"""Measure an extracted vLLM SpinCondition with real POSIX yield and ZMQ IPC."""
import argparse
import ast
import hashlib
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import platform
import statistics
import tempfile
import threading
import time
import uuid

import zmq


def load_condition(source):
    raw = Path(source).read_bytes()
    tree = ast.parse(raw)
    selected = [node for node in tree.body if isinstance(node, ast.ClassDef)
                and node.name == 'SpinCondition']
    if len(selected) != 1:
        raise ValueError('Expected exactly one SpinCondition class')
    imported_names = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported_names.update(alias.asname or alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_names.update(alias.asname or alias.name for alias in node.names)
    candidates = dict(zmq=zmq, time=time, os=os, SUB=zmq.SUB, PUB=zmq.PUB,
                     SUBSCRIBE=zmq.SUBSCRIBE,
                     sched_yield=getattr(os, 'sched_yield', lambda: time.sleep(0)),
                     get_open_zmq_inproc_path=lambda: 'inproc://' + str(uuid.uuid4()))
    namespace = {name: obj for name, obj in candidates.items() if name in imported_names}
    if any(isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
           and target.id == 'logger' for target in node.targets) for node in tree.body):
        namespace['logger'] = logging.getLogger('ipc-wait-probe')
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['SpinCondition']


class CountPolls:
    def __init__(self, poller):
        self.poller = poller
        self.calls = 0

    def poll(self, **kwargs):
        self.calls += 1
        return self.poller.poll(**kwargs)


def reader(source, busy, address, state, ack, start, pipe, count):
    ctx = zmq.Context()
    condition = load_condition(source)(True, ctx, address, busy_loop_s=busy)
    condition.poller = CountPolls(condition.poller)
    pipe.send('ready')
    start.wait(5)
    condition.record_read()
    cpu_start, wall_start = time.process_time(), time.monotonic()
    delays, waits = [], 0
    for expected in range(1, count + 1):
        deadline = time.monotonic() + 5
        while True:
            with state.get_lock():
                sequence, published = state[:]
            if sequence == expected:
                delays.append((time.monotonic_ns() - published) / 1000)
                condition.record_read()
                ack.value = expected
                break
            if sequence > expected or time.monotonic() > deadline:
                raise RuntimeError('Missing, reordered or overdue IPC message')
            condition.wait(timeout_ms=1000)
            waits += 1
    pipe.send(dict(reader_cpu_s=time.process_time() - cpu_start,
                   wall_s=time.monotonic() - wall_start,
                   latency_us=delays, waits=waits, polls=condition.poller.calls,
                   received=len(delays)))
    for socket in (condition.local_notify_socket, condition.read_cancel_socket,
                   condition.write_cancel_socket):
        socket.close(linger=0)
    ctx.term()


def run_case(source, busy, name, gaps):
    condition_class = load_condition(source)
    context = mp.get_context('spawn')
    state = context.Array('q', [0, 0])
    ack = context.Value('q', 0)
    start = context.Event()
    receive, send = context.Pipe(duplex=False)
    with tempfile.TemporaryDirectory(prefix='sparkring-ipc-') as directory:
        address = 'ipc://' + directory + '/notify.sock'
        ctx = zmq.Context()
        writer = condition_class(False, ctx, address)
        process = context.Process(target=reader,
                                  args=(str(source), busy, address, state, ack, start, send, len(gaps)))
        process.start()
        try:
            if not receive.poll(5) or receive.recv() != 'ready':
                raise RuntimeError('Reader failed to initialize')
            # ZMQ subscription propagation is outside the timed workload.
            time.sleep(0.15)
            start.set()
            for sequence, gap in enumerate(gaps, 1):
                time.sleep(gap)
                with state.get_lock():
                    state[:] = [sequence, time.monotonic_ns()]
                writer.notify()
                deadline = time.monotonic() + 3
                while ack.value != sequence:
                    if time.monotonic() > deadline or not process.is_alive():
                        raise RuntimeError('Reader did not acknowledge published state')
                    time.sleep(0.00005)
            if not receive.poll(5):
                raise RuntimeError('Reader did not return measurements')
            result = receive.recv()
            process.join(5)
            if process.exitcode != 0:
                raise RuntimeError('Reader failed after receiving its messages')
        finally:
            if process.is_alive():
                process.terminate()
                process.join(5)
            receive.close()
            send.close()
            writer.local_notify_socket.close(linger=0)
            ctx.term()
    samples = sorted(result['latency_us'])
    result.update(case=name, busy_loop_s=busy, sent=len(gaps),
                  cpu_fraction=result['reader_cpu_s'] / result['wall_s'],
                  p50_us=statistics.median(samples),
                  p95_us=samples[min(len(samples) - 1, int(len(samples) * 0.95))],
                  max_us=max(samples))
    return result


def cancellation(source, busy):
    ctx = zmq.Context()
    address = 'inproc://' + str(uuid.uuid4())
    condition_class = load_condition(source)
    writer = condition_class(False, ctx, address)
    condition = condition_class(True, ctx, address, busy_loop_s=busy)
    condition.last_read = time.monotonic() - busy - 1
    thread = threading.Thread(target=lambda: (time.sleep(0.02), condition.cancel()))
    before = time.monotonic()
    thread.start()
    condition.wait(timeout_ms=1000)
    elapsed = time.monotonic() - before
    thread.join(2)
    for socket in (condition.local_notify_socket, condition.read_cancel_socket,
                   condition.write_cancel_socket):
        socket.close(linger=0)
    writer.local_notify_socket.close(linger=0)
    ctx.term()
    if elapsed > 0.5:
        raise RuntimeError('Cancellation did not wake the parked reader')
    return elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if not hasattr(os, 'sched_yield'):
        parser.error('This measurement requires POSIX os.sched_yield')
    if not 1 <= args.repeats <= 20:
        parser.error('Repeats must be between 1 and 20')
    if args.output.exists():
        parser.error('Refusing to overwrite an existing measurement')
    results = []
    for repeat in range(args.repeats):
        policies = (1.0, 0.002) if repeat % 2 == 0 else (0.002, 1.0)
        for busy in policies:
            for name, gaps in [('idle', [0.15] * 8), ('decode-gap', [0.015] * 80),
                               ('burst', [0.0002] * 160)]:
                row = run_case(args.source, busy, name, gaps)
                row['repeat'] = repeat
                results.append(row)
                print(name, busy, repeat, row['received'], row['cpu_fraction'], row['p95_us'], flush=True)
    report = dict(schema='sparkring-ipc-wait-probe/v1',
                  harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                  platform=platform.platform(), python=platform.python_version(),
                  zmq=zmq.__version__, policy_order='counterbalanced by repetition',
                  scope='Extracted actual SpinCondition class, real ZMQ IPC, POSIX yield, synchronized shared state; no model/GPU/complete MessageQueue',
                  cancellation_seconds={str(busy): cancellation(args.source, busy) for busy in (1.0, 0.002)},
                  results=results)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    for name in ('idle', 'decode-gap', 'burst'):
        for busy in (1.0, 0.002):
            selected = [row for row in results if row['case'] == name and row['busy_loop_s'] == busy]
            print(name, busy, 'cpu%', round(100 * statistics.median(row['cpu_fraction'] for row in selected), 2),
                  'p95_us', round(statistics.median(row['p95_us'] for row in selected), 2),
                  'polls', [row['polls'] for row in selected], flush=True)


if __name__ == '__main__':
    main()
