"""Measures available memory on all four Sparks at a fast interval while a load command runs, and stops the load below a floor.

Each Spark streams one reading every --interval seconds over a persistent SSH connection
(Node A directly, the workers through Node A) using stream.sh. Every reading is appended to
--log. When any Spark reads below --floor GiB, the load command's whole process tree is killed at
once (Windows taskkill), so its requests end. Prints the lowest reading per Spark and whether the
guard fired.

Usage: fleet_guard_fast.py --node-a USER@192.0.2.10 --workers 192.0.2.11,192.0.2.12,192.0.2.13 \
           --floor 2.0 --interval 0.2 --log FILE -- COMMAND...
The workers must accept root SSH from Node A (sudo -n ssh root@ADDRESS); stream.sh must be at
--stream-script on Node A and at /tmp/stream.sh on each worker.
"""
import argparse
import subprocess
import sys
import threading
import time

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--node-a", required=True, help="SSH target of Node A")
parser.add_argument("--workers", required=True, help="comma-separated worker addresses in rank order, reachable from Node A")
parser.add_argument("--stream-script", default="$HOME/tp4mem/stream.sh", help="path of stream.sh on Node A")
parser.add_argument("--floor", type=float, required=True, help="GiB of MemAvailable on any Spark that stops the load")
parser.add_argument("--interval", type=float, default=0.2, help="seconds between readings on each Spark")
parser.add_argument("--log", required=True, help="file that receives every reading")
parser.add_argument("command", nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command[1:] if args.command[:1] == ["--"] else args.command

streams = {"r0": ["ssh", args.node_a, f"sh {args.stream_script} r0 {args.interval}"]}
for rank, address in enumerate(args.workers.split(","), start=1):
    streams[f"r{rank}"] = ["ssh", args.node_a, f"sudo -n ssh root@{address} sh /tmp/stream.sh r{rank} {args.interval}"]
lowest = {node: float("inf") for node in streams}
fired = []
lock = threading.Lock()
log = open(args.log, "a")
load = None
watchers = {node: subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1) for node, cmd in streams.items()}


def stop_load(reason):
    with lock:
        if fired or load is None or load.poll() is not None:
            return
        fired.append(reason)
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(load.pid)], capture_output=True)
    print(f"GUARD {time.strftime('%H:%M:%S')}: {reason}; load stopped", flush=True)


def read(node, pipe):
    for line in pipe:
        parts = line.split()
        if len(parts) != 3 or not parts[2].isdigit():
            continue
        gib = int(parts[2]) / 1048576
        with lock:
            log.write(line if line.endswith("\n") else line + "\n")
            lowest[node] = min(lowest[node], gib)
        if gib < args.floor:
            stop_load(f"{node} MemAvailable {gib:.2f} GiB < {args.floor} GiB")


threads = [threading.Thread(target=read, args=(node, w.stdout), daemon=True) for node, w in watchers.items()]
for thread in threads:
    thread.start()
time.sleep(2)  # first readings arrive before the load starts
load = subprocess.Popen(command)
code = load.wait()
time.sleep(1)
for w in watchers.values():
    w.kill()
log.close()
print(f"load exit {code}; lowest MemAvailable " + ", ".join(f"{n} {v:.2f} GiB" for n, v in sorted(lowest.items()))
      + f"; guard fired: {bool(fired)}", flush=True)
sys.exit(code)
