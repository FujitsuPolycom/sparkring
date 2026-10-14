"""Best-effort counters from vLLM's CPU statistics; no engine or CUDA calls.

The single-slot mailbox never waits for storage. A stalled writer retains at
most one pending snapshot; subsequent snapshots replace it. Only the writer
serializes JSON or touches the filesystem. Disabled unless the installer
supplies SPARKRING_STATS_PATH. One API process and engine per deployment is
supported; other engine indices are disabled to avoid reporting partial DP.
"""
import json
import os
from pathlib import Path
import threading
import time
import uuid

from vllm.v1.metrics.loggers import StatLoggerBase


class SparkRingStatLogger(StatLoggerBase):
    def __init__(self, vllm_config, engine_index=0):
        self.path = os.environ.get("SPARKRING_STATS_PATH") if engine_index == 0 else None
        self.enabled = bool(self.path) and getattr(vllm_config.parallel_config, "data_parallel_size", 1) == 1
        self.pending = None
        self.wake = threading.Event()
        self.next_write = 0.0
        self.last_step = None
        self.values = dict(prompt=0, gen=0, accepted=0, drafts=0, draft_tokens=0,
                           steps=0, step_accepted=0, step_seconds=0.0, ttft_sum=0.0, ttft_count=0,
                           running=0, waiting=0, kv=0.0)
        self.identity = dict(schema="sparkring-stats/v1", run_id=uuid.uuid4().hex,
                             deployment_id=os.environ.get("SPARKRING_STATS_DEPLOYMENT", ""))
        if self.enabled:
            try:
                threading.Thread(target=self._writer, name="sparkring-stats", daemon=True).start()
            except Exception:
                self.enabled = False

    def log_engine_initialized(self):
        pass

    def log(self):
        if self.enabled:
            try:
                self._publish()
            except Exception:
                pass

    def record(self, scheduler_stats, iteration_stats, mm_cache_stats=None, engine_idx=0):
        if not self.enabled or engine_idx != 0:
            return
        try:
            self._record(scheduler_stats, iteration_stats)
        except Exception:
            # A changed vLLM statistics interface must never stop serving or
            # publish counters that only partly describe an iteration.
            self.enabled = False

    def _record(self, scheduler, iteration):
        v = self.values
        if iteration is not None:
            v["prompt"] += iteration.prompt_token_stats.computed
            v["gen"] += iteration.num_generation_tokens
            times = iteration.time_to_first_tokens_iter
            v["ttft_sum"] += sum(times)
            v["ttft_count"] += len(times)
        if scheduler is not None:
            v.update(running=scheduler.num_running_reqs, waiting=scheduler.num_waiting_reqs,
                     kv=scheduler.kv_cache_usage)
            details = getattr(scheduler, "iteration_details", None)
            spec = getattr(scheduler, "spec_decoding_stats", None)
            new_step = details is not None and details.iteration_index != self.last_step and not details.is_dummy
            if new_step:
                self.last_step = details.iteration_index
                v["steps"] += 1
                v["step_seconds"] += details.elapsed_ms / 1000
                if spec is not None:
                    v["step_accepted"] += spec.num_accepted_tokens
            if spec is not None and (details is None or new_step):
                v["accepted"] += spec.num_accepted_tokens
                v["drafts"] += spec.num_drafts
                v["draft_tokens"] += spec.num_draft_tokens
        self._publish()

    def _publish(self):
        stamp = time.monotonic()
        if stamp < self.next_write:
            return
        self.next_write = stamp + 0.5
        self.pending = {**self.identity, **self.values, "at": time.time(), "monotonic": stamp,
                        "step_timing": self.last_step is not None}
        self.wake.set()

    def _writer(self):
        while True:
            self.wake.wait()
            self.wake.clear()
            snapshot, self.pending = self.pending, None
            if snapshot is not None:
                self._write(snapshot)

    def _write(self, snapshot):
        temporary = Path(self.path + ".writing")
        try:
            temporary.write_text(json.dumps(snapshot, separators=(",", ":")), encoding="utf-8")
            temporary.replace(self.path)
        except Exception:
            # No retries or logs on a full, read-only, or failed filesystem.
            pass
