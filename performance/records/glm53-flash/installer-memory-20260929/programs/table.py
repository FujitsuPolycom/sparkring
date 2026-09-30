"""Measures, per tested configuration directory: settings, KV tokens, per-Spark MemAvailable lows
(startup / idle / text load / multimodal load), startup time, and B12X compilations per stage.

Usage: table.py DIR [DIR ...]
"""
import collections
import datetime
import json
import re
import sys


def epoch(stamp):
    return int(datetime.datetime.fromisoformat(stamp[:19] + "+00:00").timestamp())


def settings(base):
    cfg = json.load(open(f"{base}/config.json"))
    env, args = cfg["environment"], cfg["vllm_args"]

    def arg(name):
        return args[args.index(name) + 1] if name in args else "-"

    workers = env.get("B12X_COMPILE_WORKERS", "8")
    if "B12X_BIND_COMPILE_WORKERS" in env:
        workers += f"/bind {env['B12X_BIND_COMPILE_WORKERS']}"
    return (f"ch {env.get('NCCL_MAX_NCHANNELS', '-')} page {arg('--block-size')} workers {workers} "
            f"KV {int(arg('--kv-cache-memory-bytes')) / 2**30:g} GiB mmcache {arg('--mm-processor-cache-gb') if '--mm-processor-cache-gb' in args else 'default'}")


for base in sys.argv[1:]:
    phases = {}
    for line in open(f"{base}/phases.txt"):
        key, value = line.split()
        phases[key] = int(value)
    log = open(f"{base}/rank0.log", encoding="utf-8", errors="replace").read().replace("\r", "\n")
    log = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", log)
    kv = re.findall(r"GPU KV cache size: ([\d,]+) tokens", log)[-1]
    ready = epoch(re.search(r"^(\S+) .*Application startup complete", log, re.M).group(1))
    kv_alloc = epoch(re.search(r"^(\S+) .*GPU KV cache size", log, re.M).group(1))
    stages, current = {}, None
    for line in log.splitlines():
        m = re.search(r"b12x (weights|state|bind) preparation uses (\d+)", line)
        if m:
            current = m.group(1)
            continue
        m = re.search(r"b12x ready .* (\d+) compilations, (\S+)", line)
        if m and current:
            stages[current] = f"{m.group(1)}c/{m.group(2)}"
    samples = collections.defaultdict(list)
    for line in open(f"{base}/memlog.txt"):
        parts = line.split()
        if len(parts) == 3 and parts[2].isdigit():
            samples[parts[0]].append((int(parts[1]), int(parts[2]) / 1048576))
    # Startup starts when the new rank-0 container starts, so the previous model's last minutes
    # (still running during the installer's checks) are not attributed to this configuration.
    container_start = epoch(re.search(r"^(\d{4}-\d\d-\d\dT\S+)", log, re.M).group(1))
    windows = {"startup": (container_start, ready + 3), "idle": (ready + 4, phases.get("bench_start", ready + 60)),
               "load": (phases.get("bench_start", 0), phases.get("bench_end", 0))}
    # mm*: mm_stress.py (4 images of 2560x2560 or one 16-frame video per request);
    # img*: image_load.py (3 unique images of 2048x2048, 3840x2160, 1600x1200 per request).
    for prefix in ("mm", "img"):
        starts = [v for k, v in phases.items() if k.startswith(prefix) and k.endswith("_start")]
        ends = [v for k, v in phases.items() if k.startswith(prefix) and k.endswith("_end")]
        if starts:
            windows[prefix] = (min(starts), max(ends) + 15)
    if "img16_start" in phases:
        windows["img16"] = (phases["img16_start"], phases["img16_end"] + 15)
    for name in ("bench2", "burst6", "mixed", "mixedfresh"):
        if f"{name}_start" in phases and f"{name}_end" in phases:
            windows[name] = (phases[f"{name}_start"], phases[f"{name}_end"] + 15)
    # Settled level after each load step: highest reading in the gap before the next step starts.
    ordered = sorted((v, k[:-6]) for k, v in phases.items() if k.endswith("_start"))
    for (start, name), following in zip(ordered, ordered[1:] + [(None, None)]):
        end = phases.get(f"{name}_end")
        if end and following[0] and following[0] - end > 20:
            windows[f"after {name}"] = (end + 15, following[0])
    # A window never extends into the next load step.
    starts_after = sorted(v for k, v in phases.items() if k.endswith("_start") and k != "install_start")
    for name, (start, end) in list(windows.items()):
        later = [s for s in starts_after if s > start]
        if later and not name.startswith("after ") and name not in ("mm", "img"):
            windows[name] = (start, min(end, later[0] - 1))
    lows = {}
    for name, (start, end) in windows.items():
        if name.startswith("after "):
            lows[name] = [max((v for t, v in samples[n] if start <= t <= end), default=float("nan")) for n in sorted(samples)]
            continue
        lows[name] = [min((v for t, v in samples[n] if start <= t <= end), default=float("nan")) for n in sorted(samples)]
    print(f"{base.split('/')[-1]}: {settings(base)}; KV tokens {kv}; container start to API ready {ready - container_start} s "
          f"(KV sizing to API ready {ready - kv_alloc} s); compilations {stages}")
    for name, values in lows.items():
        print(f"   {name:8s} " + "  ".join(f"r{i} {v:5.2f}" for i, v in enumerate(values)))
