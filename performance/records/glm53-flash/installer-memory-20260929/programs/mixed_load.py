"""Measures mixed-load memory: long-context text streams, image requests and video requests at the same time.

Starts three load generators together and waits for all of them:
- llm_decode_bench.py at --text-concurrency streams with --text-context tokens of context,
- image_load.py at --image-concurrency (3 large images per request),
- video_load.py at --video-concurrency (--frames frames per video).
Run it under guarded.py so the whole mix stops if memory gets low.
Environment: BENCH_DIR = llm-inference-bench checkout; BENCH_PYTHON = the Python that runs it.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.environ["BENCH_DIR"]
BENCH_PYTHON = os.environ["BENCH_PYTHON"]

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--host", required=True)
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--kv-budget", type=int, required=True)
parser.add_argument("--text-concurrency", type=int, default=4)
parser.add_argument("--text-context", default="64k")
parser.add_argument("--image-concurrency", type=int, default=8)
parser.add_argument("--images", type=int, default=3, help="images per image request (default 3)")
parser.add_argument("--video-concurrency", type=int, default=2)
parser.add_argument("--frames", type=int, default=64)
parser.add_argument("--rounds", type=int, default=2)
parser.add_argument("--extra", default="{}")
args = parser.parse_args()
json.loads(args.extra)
common = ["--host", args.host, "--port", str(args.port), "--model", args.model, "--extra", args.extra]
jobs = {
    "text": subprocess.Popen([BENCH_PYTHON, "llm_decode_bench.py", "--host",
                              "http://" + args.host, "--port", str(args.port), "--model", args.model, "--dcp-size", "1",
                              "--no-hw-monitor", "--kv-budget", str(args.kv_budget), "--temperature", "1.0",
                              "--token-targeting", "exact", "--display-mode", "plain", "--concurrency",
                              str(args.text_concurrency), "--contexts", args.text_context, "--skip-prefill",
                              "--max-tokens", "2048", "--duration", "60", "--decode-warmup-seconds", "5",
                              "--cell-warmup-timeout-seconds", "900", "--no-resume", "--output",
                              os.path.join(HERE, "mixed-text.json")], cwd=BENCH, stdout=subprocess.DEVNULL,
                             stderr=subprocess.STDOUT),
    "images": subprocess.Popen([sys.executable, os.path.join(HERE, "image_load.py"), *common, "--concurrency",
                                str(args.image_concurrency), "--rounds", str(args.rounds), "--images", str(args.images)]),
    "video": subprocess.Popen([sys.executable, os.path.join(HERE, "video_load.py"), *common, "--concurrency",
                               str(args.video_concurrency), "--rounds", str(args.rounds), "--frames", str(args.frames)]),
}
codes = {name: job.wait() for name, job in jobs.items()}
print("exit codes:", codes, flush=True)
sys.exit(max(codes.values()))
