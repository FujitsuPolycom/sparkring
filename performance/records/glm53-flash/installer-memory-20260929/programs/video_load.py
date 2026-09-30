"""Measures video-load memory: concurrent chat requests that each carry one unique smooth-noise video.

Each video is fresh smooth noise at 1920x1080, written as MP4 at 2 frames per
second (the GLM processor's sampling rate), so every request runs the vision
encoder and prefill for all of its frames. Use --frames to set the length (for
example 64 frames = 32 s). Run it next to a memory sampler or under guarded.py.
Video seeds are 1000 + LOAD_INDEX_START (environment, default 0) + request position.
"""
import argparse
import base64
import concurrent.futures
import json
import os
import tempfile
import time
import urllib.request

import cv2
import numpy as np


def video_url(seed, frames, width, height):
    path = os.path.join(tempfile.gettempdir(), f"video-load-{seed}-{frames}.mp4")
    rng = np.random.default_rng(seed)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 2, (width, height))
    base = rng.integers(0, 256, size=(height // 8, width // 8, 3), dtype=np.uint8)
    for frame in range(frames):
        shifted = np.roll(base, frame * 3, axis=1)
        writer.write(cv2.resize(shifted, (width, height), interpolation=cv2.INTER_CUBIC))
    writer.release()
    data = open(path, "rb").read()
    os.remove(path)
    return "data:video/mp4;base64," + base64.b64encode(data).decode()


def request(args, index):
    content = [{"type": "text", "text": f"[{index}] Describe what happens in this video in one sentence."},
               {"type": "video_url", "video_url": {"url": video_url(index, args.frames, 1920, 1080)}}]
    body = {"model": args.model, "max_tokens": args.max_tokens, "temperature": 0,
            "messages": [{"role": "user", "content": content}], **args.extra}
    started = time.time()
    try:
        http = urllib.request.Request(f"http://{args.host}:{args.port}/v1/chat/completions",
                                      json.dumps(body).encode(), {"Content-Type": "application/json"})
        usage = json.load(urllib.request.urlopen(http, timeout=args.timeout))["usage"]
        return True, usage["prompt_tokens"], time.time() - started, ""
    except Exception as error:  # noqa: BLE001 - a failed request is a result
        detail = getattr(error, "read", lambda: b"")()[:200].decode(errors="replace")
        return False, 0, time.time() - started, f"{error} {detail}".strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--frames", type=int, default=64, help="frames at 2 per second (default 64 = 32 s)")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--extra", type=json.loads, default={}, help="JSON merged into every request body")
    args = parser.parse_args()
    index = 1000 + int(os.environ.get("LOAD_INDEX_START", "0"))
    for round_number in range(1, args.rounds + 1):
        started = time.time()
        with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
            results = list(pool.map(lambda i: request(args, i), range(index, index + args.concurrency)))
        index += args.concurrency
        ok = [r for r in results if r[0]]
        failed = [r for r in results if not r[0]]
        tokens = sum(r[1] for r in ok) / len(ok) if ok else 0
        print(f"round {round_number}: {len(ok)}/{len(results)} ok, {tokens:,.0f} prompt tokens per request, "
              f"{time.time() - started:.1f} s" + (f"; first error: {failed[0][3][:200]}" if failed else ""),
              flush=True)


if __name__ == "__main__":
    main()
