"""Measures vision-load memory: concurrent chat requests that each carry several large, unique random-noise images.

Every image is fresh random noise (JPEG), so neither the prefix cache nor a media
cache can reuse work: each request runs the vision encoder and prefill for all of
its images. Use it to measure the memory the vision path needs under load; it
watches nothing itself, so run it next to a memory sampler or under guarded.py.
Request indices, and so image seeds (index * 100 + image number), start at the LOAD_INDEX_START
environment variable (default 0); give each step its own offset so it sends images the server has
not seen, because the prefix cache skips the vision encoder for repeated images.

Example:
    python image_load.py --host 192.0.2.10 --port 8015 --model GLM-5.3-Flash-NVFP4-Spark-TP4 \
        --concurrency 8 --rounds 3 --extra '{"chat_template_kwargs": {"reasoning_effort": "low"}}'
"""
import argparse
import base64
import concurrent.futures
import io
import json
import os
import time
import urllib.request

import numpy as np
from PIL import Image


def image_url(seed, size):
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def request(args, index):
    content = [{"type": "text", "text": f"[{index}] Describe each image in one short sentence."}]
    for k in range(args.images):
        size = args.sizes[(index + k) % len(args.sizes)]
        content.append({"type": "image_url", "image_url": {"url": image_url(index * 100 + k, size)}})
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
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--images", type=int, default=3, help="images per request (default 3)")
    parser.add_argument("--sizes", default="2048x2048,3840x2160,1600x1200",
                        help="image sizes WxH, cycled across images (default 2048x2048,3840x2160,1600x1200)")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--extra", type=json.loads, default={}, help="JSON merged into every request body")
    args = parser.parse_args()
    args.sizes = [tuple(int(v) for v in s.split("x")) for s in args.sizes.split(",")]
    # Offset of the first request index (and so of the image seeds), so a later run sends images the
    # server has not seen: the prefix cache would otherwise skip the vision encoder for repeats.
    index = int(os.environ.get("LOAD_INDEX_START", "0"))
    for round_number in range(1, args.rounds + 1):
        started = time.time()
        with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
            results = list(pool.map(lambda i: request(args, i), range(index, index + args.concurrency)))
        index += args.concurrency
        ok = [r for r in results if r[0]]
        failed = [r for r in results if not r[0]]
        tokens = sum(r[1] for r in ok) / len(ok) if ok else 0
        print(f"round {round_number}: {len(ok)}/{len(results)} ok, {tokens:,.0f} prompt tokens per request, "
              f"{time.time() - started:.1f} s" + (f"; first error: {failed[0][3][:160]}" if failed else ""),
              flush=True)


if __name__ == "__main__":
    main()
