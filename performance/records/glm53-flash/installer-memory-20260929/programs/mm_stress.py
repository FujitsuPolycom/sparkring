"""Measures multimodal memory load: concurrent chat requests carrying four large images each, then one 16-frame video each.

Sends CONCURRENCY simultaneous chat requests, each carrying the profile's per-request maximum of
four large images (random-noise JPEGs of SIDE x SIDE pixels, which reach the processor's
8,000-token-per-image cap), then the same number of requests that each carry one 16-frame
1920x1080 video. Prints prompt tokens and latency per request.

Usage: mm_stress.py [CONCURRENCY] [SIDE]
Environment: API_URL = chat completions URL (default http://192.0.2.10:8015/v1/chat/completions),
             MODEL = served model name (default GLM-5.3-Flash-NVFP4-Spark-TP4).
Image and video seeds depend only on the request position, so a second run repeats the first run's media.
"""
import base64
import concurrent.futures
import io
import json
import os
import sys
import tempfile
import time
import urllib.request

import cv2
import numpy
from PIL import Image

URL = os.environ.get("API_URL", "http://192.0.2.10:8015/v1/chat/completions")
MODEL = os.environ.get("MODEL", "GLM-5.3-Flash-NVFP4-Spark-TP4")
concurrency = int(sys.argv[1]) if len(sys.argv) > 1 else 4
side = int(sys.argv[2]) if len(sys.argv) > 2 else 2560
rng = numpy.random.default_rng(7)


def smooth_noise(seed, width, height):
    """Distinct low-frequency noise scaled up: full-size pixels, small compressed payload."""
    small = numpy.random.default_rng(seed).integers(0, 256, (48, 48, 3), dtype=numpy.uint8)
    return numpy.asarray(Image.fromarray(small).resize((width, height), Image.BICUBIC))


def jpeg(seed):
    buffer = io.BytesIO()
    Image.fromarray(smooth_noise(seed, side, side)).save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def mp4(seed):
    path = os.path.join(tempfile.gettempdir(), f"tp4-mm-stress-{seed}.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 2, (1920, 1080))
    for frame in range(16):
        writer.write(smooth_noise(1000 * seed + frame, 1920, 1080))
    writer.release()
    return "data:video/mp4;base64," + base64.b64encode(open(path, "rb").read()).decode()


def send(content):
    body = json.dumps({"model": MODEL, "max_tokens": 16, "temperature": 1.0,
                       "chat_template_kwargs": {"reasoning_effort": "low"},
                       "messages": [{"role": "user", "content": content}]}).encode()
    start = time.time()
    request = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    try:
        usage = json.load(urllib.request.urlopen(request, timeout=900))["usage"]
        return f"prompt {usage['prompt_tokens']:6d} tokens, {time.time() - start:6.1f} s"
    except Exception as error:  # report and continue; the memory trace is the measurement
        return f"FAILED after {time.time() - start:.1f} s: {str(error)[:200]}"


images = [jpeg(seed) for seed in range(4 * concurrency)]
videos = [mp4(seed) for seed in range(concurrency)]
print(f"{time.strftime('%H:%M:%S')} image phase: {concurrency} requests x 4 images of {side}x{side}", flush=True)
with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
    jobs = [pool.submit(send, [{"type": "image_url", "image_url": {"url": images[4 * i + k]}} for k in range(4)]
                        + [{"type": "text", "text": "Describe these four images in one short sentence."}])
            for i in range(concurrency)]
    for job in jobs:
        print("  image request:", job.result(), flush=True)
print(f"{time.strftime('%H:%M:%S')} video phase: {concurrency} requests x one 16-frame 1920x1080 video", flush=True)
with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
    jobs = [pool.submit(send, [{"type": "video_url", "video_url": {"url": videos[i]}},
                               {"type": "text", "text": f"Request {i}: describe this video in one short sentence."}])
            for i in range(concurrency)]
    for job in jobs:
        print("  video request:", job.result(), flush=True)
print(f"{time.strftime('%H:%M:%S')} done", flush=True)
