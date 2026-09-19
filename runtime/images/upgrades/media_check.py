"""Deterministic solid-color fixtures for bounded image/video input checks."""

import base64
import json
import struct
import zlib

from .contracts import require, sha
from .serving_checks import chat


def png(rgb, size=64):
    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    rows = (b"\0" + bytes(rgb) * size) * size
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def fixture(video):
    content = [
        {
            "type": "text",
            "text": "Identify the solid colors. Return only JSON with images as the three image colors in order, and video as the video color.",
        }
    ]
    images = []
    for name, rgb in (
        ("red", (255, 0, 0)),
        ("green", (0, 255, 0)),
        ("blue", (0, 0, 255)),
    ):
        data = png(rgb)
        images.append({"color": name, "sha256": sha(data)})
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(data).decode()
                },
            }
        )
    content.append(
        {
            "type": "video_url",
            "video_url": {
                "url": "data:video/mp4;base64," + base64.b64encode(video).decode()
            },
        }
    )
    return [{"role": "user", "content": content}], {
        "images": images,
        "video_sha256": sha(video),
        "video_color": "red",
    }


def verify_answer(content):
    value = content.strip()
    if value.startswith("```"):
        value = "\n".join(value.splitlines()[1:-1])
    parsed = json.loads(value)
    require(isinstance(parsed, dict), "Media response must be a JSON object")
    images, video = parsed.get("images"), parsed.get("video")
    require(
        isinstance(images, list)
        and len(images) == 3
        and all(isinstance(color, str) for color in images)
        and isinstance(video, str),
        "Media response must name three image colors and one video color",
    )
    # The fixture asks for color names, not a particular capitalization.
    require(
        [color.strip().casefold() for color in images] == ["red", "green", "blue"]
        and video.strip().casefold() == "red",
        "Solid-color image/video response differs from the fixture",
    )
    return parsed


def run(pair, spec, base, model, *, chat_template_kwargs=None):
    path = "/tmp/sparkring-media-" + pair.run_id + ".mp4"
    pair.owned(0, spec.name)
    pair.call(
        0,
        [
            "docker",
            "exec",
            spec.name,
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=224x224:r=4:d=1",
            "-threads",
            "1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-n",
            path,
        ],
        seconds=60,
    )
    video = pair.call(0, ["docker", "exec", spec.name, "cat", path])
    require(
        0 < len(video) < 2 * 1024**2,
        "Synthetic video exceeds the bounded fixture budget",
    )
    messages, identity = fixture(video)
    response = chat(
        base, model, messages, max_tokens=256, chat_template_kwargs=chat_template_kwargs
    )
    response["parsed"] = verify_answer(response["content"])
    return {
        "fixture": identity,
        "response": response,
        "scope": "Three solid-color images plus one red video; not general video accuracy.",
    }
