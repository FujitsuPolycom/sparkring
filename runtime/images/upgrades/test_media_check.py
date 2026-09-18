"""Fixture identities and exact response interpretation without model hardware."""

import json
import struct
import zlib

import pytest

from .contracts import Refused
from .media_check import fixture, png, verify_answer


def test_png_encodes_expected_rgb_and_valid_chunks():
    data = png((255, 0, 0), 2)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    position = 8
    chunks = {}
    while position < len(data):
        size = struct.unpack_from(">I", data, position)[0]
        kind = data[position + 4 : position + 8]
        payload = data[position + 8 : position + 8 + size]
        assert (
            zlib.crc32(kind + payload) & 0xFFFFFFFF
            == struct.unpack_from(">I", data, position + 8 + size)[0]
        )
        chunks[kind] = payload
        position += 12 + size
    assert zlib.decompress(chunks[b"IDAT"]) == (b"\0" + b"\xff\0\0" * 2) * 2


def test_fixture_uses_three_images_and_one_video():
    messages, identity = fixture(b"fixture video")
    types = [item["type"] for item in messages[0]["content"]]
    assert types.count("image_url") == 3 and types.count("video_url") == 1
    assert len(identity["images"]) == 3


def test_wrong_media_colors_are_not_a_pass():
    assert verify_answer(json.dumps(dict(images=["red", "green", "blue"], video="red")))
    with pytest.raises(Refused):
        verify_answer(json.dumps(dict(images=["red", "green", "blue"], video="black")))
