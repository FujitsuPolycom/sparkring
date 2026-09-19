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


def test_color_capitalization_does_not_change_media_meaning():
    value = dict(images=["Red", " GREEN ", "Blue"], video="RED")
    assert verify_answer(json.dumps(value)) == value


@pytest.mark.parametrize(
    "value",
    [
        [],
        dict(images=["red", "green"], video="red"),
        dict(images=["blue", "green", "red"], video="red"),
        dict(images=["red", "green", "blue"], video="BLACK"),
        dict(images=["red", "green", "blue"], video=None),
        dict(images=["red", "green", 3], video="red"),
    ],
)
def test_media_structure_order_and_wrong_colors_remain_failures(value):
    with pytest.raises(Refused):
        verify_answer(json.dumps(value))
