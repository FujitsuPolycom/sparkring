"""Lossless text retention and immutable binary-asset request identities."""

from runtime.images.upgrades.contracts import sha
from runtime.images.upgrades.agent import request_identity
from runtime.images.upgrades import opaque_patch as module


def test_compaction_removes_only_duplicate_binary_payloads(tmp_path):
    part = b"diff --git a/lib/profile.gz b/lib/profile.gz\nGIT binary patch\nliteral 9\nopaque-data\n"
    text = b"diff --git a/lib/code.py b/lib/code.py\n--- a/lib/code.py\n+++ b/lib/code.py\n@@ -1 +1 @@\n-old\n+retained\n"
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib/profile.gz").write_bytes(b"asset-data")
    request = {
        "carried_patch": (part + text).decode(),
        "feedback": [
            {"patch": part.decode(), "error": "conflict"},
            {"patch": text.decode(), "error": "context"},
        ],
        "files": {"lib/code.py": {"candidate": "retained"}},
    }
    result, protected = module.compact(request, roots={"candidate": tmp_path})
    assert result["carried_patch"] == text.decode()
    assert result["files"] == request["files"]
    assert result["feedback"][1] == request["feedback"][1]
    assert result["feedback"][0] == {
        "opaque_patch_sha256": sha(part),
        "error": "conflict",
    }
    assert result["complete_carried_patch_sha256"] == sha(part + text)
    assert result["opaque_patch_assets"][0]["source_files"]["lib/profile.gz"][
        "candidate"
    ]["sha256"] == sha(b"asset-data")
    assert protected == ["lib/profile.gz"]
    assert request["carried_patch"] == (part + text).decode()


def test_binary_bytes_remain_bound_to_request_identity(tmp_path):
    def request(payload):
        part = (
            "diff --git a/lib/p.gz b/lib/p.gz\nGIT binary patch\nliteral 1\n"
            + payload
            + "\n"
        )
        return module.compact(
            {"carried_patch": part, "feedback": [], "files": {}},
            roots={"upstream": tmp_path},
        )[0]

    assert request_identity(request("first")) != request_identity(request("second"))


def test_text_only_request_is_unchanged(tmp_path):
    value = {
        "carried_patch": "plain text",
        "files": {"file": {"candidate": "bytes"}},
        "feedback": [],
    }
    result, protected = module.compact(value, roots={"candidate": tmp_path})
    assert result == value and result is not value and protected == []
