import json

import pytest

from runtime.common import distribution


def test_installed_source_and_bundle_are_verified_without_git(tmp_path):
    (tmp_path / "source.py").write_text("# fixture\n")
    (tmp_path / "source.bundle").write_bytes(b"fixture bundle")
    record = {"schema": "sparkring-distribution/v1", "revision": "a" * 40,
              "files": {"source.py": distribution.digest(tmp_path / "source.py")},
              "bundle_sha256": distribution.digest(tmp_path / "source.bundle")}
    (tmp_path / "distribution.json").write_text(json.dumps(record))
    assert distribution.identity(tmp_path) == "a" * 40
    distribution.bundle(tmp_path, tmp_path / "copy.bundle")
    assert (tmp_path / "copy.bundle").read_bytes() == b"fixture bundle"
    (tmp_path / "source.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="file differs"):
        distribution.identity(tmp_path)
