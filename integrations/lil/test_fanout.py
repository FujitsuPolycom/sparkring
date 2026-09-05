import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location(
    "fanout", Path(__file__).with_name("fanout.py")
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


@pytest.mark.parametrize("pull", [False, True])
def test_direction_and_verified_publication(pull):
    calls = []
    sha = "a" * 64

    def run(host, argv):
        calls.append((host, argv))
        if argv[0] == "sha256sum":
            return sha + "  file"
        if argv[0] == "mktemp":
            return "/srv/test/.lil-fanout-12345678"
        return ""

    mod.copy_edge(
        run,
        "rank0",
        "rank1",
        "cody@192.0.2.1",
        "/srv/test/source",
        "/srv/test/result",
        sha,
        pull=pull,
    )
    scp = next(c for c in calls if c[1][0] == "scp")
    assert scp[0] == ("rank1" if pull else "rank0")
    assert calls[-2][1][0] == "ln"
    assert calls[-3][1][0] == "sha256sum"


def test_source_mismatch_never_transfers():
    calls = []

    def run(host, argv):
        calls.append(argv)
        return "b" * 64 + " file"

    with pytest.raises(ValueError, match="source checksum"):
        mod.copy_edge(
            run, "rank0", "rank1", "peer", "/srv/source", "/srv/dest", "a" * 64
        )
    assert len(calls) == 1
