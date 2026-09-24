"""The native loader gets only its io_uring calls and a CPU-only admission gate."""
import json
from types import SimpleNamespace

import pytest

from runtime.common import compose, installer, loader_policy
from runtime.common.test_installer_image import PROFILE, image_lock, ring_site
from scripts import installer_host
from scripts.test_installer_host import inspection


def test_policy_preserves_every_upstream_rule_and_adds_only_three_calls():
    baseline = json.loads((installer.ROOT / "third_party/moby_seccomp/default.json").read_text())
    policy = json.loads(loader_policy.PROFILE.read_text())
    extra = policy["syscalls"].pop()
    assert policy == baseline
    assert extra.pop("comment").startswith("SparkRing modification")
    assert extra == {"names": ["io_uring_setup", "io_uring_enter", "io_uring_register"], "action": "SCMP_ACT_ALLOW"}
    assert baseline["defaultAction"] == "SCMP_ACT_ERRNO"


def test_preparation_probe_uses_the_serving_policy_without_gpu_or_network():
    calls = []
    def run(argv):
        calls.append(argv)
        return SimpleNamespace(stdout='{"io_uring":"available","gpu_used":false}')
    loader_policy.check("sha256:" + "a" * 64, run=run)
    argv = calls[0]
    assert argv[argv.index("--runtime") + 1] == "runc"
    assert argv[argv.index("--network") + 1] == "none"
    assert "--gpus" not in argv and "seccomp=unconfined" not in argv
    assert "seccomp=" + str(loader_policy.PROFILE) in argv
    assert argv[argv.index("--cap-drop") + 1] == "ALL"
    assert "memlock=-1:-1" in argv


@pytest.mark.parametrize("change", [False, True])
def test_container_ownership_compares_full_policy_content(change, monkeypatch):
    lock = installer.make_lock(PROFILE, ring_site(), "1" * 40, "2" * 64, image_runtime=image_lock())
    spec = installer.specifications(lock)[0]
    info, image = inspection(spec)
    monkeypatch.setattr(compose, "check_project_containers", lambda *a, **k: None)
    if change:
        info["HostConfig"]["SecurityOpt"] = ["seccomp=unconfined"]
        with pytest.raises(ValueError):
            installer_host.owned(spec, info, image)
    else:
        assert installer_host.owned(spec, info, image) == info


def test_external_policy_does_not_change_the_published_profile_envelope():
    baseline = installer.make_lock(PROFILE, ring_site(), "1" * 40, "2" * 64)
    candidate = installer.make_lock(PROFILE, ring_site(), "1" * 40, "2" * 64, image_runtime=image_lock())
    for before, after, row in zip(installer.specifications(baseline), installer.specifications(candidate), candidate["site"]["ranks"], strict=True):
        assert before.security_opt == ()
        assert after.security_opt == ("seccomp=" + row["repository"] + "/" + loader_policy.RELATIVE,)
        assert before.cap_add == after.cap_add == ()
        assert before.memlock == after.memlock == -1
