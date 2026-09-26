"""Compose exports of installer profiles run the containers `sparkring install` runs.

The installer (runtime/common/installer.py) adapts each rank's canonical
container with installer_image.adapt. A Compose export applies the same
adapter for a host without the installer. For one site the two containers
differ only in:

- the runtime binding, a file the installer writes per rank with the created
  container's ID and the host's node ID, and its SPARKRING_RUNTIME_BINDING
  variable. Compose exports omit both;
- labels: each deployment labels its containers with its own identity, and
  the Compose manifest, not a label, records the image lock.
"""
import copy
import json
import shutil
import subprocess
import sys

import pytest
import yaml

from runtime.common import compose, installer, installer_image, loader_policy
from runtime.common.test_compose import compose_cli  # noqa: F401  (pytest fixture)

PROFILES = installer_image.SUPPORTED
BINDING_ENVIRONMENT = "SPARKRING_RUNTIME_BINDING"


def install_site(nodes):
    hosts = []
    for number in range(nodes):
        row = {"host": f"spark{number}", "management_ip": f"192.0.2.{20 + number}",
               "fabric_ip": f"198.18.20.{number + 1}", "interface": "enp1s0f0np0"}
        if nodes == 4:
            row["fabric"] = {"site_path": "/srv/sparkring/mesh-site.json",
                             "site_sha256": "1" * 64, "plan_sha256": "2" * 64}
        hosts.append(row)
    return {"schema": "sparkring-install-site/v1", "name": "parity", "hosts": hosts}


def installer_lock(profile):
    nodes = 4 if profile.endswith("-tp4") else 2
    return installer.make_lock(profile, install_site(nodes), "1" * 40, "2" * 64,
                               image_runtime=installer_image.for_profile(profile))


def example_site(profile):
    return compose.read_site(compose.ROOT / "profiles" / profile.removesuffix("-sparkcache")
                             / "compose/site.example.yaml")


def test_installer_profiles_are_compose_profiles_on_the_default_lock():
    assert set(PROFILES) == set(installer.INSTALLABLE) <= set(compose.SUPPORTED)
    for profile in compose.SUPPORTED:
        expected = installer_image.default_lock() if profile in PROFILES else None
        assert compose.installer_image_runtime(profile) == expected


@pytest.mark.parametrize("profile", PROFILES)
def test_compose_export_matches_installer_container_for_each_rank(profile):
    lock = installer_lock(profile)
    installed = installer.specifications(lock)
    manifest, files = compose.build(profile, installer.compose_site(lock))
    assert manifest["image_runtime"] == lock["image_runtime"]
    assert manifest["image"] == lock["selection"]["image_reference"]
    assert manifest["image_id"] == lock["selection"]["image_id"]
    for number, spec in enumerate(installed):
        expected = json.loads(compose.encoded(spec.document()))
        binding = expected["mounts"].pop()
        assert binding == {"source": installer_image.binding_path(lock, lock["site"]["ranks"][number]),
                           "target": installer_image.BINDING_TARGET, "read_only": True}
        assert expected["environment"].pop(BINDING_ENVIRONMENT) == installer_image.BINDING_TARGET
        assert expected["labels"] == {compose.LABEL: lock["id"], "io.sparkring.rank": str(number),
                                      "io.sparkring.image-lock": lock["image_runtime"]["name"]}
        expected["labels"] = {compose.LABEL: manifest["id"], "io.sparkring.rank": str(number)}
        assert json.loads(files[f"rank{number}/container.json"]) == expected


@pytest.mark.parametrize("profile", PROFILES)
def test_rendered_service_differs_from_installer_export_only_by_binding_and_labels(profile):
    lock = installer_lock(profile)
    manifest, files = compose.build(profile, installer.compose_site(lock))
    for name, text in installer.rendered(lock).items():
        if not name.endswith("compose.yaml"):
            continue
        expected = yaml.safe_load(text)["services"]["model"]
        actual = yaml.safe_load(files[name])["services"]["model"]
        assert expected["volumes"].pop()["target"] == installer_image.BINDING_TARGET
        expected["environment"].pop(BINDING_ENVIRONMENT)
        expected["labels"] = actual["labels"]
        assert actual == expected
        # The installer's toolchain, not the profile's inherited NCCL library.
        assert actual["entrypoint"] == list(installer_image.ENTRYPOINT) and actual["command"][0] == "serve"
        assert actual["environment"]["VLLM_NCCL_SO_PATH"] == "/opt/sparkring/toolchain/nccl/lib/libnccl.so.2"
        assert actual["environment"]["CUDA_VERSION"] == installer_image.CUDA_VERSION
        assert not {"LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH", BINDING_ENVIRONMENT} & set(actual["environment"])


@pytest.mark.parametrize("profile", PROFILES)
def test_loader_policy_resolves_in_each_hosts_verified_checkout(profile):
    site = example_site(profile)
    manifest, files = compose.build(profile, site)
    assert loader_policy.RELATIVE in manifest["inputs"] and "runtime/common/loader_policy.py" in manifest["inputs"]
    for rank in site["ranks"]:
        service = yaml.safe_load(files[f"rank{rank['rank']}/compose.yaml"])["services"]["model"]
        assert service["security_opt"] == ["seccomp=" + rank["repository"] + "/" + loader_policy.RELATIVE]
        assert [volume["target"] for volume in service["volumes"]] == ["/models/target", "/cache"]


def test_native_profiles_do_not_select_an_installer_image_lock():
    site = example_site("qwen38-flash-next-tp2-sparkcache")
    manifest, _ = compose.build("qwen38-flash-next-tp2-sparkcache", site)
    assert "image_runtime" not in manifest
    with pytest.raises(ValueError, match="installer image lock"):
        compose.specifications("qwen38-flash-next-tp2-sparkcache", site,
                               image_runtime=installer_image.default_lock())


def test_changed_image_lock_in_a_deployment_is_rejected(tmp_path):
    profile = "glm53-flash-nvfp4-spark-tp2"
    target = tmp_path / "deployment"
    manifest = compose.render(profile, example_site(profile), target)
    assert compose.load_deployment(target)[0] == manifest
    changed = copy.deepcopy(manifest)
    changed["image_runtime"]["transport_manifest_sha256"] = "f" * 64
    (target / "deployment.json").write_text(compose.encoded(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="inputs changed"):
        compose.load_deployment(target)


@pytest.mark.parametrize("profile", PROFILES)
def test_host_inventory_verifies_every_adapter_input(tmp_path, profile):
    """Hosts check these files in their checkout before importing the coordinator."""
    snapshot = tmp_path / "source"
    for relative in compose.source_inventory(profile):
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(compose.ROOT / relative, destination)
    script = """
import sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root))
from runtime.common import compose, container_spec, installer_image, loader_policy
for module in (compose, container_spec, installer_image, loader_policy):
    assert Path(module.__file__).is_relative_to(root), module.__file__
assert installer_image.DEFAULT_LOCK.is_relative_to(root) and installer_image.default_lock()
assert loader_policy.PROFILE.is_relative_to(root) and loader_policy.inspection_option()
"""
    result = subprocess.run([sys.executable, "-I", "-B", "-c", script, str(snapshot)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


@pytest.mark.usefixtures("compose_cli")
@pytest.mark.parametrize("profile", compose.SUPPORTED)
def test_every_public_example_resolves_to_its_specification(profile, tmp_path):
    target = tmp_path / "deployment"
    manifest = compose.render(profile, example_site(profile), target)
    assert compose.check(target) == manifest
