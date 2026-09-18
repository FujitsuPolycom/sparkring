"""Offline profile/Compose equivalence and deployment drift checks."""

import copy
import json
import os
import shutil
import subprocess
import sys

import pytest
import yaml

from runtime.common import compose, qwen_flash_next
from runtime.common.container_spec import Bind, ContainerSpec, docker_create
from scripts import generate_compose_examples


@pytest.mark.parametrize("bind", [{}, {"create_host_path": False}])
def test_resolved_bind_options_preserve_omitted_false(bind):
    from types import SimpleNamespace

    spec = ContainerSpec(name="bind-options", image_id="sha256:" + "a" * 64,
                         entrypoint=("/python",), command=("serve",), environment={},
                         mounts=(Bind("/model", "/model", True), Bind("/cache", "/cache")))
    image = "example.invalid/model@sha256:" + "a" * 64
    actual = compose.escape({"name": spec.name, "services": {"model": compose.service(spec, image)}})
    for mount in actual["services"]["model"]["volumes"]:
        mount["bind"] = dict(bind)

    def run(argv, **kwargs):
        submitted = yaml.safe_load(kwargs["input"])["services"]["model"]["volumes"]
        assert all(mount["bind"]["create_host_path"] is False for mount in submitted)
        return SimpleNamespace(stdout=json.dumps(actual))

    result = compose.check_equivalence(spec, image, compose.compose_text(spec, image), run=run)
    assert all(mount["bind"]["create_host_path"] is False
               for mount in result["services"]["model"]["volumes"])


@pytest.mark.parametrize("bind", [{}, {"create_host_path": True}, {"create_host_path": "false"}])
def test_bind_creation_requires_explicit_false_in_input(bind):
    from types import SimpleNamespace

    spec = ContainerSpec(name="bind-input", image_id="sha256:" + "a" * 64,
                         entrypoint=("/python",), command=("serve",), environment={},
                         mounts=(Bind("/model", "/model", True),))
    image = "example.invalid/model@sha256:" + "a" * 64
    submitted = yaml.safe_load(compose.compose_text(spec, image))
    submitted["services"]["model"]["volumes"][0]["bind"] = bind
    # Omitted output flags cannot prove the policy requested by the input.
    resolved = compose.escape({"name": spec.name, "services": {"model": compose.service(spec, image)}})
    resolved["services"]["model"]["volumes"][0]["bind"] = {}
    with pytest.raises(ValueError, match="explicitly disable"):
        compose.check_equivalence(spec, image, yaml.safe_dump(submitted),
                                  run=lambda *a, **k: SimpleNamespace(stdout=json.dumps(resolved)))


@pytest.mark.parametrize("profile_id", sorted(compose.TP4_PROFILES))
def test_tp4_fabric_imports_from_only_its_packaged_inventory(tmp_path, profile_id):
    snapshot = tmp_path / "source"
    for relative in compose.source_inventory(profile_id):
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(compose.ROOT / relative, destination)
    script = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from runtime.common import qwen_mesh, glm_targets
assert Path(qwen_mesh.__file__).is_relative_to(Path(sys.argv[1]))
assert qwen_mesh._network().NetworkManager
assert glm_targets.target('nvidia-nvfp4')['repository'] == 'nvidia/GLM-5.3-Flash-NVFP4'
"""
    result = subprocess.run([sys.executable, "-I", "-B", "-c", script, str(snapshot)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.fixture
def site():
    return compose.read_site(
        compose.ROOT / "profiles/qwen38-flash-next-tp2/compose/site.example.yaml"
    )


@pytest.fixture(scope="module")
def compose_cli():
    available = (
        shutil.which("docker")
        and subprocess.run(
            ["docker", "compose", "version"], capture_output=True, timeout=20
        ).returncode
        == 0
    )
    if not available:
        if os.environ.get("SPARKRING_REQUIRE_COMPOSE") == "1":
            pytest.fail("Docker Compose CLI is required by this CI job")
        pytest.skip("Docker Compose CLI unavailable; no daemon is required")


@pytest.mark.parametrize("profile", ("qwen38-flash-next-tp2", "qwen38-flash-next-tp2-sparkcache"))
def test_tp2_backends_preserve_canonical_profile(site, profile, compose_cli):
    specs, image = compose.specifications(profile, site)
    for number, spec in enumerate(specs):
        result = compose.check_equivalence(
            spec, image, compose.compose_text(spec, image)
        )
        service = result["services"]["model"]
        argv = docker_create(spec)
        assert (
            argv[argv.index(spec.image_id) + 1 :]
            == service["entrypoint"][1:] + service["command"]
        )
        assert argv[argv.index("--entrypoint") + 1] == service["entrypoint"][0]
        from runtime.common import native_candidate
        assert spec.command[0] == native_candidate.ENTRYPOINT
        for option, expected in (
            ("--max-model-len", "262144"),
            ("--max-num-seqs", "16"),
            ("--max-num-batched-tokens", "8192"),
            ("--kv-cache-memory-bytes", "25769803776"),
        ):
            assert spec.command[spec.command.index(option) + 1] == expected
        assert ("--headless" in spec.command) == bool(number)
        assert bool(spec.health_command) == (number == 0)
        assert spec.environment["SIRCL_ENABLED"] == "0"
        assert spec.environment["VLLM_GLM53_MHC_PREFILL_SHARD"] == "0"
        assert spec.environment["B12X_ROCE_PEER_HCA_MAP"] == f"{1-number}=0/1"
        assert "@sha256:" in image and image != spec.image_id
        assert spec.mounts[0].read_only and not spec.mounts[1].read_only
        assert ("--kv-transfer-config" in spec.command) == profile.endswith(
            "-sparkcache"
        )
        assert ("--health-cmd" in argv) == (number == 0)


def test_site_hca_selection_is_shared_with_docker(site):
    site["ranks"][0].update(hcas=["mlx5_2", "mlx5_3"], gid=4)
    spec = compose.specifications(compose.SUPPORTED[0], site)[0][0]
    assert spec.environment["NCCL_IB_HCA"] == "=mlx5_2,mlx5_3"
    assert spec.environment["NCCL_IB_GID_INDEX"] == "4"
    assert "B12X_ROCE_HCA=mlx5_2,mlx5_3" in docker_create(spec)


def test_literal_dollars_spaces_json_and_ambient_variables(compose_cli, monkeypatch):
    monkeypatch.setenv("VALUE", "must-not-appear")
    spec = ContainerSpec(
        name="escape-check",
        image_id="sha256:" + "a" * 64,
        entrypoint=("/python",),
        command=(
            "serve",
            "$VALUE",
            "${VALUE}",
            "$$two",
            '{"value":"$VALUE with spaces"}',
        ),
        environment={"LITERAL": "$VALUE", "BRACED": "${VALUE}", "DOUBLE": "$$two"},
        mounts=(
            Bind("/models/$VALUE with spaces", "/model", True),
            Bind("/cache/${VALUE}", "/cache"),
        ),
    )
    result = compose.check_equivalence(
        spec,
        "example.invalid/model@sha256:" + "b" * 64,
        compose.compose_text(spec, "example.invalid/model@sha256:" + "b" * 64),
    )
    assert "must-not-appear" not in json.dumps(result)
    assert result["services"]["model"]["command"][1:4] == [
        "$$VALUE",
        "$${VALUE}",
        "$$$$two",
    ]
    # Docker receives literal values directly; Compose config re-escapes its output.
    assert "LITERAL=$VALUE" in docker_create(spec)
    assert "$VALUE" in docker_create(spec)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: s.update(image="example.invalid/model:mutable"),
        lambda s: s.update(entrypoint=["/bin/sh"]),
        lambda s: s.update(restart="always"),
        lambda s: s["environment"].update(SIRCL_ENABLED="1"),
        lambda s: s["command"].append("--headless"),
        lambda s: s["volumes"][0].update(read_only=False),
        lambda s: s["volumes"][0]["bind"].update(create_host_path=True),
        lambda s: s["deploy"]["resources"]["reservations"]["devices"][0].update(
            count=0
        ),
        lambda s: s.update(healthcheck={"disable": True}),
    ],
)
def test_resolved_semantic_drift_is_rejected(site, compose_cli, mutation):
    specs, image = compose.specifications(compose.SUPPORTED[0], site)
    value = yaml.safe_load(compose.compose_text(specs[0], image))
    mutation(value["services"]["model"])
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        compose.check_equivalence(specs[0], image, yaml.safe_dump(value))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: s.update(secret="test"),
        lambda s: s.update(master="192.0.2.11"),
        lambda s: s["ranks"][1].update(rank=0),
        lambda s: s["ranks"][1].update(host="spark0"),
        lambda s: s["ranks"][1].update(host_ip="192.0.2.10"),
        lambda s: s["ranks"][0].update(host="controller"),
        lambda s: s["ranks"][0].update(model="/models/../other"),
        lambda s: s["ranks"][0].update(cache="/srv/models"),
        lambda s: s["ranks"][0].update(
            repository=s["ranks"][0]["cache"] + "/repo"
        ),
        lambda s: s["ranks"][0].update(deployment_root="/"),
        lambda s: s["ranks"][0].update(hcas=["mlx5_0", "mlx5_0"]),
        lambda s: s["ranks"][0].update(gid=True),
        lambda s: s["ranks"][0].update(interface="eth0;evil"),
    ],
)
def test_invalid_site_never_creates_export(site, tmp_path, mutation):
    mutation(site)
    target = tmp_path / "deployment"
    with pytest.raises(ValueError):
        compose.render(compose.SUPPORTED[0], site, target)
    assert not target.exists()


def test_duplicate_yaml_key_is_rejected(tmp_path):
    path = tmp_path / "site.yaml"
    path.write_text("name: first\nname: second\n")
    with pytest.raises(ValueError, match="Duplicate"):
        compose.read_site(path)


def test_profile_unsupported_before_output_creation(site, tmp_path):
    with pytest.raises(ValueError, match="unsupported"):
        compose.render(
            "glm53-flash-spark-tp4-dcp1-sparkcache", site, tmp_path / "deployment"
        )
    assert not (tmp_path / "deployment").exists()


def test_edited_export_cannot_relabel_itself_canonical(site, tmp_path):
    directory = tmp_path / "deployment"
    compose.render(compose.SUPPORTED[0], site, directory)
    path = directory / "rank0/compose.yaml"
    path.write_text(path.read_text() + "# manual edit\n", newline="\n")
    with pytest.raises(ValueError, match="custom configuration"):
        compose.load_deployment(directory)
    manifest_path = directory / "deployment.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["rank0/compose.yaml"] = compose.digest(path.read_bytes())
    manifest_path.write_text(compose.encoded(manifest))
    with pytest.raises(ValueError, match="inputs changed"):
        compose.load_deployment(directory)


def test_source_drift_and_existing_output_are_rejected(site, tmp_path, monkeypatch):
    directory = tmp_path / "deployment"
    compose.render(compose.SUPPORTED[0], site, directory)
    with pytest.raises(FileExistsError):
        compose.render(compose.SUPPORTED[0], site, directory)
    inventory = copy.deepcopy(compose.source_inventory(compose.SUPPORTED[0]))
    inventory["runtime/common/qwen_flash_next.py"] = "0" * 64
    monkeypatch.setattr(compose, "source_inventory", lambda _: inventory)
    with pytest.raises(ValueError, match="inputs changed"):
        compose.load_deployment(directory)


def test_deterministic_private_export_passes_compose(site, tmp_path, compose_cli):
    target = tmp_path / "deployment"
    manifest = compose.render(compose.SUPPORTED[0], site, target)
    assert compose.check(target) == manifest
    assert compose.load_deployment(target)[0] == manifest
    assert not any(p.suffix == ".env" for p in target.rglob("*"))


def test_public_examples_match_generator():
    for path, expected in generate_compose_examples.examples():
        assert (
            path.read_bytes() == expected.encode()
        ), "Run scripts/generate_compose_examples.py"


def test_cli_help_has_compose_command():
    result = subprocess.run(
        [
            sys.executable,
            str(compose.ROOT / "scripts/sparkring.py"),
            "compose",
            "--help",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    assert (
        "render" in result.stdout
        and "check" in result.stdout
        and "start" in result.stdout
    )
