"""CPU tests of the serve launcher (``sparkring_sircl.vllm.serve``): site, profile, plan, commands, flows.

A synthetic SparkRing checkout stands in for the repository: a four-rank and a
two-rank profile with the same file layout and Compose rendering rules as the
repository's ``glm53-flash-nvfp4-spark-tp4`` and ``glm53-flash-nvfp4-spark-tp2``.
A fake SSH runner stands in for the Sparks and records every command, so the
tests prove what each operator command would run without contacting anything.
Three tests also plan the real profiles of the SparkRing checkout that
holds this package (or of ``SPARKRING_REPOSITORY`` when it names another), among them every profile of its catalog on the
ring of eight (the adapter RUNBOOK's "Installer profiles on the ring of
eight").
"""

from __future__ import annotations

import ast
import base64
import dataclasses
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from sparkring_sircl.ring import plan as ring_plan  # noqa: E402
from sparkring_sircl.ring import remote  # noqa: E402
from sparkring_sircl.ring.site import Site, SiteError  # noqa: E402
from sparkring_sircl.vllm import catalog, pins  # noqa: E402
from sparkring_sircl.vllm.serve import bundle, checks, cli, commands, probe, staging  # noqa: E402
from sparkring_sircl.vllm.serve import plan as plan_mod  # noqa: E402
from sparkring_sircl.vllm.serve import profile as profile_mod  # noqa: E402
from sparkring_sircl.vllm.serve.plan import Options, ServePlanError  # noqa: E402
from sparkring_sircl.vllm.serve.profile import ProfileError  # noqa: E402
from sparkring_sircl.vllm.serve.sitefile import ServeSite  # noqa: E402

REVISION = "0123456789abcdef0123456789abcdef01234567"
REPOSITORY = "local-inference-lab/GLM-5.3-Flash-NVFP4-Spark"
SLUG = "local-inference-lab--GLM-5.3-Flash-NVFP4-Spark"
RELEASE = "runtime/releases/example-release/release.json"
IMAGE_REFERENCE = "ghcr.io/example/sparkring@sha256:" + "1" * 64
IMAGE_ID = "sha256:" + "a" * 64
MODEL = "/srv/models/example"
MODEL_A = f"/srv/sparkring/cluster-a/checkpoints/{SLUG}/{REVISION}"
MODEL_B = f"/srv/sparkring/cluster-b/checkpoints/{SLUG}/{REVISION}"
SECCOMP = b'{"defaultAction": "SCMP_ACT_ERRNO", "syscalls": []}\n'
CONFIG_JSON = b'{"model_type": "glm5_next", "hidden_size": 4096}\n'
INDEX_JSON = b'{"weight_map": {}}\n'
# The SparkRing checkout whose real profiles three tests plan: SPARKRING_REPOSITORY, else the repository
# that holds this package (<repository>/spark_transport/sircl/tests).
CHECKOUT = os.environ.get("SPARKRING_REPOSITORY") or next(
    (str(root) for root in Path(__file__).resolve().parents[3:4] if (root / "profiles/catalog.json").is_file()),
    "")
needs_checkout = pytest.mark.skipif(not CHECKOUT, reason="no SparkRing checkout holds this package and "
                                    "SPARKRING_REPOSITORY names none")


@dataclasses.dataclass(frozen=True)
class Spec:
    """A synthetic profile, rendered the way the repository renders its Compose files."""

    id: str
    served: str
    tp: int
    port: int
    master_port: int
    topology: str
    environment: dict
    extra: tuple[str, ...]

    @property
    def recipe(self) -> list[str]:
        return ["--tensor-parallel-size", str(self.tp), "--nnodes", str(self.tp), "--port", str(self.port),
                "--master-port", str(self.master_port), "--max-model-len", "1048576", *self.extra,
                "--pipeline-parallel-size", "1", "--decode-context-parallel-size", "1",
                "--max-num-batched-tokens", "8192", "--host", "0.0.0.0"]


COMMON_ENV = {
    "VLLM_PLUGINS": "b12x_loader",
    "VLLM_ENABLE_ROCE_ALLREDUCE": "1",
    "SPARKRING_TRANSPORT_PROFILE": "tp2-rocenante-adaptive-prepared",
    "SPARKRING_TRANSPORT_MANIFEST_SHA256": "f" * 64,
    "SIRCL_ENABLED": "0",
    "VLLM_SPARK_TP4_MODE": "",
    "VLLM_GLM53_MHC_PREFILL_SHARD": "1",
    "B12X_COMPILE_CACHE_DIR": "/cache/example/b12x",
    "VLLM_B12X_MOE_FP4_FORCE_A16": "0",
    "CUTE_DSL_CACHE_DIR": "/cache/example/cute",
    "NCCL_IB_GID_INDEX": "3",
}
TP4 = Spec("example-tp4", "Example-Model-TP4", 4, 8015, 29775, "direct-cycle-4",
           {**COMMON_ENV, "B12X_ROCE_GID_INDEX": "3", "NCCL_ALGO": "Ring",
            "NCCL_IB_HCA": "=rocep1s0f0:1,rocep1s0f1:1,roceP2p1s0f0:1,roceP2p1s0f1:1"},
           ("--gpu-memory-utilization", "0.80",
            "--speculative-config", '{"method":"mtp","num_speculative_tokens":3,"draft_tensor_parallel_size":4}',
            "--compilation-config",
            '{"cudagraph_mode":"FULL_AND_PIECEWISE","pass_config":{"fuse_allreduce_rms":false}}'))
TP2 = Spec("example-tp2", "Example-Model-TP2", 2, 8000, 29500, "direct-pair-2",
           {**COMMON_ENV, "B12X_ROCE_PAIR_PATHS": "2", "NCCL_IB_HCA": "=rocep1s0f0,roceP2p1s0f0"},
           ("--gpu-memory-utilization", "0.87",
            "--speculative-config", '{"method":"mtp","num_speculative_tokens":3}',
            "--compilation-config", '{"mode":0,"cudagraph_mode":"FULL_AND_PIECEWISE"}'))
PROFILE, SERVED, RECIPE = TP4.id, TP4.served, TP4.recipe
PEER_MAPS = {4: ("1=0/2,2=0/3,3=1/3", "0=1/3,2=0/2,3=0/3", "0=1/2,1=1/3,3=0/2", "0=0/2,1=1/2,2=1/3"),
             2: ("1=0/1", "0=0/1")}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compose_service(spec: Spec, rank: int) -> dict:
    command = ["serve", "/models/target", "--served-model-name", spec.served, "--node-rank", str(rank),
               "--master-addr", "192.0.2.10", *spec.recipe] + (["--headless"] if rank else [])
    environment = {**spec.environment, "VLLM_PLUGINS": "b12x_loader,sparkring_status", "VLLM_SPARK_TP4_VOCAB_MODE": "",
                   "VLLM_HOST_IP": f"192.0.2.{10 + rank}", "GLOO_SOCKET_IFNAME": "eth0",
                   "NCCL_SOCKET_IFNAME": "eth0", "B12X_ROCE_PEER_HCA_MAP": PEER_MAPS[spec.tp][rank]}
    service = {
        "container_name": f"sr-example-r{rank}", "image": IMAGE_REFERENCE, "platform": "linux/arm64",
        "pull_policy": "never", "restart": "no", "init": True,
        "entrypoint": ["python3", "/opt/sparkring/toolchain/toolchain.py"], "command": command,
        "labels": {"io.sparkring.deployment": "d" * 64, "io.sparkring.rank": str(rank)},
        "network_mode": "host", "ipc": "host", "ulimits": {"memlock": {"soft": -1, "hard": -1}},
        "deploy": {"resources": {"reservations": {"devices": [
            {"driver": "nvidia", "count": "all", "capabilities": ["gpu"]}]}}},
        "devices": ["/dev/infiniband"],
        "volumes": [
            {"type": "bind", "source": "/srv/models/GLM-5.3-Flash-NVFP4-Spark/0123456789ab", "target": "/models/target",
             "read_only": True, "bind": {"create_host_path": False}},
            {"type": "bind", "source": f"/srv/cache/{spec.id}", "target": "/cache", "read_only": False,
             "bind": {"create_host_path": False}}],
        "mem_limit": 115964116992, "memswap_limit": 120259084288,
        "security_opt": ["seccomp=/opt/sparkring/runtime/common/loader-seccomp.json"],
        "environment": environment,
    }
    if rank == 0:
        service["healthcheck"] = {
            "test": ["CMD", "python3", "-c", "import urllib.request; urllib.request.urlopen("
                     f"'http://127.0.0.1:{spec.port}/health', timeout=4).close()"],
            "interval": "10s", "timeout": "5s", "start_period": "1800s", "retries": 3}
    return {"name": f"sr-example-r{rank}", "services": {"model": service}}


def write_repository(root: Path) -> Path:
    def put(relative: str, data: bytes | str) -> None:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data.encode() if isinstance(data, str) else data)

    weights = {f"model-0000{i}-of-00002.safetensors": bytes([i]) * (1000 + i) for i in (1, 2)}
    files = {"config.json": CONFIG_JSON, "model.safetensors.index.json": INDEX_JSON, **weights}
    for spec in (TP4, TP2):
        put(f"profiles/{spec.id}/profile.json", json.dumps({
            "schema": "sparkring-deployment/v1", "id": spec.id, "release": RELEASE,
            "configuration": {"format": "serving-profile", "path": f"profiles/{spec.id}/config.json"}}))
        put(f"profiles/{spec.id}/config.json", json.dumps({
            "schema": "sparkring-serving-profile/v1", "status": "implemented",
            "model": {"repository": REPOSITORY, "revision": REVISION, "config_sha256": sha(CONFIG_JSON),
                      "index_sha256": sha(INDEX_JSON)},
            "topology": spec.topology, "served_model_name": spec.served,
            "smoke": {"chat_template_kwargs": {"reasoning_effort": "low"}},
            "environment": spec.environment, "vllm_args": spec.recipe}))
        put(f"profiles/{spec.id}/SHA256SUMS", "".join(f"{sha(data)}  {name}\n" for name, data in files.items()))
        for rank in range(spec.tp):
            put(f"profiles/{spec.id}/compose/compose.rank{rank}.yaml", yaml.safe_dump(compose_service(spec, rank)))
    put(f"profiles/checkpoints/{SLUG}/{REVISION}.json", json.dumps(
        {"files": {name: {"sha256": sha(data), "size": len(data)} for name, data in files.items()}}))
    put("runtime/releases/example-release/installer-image.json", json.dumps(
        {"name": "example-release", "image_reference": IMAGE_REFERENCE, "image_id": IMAGE_ID,
         "profiles": [TP4.id, TP2.id]}))
    put(profile_mod.SECCOMP_RELATIVE, SECCOMP)
    put(profile_mod.THINKING_RELATIVE, json.dumps(THINKING))
    return root


# The stand-in checkpoint behaves like GLM-5.3-Flash's chat template (the repository's profiles/thinking.json).
THINKING = {
    "schema": "sparkring-thinking/v1",
    "behaviours": {
        "glm53-flash-template": {"default": "always", "level": "max", "levels": ["low", "high", "max"],
                                 "effort": "reasoning_effort", "off": None},
        "mimo-v26-flash-template": {"default": "on", "level": None, "levels": [], "effort": None,
                                    "off": {"enable_thinking": False}},
    },
    "checkpoints": {f"{REPOSITORY}@{REVISION}": "glm53-flash-template"},
}


SITE = {
    "schema": "sircl-ring-site/v1", "image": "sha256:aba309e4610c", "lan_interface": "enP7s7",
    "control_port": 29650, "remote_dir": "/tmp/sircl-ring", "docker": "sudo -n docker",
    "ring": [{"name": f"spark{i}", "ssh": f"user@192.0.2.{20 + i}", "lan_address": f"192.0.2.{20 + i}",
              **({"docker": "/usr/bin/docker"} if i == 2 else {})} for i in range(8)],
}
# The checkpoint at two paths: positions 0, 1, 2 and 7 hold one copy, positions 3-6 another.
SITE_WITH_PATHS = {**SITE, "ring": [{**entry, "model_path": MODEL_A if i in (0, 1, 2, 7) else MODEL_B}
                                    for i, entry in enumerate(SITE["ring"])]}


@pytest.fixture
def repository(tmp_path) -> Path:
    return write_repository(tmp_path / "sparkring")


@pytest.fixture
def site_file(tmp_path) -> Path:
    path = tmp_path / "site.json"
    path.write_text(json.dumps(SITE), encoding="utf-8")
    return path


# The plans and bundles of these tests let NCCL run where the cabling allows it (--nccl auto, the opt-in) unless
# a test names another mode; the launcher's own default is never
# (test_serve_and_bundle_share_one_nccl_default_and_state_every_groups_policy).
def make_plan(repository, *, spec=TP4, site=SITE, **options):
    profile = profile_mod.load(repository, spec.id)
    values = {"positions": tuple(range(spec.tp)), "model_path": MODEL, "nccl_mode": "auto", **options}
    return plan_mod.build_plan(ServeSite.from_json(site), profile, Options(**values),
                               staged_digest=staging.staged_tree().digest, library=staging.library_name())


def make_plans(repository, groups, *, spec=TP2, site=SITE, **options):
    profile = profile_mod.load(repository, spec.id)
    values = {"positions": groups[0], "model_path": MODEL, "nccl_mode": "auto", **options}
    return plan_mod.build_plans(ServeSite.from_json(site), profile, groups, Options(**values),
                                staged_digest=staging.staged_tree().digest, library=staging.library_name())


@pytest.fixture
def serve_plan(repository):
    return make_plan(repository)


def edit_compose(repository: Path, rank: int, change) -> None:
    path = repository / f"profiles/{PROFILE}/compose/compose.rank{rank}.yaml"
    document = yaml.safe_load(path.read_text())
    change(document["services"]["model"])
    path.write_text(yaml.safe_dump(document))


# -- profile ----------------------------------------------------------------------------------------


def test_the_profile_is_read_with_its_pins_and_request_settings(repository):
    profile = profile_mod.load(repository, PROFILE)
    assert (profile.tensor_parallel, profile.served_model_name, profile.api_port, profile.master_port) == (
        4, SERVED, 8015, 29775)
    assert profile.image_id == IMAGE_ID and profile.image_reference == IMAGE_REFERENCE
    assert profile.max_num_batched_tokens == 8192 and profile.topology == "direct-cycle-4"
    assert profile.request_settings == {"chat_template_kwargs": {"reasoning_effort": "low"}}
    assert [item.name for item in profile.checkpoint_files][:2] == ["config.json", "model.safetensors.index.json"]
    assert profile.checkpoint_dir_name == f"{SLUG}/{REVISION}"
    assert profile.seccomp_policy == SECCOMP
    assert f"profiles/{PROFILE}/compose/compose.rank3.yaml" in profile.sources
    pair = profile_mod.load(repository, TP2.id)
    assert (pair.tensor_parallel, pair.api_port, pair.master_port, pair.topology) == (2, 8000, 29500, "direct-pair-2")


@pytest.mark.parametrize("change, message", [
    (lambda service: service["environment"].update({"NCCL_ALGO": "Tree"}), "NCCL_ALGO is 'Tree'"),
    (lambda service: service.update({"shm_size": "8g"}), "no docker run translation"),
    (lambda service: service.update({"image": "ghcr.io/example/other:1"}), "differs from the release"),
    (lambda service: service["command"].remove("--headless"), "differ from config.json's recipe"),
    (lambda service: service.update({"deploy": {}}), "deploy must reserve all NVIDIA GPUs"),
    (lambda service: service["environment"].update({"VLLM_PLUGINS": "b12x_loader,other"}), "VLLM_PLUGINS"),
])
def test_profile_drift_is_reported_with_the_file_that_differs(repository, change, message):
    edit_compose(repository, 1, change)
    with pytest.raises(ProfileError, match=re.escape(message)) as raised:
        profile_mod.load(repository, PROFILE)
    assert "compose.rank1.yaml" in str(raised.value)


def test_profiles_in_other_formats_and_search_paths_the_installer_removes(repository):
    deployment = repository / f"profiles/{PROFILE}/profile.json"
    record = json.loads(deployment.read_text())
    deployment.write_text(json.dumps({**record, "configuration": {
        "format": "recipe", "path": f"profiles/{PROFILE}/recipe.json"},
        "launcher": {"kind": "bash", "path": "scripts/example_serve.sh"}}))
    with pytest.raises(ProfileError, match=r"recipe profile .* runs from its own launcher "
                                           r"\(scripts/example_serve.sh\).*bundle"):
        profile_mod.load(repository, PROFILE)
    # A serving profile whose configuration lives elsewhere, as SparkRing's cache variants keep theirs.
    shutil.copy(repository / f"profiles/{PROFILE}/config.json", repository / f"profiles/{PROFILE}/variant.json")
    deployment.write_text(json.dumps({**record, "configuration": {
        "format": "serving-profile", "path": f"profiles/{PROFILE}/variant.json"}}))
    assert f"profiles/{PROFILE}/variant.json" in profile_mod.load(repository, PROFILE).sources
    # The installer image's adaptation drops PYTHONPATH from every rendered container.
    path = repository / f"profiles/{PROFILE}/variant.json"
    config = json.loads(path.read_text())
    config["environment"]["PYTHONPATH"] = ""
    path.write_text(json.dumps(config))
    assert profile_mod.load(repository, PROFILE).recipe_environment["PYTHONPATH"] == ""
    config["topology"] = "switched"
    path.write_text(json.dumps(config))
    with pytest.raises(ProfileError, match="switched fabric"):
        profile_mod.load(repository, PROFILE)


def test_a_checkpoint_pin_that_differs_from_sha256sums_is_drift(repository):
    path = repository / f"profiles/{PROFILE}/config.json"
    config = json.loads(path.read_text())
    config["model"]["index_sha256"] = "0" * 64
    path.write_text(json.dumps(config))
    with pytest.raises(ProfileError, match="model.safetensors.index.json"):
        profile_mod.load(repository, PROFILE)


# -- site file --------------------------------------------------------------------------------------


def test_the_site_file_adds_per_spark_model_paths_and_host_sudo_prefixes():
    document = {**SITE_WITH_PATHS, "ring": [dict(entry) for entry in SITE_WITH_PATHS["ring"]]}
    document["ring"][5]["sudo"] = ""                       # explicit: plain although its docker uses sudo
    document["ring"][2]["sudo"] = "sudo -n -u root"         # explicit: sudo although its docker is plain
    site = ServeSite.from_json(document)
    assert site.model_paths[0] == MODEL_A and site.model_paths[3] == MODEL_B and site.model_paths[7] == MODEL_A
    assert site.sudo[0] == "sudo -n" and site.sudo_sources[0] == "the Spark's Docker command"
    assert site.sudo[2] == "sudo -n -u root" and site.sudo_sources[2] == "site file, ring entry"
    assert site.sudo[5] == "" and site.sudo_sources[5] == "site file, ring entry"
    assert Site.from_json(document).host(0).ssh == "user@192.0.2.20"   # the harness's reader ignores both keys
    plain = ServeSite.from_json({**SITE, "docker": "docker", "sudo": "sudo -n"})
    assert set(plain.sudo.values()) == {"sudo -n"} and plain.sudo_sources[0] == "site file"
    untouched = ServeSite.from_json({**SITE, "docker": "docker",
                                     "ring": [{key: value for key, value in entry.items() if key != "docker"}
                                              for entry in SITE["ring"]]})
    assert set(untouched.sudo.values()) == {""} and untouched.model_paths == {}


@pytest.mark.parametrize("entry, message", [
    ({"model_path": "relative/path"}, "model_path 'relative/path' must be an absolute path"),
    ({"model_path": "/srv/a,b"}, "must be an absolute path"),
    ({"model_path": "/srv/../etc"}, "'..' component"),
    ({"model_path": "/srv/REPLACE_ME"}, "must be an absolute path"),
    ({"sudo": "doas"}, "must start with sudo"),
    ({"sudo": "sudo -n; rm"}, "characters a shell would read"),
    ({"sudo": 1}, "must be a string"),
])
def test_bad_site_keys_are_refused(entry, message):
    document = {**SITE, "ring": [dict(item) for item in SITE["ring"]]}
    document["ring"][4].update(entry)
    with pytest.raises(SiteError, match=re.escape(message)):
        ServeSite.from_json(document)


# -- plan -----------------------------------------------------------------------------------------


def test_every_rank_runs_the_profile_with_sircl_in_front_of_every_collective(serve_plan):
    profile = serve_plan.profile
    assert serve_plan.run_id == "tp4-0-3"
    assert serve_plan.group.fabric.describe() == "path:0-1-2-3" and serve_plan.nccl_policy.value == "none"
    for launch, container in zip(serve_plan.ranks, profile.ranks):
        env = launch.environment
        assert env["SIRCL_MODE"] == "custom" and env["SIRCL_FABRIC"] == "ring:8"
        assert env["SIRCL_RANK_POSITIONS"] == "0,1,2,3" and env["SIRCL_GROUPS"] == "tp"
        assert env["SIRCL_NCCL"] == "auto" and env["SIRCL_LARGE_ALLREDUCE"] == "auto"
        assert env["SIRCL_SESSION_MODULE"] == "sparkring_sircl.oneshot"
        assert env["SIRCL_NATIVE_LIBRARY"] == f"/sircl/build-cache/{serve_plan.library}"
        assert env["SIRCL_ALLREDUCE_CAPACITY_BYTES"] == env["SIRCL_ALLREDUCE_DISPATCH_LIMIT_BYTES"] == "131072"
        assert env["SIRCL_ALLGATHER_MAX_BYTES"] == "155648" and env["SIRCL_GID_INDEX"] == "3"
        assert env["PYTHONPATH"] == "/sircl/src" and env["SIRCL_RECEIPT_DIR"] == "/sircl/run/receipts"
        assert env["VLLM_PLUGINS"] == "b12x_loader,sparkring_status,sircl"
        for name, value in plan_mod.DISABLED_TRANSPORTS.items():
            assert env[name] == value
        assert env["VLLM_GLM53_MHC_PREFILL_SHARD"] == "1"           # the profile's value; the shim carries it
        assert env["VLLM_HOST_IP"] == SITE["ring"][launch.position]["lan_address"]
        assert env["GLOO_SOCKET_IFNAME"] == env["NCCL_SOCKET_IFNAME"] == "enP7s7"
        for name, value in container.environment.items():      # compile-cache keys and NCCL's devices unchanged
            if name.startswith(("B12X_", "CUTE_", "NCCL_IB", "NCCL_ALGO")):
                assert env[name] == value
        assert not any(name in env for name in ("SIRCL_PEER_ROUTES", "SIRCL_LAYOUT", "VLLM_DISABLE_PYNCCL"))
        command = list(launch.command)
        assert command[command.index("--port") + 1] == "8017"
        assert command[command.index("--master-addr") + 1] == "192.0.2.20"
        assert command[command.index("--master-port") + 1] == "29775"
        assert ("--headless" in command) == (launch.rank > 0)
        assert command[command.index("--speculative-config") + 1] == RECIPE[RECIPE.index("--speculative-config") + 1]
        assert launch.container == f"sircl-serve-tp4-0-3-r{launch.rank}"
        assert launch.labels == {"sircl-serve": "tp4-0-3", "sircl-serve.rank": str(launch.rank),
                                 "sircl-serve.position": str(launch.position)}
        assert launch.image == IMAGE_ID and launch.model_source == "--model-path"
        mounts = {mount.target: mount for mount in launch.mounts}
        assert mounts["/models/target"].source == MODEL and mounts["/models/target"].read_only
        assert mounts["/cache"].source == f"/tmp/sircl-ring/serve/cache/{PROFILE}" and not mounts["/cache"].read_only
        assert mounts["/sircl/src"].source == f"/tmp/sircl-ring/serve/src/{serve_plan.staged_digest}"
        assert mounts["/sircl/src"].read_only and mounts["/sircl/build-cache"].read_only
        assert mounts["/sircl/run"].source == "/tmp/sircl-ring/serve/runs/tp4-0-3" and not mounts["/sircl/run"].read_only
    rank0 = serve_plan.ranks[0]
    assert rank0.health and ":8017/health" in rank0.health[-1]
    assert all(launch.health is None for launch in serve_plan.ranks[1:])
    assert "io.sparkring" not in json.dumps([launch.labels for launch in serve_plan.ranks])
    changed = {change.name for change in serve_plan.changes}
    assert {"VLLM_ENABLE_ROCE_ALLREDUCE", "SPARKRING_TRANSPORT_PROFILE", "VLLM_PLUGINS", "PYTHONPATH"} <= changed
    assert "NCCL_IB_HCA" not in changed and "VLLM_GLM53_MHC_PREFILL_SHARD" not in changed
    assert serve_plan.mhc_prefill_shard and serve_plan.mhc_world == 4


def test_docker_commands_use_each_sparks_docker_command_and_quote_every_argument(serve_plan):
    for launch in serve_plan.ranks:
        words = shlex.split(launch.shell())
        prefix = launch.docker.split()
        assert words == prefix + launch.argv()
        assert words[len(prefix):len(prefix) + 4] == ["run", "-d", "--name", launch.container]
        assert "--mount" in words and "-v" not in words
        assert f"seccomp={serve_plan.seccomp_path}" in words
        assert words[words.index("--entrypoint") + 1] == "python3"
        image_at = words.index(IMAGE_ID)
        assert words[image_at + 1:image_at + 3] == ["/opt/sparkring/toolchain/toolchain.py", "serve"]
        assert words[words.index("SIRCL_MODE=custom") - 1] == "--env"
    assert serve_plan.ranks[0].shell().startswith("sudo -n docker run -d ")
    assert serve_plan.ranks[2].shell().startswith("/usr/bin/docker run -d ")


@pytest.mark.parametrize("options, message", [
    (dict(positions=(0, 1, 2)), "serves 4 ranks"),
    (dict(positions=(0, 1, 1, 2)), "distinct"),
    (dict(positions=(0, 2, 4, 6)), "are not consecutive Sparks of the ring:8 layout"),
    (dict(model_path=None, model_paths={0: MODEL}), "no model directory for Sparks [1, 2, 3]"),
    (dict(model_path="relative/path"), "absolute path"),
    (dict(api_port=29775), "master port"),
    (dict(capacity=131072, dispatch=262144), "cannot exceed"),
    (dict(oneshot_max=262144), "--oneshot-max 262144 exceeds the dispatch ceiling of 131072 bytes"),
    (dict(oneshot_max=100), "--oneshot-max must be a non-negative multiple of 16 bytes"),
    (dict(gather=100), "multiple of 16"),
    (dict(run_id="Bad Name"), "run id"),
    (dict(large_allreduce="nccl"), "SIRCL_LARGE_ALLREDUCE=nccl needs NCCL"),
    (dict(nccl_mode="sometimes"), "SIRCL_NCCL must be one of"),
])
def test_plans_that_cannot_run_are_refused(repository, options, message):
    with pytest.raises(ServePlanError, match=re.escape(message)):
        make_plan(repository, **options)


def test_recipe_settings_sircl_cannot_carry_stop_the_plan_and_row_ownership_is_carried(repository):
    shard = make_plan(repository, extra_env={"VLLM_QWEN3_8_HC_PREFILL_MODE": "shard"})
    assert shard.hc_prefill_mode == "shard" and shard.to_json()["hc_prefill_mode"] == "shard"
    assert ("hyper-connection prefill row ownership: shard (VLLM_QWEN3_8_HC_PREFILL_MODE), carried by SIRCL's "
            "reduce-scatter and all-gather (shim qwen_hc_prefill_shard)") in plan_mod.render_text(shard)
    pair = make_plan(repository, spec=TP2, extra_env={"VLLM_QWEN3_8_HC_PREFILL_MODE": "shard"})
    assert pair.nccl_policy.value == "all"                  # PyNccl runs on a pair and carries it
    assert "row ownership: shard (VLLM_QWEN3_8_HC_PREFILL_MODE), carried by vLLM's PyNccl path" in (
        plan_mod.render_text(pair))
    plain = make_plan(repository)
    assert plain.hc_prefill_mode is None and "hyper-connection" not in plan_mod.render_text(plain)
    assert plan_mod.relay_conflicts([], {"VLLM_QWEN3_8_HC_PREFILL_MODE": "shard"}) == []
    recipe = ["--compilation-config", '{"pass_config":{"fuse_allreduce_rms":true}}',
              "--enable-batch-sharded-sampling", "--enable-expert-parallel", "--all2all-backend",
              "deepep_low_latency"]
    found = plan_mod.relay_conflicts(recipe, {})
    assert len(found) == 3 and "pass_config.fuse_allreduce_rms" in found[0] and "uneven" in found[1]
    assert "deepep_low_latency" in found[2]
    assert plan_mod.relay_conflicts(["--compilation-config", '{"pass_config":{"fuse_act_quant":true}}'], {}) == []


def test_a_profile_that_sets_a_variable_the_launcher_owns_is_refused(repository):
    for rank in range(4):
        edit_compose(repository, rank, lambda service: service["environment"].update({"SIRCL_PEER_ROUTES": "1=a"}))
    with pytest.raises(ServePlanError, match="SIRCL_PEER_ROUTES"):
        make_plan(repository)


def test_model_paths_come_from_the_command_line_and_the_site_file_in_that_order(repository):
    plan = make_plan(repository, site=SITE_WITH_PATHS, model_path=None)
    assert [launch.mount("/models/target").source for launch in plan.ranks] == [MODEL_A, MODEL_A, MODEL_A, MODEL_B]
    assert {launch.model_source for launch in plan.ranks} == {"site file model_path"}
    plan = make_plan(repository, site=SITE_WITH_PATHS, model_path="/srv/shared", model_paths={3: "/srv/own"})
    assert [launch.mount("/models/target").source for launch in plan.ranks] == [MODEL_A, MODEL_A, MODEL_A, "/srv/own"]
    assert plan.ranks[3].model_source == "--model-path 3=PATH"
    plan = make_plan(repository, model_path="/srv/shared")          # a site without model_path keys
    assert {launch.mount("/models/target").source for launch in plan.ranks} == {"/srv/shared"}
    text = plan_mod.render_text(make_plan(repository, site=SITE_WITH_PATHS, model_path=None))
    assert f"mount {MODEL_B} -> /models/target (ro, site file model_path)" in text
    shared, per_spark = plan_mod.split_paths([MODEL, "3=/srv/own"], (0, 1, 2, 3), "--model-path")
    assert shared == MODEL and per_spark == {3: "/srv/own"}
    with pytest.raises(ServePlanError, match="serves no rank"):
        plan_mod.split_paths(["5=/x"], (0, 1, 2, 3), "--model-path")
    with pytest.raises(ServePlanError, match="twice"):
        plan_mod.split_paths(["/a", "/b"], (0, 1, 2, 3), "--model-path")


def test_a_group_across_the_rings_last_cable_routes_and_relays_through_sparks_zero_and_one(repository):
    assert plan_mod.parse_positions("7-2", 8) == plan_mod.parse_positions("7,0,1,2", 8) == (7, 0, 1, 2)
    with pytest.raises(ServePlanError, match="needs the ring's size"):
        plan_mod.parse_positions("7-2")
    plan = make_plan(repository, site=SITE_WITH_PATHS, model_path=None, positions=(7, 0, 1, 2))
    assert plan.run_id == "tp4-7-2" and plan.group.fabric.describe() == "path:7-0-1-2"
    assert plan.nccl_policy.value == "none" and "positions 2-7" in plan.nccl_reason
    assert plan.ranks[0].environment["SIRCL_RANK_POSITIONS"] == "7,0,1,2"
    command = list(plan.ranks[1].command)
    assert command[command.index("--master-addr") + 1] == "192.0.2.27"      # rank 0 is on Spark 7
    assert plan.group.session_layout() == ring_plan.layout_text(ring_plan.group_layout(8, (7, 0, 1, 2)))
    assert plan.group.session_layout() == "cables=7.port0-0.port1,0.port0-1.port1,1.port0-2.port1;positions=7,0,1,2"
    record = plan.to_json()
    assert record["route_maps"]["0"] == "1=rocep1s0f0/roceP2p1s0f0,2=rocep1s0f0/roceP2p1s0f0,3=rocep1s0f0/roceP2p1s0f0"
    assert record["route_maps"]["3"] == "0=rocep1s0f1/roceP2p1s0f1,1=rocep1s0f1/roceP2p1s0f1,2=rocep1s0f1/roceP2p1s0f1"
    assert record["relays"] == [
        {"ranks": [0, 2], "positions": [7, 1], "lanes": 2, "relays": [0]},
        {"ranks": [0, 3], "positions": [7, 2], "lanes": 2, "relays": [0, 1]},
        {"ranks": [1, 3], "positions": [0, 2], "lanes": 2, "relays": [1]}]
    assert record["max_relays"] == 2 and record["relay_factor"] == 1.0
    text = plan_mod.render_text(plan)
    for expected in ("derived route map, rank 0 (position 7): 1=rocep1s0f0/roceP2p1s0f0",
                     "ranks 0-3 (positions 7-2): 2 lanes through positions 0, 1",
                     "ranks 0-2 (positions 7-1): 2 lanes through position 0",
                     "at most 2 relays per lane; relay-load factor 1"):
        assert expected in text
    second = make_plan(repository, site=SITE_WITH_PATHS, model_path=None, positions=(4, 5, 6, 7))
    assert [launch.mount("/models/target").source for launch in second.ranks] == [MODEL_B, MODEL_B, MODEL_B, MODEL_A]
    assert [row["relays"] for row in second.to_json()["relays"]] == [[5], [5, 6], [6]]


def test_four_pairs_at_once_get_their_own_ports_and_nccl_devices_facing_the_partner(repository):
    groups = plan_mod.parse_groups("0-1;2-3;4-5;6-7", 8)
    plans = make_plans(repository, groups, site=SITE_WITH_PATHS, model_path=None)
    assert [plan.run_id for plan in plans] == ["tp2-0-1", "tp2-2-3", "tp2-4-5", "tp2-6-7"]
    assert [plan.api_port for plan in plans] == [8017, 8018, 8019, 8020]
    assert [plan.master_port for plan in plans] == [29500, 29501, 29502, 29503]
    for plan in plans:
        assert plan.nccl_policy.value == "all" and plan.group.fabric.describe().startswith("pair:")
        assert plan.to_json()["relays"] == [] and plan.prefill() is None
        first, second = plan.ranks
        assert first.environment["NCCL_IB_HCA"] == "=rocep1s0f0,roceP2p1s0f0"      # port 0 faces the next Spark
        assert second.environment["NCCL_IB_HCA"] == "=rocep1s0f1,roceP2p1s0f1"     # port 1 faces the previous one
        assert first.environment["VLLM_GLM53_MHC_PREFILL_SHARD"] == "1"             # PyNccl may run on a pair
        assert first.environment["SIRCL_RANK_POSITIONS"] == ",".join(str(p) for p in plan.positions)
        assert first.health and f":{plan.api_port}/health" in first.health[-1]
        command = list(second.command)
        assert command[command.index("--master-port") + 1] == str(plan.master_port)
        assert command[command.index("--master-addr") + 1] == first.lan_address
    assert plans[1].ranks[1].mount("/models/target").source == MODEL_B             # Spark 3
    text = plan_mod.render_text(plans[0])
    assert "this group is pair:0-1 with NCCL policy all (every pair of ranks shares a cable)" in text
    assert "relays: none" in text and "NCCL carries larger eager all-reduces" in text
    assert "expected prefill: all-reduces above the dispatch ceiling run on NCCL" in text
    assert "NCCL_IB_HCA=" + "=rocep1s0f1,roceP2p1s0f1" in text
    wrapped = make_plans(repository, plan_mod.parse_groups("7-0", 8))[0]
    assert wrapped.run_id == "tp2-7-0" and wrapped.ranks[0].environment["NCCL_IB_HCA"] == "=rocep1s0f0,roceP2p1s0f0"
    assert wrapped.ranks[1].environment["NCCL_IB_HCA"] == "=rocep1s0f1,roceP2p1s0f1"


def test_a_pair_without_nccl_or_with_sircl_for_large_messages(repository):
    never = make_plans(repository, ((2, 3),), nccl_mode="never")[0]
    assert never.nccl_policy.value == "none" and never.nccl_reason == "SIRCL_NCCL=never"
    assert never.ranks[0].environment["VLLM_GLM53_MHC_PREFILL_SHARD"] == "1" and never.mhc_world == 2
    assert never.ranks[1].environment["NCCL_IB_HCA"] == "=rocep1s0f0,roceP2p1s0f0"   # NCCL never runs: unchanged
    # In each of two chunks: 4 all-reduces and 90 reduce-scatters of 64 MiB, 90 gathers of 4,096-row shards
    estimate = never.prefill()
    assert (estimate.allreduces, estimate.scatters, estimate.gathers) == (
        ((8, 64 << 20),), ((180, 64 << 20),), ((180, 4096, 8192),))
    sircl = make_plans(repository, ((2, 3),), large_allreduce="sircl")[0]
    assert sircl.nccl_policy.value == "all" and sircl.prefill() is not None
    assert "every all-reduce runs on SIRCL" in plan_mod.render_text(sircl)


def test_mhc_prefill_sharding_follows_the_profile_unless_turned_off(repository):
    on = make_plan(repository)
    text = plan_mod.render_text(on)
    assert "mHC prefill sharding: on (VLLM_GLM53_MHC_PREFILL_SHARD of the profile), carried by SIRCL's " \
           "reduce-scatter and all-gather (shim mhc_prefill_shard)" in text
    assert on.to_json()["mhc_prefill_shard"] is True and on.prefill().gathers == ((180, 2048, 8192),)
    off = make_plan(repository, mhc_prefill_shard="off")
    assert {launch.environment["VLLM_GLM53_MHC_PREFILL_SHARD"] for launch in off.ranks} == {"0"}
    assert not off.mhc_prefill_shard and off.mhc_world is None and off.prefill().gathers == ()
    change = next(change for change in off.changes if change.name == "VLLM_GLM53_MHC_PREFILL_SHARD")
    assert (change.before, change.after) == ("1", "0") and "--mhc-prefill-shard off" in change.reason
    assert "mHC prefill sharding: off" in plan_mod.render_text(off)
    pair = make_plans(repository, ((0, 1),))[0]
    assert pair.mhc_prefill_shard and pair.mhc_world is None            # PyNccl carries it on a pair
    assert "carried by vLLM's PyNccl path (NCCL may run here)" in plan_mod.render_text(pair)
    with pytest.raises(ServePlanError, match="--mhc-prefill-shard"):
        make_plan(repository, mhc_prefill_shard="sometimes")


def test_instances_must_use_disjoint_sparks_and_distinct_ports(repository):
    with pytest.raises(ServePlanError, match=r"Sparks \[1\] appear in more than one group"):
        make_plans(repository, ((0, 1), (1, 2)))
    with pytest.raises(ServePlanError, match="ports overlap"):
        make_plans(repository, ((0, 1), (2, 3), (4, 5)), master_port=8018)
    plans = make_plans(repository, ((0, 1), (2, 3)), run_id="pair")
    assert [plan.run_id for plan in plans] == ["pair-0-1", "pair-2-3"]


def test_sixteen_thousand_token_prefill_estimate():
    estimate = plan_mod.prefill_estimate(16384, chunk_tokens=8192)
    assert (estimate.chunks, estimate.reductions_per_chunk, estimate.allreduces) == (2, 94, ((188, 64 << 20),))
    assert estimate.oneshot_ops == 96256 and estimate.gathers == () and estimate.scatters == ()
    assert [round(seconds, 2) for seconds in estimate.collective_bound_seconds] == [4.62, 4.81]
    assert round(estimate.compute_seconds, 2) == 4.65
    low, high = estimate.bound_seconds
    assert 9.2 < low < high < 9.5
    short = plan_mod.prefill_estimate(50, chunk_tokens=8192)
    assert short.allreduces == ((94, 409600),) and short.oneshot_ops == 94 * 4
    unknown = plan_mod.prefill_estimate(16384, chunk_tokens=8192, prefill_rate=None)
    assert unknown.bound_seconds is None and "compute unknown" in unknown.describe()
    # The one-shot path measurement (17,055 tokens in 9.37 s, mHC sharding off) is reproduced by its compute rate.
    measured = plan_mod.prefill_estimate(17055, chunk_tokens=8192, compute_per_token=0.26e-3)
    assert measured.oneshot_ops == 100204 and 9.2 < measured.bound_seconds[0] < 9.37 < measured.bound_seconds[1]
    # Row ownership at TP4: per full chunk 90 reduce-scatters of 64 MiB in place of all-reduces and 90 gathers
    # of 2,048-row shards, 0.41 s less mixing; the one-shot bound prices a reduce-scatter as an all-reduce.
    sharded = plan_mod.prefill_estimate(16384, chunk_tokens=8192, compute_per_token=0.26e-3, mhc_world=4)
    assert (sharded.sharded_chunks, sharded.allreduces, sharded.scatters, sharded.gathers) == (
        2, ((8, 64 << 20),), ((180, 64 << 20),), ((180, 2048, 8192),))
    assert sharded.oneshot_ops == 188 * 512 + 180 * 128
    assert round(sharded.mhc_saving_seconds, 3) == 0.825
    assert ("94 reductions per chunk (64 MiB in a full chunk); with mHC row ownership in 2 chunk(s) they are 8 "
            "all-reduces and 180 reduce-scatters (the session's reduce-scatter where it has one, otherwise "
            "all-reduces and this rank's rows), plus 180 all-gathers of 16 MiB shards") in sharded.describe()
    assert "0.8 s less mHC mixing with row ownership" in sharded.describe()


def test_session_ops_come_from_the_session_statistics_and_price_a_measured_prefill():
    stats = {"dispatch_limit_bytes": 131072, "large_piece_bytes": 4 << 20, "gather_piece_bytes": 4 << 20,
             "large_schedule": "auto", "chain_available": False, "chain_min_bytes": 1 << 20}
    session = plan_mod.SessionOps.from_stats(stats)
    assert session == plan_mod.SessionOps(131072, 4 << 20, 4 << 20, None)
    assert [session.allreduce_ops(n) for n in (16, 131072, 131088, 64 << 20, (64 << 20) + 8)] == [1, 1, 1, 16, 17]
    assert session.gather_ops(2048, 8192) == 4 and session.gather_ops(3, 5 << 20) == 6
    two_mib = plan_mod.SessionOps.from_stats({**stats, "large_piece_bytes": 2 << 20, "gather_piece_bytes": 155648})
    assert two_mib.allreduce_ops(64 << 20) == 32 and two_mib.gather_ops(2048, 8192) == 108
    chained = plan_mod.SessionOps.from_stats({**stats, "chain_available": True})
    assert chained.chain_min == 1 << 20 and chained.allreduce_ops(64 << 20) == 1
    assert chained.allreduce_ops(512 << 10) == 1 and "one chain op" in chained.describe()
    assert plan_mod.SessionOps.from_stats({**stats, "chain_available": True, "large_schedule": "pieces"}).chain_min is None
    assert plan_mod.SessionOps.from_stats({"large_piece_bytes": 4 << 20}) is None
    assert plan_mod.SessionOps.from_stats(None) is None
    # 17,057 tokens: two full chunks and 673 rows; 94 all-reduces each; 16 + 16 + 2 pieces of 4 MiB.
    measured = plan_mod.prefill_estimate(17057, chunk_tokens=8192, compute_per_token=0.26e-3)
    assert measured.ops(session) == (94 * (16 + 16 + 2), 0, 0)
    per_op = measured.implied_op_seconds(6.25, session)
    assert 0.56e-3 < per_op < 0.58e-3
    # Without a session reduce-scatter the 180 reduce-scatters are all-reduces of 16 pieces each.
    sharded = plan_mod.prefill_estimate(17057, chunk_tokens=8192, compute_per_token=0.26e-3, mhc_world=4)
    assert "reduce-scatters carried as all-reduces" in session.describe()
    assert sharded.ops(session) == (94 * 34, 0, 180 * 4)


def test_plan_text_and_json_name_everything_the_operator_checks(serve_plan):
    text = plan_mod.render_text(serve_plan)
    for expected in ("path:0-1-2-3 with NCCL policy none", "every collective runs on SIRCL",
                     "direct-cycle-4", "/srv/models/GLM-5.3-Flash-NVFP4-Spark/0123456789ab (documentation path)",
                     f"/srv/sparkring/<cluster>/checkpoints/{SLUG}/{REVISION}",
                     "8 all-reduces and 180 reduce-scatters", "180 all-gathers of 16 MiB shards",
                     "--port 8015 -> 8017",
                     "docker command \"/usr/bin/docker\"",
                     "SIRCL flag waits: startup regime up to 600 s", "serving regime up to 20 s from the first step "
                     "after warm-up", "SIRCL_SPIN_LIMIT unset",
                     "host file operations through \"sudo -n\" (the Spark's Docker command)",
                     "host file operations through none (default)"):
        assert expected in text
    record = serve_plan.to_json()
    assert record["schema"] == "sircl-serve-plan/v1" and record["nccl"] == "none"
    assert record["route_maps"]["0"] == "1=rocep1s0f0/roceP2p1s0f0,2=rocep1s0f0/roceP2p1s0f0,3=rocep1s0f0/roceP2p1s0f0"
    estimate = json.loads(json.dumps(record))["prefill_estimate_16k"]
    assert estimate["gathers"] == [[180, 2048, 8192]] and estimate["scatters"] == [[180, 64 << 20]]
    assert record["mhc_prefill_shard"] is True
    assert record["waits"] == {"startup_s": 600.0, "serving_s": 20.0} and record["sizes"]["spin_limit"] is None
    assert record["ranks"][0]["sudo"] == "sudo -n" and record["ranks"][2]["sudo"] == ""
    json.dumps(record)


# -- staging and probe --------------------------------------------------------------------------------


def test_the_staged_tree_carries_the_package_and_the_entry_points_of_pyproject(tmp_path):
    pyproject = tomllib.loads((staging.PROJECT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = {group: dict(points) for group, points in pyproject["project"]["entry-points"].items()
                if group.startswith("vllm.")}
    assert declared == {group: dict(points) for group, points in staging.ENTRY_POINTS.items()}
    tree = staging.staged_tree()
    names = tree.names()
    assert "sparkring_sircl/vllm/serve/probe.py" in names and "sparkring_sircl/oneshot/_roce_proxy.c" in names
    assert not any("__pycache__" in name for name in names)
    assert tree.digest == staging.staged_tree().digest
    with tarfile.open(fileobj=io.BytesIO(tree.tar())) as archive:
        archive.extractall(tmp_path / "src", filter="data")
    folder = tmp_path / "src"
    assert (folder / "sparkring_sircl/vllm/plugin.py").read_bytes() == (staging.PACKAGE / "vllm/plugin.py").read_bytes()
    points = probe.entry_points([str(folder)])
    assert ["sircl", "sparkring_sircl.vllm.plugin:register", "sparkring-sircl"] in points["vllm.general_plugins"]
    assert ["sircl", "sparkring_sircl.vllm.platform:activate", "sparkring-sircl"] in points["vllm.platform_plugins"]
    assert re.fullmatch(r"roce_proxy-[0-9a-f]{16}\.so", staging.library_name())


def good_record(library: str) -> dict:
    return {"package": "/sircl/src/sparkring_sircl/__init__.py",
            "entry_points": {group: [["sircl", value, "sparkring-sircl"], ["b12x_loader", "b12x.x:y", "b12x"]]
                             for group, value in probe.OURS.items()},
            "vllm": {"root": "/usr/local/lib/python3.12/dist-packages/vllm", "matches": ["lil-image-aba309e4610c"]},
            "library": {"path": f"/sircl/build-cache/{library}", "exists": True}}


def test_the_stage_probe_record_is_judged_for_shadowing_entry_points_and_the_library():
    library = "roce_proxy-" + "1" * 16 + ".so"
    blockers, notes = probe.evaluate(good_record(library), staged_root="/sircl/src", library=library)
    assert blockers == [] and notes == [
        "vLLM at /usr/local/lib/python3.12/dist-packages/vllm matches pinned build lil-image-aba309e4610c"]
    record = good_record(library)
    record["package"] = "/opt/sparkring/addons/python/sparkring_sircl/__init__.py"
    record["entry_points"]["vllm.general_plugins"].append(["sircl", "other.module:register", "other"])
    record["library"]["exists"] = False
    record["vllm"]["matches"] = []
    blockers, notes = probe.evaluate(record, staged_root="/sircl/src", library=library)
    assert len(blockers) == 3 and "shadow" in blockers[0] and "other.module:register" in blockers[1]
    assert "matches no pinned build" in notes[0]
    assert probe.parse("noise\n" + probe.PREFIX + json.dumps(record) + "\n") == record
    record = good_record(library)
    record["p2p_library"] = {"path": "/sircl/build-cache/p2p_proxy-" + "2" * 16 + ".so", "exists": True}
    assert probe.evaluate(record, staged_root="/sircl/src", library=library)[0] == []
    record["p2p_library"] = {"error": "BuildError: building SIRCL's point-to-point library failed"}
    assert probe.evaluate(record, staged_root="/sircl/src", library=library)[0] == [
        "the point-to-point library could not be built: BuildError: building SIRCL's point-to-point library failed"]
    assert probe.serve_path(["/x", "/opt/sparkring/addons/python"], prefixes=("/nonexistent",)) == [
        "/x", "/opt/sparkring/addons/python"]


# -- remote commands ----------------------------------------------------------------------------------


def test_host_file_operations_run_through_each_sparks_sudo_prefix(serve_plan):
    rank0, rank2 = serve_plan.ranks[0], serve_plan.ranks[2]
    assert rank0.sudo == "sudo -n" and rank2.sudo == ""
    made = commands.make_directories(rank0.directories, rank0.sudo)
    assert made.startswith("sudo -n sh -c ") and shlex.split(made)[4].startswith("mkdir -p ")
    assert commands.make_directories(rank2.directories, rank2.sudo).startswith("mkdir -p ")
    facts = commands.host_facts(serve_plan, rank0, serve_plan.profile)
    words = shlex.split(facts)
    assert words[:4] == ["sudo", "-n", "sh", "-c"] and len(words) == 5 and MODEL in words[4]
    assert not commands.host_facts(serve_plan, rank2, serve_plan.profile).startswith("sudo")
    assert commands.stage_tree(serve_plan.source_dir, serve_plan.staged_digest).startswith("if [ -f ")


def test_every_remote_command_is_valid_shell(serve_plan):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("no bash on this machine")
    launch = serve_plan.ranks[2]
    built = [
        commands.running_containers(launch.docker), commands.run_containers(serve_plan.run_id, launch.docker),
        commands.remove_containers(serve_plan.run_id, launch.docker), commands.remove_containers(None, launch.docker),
        commands.container_state(launch.container, launch.docker),
        commands.container_logs(launch.container, launch.docker, 40),
        commands.log_lines(launch.container, launch.docker, cli.FAILURES),
        commands.receipts(serve_plan.receipt_dir, 2), commands.api_get(8017, "/v1/models"),
        commands.api_post(8017, "/v1/chat/completions", 300), commands.make_directories(launch.directories),
        commands.stage_tree(serve_plan.source_dir, serve_plan.staged_digest),
        commands.write_once(serve_plan.seccomp_path, serve_plan.profile.seccomp_sha256),
        commands.stage_container(serve_plan, launch), commands.staged_state(serve_plan),
        commands.listening_ports((8017, 29775)), commands.host_facts(serve_plan, launch, serve_plan.profile),
        launch.shell(), commands.make_directories(launch.directories, "sudo -n"),
        commands.host_facts(serve_plan, serve_plan.ranks[0], serve_plan.profile),
    ]
    for command in built:
        checked = subprocess.run([bash, "-n", "-c", command], capture_output=True, text=True)
        assert checked.returncode == 0, (command[:200], checked.stderr)
        inner = shlex.split(command)
        if inner[:2] == ["sudo", "-n"]:                   # the script under sudo is valid sh too
            checked = subprocess.run([bash, "-n", "-c", inner[4]], capture_output=True, text=True)
            assert checked.returncode == 0, (inner[4][:200], checked.stderr)
    assert "--filter label=sircl-serve=tp4-0-3" in built[2] and "--filter label=sircl-serve)" in built[3]
    assert built[13].startswith("mkdir -p /tmp/sircl-ring/build-cache && /usr/bin/docker run --rm ")


# -- fake Sparks ------------------------------------------------------------------------------------


class FakeSparks:
    """Answers the launcher's commands like healthy, idle Sparks and records every call."""

    def __init__(self, plans, *, running=None, failing_run=None, answers=None, receipts=None, states=None,
                 failing_mkdir=None):
        self.plans = [plans] if isinstance(plans, plan_mod.ServePlan) else list(plans)
        self.plan = self.plans[0]
        self.calls: list[tuple[str, str, bytes | None]] = []
        self.running = running or {}
        self.failing_run = failing_run
        self.failing_mkdir = failing_mkdir
        self.answers = answers or {prompt.name: good for prompt, good in
                                   zip(checks.PROMPTS, (", ".join(str(i) for i in range(1, 21)), "391", "Paris"))}
        self.receipts = receipts
        self.states = states or {}

    def receipt(self, rank: int, **extra) -> dict:
        record = {"group": "tp:0", "global_rank": rank, "rank": rank, "world": 4, "nccl": "none",
                  "pynccl": "skipped", "session": "ring", "state": "ready",
                  "decisions": [{"collective": "all_reduce", "backend": "sircl", "method": "chunked", "calls": 9},
                                {"collective": "all_gather", "backend": "sircl", "method": "rows", "calls": 3}]}
        record.update(extra)
        return record

    def launches(self):
        return [launch for plan in self.plans for launch in plan.ranks]

    def __call__(self, target, command, *, timeout=60, input_bytes=None):
        self.calls.append((target, command, input_bytes))
        ok = remote.Result(0, "", "")
        if "/proc/meminfo" in command:
            profile = self.plan.profile
            sizes = "".join(f"size:{item.name}\t{item.size}\n" for item in profile.checkpoint_files)
            return remote.Result(0, "mem:MemTotal\t125000000\nmem:MemAvailable\t120000000\ncurl\t/usr/bin/curl\n"
                                    f"model\tpresent\n{sizes}sha256:config.json\t{profile.config_sha256}\n"
                                    f"sha256:model.safetensors.index.json\t{profile.index_sha256}\n"
                                    "cache\tpresent\nspace_kib\t3000000000\n", "")
        if " ps --format " in command:
            return remote.Result(0, self.running.get(target, ""), "")
        if "ps -a --filter" in command:
            return ok
        if ".sircl-staged ]; then echo" in command and "library\t" in command:
            return remote.Result(0, f"source\tpresent\nlibrary\tpresent\nseccomp\t{sha(SECCOMP)}\n", "")
        if "ss -Hltn" in command:
            ports = re.findall(r"sport = :(\d+)", command)
            return remote.Result(0, "".join(f"port:{port}\tfree\n" for port in ports), "")
        if " run -d --name " in command:
            if self.failing_run and self.failing_run in command:
                return remote.Result(125, "", "docker: Error response from daemon: bind source path does not exist")
            return remote.Result(0, "c0ffee" * 10 + "\n", "")
        if "ps -aq --filter" in command:
            return remote.Result(0, "0\n", "")
        if "mkdir -p" in command and "docker" not in command:
            if self.failing_mkdir and self.failing_mkdir == target:
                return remote.Result(1, "", "mkdir: cannot create directory '/srv/sparkring/x': Permission denied")
            return ok
        if "inspect -f" in command:
            name = next(launch.container for launch in self.launches() if launch.container in command)
            return remote.Result(0, self.states.get(name, "running 0 healthy") + "\n", "")
        if "/v1/models" in command:
            plan = next(plan for plan in self.plans if f":{plan.api_port}/" in command)
            return remote.Result(0, json.dumps({"data": [{"id": plan.profile.served_model_name}]}), "")
        if "/v1/chat/completions" in command:
            body = json.loads(input_bytes)
            content = body["messages"][0]["content"]
            if content.startswith("Session "):
                answer = {"choices": [{"message": {"content": "OK"}}], "usage": {"prompt_tokens": 16400}}
                return remote.Result(0, json.dumps(answer) + "\n9.80\n", "")
            name = next(prompt.name for prompt in checks.PROMPTS if prompt.content == content)
            answer = {"choices": [{"message": {"content": self.answers[name], "reasoning_content": "brief"}}],
                      "usage": {"prompt_tokens": 20, "completion_tokens": 5}}
            return remote.Result(0, json.dumps(answer) + "\n0.42\n", "")
        if "| grep -F" in command:
            if "SIRCL receipt" in command:
                rank = next(launch.rank for launch in self.launches() if launch.container in command)
                return remote.Result(0, f"INFO adapter.py:292] SIRCL receipt group=tp:0 global_rank={rank} "
                                        "nccl=none pynccl=skipped session=ring state=ready\n"
                                        "INFO plugin.py:72] SIRCL vLLM adapter registered: layout ring:8\n", "")
            return ok
        if "rank" in command and "-*.json" in command:
            rank = int(re.search(r"for f in rank(\d+)-", command).group(1))
            records = self.receipts(rank) if self.receipts else [self.receipt(rank)]
            return remote.Result(0, "".join(f"rank{rank}-{r['group'].replace(':', '-')}.json\t{json.dumps(r)}\n"
                                            for r in records), "")
        if " logs " in command:
            return remote.Result(0, "INFO last line\n", "")
        raise AssertionError(f"unexpected command on {target}: {command[:160]}")


def context(plan, fake, lines=None):
    output = lines if lines is not None else []
    return cli.Context(plan=plan, tree=staging.staged_tree(), run=fake, out=output.append, sleep=lambda _: None)


def contexts(plans, fake, lines):
    return [context(plan, fake, lines) for plan in plans]


def test_print_shows_the_plan_and_every_remote_command_and_contacts_nothing(repository, site_file, capsys,
                                                                             monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("--print must not contact anything")

    monkeypatch.setattr(remote, "ssh", refuse)
    monkeypatch.setattr("subprocess.run", refuse)
    base = ["--site", str(site_file), "--repository", str(repository)]
    code = cli.main(["start", "--print", *base, "--profile", PROFILE, "--model-path", MODEL], run=refuse)
    output = capsys.readouterr().out
    assert code == 0
    assert "nothing was contacted" in output and output.count(" run -d --name sircl-serve-tp4-0-3-r") == 8
    assert "-m sparkring_sircl.vllm.serve.probe --build" in output and "sudo -n sh -c 'mkdir -p" in output
    assert cli.main(["plan", "--json", *base, "--profile", PROFILE, "--model-path", MODEL], run=refuse) == 0
    assert json.loads(capsys.readouterr().out)["run_id"] == "tp4-0-3"
    assert cli.main(["start", "--print", *base, "--profile", TP2.id, "--groups", "0-1;2-3;4-5;6-7",
                     "--model-path", MODEL], run=refuse) == 0
    output = capsys.readouterr().out
    assert all(f"--name sircl-serve-tp2-{a}-{b}-r1" in output for a, b in ((0, 1), (2, 3), (4, 5), (6, 7)))
    assert cli.main(["plan", "--json", *base, "--profile", TP2.id, "--groups", "0-1;2-3", "--model-path", MODEL],
                    run=refuse) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["schema"] == "sircl-serve-plans/v1" and [item["api_port"] for item in record["instances"]] == [
        8017, 8018]
    assert cli.main(["plan", *base, "--profile", PROFILE, "--positions", "7-2", "--model-path", MODEL],
                    run=refuse) == 0
    assert "ranks 0-3 (positions 7-2): 2 lanes through positions 0, 1" in capsys.readouterr().out
    assert cli.main(["plan", *base, "--profile", PROFILE, "--positions", "0-3"], run=refuse) == 2
    assert "no model directory for Sparks [0, 1, 2, 3]" in capsys.readouterr().err


def test_start_refuses_while_another_container_runs_and_starts_nothing(serve_plan):
    running = "sr-glm-tp8-r1\tghcr.io/example/sparkring@sha256:1111\tUp 2 hours\t\t\t" + "e" * 64 + "\n"
    fake = FakeSparks(serve_plan, running={"user@192.0.2.21": running})
    lines: list[str] = []
    assert cli.start(context(serve_plan, fake, lines)) == 1
    assert not any(" run -d " in command for _, command, _ in fake.calls)
    assert any("sr-glm-tp8-r1 (SparkRing deployment eeeeeeeeeeee" in line for line in lines)
    assert lines[-1] == "nothing was started"
    fake.calls.clear()
    assert cli.start(context(serve_plan, fake, []), allow_running=["sr-glm-tp8-r1"]) == 0


def test_start_creates_directories_through_sudo_then_starts_ranks_in_order(serve_plan):
    fake = FakeSparks(serve_plan)
    lines: list[str] = []
    assert cli.start(context(serve_plan, fake, lines)) == 0
    started = [(host, command) for host, command, _ in fake.calls if " run -d " in command]
    assert [host for host, _ in started] == [f"user@192.0.2.{20 + rank}" for rank in range(4)]
    assert [command for _, command in started] == [launch.shell() for launch in serve_plan.ranks]
    first_run = next(index for index, (_, command, _) in enumerate(fake.calls) if " run -d " in command)
    made = [(host, command) for host, command, _ in fake.calls[:first_run] if "mkdir -p" in command]
    assert [command.startswith("sudo -n sh -c ") for _, command in made] == [True, True, False, True]


def test_a_failed_mkdir_names_the_sudo_prefix_and_starts_nothing(serve_plan):
    fake = FakeSparks(serve_plan, failing_mkdir="user@192.0.2.21")
    lines: list[str] = []
    assert cli.start(context(serve_plan, fake, lines)) == 1
    assert not any(" run -d " in command for _, command, _ in fake.calls)
    assert any("through \"sudo -n\" failed: mkdir: cannot create directory" in line for line in lines)


def test_a_failed_docker_run_removes_only_these_runs_containers(serve_plan, repository):
    fake = FakeSparks(serve_plan, failing_run="sircl-serve-tp4-0-3-r2")
    lines: list[str] = []
    assert cli.start(context(serve_plan, fake, lines)) == 1
    removals = [(host, command) for host, command, _ in fake.calls if "ps -aq --filter" in command]
    assert {host for host, _ in removals} == {f"user@192.0.2.{20 + rank}" for rank in range(4)}
    assert all("label=sircl-serve=tp4-0-3" in command for _, command in removals)
    assert any("bind source path does not exist" in line for line in lines)
    plans = make_plans(repository, ((0, 1), (2, 3), (4, 5), (6, 7)))
    fake = FakeSparks(plans, failing_run="sircl-serve-tp2-4-5-r0")
    assert cli.start(contexts(plans, fake, [])) == 1
    removed = [command for _, command, _ in fake.calls if "rm -f" in command]
    assert sorted({re.search(r"label=sircl-serve=(tp2-\d-\d)", command).group(1) for command in removed}) == [
        "tp2-0-1", "tp2-2-3", "tp2-4-5", "tp2-6-7"]


def test_four_pair_instances_start_wait_and_check_together(repository):
    plans = make_plans(repository, ((0, 1), (2, 3), (4, 5), (6, 7)), site=SITE_WITH_PATHS, model_path=None)

    def pair_receipts(rank):
        return [{**FakeSparks(plans).receipt(rank), "world": 2, "nccl": "all", "pynccl": "built",
                 "decisions": [{"collective": "all_reduce", "backend": "sircl", "method": "direct", "calls": 40},
                               {"collective": "all_reduce", "backend": "nccl", "method": "nccl", "calls": 6}]}]

    fake = FakeSparks(plans, receipts=pair_receipts)
    lines: list[str] = []
    assert cli.start(contexts(plans, fake, lines), wait_after=True) == 0
    assert sum(" run -d " in command for _, command, _ in fake.calls) == 8
    assert sum(line.startswith("API ready after") for line in lines) == 4
    lines.clear()
    assert cli.check(contexts(plans, fake, lines)) == 0
    assert sum(line.endswith(": check passed") for line in lines) == 4
    bodies = [json.loads(data) for _, command, data in fake.calls if "/v1/chat/completions" in command]
    assert {body["model"] for body in bodies} == {TP2.served}
    ports = {re.search(r"127\.0\.0\.1:(\d+)", command).group(1) for _, command, _ in fake.calls
             if "/v1/chat/completions" in command}
    assert ports == {"8017", "8018", "8019", "8020"}


def test_stop_removes_by_each_runs_label_and_all_runs_covers_every_spark(serve_plan, tmp_path):
    fake = FakeSparks(serve_plan)
    assert cli.stop(context(serve_plan, fake), output=tmp_path / "results") == 0
    removals = [(host, command) for host, command, _ in fake.calls if "rm -f" in command]
    assert len(removals) == 4
    for _, command in removals:
        assert "--filter label=sircl-serve=tp4-0-3" in command
        assert "io.sparkring" not in command and "sircl-ring" not in command
    saved = tmp_path / "results/tp4-0-3"
    assert json.loads((saved / "plan.json").read_text())["run_id"] == "tp4-0-3"
    assert json.loads((saved / "rank-3-receipts.json").read_text())[0]["global_rank"] == 3
    fake.calls.clear()
    assert cli.stop(context(serve_plan, fake), all_runs=True, collect_first=False) == 0
    removals = [(host, command) for host, command, _ in fake.calls if "rm -f" in command]
    assert len(removals) == 8 and all("--filter label=sircl-serve)" in command for _, command in removals)
    assert {host for host, _ in removals} == {entry["ssh"] for entry in SITE["ring"]}


def test_wait_reports_an_exited_rank_with_its_log_and_leaves_the_containers(serve_plan):
    fake = FakeSparks(serve_plan, states={"sircl-serve-tp4-0-3-r1": "exited 1 none"})
    lines: list[str] = []
    assert cli.wait(context(serve_plan, fake, lines), timeout=60) == 1
    assert any("sircl-serve-tp4-0-3-r1 is exited (exit code 1)" in line for line in lines)
    assert not any("rm -f" in command for _, command, _ in fake.calls)
    fake = FakeSparks(serve_plan, states={"sircl-serve-tp4-0-3-r1": "exited 1 none"})
    assert cli.wait(context(serve_plan, fake, []), timeout=60, stop_on_failure=True) == 1
    assert sum("rm -f" in command for _, command, _ in fake.calls) == 4


def test_wait_returns_when_the_api_serves_the_model_and_every_rank_logged_its_receipt(serve_plan):
    lines: list[str] = []
    assert cli.wait(context(serve_plan, FakeSparks(serve_plan), lines), timeout=60) == 0
    assert any(line.startswith("API ready after") for line in lines)
    assert sum("SIRCL receipt group=tp:0" in line for line in lines) == 4


class RedrawingSparks(FakeSparks):
    """Healthy Sparks whose API answers from the second poll and whose log tails end in a progress bar."""

    LOG = "Loading safetensors \u2595\u2588\u2588\u2588\u2588\u258f 100% \u2502 55/55\n"

    def __init__(self, plans, **kwargs):
        super().__init__(plans, **kwargs)
        self.polls = 0

    def __call__(self, target, command, *, timeout=60, input_bytes=None):
        if "/v1/models" in command:
            self.polls += 1
            if self.polls == 1:
                self.calls.append((target, command, input_bytes))
                return remote.Result(7, "", "curl: (7) Failed to connect to 127.0.0.1 port 8017")
        if " logs --tail " in command:
            self.calls.append((target, command, input_bytes))
            return remote.Result(0, self.LOG, "")
        return super().__call__(target, command, timeout=timeout, input_bytes=input_bytes)


def test_wait_prints_log_lines_a_cp1252_stream_cannot_encode(repository, site_file, monkeypatch):
    """Output redirected on Windows is encoded in cp1252; a progress bar in a log tail must not end the wait."""
    console = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict", write_through=True)
    with pytest.raises(UnicodeEncodeError):
        console.write(RedrawingSparks.LOG)
    monkeypatch.setattr(sys, "stdout", console)
    fake = RedrawingSparks(make_plan(repository))
    code = cli.main(["wait", "--site", str(site_file), "--repository", str(repository), "--profile", PROFILE,
                     "--model-path", MODEL, "--interval", "0", "--timeout", "60"], run=fake)
    text = console.buffer.getvalue().decode("cp1252")
    assert code == 0 and fake.polls == 2
    assert "r0: Loading safetensors ?????? 100% ? 55/55" in text and "API ready after" in text
    assert text.count("SIRCL receipt group=tp:0") == 4


def test_check_passes_on_known_answers_and_clean_receipts(serve_plan):
    fake = FakeSparks(serve_plan)
    lines: list[str] = []
    assert cli.check(context(serve_plan, fake, lines), long_prompt=16384) == 0
    assert [line.split(":")[0] for line in lines[:3]] == ["count", "arithmetic", "capital"]
    assert all("passed" in line for line in lines[:3])
    bodies = [json.loads(data) for _, command, data in fake.calls if "/v1/chat/completions" in command]
    assert all(body["temperature"] == 0 and body["chat_template_kwargs"] == {"reasoning_effort": "low"}
               for body in bodies)
    long_body = next(body for body in bodies if body["messages"][0]["content"].startswith("Session "))
    assert long_body["max_tokens"] == 1 and len(long_body["messages"][0]["content"]) > 16384
    assert bodies[-1]["messages"][0]["content"] == checks.PROMPTS[0].content    # refreshes the receipts' counts
    line = next(line for line in lines if line.startswith("long prompt: 16,400 prompt tokens answered in 9.80 s"))
    assert "rank 0's receipt states no session op sizes" in line
    assert lines[-1] == "run tp4-0-3: check passed"


def test_check_fails_on_a_wrong_answer_or_any_nccl_row_on_a_group_nccl_may_not_run(serve_plan):
    fake = FakeSparks(serve_plan, answers={"count": "1, 2, 3", "arithmetic": "391", "capital": "Paris"})
    lines: list[str] = []
    assert cli.check(context(serve_plan, fake, lines)) == 1
    assert lines[0].startswith("count: FAILED (expected contains 1, 2, ..., 20)")

    def with_nccl(rank):
        record = FakeSparks(serve_plan).receipt(rank)
        if rank == 2:
            record["decisions"].append({"collective": "all_reduce", "backend": "nccl", "method": "nccl",
                                        "calls": 1})
        return [record, {**FakeSparks(serve_plan).receipt(rank), "group": "ep:0", "decisions": []}]

    lines = []
    assert cli.check(context(serve_plan, FakeSparks(serve_plan, receipts=with_nccl), lines)) == 1
    assert any(line == "PROBLEM: rank 2 group tp:0: 1 calls reached NCCL (['all_reduce'])" for line in lines)


def test_receipt_evaluation_rules():
    good = {"group": "tp:0", "nccl": "none", "pynccl": "skipped", "session": "ring", "state": "ready",
            "decisions": [{"collective": "all_reduce", "backend": "sircl", "method": "direct", "calls": 4}]}
    problems, lines = checks.evaluate_receipts({0: [good], 1: [good]}, 2)
    assert problems == [] and len(lines) == 2
    problems, _ = checks.evaluate_receipts({0: [{**good, "pynccl": "built"}], 1: []}, 2)
    assert problems == ["rank 0 group tp:0: PyNccl was built on a group NCCL may not run",
                        "rank 1: no tensor-parallel receipt"]
    refused = {**good, "decisions": [{"collective": "broadcast", "backend": "refuse", "method": "refuse"}]}
    problems, _ = checks.evaluate_receipts({0: [refused]}, 1)
    assert problems == ["rank 0 group tp:0: refused collectives ['broadcast']",
                        "rank 0 group tp:0: no all-reduce ran on SIRCL"]
    pair = {**good, "nccl": "all", "pynccl": "built",
            "decisions": good["decisions"] + [{"collective": "all_reduce", "backend": "nccl", "method": "nccl"}]}
    assert checks.evaluate_receipts({0: [pair]}, 1)[0] == []
    answer = checks.read_answer("arithmetic", json.dumps({"choices": [{"message": {"content": " 391\n"}}]})
                                + "\n1.5\n", checks.PROMPTS[1].accepts)
    assert answer.ok and answer.seconds == 1.5
    assert not checks.read_answer("arithmetic", "<html>bad gateway</html>\n", checks.PROMPTS[1].accepts).ok


def test_preflight_checks_each_ranks_own_model_directory_through_sudo(repository):
    plan = make_plan(repository, site=SITE_WITH_PATHS, model_path=None)
    profile = plan.profile
    fake = FakeSparks(plan)

    def facts_for(target, command, **kwargs):
        if "/proc/meminfo" in command:
            fake.calls.append((target, command, None))
            sizes = "".join(f"size:{item.name}\t{item.size}\n" for item in profile.checkpoint_files)
            digests = (f"sha256:config.json\t{profile.config_sha256}\n"
                       f"sha256:model.safetensors.index.json\t{'0' * 64 if target.endswith('.23') else profile.index_sha256}\n")
            memory = "mem:MemTotal\t125000000\nmem:MemAvailable\t" + ("20000000" if target.endswith(".21") else "120000000")
            return remote.Result(0, f"{memory}\ncurl\t/usr/bin/curl\nmodel\tpresent\n{sizes}{digests}cache\tabsent\n"
                                    "space_kib\t3000000000\n", "")
        return fake(target, command, **kwargs)

    lines: list[str] = []
    assert cli.preflight(context(plan, facts_for, lines), fabric=False) == 1
    blockers = [line for line in lines if line.startswith("BLOCKER: ")]
    assert len(blockers) == 2
    assert "rank 1 (spark1): 19.1 GiB available is below" in blockers[0]
    assert f"rank 3 (spark3): {MODEL_B}/model.safetensors.index.json has SHA-256 " + "0" * 64 in blockers[1]
    assert any(f"model directory {MODEL_A} (site file model_path) holds the 4 pinned files" in line and
               "checked through \"sudo -n\"" in line for line in lines)
    facts_commands = {target: command for target, command, _ in fake.calls if "/proc/meminfo" in command}
    assert MODEL_B in facts_commands["user@192.0.2.23"] and MODEL_A not in facts_commands["user@192.0.2.23"]
    assert facts_commands["user@192.0.2.20"].startswith("sudo -n sh -c ")
    assert not facts_commands["user@192.0.2.22"].startswith("sudo")
    assert any(line.endswith(f"cache /tmp/sircl-ring/serve/cache/{PROFILE} absent, 2861 GiB free there")
               for line in lines)


def test_preflight_runs_the_ring_harness_fabric_checks_with_the_serving_image(serve_plan, repository, monkeypatch):
    seen = []

    def harness_ssh(target, command, *, timeout=60, input_bytes=None, binary="ssh"):
        seen.append((target, command))
        if "hostname" in command:
            return remote.Result(0, "hostname\tspark\ndocker\t28.0.1\nimage\tmissing\ngpu\t0, NVIDIA GB10\n", "")
        return remote.Result(0, "", "")

    monkeypatch.setattr(remote, "ssh", harness_ssh)
    ok, harness_lines = cli.harness_preflight(serve_plan)
    assert not ok and any(f"image {IMAGE_ID} is not present" in line for line in harness_lines)
    assert {target for target, _ in seen} == {f"user@192.0.2.{20 + rank}" for rank in range(4)}
    assert any(IMAGE_ID in command for _, command in seen)
    seen.clear()
    pairs = make_plans(repository, ((7, 0), (2, 3)))
    cli.harness_preflight(pairs)
    assert {target for target, _ in seen} == {"user@192.0.2.27", "user@192.0.2.20", "user@192.0.2.22",
                                              "user@192.0.2.23"}
    lines: list[str] = []
    ctx = context(serve_plan, FakeSparks(serve_plan), lines)
    monkeypatch.setattr(cli, "harness_preflight", lambda plans: (False, ["BLOCKER: spark0: RDMA device x is DOWN"]))
    assert cli.preflight(ctx) == 1
    assert "fabric: BLOCKER: spark0: RDMA device x is DOWN" in lines
    assert "BLOCKER: spark0: RDMA device x is DOWN" in lines
    lines.clear()
    monkeypatch.setattr(cli, "harness_preflight", lambda plans: (True, ["spark0: docker command \"docker\""]))
    assert cli.preflight(ctx) == 0 and lines[-1] == "preflight passed"


@needs_checkout
def test_the_repositorys_glm53_flash_profiles_plan_on_the_ring():
    root = CHECKOUT
    profile = profile_mod.load(root)
    assert profile.id == "glm53-flash-nvfp4-spark-tp4" and profile.tensor_parallel == 4
    assert profile.image_id.startswith("sha256:aba309e4610c")
    site = ServeSite.from_json(SITE_WITH_PATHS)
    tree = staging.staged_tree()
    plan = plan_mod.build_plan(site, profile, Options((7, 0, 1, 2)), staged_digest=tree.digest,
                               library=staging.library_name())
    assert plan.prefill().gathers == ((180, 2048, 8192),) and plan.api_port == 8017
    assert plan.prefill().compute_measured
    assert plan.ranks[0].environment["VLLM_GLM53_MHC_PREFILL_SHARD"] == "1"
    off = plan_mod.build_plan(site, profile, Options((7, 0, 1, 2), mhc_prefill_shard="off"),
                              staged_digest=tree.digest, library=staging.library_name())
    assert off.prefill().oneshot_ops == 96256 and 8.8 < off.prefill().bound_seconds[0] < 9.0
    pair_profile = profile_mod.load(root, "glm53-flash-nvfp4-spark-tp2")
    plans = plan_mod.build_plans(site, pair_profile, ((0, 1), (2, 3), (4, 5), (6, 7)),
                                 Options((0, 1), nccl_mode="auto"),
                                 staged_digest=tree.digest, library=staging.library_name())
    assert [plan.api_port for plan in plans] == [8017, 8018, 8019, 8020]
    assert plans[0].ranks[1].environment["NCCL_IB_HCA"] == "=rocep1s0f1,roceP2p1s0f1"
    assert plans[1].ranks[1].mount("/models/target").source == MODEL_B
    # GLM-5.3-Flash's chat template: always thinking, max when a request names no effort; high as the default.
    assert (profile.thinking.id, profile.thinking.level, profile.thinking.levels) == (
        "glm53-flash-template", "max", ("low", "high", "max"))
    high = plan_mod.build_plan(site, profile, Options((7, 0, 1, 2), reasoning_effort="high"),
                               staged_digest=tree.digest, library=staging.library_name())
    assert high.ranks[0].command[-2:] == ("--default-chat-template-kwargs", '{"reasoning_effort":"high"}')
    assert "--default-chat-template-kwargs" not in high.ranks[1].command
    mimo = profile_mod.load(root, "mimo-v26-flash-mopd-tp4")
    with pytest.raises(ServePlanError, match=r"\(mimo-v26-flash-template\) has no effort levels"):
        plan_mod.build_plan(site, mimo, Options((0, 1, 2, 3), reasoning_effort="high"), staged_digest=tree.digest,
                            library=staging.library_name())


# The adapter RUNBOOK's "Installer profiles on the ring of eight": catalog profiles that plan on Sparks
# 0-1 (tensor parallelism 2) or 0-3 (4) as they are, those whose hyper-connection prefill row ownership
# SIRCL carries there under --nccl auto (with NCCL off, the default, SIRCL carries it for every planned
# Qwen3.8 profile), and the reason every other profile names.
PLANNED = {"glm53-flash-nvfp4-spark-tp2", "qwen38-flash-next-tp2", "swift15-qwen38-flash-next-tp2",
           "mimo-v26-flash-mopd-tp2", "glm53-flash-nvfp4-spark-tp4", "mimo-v26-flash-mopd-tp4",
           "deepseek-v41-flash-tp4", "qwen38-flash-next-qad-tp4", "swift15-qwen38-flash-next-tp4"}
ROW_OWNERSHIP_ON_SIRCL = {"qwen38-flash-next-qad-tp4", "swift15-qwen38-flash-next-tp4"}
BLOCKERS = ("runs from its own launcher", "switched fabric", "has no installer image lock")


@needs_checkout
def test_every_catalog_profile_plans_on_the_ring_or_names_what_blocks_it():
    root = Path(CHECKOUT)
    catalog = json.loads((root / "profiles/catalog.json").read_text(encoding="utf-8"))
    site = ServeSite.from_json(SITE_WITH_PATHS)
    tree = staging.staged_tree()
    outcomes = {}
    for entry in catalog["profiles"]:
        try:
            profile = profile_mod.load(root, entry["id"])
        except ProfileError as error:
            outcomes[entry["id"]] = str(error)
            continue
        assert profile.tensor_parallel in (2, 4) and profile.image_id.startswith("sha256:aba309e4610c")
        positions = (0, 1) if profile.tensor_parallel == 2 else (0, 1, 2, 3)
        try:
            plan = plan_mod.build_plan(site, profile, Options(positions), staged_digest=tree.digest,
                                       library=staging.library_name())
        except ServePlanError as error:
            outcomes[entry["id"]] = str(error)
            continue
        outcomes[entry["id"]] = "planned"
        # By default (--nccl never) SIRCL carries every group, prefill row ownership among them; with --nccl
        # auto a pair keeps vLLM's PyNccl for it.
        assert not plan.nccl_policy.allows("all_reduce"), entry["id"]
        ruled = plan_mod.build_plan(site, profile, Options(positions, nccl_mode="auto"),
                                    staged_digest=tree.digest, library=staging.library_name())
        carried = ruled.hc_prefill_mode == "shard" and not ruled.nccl_policy.allows("all_reduce")
        assert carried == (entry["id"] in ROW_OWNERSHIP_ON_SIRCL), entry["id"]
    assert {key for key, outcome in outcomes.items() if outcome == "planned"} == PLANNED
    blocked = {key: outcome for key, outcome in outcomes.items() if not outcome.startswith("planned")}
    assert blocked and all(any(reason in outcome for reason in BLOCKERS) for outcome in blocked.values()), blocked


def test_the_long_prompt_check_prices_session_ops_from_rank_zeros_receipt(serve_plan, monkeypatch):
    monkeypatch.setitem(plan_mod.PATH_COMPUTE_SECONDS_PER_TOKEN, serve_plan.profile.id, 0.26e-3)
    stats = {"dispatch_limit_bytes": 131072, "large_piece_bytes": 4 << 20, "gather_piece_bytes": 4 << 20,
             "large_schedule": "pieces", "chain_available": False}

    def receipts(rank):
        return [FakeSparks(serve_plan).receipt(rank, session_stats=stats)]

    fake = FakeSparks(serve_plan, receipts=receipts)
    lines: list[str] = []
    assert cli.check(context(serve_plan, fake, lines), long_prompt=16384) == 0
    line = next(line for line in lines if line.startswith("long prompt:"))
    # 16,400 tokens with mHC row ownership on a session without a reduce-scatter: 94 * (16 + 16 + 1)
    # all-reduce ops (the 180 reduce-scatters among them), 180 gathers of 4 ops.
    assert "reduce-scatters carried as all-reduces; all-gather pieces of 4,194,304 B" in line
    assert "3,102 all-reduce, 0 reduce-scatter and 720 all-gather ops" in line
    # 16,400 tokens at 0.26 ms less 0.8 s of mixing leave 6.4 s for 3,822 ops.
    assert "after compute of 3.4 s (measured on the path) the collectives took 6.4 s, 1.66 ms per op" in line


def test_waits_and_the_spin_limit_reach_every_rank_and_nothing_pins_the_piece_size(repository):
    plan = make_plan(repository)
    for launch in plan.ranks:
        assert launch.environment["SIRCL_STARTUP_WAIT_S"] == "600"
        assert launch.environment["SIRCL_SERVING_WAIT_S"] == "20"
        assert "SIRCL_SPIN_LIMIT" not in launch.environment
        assert "SIRCL_LARGE_PIECE_BYTES" not in launch.environment
    custom = make_plan(repository, spin_limit=123, startup_wait=900, serving_wait=7.5)
    assert {launch.environment["SIRCL_SPIN_LIMIT"] for launch in custom.ranks} == {"123"}
    assert {launch.environment["SIRCL_SERVING_WAIT_S"] for launch in custom.ranks} == {"7.5"}
    text = plan_mod.render_text(custom)
    assert "startup regime up to 900 s" in text and "serving regime up to 7.5 s" in text
    assert "SIRCL_SPIN_LIMIT=123 (--spin-limit; time-limited waits ignore it)" in text
    with pytest.raises(ServePlanError, match="serving wait"):
        make_plan(repository, serving_wait=0)


def test_session_schedules_reach_every_rank_of_start_and_bundle_and_the_plan_records_them(repository, site_file,
                                                                                         capsys):
    ring = make_plan(repository, large_schedule="ring", gather_schedule="ring", scatter_schedule="ring")
    for launch in ring.ranks:
        assert {key: launch.environment[key] for key in ("SIRCL_LARGE_SCHEDULE", "SIRCL_GATHER_SCHEDULE",
                                                         "SIRCL_SCATTER_SCHEDULE")} == dict.fromkeys(
            ("SIRCL_LARGE_SCHEDULE", "SIRCL_GATHER_SCHEDULE", "SIRCL_SCATTER_SCHEDULE"), "ring")
    reasons = {change.name: change.reason for change in ring.changes}
    assert reasons["SIRCL_GATHER_SCHEDULE"].startswith("--gather-schedule")
    assert "SIRCL_SCATTER_SCHEDULE: unset -> 'ring'" in plan_mod.render_text(ring)
    default = make_plan(repository, gather_schedule="chain")
    assert default.ranks[0].environment["SIRCL_GATHER_SCHEDULE"] == "chain"
    assert not {"SIRCL_LARGE_SCHEDULE", "SIRCL_SCATTER_SCHEDULE"} & set(default.ranks[0].environment)
    with pytest.raises(ServePlanError, match="--large-schedule must be one of"):
        make_plan(repository, large_schedule="fastest")
    with pytest.raises(ServePlanError, match="--large-schedule, --gather-schedule"):
        make_plan(repository, extra_env={"SIRCL_LARGE_SCHEDULE": "ring"})
    base = ["--site", str(site_file), "--repository", str(repository), "--profile", PROFILE, "--model-path", MODEL]
    assert cli.main(["plan", "--json", *base, "--large-schedule", "ring", "--scatter-schedule", "pieces"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert {item["name"]: item["after"] for item in record["changes"]}["SIRCL_LARGE_SCHEDULE"] == "ring"
    with pytest.raises(SystemExit):
        cli.main(["plan", *base, "--gather-schedule", "fastest"])
    capsys.readouterr()
    bundled = make_bundle(large_schedule="ring", gather_schedule="ring", scatter_schedule="ring")
    assert bundled.ranks[3].environment["SIRCL_SCATTER_SCHEDULE"] == "ring"
    assert "SIRCL_LARGE_SCHEDULE" not in make_bundle().ranks[0].environment
    assert cli.main(["bundle", "--site", str(site_file), "--positions", "0-7", "--gather-schedule", "chain"]) == 0
    assert json.loads(capsys.readouterr().out)["ranks"][0]["environment"]["SIRCL_GATHER_SCHEDULE"] == "chain"


def test_link_sizes_reach_every_rank_of_start_and_bundle_and_the_plan_records_them(repository, site_file, capsys):
    mib = 1 << 20
    linked = make_plan(repository, link_sizes={"link_slot": mib, "link_chunk": mib})
    for launch in linked.ranks:
        assert (launch.environment["SIRCL_LINK_SLOT_BYTES"], launch.environment["SIRCL_LINK_CHUNK_BYTES"]) == (
            "1048576", "1048576")
    reasons = {change.name: change.reason for change in linked.changes}
    assert reasons["SIRCL_LINK_SLOT_BYTES"].startswith("--link-slot: the sessions' link slot")
    assert reasons["SIRCL_LINK_CHUNK_BYTES"].startswith("--link-chunk: the piece of every chain or ring collective")
    text = plan_mod.render_text(linked)
    assert "SIRCL_LINK_CHUNK_BYTES: unset -> '1048576'" in text
    assert "; link slot 1,048,576 B (--link-slot), link piece 1,048,576 B (--link-chunk)" in text
    assert linked.to_json()["sizes"]["link_chunk"] == mib
    # Unset sizes keep the session's: a 512 KiB slot and a piece of the smaller of 512 KiB and the slot.
    default = make_plan(repository)
    assert not {"SIRCL_LINK_SLOT_BYTES", "SIRCL_LINK_CHUNK_BYTES"} & set(default.ranks[0].environment)
    assert ("; link slot 524,288 B (the session's default), link piece 524,288 B (the session's default)"
            in plan_mod.render_text(default))
    assert default.to_json()["sizes"]["link_slot"] is None
    small = make_plan(repository, link_sizes={"link_slot": 256 << 10})
    assert "SIRCL_LINK_CHUNK_BYTES" not in small.ranks[0].environment
    assert "link piece 262,144 B (the session's default)" in plan_mod.render_text(small)
    # Each link collective's piece: the link piece unless set.
    pieces = make_plan(repository, link_sizes={"gather_link_chunk": mib, "scatter_link_chunk": 256 << 10})
    assert {key: pieces.ranks[2].environment.get(key) for key in (
        "SIRCL_GATHER_LINK_CHUNK_BYTES", "SIRCL_SCATTER_LINK_CHUNK_BYTES", "SIRCL_REDUCE_LINK_CHUNK_BYTES",
        "SIRCL_LINK_SLOT_BYTES")} == {"SIRCL_GATHER_LINK_CHUNK_BYTES": "1048576",
                                      "SIRCL_SCATTER_LINK_CHUNK_BYTES": "262144",
                                      "SIRCL_REDUCE_LINK_CHUNK_BYTES": None, "SIRCL_LINK_SLOT_BYTES": None}
    assert {change.name: change.reason for change in pieces.changes}["SIRCL_GATHER_LINK_CHUNK_BYTES"].startswith(
        "--gather-link-chunk: the piece of chain and ring all-gathers")
    # Without --link-slot the slot grows to the largest piece set, rounded up to 4 KiB, up to 1 MiB.
    assert ("; link slot 1,048,576 B (the session's default), link piece 524,288 B (the session's default), "
            "all-gather link piece 1,048,576 B (--gather-link-chunk), reduce-scatter link piece 262,144 B "
            "(--scatter-link-chunk), all-reduce link piece 524,288 B (the link piece)") in plan_mod.render_text(pieces)
    assert pieces.to_json()["sizes"]["gather_link_chunk"] == mib and pieces.to_json()["sizes"]["reduce_link_chunk"] is None
    assert [plan_mod.link_slot(sizes) for sizes in (
        {}, {"link_chunk": 1 << 19}, {"link_chunk": (1 << 19) + 16}, {"reduce_link_chunk": mib},
        {"gather_link_chunk": mib + 16}, {"link_slot": 4096, "link_chunk": mib})] == [
        1 << 19, 1 << 19, (1 << 19) + 4096, mib, 1 << 19, 4096]
    assert make_plan(repository, link_sizes={"link_chunk": mib}).ranks[0].environment["SIRCL_LINK_CHUNK_BYTES"] == (
        "1048576")
    for sizes, message in (
            ({"link_chunk": mib + 16}, r"--link-chunk 1048592 exceeds the link slot of 524288 bytes \(without "
                                       r"--link-slot the sessions' slot holds pieces up to 1048576 bytes; set "
                                       r"--link-slot\)"),
            ({"scatter_link_chunk": 2 * mib}, r"--scatter-link-chunk 2097152 exceeds the link slot of 524288"),
            ({"link_slot": mib, "link_chunk": mib + 16}, r"exceeds the link slot of 1048576 bytes \(--link-slot"),
            ({"link_slot": 256 << 10, "gather_link_chunk": 512 << 10},
             r"--gather-link-chunk 524288 exceeds the link slot of 262144 bytes \(--link-slot 262144\)"),
            ({"link_chunk": 1000}, "--link-chunk must be a positive multiple of 16 bytes"),
            ({"reduce_link_chunk": 1000}, "--reduce-link-chunk must be a positive multiple of 16 bytes"),
            ({"link_chunk": 0}, "--link-chunk must be a positive multiple of 16 bytes"),
            ({"link_slot": 6000}, "--link-slot must be a positive multiple of 4096 bytes up to 2147483648"),
            ({"link_slot": (1 << 31) + 4096}, "--link-slot must be a positive multiple of 4096 bytes"),
            ({"link_slot": True}, "--link-slot must be a byte count"),
            ({"window_link_chunk": mib}, r"unknown link sizes \['window_link_chunk'\]")):
        with pytest.raises(ServePlanError, match=message):
            make_plan(repository, link_sizes=sizes)
    assert make_plan(repository, link_sizes={"link_slot": 2 * mib, "scatter_link_chunk": 2 * mib}).ranks[
        1].environment["SIRCL_SCATTER_LINK_CHUNK_BYTES"] == "2097152"
    for variable, option in (("SIRCL_LINK_SLOT_BYTES", "--link-slot"),
                             ("SIRCL_REDUCE_LINK_CHUNK_BYTES", "--reduce-link-chunk")):
        with pytest.raises(ServePlanError, match=f"--env {variable}: the launcher owns every SIRCL_\\* "
                                                 f"variable; use {option}"):
            make_plan(repository, extra_env={variable: "1048576"})
    base = ["--site", str(site_file), "--repository", str(repository), "--profile", PROFILE, "--model-path", MODEL]
    assert cli.main(["plan", "--json", *base, "--link-slot", "1048576", "--link-chunk", "1048576"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert {item["name"]: item["after"] for item in record["changes"]}["SIRCL_LINK_SLOT_BYTES"] == "1048576"
    assert (record["sizes"]["link_slot"], record["sizes"]["link_chunk"]) == (mib, mib)
    assert cli.main(["plan", *base, "--gather-link-chunk", str(2 * mib)]) == 2
    assert "--gather-link-chunk 2097152 exceeds the link slot of 524288 bytes" in capsys.readouterr().err
    flags = ["--link-slot", "4096", "--link-chunk", "16", "--gather-link-chunk", "32", "--scatter-link-chunk", "48",
             "--reduce-link-chunk", "64"]
    expected = {"link_slot": 4096, "link_chunk": 16, "gather_link_chunk": 32, "scatter_link_chunk": 48,
                "reduce_link_chunk": 64}
    for name in ("plan", "preflight", "stage", "start", "wait", "status", "check", "logs", "collect", "stop"):
        args = cli.parser().parse_args([name, "--site", "s", "--repository", "r", *flags])
        assert plan_mod.link_values(args) == expected, name
    args = cli.parser().parse_args(["bundle", "--site", "s", "--positions", "0-7", *flags])
    assert plan_mod.link_values(args) == expected
    bundled = make_bundle(link_sizes={"link_slot": mib, "link_chunk": 256 << 10, "reduce_link_chunk": mib})
    assert bundled.ranks[7].environment["SIRCL_LINK_CHUNK_BYTES"] == "262144"
    assert bundled.ranks[7].environment["SIRCL_REDUCE_LINK_CHUNK_BYTES"] == "1048576"
    assert "SIRCL_LINK_SLOT_BYTES" not in make_bundle().ranks[0].environment
    with pytest.raises(ServePlanError, match="--link-chunk 2097152 exceeds the link slot"):
        make_bundle(link_sizes={"link_chunk": 2 * mib})
    assert cli.main(["bundle", "--site", str(site_file), "--positions", "0-7", "--link-slot", "1048576",
                     "--scatter-link-chunk", "262144"]) == 0
    environment = json.loads(capsys.readouterr().out)["ranks"][0]["environment"]
    assert (environment["SIRCL_LINK_SLOT_BYTES"], environment["SIRCL_SCATTER_LINK_CHUNK_BYTES"]) == (
        "1048576", "262144")


def test_the_ring_minimum_reaches_every_rank_of_start_and_bundle_and_the_plan_records_it(repository, site_file,
                                                                                         capsys):
    lowered = make_plan(repository, ring_min=0, large_schedule="ring")
    assert {launch.environment["SIRCL_RING_MIN_BYTES"] for launch in lowered.ranks} == {"0"}
    assert {change.name: change.reason for change in lowered.changes}["SIRCL_RING_MIN_BYTES"].startswith(
        "--ring-min: the smallest collective a ring schedule runs as a ring op")
    text = plan_mod.render_text(lowered)
    assert "SIRCL_RING_MIN_BYTES: unset -> '0'" in text
    assert ("; ring schedules run ring ops from 0 B of every collective (--ring-min), smaller collectives as under "
            "auto") in text
    assert lowered.to_json()["sizes"]["ring_min"] == 0
    default = make_plan(repository)
    assert "SIRCL_RING_MIN_BYTES" not in default.ranks[0].environment and default.to_json()["sizes"]["ring_min"] is None
    assert ("ring schedules run ring ops from 4,194,304 B of all-reduce message, 8,388,608 B of all-gather output, "
            "4,194,304 B of reduce-scatter input (the session's defaults), smaller collectives as under auto"
            ) in plan_mod.render_text(default)
    for value, message in ((-1, "--ring-min must not be negative, got -1"), (True, "--ring-min must be a byte count")):
        with pytest.raises(ServePlanError, match=re.escape(message)):
            make_plan(repository, ring_min=value)
    with pytest.raises(ServePlanError, match="--env SIRCL_RING_MIN_BYTES: the launcher owns every SIRCL_\\* "
                                             "variable; use --ring-min"):
        make_plan(repository, extra_env={"SIRCL_RING_MIN_BYTES": "0"})
    base = ["--site", str(site_file), "--repository", str(repository), "--profile", PROFILE, "--model-path", MODEL]
    assert cli.main(["plan", "--json", *base, "--ring-min", "4194304"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert {item["name"]: item["after"] for item in record["changes"]}["SIRCL_RING_MIN_BYTES"] == "4194304"
    assert record["sizes"]["ring_min"] == 4194304
    for name in ("plan", "preflight", "stage", "start", "wait", "status", "check", "logs", "collect", "stop"):
        assert cli.parser().parse_args([name, "--site", "s", "--repository", "r", "--ring-min", "0"]).ring_min == 0
    assert make_bundle(ring_min=1 << 20).ranks[5].environment["SIRCL_RING_MIN_BYTES"] == "1048576"
    assert "SIRCL_RING_MIN_BYTES" not in make_bundle().ranks[0].environment
    assert cli.main(["bundle", "--site", str(site_file), "--positions", "0-7", "--ring-min", "1048576"]) == 0
    assert json.loads(capsys.readouterr().out)["ranks"][0]["environment"]["SIRCL_RING_MIN_BYTES"] == "1048576"


def test_the_oneshot_limit_reaches_every_rank_of_start_and_bundle_and_the_plan_records_it(repository, site_file,
                                                                                         capsys):
    limited = make_plan(repository, oneshot_max=65536)
    assert {launch.environment["SIRCL_ONESHOT_MAX_BYTES"] for launch in limited.ranks} == {"65536"}
    reasons = {change.name: change.reason for change in limited.changes}
    assert reasons["SIRCL_ONESHOT_MAX_BYTES"].startswith("--oneshot-max: the largest all-reduce the sessions' auto "
                                                         "algorithm runs one-shot")
    text = plan_mod.render_text(limited)
    assert "SIRCL_ONESHOT_MAX_BYTES: unset -> '65536'" in text
    assert ("dispatch ceiling 131072 B, one-shot all-reduces up to 65,536 B and two-shot above (--oneshot-max), "
            "two-shot and large-message launches on grids of up to 32 blocks (the session's default), all-gather "
            "capacity") in text
    assert limited.to_json()["sizes"]["oneshot_max"] == 65536
    # Unset leaves the limit the sessions derive for their layout; 0 runs every all-reduce two-shot.
    default = make_plan(repository)
    assert "SIRCL_ONESHOT_MAX_BYTES" not in default.ranks[0].environment
    assert default.to_json()["sizes"]["oneshot_max"] is None
    assert default.to_json()["sizes"]["oneshot_max_derived"] == 73728
    assert "one-shot all-reduces up to 73,728 B and two-shot above (the session's limit: the latency model's for this layout, lane count and posting order)" in (
        plan_mod.render_text(default))
    zero = make_plan(repository, oneshot_max=0)
    assert zero.ranks[3].environment["SIRCL_ONESHOT_MAX_BYTES"] == "0"
    assert "two-shot all-reduces only (--oneshot-max 0)" in plan_mod.render_text(zero)
    # The limit is checked against the dispatch ceiling the plan uses (--dispatch, default the capacity).
    assert make_plan(repository, capacity=262144, oneshot_max=262144).ranks[0].environment[
        "SIRCL_ONESHOT_MAX_BYTES"] == "262144"
    for options, message in ((dict(capacity=262144, dispatch=65536, oneshot_max=131072),
                              "--oneshot-max 131072 exceeds the dispatch ceiling of 65536 bytes"),
                             (dict(oneshot_max=-16), "non-negative multiple of 16"),
                             (dict(oneshot_max=True), "--oneshot-max must be a byte count")):
        with pytest.raises(ServePlanError, match=re.escape(message)):
            make_plan(repository, **options)
    with pytest.raises(ServePlanError, match="--env SIRCL_ONESHOT_MAX_BYTES: the launcher owns every SIRCL_\\* "
                                             "variable; use --oneshot-max"):
        make_plan(repository, extra_env={"SIRCL_ONESHOT_MAX_BYTES": "65536"})
    base = ["--site", str(site_file), "--repository", str(repository), "--profile", PROFILE, "--model-path", MODEL]
    assert cli.main(["plan", "--json", *base, "--oneshot-max", "65536"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert {item["name"]: item["after"] for item in record["changes"]}["SIRCL_ONESHOT_MAX_BYTES"] == "65536"
    assert record["sizes"]["oneshot_max"] == 65536
    assert cli.main(["plan", *base, "--oneshot-max", "1000"]) == 2
    assert "--oneshot-max must be a non-negative multiple of 16 bytes, got 1000" in capsys.readouterr().err
    for name in ("plan", "preflight", "stage", "start", "wait", "status", "check", "logs", "collect", "stop"):
        args = cli.parser().parse_args([name, "--site", "s", "--repository", "r", "--oneshot-max", "65536"])
        assert args.oneshot_max == 65536, name
    bundled = make_bundle(oneshot_max=65536)
    assert {launch.environment["SIRCL_ONESHOT_MAX_BYTES"] for launch in bundled.ranks} == {"65536"}
    assert "SIRCL_ONESHOT_MAX_BYTES" not in make_bundle().ranks[0].environment
    with pytest.raises(ServePlanError, match="--oneshot-max 262144 exceeds the dispatch ceiling of 131072 bytes"):
        make_bundle(oneshot_max=262144)
    assert cli.main(["bundle", "--site", str(site_file), "--positions", "0-7", "--oneshot-max", "65536"]) == 0
    assert json.loads(capsys.readouterr().out)["ranks"][0]["environment"]["SIRCL_ONESHOT_MAX_BYTES"] == "65536"


def test_the_profile_reads_its_checkpoints_thinking_behaviour(repository):
    profile = profile_mod.load(repository, PROFILE)
    assert profile.thinking == profile_mod.ThinkingBehaviour("glm53-flash-template", "always", "max",
                                                             ("low", "high", "max"), "reasoning_effort")
    assert profile.sources[profile_mod.THINKING_RELATIVE] == sha(json.dumps(THINKING).encode())
    path = repository / profile_mod.THINKING_RELATIVE
    path.write_text(json.dumps({**THINKING, "checkpoints": {}}))
    assert profile_mod.load(repository, PROFILE).thinking is None             # the checkpoint is not listed
    path.write_text(json.dumps({**THINKING, "checkpoints": {f"{REPOSITORY}@{REVISION}": "unknown-template"}}))
    with pytest.raises(ProfileError, match="names behaviour 'unknown-template', which it does not define"):
        profile_mod.load(repository, PROFILE)
    path.write_text(json.dumps({**THINKING, "schema": "other/v1"}))
    with pytest.raises(ProfileError, match="is not a sparkring-thinking/v1 record"):
        profile_mod.load(repository, PROFILE)
    path.unlink()
    profile = profile_mod.load(repository, PROFILE)
    assert profile.thinking is None and profile_mod.THINKING_RELATIVE not in profile.sources


def test_reasoning_effort_sets_the_api_ranks_chat_template_default_and_the_plan_records_it(repository, site_file,
                                                                                          capsys):
    default = make_plan(repository)
    high = make_plan(repository, reasoning_effort="high")
    # Rank 0 serves the API and gains the argument; the headless ranks run the profile's command unchanged.
    assert high.ranks[0].command == default.ranks[0].command + (
        "--default-chat-template-kwargs", '{"reasoning_effort":"high"}')
    assert [launch.command for launch in high.ranks[1:]] == [launch.command for launch in default.ranks[1:]]
    assert "--default-chat-template-kwargs '{\"reasoning_effort\":\"high\"}'" in high.ranks[0].shell()
    assert high.to_json()["serving"] == {"reasoning_effort": "high",
                                         "default_chat_template_kwargs": {"reasoning_effort": "high"},
                                         "thinking_behaviour": "glm53-flash-template", "template_level": "max",
                                         "thinking_source": "profiles/thinking.json"}
    text = plan_mod.render_text(high)
    assert "--default-chat-template-kwargs '{\"reasoning_effort\":\"high\"}' added on rank 0 (--reasoning-effort)" in text
    assert ("thinking (glm53-flash-template): requests that name no reasoning_effort run at high "
            "(--reasoning-effort; the chat template's own default is max); the launcher's checks keep the "
            "profile's request settings {\"chat_template_kwargs\": {\"reasoning_effort\": \"low\"}}") in text
    assert default.to_json()["serving"]["reasoning_effort"] is None
    assert default.to_json()["serving"]["default_chat_template_kwargs"] is None
    assert ("thinking (glm53-flash-template): requests that name no reasoning_effort run at the chat template's "
            "default, max; --reasoning-effort sets one of low, high, max") in plan_mod.render_text(default)
    checkpoint = f"{REPOSITORY}@{REVISION}"
    with pytest.raises(ServePlanError, match=re.escape(
            f"--reasoning-effort 'medium' is not an effort level of the checkpoint {checkpoint} "
            "(glm53-flash-template); it accepts low, high, max")):
        make_plan(repository, reasoning_effort="medium")
    # The launcher's own checks keep the profile's explicit per-request settings.
    fake = FakeSparks(high)
    assert cli.check(context(high, fake), long_prompt=0) == 0
    bodies = [json.loads(data) for _, command, data in fake.calls if "/v1/chat/completions" in command]
    assert bodies and all(body["chat_template_kwargs"] == {"reasoning_effort": "low"} for body in bodies)
    base = ["--site", str(site_file), "--repository", str(repository), "--profile", PROFILE, "--model-path", MODEL]
    assert cli.main(["plan", "--json", *base, "--reasoning-effort", "high"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["serving"]["reasoning_effort"] == "high"
    assert record["ranks"][0]["argv"][-2:] == ["--default-chat-template-kwargs", '{"reasoning_effort":"high"}']
    assert "--default-chat-template-kwargs" not in record["ranks"][1]["argv"]
    assert cli.main(["plan", *base, "--reasoning-effort", "xhigh"]) == 2
    assert "it accepts low, high, max" in capsys.readouterr().err
    for name in ("plan", "preflight", "stage", "start", "wait", "status", "check", "logs", "collect", "stop"):
        args = cli.parser().parse_args([name, "--site", "s", "--repository", "r", "--reasoning-effort", "low"])
        assert args.reasoning_effort == "low", name
    # A behaviour without levels, a checkpoint the record does not list, and a checkout without the record.
    path = repository / profile_mod.THINKING_RELATIVE
    path.write_text(json.dumps({**THINKING, "checkpoints": {checkpoint: "mimo-v26-flash-template"}}))
    with pytest.raises(ServePlanError, match=r"\(mimo-v26-flash-template\) has no effort levels"):
        make_plan(repository, reasoning_effort="high")
    assert "thinking (mimo-v26-flash-template): the model has no effort levels" in plan_mod.render_text(
        make_plan(repository))
    path.unlink()
    with pytest.raises(ServePlanError, match="records no thinking behaviour for the checkpoint"):
        make_plan(repository, reasoning_effort="high")
    assert f"thinking: profiles/thinking.json records no behaviour for {checkpoint}" in plan_mod.render_text(
        make_plan(repository))


def test_the_chat_template_default_merges_into_a_value_the_command_already_gives():
    command = ["serve", "--default-chat-template-kwargs", '{"clear_thinking":true}', "--port", "8017"]
    plan_mod.with_chat_template_kwargs(command, {"reasoning_effort": "high"})
    assert command[2] == '{"clear_thinking":true,"reasoning_effort":"high"}' and command[3:] == ["--port", "8017"]
    inline = ["serve", '--default-chat-template-kwargs={"reasoning_effort":"max"}']
    plan_mod.with_chat_template_kwargs(inline, {"reasoning_effort": "low"})
    assert inline == ["serve", '--default-chat-template-kwargs={"reasoning_effort":"low"}']
    for broken, message in ((["--default-chat-template-kwargs", "[1]"], "is not a JSON object"),
                            (["--default-chat-template-kwargs"], "at most once, with a value"),
                            (["--default-chat-template-kwargs", "{}", "--default-chat-template-kwargs", "{}"],
                             "at most once")):
        with pytest.raises(ServePlanError, match=re.escape(message)):
            plan_mod.with_chat_template_kwargs(list(broken), {"reasoning_effort": "high"})


def test_the_bundle_gives_rank_zero_the_chat_template_default_of_a_checkpoint(repository, site_file, capsys):
    checkpoint = f"{REPOSITORY}@{REVISION}"
    bundled = make_bundle(reasoning_effort="high", repository=str(repository), checkpoint=checkpoint)
    assert bundled.ranks[0].vllm_arguments == ("--default-chat-template-kwargs", '{"reasoning_effort":"high"}')
    assert all(launch.vllm_arguments == () for launch in bundled.ranks[1:])
    document = bundled.to_json()
    assert document["serving"] == {"reasoning_effort": "high",
                                   "default_chat_template_kwargs": {"reasoning_effort": "high"},
                                   "checkpoint": checkpoint, "thinking_behaviour": "glm53-flash-template",
                                   "template_level": "max", "thinking_source": "profiles/thinking.json"}
    assert document["merge"]["--default-chat-template-kwargs"].startswith("rank 0 only, the rank that serves the API")
    assert document["ranks"][0]["vllm_arguments"] == ["--default-chat-template-kwargs", '{"reasoning_effort":"high"}']
    plain = make_bundle().to_json()
    assert plain["serving"]["reasoning_effort"] is None and "--default-chat-template-kwargs" not in plain["merge"]
    assert plain["ranks"][0]["vllm_arguments"] == []
    for options, message in (
            ({"reasoning_effort": "high"}, "--reasoning-effort needs --repository"),
            ({"reasoning_effort": "high", "repository": str(repository)}, "--reasoning-effort needs --repository"),
            ({"repository": str(repository)}, "give it with one of them"),
            ({"reasoning_effort": "high", "repository": str(repository), "checkpoint": REPOSITORY},
             "is not REPOSITORY@REVISION"),
            ({"reasoning_effort": "high", "repository": str(repository), "checkpoint": "other/model@abc"},
             "records no thinking behaviour for the checkpoint other/model@abc"),
            ({"reasoning_effort": "medium", "repository": str(repository), "checkpoint": checkpoint},
             "it accepts low, high, max"),
            ({"reasoning_effort": "high", "repository": str(repository / "missing"), "checkpoint": checkpoint},
             "has no profiles/thinking.json")):
        with pytest.raises(ServePlanError, match=re.escape(message)):
            make_bundle(**options)
    assert cli.main(["bundle", "--site", str(site_file), "--positions", "0-7", "--reasoning-effort", "low",
                     "--repository", str(repository), "--checkpoint", checkpoint]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["ranks"][0]["vllm_arguments"] == ["--default-chat-template-kwargs", '{"reasoning_effort":"low"}']


def _module_constants(path: Path) -> dict[str, object]:
    """Module-level ``NAME = <integer expression or constant>`` assignments of a source file, evaluated."""
    operators = {ast.LShift: lambda a, b: a << b, ast.Mult: lambda a, b: a * b, ast.Add: lambda a, b: a + b,
                 ast.Sub: lambda a, b: a - b}

    def value(node: ast.AST) -> object:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            return operators[type(node.op)](value(node.left), value(node.right))
        if isinstance(node, ast.Dict) and None not in node.keys:
            return {value(key): value(item) for key, item in zip(node.keys, node.values)}
        if isinstance(node, ast.Tuple):
            return tuple(value(item) for item in node.elts)
        raise ValueError(ast.dump(node))

    constants = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                constants[node.targets[0].id] = value(node.value)
            except ValueError:
                continue
    return constants


def test_the_launchers_link_size_defaults_and_limits_are_the_sessions():
    package = Path(__file__).resolve().parents[1] / "sparkring_sircl"
    if not (package / "oneshot" / "runtime.py").is_file():
        pytest.skip("the session package is not present")
    from sparkring_sircl import protocol

    runtime = _module_constants(package / "oneshot" / "runtime.py")
    assert runtime["DEFAULT_LINK_SLOT_BYTES"] == plan_mod.DEFAULT_LINK_SLOT_BYTES
    assert runtime["DEFAULT_LINK_CHUNK_BYTES"] == plan_mod.DEFAULT_LINK_CHUNK_BYTES
    assert runtime["MAX_AUTO_LINK_SLOT_BYTES"] == plan_mod.MAX_AUTO_LINK_SLOT_BYTES
    assert runtime["DEFAULT_CHAIN_MINS"] == plan_mod.DEFAULT_CHAIN_MINS
    assert runtime["DEFAULT_RING_MINS"] == plan_mod.DEFAULT_RING_MINS
    from sparkring_sircl.vllm import sessionapi as api

    assert runtime["MIN_COLLECTIVES"] == plan_mod.MIN_COLLECTIVES == api.MIN_COLLECTIVES
    collectives = next(ast.literal_eval(node.value) for node in ast.parse(
        (package / "oneshot" / "runtime.py").read_text(encoding="utf-8")).body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "LINK_COLLECTIVES")
    from sparkring_sircl.vllm import sessionapi

    assert tuple(collectives) == sessionapi.LINK_COLLECTIVES
    assert {f"{name}_link_chunk": variable for name, variable in collectives.items()} == {
        size.attribute: size.variable for size in plan_mod.LINK_SIZES if size.attribute.endswith("_link_chunk")}
    assert protocol.SLOT_ALIGNMENT == plan_mod.LINK_SLOT_ALIGNMENT
    with pytest.raises(protocol.ProtocolError):
        protocol.LinkLayout(2, 8, plan_mod.MAX_LINK_SLOT_BYTES + plan_mod.LINK_SLOT_ALIGNMENT)
    protocol.LinkLayout(2, 8, plan_mod.MAX_LINK_SLOT_BYTES)
    from sparkring_sircl import env as session_env

    variables = {variable.name for variable in session_env.VARIABLES}
    assert {size.variable for size in plan_mod.LINK_SIZES} | {plan_mod.RING_MIN_VARIABLE,
                                                               plan_mod.CHAIN_MIN_VARIABLE} <= variables


def test_the_emulated_sessions_minimums_are_the_launchers():
    """The CPU reference session (it imports torch) reports the session package's default minimums."""
    pytest.importorskip("torch")
    from sparkring_sircl.vllm import emulation

    assert (emulation.DEFAULT_CHAIN_MINS, emulation.DEFAULT_RING_MINS) == (plan_mod.DEFAULT_CHAIN_MINS,
                                                                           plan_mod.DEFAULT_RING_MINS)


def test_the_launchers_oneshot_default_is_the_sessions():
    package = Path(__file__).resolve().parents[1] / "sparkring_sircl"
    if not (package / "oneshot" / "runtime.py").is_file():
        pytest.skip("the session package is not present")
    from sparkring_sircl import env as session_env
    from sparkring_sircl import protocol

    runtime = _module_constants(package / "oneshot" / "runtime.py")
    assert runtime["DEFAULT_ONESHOT_MAX_BYTES"] == plan_mod.DEFAULT_ONESHOT_MAX_BYTES
    assert plan_mod.ONESHOT_VARIABLE in {variable.name for variable in session_env.VARIABLES}
    # The session's auto choice at the limit and one pack above it.
    choose = protocol.select_algorithm
    assert (choose(65536, oneshot_max_bytes=65536), choose(65552, oneshot_max_bytes=65536)) == ("oneshot", "twoshot")
    assert choose(16, oneshot_max_bytes=0) == "twoshot"


def test_recipes_with_vllm_micro_batching_do_not_plan(repository):
    profile = profile_mod.load(repository, PROFILE)
    site = ServeSite.from_json(SITE)
    tree = staging.staged_tree()
    for extra in (("--enable-dbo",), ("--ubatch-size", "2")):
        batched = dataclasses.replace(profile, recipe_arguments=profile.recipe_arguments + extra)
        with pytest.raises(ServePlanError, match=r"refuses this profile: vLLM's micro-batching \(--enable-dbo"):
            plan_mod.build_plan(site, batched, Options((0, 1, 2, 3), model_path=MODEL), staged_digest=tree.digest,
                                library=staging.library_name())
    assert plan_mod.micro_batching(["--ubatch-size", "1"]) is None and plan_mod.micro_batching([]) is None


# Statistics of a TP4 session whose links and ring run: one-shot all-reduces up to 131,072 B, all-reduce pieces
# of 4 MiB, all-gather pieces of 155,648 B (19 rows of 8,192 B), plain reduce-scatter ops of 256 KiB (64 KiB of
# each rank's chunk), a 2 MiB chain threshold, 512 KiB link chunks and every schedule auto.
LINK_SESSION = {"world_size": 4, "dispatch_limit_bytes": 131072, "large_piece_bytes": 4 << 20,
                "gather_piece_bytes": 155648, "chain_available": True, "chain_min_bytes": 2 << 20,
                "link_available": True, "ring_available": True, "link_chunk_bytes": 512 << 10,
                "relay_safe_bytes": None, "scatter_op_bytes": 256 << 10, "large_schedule": "auto",
                "gather_schedule": "auto", "scatter_schedule": "auto"}
REDUCE_CLAUSE = "one chain op per all-reduce from 2,097,152 B, else all-reduce pieces of 4,194,304 B above 131,072 B"
SCATTER_AUTO = "one chain op per reduce-scatter from 2,097,152 B, else reduce-scatter pieces of 262,144 B"
GATHER_AUTO = "one chain op per all-gather from 2,097,152 B, else all-gather pieces of 155,648 B"


def test_session_ops_count_ring_schedules_from_the_session_statistics():
    stats = {"dispatch_limit_bytes": 131072, "large_piece_bytes": 4 << 20, "gather_piece_bytes": 4 << 20,
             "world_size": 4, "chain_available": True, "chain_min_bytes": 2 << 20, "ring_available": True,
             "large_schedule": "ring", "gather_schedule": "ring"}
    # A session without ring_min_bytes runs a ring schedule as ring ops at every size (from 16 bytes per rank).
    ring = plan_mod.SessionOps.from_stats(stats)
    assert (ring.reduce_ring, ring.ring_min, ring.chain_min, ring.gather_ring) == (True, 0, 2 << 20, True)
    assert ring.allreduce_ops(64 << 20) == 1 and ring.gather_ops(8192, 8192) == 1
    assert ring.describe().startswith("one ring op per all-reduce from 64 B, else all-reduce pieces")
    assert "one ring op per all-gather from 64 B" in ring.describe()
    lacking = plan_mod.SessionOps.from_stats({**stats, "ring_available": False})
    assert (lacking.reduce_ring, lacking.chain_min, lacking.gather_ring) == (False, 2 << 20, False)  # runs as auto
    pieces = plan_mod.SessionOps.from_stats({**stats, "large_schedule": "pieces", "gather_schedule": "auto"})
    assert pieces.allreduce_ops(64 << 20) == 16 and pieces.gather_ops(8192, 8192) == 16


@pytest.mark.parametrize("ring_min, clause, ops", [
    # 1 MiB, 3 MiB, 64 MiB, 64 MiB + 8 and 64 MiB + 32 bytes on four ranks: a ring op takes the largest
    # prefix of four chunks of whole 16-byte packs, its remainder one piece, and a tail below 16 bytes one op.
    (0, "one ring op per all-reduce from 64 B, else all-reduce pieces of 4,194,304 B above 131,072 B",
     (1, 1, 1, 2, 2)),
    (2 << 20, "one ring op per all-reduce from 2,097,152 B, else all-reduce pieces of 4,194,304 B above "
              "131,072 B", (1, 1, 1, 2, 2)),
    # Below a 4 MiB ring minimum the ring schedule runs as auto: one chain op from the 2 MiB chain threshold.
    (4 << 20, "one ring op per all-reduce from 4,194,304 B, else one chain op per all-reduce from 2,097,152 B, "
              "else all-reduce pieces of 4,194,304 B above 131,072 B", (1, 1, 1, 2, 2)),
    (128 << 20, "one ring op per all-reduce from 134,217,728 B, else one chain op per all-reduce from "
                "2,097,152 B, else all-reduce pieces of 4,194,304 B above 131,072 B", (1, 1, 1, 2, 1)),
])
def test_session_ops_run_a_ring_all_reduce_schedule_as_auto_below_the_ring_minimum(ring_min, clause, ops):
    session = plan_mod.SessionOps.from_stats({**LINK_SESSION, "large_schedule": "ring", "ring_min_bytes": ring_min})
    assert session.describe().split("; ")[0] == clause
    sizes = (1 << 20, 3 << 20, 64 << 20, (64 << 20) + 8, (64 << 20) + 32)
    assert tuple(session.allreduce_ops(nbytes) for nbytes in sizes) == ops
    # Without the chain, a ring schedule below its minimum runs in pieces: 12 MiB in three of 4 MiB.
    unchained = plan_mod.SessionOps.from_stats({**LINK_SESSION, "large_schedule": "ring", "ring_min_bytes": 16 << 20,
                                                "chain_available": False})
    assert unchained.allreduce_ops(12 << 20) == 3 and unchained.allreduce_ops(16 << 20) == 1


@pytest.mark.parametrize("schedule, ring_runs, links_run, ring_min, clause, ops", [
    ("auto", True, True, 2 << 20, GATHER_AUTO, (2, 1, 1, 57)),
    ("chain", True, True, 2 << 20, "one chain op per all-gather from 64 B, else all-gather pieces of 155,648 B",
     (1, 1, 1, 57)),
    ("ring", True, True, 0, "one ring op per all-gather from 64 B, else all-gather pieces of 155,648 B",
     (1, 1, 1, 57)),
    # Below the ring minimum a ring schedule runs as auto: the 1 MiB output in pieces, chain ops from 2 MiB.
    ("ring", True, True, 2 << 20, "one ring op per all-gather from 2,097,152 B, else all-gather pieces of "
                                  "155,648 B", (2, 1, 1, 57)),
    ("ring", True, True, 4 << 20, "one ring op per all-gather from 4,194,304 B, else one chain op per all-gather "
                                  "from 2,097,152 B, else all-gather pieces of 155,648 B", (2, 1, 1, 57)),
    ("pieces", True, True, 2 << 20, "all-gather pieces of 155,648 B", (2, 4, 108, 57)),
    ("ring", False, True, 2 << 20, GATHER_AUTO, (2, 1, 1, 57)),        # a ring schedule without a ring runs as auto
    ("auto", False, False, 2 << 20, "all-gather pieces of 155,648 B", (2, 4, 108, 57)),   # no links: pieces only
])
def test_session_ops_describe_and_count_all_gathers_as_the_gather_schedule_runs_them(schedule, ring_runs, links_run,
                                                                                      ring_min, clause, ops):
    session = plan_mod.SessionOps.from_stats({**LINK_SESSION, "gather_schedule": schedule, "ring_min_bytes": ring_min,
                                              "ring_available": ring_runs, "link_available": links_run})
    scatter = SCATTER_AUTO if links_run else "reduce-scatter pieces of 262,144 B"
    assert session.describe() == f"{REDUCE_CLAUSE}; {scatter}; {clause}"
    # Shards of 32, 64 and 2,048 rows of 8,192 B (outputs of 1 MiB, 2 MiB and 64 MiB) and of 1,025 rows of
    # 8,200 B, a shard that is not a multiple of 16 bytes and so never a link op.
    shapes = ((32, 8192), (64, 8192), (2048, 8192), (1025, 8200))
    assert tuple(session.gather_ops(rows, row_bytes) for rows, row_bytes in shapes) == ops


@pytest.mark.parametrize("schedule, ring_runs, ring_min, clause, ops", [
    ("auto", True, 2 << 20, SCATTER_AUTO, (4, 1, 1)),
    ("chain", True, 2 << 20, "one chain op per reduce-scatter from 64 B, else reduce-scatter pieces of 262,144 B",
     (1, 1, 1)),
    ("ring", True, 0, "one ring op per reduce-scatter from 64 B, else reduce-scatter pieces of 262,144 B", (1, 1, 1)),
    # Below the ring minimum a ring schedule runs as auto: the 1 MiB input in four plain ops, chain ops from 2 MiB.
    ("ring", True, 2 << 20, "one ring op per reduce-scatter from 2,097,152 B, else reduce-scatter pieces of "
                            "262,144 B", (4, 1, 1)),
    ("ring", True, 4 << 20, "one ring op per reduce-scatter from 4,194,304 B, else one chain op per reduce-scatter "
                            "from 2,097,152 B, else reduce-scatter pieces of 262,144 B", (4, 1, 1)),
    ("pieces", True, 2 << 20, "reduce-scatter pieces of 262,144 B", (4, 8, 256)),
    ("ring", False, 2 << 20, SCATTER_AUTO, (4, 1, 1)),                 # a ring schedule without a ring runs as auto
])
def test_session_ops_describe_and_count_reduce_scatters_as_the_scatter_schedule_runs_them(schedule, ring_runs,
                                                                                          ring_min, clause, ops):
    session = plan_mod.SessionOps.from_stats({**LINK_SESSION, "scatter_schedule": schedule,
                                              "ring_available": ring_runs, "ring_min_bytes": ring_min})
    assert session.describe() == f"{REDUCE_CLAUSE}; {clause}; {GATHER_AUTO}"
    # Inputs of 1 MiB, 2 MiB and 64 MiB: each rank's chunk is 256 KiB, 512 KiB and 16 MiB.
    assert tuple(session.scatter_ops(nbytes) for nbytes in (1 << 20, 2 << 20, 64 << 20)) == ops


def test_session_ops_bound_link_reduce_scatters_and_carry_them_as_all_reduces_without_a_session_reduce_scatter():
    capped = plan_mod.SessionOps.from_stats({**LINK_SESSION, "scatter_schedule": "pieces", "relay_safe_bytes": 32768})
    assert capped.scatter_piece == 131072 and capped.scatter_ops(64 << 20) == 512        # 32 KiB of each chunk
    chained = plan_mod.SessionOps.from_stats({**LINK_SESSION, "scatter_schedule": "chain"})
    assert chained.scatter_ops((1 << 31) - 64) == 1 and chained.scatter_ops(1 << 31) == 8192   # below 2 GiB only
    assert chained.scatter_ops((64 << 20) + 16) == 257                  # not 16 bytes per rank: plain ops
    # A chunk may span at most 65,536 link chunks: 1 MiB chunks of 16-byte link chunks do, 16 MiB ones do not.
    tiny = plan_mod.SessionOps.from_stats({**LINK_SESSION, "scatter_schedule": "chain", "link_chunk_bytes": 16})
    assert tiny.scatter_ops(4 << 20) == 1 and tiny.scatter_ops(64 << 20) == 256
    # The reduce-scatter's own link piece (link_chunks["scatter"]) sets that limit, not the session's link piece.
    own = plan_mod.SessionOps.from_stats({**LINK_SESSION, "scatter_schedule": "chain",
                                          "link_chunks": {"gather": 1 << 20, "scatter": 16, "reduce": 512 << 10}})
    assert own.scatter_link_chunk == 16 and (own.scatter_ops(4 << 20), own.scatter_ops(64 << 20)) == (1, 256)
    assert plan_mod.SessionOps.from_stats({**LINK_SESSION, "link_chunks": {"scatter": None}}).scatter_link_chunk == (
        512 << 10)                                                          # no piece of its own: the link piece
    plain = plan_mod.SessionOps.from_stats({**LINK_SESSION, "scatter_op_bytes": None})
    assert plain.scatter_piece is None
    assert plain.describe() == f"{REDUCE_CLAUSE}; reduce-scatters carried as all-reduces; {GATHER_AUTO}"
    with pytest.raises(ValueError, match="carries reduce-scatters as all-reduces"):
        plain.scatter_ops(64 << 20)
    sharded = plan_mod.prefill_estimate(16384, chunk_tokens=8192, mhc_world=4)
    assert sharded.ops(plain) == (8 + 180, 0, 180) and sharded.ops(chained) == (8, 180, 180)


def test_the_long_prompt_line_counts_one_op_per_chain_reduce_scatter_and_all_gather(serve_plan, monkeypatch):
    """Rank 0's statistics of a run with ``--large-schedule auto --gather-schedule auto --scatter-schedule
    chain``, 4 MiB pieces and mHC row ownership at TP4: each reduce-scatter of 64 MiB and each all-gather of a
    16 MiB shard is one chain op."""
    monkeypatch.setitem(plan_mod.PATH_COMPUTE_SECONDS_PER_TOKEN, serve_plan.profile.id, 0.26e-3)
    stats = {**LINK_SESSION, "gather_piece_bytes": 4 << 20, "scatter_op_bytes": 4 << 20, "scatter_schedule": "chain"}
    line = cli._long_prompt_line(serve_plan, 17056, 5.17, stats)
    # 17,056 tokens: two sharded chunks (4 all-reduces, 90 reduce-scatters and 90 all-gathers each) and 672
    # rows (94 all-reduces of 5.25 MiB); every one of the 462 collectives is one chain op.
    assert ("session ops (one chain op per all-reduce from 2,097,152 B, else all-reduce pieces of 4,194,304 B "
            "above 131,072 B; one chain op per reduce-scatter from 64 B, else reduce-scatter pieces of 4,194,304 B; "
            "one chain op per all-gather from 2,097,152 B, else all-gather pieces of 4,194,304 B, from rank 0's "
            "receipt): 102 all-reduce, 180 reduce-scatter and 180 all-gather ops") in line
    assert "after compute of 3.6 s (measured on the path) the collectives took 1.6 s, 3.38 ms per op" in line


def test_the_link_piece_limit_is_the_sessions():
    links = Path(__file__).resolve().parents[1] / "sparkring_sircl" / "oneshot" / "_links_cute.py"
    if not links.is_file():
        pytest.skip("the session package is not present")
    tree = ast.parse(links.read_text(encoding="utf-8"))
    constants = {target.id: node.value.value for node in tree.body if isinstance(node, ast.Assign)
                 for target in node.targets if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant)}
    assert constants["PIECE_COUNTERS"] == plan_mod.LINK_PIECE_COUNTERS


def test_the_launchers_wait_defaults_are_the_sessions():
    runtime = Path(__file__).resolve().parents[1] / "sparkring_sircl" / "oneshot" / "runtime.py"
    if not runtime.is_file():
        pytest.skip("the session package is not present")
    tree = ast.parse(runtime.read_text(encoding="utf-8"))
    constants = {target.id: node.value.value for node in tree.body if isinstance(node, ast.Assign)
                 for target in node.targets if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant)}
    assert constants["DEFAULT_STARTUP_WAIT_S"] == plan_mod.DEFAULT_STARTUP_WAIT_S
    assert constants["DEFAULT_SERVING_WAIT_S"] == plan_mod.DEFAULT_SERVING_WAIT_S


def test_env_pass_through_reaches_every_rank_and_refuses_launcher_keys(repository):
    extra = plan_mod.parse_env(["VLLM_TORCH_PROFILER_DIR=/tmp/profile", "B12X_POLICY_MODE=fixed",
                                "VLLM_TORCH_PROFILER_DIR=/run/profile"])
    assert extra == {"VLLM_TORCH_PROFILER_DIR": "/run/profile", "B12X_POLICY_MODE": "fixed"}
    plan = make_plan(repository, extra_env=extra)
    for launch in plan.ranks:
        assert launch.environment["VLLM_TORCH_PROFILER_DIR"] == "/run/profile"
        assert "VLLM_TORCH_PROFILER_DIR=/run/profile" in launch.shell()
    change = next(change for change in plan.changes if change.name == "VLLM_TORCH_PROFILER_DIR")
    assert (change.before, change.after, change.reason) == (None, "/run/profile", "--env")
    assert "VLLM_TORCH_PROFILER_DIR: unset -> '/run/profile'  (--env)" in plan_mod.render_text(plan)
    for item, message in (("SIRCL_SPIN_LIMIT=5", "owns every SIRCL_\\* variable"),
                          ("VLLM_PLUGINS=x", "adds the sircl plugins"), ("PYTHONPATH=/x", "staged package"),
                          ("VLLM_ENABLE_ROCE_ALLREDUCE=1", "own transports off"),
                          ("VLLM_GLM53_MHC_PREFILL_SHARD=0", "--mhc-prefill-shard"),
                          ("NOT A NAME=1", "not an environment variable name"), ("NOEQUALS", "not KEY=VALUE")):
        with pytest.raises(ServePlanError, match=message):
            plan_mod.parse_env([item])
    with pytest.raises(ServePlanError, match="owns every SIRCL"):
        make_plan(repository, extra_env={"SIRCL_GROUPS": "tp,dcp"})


# -- bundle: SIRCL for another launcher's containers ----------------------------------------------------


def make_bundle(**options):
    values = {"positions": tuple(range(8)), "nccl_mode": "auto", **options}
    return bundle.build_bundle(ServeSite.from_json(SITE), bundle.BundleOptions(**values),
                               staged_digest=staging.staged_tree().digest, library=staging.library_name())


def test_a_bundle_for_the_whole_ring_states_every_ranks_mounts_environment_and_merges():
    plan = make_bundle(nccl_mode="never")
    assert plan.run_id == "bundle-0-7" and plan.group.fabric.describe() == "cycle:0-1-2-3-4-5-6-7"
    assert plan.nccl_policy.value == "none" and plan.nccl_reason == "SIRCL_NCCL=never"
    record = json.loads(json.dumps(plan.to_json("/etc/sircl/site.json")))
    assert record["schema"] == "sircl-vllm-bundle/v1" and len(record["ranks"]) == 8
    assert record["security_options"] == [] and record["remote"]["receipts"] == (
        "/tmp/sircl-ring/serve/runs/bundle-0-7/receipts")
    assert record["check"] == ("python -m sparkring_sircl.vllm.serve bundle-check --site /etc/sircl/site.json "
                               "--positions 0,1,2,3,4,5,6,7 --run-id bundle-0-7 --nccl never --container "
                               "'NAME-{rank}'")
    for rank, entry in enumerate(record["ranks"]):
        environment = entry["environment"]
        assert entry["position"] == rank and environment["VLLM_HOST_IP"] == f"192.0.2.{20 + rank}"
        assert environment["SIRCL_FABRIC"] == "ring:8" and environment["SIRCL_RANK_POSITIONS"] == "0,1,2,3,4,5,6,7"
        assert environment["SIRCL_GROUPS"] == "tp" and environment["SIRCL_NCCL"] == "never"
        assert environment["GLOO_SOCKET_IFNAME"] == environment["NCCL_SOCKET_IFNAME"] == "enP7s7"
        assert environment["VLLM_ENABLE_ROCE_ALLREDUCE"] == "0" and environment["SIRCL_STARTUP_WAIT_S"] == "600"
        assert environment["SIRCL_NATIVE_LIBRARY"] == f"/sircl/build-cache/{plan.library}"
        assert environment["SIRCL_FUSED_NORM"] == "0"
        assert not {"PYTHONPATH", "VLLM_PLUGINS", "NCCL_IB_HCA", "SIRCL_LARGE_PIECE_BYTES"} & set(environment)
        assert (entry["pythonpath_prepend"], entry["vllm_plugins_add"]) == ("/sircl/src", "sircl")
        assert [mount["target"] for mount in entry["mounts"]] == ["/sircl/src", "/sircl/build-cache", "/sircl/run"]
        assert entry["mounts"][0]["option"] == (f"type=bind,src=/tmp/sircl-ring/serve/src/{plan.staged_digest},"
                                                "dst=/sircl/src,readonly")
        assert "--env" in entry["docker_args"] and "SIRCL_MODE=custom" in entry["docker_args"]
    assert record["route_maps"]["0"].startswith("1=rocep1s0f0/roceP2p1s0f0")
    assert any(row["ranks"] == [0, 4] for row in record["relays"])


def test_bundle_options_reach_the_environment_and_bad_ones_are_refused():
    ring = make_bundle(nccl_mode="topology", capacity=4194304, dispatch=131072, gather=4194304,
                       session_groups="tp,dcp", extra_env={"VLLM_TORCH_PROFILER_DIR": "/run/profile"},
                       startup_wait=1800, run_id="glm53-tp8", fused_norm=True)
    assert ring.nccl_policy.value == "ring"
    first = ring.ranks[0].environment
    assert (first["NCCL_ALGO"], first["NCCL_SKIP_TREE_CONNECT"]) == ("Ring", "1")
    assert "NCCL_SKIP_TREE_CONNECT" in ring.to_json()["merge"]
    assert "NCCL_ALGO" not in make_bundle(nccl_mode="never").ranks[0].environment   # no NCCL settings
    with pytest.raises(ServePlanError, match="NCCL's ring on this cycle needs"):
        make_bundle(nccl_mode="topology", extra_env={"NCCL_ALGO": "Tree"})
    assert first["NCCL_IB_HCA"].startswith("=") and first["SIRCL_GROUPS"] == "tp,dcp"
    assert (first["SIRCL_ALLREDUCE_CAPACITY_BYTES"], first["SIRCL_ALLREDUCE_DISPATCH_LIMIT_BYTES"],
            first["SIRCL_ALLGATHER_MAX_BYTES"]) == ("4194304", "131072", "4194304")
    assert first["VLLM_TORCH_PROFILER_DIR"] == "/run/profile" and first["SIRCL_STARTUP_WAIT_S"] == "1800"
    assert first["SIRCL_FUSED_NORM"] == "1"
    assert ring.ranks[0].directories == ("/tmp/sircl-ring/serve/runs/glm53-tp8",
                                         "/tmp/sircl-ring/serve/runs/glm53-tp8/receipts")
    path = make_bundle(positions=(7, 0, 1, 2))
    assert path.group.fabric.describe() == "path:7-0-1-2" and path.ranks[0].environment["SIRCL_RANK_POSITIONS"] == (
        "7,0,1,2")
    for options, message in (({"positions": (0, 2)}, "consecutive"), ({"positions": (0,)}, "at least two"),
                             ({"extra_env": {"SIRCL_GROUPS": "dcp"}}, "owns every SIRCL"),
                             ({"session_groups": "dcp"}, "session groups"), ({"dispatch": 1 << 30}, "dispatch"),
                             ({"run_id": "Bad Id"}, "run id")):
        with pytest.raises(ServePlanError, match=message):
            make_bundle(**options)


class BundleSparks:
    """Answers the bundle's remote commands like healthy Sparks with one running container each."""

    def __init__(self, plan, *, receipts=None, state="running 0 none"):
        self.plan = plan
        self.calls = []
        self.receipts = receipts
        self.state = state

    def __call__(self, target, command, *, timeout=60, input_bytes=None):
        self.calls.append((target, command, input_bytes))
        if ".sircl-staged" in command and "tar -C" in command:
            return remote.Result(0, "staged\n", "")
        if "sparkring_sircl.vllm.serve.probe" in command:
            return remote.Result(0, probe.PREFIX + json.dumps(good_record(self.plan.library)) + "\n", "")
        if "mkdir -p" in command:
            return remote.Result(0, "", "")
        if " inspect -f " in command:
            return remote.Result(0, self.state + "\n", "")
        if "| grep -F" in command:
            rank = next(launch.rank for launch in self.plan.ranks if launch.ssh == target)
            return remote.Result(0, f"INFO adapter.py:292] SIRCL receipt group=tp:0 global_rank={rank} nccl=none "
                                    "pynccl=skipped session=ring state=ready\n"
                                    "INFO plugin.py:72] SIRCL vLLM adapter registered: layout ring:8\n", "")
        if "-*.json" in command:
            rank = int(re.search(r"for f in rank(\d+)-", command).group(1))
            record = {"group": "tp:0", "global_rank": rank, "rank": rank, "world": 8, "nccl": "none",
                      "pynccl": "skipped", "session": "ring", "state": "ready",
                      "decisions": [{"collective": "all_reduce", "backend": "sircl", "method": "direct", "calls": 9}]}
            records = self.receipts(rank) if self.receipts else [record]
            return remote.Result(0, "".join(f"rank{rank}-tp-0.json\t{json.dumps(r)}\n" for r in records), "")
        raise AssertionError(f"unexpected command on {target}: {command[:160]}")


def test_bundle_stage_builds_in_the_given_image_and_check_reads_each_named_container(site_file):
    plan = make_bundle(image="registry.example/glm53:tp8")
    fake = BundleSparks(plan)
    lines: list[str] = []
    assert bundle.stage(plan, staging.staged_tree(), fake, lines.append) == 0
    assert lines[-1] == "bundle staged"
    builds = [command for _, command, _ in fake.calls if "serve.probe" in command]
    assert len(builds) == 8 and all("registry.example/glm53:tp8" in command for command in builds)
    assert all("sudo -n sh -c 'mkdir -p /tmp/sircl-ring/serve/runs/bundle-0-7" in command
               for _, command, _ in fake.calls if command.startswith("sudo -n sh -c 'mkdir"))
    names = bundle.container_names(["glm53-r{rank}", "3=special"], plan)
    assert names[0] == "glm53-r0" and names[3] == "special"
    fake.calls.clear()
    lines.clear()
    assert bundle.check(plan, names, fake, lines.append) == 0
    assert lines[-1] == "bundle bundle-0-7: check passed"
    assert any("container special" in line for line in lines)
    assert any("docker inspect" in command and "special" in command for _, command, _ in fake.calls)
    stopped = BundleSparks(plan, state="exited 1 none")
    lines.clear()
    assert bundle.check(plan, names, stopped, lines.append) == 1
    assert any("container state exited 1 none" in line for line in lines)
    with pytest.raises(ServePlanError, match="no valid container name"):
        bundle.container_names(["3=only"], plan)


def test_the_bundle_command_prints_json_and_contacts_nothing_without_stage(site_file, capsys):
    calls = []

    def run(*args, **kwargs):
        calls.append(args)
        raise AssertionError("contacted a Spark")

    assert cli.main(["bundle", "--site", str(site_file), "--positions", "0-7", "--fused-norm", "on"],
                    run=run) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["schema"] == "sircl-vllm-bundle/v1" and record["run_id"] == "bundle-0-7" and calls == []
    assert record["ranks"][5]["environment"]["SIRCL_FUSED_NORM"] == "1"
    assert cli.main(["bundle", "--site", str(site_file), "--positions", "0,2"], run=run) == 2


def test_bundle_stage_writes_only_the_json_document_to_the_standard_output_stream(site_file):
    """The same at the process level: a real interpreter, its standard output and error as separate pipes."""
    tests = Path(__file__).resolve().parent
    script = ("import sys\n"
              "import test_vllm_serve as t\n"
              "plan = t.make_bundle(run_id='glm53-tp8')\n"
              "sys.exit(t.cli.main(['bundle', '--site', sys.argv[1], '--positions', '0-7', '--run-id', "
              "'glm53-tp8', '--stage'], run=t.BundleSparks(plan)))\n")
    paths = [str(tests), str(tests.parent), os.environ.get("PYTHONPATH", "")]
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(path for path in paths if path)}
    done = subprocess.run([sys.executable, "-c", script, str(site_file)], capture_output=True, cwd=tests.parent,
                          env=environment, timeout=600)
    assert done.returncode == 0, done.stderr.decode(errors="replace")[-2000:]
    record = json.loads(done.stdout)                      # standard output is the JSON document alone
    assert record["schema"] == "sircl-vllm-bundle/v1" and record["run_id"] == "glm53-tp8"
    progress = done.stderr.decode().splitlines()
    assert progress[-1] == "bundle staged"
    assert sum("package tree" in line for line in progress) == 8


def test_bundle_stage_keeps_standard_output_for_the_json_document(site_file, capsys):
    plan = make_bundle(run_id="glm53-tp8")
    fake = BundleSparks(plan)
    code = cli.main(["bundle", "--site", str(site_file), "--positions", "0-7", "--run-id", "glm53-tp8",
                     "--stage"], run=fake)
    captured = capsys.readouterr()
    assert code == 0
    record = json.loads(captured.out)                     # the whole of standard output parses
    assert record["schema"] == "sircl-vllm-bundle/v1" and record["run_id"] == "glm53-tp8"
    lines = captured.err.splitlines()
    assert lines[-1] == "bundle staged"
    assert sum(line.startswith("bundle glm53-tp8 rank ") and "package tree" in line for line in lines) == 8
    assert any("serve.probe" in command for _, command, _ in fake.calls)


# -- serving another checkpoint or vLLM build ----------------------------------------------------------

OTHER = "example/Other-Checkpoint@" + "b" * 40
OVERLAY = "/srv/overlays/csf"


def edits(set_values=(), drop_values=(), fields=()):
    return plan_mod.vllm_edits(set_values, drop_values, fields)


def test_vllm_argument_edits_reach_every_rank_and_the_plan_records_them(repository, site_file, capsys):
    default = make_plan(repository)
    plan = make_plan(repository, vllm_edits=edits(["--gpu-memory-utilization=0.85", "--enforce-eager",
                                                    "--quantization=nvfp4_csf"], ["--host"],
                                                   ["moe_backend=marlin", "num_speculative_tokens=2"]))
    for launch, before in zip(plan.ranks, default.ranks):
        command = list(launch.command)
        assert command[command.index("--gpu-memory-utilization") + 1] == "0.85"
        assert "--host" not in command and "0.0.0.0" not in command
        assert json.loads(command[command.index("--speculative-config") + 1]) == {
            "method": "mtp", "num_speculative_tokens": 2, "draft_tensor_parallel_size": 4, "moe_backend": "marlin"}
        assert command[-3:] == ["--enforce-eager", "--quantization", "nvfp4_csf"]
        assert ("--headless" in command) == (launch.rank > 0) and len(command) == len(before.command) + 1
    assert [change.describe() for change in plan.argument_changes] == [
        "--gpu-memory-utilization 0.80 -> --gpu-memory-utilization 0.85  (--vllm-arg)",
        "unset -> --enforce-eager  (--vllm-arg)",
        "unset -> --quantization nvfp4_csf  (--vllm-arg)",
        "--host 0.0.0.0 -> removed  (--drop-vllm-arg)",
        "--speculative-config: moe_backend unset -> \"marlin\"; num_speculative_tokens 3 -> 2  (--speculative-set)"]
    assert ("  vLLM arguments edited on every rank:\n    --gpu-memory-utilization 0.80 -> --gpu-memory-utilization "
            "0.85  (--vllm-arg)") in plan_mod.render_text(plan)
    record = json.loads(json.dumps(plan.to_json()))["vllm_arguments"]
    assert record["changes"][3] == {"flag": "--host", "before": ["0.0.0.0"], "after": None, "option": "--drop-vllm-arg"}
    assert record["recipe"] == list(plan.recipe_arguments) and "--host" not in record["recipe"]
    assert cli.memory_utilization(plan) == 0.85 and cli.memory_utilization(default) == 0.80
    assert json.loads(json.dumps(default.to_json()))["vllm_arguments"] == {
        "changes": [], "recipe": list(default.profile.recipe_arguments)}
    assert "vLLM arguments edited" not in plan_mod.render_text(default)
    # The command line: an option value that is itself a vLLM option.
    assert plan_mod.join_dash_values(["plan", "--vllm-arg", "--a=1", "--drop-vllm-arg", "--b", "--vllm-arg=--c",
                                      "--vllm-arg"]) == [
        "plan", "--vllm-arg=--a=1", "--drop-vllm-arg=--b", "--vllm-arg=--c", "--vllm-arg"]
    base = ["--site", str(site_file), "--repository", str(repository), "--profile", PROFILE, "--model-path", MODEL]
    assert cli.main(["plan", "--json", *base, "--vllm-arg", "--quantization=nvfp4_csf", "--vllm-arg",
                     "--load-format=nvfp4_csf", "--drop-vllm-arg", "--host", "--speculative-set",
                     "moe_backend=marlin"]) == 0
    record = json.loads(capsys.readouterr().out)
    argv = record["ranks"][2]["argv"]
    assert argv[argv.index("--quantization") + 1] == "nvfp4_csf" and "--host" not in argv
    assert '"moe_backend":"marlin"' in argv[argv.index("--speculative-config") + 1]
    assert [change["flag"] for change in record["vllm_arguments"]["changes"]] == [
        "--quantization", "--load-format", "--host", "--speculative-config"]
    assert cli.main(["plan", *base, "--vllm-arg", "--port=9000"]) == 2
    assert "--vllm-arg --port: the launcher owns this argument" in capsys.readouterr().err
    for name in cli.COMMANDS:
        args = cli.parser().parse_args(plan_mod.join_dash_values(
            [name, "--site", "s", "--repository", "r", "--vllm-arg", "--enforce-eager", "--drop-vllm-arg", "--host",
             "--speculative-set", "moe_backend=marlin"]))
        assert (args.vllm_arg, args.drop_vllm_arg, args.speculative_set) == (
            ["--enforce-eager"], ["--host"], ["moe_backend=marlin"]), name


@pytest.mark.parametrize("arguments, message", [
    ((["--port=9000"],), "--vllm-arg --port: the launcher owns this argument (the launcher sets each instance's API"),
    ((["--tensor-parallel-size=8"],), "the group's size"),
    ((["--pipeline-parallel-size=2"],), "tensor parallelism only"),
    ((["--data-parallel-size-local=2"],), "tensor parallelism only"),
    ((["--served-model-name=other"],), "served model name"),
    (([], ["--headless"]), "--drop-vllm-arg --headless: the launcher owns this argument"),
    ((["--default-chat-template-kwargs={}"],), "use --reasoning-effort"),
    ((["-q=nvfp4_csf"],), "give a long vLLM option"),
    ((["--quantization="],), "give a value after '='"),
    ((["--seed=1", "--seed=2"],), "--vllm-arg names --seed more than once"),
    ((["--seed=1"], ["--seed"]), "--seed: given to both --vllm-arg and --drop-vllm-arg"),
    (([], ["--speculative-config"], ["moe_backend=marlin"]), "which --drop-vllm-arg removes"),
    (([], [], ["moe backend=marlin"]), "is not KEY=VALUE"),
    (([], [], ["moe_backend"]), "is not KEY=VALUE"),
], ids=["port", "tp", "pp", "dp", "name", "headless", "template", "short", "empty", "twice", "both", "dropped",
        "key", "no-value"])
def test_vllm_argument_edits_refuse_owned_arguments_and_contradictions(arguments, message):
    with pytest.raises(ServePlanError, match=re.escape(message)):
        plan_mod.vllm_edits(*arguments)


def test_vllm_argument_edits_follow_how_the_command_gives_each_argument(repository):
    edit = plan_mod.edit_arguments
    assert edit(["--a=1", "--b"], edits(["--a=2"]))[0] == ["--a=2", "--b"]
    assert edit(["--a", "1", "--b"], edits(["--a=-5"]))[0] == ["--a=-5", "--b"]   # a value with a dash goes inline
    assert edit(["--b"], edits(["--seed=-1"]))[0] == ["--b", "--seed=-1"]
    assert edit(["--a=1", "--b", "--c", "2"], edits([], ["--a", "--b", "--c"]))[0] == []
    unchanged = edit(["--b"], edits(["--b"]))[1]
    assert unchanged == [plan_mod.ArgumentChange("--b", (), (), "--vllm-arg")]
    assert unchanged[0].describe() == "--b: as the profile gives it  (--vllm-arg)"
    assert edit(['--speculative-config={"n":3}'], edits([], [], ["n=2"]))[0] == ['--speculative-config={"n":2}']
    assert plan_mod.vllm_edits(["--load_format=b12x"]).set == (("--load-format", "b12x"),)
    assert plan_mod.vllm_edits([], [], ["n=2", "m=marlin", "f=true", 's="x"']).speculative == (
        ("n", 2), ("m", "marlin"), ("f", True), ("s", "x"))
    for arguments, change, message in (
            (["--a", "1"], edits(["--a"]), "gives it the value '1'; give --a=VALUE"),
            (["--a", "--b"], edits(["--a=1"]), "gives --a as a switch without a value"),
            (["--a", "1", "--a", "2"], edits(["--a=3"]), "gives it 2 times"),
            (["--b"], edits([], ["--a"]), "does not give it"),
            (["--b"], edits([], [], ["n=1"]), "needs exactly one --speculative-config"),
            (["--speculative-config", "[1]"], edits([], [], ["n=1"]), "is not a JSON object")):
        with pytest.raises(ServePlanError, match=re.escape(message)):
            edit(arguments, change)
    # The recipe checks run on the edited arguments.
    with pytest.raises(ServePlanError, match="micro-batching"):
        make_plan(repository, vllm_edits=edits(["--enable-dbo"]))
    with pytest.raises(ServePlanError, match="pass_config.fuse_allreduce_rms"):
        make_plan(repository, vllm_edits=edits(['--compilation-config={"pass_config":{"fuse_allreduce_rms":true}}']))
    assert make_plan(repository, vllm_edits=edits(["--max-num-batched-tokens=4096"])).max_num_batched_tokens == 4096


def test_env_accepts_vllm_and_b12x_runtime_switches_and_the_plan_shows_what_it_replaces(repository):
    switches = ["VLLM_B12X_MOE_FP4_FORCE_A16=1", "B12X_W4A16_FP32_TOPK_WEIGHTS=1",
                "B12X_W4A16_A4_PREFILL_MIN_TOKENS=4096", "VLLM_GLM53_MTP_DRAFT_HEAD=1", "B12X_AUTOTUNE=0"]
    plan = make_plan(repository, extra_env=plan_mod.parse_env(switches))
    for launch in plan.ranks:
        assert all(launch.environment[key] == value for key, _, value in (item.partition("=") for item in switches))
    changes = {change.name: change for change in plan.changes}
    assert (changes["VLLM_B12X_MOE_FP4_FORCE_A16"].before, changes["VLLM_B12X_MOE_FP4_FORCE_A16"].after) == ("0", "1")
    assert all(changes[key].reason == "--env" for key in ("B12X_AUTOTUNE", "VLLM_GLM53_MTP_DRAFT_HEAD"))
    assert changes["B12X_AUTOTUNE"].before is None
    assert "VLLM_B12X_MOE_FP4_FORCE_A16: '0' -> '1'  (--env)" in plan_mod.render_text(plan)
    with pytest.raises(ServePlanError, match="use --b12x-cache-dir"):
        plan_mod.parse_env(["B12X_COMPILE_CACHE_DIR=/cache/other"])


def overlay_answer(tree="t" * 64, missing=(), metadata=""):
    """What ``commands.overlay_facts`` prints for an overlay whose OVERLAY.json states ``tree`` (None: no tree)."""
    record = json.dumps({"schema": "example-overlay/v1", **({"tree_sha256": tree} if tree else {})}).encode()
    lines = ["overlay\tpresent"]
    lines += [f"file:{name}\t{'missing' if name in missing else 'present'}" for name in plan_mod.OVERLAY_FILES]
    lines += [f"record_sha256\t{sha(record)}", f"record\t{base64.b64encode(record).decode()}", f"metadata\t{metadata}"]
    return remote.Result(0, "\n".join(lines) + "\n", "")


class OverlaySparks(FakeSparks):
    """Healthy Sparks whose overlay directories hold the tree ``trees[position]`` (None: no tree in OVERLAY.json)."""

    def __init__(self, plans, *, trees=None, missing=None, metadata="", **kwargs):
        super().__init__(plans, **kwargs)
        self.trees, self.missing, self.metadata = trees or {}, missing or {}, metadata

    def __call__(self, target, command, *, timeout=60, input_bytes=None):
        if "OVERLAY.json" in command and "record_sha256" in command:
            self.calls.append((target, command, input_bytes))
            position = next(launch.position for launch in self.launches() if launch.ssh == target)
            return overlay_answer(self.trees.get(position, "t" * 64), self.missing.get(position, ()), self.metadata)
        return super().__call__(target, command, timeout=timeout, input_bytes=input_bytes)


def test_an_overlay_is_mounted_read_only_first_on_pythonpath_and_preflight_compares_its_trees(repository):
    plan = make_plan(repository, overlay=OVERLAY)
    for launch in plan.ranks:
        mount = launch.mount(plan_mod.OVERLAY_TARGET)
        assert (mount.source, mount.read_only) == (OVERLAY, True)
        assert launch.environment["PYTHONPATH"] == "/opt/sparkring-overlay:/sircl/src"
        assert f"--mount type=bind,src={OVERLAY},dst=/opt/sparkring-overlay,readonly" in launch.shell()
        stage = commands.stage_container(plan, launch)
        assert f"src={OVERLAY},dst=/opt/sparkring-overlay,readonly" in stage
        assert "PYTHONPATH=/opt/sparkring-overlay:/sircl/src" in stage
    change = next(change for change in plan.changes if change.name == "PYTHONPATH")
    assert change.after == "/opt/sparkring-overlay:/sircl/src" and change.reason == plan_mod.OVERLAY_PYTHONPATH_REASON
    assert json.loads(json.dumps(plan.to_json()))["overlay"] == {
        "target": "/opt/sparkring-overlay", "sources": {str(position): OVERLAY for position in range(4)},
        "pythonpath": "/opt/sparkring-overlay:/sircl/src", "required_files": list(plan_mod.OVERLAY_FILES)}
    text = plan_mod.render_text(plan)
    assert f"source overlay: {OVERLAY} (read-only) at /opt/sparkring-overlay on every rank, first on PYTHONPATH" in text
    assert "the mhc_prefill_shard shim this group needs installs only where the overlay's vLLM files match" in text
    assert "with the overlay, its b12x reuses kernels the image's b12x compiled there" in text
    plain = make_plan(repository)
    assert plain.to_json()["overlay"] is None and "source overlay" not in plan_mod.render_text(plain)
    assert "PYTHONPATH=/sircl/src " in commands.stage_container(plain, plain.ranks[0])
    # One directory for every Spark, with per-Spark exceptions; every rank or none.
    mixed = make_plan(repository, overlay=OVERLAY, overlays={3: "/srv/other/csf"})
    assert [launch.mount(plan_mod.OVERLAY_TARGET).source for launch in mixed.ranks] == (
        [OVERLAY] * 3 + ["/srv/other/csf"])
    for options, message in (({"overlays": {0: OVERLAY}}, "names no directory for Sparks [1, 2, 3]"),
                             ({"overlay": "relative/dir"}, "must be an absolute path")):
        with pytest.raises(ServePlanError, match=re.escape(message)):
            make_plan(repository, **options)
    # Preflight: every Spark's directory holds the required files, and one tree everywhere.
    lines: list[str] = []
    assert cli.preflight(context(plan, OverlaySparks(plan), lines), fabric=False) == 0
    assert sum(f"overlay {OVERLAY} holds vllm/__init__.py, b12x/__init__.py, OVERLAY.json; tree {'t' * 64} "
               "(OVERLAY.json tree_sha256)" in line for line in lines) == 4
    lines.clear()
    fake = OverlaySparks(plan, trees={3: "u" * 64}, missing={1: ("b12x/__init__.py",)},
                         metadata="vllm-0.0.1.dist-info ")
    assert cli.preflight(context(plan, fake, lines), fabric=False) == 1
    blockers = [line for line in lines if line.startswith("BLOCKER: ")]
    assert any(f"rank 1 (spark1): overlay directory {OVERLAY} lacks b12x/__init__.py" in line for line in blockers)
    assert any("the overlay trees differ between Sparks" in line and "u" * 64 in line for line in blockers)
    assert any("carries distribution metadata vllm-0.0.1.dist-info" in line for line in lines)
    facts = [command for _, command, _ in fake.calls if "OVERLAY.json" in command]
    assert len(facts) == 4 and facts[0].startswith("sudo -n sh -c ") and not facts[2].startswith("sudo")
    # Start refuses on the same blockers and starts nothing.
    lines.clear()
    fake = OverlaySparks(plan, trees={3: "u" * 64})
    assert cli.start(context(plan, fake, lines)) == 1 and lines[-1] == "nothing was started"
    assert not any(" run -d " in command for _, command, _ in fake.calls)
    # Without a tree in OVERLAY.json the file's own digest identifies the tree.
    lines.clear()
    assert cli.preflight(context(plan, OverlaySparks(plan, trees=dict.fromkeys(range(4))), lines), fabric=False) == 0
    assert sum("(SHA-256 of OVERLAY.json (it states no tree hash))" in line for line in lines) == 4


def test_the_overlay_and_checkpoint_commands_are_valid_shell(repository, tmp_path):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("no bash on this machine")
    plan = make_plan(repository, overlay=OVERLAY, checkpoint_id=OTHER)
    for command in (commands.overlay_facts(OVERLAY, "sudo -n"), commands.overlay_facts(OVERLAY),
                    commands.host_facts(plan, plan.ranks[0], plan.profile),
                    commands.stage_container(plan, plan.ranks[2])):
        checked = subprocess.run([bash, "-n", "-c", command], capture_output=True, text=True)
        assert checked.returncode == 0, (command[:200], checked.stderr)
        inner = shlex.split(command)
        if inner[:2] == ["sudo", "-n"]:
            checked = subprocess.run([bash, "-n", "-c", inner[4]], capture_output=True, text=True)
            assert checked.returncode == 0, (inner[4][:200], checked.stderr)
    if os.name != "posix" or not all(shutil.which(tool) for tool in ("base64", "sha256sum")):
        return
    # The facts of a real directory, read back as preflight reads them.
    for name in ("vllm", "b12x", "vllm-0.0.1.dist-info"):
        (tmp_path / name).mkdir()
    for name in ("vllm/__init__.py", "b12x/__init__.py"):
        (tmp_path / name).write_text("")
    (tmp_path / "OVERLAY.json").write_text(json.dumps({"tree_sha256": "f" * 64}))
    output = subprocess.run([bash, "-c", commands.overlay_facts(str(tmp_path))], capture_output=True, text=True)
    lines, blockers = checks.overlay_findings([("here", str(tmp_path), cli.facts(output.stdout))])
    assert blockers == [] and f"tree {'f' * 64} (OVERLAY.json tree_sha256)" in lines[0]
    assert "vllm-0.0.1.dist-info" in lines[1]


class StageSparks(FakeSparks):
    """Answers the stage step: the package tree, the seccomp policy and the stage container's probe record."""

    def __init__(self, plans, record, **kwargs):
        super().__init__(plans, **kwargs)
        self.record = record

    def __call__(self, target, command, *, timeout=60, input_bytes=None):
        if "serve.probe" in command:
            self.calls.append((target, command, input_bytes))
            return remote.Result(0, probe.PREFIX + json.dumps(self.record) + "\n", "")
        if "tar -C" in command and ".sircl-staged" in command:
            self.calls.append((target, command, input_bytes))
            return remote.Result(0, "staged\n", "")
        if "loader-seccomp" in command and ".tmp." in command:
            self.calls.append((target, command, input_bytes))
            return remote.Result(0, "written\n", "")
        return super().__call__(target, command, timeout=timeout, input_bytes=input_bytes)


def test_the_stage_probe_requires_the_overlays_packages_and_the_shims_the_launch_needs(repository, tmp_path):
    library = "roce_proxy-" + "1" * 16 + ".so"
    record = good_record(library)
    record["modules"] = {name: f"/opt/sparkring-overlay/{name}/__init__.py" for name in ("vllm", "b12x")}
    record["vllm"] = {"root": "/opt/sparkring-overlay/vllm", "matches": [], "describe": "no pinned build"}
    record["shims"] = {"mhc_prefill_shard": None, "worker_regimes": "lil-image-aba309e4610c"}
    blockers, notes = probe.evaluate(record, staged_root="/sircl/src", library=library,
                                     overlay="/opt/sparkring-overlay")
    assert blockers == [] and "shims that match: worker_regimes (lil-image-aba309e4610c)" in notes[0]
    blockers, _ = probe.evaluate(record, staged_root="/sircl/src", library=library, overlay="/opt/sparkring-overlay",
                                 required_shims=("mhc_prefill_shard",))
    assert len(blockers) == 1 and "needs the mhc_prefill_shard shim" in blockers[0]
    assert "--mhc-prefill-shard off serves without it" in blockers[0]
    # A launch with mHC on and DCP above 1 needs the build its mHC files match to admit its sizes.
    record["shims"]["mhc_prefill_shard"] = "sparkring-kraken-beta-20261007-bc9ea774"
    for sizes, refused in (((4, 2), False), ((2, 2), True), (None, False)):
        blockers, _ = probe.evaluate(record, staged_root="/sircl/src", library=library,
                                     overlay="/opt/sparkring-overlay", required_shims=("mhc_prefill_shard",),
                                     mhc_sizes=sizes)
        assert bool(blockers) == refused, sizes
    assert blockers == [] and "refuses TP2 with DCP 2 at startup" in probe.evaluate(
        record, staged_root="/sircl/src", library=library, overlay="/opt/sparkring-overlay",
        mhc_sizes=(2, 2))[0][0]
    record["shims"]["mhc_prefill_shard"] = None
    record["modules"]["b12x"] = "/usr/local/lib/python3.12/dist-packages/b12x/__init__.py"
    blockers, _ = probe.evaluate(record, staged_root="/sircl/src", library=library, overlay="/opt/sparkring-overlay")
    assert blockers == ["import b12x resolves to /usr/local/lib/python3.12/dist-packages/b12x/__init__.py, not the "
                        "source overlay /opt/sparkring-overlay"]
    # Each package resolves to the first directory of the path that holds it.
    for directory, names in (("first", ("vllm",)), ("second", ("vllm", "b12x"))):
        for name in names:
            (tmp_path / directory / name).mkdir(parents=True)
            (tmp_path / directory / name / "__init__.py").write_text("")
    origins = probe.module_origins([str(tmp_path / "first"), str(tmp_path / "second")])
    assert Path(origins["vllm"]).parent.parent.name == "first" and Path(origins["b12x"]).parent.parent.name == "second"
    assert probe.module_origins([str(tmp_path / "first")], ("sircl_no_such_package",)) == {
        "sircl_no_such_package": None}
    # The shims a plan needs: prefill row ownership on a group without NCCL.
    assert make_plan(repository).required_shims == ("mhc_prefill_shard",)
    assert make_plan(repository, mhc_prefill_shard="off").required_shims == ()
    assert make_plans(repository, ((0, 1),))[0].required_shims == ()          # NCCL runs on a pair


def test_stage_probes_the_overlay_and_refuses_a_launch_its_vllm_cannot_carry(repository):
    plan = make_plan(repository, overlay=OVERLAY)
    record = good_record(plan.library)
    record["modules"] = {name: f"/opt/sparkring-overlay/{name}/__init__.py" for name in ("vllm", "b12x")}
    record["vllm"] = {"root": "/opt/sparkring-overlay/vllm", "matches": [], "describe": "no pinned build"}
    record["shims"] = {"mhc_prefill_shard": None, "worker_regimes": None}
    lines: list[str] = []
    fake = StageSparks(plan, record)
    assert cli.stage(context(plan, fake, lines)) == 1
    assert sum("needs the mhc_prefill_shard shim" in line for line in lines if line.startswith("BLOCKER: ")) == 4
    assert any("vllm from /opt/sparkring-overlay/vllm/__init__.py, b12x from /opt/sparkring-overlay/b12x/__init__.py"
               in line for line in lines)
    builds = [command for _, command, _ in fake.calls if "serve.probe" in command]
    assert len(builds) == 4 and all("dst=/opt/sparkring-overlay,readonly" in command for command in builds)
    lines.clear()
    off = make_plan(repository, overlay=OVERLAY, mhc_prefill_shard="off")
    assert cli.stage(context(off, StageSparks(off, record), lines)) == 0 and lines[-1] == "stage complete"


def test_stage_refuses_mhc_at_dcp_sizes_the_matched_build_does_not_admit(repository, monkeypatch):
    """With mHC on and DCP above 1 the plan needs some pinned build to admit the sizes, and stage needs the
    build each Spark's mHC files match to admit them."""
    monkeypatch.setitem(pins.MHC_ADMITS, "lil-image-aba309e4610c", pins.MHC_ADMITS_DEFAULT + ((2, 2),))
    b12x = plan_mod.vllm_edits(["--attention-backend=B12X"])
    plan = make_plans(repository, ((0, 1),), dcp_size=2, nccl_mode="never", vllm_edits=b12x, overlay=OVERLAY)[0]
    assert plan.mhc_prefill_shard and "mhc_prefill_shard" in plan.required_shims
    record = good_record(plan.library)
    record["modules"] = {name: f"/opt/sparkring-overlay/{name}/__init__.py" for name in ("vllm", "b12x")}
    record["vllm"] = {"root": "/opt/sparkring-overlay/vllm", "matches": ["sparkring-kraken-beta-20261007-bc9ea774"]}
    record["shims"] = {name: "sparkring-kraken-beta-20261007-bc9ea774" for name in plan.required_shims}
    lines: list[str] = []
    assert cli.stage(context(plan, StageSparks(plan, record), lines)) == 1
    assert sum("refuses TP2 with DCP 2 at startup" in line for line in lines if line.startswith("BLOCKER: ")) == 2
    record["shims"]["mhc_prefill_shard"] = "lil-image-aba309e4610c"
    lines.clear()
    assert cli.stage(context(plan, StageSparks(plan, record), lines)) == 0 and lines[-1] == "stage complete"


def test_check_with_an_overlay_needs_every_ranks_vllm_from_the_overlay(repository):
    plan = make_plan(repository, overlay=OVERLAY)

    def receipts(rank):
        return [FakeSparks(plan).receipt(rank, vllm="/opt/sparkring-overlay/vllm", b12x="/opt/sparkring-overlay/b12x")]

    lines: list[str] = []
    assert cli.check(context(plan, FakeSparks(plan, receipts=receipts), lines), long_prompt=0) == 0
    assert ("imports of rank(s) 0,1,2,3: vllm from /opt/sparkring-overlay/vllm, b12x from "
            "/opt/sparkring-overlay/b12x") in lines
    lines.clear()
    assert cli.check(context(plan, FakeSparks(plan), lines), long_prompt=0) == 1    # receipts without the fields
    assert any("rank 0: vllm was imported from an unstated directory, not the source overlay" in line for line in lines)
    plain = make_plan(repository)
    assert cli.check(context(plain, FakeSparks(plain), []), long_prompt=0) == 0
    found, problems = checks.import_findings({0: [{"group": "tp:0", "vllm": "/opt/sparkring-overlay/vllm"}],
                                              1: [{"group": "tp:0", "vllm": "/opt/sparkring-overlay/vllm",
                                                   "b12x": "/usr/lib/b12x"}]}, "/opt/sparkring-overlay")
    assert problems == ["rank 1: b12x was imported from /usr/lib/b12x, not the source overlay /opt/sparkring-overlay"]
    assert found[0] == ("imports of rank(s) 0: vllm from /opt/sparkring-overlay/vllm, b12x from not loaded when the "
                        "receipt was written")


def test_a_checkpoint_id_names_the_served_checkpoint_and_its_thinking_behaviour(repository, site_file, capsys):
    plan = make_plan(repository, checkpoint_id=OTHER)
    assert plan.checkpoint == OTHER and plan.foreign_checkpoint and plan.thinking is None and plan.prefill() is None
    assert json.loads(json.dumps(plan.to_json()))["checkpoint"] == {
        "id": OTHER, "source": "--checkpoint-id", "profile_checkpoint": f"{REPOSITORY}@{REVISION}", "manifest": False}
    text = plan_mod.render_text(plan)
    assert f"checkpoint {OTHER} (--checkpoint-id), not the profile's {REPOSITORY}@{REVISION}" in text
    assert f"thinking: profiles/thinking.json records no behaviour for {OTHER}; --thinking-behaviour names one" in text
    assert f"expected prefill: not estimated for {OTHER}" in text
    with pytest.raises(ServePlanError, match=re.escape(f"records no thinking behaviour for the checkpoint {OTHER}; "
                                                       "--thinking-behaviour names one")):
        make_plan(repository, checkpoint_id=OTHER, reasoning_effort="high")
    named = make_plan(repository, checkpoint_id=OTHER, thinking_behaviour="glm53-flash-template",
                      reasoning_effort="high")
    assert named.ranks[0].command[-2:] == ("--default-chat-template-kwargs", '{"reasoning_effort":"high"}')
    assert named.to_json()["serving"]["thinking_source"] == "--thinking-behaviour"
    assert ("thinking (glm53-flash-template by --thinking-behaviour): requests that name no reasoning_effort run at "
            "high") in plan_mod.render_text(named)
    own = make_plan(repository, checkpoint_id=f"{REPOSITORY}@{REVISION}")
    assert not own.foreign_checkpoint and own.to_json()["checkpoint"]["source"] == "profile"
    assert own.prefill() is not None
    for options, message in (
            ({"checkpoint_id": OTHER, "thinking_behaviour": "unknown-template"},
             "defines no behaviour 'unknown-template' with a list of levels; it defines glm53-flash-template, "
             "mimo-v26-flash-template"),
            ({"thinking_behaviour": "mimo-v26-flash-template"},
             "--thinking-behaviour mimo-v26-flash-template: profiles/thinking.json records glm53-flash-template for "
             f"the checkpoint {REPOSITORY}@{REVISION}"),
            ({"checkpoint_id": "no-revision"}, "--checkpoint-id 'no-revision' is not REPOSITORY@REVISION")):
        with pytest.raises(ServePlanError, match=re.escape(message)):
            make_plan(repository, **options)
    # A checkpoint the record lists takes its behaviour from the record.
    path = repository / profile_mod.THINKING_RELATIVE
    path.write_text(json.dumps({**THINKING, "checkpoints": {**THINKING["checkpoints"], OTHER: "glm53-flash-template"}}))
    listed = make_plan(repository, checkpoint_id=OTHER, reasoning_effort="low")
    assert listed.thinking.id == "glm53-flash-template" and listed.thinking_source == "profiles/thinking.json"
    base = ["--site", str(site_file), "--repository", str(repository), "--profile", PROFILE, "--model-path", MODEL]
    assert cli.main(["plan", "--json", *base, "--checkpoint-id", OTHER, "--thinking-behaviour", "glm53-flash-template",
                     "--reasoning-effort", "max"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["checkpoint"]["id"] == OTHER
    assert record["serving"]["default_chat_template_kwargs"] == {"reasoning_effort": "max"}


def test_preflight_compares_the_copies_of_a_checkpoint_the_profile_does_not_pin(repository):
    plan = make_plan(repository, checkpoint_id=OTHER)
    fake = FakeSparks(plan)

    def facts_for(target, command, **kwargs):
        if "/proc/meminfo" in command:
            fake.calls.append((target, command, None))
            index = "e" * 64 if target.endswith(".23") else "d" * 64
            config = "" if target.endswith(".22") else "c" * 64
            return remote.Result(0, "mem:MemTotal\t125000000\nmem:MemAvailable\t120000000\ncurl\t/usr/bin/curl\n"
                                    f"model\tpresent\nfiles\t36 123456789\nsha256:config.json\t{config}\n"
                                    f"sha256:model.safetensors.index.json\t{index}\ncache\tpresent\n"
                                    "space_kib\t3000000000\n", "")
        return fake(target, command, **kwargs)

    lines: list[str] = []
    assert cli.preflight(context(plan, facts_for, lines), fabric=False) == 1
    blockers = [line for line in lines if line.startswith("BLOCKER: ")]
    assert len(blockers) == 2
    assert (f"rank 2 (spark2): {MODEL} (--model-path) has no readable config.json or model.safetensors.index.json"
            in blockers[0])
    assert f"the copies of checkpoint {OTHER} differ between Sparks" in blockers[1] and "e" * 64 in blockers[1]
    assert any(f"for checkpoint {OTHER}, which the profile's manifest does not describe: config.json {'c' * 64}, "
               f"model.safetensors.index.json {'d' * 64}, 36 files, 123,456,789 bytes" in line for line in lines)
    command = next(command for _, command, _ in fake.calls if "/proc/meminfo" in command)
    assert "find -L" in command and "size:" not in command


def test_the_b12x_cache_directory_option_places_b12x_compile_cache(repository):
    default = make_plan(repository)
    assert default.b12x_cache == "/cache/example/b12x"
    assert ("B12X compile cache: B12X_COMPILE_CACHE_DIR=/cache/example/b12x (the profile's), in the /cache mount, "
            f"on each Spark at /tmp/sircl-ring/serve/cache/{PROFILE}/example/b12x") in plan_mod.render_text(default)
    assert default.to_json()["b12x_cache"] == {"variable": "B12X_COMPILE_CACHE_DIR", "value": "/cache/example/b12x",
                                               "source": "profile", "mount": "/cache"}
    fresh = make_plan(repository, b12x_cache_dir="/cache/b12x-overlay", overlay=OVERLAY)
    assert all(launch.environment["B12X_COMPILE_CACHE_DIR"] == "/cache/b12x-overlay" for launch in fresh.ranks)
    change = next(change for change in fresh.changes if change.name == "B12X_COMPILE_CACHE_DIR")
    assert (change.before, change.after) == ("/cache/example/b12x", "/cache/b12x-overlay")
    text = plan_mod.render_text(fresh)
    assert ("B12X compile cache: B12X_COMPILE_CACHE_DIR=/cache/b12x-overlay (--b12x-cache-dir), in the /cache mount, "
            f"on each Spark at /tmp/sircl-ring/serve/cache/{PROFILE}/b12x-overlay") in text
    assert "reuses kernels" not in text and fresh.to_json()["b12x_cache"]["source"] == "--b12x-cache-dir"
    run = make_plan(repository, b12x_cache_dir="/sircl/run/b12x")
    assert ("in the run directory, on each Spark at /tmp/sircl-ring/serve/runs/tp4-0-3/b12x (new for every run id)"
            in plan_mod.render_text(run))
    own = make_plan(repository, b12x_cache_dir="/var/cache/b12x")
    assert "the container's own file system, discarded with the container" in plan_mod.render_text(own)
    assert own.to_json()["b12x_cache"]["mount"] is None
    for path, message in (("/opt/sparkring-overlay/b12x", "lies in a read-only mount"),
                          ("/models/target/b12x", "lies in a read-only mount"),
                          ("/", "must name a directory below /"), ("cache/b12x", "must be an absolute path")):
        with pytest.raises(ServePlanError, match=re.escape(message)):
            make_plan(repository, b12x_cache_dir=path)
    for name in cli.COMMANDS:
        args = cli.parser().parse_args([name, "--site", "s", "--repository", "r", "--b12x-cache-dir", "/cache/x",
                                        "--overlay", "/o", "--checkpoint-id", "a/b@c", "--thinking-behaviour", "t"])
        assert (args.b12x_cache_dir, args.overlay, args.checkpoint_id, args.thinking_behaviour) == (
            "/cache/x", ["/o"], "a/b@c", "t"), name


@needs_checkout
def test_the_repositorys_glm53_flash_profile_serves_another_checkpoint_through_an_overlay():
    root = CHECKOUT
    profile = profile_mod.load(root)
    csf = "local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD@dec48abd33efa73c3bb7c95b74eee10cad34f9be"
    options = Options((0, 1, 2, 3), overlay="/srv/sparkring/overlays/csf-test", checkpoint_id=csf,
                      vllm_edits=edits(["--quantization=nvfp4_csf", "--load-format=nvfp4_csf"], (),
                                       ["moe_backend=marlin"]),
                      thinking_behaviour="glm53-flash-template", reasoning_effort="high",
                      extra_env=plan_mod.parse_env(["VLLM_B12X_MOE_FP4_FORCE_A16=1"]), b12x_cache_dir="/cache/b12x-csf")
    plan = plan_mod.build_plan(ServeSite.from_json(SITE_WITH_PATHS), profile, options,
                               staged_digest=staging.staged_tree().digest, library=staging.library_name())
    command = plan.ranks[0].command
    assert command[command.index("--quantization") + 1] == "nvfp4_csf"
    assert command[command.index("--load-format") + 1] == "nvfp4_csf"
    assert json.loads(command[command.index("--speculative-config") + 1])["moe_backend"] == "marlin"
    assert command[-2:] == ("--default-chat-template-kwargs", '{"reasoning_effort":"high"}')
    changes = {change.name: change for change in plan.changes}
    assert (changes["VLLM_B12X_MOE_FP4_FORCE_A16"].before, changes["VLLM_B12X_MOE_FP4_FORCE_A16"].after) == ("0", "1")
    assert changes["B12X_COMPILE_CACHE_DIR"].before == "/cache/glm53-flash-nvfp4-spark-cuda13.4.2-a608241037e4/b12x"
    assert plan.required_shims == ("mhc_prefill_shard",) and plan.foreign_checkpoint
    assert (plan.thinking.id, plan.thinking_source) == ("glm53-flash-template", "--thinking-behaviour")


def test_the_bundle_carries_an_overlay_vllm_edits_the_b12x_cache_and_a_named_thinking_behaviour(repository, site_file,
                                                                                                capsys):
    bundled = make_bundle(overlay=OVERLAY, vllm_edits=edits(["--quantization=nvfp4_csf"], ["--host"],
                                                            ["moe_backend=marlin"]),
                          b12x_cache_dir="/cache/b12x-overlay", checkpoint=OTHER,
                          thinking_behaviour="glm53-flash-template", repository=str(repository),
                          reasoning_effort="high")
    document = json.loads(json.dumps(bundled.to_json()))
    for entry in document["ranks"]:
        assert entry["mounts"][0]["option"] == f"type=bind,src={OVERLAY},dst=/opt/sparkring-overlay,readonly"
        assert entry["pythonpath_prepend"] == "/opt/sparkring-overlay:/sircl/src"
        assert entry["environment"]["B12X_COMPILE_CACHE_DIR"] == "/cache/b12x-overlay"
    assert document["merge"]["PYTHONPATH"].startswith("prepend /opt/sparkring-overlay:/sircl/src")
    assert document["vllm_edits"] == {"set": [["--quantization", "nvfp4_csf"]], "remove": ["--host"],
                                      "speculative_config": {"moe_backend": "marlin"}}
    assert document["merge"]["vllm_arguments"].startswith("every rank: replace or add each argument of vllm_edits.set")
    assert document["overlay"] == {"target": "/opt/sparkring-overlay",
                                   "sources": {str(position): OVERLAY for position in range(8)},
                                   "required_files": list(plan_mod.OVERLAY_FILES)}
    assert document["serving"] == {"reasoning_effort": "high",
                                   "default_chat_template_kwargs": {"reasoning_effort": "high"},
                                   "checkpoint": OTHER, "thinking_behaviour": "glm53-flash-template",
                                   "template_level": "max", "thinking_source": "--thinking-behaviour"}
    assert document["ranks"][0]["vllm_arguments"] == ["--default-chat-template-kwargs", '{"reasoning_effort":"high"}']
    plain = make_bundle().to_json()
    assert plain["vllm_edits"] is None and plain["overlay"] is None and "vllm_arguments" not in plain["merge"]
    named = make_bundle(checkpoint=OTHER)
    assert named.to_json()["serving"]["checkpoint"] == OTHER and named.thinking is None
    for options, message in (({"thinking_behaviour": "glm53-flash-template"},
                              "--thinking-behaviour needs --repository"),
                             ({"repository": str(repository)}, "give it with one of them"),
                             ({"overlays": {0: OVERLAY}}, "names no directory for Sparks [1, 2, 3, 4, 5, 6, 7]"),
                             ({"b12x_cache_dir": "/sircl/src/b12x"}, "lies in a read-only mount")):
        with pytest.raises(ServePlanError, match=re.escape(message)):
            make_bundle(**options)
    assert cli.main(["bundle", "--site", str(site_file), "--positions", "0-7", "--overlay", OVERLAY, "--vllm-arg",
                     "--load-format=nvfp4_csf", "--speculative-set", "moe_backend=marlin", "--checkpoint-id", OTHER,
                     "--b12x-cache-dir", "/cache/b12x-overlay"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["vllm_edits"]["set"] == [["--load-format", "nvfp4_csf"]]
    assert document["serving"]["checkpoint"] == OTHER
    assert document["ranks"][3]["pythonpath_prepend"] == "/opt/sparkring-overlay:/sircl/src"
    args = cli.parser().parse_args(["bundle-check", "--site", "s", "--positions", "0-7", "--container", "x-{rank}",
                                    "--overlay", OVERLAY])
    assert args.overlay == [OVERLAY]


class OverlayBundleSparks(BundleSparks):
    """Bundle Sparks whose overlay directories hold the tree ``trees[position]`` and whose probe sees the overlay."""

    def __init__(self, plan, *, trees=None, **kwargs):
        super().__init__(plan, **kwargs)
        self.trees = trees or {}

    def __call__(self, target, command, *, timeout=60, input_bytes=None):
        if "OVERLAY.json" in command and "record_sha256" in command:
            self.calls.append((target, command, input_bytes))
            position = next(launch.position for launch in self.plan.ranks if launch.ssh == target)
            return overlay_answer(self.trees.get(position, "t" * 64))
        if "sparkring_sircl.vllm.serve.probe" in command:
            self.calls.append((target, command, input_bytes))
            record = good_record(self.plan.library)
            record["modules"] = {name: f"/opt/sparkring-overlay/{name}/__init__.py" for name in ("vllm", "b12x")}
            return remote.Result(0, probe.PREFIX + json.dumps(record) + "\n", "")
        return super().__call__(target, command, timeout=timeout, input_bytes=input_bytes)


def test_bundle_stage_checks_every_overlay_and_bundle_check_the_imports():
    plan = make_bundle(overlay=OVERLAY, run_id="glm53-tp8")
    fake = OverlayBundleSparks(plan)
    lines: list[str] = []
    assert bundle.stage(plan, staging.staged_tree(), fake, lines.append) == 0 and lines[-1] == "bundle staged"
    assert sum(f"overlay {OVERLAY} holds" in line for line in lines) == 8
    builds = [command for _, command, _ in fake.calls if "serve.probe" in command]
    assert len(builds) == 8 and all("dst=/opt/sparkring-overlay,readonly" in command for command in builds)
    lines.clear()
    assert bundle.stage(plan, staging.staged_tree(), OverlayBundleSparks(plan, trees={5: "u" * 64}), lines.append) == 1
    assert any(line.startswith("BLOCKER: the overlay trees differ between Sparks") for line in lines)
    names = bundle.container_names(["glm53-r{rank}"], plan)

    def receipts(rank):
        vllm = "/usr/local/lib/python3.12/dist-packages/vllm" if rank == 6 else "/opt/sparkring-overlay/vllm"
        return [{"group": "tp:0", "global_rank": rank, "rank": rank, "world": 8, "nccl": "none", "pynccl": "skipped",
                 "session": "ring", "state": "ready", "vllm": vllm, "b12x": "/opt/sparkring-overlay/b12x",
                 "decisions": [{"collective": "all_reduce", "backend": "sircl", "method": "direct", "calls": 9}]}]

    lines.clear()
    assert bundle.check(plan, names, BundleSparks(plan, receipts=receipts), lines.append) == 1
    assert any("rank 6: vllm was imported from /usr/local/lib/python3.12/dist-packages/vllm, not the source overlay"
               in line for line in lines)
    assert any(line.startswith("imports of rank(s) 0,1,2,3,4,5,7: vllm from /opt/sparkring-overlay/vllm")
               for line in lines)



# -- chain and ring minimums per collective ------------------------------------------------------------


def test_the_chain_minimum_reaches_every_rank_of_start_and_bundle_and_the_plan_records_it(repository, site_file,
                                                                                          capsys):
    lowered = make_plan(repository, chain_min=0, ring_min=1 << 20)
    assert {launch.environment["SIRCL_CHAIN_MIN_BYTES"] for launch in lowered.ranks} == {"0"}
    assert {change.name: change.reason for change in lowered.changes}["SIRCL_CHAIN_MIN_BYTES"].startswith(
        "--chain-min: the smallest collective auto runs as a chain op")
    text = plan_mod.render_text(lowered)
    assert "SIRCL_CHAIN_MIN_BYTES: unset -> '0'" in text
    assert ("; auto schedules run chain ops from 0 B of every collective (--chain-min) where the ranks form a chain "
            "of cable neighbors; ring schedules run ring ops from 1,048,576 B of every collective (--ring-min), "
            "smaller collectives as under auto") in text
    assert (lowered.to_json()["sizes"]["chain_min"], lowered.to_json()["sizes"]["ring_min"]) == (0, 1 << 20)
    default = make_plan(repository)
    assert "SIRCL_CHAIN_MIN_BYTES" not in default.ranks[0].environment
    assert default.to_json()["sizes"]["chain_min"] is None
    assert ("auto schedules run chain ops from 8,388,608 B of all-reduce message, 8,388,608 B of all-gather output, "
            "4,194,304 B of reduce-scatter input (the session's defaults) where the ranks form a chain of cable "
            "neighbors") in plan_mod.render_text(default)
    for value, message in ((-1, "--chain-min must not be negative, got -1"), (True, "--chain-min must be a byte count")):
        with pytest.raises(ServePlanError, match=re.escape(message)):
            make_plan(repository, chain_min=value)
    with pytest.raises(ServePlanError, match="--env SIRCL_CHAIN_MIN_BYTES: the launcher owns every SIRCL_\\* "
                                             "variable; use --chain-min"):
        make_plan(repository, extra_env={"SIRCL_CHAIN_MIN_BYTES": "0"})
    base = ["--site", str(site_file), "--repository", str(repository), "--profile", PROFILE, "--model-path", MODEL]
    assert cli.main(["plan", "--json", *base, "--chain-min", "4194304"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert {item["name"]: item["after"] for item in record["changes"]}["SIRCL_CHAIN_MIN_BYTES"] == "4194304"
    assert record["sizes"]["chain_min"] == 4194304
    for name in cli.COMMANDS:
        assert cli.parser().parse_args([name, "--site", "s", "--repository", "r", "--chain-min", "0"]).chain_min == 0
    assert make_bundle(chain_min=1 << 20).ranks[5].environment["SIRCL_CHAIN_MIN_BYTES"] == "1048576"
    assert "SIRCL_CHAIN_MIN_BYTES" not in make_bundle().ranks[0].environment
    assert cli.main(["bundle", "--site", str(site_file), "--positions", "0-7", "--chain-min", "2097152"]) == 0
    assert json.loads(capsys.readouterr().out)["ranks"][0]["environment"]["SIRCL_CHAIN_MIN_BYTES"] == "2097152"


# Statistics of a session with the default minimums: per-collective maps, and no single size.
DEFAULT_MINIMUMS = {**LINK_SESSION, "chain_min_bytes": None, "ring_min_bytes": None,
                    "chain_mins": {"reduce": 8 << 20, "gather": 8 << 20, "scatter": 4 << 20},
                    "ring_mins": {"reduce": 4 << 20, "gather": 8 << 20, "scatter": 4 << 20}}


def test_session_ops_read_each_collectives_minimums_from_the_session_statistics():
    auto = plan_mod.SessionOps.from_stats(DEFAULT_MINIMUMS)
    assert (auto.chain_min, auto.gather_chain_min, auto.scatter_chain_min) == (8 << 20, 8 << 20, 4 << 20)
    assert auto.describe() == (
        "one chain op per all-reduce from 8,388,608 B, else all-reduce pieces of 4,194,304 B above 131,072 B; "
        "one chain op per reduce-scatter from 4,194,304 B, else reduce-scatter pieces of 262,144 B; "
        "one chain op per all-gather from 8,388,608 B, else all-gather pieces of 155,648 B")
    # 6 MiB all-reduce: two pieces below the 8 MiB chain minimum; 8 MiB: one chain op. Reduce-scatter inputs of
    # 2 and 4 MiB; all-gather outputs of 4 and 8 MiB (shards of 128 and 256 rows of 8,192 B).
    assert (auto.allreduce_ops(6 << 20), auto.allreduce_ops(8 << 20)) == (2, 1)
    assert (auto.scatter_ops(2 << 20), auto.scatter_ops(4 << 20)) == (8, 1)
    assert (auto.gather_ops(128, 8192), auto.gather_ops(256, 8192)) == (7, 1)
    # Ring schedules with all-reduce pieces of 1 MiB: a ring op from each collective's ring minimum, below it
    # as auto (the chain minimums lie at or above the ring minimums, so pieces).
    ring = plan_mod.SessionOps.from_stats({**DEFAULT_MINIMUMS, "large_piece_bytes": 1 << 20, "large_schedule": "ring",
                                           "gather_schedule": "ring", "scatter_schedule": "ring"})
    assert (ring.ring_min, ring.gather_ring_min, ring.scatter_ring_min) == (4 << 20, 8 << 20, 4 << 20)
    assert ring.describe() == (
        "one ring op per all-reduce from 4,194,304 B, else all-reduce pieces of 1,048,576 B above 131,072 B; "
        "one ring op per reduce-scatter from 4,194,304 B, else reduce-scatter pieces of 262,144 B; "
        "one ring op per all-gather from 8,388,608 B, else all-gather pieces of 155,648 B")
    assert (ring.allreduce_ops(3 << 20), ring.allreduce_ops(4 << 20), ring.allreduce_ops(6 << 20)) == (3, 1, 1)
    assert (ring.scatter_ops(2 << 20), ring.scatter_ops(4 << 20)) == (8, 1)
    assert (ring.gather_ops(128, 8192), ring.gather_ops(256, 8192)) == (7, 1)
    # A map without a collective falls back to the size the session states for all three.
    partial = plan_mod.SessionOps.from_stats({**LINK_SESSION, "chain_min_bytes": 1 << 20,
                                              "chain_mins": {"reduce": 16 << 20}})
    assert (partial.chain_min, partial.gather_chain_min, partial.scatter_chain_min) == (16 << 20, 1 << 20, 1 << 20)


def test_stage_prints_every_sparks_shim_statuses(repository):
    """The stage probe's shim statuses (catalog.tree_status in the serving image) reach the stage report."""
    plan = make_plan(repository, mhc_prefill_shard="off")
    record = good_record(plan.library)
    record["shim_status"] = {"schema": catalog.STATUS_SCHEMA, "tree": "/opt/sparkring-overlay/vllm", "matches": [],
                             "shims": [{"name": "mhc_prefill_shard", "status": "applicable-unverified", "build": None,
                                        "installs": False, "missing": []},
                                       {"name": "roce_slot", "status": "verified", "build": "lil-image-aba309e4610c",
                                        "installs": True, "missing": []}]}
    lines: list[str] = []
    assert cli.stage(context(plan, StageSparks(plan, record), lines)) == 0
    assert sum(line.endswith(": shims: mhc_prefill_shard applicable-unverified, roce_slot verified "
                             "(lil-image-aba309e4610c)") for line in lines) == 4
    assert sum(line.endswith(": shim mhc_prefill_shard applicable-unverified") for line in lines) == 4



def test_the_large_grid_cap_reaches_every_rank_of_start_and_bundle_and_check_reads_it_back(repository, site_file,
                                                                                          capsys):
    capped = make_plan(repository, large_blocks=4)
    assert {launch.environment["SIRCL_LARGE_BLOCKS"] for launch in capped.ranks} == {"4"}
    assert {change.name: change.reason for change in capped.changes}["SIRCL_LARGE_BLOCKS"].startswith(
        "--large-blocks: the largest grid, in blocks, of the sessions' two-shot and large-message launches")
    text = plan_mod.render_text(capped)
    assert "SIRCL_LARGE_BLOCKS: unset -> '4'" in text
    assert ("two-shot above (the session's limit: the latency model's for this layout, lane count and posting "
            "order), two-shot and large-message launches on grids of up to 4 blocks (--large-blocks), all-gather "
            "capacity") in text
    assert capped.to_json()["sizes"]["large_blocks"] == 4
    # Unset keeps the session's 32.
    default = make_plan(repository)
    assert "SIRCL_LARGE_BLOCKS" not in default.ranks[0].environment and default.to_json()["sizes"]["large_blocks"] is None
    assert "launches on grids of up to 32 blocks (the session's default)" in plan_mod.render_text(default)
    assert make_plan(repository, large_blocks=1).ranks[2].environment["SIRCL_LARGE_BLOCKS"] == "1"
    assert make_plan(repository, large_blocks=1024).ranks[2].environment["SIRCL_LARGE_BLOCKS"] == "1024"
    for value, message in ((12, "--large-blocks must be a power of two from 1 to 1024, got 12"),
                           (0, "power of two from 1 to 1024, got 0"), (2048, "got 2048"),
                           (True, "--large-blocks must be a block count")):
        with pytest.raises(ServePlanError, match=re.escape(message)):
            make_plan(repository, large_blocks=value)
    with pytest.raises(ServePlanError, match="--env SIRCL_LARGE_BLOCKS: the launcher owns every SIRCL_\\* "
                                             "variable; use --large-blocks"):
        make_plan(repository, extra_env={"SIRCL_LARGE_BLOCKS": "4"})
    base = ["--site", str(site_file), "--repository", str(repository), "--profile", PROFILE, "--model-path", MODEL]
    assert cli.main(["plan", "--json", *base, "--large-blocks", "8"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert {item["name"]: item["after"] for item in record["changes"]}["SIRCL_LARGE_BLOCKS"] == "8"
    assert record["sizes"]["large_blocks"] == 8
    assert cli.main(["plan", *base, "--large-blocks", "6"]) == 2
    assert "--large-blocks must be a power of two from 1 to 1024, got 6" in capsys.readouterr().err
    for name in cli.COMMANDS:
        assert cli.parser().parse_args([name, "--site", "s", "--repository", "r", "--large-blocks", "4"]).large_blocks == 4
    bundled = make_bundle(large_blocks=8)
    assert {launch.environment["SIRCL_LARGE_BLOCKS"] for launch in bundled.ranks} == {"8"}
    assert "SIRCL_LARGE_BLOCKS" not in make_bundle().ranks[0].environment
    with pytest.raises(ServePlanError, match="power of two"):
        make_bundle(large_blocks=3)
    assert cli.main(["bundle", "--site", str(site_file), "--positions", "0-7", "--large-blocks", "4"]) == 0
    assert json.loads(capsys.readouterr().out)["ranks"][0]["environment"]["SIRCL_LARGE_BLOCKS"] == "4"
    args = cli.parser().parse_args(["bundle-check", "--site", "s", "--positions", "0-7", "--container", "x-{rank}",
                                    "--large-blocks", "4"])
    assert args.large_blocks == 4
    # check reads back the cap each rank's session used (its statistics in the receipts).
    def receipts(stated):
        return lambda rank: [FakeSparks(capped).receipt(rank, session_stats={"large_blocks": stated(rank)})]

    lines: list[str] = []
    assert cli.check(context(capped, FakeSparks(capped, receipts=receipts(lambda rank: 4)), lines),
                     long_prompt=0) == 0
    assert ("large_blocks of rank(s) 0,1,2,3: 4 (two-shot and large-message grid cap; --large-blocks 4)") in lines
    lines.clear()
    assert cli.check(context(capped, FakeSparks(capped, receipts=receipts(lambda rank: 32 if rank == 2 else 4)),
                             lines), long_prompt=0) == 1
    assert any("rank 2: the session's two-shot and large-message grid cap is 32 blocks, --large-blocks asked for 4"
               in line for line in lines)
    lines.clear()
    assert cli.check(context(default, FakeSparks(default), lines), long_prompt=0) == 0
    assert ("large_blocks of rank(s) 0,1,2,3: the receipts' session statistics state none (the session's default "
            "not confirmed)") in lines


def test_bundle_check_reads_back_each_ranks_large_grid_cap():
    plan = make_bundle(large_blocks=8)

    def receipts(rank):
        return [{"group": "tp:0", "global_rank": rank, "rank": rank, "world": 8, "nccl": "none", "pynccl": "skipped",
                 "session": "ring", "state": "ready", "session_stats": {"large_blocks": 8 if rank != 5 else 32},
                 "decisions": [{"collective": "all_reduce", "backend": "sircl", "method": "direct", "calls": 9}]}]

    lines: list[str] = []
    names = bundle.container_names(["glm53-r{rank}"], plan)
    assert bundle.check(plan, names, BundleSparks(plan, receipts=receipts), lines.append) == 1
    assert "large_blocks of rank(s) 0,1,2,3,4,6,7: 8 (two-shot and large-message grid cap; --large-blocks 8)" in lines
    assert any("rank 5: the session's two-shot and large-message grid cap is 32 blocks" in line for line in lines)


def test_the_launchers_large_grid_cap_default_and_limit_are_the_sessions():
    package = Path(__file__).resolve().parents[1] / "sparkring_sircl"
    if not (package / "oneshot" / "runtime.py").is_file():
        pytest.skip("the session package is not present")
    from sparkring_sircl import env as session_env

    source = (package / "oneshot" / "runtime.py").read_text(encoding="utf-8")
    assert _module_constants(package / "oneshot" / "runtime.py")["DEFAULT_LARGE_BLOCKS"] == plan_mod.DEFAULT_LARGE_BLOCKS
    # The session refuses a cap above MAX_LARGE_BLOCKS (its SIRCL_LARGE_BLOCKS check).
    assert f"large_blocks > {plan_mod.MAX_LARGE_BLOCKS}" in source
    assert plan_mod.LARGE_BLOCKS_VARIABLE in {variable.name for variable in session_env.VARIABLES}



# -- NCCL-free serving ----------------------------------------------------------------------------------


def test_require_no_nccl_plans_only_groups_nccl_may_not_run_and_logs_ncclss_communicators(repository, site_file,
                                                                                         capsys):
    path = make_plan(repository, require_no_nccl=True)          # Sparks 0-3: NCCL may not run there anyway
    for launch in path.ranks:
        assert (launch.environment["NCCL_DEBUG"], launch.environment["NCCL_DEBUG_SUBSYS"]) == ("INFO", "INIT")
    changes = {change.name: change for change in path.changes}
    assert changes["NCCL_DEBUG"].reason.startswith("--nccl-debug: NCCL logs every communicator it creates")
    record = json.loads(json.dumps(path.to_json()))
    assert record["nccl_free"] == {"required": True, "nccl_debug": True, "nccl_allowed": False,
                                   "log_patterns": list(plan_mod.NCCL_INIT_PATTERNS),
                                   "library_patterns": ["NCCL INFO", "NCCL WARN"]}
    assert [row["name"] for row in record["carriers"]["groups"]] == ["world", "tp", "ep", "dcp", "pp", "dp", "pcp"]
    assert dict((row["collective"], row["carrier"]) for row in record["carriers"]["collectives"])["PyNccl"] == (
        "not built: no PyNccl communicator, no warm-up")
    text = plan_mod.render_text(path)
    assert "  NCCL-free (--require-no-nccl): no group lets NCCL run" in text
    assert ("    ep (one group of ranks 0-3, built for mixture-of-experts models): NCCL policy none (no cable "
            "between ranks 3-0 (positions 3-0)); SIRCL's communicator; the same ranks as tp, so it shares tp's "
            "session") in text
    assert "    send, recv: SIRCL point-to-point channels between every pair of the group's ranks" in text
    assert "refused where the group has no channels" in text
    assert "    all-reduce: one session op up to 131,072 B; larger ones the session's large-message ops" in text
    # A pair lets NCCL run unless --nccl never; then mHC prefill sharding runs on the session through its shim.
    with pytest.raises(ServePlanError, match=re.escape("--require-no-nccl: NCCL may run on pair:0-1 (")):
        make_plans(repository, ((0, 1),), require_no_nccl=True)
    pair = make_plans(repository, ((0, 1),), nccl_mode="never", require_no_nccl=True)[0]
    assert pair.nccl_policy.value == "none" and pair.required_shims == ("mhc_prefill_shard",)
    assert "NCCL_IB_HCA" not in {change.name for change in pair.changes}
    text = plan_mod.render_text(pair)
    assert "mHC prefill sharding: on (VLLM_GLM53_MHC_PREFILL_SHARD of the profile), carried by SIRCL's" in text
    topology = make_plans(repository, ((0, 1),))[0]
    text = plan_mod.render_text(topology)
    assert ("    all-reduce: one session op up to 131,072 B; eager calls above it go to NCCL where the group's NCCL "
            "policy allows the collective (SIRCL_LARGE_ALLREDUCE=auto)") in text
    assert "    PyNccl: built (NCCL may run here)" in text
    assert "  NCCL: may run on this group; --nccl never and --require-no-nccl serve without it" in text
    assert "NCCL_DEBUG" not in {change.name for change in topology.changes}
    debug = make_plan(repository, nccl_debug=True)
    assert debug.ranks[0].environment["NCCL_DEBUG"] == "INFO" and not debug.require_no_nccl
    assert "NCCL: no group lets NCCL run; --require-no-nccl also proves it with check (NCCL_DEBUG=INFO, INIT" in (
        plan_mod.render_text(debug))
    # Settings that would create NCCL communicators outside SIRCL's groups, or hand them to code SIRCL does not see.
    for options, message in (
            ({"require_no_nccl": True, "extra_env": {"NCCL_DEBUG": "WARN"}},
             "--env NCCL_DEBUG: --nccl-debug and --require-no-nccl set NCCL_DEBUG=INFO"),
            ({"nccl_mode": "never", "extra_env": {"VLLM_DISTRIBUTED_USE_SPLIT_GROUP": "1"}},
             "VLLM_DISTRIBUTED_USE_SPLIT_GROUP=1 binds vLLM's default process group to the GPU"),
            ({"require_no_nccl": True, "vllm_edits": plan_mod.vllm_edits(["--load-format=instanttensor"])},
             "--load-format instanttensor hands the world group's NCCL process group to the InstantTensor loader"),
            ({"nccl_mode": "never", "vllm_edits": plan_mod.vllm_edits(["--enable-eplb"])},
             "--enable-eplb: expert load balancing calls torch.distributed all_reduce and all_gather through "
             "names bound at import")):
        with pytest.raises(ServePlanError, match=re.escape(message)):
            make_plan(repository, **options)
    assert make_plan(repository, extra_env={"VLLM_DISTRIBUTED_USE_SPLIT_GROUP": "1"}).ranks[0].environment[
        "VLLM_DISTRIBUTED_USE_SPLIT_GROUP"] == "1"                     # the topology rule is the plugin's
    base = ["--site", str(site_file), "--repository", str(repository), "--profile", TP2.id, "--model-path", MODEL]
    assert cli.main(["plan", "--json", *base, "--positions", "0-1", "--nccl", "never", "--require-no-nccl"]) == 0
    assert json.loads(capsys.readouterr().out)["nccl_free"]["required"] is True
    assert cli.main(["plan", *base, "--positions", "0-1", "--nccl", "topology", "--require-no-nccl"]) == 2
    assert "add --nccl never" in capsys.readouterr().err
    assert cli.main(["plan", *base, "--positions", "0-1", "--require-no-nccl"]) == 0     # never by default
    capsys.readouterr()
    for name in cli.COMMANDS:
        args = cli.parser().parse_args([name, "--site", "s", "--repository", "r", "--require-no-nccl", "--nccl-debug"])
        assert args.require_no_nccl and args.nccl_debug, name


class NcclLogSparks(FakeSparks):
    """Sparks whose rank ``noisy`` logs an NCCL communicator."""

    def __init__(self, plans, *, noisy=None, line="NCCL INFO ncclCommInitRank comm 0x5 rank 1 nRanks 4 - Init COMPLETE",
                 **kwargs):
        super().__init__(plans, **kwargs)
        self.noisy, self.line = noisy, line

    def __call__(self, target, command, *, timeout=60, input_bytes=None):
        if "| grep -F" in command and "Init COMPLETE" in command:
            self.calls.append((target, command, input_bytes))
            rank = next(launch.rank for launch in self.launches() if launch.container in command)
            return remote.Result(0, self.line + "\n" if rank == self.noisy else "", "")
        return super().__call__(target, command, timeout=timeout, input_bytes=input_bytes)


def test_check_require_no_nccl_reads_every_receipt_and_scans_every_log(repository):
    plan = make_plan(repository, require_no_nccl=True)
    lines: list[str] = []
    assert cli.check(context(plan, NcclLogSparks(plan), lines), long_prompt=0) == 0
    assert ("NCCL-free receipts: every rank's groups (tp:0) show nccl=none, pynccl=skipped and no NCCL decision "
            "row") in lines
    assert "NCCL log scan: no rank's log shows an NCCL communicator" in lines
    lines.clear()
    assert cli.check(context(plan, NcclLogSparks(plan, noisy=1), lines), long_prompt=0) == 1
    assert any(line.startswith("PROBLEM: rank 1: its log shows 1 NCCL communicator line(s), first: NCCL INFO "
                               "ncclCommInitRank") for line in lines)

    def receipts(rank):
        fake = FakeSparks(plan)
        tp = fake.receipt(rank)
        ep = fake.receipt(rank, group="ep:0", nccl="ring", pynccl="built", nccl_reason="NCCL ring",
                          decisions=[{"collective": "all_reduce", "backend": "nccl", "method": "nccl", "calls": 7}])
        return [tp, ep]

    lines.clear()
    assert cli.check(context(plan, NcclLogSparks(plan, receipts=receipts), lines), long_prompt=0) == 1
    problems = [line for line in lines if line.startswith("PROBLEM: rank 2 group ep:0")]
    assert [problem.split(": ", 2)[2] for problem in problems] == [
        "NCCL policy ring (NCCL ring): NCCL may run on this group", "PyNccl built: vLLM built an NCCL communicator",
        "7 calls on NCCL (all_reduce/nccl=7)"]
    # Without --require-no-nccl check reads no log for NCCL lines.
    plain = make_plan(repository)
    fake = NcclLogSparks(plain, noisy=0)
    assert cli.check(context(plain, fake, []), long_prompt=0) == 0
    assert not any("Init COMPLETE" in command for _, command, _ in fake.calls)
    found, problems = checks.nccl_free_findings({0: [], 1: [FakeSparks(plan).receipt(1)]}, 2)
    assert problems == ["rank 0: no receipt, so its groups' use of NCCL is unknown"] and found == []
    scan, problems = checks.nccl_log_findings({0: ["INFO vLLM is using nccl==2.32.3"]}, debug=False)
    assert problems == ["rank 0: its log shows 1 NCCL communicator line(s), first: INFO vLLM is using nccl==2.32.3"]
    scan, problems = checks.nccl_log_findings({0: []}, debug=False)
    assert problems == [] and "only vLLM's PyNccl line was in view" in scan[0]


def test_the_bundle_proves_a_tp8_launch_nccl_free_and_maps_every_group(site_file, capsys):
    plan = make_bundle(run_id="glm53-tp8", nccl_mode="never", large_allreduce="sircl", require_no_nccl=True)
    assert plan.nccl_policy.value == "none"
    for launch in plan.ranks:
        assert (launch.environment["NCCL_DEBUG"], launch.environment["NCCL_DEBUG_SUBSYS"]) == ("INFO", "INIT")
        assert launch.environment["SIRCL_LARGE_ALLREDUCE"] == "sircl"
    document = json.loads(json.dumps(plan.to_json()))
    assert document["nccl_free"]["required"] and not document["nccl_free"]["nccl_allowed"]
    assert [item.split(":")[0] for item in document["requirements"][-3:]] == [
        "VLLM_DISTRIBUTED_USE_SPLIT_GROUP unset or 0", "no --load-format instanttensor", "no --enable-eplb"]
    assert not any("--enable-eplb" in item for item in make_bundle().to_json()["requirements"])   # NCCL's ring
    assert [row["name"] for row in document["carriers"]["groups"]] == ["world", "tp", "ep", "dcp", "pp", "dp", "pcp"]
    text = plan.render_text()
    assert text.startswith("bundle glm53-tp8: SIRCL for a vLLM launch on Sparks [0, 1, 2, 3, 4, 5, 6, 7]")
    assert "group cycle:0-1-2-3-4-5-6-7 with NCCL policy none (SIRCL_NCCL=never)" in text
    assert ("    tp (one group of ranks 0-7): NCCL policy none (SIRCL_NCCL=never); SIRCL's communicator and the "
            "group's SIRCL session") in text
    assert plan.check_command().endswith("--nccl never --require-no-nccl --container 'NAME-{rank}'")
    assert "    all-gather: one session op up to 155,648 B; larger ones the session's large-message ops" in text
    # Decode-context-parallel groups need their own session.
    dcp = make_bundle(session_groups="tp,dcp", dcp_size=4)
    groups = {row.name: row for row in dcp.carriers()[0]}
    assert groups["dcp"].groups == "2 groups of 4 consecutive ranks"
    assert groups["dcp"].carrier.startswith("SIRCL's communicator and a session of its own per group "
                                            "(SIRCL_GROUPS tp,dcp)")
    for options, message in (({"nccl_mode": "topology", "require_no_nccl": True},
                              "--require-no-nccl: NCCL may run on cycle:0-1-2-3-4-5-6-7"),
                             ({"dcp_size": 4}, "--dcp-size 4: decode-context-parallel groups need a SIRCL session"),
                             ({"dcp_size": 3, "session_groups": "tp,dcp"}, "--dcp-size 3 must divide"),
                             ({"nccl_mode": "never", "extra_env": {"VLLM_DISTRIBUTED_USE_SPLIT_GROUP": "1"}},
                              "VLLM_DISTRIBUTED_USE_SPLIT_GROUP=1 binds"),
                             ({"nccl_mode": "never", "require_no_nccl": True,
                               "extra_env": {"NCCL_DEBUG_SUBSYS": "ALL"}},
                              "--env NCCL_DEBUG_SUBSYS")):
        with pytest.raises(ServePlanError, match=re.escape(message)):
            make_bundle(**options)
    assert "NCCL_DEBUG" not in make_bundle().ranks[0].environment
    assert make_bundle(nccl_debug=True).ranks[3].environment["NCCL_DEBUG"] == "INFO"
    assert cli.main(["bundle", "--site", str(site_file), "--positions", "0-7", "--nccl", "never",
                     "--large-allreduce", "sircl", "--require-no-nccl", "--text"]) == 0
    assert "NCCL-free (--require-no-nccl)" in capsys.readouterr().out
    args = cli.parser().parse_args(["bundle-check", "--site", "s", "--positions", "0-7", "--container", "x-{rank}",
                                    "--nccl", "never", "--require-no-nccl", "--nccl-debug"])
    assert args.require_no_nccl and args.nccl_debug and args.nccl == "never"


class NcclBundleSparks(BundleSparks):
    """Bundle Sparks whose rank ``noisy`` logs an NCCL communicator."""

    def __init__(self, plan, *, noisy=None, **kwargs):
        super().__init__(plan, **kwargs)
        self.noisy = noisy

    def __call__(self, target, command, *, timeout=60, input_bytes=None):
        if "| grep -F" in command and "Init COMPLETE" in command:
            self.calls.append((target, command, input_bytes))
            rank = next(launch.rank for launch in self.plan.ranks if launch.ssh == target)
            return remote.Result(0, "NCCL INFO comm 0x1 rank 0 nRanks 8 - Init COMPLETE\n"
                                 if rank == self.noisy else "", "")
        return super().__call__(target, command, timeout=timeout, input_bytes=input_bytes)


def test_bundle_check_require_no_nccl_reads_every_receipt_and_scans_every_log():
    plan = make_bundle(run_id="glm53-tp8", nccl_mode="never", require_no_nccl=True)
    names = bundle.container_names(["glm53-r{rank}"], plan)
    lines: list[str] = []
    assert bundle.check(plan, names, NcclBundleSparks(plan), lines.append) == 0
    assert "NCCL log scan: no rank's log shows an NCCL communicator" in lines
    lines.clear()
    assert bundle.check(plan, names, NcclBundleSparks(plan, noisy=6), lines.append) == 1
    assert any("rank 6: its log shows 1 NCCL communicator line(s)" in line for line in lines)


def test_the_general_plugin_refuses_vllms_split_group_initialization_under_sircl_nccl_never():
    from sparkring_sircl.vllm import guard

    never = guard.environment_problems({"SIRCL_NCCL": "never", "VLLM_DISTRIBUTED_USE_SPLIT_GROUP": "1"})
    assert len(never) == 1 and "creates an NCCL communicator over every rank at startup" in never[0]
    assert guard.environment_problems({"SIRCL_NCCL": "topology", "VLLM_DISTRIBUTED_USE_SPLIT_GROUP": "1"}) == []
    assert guard.environment_problems({"SIRCL_NCCL": "never", "VLLM_DISTRIBUTED_USE_SPLIT_GROUP": "0"}) == []
    assert guard.environment_problems({"VLLM_DISTRIBUTED_USE_SPLIT_GROUP": "1"}) == never     # unset is never
    connect = guard.environment_problems({"SIRCL_NCCL": "topology", "VLLM_DISTRIBUTED_USE_SPLIT_GROUP": "1",
                                          "NCCL_RUNTIME_CONNECT": "0"})
    assert len(connect) == 1 and "NCCL_RUNTIME_CONNECT=0" in connect[0]


def test_profile_serving_runs_decode_context_parallelism_with_a_session_per_group(repository, monkeypatch):
    """--dcp-size N: N divides the tensor parallelism; above 1 the launcher sets vLLM's decode-context parallelism
    and a SIRCL session per DCP group (SIRCL_GROUPS tp,dcp), the plan states the groups and sessions, check wants
    every rank's DCP receipt with a session, and the plan refuses where the checkpoint or its attention backend
    does not run DCP. GLM-5.3-Flash at TP2 with DCP 2 on a cabled pair, NCCL off and mHC prefill sharding off,
    and TP4 with DCP 2 on a path."""
    b12x = plan_mod.vllm_edits(["--attention-backend=B12X"])
    pair = make_plans(repository, ((0, 1),), dcp_size=2, nccl_mode="never", vllm_edits=b12x,
                      mhc_prefill_shard="off")[0]
    assert pair.dcp_size == 2 and pair.to_json()["dcp_size"] == 2
    for launch in pair.ranks:
        command = list(launch.command)
        assert command[command.index("--decode-context-parallel-size") + 1] == "2"
        assert command[command.index("--cp-kv-cache-interleave-size") + 1] == "4"    # set where the recipe has none
        assert launch.environment["SIRCL_GROUPS"] == "tp,dcp" and launch.environment["SIRCL_NCCL"] == "never"
    assert plan_mod.recipe_dcp(pair.recipe_arguments) == 2
    text = plan_mod.render_text(pair)
    assert "--decode-context-parallel-size 1 -> --decode-context-parallel-size 2  (--dcp-size)" in text
    assert "unset -> --cp-kv-cache-interleave-size 4  (--dcp-size)" in text
    assert "    decode-context-parallel sessions (dcp, one group of ranks 0-1): schedules:" in text
    rows = {row.name: row for row in pair.carriers()[0]}
    assert rows["dcp"].groups == "one group of ranks 0-1" and rows["dcp"].nccl == "none"
    assert rows["dcp"].carrier.startswith("SIRCL's communicator and a session of its own per group")
    assert [session.name for session in pair.sessions()] == ["tp", "dcp"]
    assert pair.required_shims[:2] == ("dcp_all_to_all", "dcp_b12x_transport")   # the stage probe checks them
    # The default keeps the profile's decode-context parallelism of 1 and the tensor-parallel session alone.
    plain = make_plans(repository, ((0, 1),), nccl_mode="never")[0]
    assert plain.ranks[0].environment["SIRCL_GROUPS"] == "tp" and plain.dcp_size == 1
    assert "--cp-kv-cache-interleave-size" not in plain.ranks[0].command
    assert [session.name for session in plain.sessions()] == ["tp"]
    # TP4 with DCP 2 on the path of Sparks 0-3: two groups of two cabled ranks.
    tp4 = make_plan(repository, dcp_size=2, nccl_mode="never", vllm_edits=b12x)
    assert {row.name: row for row in tp4.carriers()[0]}["dcp"].groups == "2 groups of 2 consecutive ranks"
    for options, message in (
            ({"dcp_size": 3}, "--dcp-size 3 must be a positive divisor of the profile's tensor parallelism 2"),
            ({"dcp_size": 0}, "--dcp-size 0 must be a positive divisor"),
            ({"dcp_size": 2}, "the recipe's attention backend (unset: vLLM chooses) does not run GLM-5.3-Flash's "
                              "attention with decode-context parallelism; B12X does"),
            ({"dcp_size": 2, "vllm_edits": plan_mod.vllm_edits(["--attention-backend=FLASHINFER"])},
             "the recipe's attention backend FLASHINFER does not run")):
        with pytest.raises(ServePlanError, match=re.escape(message)):
            make_plans(repository, ((0, 1),), nccl_mode="never", **options)
    # The CSF checkpoint served with --checkpoint-id runs it too.
    assert plan_mod.dcp_problem(2, 2, "local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD@dec48abd",
                                ["--attention-backend", "B12X"], {}) is None
    monkeypatch.setattr(plan_mod, "DCP_MODELS", {"other/model": "Other"})
    with pytest.raises(ServePlanError, match=re.escape(f"the served checkpoint {REPOSITORY} is not one whose "
                                                       "attention the served images run with decode-context")):
        make_plans(repository, ((0, 1),), nccl_mode="never", dcp_size=2, vllm_edits=b12x)
    monkeypatch.undo()
    # The KV-cache interleave GLM-5.3-Flash's DCP needs: a recipe value divisible by 4 stays, another is refused.
    eight = make_plans(repository, ((0, 1),), dcp_size=2, nccl_mode="never", mhc_prefill_shard="off",
                       vllm_edits=plan_mod.vllm_edits(["--attention-backend=B12X",
                                                       "--cp-kv-cache-interleave-size=8"]))[0]
    for launch in eight.ranks:
        command = list(launch.command)
        assert command.count("--cp-kv-cache-interleave-size") == 1
        assert command[command.index("--cp-kv-cache-interleave-size") + 1] == "8"
    with pytest.raises(ServePlanError, match=re.escape("GLM-5.3-Flash's attention under decode-context parallelism "
                                                       "needs --cp-kv-cache-interleave-size divisible by 4; the "
                                                       "recipe gives 2")):
        make_plans(repository, ((0, 1),), dcp_size=2, nccl_mode="never", mhc_prefill_shard="off",
                   vllm_edits=plan_mod.vllm_edits(["--attention-backend=B12X", "--cp-kv-cache-interleave-size=2"]))
    # mHC prefill row ownership starts at TP2 only with DCP 1 in every pinned build: TP2 with DCP 2 and the
    # profile's mHC sharding is refused, TP4 with DCP 2 keeps it, and a build pinned to admit TP2/DCP2 serves it.
    with pytest.raises(ServePlanError, match=re.escape(
            "--dcp-size 2: GLM-5.3-Flash's mHC prefill row ownership (VLLM_GLM53_MHC_PREFILL_SHARD=1, the "
            "profile's) starts only at TP2/DCP1, TP4/DCP1, TP4/DCP2, TP4/DCP4 in the pinned vLLM builds, and the "
            "model refuses TP2 with DCP 2 at startup; serve it with --mhc-prefill-shard off")):
        make_plans(repository, ((0, 1),), dcp_size=2, nccl_mode="never", vllm_edits=b12x)
    assert tp4.mhc_prefill_shard and tp4.dcp_size == 2
    monkeypatch.setitem(pins.MHC_ADMITS, "sparkring-kraken-beta-20261007-bc9ea774",
                        pins.MHC_ADMITS_DEFAULT + ((2, 2),))
    admitted = make_plans(repository, ((0, 1),), dcp_size=2, nccl_mode="never", vllm_edits=b12x)[0]
    assert admitted.mhc_prefill_shard and pins.mhc_builds(2, 2) == ["sparkring-kraken-beta-20261007-bc9ea774"]
    monkeypatch.undo()
    # vLLM's parallel sizes stay the launcher's: --vllm-arg may not set it.
    with pytest.raises(ServePlanError, match="the launcher owns this argument"):
        plan_mod.vllm_edits(["--decode-context-parallel-size=2"])
    for name in cli.COMMANDS:
        args = cli.parser().parse_args([name, "--site", "s", "--repository", "r", "--dcp-size", "2"])
        assert args.dcp_size == 2, name
    # check: every rank needs its decode-context-parallel receipt with a session.
    tp = {"group": "tp:0", "state": "ready", "nccl": "none", "pynccl": "skipped", "session": "sircl",
          "decisions": [{"collective": "all_reduce", "backend": "sircl", "method": "oneshot", "calls": 3}]}
    dcp = {"group": "dcp:0", "state": "ready", "nccl": "none", "pynccl": "skipped", "session": "sircl",
           "decisions": []}
    found, _ = checks.evaluate_receipts({0: [tp, dcp], 1: [tp]}, 2, dcp=2)
    assert found == ["rank 1: no decode-context-parallel receipt with a SIRCL session (--dcp-size 2)"]
    assert checks.evaluate_receipts({0: [tp], 1: [tp]}, 2)[0] == []


def test_serve_and_bundle_share_one_nccl_default_and_state_every_groups_policy(repository):
    from sparkring_sircl.vllm import fabric

    assert plan_mod.DEFAULT_NCCL_MODE == "never" and plan_mod.NCCL_MODES == ("never", "auto")
    assert Options(positions=(0,)).nccl_mode == bundle.BundleOptions(positions=(0,)).nccl_mode == (
        plan_mod.DEFAULT_NCCL_MODE)
    for command in (["plan", "--site", "s", "--repository", "r"],
                    ["bundle", "--site", "s", "--positions", "0-7"],
                    ["bundle-check", "--site", "s", "--positions", "0-7", "--container", "x-{rank}"]):
        assert cli.parser().parse_args(command).nccl == "never", command[0]
        assert cli.parser().parse_args([*command, "--nccl", "auto"]).nccl == "auto", command[0]
        assert cli.parser().parse_args([*command, "--nccl", "topology"]).nccl == "auto", command[0]
        with pytest.raises(SystemExit):
            cli.parser().parse_args([*command, "--nccl", "sometimes"])
    # By default no group of the ring of eight lets NCCL run.
    default = make_bundle(nccl_mode=plan_mod.DEFAULT_NCCL_MODE, session_groups="tp,dcp", dcp_size=4)
    assert default.nccl_policy.value == "none" and default.to_json()["nccl_mode"] == "never"
    # TP8 on the ring of eight with --nccl auto (and its other name topology): NCCL's ring on tp and ep, the
    # decode-context-parallel paths on SIRCL.
    ring = make_bundle(session_groups="tp,dcp", dcp_size=4)
    assert ring.nccl_policy.value == "ring" and ring.ranks[0].environment["NCCL_ALGO"] == "Ring"
    assert ring.to_json()["nccl_mode"] == "auto"
    assert "--nccl auto --container" in ring.check_command()
    named = make_bundle(nccl_mode="topology", session_groups="tp,dcp", dcp_size=4)
    assert named.to_json() == ring.to_json() and named.ranks[0].environment["SIRCL_NCCL"] == "auto"
    # Plan text and JSON state the NCCL rule and the launch's mode, in profile serving and in a bundle.
    rule = "  NCCL: opt-in only (auto); tables choose among SIRCL options; this launch: --nccl "
    for mode, resolved in (("never", "never"), ("auto", "auto"), ("topology", "auto")):
        served = make_plans(repository, ((0, 1),), nccl_mode=mode)[0]
        bundled = make_bundle(nccl_mode=mode)
        views = ((plan_mod.render_text(served), served.to_json(), served.ranks[0].environment),
                 (bundled.render_text(), bundled.to_json(), bundled.ranks[0].environment))
        for text, record, environment in views:
            assert rule + resolved + ", so NCCL " in text, mode
            assert (record["nccl_mode"], record["nccl_rule"]) == (resolved, plan_mod.NCCL_RULE), mode
            assert environment["SIRCL_NCCL"] == resolved, mode
    assert plan_mod.NCCL_RULE == "NCCL: opt-in only (auto); tables choose among SIRCL options"
    groups = {row.name: row for row in ring.carriers()[0]}
    assert {name: row.nccl for name, row in groups.items()} == {
        "world": "ring", "tp": "ring", "ep": "ring", "dcp": "none", "pp": "-", "dp": "-", "pcp": "-"}
    assert groups["tp"].nccl_reason == ("consecutive ranks share cables around the whole group; NCCL ring algorithm "
                                        "only (NCCL_ALGO=Ring and NCCL_SKIP_TREE_CONNECT=1)")
    assert groups["ep"].carrier.startswith("NCCL: SIRCL's communicator without a session (only a group NCCL may "
                                           "not run shares tp's), so NCCL carries its collectives")
    assert groups["dcp"].nccl_reason == ("ranks 0-3: none (no cable between ranks 3-0 (positions 3-0, 2 relays)); "
                                         "ranks 4-7: none (no cable between ranks 3-0 (positions 7-4, 2 relays))")
    assert groups["world"].carrier.endswith("direct torch.distributed calls on its device group or the default "
                                            "group run on NCCL")
    text = ring.render_text()
    assert ("    ep (one group of ranks 0-7, built for mixture-of-experts models): NCCL policy ring (consecutive ranks "
            "share cables around the whole group; NCCL ring algorithm only (NCCL_ALGO=Ring and "
            "NCCL_SKIP_TREE_CONNECT=1)); NCCL: SIRCL's communicator") in text
    assert "    pp (pipeline parallelism 1): one rank per group" in text
    # --nccl never: every group's policy is none, and ep shares tp's session.
    never = make_bundle(nccl_mode="never", session_groups="tp,dcp", dcp_size=4)
    rows = never.carriers()[0]
    assert {(row.nccl, row.nccl_reason) for row in rows if row.nccl != "-"} == {("none", "SIRCL_NCCL=never")}
    assert {row.name: row for row in rows}["ep"].carrier.startswith("SIRCL's communicator; the same ranks as tp")
    # Groups without a session: refused where NCCL may not run, NCCL's where the cabling lets it run.
    layout = fabric.Layout.ring(8)
    refused = {row.name: row for row in plan_mod.group_carriers(layout, range(8), nccl_mode="topology",
                                                                 environment=bundle.RING_SETTINGS, dcp=4)}
    assert refused["dcp"].carrier.startswith("refused at startup: a decode-context-parallel group NCCL may not run")
    pairs = {row.name: row for row in plan_mod.group_carriers(layout, range(4), nccl_mode="topology", dcp=2)}
    assert (pairs["dcp"].nccl, pairs["dcp"].nccl_reason) == ("all", "every pair of ranks shares a cable")
    assert pairs["dcp"].carrier.startswith("NCCL: the groups have no SIRCL session")
    # The serve plan: with --nccl auto a pair lets NCCL run every collective; the recipe gives
    # decode-context parallelism.
    pair = make_plans(repository, ((0, 1),))[0]
    rows = {row.name: row for row in pair.carriers()[0]}
    assert (rows["tp"].nccl, rows["ep"].nccl) == ("all", "all")
    assert rows["dcp"].groups == "decode-context parallelism 1"
    assert [row["nccl"] for row in pair.to_json()["carriers"]["groups"]][:3] == ["all", "all", "all"]
    assert plan_mod.recipe_dcp(["--decode-context-parallel-size", "2"]) == 2
    assert plan_mod.recipe_dcp(["-dcp=4"]) == 4
    assert plan_mod.recipe_dcp(["--decode-context-parallel-size", "x"]) == plan_mod.recipe_dcp([]) == 1


def test_the_plan_shows_the_one_shot_limit_and_posting_order_the_sessions_derive(repository):
    from sparkring_sircl import env as session_env

    path = make_plan(repository)                                      # Sparks 0-3, a path of four
    assert path.derived_oneshot_max == 73728
    sizes = path.to_json()["sizes"]
    assert (sizes["oneshot_max"], sizes["oneshot_max_derived"], sizes["post_order"]) == (None, 73728, "farthest")
    text = plan_mod.render_text(path)
    assert "dispatch ceiling 131072 B, one-shot all-reduces up to 73,728 B and two-shot above (the session's " in text
    assert "; lanes posted to the peers with the most relays first (SIRCL_POST_ORDER unset: the session's order " in text
    ring = make_bundle()                                              # the ring of eight
    assert ring.derived_oneshot_max == 28672
    assert ring.to_json()["session_defaults"] == {"oneshot_max": None, "oneshot_max_derived": 28672,
                                                  "post_order": "farthest"}
    assert ("  tensor-parallel session: one-shot all-reduces up to 28,672 B and two-shot above (the session's limit"
            in ring.render_text())
    given = make_bundle(oneshot_max=65536)
    assert "one-shot all-reduces up to 65,536 B and two-shot above (--oneshot-max)" in given.render_text()
    assert make_bundle(capacity=16384, dispatch=16384).derived_oneshot_max <= 16384
    pair = make_plans(repository, ((0, 1),))[0]
    assert pair.derived_oneshot_max == 131072
    # The values the core documents for its own default, from the same rule.
    documented = {variable.name: variable.default for variable in session_env.VARIABLES}["SIRCL_ONESHOT_MAX_BYTES"]
    assert "28672 on a ring of eight farthest first" in documented and "73728 on a path of four" in documented
    source = (Path(session_env.__file__).parent / "oneshot" / "runtime.py").read_text(encoding="utf-8")
    assert "latency_model.oneshot_limit(layout, self.lane_count, order," in source
    assert "cap=min(self.max_size, DEFAULT_ONESHOT_MAX_BYTES)" in source
    assert 'self._post_order_name = "farthest" if layout is not None else "rank"' in source


# -- SIRCL sessions: scope of the schedule and link options -------------------------------------------


TP8_RING = dict(run_id="glm53-tp8", session_groups="tp,dcp", dcp_size=4, fused_norm=True, capacity=1048576,
                oneshot_max=28672, nccl_mode="never", large_allreduce="sircl", require_no_nccl=True,
                large_schedule="ring", link_sizes={"reduce_link_chunk": 2097152, "link_slot": 2097152},
                link_slots=12)


def test_schedule_and_link_options_reach_the_tensor_parallel_session_and_dcp_sessions_keep_the_defaults(
        repository, site_file, capsys):
    plan = make_bundle(**TP8_RING)
    first = plan.ranks[0].environment
    assert (first["SIRCL_LARGE_SCHEDULE"], first["SIRCL_LINK_SLOTS"], first["SIRCL_LINK_SLOT_BYTES"],
            first["SIRCL_REDUCE_LINK_CHUNK_BYTES"]) == ("ring", "12", "2097152", "2097152")
    tp, dcp = plan.sessions()
    assert (tp.name, tp.scoped, tp.covers_fabric, tp.chain_order) == ("tp", True, True, tuple(range(8)))
    assert tp.schedules == {"large_schedule": "ring", "gather_schedule": "auto", "scatter_schedule": "pieces"}
    assert (tp.chain_area, tp.link_area, tp.ring, tp.link_slots) == (
        True, True, "available, every ring edge is a cable", 12)
    assert (dcp.name, dcp.scoped, dcp.groups, dcp.ranks_on_fabric) == (
        "dcp", False, "2 groups of 4 consecutive ranks", (4, 8))
    assert dcp.schedules == plan_mod.DEFAULT_SCHEDULES and (dcp.chain_area, dcp.link_area) == (False, False)
    assert (dcp.link_slots, dcp.link_sizes, dcp.problems) == (plan_mod.DEFAULT_LINK_SLOTS, {}, ())
    text = plan.render_text()
    assert ("    tensor-parallel session (tp, one group of ranks 0-7): schedules: large all-reduces ring "
            "(--large-schedule), large all-gathers auto (the session's default), reduce-scatters pieces (the "
            "session's default); chain order 0-1-2-3-4-5-6-7; chain area yes, link area yes; ring: available, every "
            "ring edge is a cable; link slot 2,097,152 B (--link-slot)") in text
    assert "all-reduce link piece 2,097,152 B (--reduce-link-chunk), link slots 12 (--link-slots); " in text
    assert ("    decode-context-parallel sessions (dcp, 2 groups of 4 consecutive ranks): schedules: large all-reduces "
            "auto (the session's default)") in text
    assert ("no chain or link area: its ranks occupy 4 of the 8 Sparks of its fabric, so every large-message op "
            "runs in pieces and the link sizes and chain and ring minimums do not apply; SIRCL's adapter builds "
            "these sessions without the tensor-parallel session's schedule and link variables "
            "(settings.TP_SESSION_VARIABLES)") in text
    sessions = json.loads(json.dumps(plan.to_json()))["sessions"]
    assert [(row["name"], row["scoped"], row["schedules"]["large"], row["link_slots"]) for row in sessions] == [
        ("tp", True, "ring", 12), ("dcp", False, "auto", 8)]
    assert sessions[0]["link_slot"] == 2097152 and sessions[1]["covers_fabric"] is False
    assert any(item.startswith("no SIRCL_* variable from the launcher's own environment")
               for item in plan.to_json()["requirements"])
    assert [row.name for row in make_bundle(session_groups="tp,dcp").sessions()] == ["tp"]    # DCP size 1
    assert [row.name for row in make_bundle(dcp_size=4, session_groups="tp,dcp", nccl_mode="never").sessions()] == [
        "tp", "dcp"]
    # A serve plan: the tensor-parallel session on a path of four, whose ring crosses the path's relays.
    path = make_plan(repository, large_schedule="ring", link_slots=16)
    (session,) = path.sessions()
    assert session.ring == "available, relayed ring lanes keep up to 393,216 B unacknowledged"
    assert path.ranks[0].environment["SIRCL_LINK_SLOTS"] == "16" and path.to_json()["sizes"]["link_slots"] == 16
    assert {change.name: change.reason for change in path.changes}["SIRCL_LINK_SLOTS"].startswith(
        "--link-slots: the receive and own slots of each link of the tensor-parallel session's chain and ring ops")
    assert "  SIRCL sessions (the schedule and link options apply to the tensor-parallel session; other sessions " \
           "keep the session defaults):" in plan_mod.render_text(path)
    assert json.loads(json.dumps(path.to_json()))["sessions"][0]["chain_order"] == [0, 1, 2, 3]
    for value, message in ((1, "--link-slots must be 2 to 32, got 1"), (33, "got 33"),
                           (True, "--link-slots must be a slot count")):
        with pytest.raises(ServePlanError, match=re.escape(message)):
            make_plan(repository, link_slots=value)
        with pytest.raises(ServePlanError, match=re.escape(message)):
            make_bundle(link_slots=value)
    with pytest.raises(ServePlanError, match=re.escape("--env SIRCL_LINK_SLOTS: the launcher owns every SIRCL_* "
                                                      "variable; use --link-slots")):
        make_plan(repository, extra_env={"SIRCL_LINK_SLOTS": "12"})
    assert cli.main(["bundle", "--site", str(site_file), "--positions", "0-7", "--session-groups", "tp,dcp",
                     "--dcp-size", "4", "--large-schedule", "ring", "--link-slots", "12", "--text"]) == 0
    assert "link slots 12 (--link-slots)" in capsys.readouterr().out
    for name in cli.COMMANDS:
        assert cli.parser().parse_args([name, "--site", "s", "--repository", "r", "--link-slots", "12"]).link_slots == 12


def test_settings_a_session_would_refuse_at_setup_are_refused_by_plan_and_bundle(repository, monkeypatch):
    from sparkring_sircl import routes as routes_mod
    from sparkring_sircl.vllm import fabric

    layout = fabric.Layout.ring(8)
    sub = fabric.describe_group(layout, [0, 1, 2, 3], parent=list(range(8)))
    for schedule in ("ring", "chain"):
        refused = plan_mod.session_settings(sub, name="dcp", groups="2 groups of 4", scoped=True,
                                            schedules={"large_schedule": schedule})
        assert refused.problems == (f"--large-schedule {schedule}: the dcp session's ranks occupy 4 of the 8 Sparks "
                                    "of its fabric, and a session runs chain and ring ops only when every Spark of "
                                    "its fabric hosts one of its ranks",)
        with pytest.raises(ServePlanError, match=re.escape("SIRCL's sessions would refuse these settings at setup "
                                                           f"on every rank: --large-schedule {schedule}: the dcp")):
            plan_mod.session_problems([refused])
    assert plan_mod.session_settings(sub, name="dcp", groups="", scoped=False).problems == ()
    # A ring that cannot run, and ranks that form no chain: refused before any container starts.
    monkeypatch.setattr(routes_mod, "ring_window", lambda *args, **kwargs: (0, ["queue q carries two ring lanes"]))
    with pytest.raises(ServePlanError, match=re.escape("--large-schedule ring: the tp session's ring cannot run: "
                                                       "queue q carries two ring lanes")):
        make_bundle(large_schedule="ring")
    with pytest.raises(ServePlanError, match=re.escape("--gather-schedule ring: the tp session's ring cannot run")):
        make_plan(repository, gather_schedule="ring")
    assert make_bundle(gather_schedule="chain").sessions()[0].problems == ()        # chains need no ring
    assert make_bundle().sessions()[0].ring == "cannot run: queue q carries two ring lanes"
    monkeypatch.setattr(routes_mod, "chain_order", lambda *args, **kwargs: None)
    with pytest.raises(ServePlanError, match=re.escape("--scatter-schedule chain: the tp session's ranks do not "
                                                       "form a chain of cable neighbors joined by direct lanes")):
        make_plan(repository, scatter_schedule="chain")


def test_the_tensor_parallel_session_variables_are_every_schedule_and_link_variable_the_session_reads():
    from sparkring_sircl.vllm import settings as adapter_settings

    package = Path(plan_mod.__file__).resolve().parents[2]
    runtime = package / "oneshot" / "runtime.py"
    source = runtime.read_text(encoding="utf-8")
    tree = ast.parse(source)
    configure = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_configure")
    names = {node.value for node in ast.walk(configure) if isinstance(node, ast.Constant)
             and isinstance(node.value, str) and re.fullmatch(r"SIRCL_[A-Z_]+", node.value)}
    constants = _module_constants(runtime)
    scoped = {name for name in names if re.fullmatch(r"SIRCL_(LARGE|GATHER|SCATTER)_SCHEDULE|SIRCL_(CHAIN|LINK|RING)_\w+",
                                                     name)} | set(constants["LINK_COLLECTIVES"].values())
    assert scoped == set(adapter_settings.TP_SESSION_VARIABLES)
    assert len(adapter_settings.TP_SESSION_VARIABLES) == len(set(adapter_settings.TP_SESSION_VARIABLES))
    # The plan's copies of the session's defaults and rules.
    assert constants["DEFAULT_LINK_SLOTS"] == plan_mod.DEFAULT_LINK_SLOTS
    assert constants["DEFAULT_RING_STAGGER"] == plan_mod.DEFAULT_RING_STAGGER
    assert constants["DEFAULT_RING_GATHER_STAGGER"] == plan_mod.DEFAULT_RING_GATHER_STAGGER
    for attribute, variable in plan_mod.SCHEDULE_VARIABLES:
        assert f'_env_text("{variable}", default="{plan_mod.DEFAULT_SCHEDULES[attribute]}")' in source
    for rule in ("fabric_full = layout is not None and sorted(layout.positions) == list(layout.fabric.positions)",
                 'self._chain_region = self.large_schedule != "pieces" and fabric_full and chain_threads',
                 "self._link_region = fabric_full and chain_threads and (gather_links or scatter_links or reduce_links",
                 "or table_links)",
                 "table_links = self._tuning is not None and any(",
                 'reduce_links = self.large_schedule == "ring"',
                 "proto.LinkLayout(self.lane_count, self.link_slots, self.link_slot_bytes)",
                 "routes_mod.ring_window(self._layout_identity_object, route_maps, order,",
                 "chunk=proto.LINK_WINDOW_CHUNK,"):
        assert rule in source, rule
    dcp = (package / "vllm" / "dcp_collectives.py").read_text(encoding="utf-8")
    assert "**dict.fromkeys(TP_SESSION_VARIABLES)," in dcp


def test_check_reports_nccl_library_activity_without_counting_it_as_a_communicator(repository):
    plan = make_plan(repository, require_no_nccl=True)
    assert plan.to_json()["nccl_free"]["library_patterns"] == ["NCCL INFO", "NCCL WARN"]
    exit_line = "spark-b:150:263 [0] NCCL INFO ENV/Plugin: Could not find: libnccl-env.so"
    lines: list[str] = []
    fake = NcclLogSparks(plan, noisy=2, line=exit_line)
    assert cli.check(context(plan, fake, lines), long_prompt=0) == 0
    assert "NCCL log scan: no rank's log shows an NCCL communicator" in lines
    assert ("NCCL library activity without a communicator: rank 2, 1 line(s), first: " + exit_line) in lines
    assert any("-e 'NCCL INFO' -e 'NCCL WARN'" in command for _, command, _ in fake.calls)
    scan, problems = checks.nccl_log_findings(
        {0: [exit_line, "x:1:1 [0] NCCL INFO comm 0x1 rank 0 nRanks 8 - Init COMPLETE"], 1: []}, debug=True)
    assert problems == ["rank 0: its log shows 1 NCCL communicator line(s), first: x:1:1 [0] NCCL INFO comm 0x1 "
                        "rank 0 nRanks 8 - Init COMPLETE"]
    assert scan == ["NCCL library activity without a communicator: rank 0, 1 line(s), first: " + exit_line]
    ring = make_bundle(nccl_mode="never", require_no_nccl=True)
    assert ring.to_json()["nccl_free"]["library_patterns"] == ["NCCL INFO", "NCCL WARN"]
    names = bundle.container_names(["glm53-r{rank}"], ring)
    out: list[str] = []
    assert bundle.check(ring, names, NcclBundleSparks(ring), out.append) == 0
    assert "NCCL log scan: no rank's log shows an NCCL communicator" in out


# -- measured tuning tables ------------------------------------------------------------------------------


def _session_layout(positions, parent=None):
    from sparkring_sircl.vllm import fabric

    return fabric.describe_group(fabric.Layout.ring(8), list(positions),
                                 parent=list(parent) if parent is not None else None).session_layout()


def _tuning_table(tmp_path, name, positions, *, parent=None, rows=None, session=None, **key):
    """A tuning table measured on the group of ``positions`` of the ring of eight with this tree's build, in a
    session whose stats() fields are ``session``."""
    from sparkring_sircl import tuning

    facts = tuning.facts_for_layout(_session_layout(positions, parent), 2)
    rows = rows or [{"collective": "all_reduce", "mode": "eager", "bytes": 4096, "choice": {"algorithm": "oneshot"},
                     "p50_us": 20.0}]
    document = tuning.build_document({**facts, "image": "sha256:test", **key}, rows, session=session)
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(document, indent=1), encoding="utf-8")
    return path, tuning.document_hash(document), document


PAIR_ROWS = [{"collective": "all_reduce", "mode": mode, "bytes": size, "choice": choice, "p50_us": micros}
             for mode in ("eager", "graph")
             for size, choice, micros in ((4096, {"algorithm": "oneshot"}, 20.0), (4096, {"backend": "nccl"}, 15.0),
                                          (1 << 20, {"algorithm": "twoshot"}, 100.0),
                                          (1 << 20, {"backend": "nccl"}, 150.0))]


def test_tuning_tables_are_matched_to_each_session_staged_and_named_in_the_plan(repository, tmp_path, capsys,
                                                                                site_file):
    path4, path_hash, _ = _tuning_table(tmp_path, "path4", range(4))
    plan = make_plan(repository, tuning_tables=(str(path4),))
    container = f"/sircl/run/tuning/{path_hash}.json"
    assert {launch.environment["SIRCL_TUNING_TABLE"] for launch in plan.ranks} == {container}
    assert {change.name: change.reason for change in plan.changes}["SIRCL_TUNING_TABLE"].startswith(
        "--tuning-table: the measured tuning tables staged in the run directory")
    (table,) = plan.tuning.tables
    assert (table.hash, table.sessions, table.container_path) == (path_hash, ("tp",), container)
    assert table.host_path(plan.run_dir) == f"{plan.run_dir}/tuning/{path_hash}.json"
    assert table.sha256 == hashlib.sha256(path4.read_bytes()).hexdigest()
    assert plan.tuning.expected() == {"tp": path_hash, "ep": path_hash}       # the path: EP shares TP's session
    text = plan_mod.render_text(plan)
    assert f"  tuning tables (SIRCL_TUNING_TABLE={container}):" in text
    assert (f"    table {path_hash} from {path4}: path:4, 4 ranks, 2 lanes, 2 relays, native "
            f"{table.key['native']}") in text
    assert (f"    tensor-parallel session (tp): table {path_hash}; SIRCL carries every size: NCCL may not run on this "
            "group") in text
    assert "      all_reduce eager: from 4,096 B oneshot" in text
    assert "    ep: shares the tensor-parallel session and its table" in text
    record = json.loads(json.dumps(plan.to_json()))["tuning"]
    assert record["tables"][0]["container_path"] == container and record["sessions"][0]["table"] == path_hash
    assert record["sessions"][0]["facts"]["shape"] == "path:4" and record["expected"] == {"tp": path_hash,
                                                                                            "ep": path_hash}
    # Without a table the plan says the rules choose; the same table named twice is staged once.
    assert "  tuning tables: none (--tuning-table); every session's rules choose" in plan_mod.render_text(
        make_plan(repository))
    assert "SIRCL_TUNING_TABLE" not in make_plan(repository).ranks[0].environment
    assert len(make_plan(repository, tuning_tables=(str(path4), str(path4))).tuning.tables) == 1
    # A pair under --nccl auto and its other name topology, where NCCL may run by the rules: the table's NCCL
    # marks (NCCL measured faster at 4 KiB) are listed as measurements and route no call.
    pair_table, pair_hash, document = _tuning_table(tmp_path, "pair", (0, 1), rows=PAIR_ROWS)
    eager = next(entry for entry in document["decisions"]
                 if entry["collective"] == "all_reduce" and entry["mode"] == "eager")["intervals"]
    assert eager[0]["nccl"] and eager[0]["from"] == 4096
    for mode in ("auto", "topology"):
        pair = make_plans(repository, ((0, 1),), tuning_tables=(str(pair_table),), nccl_mode=mode)[0]
        text = plan_mod.render_text(pair)
        assert (f"    tensor-parallel session (tp): table {pair_hash}; the table chooses among SIRCL options only: "
                "its NCCL marks are measurements and route no call; the rules decide what NCCL carries here") in text
        assert "      all_reduce eager: from 4,096 B oneshot (NCCL faster)" in text
        assert "  NCCL: opt-in only (auto); tables choose among SIRCL options; this launch: --nccl auto" in text
        assert "eager calls go to NCCL where the table" not in text and "the table decides stay" not in text
        assert pair.tuning.expected() == {"tp": pair_hash}                    # a pair's EP group runs on NCCL
        assert json.loads(json.dumps(pair.to_json()))["tuning"]["sessions"][0]["nccl"] == (
            "the table chooses among SIRCL options only: its NCCL marks are measurements and route no call; the "
            "rules decide what NCCL carries here")
    never = make_plans(repository, ((0, 1),), tuning_tables=(str(pair_table),), nccl_mode="never")[0]
    assert "SIRCL carries every size: NCCL may not run on this group" in plan_mod.render_text(never)
    kept = make_plans(repository, ((0, 1),), tuning_tables=(str(pair_table),), large_allreduce="sircl",
                      nccl_mode="auto")[0]
    assert "the table chooses among SIRCL options only" in plan_mod.render_text(kept)
    # Every serve command takes the option; --env may not set the variable.
    for name in cli.COMMANDS:
        args = cli.parser().parse_args([name, "--site", "s", "--repository", "r", "--tuning-table", "a.json",
                                        "--tuning-table", "b.json"])
        assert args.tuning_table == ["a.json", "b.json"], name
    with pytest.raises(ServePlanError, match=re.escape("--env SIRCL_TUNING_TABLE: the launcher owns every SIRCL_* "
                                                      "variable; use --tuning-table")):
        make_plan(repository, extra_env={"SIRCL_TUNING_TABLE": "/x.json"})


def test_a_tables_settings_reach_its_sessions_and_smaller_launcher_options_are_refused(repository, tmp_path):
    """A table records the settings its choices ran under: the plan names them, the sessions that take the
    table apply them (the launcher sets no variable for them), a launcher option below them is refused in
    profile serving and in a bundle, and the session checks count the table's link slots."""
    rows = [{"collective": "all_gather", "mode": "eager", "bytes": 1 << 20,
             "choice": {"schedule": "ring", "piece": 2 << 20, "gather_stagger": 1}, "p50_us": 400.0}]
    session = {"link_slots": 12, "link_slot_bytes": 2 << 20}
    path4, _, document = _tuning_table(tmp_path, "path4", range(4), rows=rows, session=session)
    assert document["settings"] == {"SIRCL_LINK_SLOTS": 12, "SIRCL_LINK_SLOT_BYTES": 2 << 20}
    plan = make_plan(repository, tuning_tables=(str(path4),))
    assert "SIRCL_LINK_SLOTS" not in plan.ranks[0].environment
    assert "SIRCL_LINK_SLOT_BYTES" not in plan.ranks[0].environment
    assert ("      settings its sessions apply where the launcher leaves them unset: SIRCL_LINK_SLOTS=12, "
            "SIRCL_LINK_SLOT_BYTES=2,097,152") in plan_mod.render_text(plan)
    assert json.loads(json.dumps(plan.to_json()))["tuning"]["tables"][0]["settings"] == document["settings"]
    with pytest.raises(ServePlanError, match=re.escape("its choices need more than SIRCL_LINK_SLOTS=8 (the table's "
                                                       "12) (--link-slots)")):
        make_plan(repository, tuning_tables=(str(path4),), link_slots=8)
    with pytest.raises(ServePlanError, match=re.escape("SIRCL_LINK_SLOT_BYTES=1048576 (the table's 2097152) "
                                                       "(--link-slot)")):
        make_plan(repository, tuning_tables=(str(path4),), link_sizes={"link_slot": 1 << 20})
    assert make_plan(repository, tuning_tables=(str(path4),), link_slots=16).ranks[0].environment[
        "SIRCL_LINK_SLOTS"] == "16"
    # The session checks count the table's 12 link slots: an all-gather stagger of 3 on 4 ranks needs 11.
    with pytest.raises(ServePlanError, match="--ring-gather-stagger 3"):
        make_plan(repository, ring_gather_stagger=3)
    make_plan(repository, tuning_tables=(str(path4),), ring_gather_stagger=3)
    ring, _, _ = _tuning_table(tmp_path, "ring8", range(8), rows=rows, session=session)
    bundled = make_bundle(nccl_mode="never", tuning_tables=(str(ring),))
    assert "SIRCL_LINK_SLOTS" not in bundled.ranks[0].environment
    assert "link slots 12 (the tuning table's)" in bundled.render_text()
    with pytest.raises(ServePlanError, match=re.escape("SIRCL_LINK_SLOTS=10 (the table's 12) (--link-slots)")):
        make_bundle(nccl_mode="never", tuning_tables=(str(ring),), link_slots=10)


def test_tuning_tables_that_no_session_takes_or_that_compete_are_refused(repository, tmp_path):
    ring, _, _ = _tuning_table(tmp_path, "ring8", range(8))
    with pytest.raises(ServePlanError, match=re.escape(f"--tuning-table {ring} (table ")) as raised:
        make_plan(repository, tuning_tables=(str(ring),))
    assert ("matches no session of this launch (tensor-parallel session: shape: table 'cycle:8', here 'path:4', "
            "world: table 8, here 4") in str(raised.value)
    other_build, _, _ = _tuning_table(tmp_path, "build", range(4), kernels="0" * 16)
    with pytest.raises(ServePlanError, match=re.escape("kernels: table '0000000000000000', here ")):
        make_plan(repository, tuning_tables=(str(other_build),))
    first, first_hash, _ = _tuning_table(tmp_path, "first", range(4))
    rows = [{"collective": "all_gather", "mode": "graph", "bytes": 8192, "choice": {"schedule": "pieces"},
             "p50_us": 30.0}]
    second, second_hash, _ = _tuning_table(tmp_path, "second", range(4), rows=rows)
    with pytest.raises(ServePlanError, match=re.escape(f"--tuning-table: tables {first} ({first_hash}), {second} "
                                                       f"({second_hash}) all match the tensor-parallel session")):
        make_plan(repository, tuning_tables=(str(first), str(second)))
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({"schema": "sircl-tuning-table/v0", "key": {}}), encoding="utf-8")
    with pytest.raises(ServePlanError, match=re.escape(f"--tuning-table {broken}: tuning table schema must be "
                                                       "sircl-tuning-table/v1")):
        make_plan(repository, tuning_tables=(str(broken),))
    with pytest.raises(ServePlanError, match=re.escape(f"--tuning-table {tmp_path / 'missing.json'}: ")):
        make_plan(repository, tuning_tables=(str(tmp_path / "missing.json"),))
    (tmp_path / "text.json").write_text("not json", encoding="utf-8")
    with pytest.raises(ServePlanError, match=re.escape("--tuning-table ")):
        make_plan(repository, tuning_tables=(str(tmp_path / "text.json"),))


def test_a_bundle_matches_the_tp_and_dcp_sessions_and_stage_writes_every_table(tmp_path, site_file, capsys):
    ring, ring_hash, _ = _tuning_table(tmp_path, "ring8", range(8))
    dcp, dcp_hash, _ = _tuning_table(tmp_path, "dcp4", range(4), parent=range(8))
    plan = make_bundle(run_id="glm53-tp8", session_groups="tp,dcp", dcp_size=4, nccl_mode="never",
                       tuning_tables=(str(ring), str(dcp)))
    paths = {table.hash: table.container_path for table in plan.tuning.tables}
    assert plan.ranks[5].environment["SIRCL_TUNING_TABLE"] == f"{paths[ring_hash]},{paths[dcp_hash]}"
    assert plan.tuning.expected() == {"tp": ring_hash, "dcp": dcp_hash, "ep": ring_hash}
    text = plan.render_text()
    assert f"    tensor-parallel session (tp): table {ring_hash}; SIRCL carries every size" in text
    assert f"    decode-context-parallel sessions (dcp): table {dcp_hash}; SIRCL carries every size" in text
    document = json.loads(json.dumps(plan.to_json()))
    assert [row["sessions"] for row in document["tuning"]["tables"]] == [["tp"], ["dcp"]]
    assert document["tuning"]["tables"][1]["host_path"] == f"/tmp/sircl-ring/serve/runs/glm53-tp8/tuning/{dcp_hash}.json"
    assert any(item.startswith("the tuning tables at the host paths of tuning.tables") for item in document["requirements"])
    # A table measured on a TP4 path has the key of a DCP 4 group of the ring of eight: path:4, 4 ranks, 2 relays.
    same, same_hash, _ = _tuning_table(tmp_path, "tp4", range(4))
    assert same_hash == dcp_hash
    # Without DCP sessions the DCP table matches nothing; where --nccl auto (or topology) lets NCCL's ring run on
    # the whole ring, the TP table still chooses among SIRCL options only.
    with pytest.raises(ServePlanError, match="matches no session of this launch"):
        make_bundle(tuning_tables=(str(ring), str(dcp)))
    for mode in ("auto", "topology"):
        ringed = make_bundle(tuning_tables=(str(ring),), nccl_mode=mode)
        assert ringed.tuning.expected() == {"tp": ring_hash} and ringed.nccl_policy.value == "ring"
        assert (f"    tensor-parallel session (tp): table {ring_hash}; the table chooses among SIRCL options only: "
                "its NCCL marks are measurements and route no call") in ringed.render_text()
    fake = BundleSparks(plan)
    lines: list[str] = []
    assert bundle.stage(plan, staging.staged_tree(), fake, lines.append) == 0
    writes = [(target, command, data) for target, command, data in fake.calls if "sha256sum" in command]
    assert len(writes) == 16 and {data for _, _, data in writes} == {ring.read_bytes(), dcp.read_bytes()}
    assert all("/runs/glm53-tp8/tuning/" in command for _, command, _ in writes)
    assert sum(line.endswith(f": tuning tables {ring_hash}, {dcp_hash}") for line in lines) == 8
    assert cli.main(["bundle", "--site", str(site_file), "--positions", "0-7", "--session-groups", "tp,dcp",
                     "--dcp-size", "4", "--nccl", "never", "--tuning-table", str(ring), "--tuning-table", str(dcp),
                     "--text"]) == 0
    assert f"table {dcp_hash}" in capsys.readouterr().out
    args = cli.parser().parse_args(["bundle-check", "--site", "s", "--positions", "0-7", "--container", "x-{rank}",
                                    "--tuning-table", "t.json", "--session-groups", "tp,dcp", "--dcp-size", "4"])
    assert (args.tuning_table, args.session_groups, args.dcp_size) == (["t.json"], "tp,dcp", 4)


def test_start_writes_every_tuning_table_before_any_container_starts(repository, tmp_path):
    path4, path_hash, _ = _tuning_table(tmp_path, "path4", range(4))
    plan = make_plan(repository, tuning_tables=(str(path4),))
    fake = FakeSparks(plan)
    assert cli.start(context(plan, fake, [])) == 0
    first_run = next(index for index, (_, command, _) in enumerate(fake.calls) if " run -d " in command)
    writes = [(target, data) for target, command, data in fake.calls[:first_run]
              if f"{plan.run_dir}/tuning/{path_hash}.json" in command and "sha256sum" in command]
    assert [target for target, _ in writes] == [launch.ssh for launch in plan.ranks]
    assert {data for _, data in writes} == {path4.read_bytes()}
    assert any(f"tuning/{path_hash}.json" in line and "< tuning table" in line for line in commands.preview(plan))


def test_check_reports_each_sessions_tuning_from_the_receipts(repository, tmp_path):
    path4, path_hash, _ = _tuning_table(tmp_path, "path4", range(4))
    plan = make_plan(repository, tuning_tables=(str(path4),))
    stats = {"tuning": {"table": path_hash, "path": f"/sircl/run/tuning/{path_hash}.json",
                        "key": {"shape": "path:4", "world": 4, "lanes": 2, "max_relays": 2},
                        "decisions": {"all_reduce/eager/oneshot": 12}, "unusable": {"all_gather/graph/ring": 2},
                        "unmatched": {}}}

    def receipts(table):
        def build(rank):
            fake = FakeSparks(plan)
            return [fake.receipt(rank, tuning=table, session_stats=stats if table else {}),
                    fake.receipt(rank, group="ep:0", session="shared:tp:0", tuning=table,
                                 session_stats=stats if table else {})]
        return build

    lines: list[str] = []
    assert cli.check(context(plan, FakeSparks(plan, receipts=receipts(path_hash)), lines), long_prompt=0) == 0
    assert (f"tuning, rank 2 group tp:0: table {path_hash}; key path:4, 4 ranks, 2 lanes, 2 relays; decisions "
            "all_reduce/eager/oneshot=12; choices the session could not run (the rules carried them) "
            "all_gather/graph/ring=2") in lines
    lines.clear()
    assert cli.check(context(plan, FakeSparks(plan, receipts=receipts(None)), lines), long_prompt=0) == 1
    assert f"PROBLEM: rank 0 group tp:0: tuning table none (the rules choose), the plan matched {path_hash}" in lines
    found, problems = checks.tuning_findings({0: [{"group": "dcp:0", "tuning": "f" * 16}]}, {"tp": None})
    assert problems == [] and found == [f"tuning, rank 0 group dcp:0: table {'f' * 16}; decisions none yet"]
    found, problems = checks.tuning_findings(
        {1: [{"group": "tp:0", "session_stats": {"tuning": {"table": None, "unmatched": {"/t.json": ["shape: x"]}}}}]},
        {"tp": None})
    assert problems == [] and found == ["tuning, rank 1 group tp:0: rules; decisions none yet; tables that do not "
                                        "match it /t.json (shape: x)"]


def test_the_ring_gather_stagger_reaches_the_tensor_parallel_session_and_is_checked_against_its_slots(
        repository, site_file, capsys):
    from sparkring_sircl import protocol

    plan = make_bundle(session_groups="tp,dcp", dcp_size=4, large_schedule="ring", link_slots=16,
                       ring_gather_stagger=2)
    assert plan.ranks[0].environment["SIRCL_RING_GATHER_STAGGER"] == "2"
    tp, dcp = plan.sessions()
    assert (tp.ring_stagger, tp.ring_gather_stagger, tp.ring_gather_stagger_given) == (1, 2, True)
    assert (dcp.ring_gather_stagger, dcp.ring_gather_stagger_given) == (1, False)   # defaults: 2 x 3 + 2 <= 8 slots
    text = plan.render_text()
    assert ("link slots 16 (--link-slots); ring staggers: reduce-scatter 1 (the session's default), all-gather 2 "
            "(--ring-gather-stagger); ") in text
    sessions = json.loads(json.dumps(plan.to_json()))["sessions"]
    assert (sessions[0]["ring_stagger"], sessions[0]["ring_gather_stagger"]) == (1, 2)
    # The default needs 1 x 7 + 2 = 9 slots on the ring of eight, which its default 16 slots (twice its ranks)
    # hold; with 8 slots both staggers are 0.
    default = make_bundle(large_schedule="ring").sessions()[0]
    assert (default.link_slots, default.ring_stagger, default.ring_gather_stagger) == (16, 1, 1)
    assert "link slots 16 (the session's default)" in make_bundle(large_schedule="ring").render_text()
    eight = make_bundle(large_schedule="ring", link_slots=8).sessions()[0]
    assert (eight.ring_stagger, eight.ring_gather_stagger) == (0, 0)
    assert protocol.ring_stagger_slots(8, 1) == 9
    assert [protocol.default_link_slots(world) for world in (2, 4, 8, 16, 32)] == [8, 8, 16, 32, 32]
    with pytest.raises(ServePlanError, match=re.escape("--ring-gather-stagger 2: the tp session's 8 ranks need 16 "
                                                       "link slots for it, and it has 12 (--link-slots)")):
        make_bundle(large_schedule="ring", link_slots=12, ring_gather_stagger=2)
    for value, message in ((5, "--ring-gather-stagger must be 0 to 4, got 5"), (-1, "got -1"),
                           (True, "--ring-gather-stagger must be a number of rounds")):
        with pytest.raises(ServePlanError, match=re.escape(message)):
            make_bundle(ring_gather_stagger=value)
    path = make_plan(repository, ring_gather_stagger=0)
    assert path.ranks[1].environment["SIRCL_RING_GATHER_STAGGER"] == "0"
    assert path.to_json()["sizes"]["ring_gather_stagger"] == 0
    assert {change.name: change.reason for change in path.changes}["SIRCL_RING_GATHER_STAGGER"].startswith(
        "--ring-gather-stagger: rounds the tensor-parallel session's ring all-gather")
    with pytest.raises(ServePlanError, match=re.escape("use --ring-gather-stagger")):
        make_plan(repository, extra_env={"SIRCL_RING_GATHER_STAGGER": "1"})
    assert cli.main(["bundle", "--site", str(site_file), "--positions", "0-7", "--link-slots", "16",
                     "--ring-gather-stagger", "2", "--text"]) == 0
    assert "all-gather 2 (--ring-gather-stagger)" in capsys.readouterr().out
    for name in cli.COMMANDS:
        assert cli.parser().parse_args([name, "--site", "s", "--repository", "r", "--ring-gather-stagger",
                                        "1"]).ring_gather_stagger == 1
