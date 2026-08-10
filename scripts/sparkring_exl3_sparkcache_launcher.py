#!/usr/bin/env python3
"""Confirmation-gated EXL3/DCP4 SparkCache candidate orchestrator."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import shlex
import sys
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import connector_bundle_manifest as bundle  # noqa: E402
import exl3_sparkcache_config as config  # noqa: E402
import exl3_verified_start as verified_start  # noqa: E402
import sparkring_exl3_launcher as exl3  # noqa: E402
import sparkring_exl3_lmcache_launcher as lmcache  # noqa: E402
from sparkcache_patch_semantic_attest import (  # noqa: E402
    attest_kv_output_aggregator_source,
    attest_sources,
)
from sparkring_site import SiteConfigError, load_site  # noqa: E402


PLAN_SCHEMA = "sparkring-exl3-sparkcache-launch-plan/v1"
CANDIDATE_COMPONENT = "sparkcache-engine"
CANDIDATE_DIGEST_LABEL = "org.sparkring.sparkcache-candidate-sha256"
BUNDLE_LABEL = "org.sparkring.sparkcache-bundle-sha256"
CONFIRMATIONS = {
    "status": "ATTEST-EXL3-SPARKCACHE-STATUS-ALL-FOUR",
    "cutover": "STOP-EXL3-LMCACHE-START-SPARKCACHE-ALL-FOUR",
    "restart-engines": "RESTART-EXL3-SPARKCACHE-ENGINES-ALL-FOUR",
    "restart-stack": "RECREATE-EXL3-SPARKCACHE-STACK-ALL-FOUR",
    "rollback": "STOP-SPARKCACHE-RESTORE-EXL3-LMCACHE-ALL-FOUR",
}


class LauncherError(ValueError):
    """A local candidate, lifecycle, or execution-contract failure."""


def _emit(document: dict, output: str | None) -> None:
    payload = json.dumps(document, indent=2) + "\n"
    if output is None:
        print(payload, end="")
        return
    path = Path(output)
    if not path.parent.is_dir():
        raise LauncherError(f"output parent does not exist: {path.parent}")
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise LauncherError(f"refusing to overwrite output: {path}") from error
    print(
        json.dumps(
            {
                "output": str(path),
                "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            }
        )
    )


def canonical_sha256(document: dict) -> str:
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def shell_action(rank, script: str) -> exl3.RemoteAction:
    return exl3.RemoteAction(rank.id, rank.ssh_target, ("sh", "-lc", script))


def production_link_attest(rank) -> str:
    interfaces = [port.interface for port in rank.ring_ports]
    if len(interfaces) != 2 or len(set(interfaces)) != 2:
        raise LauncherError(f"rank {rank.id} must define exactly two ring interfaces")
    checks = []
    for interface in interfaces:
        root = f"/sys/class/net/{interface}"
        checks.extend(
            (
                f"test \"$(cat {shlex.quote(root + '/carrier')})\" = 1",
                f"test \"$(cat {shlex.quote(root + '/operstate')})\" = up",
            )
        )
    return " && ".join(checks)


def _candidate_inputs(candidate: dict) -> tuple[str, str, str]:
    mounts = candidate.get("required_mounts")
    if not isinstance(mounts, list) or len(mounts) != 2:
        raise LauncherError("candidate requires exactly two mounts")
    by_destination = {
        item.get("destination"): item
        for item in mounts
        if isinstance(item, dict)
    }
    staging = by_destination.get(config.STAGING_DESTINATION)
    cache = by_destination.get(config.CACHE_DESTINATION)
    if staging is None or cache is None:
        raise LauncherError("candidate staging/cache mount contract is incomplete")
    if staging.get("read_only") is not True or cache.get("read_only") is not False:
        raise LauncherError("candidate mount access modes are wrong")
    return (
        config._absolute_posix(staging.get("source"), "connector staging source"),
        config._absolute_posix(cache.get("source"), "cache root source"),
        candidate.get("connector_bundle_identity_sha256", ""),
    )


def validate_candidate(candidate: dict, baseline: exl3.Profile) -> dict:
    if baseline.engine != "docker":
        raise LauncherError(
            "SparkCache candidate orchestration currently requires engine=docker"
        )
    if candidate.get("schema") != config.SCHEMA:
        raise LauncherError("wrong SparkCache candidate schema")
    if candidate.get("execution_supported") is not False:
        raise LauncherError("candidate must retain direct execution_supported=false")
    staging, cache_root, bundle_identity = _candidate_inputs(candidate)
    profile_doc = candidate.get("profile")
    if not isinstance(profile_doc, dict):
        raise LauncherError("candidate profile must be an object")
    receipt = candidate.get("checkpoint_receipt")
    target_checkpoint = config.validate_checkpoint_receipt(receipt)
    args = profile_doc.get("extra_vllm_args", [])
    positions = [i for i, value in enumerate(args[:-1]) if value == "--kv-transfer-config"]
    if len(positions) != 1:
        raise LauncherError("candidate requires exactly one --kv-transfer-config")
    try:
        kv = json.loads(args[positions[0] + 1])
        configured_checkpoint = kv["kv_connector_extra_config"][
            "spark_cache_target_checkpoint_sha256"
        ]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise LauncherError(f"invalid candidate KV configuration: {error}") from error
    if configured_checkpoint != target_checkpoint:
        raise LauncherError(
            "candidate KV namespace is not the embedded checkpoint receipt identity"
        )
    expected = config.build_candidate(
        baseline.document,
        checkpoint_receipt=receipt,
        connector_bundle_identity=bundle_identity,
        connector_staging_host=staging,
        cache_root_host=cache_root,
    )
    if candidate != expected:
        raise LauncherError(
            "candidate is not the canonical transformation of the supplied baseline"
        )
    return {
        "profile": exl3.Profile(profile_doc),
        "staging": staging,
        "cache_root": cache_root,
        "bundle_identity": bundle_identity,
        "checkpoint_receipt": receipt,
        "target_checkpoint": target_checkpoint,
        "candidate_sha256": canonical_sha256(candidate),
    }


_BUNDLE_VERIFY = r"""
import hashlib,json,os,stat,sys
from pathlib import Path
root=Path(sys.argv[1])
expected=sys.argv[2]
required=json.loads(sys.argv[3])
domain=sys.argv[4]
entries=[]
for rel in required:
 p=root/rel
 before=os.lstat(p)
 assert stat.S_ISREG(before.st_mode) and not stat.S_ISLNK(before.st_mode), rel
 h=hashlib.sha256()
 with p.open('rb') as stream:
  while True:
   block=stream.read(1048576)
   if not block: break
   h.update(block)
 after=os.lstat(p)
 assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_mode)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_mode), rel
 entries.append((rel,before.st_size,h.hexdigest()))
actual=[]
def walk_error(error): raise error
for base,dirs,files in os.walk(root,followlinks=False,onerror=walk_error):
 for name in dirs:
  assert not stat.S_ISLNK(os.lstat(Path(base)/name).st_mode), name
 for name in files:
  actual.append((Path(base)/name).relative_to(root).as_posix())
assert sorted(actual)==required,(sorted(actual),required)
h=hashlib.sha256();h.update(domain.encode())
for rel,size,digest in entries:
 h.update(b'\0');h.update(rel.encode());h.update(b'\0');h.update(str(size).encode());h.update(b'\0');h.update(digest.encode())
assert h.hexdigest()==expected,(h.hexdigest(),expected)
""".strip()


_PATCH_ATTEST = (
    "import ast,inspect,textwrap\n"
    "class SemanticAttestationError(ValueError): pass\n"
    + inspect.getsource(attest_sources)
    + inspect.getsource(attest_kv_output_aggregator_source)
    + r"""
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.config.vllm import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
s1=inspect.getsource(Scheduler._handle_invalid_blocks)
vmm_names=('_validate_kv_transfer_vmm','_verify_kv_transfer_compat')
vmm_methods=[getattr(VllmConfig,name) for name in vmm_names if hasattr(VllmConfig,name)]
assert len(vmm_methods)==1
s2=inspect.getsource(vmm_methods[0])
attest_sources(s1,s2)
s3=inspect.getsource(KVOutputAggregator.aggregate)
attest_kv_output_aggregator_source(s3)
"""
).strip()


_CONNECTOR_ATTEST = r"""
import importlib.util
assert importlib.util.find_spec('spark_context_cache_connector') is not None
from spark_context_cache_connector import SparkContextCacheConnector
assert SparkContextCacheConnector.__name__=='SparkContextCacheConnector'
""".strip()


_CHECKPOINT_IDENTITY_ATTEST = r"""
import hashlib,json,os,stat,sys,unicodedata
from pathlib import Path,PurePosixPath
root=Path(sys.argv[1]);expected=sys.argv[2]
assert not root.is_symlink() and root.is_dir()
root=root.resolve();entries=[];seen=set();seen_dirs=set()
def walk_error(error): raise error
for base,dirs,files in os.walk(root,followlinks=False,onerror=walk_error):
 dirs.sort()
 for name in dirs:
  path=Path(base)/name;rel=unicodedata.normalize('NFC',PurePosixPath(*path.relative_to(root).parts).as_posix())
  mode=os.lstat(path).st_mode
  assert stat.S_ISDIR(mode) and not stat.S_ISLNK(mode) and rel not in seen_dirs,rel
  seen_dirs.add(rel)
 for name in sorted(files):
  path=Path(base)/name;rel=unicodedata.normalize('NFC',PurePosixPath(*path.relative_to(root).parts).as_posix())
  before=os.lstat(path)
  assert stat.S_ISREG(before.st_mode) and not stat.S_ISLNK(before.st_mode) and rel not in seen,rel
  seen.add(rel);h=hashlib.sha256();size=0
  with path.open('rb') as stream:
   while True:
    block=stream.read(1048576)
    if not block: break
    h.update(block);size+=len(block)
  after=os.lstat(path)
  assert (before.st_size,before.st_ino,before.st_dev,before.st_mtime_ns,before.st_ctime_ns)==(after.st_size,after.st_ino,after.st_dev,after.st_mtime_ns,after.st_ctime_ns)
  assert size==before.st_size
  entries.append({'rel_path':rel,'byte_size':size,'content_sha256':h.hexdigest()})
assert entries
entries.sort(key=lambda item:item['rel_path'])
receipt={'manifest_version':2,'path_normalization':'POSIX separators + Unicode NFC','file_count':len(entries),'files':entries}
canonical=json.dumps(receipt,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode('ascii')
h=hashlib.sha256();h.update(b'sparkcache-checkpoint-manifest-v2');h.update(b'\0');h.update(canonical)
assert h.hexdigest()==expected,(h.hexdigest(),expected)
""".strip()


_CHECKPOINT_LAYOUT_PRECHECK = r"""
import json,os,stat,sys,unicodedata
from pathlib import Path,PurePosixPath
root=Path(sys.argv[1]);receipt=json.loads(sys.argv[2])
assert receipt['checkpoint_identity_sha256']==sys.argv[3]
expected={item['rel_path']:item['byte_size'] for item in receipt['files']}
assert len(expected)==receipt['file_count'] and expected
assert not root.is_symlink() and root.is_dir()
root=root.resolve();actual={};seen_dirs=set()
def walk_error(error): raise error
for base,dirs,files in os.walk(root,followlinks=False,onerror=walk_error):
 dirs.sort()
 for name in dirs:
  path=Path(base)/name;rel=unicodedata.normalize('NFC',PurePosixPath(*path.relative_to(root).parts).as_posix())
  mode=os.lstat(path).st_mode
  assert stat.S_ISDIR(mode) and not stat.S_ISLNK(mode) and rel not in seen_dirs,rel
  seen_dirs.add(rel)
 for name in sorted(files):
  path=Path(base)/name;rel=unicodedata.normalize('NFC',PurePosixPath(*path.relative_to(root).parts).as_posix())
  item=os.lstat(path)
  assert stat.S_ISREG(item.st_mode) and not stat.S_ISLNK(item.st_mode) and rel not in actual,rel
  actual[rel]=item.st_size
assert actual==expected,(actual,expected)
""".strip()


_NO_MODEL_CONTAINERS_ATTEST = r"""
import json,os,subprocess,sys
model=os.path.realpath(sys.argv[1]);forbidden=set(sys.argv[2:])
def overlaps(left,right):
 try: common=os.path.commonpath((left,right))
 except ValueError: return False
 return common==left or common==right
ids=subprocess.check_output(['docker','ps','-q'],text=True).split()
for ident in ids:
 doc=json.loads(subprocess.check_output(['docker','inspect',ident]))[0]
 assert doc['Name'].removeprefix('/') not in forbidden,doc['Name']
 for mount in doc.get('Mounts') or []:
  source=os.path.realpath(mount['Source'])
  assert not overlaps(source,model),(doc['Name'],mount['Source'])
""".strip()


def _helper_name(rank_id: int) -> str:
    return f"glm52-sparkring-exl3-checkpoint-attest-r{rank_id}"


def precheck_actions(
    site, baseline: exl3.Profile, state: dict
) -> list[exl3.RemoteAction]:
    profile = state["profile"]
    candidate_contracts = {
        action.rank: docker_run_contract(action, profile.image_id)
        for action in candidate_start_actions(site, state)
    }
    baseline_contracts = {
        action.rank: docker_run_contract(action, baseline.image_id)
        for action in baseline_verified_start_actions(site, baseline)
    }
    required = json.dumps(sorted(bundle.REQUIRED_FILES), separators=(",", ":"))
    actions = []
    for rank in site.ranks:
        candidate_name = exl3.container_name(profile, rank.id)
        baseline_name = exl3.container_name(baseline, rank.id)
        verify_bundle = shlex.join(
            (
                "python3",
                "-c",
                _BUNDLE_VERIFY,
                state["staging"],
                state["bundle_identity"],
                required,
                bundle.BUNDLE_DOMAIN_SEPARATOR,
            )
        )
        candidate_contract_attest = shlex.join(
            (
                "python3",
                "-c",
                _FULL_CONTRACT_ATTEST,
                candidate_name,
                profile.image_id,
                json.dumps(
                    candidate_contracts[rank.id], sort_keys=True, separators=(",", ":")
                ),
            )
        )
        baseline_contract_attest = shlex.join(
            (
                "python3",
                "-c",
                _FULL_CONTRACT_ATTEST,
                baseline_name,
                baseline.image_id,
                json.dumps(
                    baseline_contracts[rank.id], sort_keys=True, separators=(",", ":")
                ),
            )
        )
        patch_attest = shlex.join(
            ("docker", "exec", "$engine", "/opt/venv/bin/python", "-c", _PATCH_ATTEST)
        ).replace("'$engine'", '"$engine"')
        connector_attest = shlex.join(
            (
                "docker",
                "exec",
                candidate_name,
                "/opt/venv/bin/python",
                "-c",
                _CONNECTOR_ATTEST,
            )
        )
        layout_precheck = shlex.join(
            (
                "python3",
                "-c",
                _CHECKPOINT_LAYOUT_PRECHECK,
                profile.model_host_path,
                json.dumps(state["checkpoint_receipt"], sort_keys=True, separators=(",", ":")),
                state["target_checkpoint"],
            )
        )
        script = (
            "docker info >/dev/null"
            f" && test \"$(docker image inspect --format '{{{{.Id}}}}' {shlex.quote(profile.image)})\" = {shlex.quote(profile.image_id)}"
            f" && test -d {shlex.quote(state['staging'])}"
            f" && test -d {shlex.quote(state['cache_root'])}"
            f" && test -w {shlex.quote(state['cache_root'])}"
            f" && {verify_bundle}"
            f" && if test \"$(docker inspect -f '{{{{.State.Running}}}}' {shlex.quote(candidate_name)} 2>/dev/null)\" = true; then "
            f"engine={shlex.quote(candidate_name)}; connector=1; {candidate_contract_attest}; "
            f"elif test \"$(docker inspect -f '{{{{.State.Running}}}}' {shlex.quote(baseline_name)} 2>/dev/null)\" = true; then "
            f"engine={shlex.quote(baseline_name)}; connector=0; {baseline_contract_attest}; "
            "else exit 69; fi"
            f" && {patch_attest}"
            f" && if test \"$connector\" = 1; then {connector_attest}; fi"
            f" && {layout_precheck}"
        )
        actions.append(shell_action(rank, script))
    return actions


def checkpoint_quiescent_actions(
    site, baseline: exl3.Profile, state: dict
) -> list[exl3.RemoteAction]:
    profile = state["profile"]
    actions = []
    for rank in site.ranks:
        names = (
            exl3.container_name(baseline, rank.id),
            exl3.container_name(profile, rank.id),
            _helper_name(rank.id),
        )
        container_attest = shlex.join(
            (
                "python3",
                "-c",
                _NO_MODEL_CONTAINERS_ATTEST,
                profile.model_host_path,
                *names,
            )
        )
        check = (
            f"{container_attest}"
            " && gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
            " && test -z \"$(printf '%s\\n' \"$gpu_pids\" | sed '/^[[:space:]]*$/d')\""
        )
        script = (
            "docker info >/dev/null"
            f" && for i in $(seq 1 60); do {check} && exit 0; sleep 1; done"
            f" && {check}"
        )
        actions.append(shell_action(rank, script))
    return actions


def full_checkpoint_attestation_actions(site, state: dict) -> list[exl3.RemoteAction]:
    profile = state["profile"]
    actions = []
    for rank in site.ranks:
        name = _helper_name(rank.id)
        command = shlex.join(
            (
                profile.engine,
                "run",
                "--rm",
                "--name",
                name,
                "--label",
                "org.sparkring.managed=true",
                "--label",
                f"org.sparkring.exl3-profile={profile.profile_id}",
                "--label",
                "org.sparkring.component=sparkcache-checkpoint-attestor",
                "--label",
                f"{CANDIDATE_DIGEST_LABEL}={state['candidate_sha256']}",
                "--label",
                f"org.sparkring.checkpoint-identity-sha256={state['target_checkpoint']}",
                "--network",
                "none",
                "--ipc",
                "none",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--pids-limit",
                "128",
                "--env",
                "PYTHONDONTWRITEBYTECODE=1",
                "--volume",
                f"{profile.model_host_path}:{profile.model_container_path}:ro",
                "--entrypoint",
                "/opt/venv/bin/python",
                profile.image_id,
                "-c",
                _CHECKPOINT_IDENTITY_ATTEST,
                profile.model_container_path,
                state["target_checkpoint"],
            )
        )
        script = (
            f"test \"$({profile.engine} image inspect --format '{{{{.Id}}}}' {shlex.quote(profile.image)})\" = {shlex.quote(profile.image_id)}"
            f" && exec {command}"
        )
        actions.append(shell_action(rank, script))
    return actions


def candidate_start_actions(site, state: dict) -> list[exl3.RemoteAction]:
    profile = state["profile"]
    actions = []
    for action in exl3.start_actions(site, profile):
        command = action.argv[-1]
        marker = f"{profile.image_id} serve "
        if marker not in command:
            raise LauncherError("cannot locate candidate image insertion point")
        additions = shlex.join(
            (
                "--privileged",
                "--label",
                f"org.sparkring.component={CANDIDATE_COMPONENT}",
                "--label",
                f"{BUNDLE_LABEL}={state['bundle_identity']}",
                "--label",
                f"{CANDIDATE_DIGEST_LABEL}={state['candidate_sha256']}",
                "--volume",
                f"{state['staging']}:{config.STAGING_DESTINATION}:ro",
                "--volume",
                f"{state['cache_root']}:{config.CACHE_DESTINATION}",
            )
        )
        command = command.replace(marker, f"{additions} {marker}", 1)
        actions.append(
            exl3.RemoteAction(action.rank, action.ssh_target, ("sh", "-lc", command))
        )
    # ``checkpoint_full_attest`` has already hashed and identity-bound this
    # exact read-only model view.  Remove the ordinary launcher's redundant
    # full verifier, mount the hash-attested entrypoint that skips only the
    # image's duplicate verifier, and reclaim the verifier's unified-memory
    # page cache immediately before Docker starts.
    actions = verified_start.without_embedded_model_verification(actions, profile)
    return verified_start.decorate_verified_start(actions, profile)


def docker_run_contract(action: exl3.RemoteAction, image: str) -> dict:
    """Recover the exact container contract from a generated guarded action."""
    command = action.argv[-1]
    if " && exec " not in command:
        raise LauncherError("generated action has no exec boundary")
    argv = shlex.split(command.rsplit(" && exec ", 1)[1])
    try:
        image_index = argv.index(image)
    except ValueError as error:
        raise LauncherError("generated action does not contain exact image ID") from error
    prefix = argv[:image_index]
    labels = {}
    mounts = {}
    environment = {}
    entrypoint = None
    # Complete operational HostConfig projection for the canonical command.
    # Every namespace, security, lifecycle, device, mount, network, resource,
    # and daemon-runtime control is bound; presentation-only fields such as
    # ConsoleSize are intentionally outside the projection.
    host_config = {
        "AutoRemove": False,
        "Binds": [],
        "BlkioDeviceReadBps": [],
        "BlkioDeviceReadIOps": [],
        "BlkioDeviceWriteBps": [],
        "BlkioDeviceWriteIOps": [],
        "BlkioWeight": 0,
        "BlkioWeightDevice": [],
        "CapAdd": [],
        "CapDrop": None,
        "Cgroup": "",
        "CgroupParent": "",
        "CgroupnsMode": "private",
        "ContainerIDFile": "",
        "CpuCount": 0,
        "CpuPercent": 0,
        "CpuPeriod": 0,
        "CpuQuota": 0,
        "CpuRealtimePeriod": 0,
        "CpuRealtimeRuntime": 0,
        "CpuShares": 0,
        "CpusetCpus": "",
        "CpusetMems": "",
        "DeviceCgroupRules": None,
        "DeviceRequests": [],
        "Devices": [],
        "Dns": [],
        "DnsOptions": [],
        "DnsSearch": [],
        "ExtraHosts": None,
        "GroupAdd": None,
        "IOMaximumBandwidth": 0,
        "IOMaximumIOps": 0,
        "Init": None,
        "IpcMode": None,
        "Isolation": "",
        "Links": None,
        "LogConfig": {"Type": "json-file", "Config": {}},
        "MaskedPaths": [],
        "Memory": 0,
        "MemoryReservation": 0,
        "MemorySwap": 0,
        "MemorySwappiness": None,
        "Mounts": None,
        "NanoCpus": 0,
        "NetworkMode": None,
        "OomKillDisable": None,
        "OomScoreAdj": 0,
        "PidMode": "",
        "PidsLimit": None,
        "PortBindings": {},
        "Privileged": False,
        "PublishAllPorts": False,
        "ReadonlyPaths": [],
        "ReadonlyRootfs": False,
        "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0},
        "Runtime": "runc",
        "SecurityOpt": None,
        "ShmSize": None,
        "StorageOpt": {},
        "Sysctls": None,
        "Tmpfs": None,
        "UTSMode": "",
        "Ulimits": [],
        "UsernsMode": "",
        "VolumeDriver": "",
        "VolumesFrom": None,
    }

    def bytes_value(raw: str) -> int:
        suffixes = {"k": 1024, "m": 1024**2, "g": 1024**3}
        if raw[-1:].lower() in suffixes:
            return int(raw[:-1]) * suffixes[raw[-1].lower()]
        return int(raw)

    index = 0
    while index < len(prefix):
        value = prefix[index]
        if value in ("--label", "--volume", "--env", "--entrypoint"):
            if index + 1 >= len(prefix):
                raise LauncherError(f"generated action has valueless {value}")
            supplied = prefix[index + 1]
            if value == "--label":
                name, separator, item = supplied.partition("=")
                if not separator or name in labels:
                    raise LauncherError("generated action has malformed/duplicate label")
                labels[name] = item
            elif value == "--env":
                name, separator, item = supplied.partition("=")
                if not separator or name in environment:
                    raise LauncherError("generated action has malformed/duplicate env")
                environment[name] = item
            elif value == "--volume":
                parts = supplied.split(":")
                if len(parts) not in (2, 3) or parts[1] in mounts:
                    raise LauncherError("generated action has malformed/duplicate mount")
                mounts[parts[1]] = {
                    "source": parts[0],
                    "read_only": len(parts) == 3 and parts[2] == "ro",
                }
                host_config["Binds"].append(supplied)
            else:
                entrypoint = supplied
            index += 2
            continue
        if value == "--privileged":
            host_config["Privileged"] = True
            index += 1
            continue
        elif value in (
            "--network",
            "--ipc",
            "--gpus",
            "--shm-size",
            "--ulimit",
            "--cap-add",
            "--device",
        ):
            if index + 1 >= len(prefix):
                raise LauncherError(f"generated action has valueless {value}")
            supplied = prefix[index + 1]
            if value == "--network":
                host_config["NetworkMode"] = supplied
            elif value == "--ipc":
                host_config["IpcMode"] = supplied
            elif value == "--gpus":
                if supplied != "all":
                    raise LauncherError("only the canonical --gpus all is supported")
                host_config["DeviceRequests"] = [
                    {
                        "Driver": "",
                        "Count": -1,
                        "DeviceIDs": None,
                        "Capabilities": [["gpu"]],
                        "Options": {},
                    }
                ]
            elif value == "--shm-size":
                host_config["ShmSize"] = bytes_value(supplied)
            elif value == "--ulimit":
                name, limits = supplied.split("=", 1)
                soft, hard = limits.split(":", 1)
                host_config["Ulimits"].append(
                    {"Name": name, "Soft": int(soft), "Hard": int(hard)}
                )
            elif value == "--cap-add":
                host_config["CapAdd"].append(supplied)
            else:
                parts = supplied.split(":")
                host_config["Devices"].append(
                    {
                        "PathOnHost": parts[0],
                        "PathInContainer": parts[1] if len(parts) > 1 else parts[0],
                        "CgroupPermissions": parts[2] if len(parts) > 2 else "rwm",
                    }
                )
            index += 2
            continue
        if value in ("docker", "run", "--detach"):
            index += 1
            continue
        if value == "--name":
            index += 2
            continue
        if value.startswith("--"):
            raise LauncherError(f"unprojected Docker run option {value}")
        index += 1
    return {
        "cmd": argv[image_index + 1 :],
        "environment_overrides": environment,
        "entrypoint_override": entrypoint,
        "labels": labels,
        "mounts": mounts,
        "host_config": host_config,
    }


def normalize_docker_host_config(actual: dict, expected: dict) -> dict:
    """Project Docker-inspect defaults onto their generated-run semantics."""
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        raise AssertionError("HostConfig must be an object")
    missing_inert_defaults = {
        "Init": None,
        "Mounts": None,
        "StorageOpt": {},
        "Sysctls": None,
        "Tmpfs": None,
    }
    projected = {}
    for key, value in expected.items():
        if key in actual:
            projected[key] = actual[key]
        elif key in missing_inert_defaults and value == missing_inert_defaults[key]:
            projected[key] = value
        else:
            raise AssertionError(f"HostConfig is missing non-inert field {key}")
    null_empty_lists = {
        "Binds",
        "CapAdd",
        "Dns",
        "MaskedPaths",
        "ReadonlyPaths",
    }
    for key in null_empty_lists:
        if expected.get(key) == [] and projected.get(key) is None:
            projected[key] = []
    actual_binds = projected.get("Binds")
    expected_binds = expected.get("Binds")
    if isinstance(actual_binds, list) and isinstance(expected_binds, list):
        # Docker may report bind mounts in a different order than ``docker
        # run`` received them.  Bind order has no runtime semantics; retain
        # exact multiset equality so changed sources, destinations, modes, or
        # duplicate counts still fail attestation.
        if sorted(actual_binds) == sorted(expected_binds):
            projected["Binds"] = expected_binds
    actual_caps = projected.get("CapAdd")
    expected_caps = expected.get("CapAdd")
    if isinstance(actual_caps, list) and isinstance(expected_caps, list):
        def strip_prefix(value):
            return (
                value[4:]
                if isinstance(value, str) and value.startswith("CAP_")
                else value
            )

        if [strip_prefix(value) for value in actual_caps] == [
            strip_prefix(value) for value in expected_caps
        ]:
            projected["CapAdd"] = expected_caps
    if expected.get("StorageOpt") == {} and projected.get("StorageOpt") is None:
        projected["StorageOpt"] = {}
    if (
        expected.get("Privileged") is True
        and expected.get("SecurityOpt") is None
        and projected.get("SecurityOpt") == ["label=disable"]
    ):
        projected["SecurityOpt"] = None
    return projected


_HOST_CONFIG_NORMALIZE = inspect.getsource(normalize_docker_host_config)


_STATUS_ATTEST = _HOST_CONFIG_NORMALIZE + r"""
import json,subprocess,sys
name,image,profile,bundle,candidate,model_src,model_dst,stage_src,stage_dst,cache_src,cache_dst,kv,contract_json=sys.argv[1:]
contract=json.loads(contract_json)
doc=json.loads(subprocess.check_output(['docker','inspect',name]))[0]
image_doc=json.loads(subprocess.check_output(['docker','image','inspect',image]))[0]
assert doc['State']['Running'] is True
assert doc['State']['OOMKilled'] is False and doc['RestartCount']==0
assert doc['Image']==image
labels=doc['Config']['Labels'] or {};expected_labels=dict(image_doc['Config'].get('Labels') or {});expected_labels.update(contract['labels']);assert labels==expected_labels
assert labels['org.sparkring.managed']=='true'
assert labels['org.sparkring.exl3-profile']==profile
assert labels['org.sparkring.component']=='sparkcache-engine'
assert labels['org.sparkring.sparkcache-bundle-sha256']==bundle
assert labels['org.sparkring.sparkcache-candidate-sha256']==candidate
mounts={m['Destination']:(m['Source'],m['RW']) for m in doc['Mounts']}
assert mounts[model_dst]==(model_src,False)
assert mounts[stage_dst]==(stage_src,False)
assert mounts[cache_dst]==(cache_src,True)
assert not (image_doc['Config'].get('Volumes') or {})
assert set(mounts)==set(contract['mounts'])
for destination,item in contract['mounts'].items(): assert mounts[destination]==(item['source'],not item['read_only'])
cmd=doc['Config']['Cmd'];indices=[i for i,x in enumerate(cmd[:-1]) if x=='--kv-transfer-config']
assert len(indices)==1 and cmd[indices[0]+1]==kv
assert cmd.count('--disable-hybrid-kv-cache-manager')==1
assert cmd==contract['cmd']
actual_items=doc['Config']['Env'] or [];env=dict(item.split('=',1) for item in actual_items if '=' in item)
assert len(env)==len(actual_items)
base_items=image_doc['Config'].get('Env') or [];expected=dict(item.split('=',1) for item in base_items if '=' in item)
expected.update(contract['environment_overrides'])
assert env==expected
expected_entrypoint=contract['entrypoint_override']
if expected_entrypoint is None:
 assert doc['Config'].get('Entrypoint')==image_doc['Config'].get('Entrypoint')
else:
 assert doc['Config'].get('Entrypoint')==[expected_entrypoint]
hc=doc['HostConfig'];eh=contract['host_config']
assert normalize_docker_host_config(hc,eh)==eh
assert 'SPARK_CONTEXT_CACHE_ENABLE' not in env
""".strip()


_FULL_CONTRACT_ATTEST = _HOST_CONFIG_NORMALIZE + r"""
import json,subprocess,sys
name,image,contract_json=sys.argv[1:];contract=json.loads(contract_json)
doc=json.loads(subprocess.check_output(['docker','inspect',name]))[0]
image_doc=json.loads(subprocess.check_output(['docker','image','inspect',image]))[0]
assert doc['State']['Running'] is True and doc['State']['OOMKilled'] is False and doc['RestartCount']==0
assert doc['Image']==image and doc['Config']['Cmd']==contract['cmd']
labels=doc['Config']['Labels'] or {};expected_labels=dict(image_doc['Config'].get('Labels') or {});expected_labels.update(contract['labels']);assert labels==expected_labels
mounts={m['Destination']:(m['Source'],m['RW']) for m in doc['Mounts']}
assert not (image_doc['Config'].get('Volumes') or {}) and set(mounts)==set(contract['mounts'])
for destination,item in contract['mounts'].items(): assert mounts[destination]==(item['source'],not item['read_only'])
actual_items=doc['Config']['Env'] or [];actual=dict(item.split('=',1) for item in actual_items if '=' in item);assert len(actual)==len(actual_items)
base_items=image_doc['Config'].get('Env') or [];expected=dict(item.split('=',1) for item in base_items if '=' in item);expected.update(contract['environment_overrides']);assert actual==expected
entrypoint=contract['entrypoint_override']
assert doc['Config'].get('Entrypoint')==(image_doc['Config'].get('Entrypoint') if entrypoint is None else [entrypoint])
hc=doc['HostConfig'];eh=contract['host_config']
assert normalize_docker_host_config(hc,eh)==eh
""".strip()


def baseline_final_status_actions(
    site, baseline: exl3.Profile, state: dict
) -> list[exl3.RemoteAction]:
    engines = {
        action.rank: (action, docker_run_contract(action, baseline.image_id))
        for action in baseline_verified_start_actions(site, baseline)
    }
    servers = {
        action.rank: (action, docker_run_contract(action, baseline.image_id))
        for action in lmcache.server_start_actions(site, baseline)
    }
    actions = []
    for rank in site.ranks:
        engine_name = exl3.container_name(baseline, rank.id)
        server_name = lmcache.server_name(rank.id)
        engine_contract = engines[rank.id][1]
        server_contract = servers[rank.id][1]
        engine_attest = shlex.join(
            (
                "python3",
                "-c",
                _FULL_CONTRACT_ATTEST,
                engine_name,
                baseline.image_id,
                json.dumps(engine_contract, sort_keys=True, separators=(",", ":")),
            )
        )
        server_attest = shlex.join(
            (
                "python3",
                "-c",
                _FULL_CONTRACT_ATTEST,
                server_name,
                baseline.image_id,
                json.dumps(server_contract, sort_keys=True, separators=(",", ":")),
            )
        )
        script = (
            f"{engine_attest} && {server_attest} && {production_link_attest(rank)}"
        )
        actions.append(shell_action(rank, script))
    return actions


def candidate_api_barrier_actions(site, state: dict) -> list[exl3.RemoteAction]:
    profile = state["profile"]
    attempts = max(1, int(profile.startup_timeout_seconds) // 5)
    actions = []
    for rank in site.ranks:
        name = exl3.container_name(profile, rank.id)
        if rank.id == site.serving.master_rank:
            readiness = (
                f"curl -fsS http://127.0.0.1:{site.serving.api_port}/health >/dev/null"
                f" && curl -fsS http://127.0.0.1:{site.serving.api_port}/v1/models >/dev/null"
            )
            script = (
                f"test \"$(docker inspect -f '{{{{.State.Running}}}}' {shlex.quote(name)})\" = true"
                f" && for i in $(seq 1 {attempts}); do {readiness} && break; sleep 5; done"
                f" && {readiness}"
            )
        else:
            script = (
                f"test \"$(docker inspect -f '{{{{.State.Running}}}}' {shlex.quote(name)})\" = true"
            )
        actions.append(shell_action(rank, script))
    return actions


def candidate_final_status_actions(
    site, baseline: exl3.Profile, state: dict
) -> list[exl3.RemoteAction]:
    profile = state["profile"]
    args = profile.extra_vllm_args
    kv = args[args.index("--kv-transfer-config") + 1]
    contracts = {
        action.rank: docker_run_contract(action, profile.image_id)
        for action in candidate_start_actions(site, state)
    }
    actions = []
    final_attest = {
        action.rank: action
        for action in precheck_actions(site, baseline, state)
    }
    for rank in site.ranks:
        name = exl3.container_name(profile, rank.id)
        inspect_cmd = shlex.join(
            (
                "python3",
                "-c",
                _STATUS_ATTEST,
                name,
                profile.image_id,
                profile.profile_id,
                state["bundle_identity"],
                state["candidate_sha256"],
                profile.model_host_path,
                profile.model_container_path,
                state["staging"],
                config.STAGING_DESTINATION,
                state["cache_root"],
                config.CACHE_DESTINATION,
                kv,
                json.dumps(contracts[rank.id], sort_keys=True, separators=(",", ":")),
            )
        )
        readiness = (
            f"curl -fsS http://127.0.0.1:{site.serving.api_port}/health >/dev/null"
            f" && curl -fsS http://127.0.0.1:{site.serving.api_port}/v1/models >/dev/null"
            if rank.id == site.serving.master_rank
            else "true"
        )
        script = (
            f"{final_attest[rank.id].argv[-1]}"
            f" && {inspect_cmd}"
            f" && {readiness}"
            f" && {production_link_attest(rank)}"
            f" && ! docker logs {shlex.quote(name)} 2>&1 | "
            "grep -E 'CUDA out of memory|OutOfMemoryError|OOMKilled'"
        )
        actions.append(shell_action(rank, script))
    return actions


def candidate_absent_actions(site, state: dict) -> list[exl3.RemoteAction]:
    profile = state["profile"]
    return [
        shell_action(
            rank,
            "docker info >/dev/null"
            f" && ids=$(docker ps -aq --filter name=^/{shlex.quote(exl3.container_name(profile, rank.id))}$)"
            " && test -z \"$ids\"",
        )
        for rank in site.ranks
    ]


def checkpoint_helper_absent_actions(site, state: dict) -> list[exl3.RemoteAction]:
    return [
        shell_action(
            rank,
            "docker info >/dev/null"
            f" && ids=$(docker ps -aq --filter name=^/{shlex.quote(_helper_name(rank.id))}$)"
            " && test -z \"$ids\"",
        )
        for rank in site.ranks
    ]


_HELPER_OWNERSHIP_ATTEST = r"""
import json,subprocess,sys
name,image,profile,candidate,checkpoint,model_src,model_dst=sys.argv[1:]
doc=json.loads(subprocess.check_output(['docker','inspect',name]))[0]
assert doc['Image']==image
labels=doc['Config']['Labels'] or {}
assert labels['org.sparkring.managed']=='true'
assert labels['org.sparkring.exl3-profile']==profile
assert labels['org.sparkring.component']=='sparkcache-checkpoint-attestor'
assert labels['org.sparkring.sparkcache-candidate-sha256']==candidate
assert labels['org.sparkring.checkpoint-identity-sha256']==checkpoint
mounts={m['Destination']:(m['Source'],m['RW']) for m in doc['Mounts']}
assert mounts=={model_dst:(model_src,False)}
""".strip()


_EXACT_REMOVAL_ATTEST = _HOST_CONFIG_NORMALIZE + r"""
import json,subprocess,sys
name,image,contract_json=sys.argv[1:];contract=json.loads(contract_json)
doc=json.loads(subprocess.check_output(['docker','inspect',name]))[0]
image_doc=json.loads(subprocess.check_output(['docker','image','inspect',image]))[0]
assert doc['Name']=='/'+name and doc['Image']==image and doc['Config']['Cmd']==contract['cmd']
labels=doc['Config']['Labels'] or {};expected_labels=dict(image_doc['Config'].get('Labels') or {});expected_labels.update(contract['labels']);assert labels==expected_labels
mount_items=doc.get('Mounts') or [];mounts={m['Destination']:(m['Source'],m['RW']) for m in mount_items};assert len(mounts)==len(mount_items)
assert not (image_doc['Config'].get('Volumes') or {}) and set(mounts)==set(contract['mounts'])
for destination,item in contract['mounts'].items(): assert mounts[destination]==(item['source'],not item['read_only'])
actual_items=doc['Config'].get('Env') or [];actual=dict(item.split('=',1) for item in actual_items if '=' in item);assert len(actual)==len(actual_items)
base_items=image_doc['Config'].get('Env') or [];expected=dict(item.split('=',1) for item in base_items if '=' in item);expected.update(contract['environment_overrides']);assert actual==expected
entrypoint=contract['entrypoint_override'];assert doc['Config'].get('Entrypoint')==(image_doc['Config'].get('Entrypoint') if entrypoint is None else [entrypoint])
hc=doc['HostConfig'];eh=contract['host_config']
assert normalize_docker_host_config(hc,eh)==eh
ident=doc.get('Id');assert isinstance(ident,str) and ident
print(ident)
""".strip()


def baseline_exact_remove_actions(
    site, baseline: exl3.Profile, *, component: str
) -> list[exl3.RemoteAction]:
    """Remove only the exact generated baseline container, by attested ID."""
    if component == "engine":
        # Canonical engines are restored through the shared verified-start
        # transaction (outer entrypoint + page-cache reclaim).  Exact removal
        # must attest that actual contract, not the older undecorated command.
        generated = baseline_verified_start_actions(site, baseline)
        names = {
            rank.id: exl3.container_name(baseline, rank.id) for rank in site.ranks
        }
    elif component == "lmcache-server":
        generated = lmcache.server_start_actions(site, baseline)
        names = {rank.id: lmcache.server_name(rank.id) for rank in site.ranks}
    else:
        raise LauncherError(f"unsupported baseline removal component {component}")
    contracts = {
        action.rank: docker_run_contract(action, baseline.image_id)
        for action in generated
    }
    actions = []
    for rank in site.ranks:
        name = names[rank.id]
        attest = shlex.join(
            (
                "python3",
                "-c",
                _EXACT_REMOVAL_ATTEST,
                name,
                baseline.image_id,
                json.dumps(contracts[rank.id], sort_keys=True, separators=(",", ":")),
            )
        )
        script = (
            "docker info >/dev/null || exit 70; "
            f"name={shlex.quote(name)}; "
            "if ! docker inspect \"$name\" >/dev/null 2>&1; then "
            "ids=$(docker ps -aq --filter name=^/$name$) || exit 71; "
            "test -z \"$ids\" || exit 77; exit 0; fi; "
            f"ident=$({attest}) || exit 78; "
            "test -n \"$ident\" || exit 79; "
            "exec docker rm --force \"$ident\""
        )
        actions.append(shell_action(rank, script))
    return actions


def baseline_verified_start_actions(
    site, baseline: exl3.Profile
) -> list[exl3.RemoteAction]:
    """Restore canonical LMCache engines with one verifier and cache reclaim."""
    starts = lmcache.engine_start_actions(site, baseline)
    # Unlike the SparkCache candidate, rollback has no preceding checkpoint
    # helper receipt.  Retain the launcher's outer verifier and skip only the
    # image-entrypoint duplicate.
    return verified_start.decorate_verified_start(starts, baseline)


def checkpoint_helper_remove_actions(site, state: dict) -> list[exl3.RemoteAction]:
    profile = state["profile"]
    actions = []
    for rank in site.ranks:
        name = _helper_name(rank.id)
        ownership = shlex.join(
            (
                "python3",
                "-c",
                _HELPER_OWNERSHIP_ATTEST,
                name,
                profile.image_id,
                profile.profile_id,
                state["candidate_sha256"],
                state["target_checkpoint"],
                profile.model_host_path,
                profile.model_container_path,
            )
        )
        script = (
            "docker info >/dev/null || exit 70; "
            f"name={shlex.quote(name)}; "
            "if ! docker inspect \"$name\" >/dev/null 2>&1; then "
            "ids=$(docker ps -aq --filter name=^/$name$) || exit 71; "
            "test -z \"$ids\" || exit 77; exit 0; fi; "
            f"{ownership} || exit 78; "
            "exec docker rm --force \"$name\""
        )
        actions.append(shell_action(rank, script))
    return actions


_OWNERSHIP_ATTEST = _HOST_CONFIG_NORMALIZE + r"""
import json,subprocess,sys
name,image,profile,bundle,candidate,model_src,model_dst,stage_src,stage_dst,cache_src,cache_dst,kv,contract_json=sys.argv[1:]
contract=json.loads(contract_json)
doc=json.loads(subprocess.check_output(['docker','inspect',name]))[0]
image_doc=json.loads(subprocess.check_output(['docker','image','inspect',image]))[0]
assert doc['Image']==image
labels=doc['Config']['Labels'] or {};expected_labels=dict(image_doc['Config'].get('Labels') or {});expected_labels.update(contract['labels']);assert labels==expected_labels
assert labels['org.sparkring.managed']=='true'
assert labels['org.sparkring.exl3-profile']==profile
assert labels['org.sparkring.component']=='sparkcache-engine'
assert labels['org.sparkring.sparkcache-bundle-sha256']==bundle
assert labels['org.sparkring.sparkcache-candidate-sha256']==candidate
mounts={m['Destination']:(m['Source'],m['RW']) for m in doc['Mounts']}
assert mounts[model_dst]==(model_src,False)
assert mounts[stage_dst]==(stage_src,False)
assert mounts[cache_dst]==(cache_src,True)
assert not (image_doc['Config'].get('Volumes') or {}) and set(mounts)==set(contract['mounts'])
for destination,item in contract['mounts'].items(): assert mounts[destination]==(item['source'],not item['read_only'])
cmd=doc['Config']['Cmd'];indices=[i for i,x in enumerate(cmd[:-1]) if x=='--kv-transfer-config']
assert len(indices)==1 and cmd[indices[0]+1]==kv
assert cmd.count('--disable-hybrid-kv-cache-manager')==1
assert cmd==contract['cmd']
actual_items=doc['Config']['Env'] or [];actual=dict(item.split('=',1) for item in actual_items if '=' in item);assert len(actual)==len(actual_items)
base_items=image_doc['Config'].get('Env') or [];expected=dict(item.split('=',1) for item in base_items if '=' in item);expected.update(contract['environment_overrides']);assert actual==expected
entrypoint=contract['entrypoint_override'];assert doc['Config'].get('Entrypoint')==(image_doc['Config'].get('Entrypoint') if entrypoint is None else [entrypoint])
hc=doc['HostConfig'];eh=contract['host_config']
assert normalize_docker_host_config(hc,eh)==eh
""".strip()


def _owned_action(site, state: dict, operation: str) -> list[exl3.RemoteAction]:
    profile = state["profile"]
    if operation not in ("remove", "restart"):
        raise LauncherError(f"unknown candidate operation {operation}")
    actions = []
    contracts = {
        action.rank: docker_run_contract(action, profile.image_id)
        for action in candidate_start_actions(site, state)
    }
    for rank in site.ranks:
        name = exl3.container_name(profile, rank.id)
        verb = "rm --force" if operation == "remove" else "restart"
        args = profile.extra_vllm_args
        kv = args[args.index("--kv-transfer-config") + 1]
        ownership_attest = shlex.join(
            (
                "python3",
                "-c",
                _OWNERSHIP_ATTEST,
                name,
                profile.image_id,
                profile.profile_id,
                state["bundle_identity"],
                state["candidate_sha256"],
                profile.model_host_path,
                profile.model_container_path,
                state["staging"],
                config.STAGING_DESTINATION,
                state["cache_root"],
                config.CACHE_DESTINATION,
                kv,
                json.dumps(contracts[rank.id], sort_keys=True, separators=(",", ":")),
            )
        )
        absence_guard = ""
        if operation == "remove":
            absence_guard = (
                "docker info >/dev/null || exit 70; "
                "if ! docker inspect \"$name\" >/dev/null 2>&1; then "
                "ids=$(docker ps -aq --filter name=^/$name$) || exit 71; "
                "test -z \"$ids\" || exit 77; "
                "exit 0; fi; "
            )
        script = (
            f"name={shlex.quote(name)}; "
            f"{absence_guard}"
            "test \"$(docker inspect -f '{{index .Config.Labels \"org.sparkring.exl3-profile\"}}' \"$name\")\" = "
            f"{shlex.quote(profile.profile_id)} || exit 73; "
            "test \"$(docker inspect -f '{{index .Config.Labels \"org.sparkring.component\"}}' \"$name\")\" = "
            f"{shlex.quote(CANDIDATE_COMPONENT)} || exit 74; "
            "test \"$(docker inspect -f '{{index .Config.Labels \"org.sparkring.sparkcache-bundle-sha256\"}}' \"$name\")\" = "
            f"{shlex.quote(state['bundle_identity'])} || exit 75; "
            "test \"$(docker inspect -f '{{index .Config.Labels \"org.sparkring.sparkcache-candidate-sha256\"}}' \"$name\")\" = "
            f"{shlex.quote(state['candidate_sha256'])} || exit 76; "
            f"{ownership_attest} || exit 78; "
            f"exec docker {verb} \"$name\""
        )
        actions.append(shell_action(rank, script))
    return actions


def build_phases(site, baseline: exl3.Profile, state: dict) -> dict:
    baseline_phases = lmcache.build_phases(site, baseline)
    return {
        "verified_start_prepare": verified_start.prepare_verified_start_actions(site),
        "candidate_precheck": precheck_actions(site, baseline, state),
        "candidate_absent": candidate_absent_actions(site, state),
        "candidate_absent_final": candidate_absent_actions(site, state),
        "checkpoint_quiescent": checkpoint_quiescent_actions(
            site, baseline, state
        ),
        "checkpoint_full_attest": full_checkpoint_attestation_actions(site, state),
        "checkpoint_helper_absent": checkpoint_helper_absent_actions(site, state),
        "checkpoint_helper_remove": checkpoint_helper_remove_actions(site, state),
        "candidate_start": candidate_start_actions(site, state),
        "candidate_api_barrier": candidate_api_barrier_actions(site, state),
        "candidate_final_status": candidate_final_status_actions(
            site, baseline, state
        ),
        "candidate_restart": _owned_action(site, state, "restart"),
        "candidate_remove": _owned_action(site, state, "remove"),
        "baseline_server_health": baseline_phases["server_health"],
        "baseline_ready": baseline_phases["ready"],
        "baseline_remove_engines": baseline_exact_remove_actions(
            site, baseline, component="engine"
        ),
        "baseline_remove_servers": baseline_exact_remove_actions(
            site, baseline, component="lmcache-server"
        ),
        "baseline_start_servers": baseline_phases["start_servers"],
        "baseline_start_engines": baseline_verified_start_actions(site, baseline),
        "baseline_final_status": baseline_final_status_actions(site, baseline, state),
    }


def lifecycle(command: str) -> list[str]:
    if command == "status":
        return ["candidate_precheck", "candidate_api_barrier", "candidate_final_status"]
    if command == "cutover":
        return [
            "candidate_precheck",
            "candidate_absent",
            "checkpoint_helper_absent",
            "baseline_server_health",
            "baseline_ready",
            "verified_start_prepare",
            "baseline_remove_engines",
            "baseline_remove_servers",
            "checkpoint_quiescent",
            "checkpoint_full_attest",
            "checkpoint_helper_absent",
            "candidate_start",
            "candidate_api_barrier",
            "candidate_final_status",
        ]
    if command == "restart-engines":
        return [
            "candidate_precheck",
            "candidate_api_barrier",
            "candidate_final_status",
            "candidate_restart",
            "candidate_api_barrier",
            "candidate_final_status",
        ]
    if command == "restart-stack":
        return [
            "candidate_precheck",
            "candidate_api_barrier",
            "candidate_final_status",
            "verified_start_prepare",
            "candidate_remove",
            "checkpoint_helper_absent",
            "checkpoint_quiescent",
            "checkpoint_full_attest",
            "checkpoint_helper_absent",
            "candidate_start",
            "candidate_api_barrier",
            "candidate_final_status",
        ]
    if command == "rollback":
        return rollback_lifecycle()
    raise LauncherError(f"unsupported command {command}")


def rollback_lifecycle() -> list[str]:
    return [
        "verified_start_prepare",
        "checkpoint_helper_remove",
        "candidate_remove",
        "baseline_remove_engines",
        "baseline_remove_servers",
        "checkpoint_quiescent",
        "baseline_start_servers",
        "baseline_server_health",
        "baseline_start_engines",
        "baseline_ready",
        "baseline_server_health",
        "baseline_final_status",
        "checkpoint_helper_absent",
        "candidate_absent_final",
    ]


def _render(actions) -> list[dict]:
    return [
        {
            "rank": action.rank,
            "ssh_target": action.ssh_target,
            "remote_command": action.shell_command,
        }
        for action in actions
    ]


_EXPECTED_RANKS = frozenset(range(4))
_RANK_RECEIPT_KEYS = frozenset(
    ("exit_code", "stdout", "stderr", "failure_kind")
)
_EXECUTOR_RESULT_KEYS = frozenset(("exit_code", "stdout", "stderr"))


def _failure_receipts(kind: str, detail: str) -> dict[int, dict]:
    message = f"launcher_failure[{kind}]: {detail}"[:4096]
    return {
        rank: {
            "exit_code": 125,
            "stdout": "",
            "stderr": message,
            "failure_kind": kind,
        }
        for rank in sorted(_EXPECTED_RANKS)
    }


def _execute_phase(
    executor: Callable, actions: list[exl3.RemoteAction], *, timeout: int
) -> dict[int, dict]:
    action_ranks = [action.rank for action in actions]
    if (
        len(action_ranks) != 4
        or any(type(rank) is not int for rank in action_ranks)
        or set(action_ranks) != _EXPECTED_RANKS
    ):
        return _failure_receipts(
            "action_rank_contract",
            f"expected exactly ranks 0,1,2,3; got {action_ranks!r}",
        )
    try:
        raw = executor(actions, timeout=timeout)
    except Exception as error:
        return _failure_receipts(
            "executor_exception", f"{type(error).__name__}: {error}"
        )
    if (
        not isinstance(raw, dict)
        or any(type(rank) is not int for rank in raw)
        or set(raw) != _EXPECTED_RANKS
    ):
        observed = list(raw) if isinstance(raw, dict) else type(raw).__name__
        return _failure_receipts(
            "executor_rank_contract",
            f"expected exactly ranks 0,1,2,3; got {observed!r}",
        )
    normalized = {}
    for rank in sorted(_EXPECTED_RANKS):
        item = raw[rank]
        if not isinstance(item, dict) or set(item) != _EXECUTOR_RESULT_KEYS:
            return _failure_receipts(
                "executor_result_shape",
                f"rank {rank} result keys are not exactly exit_code/stdout/stderr",
            )
        exit_code = item["exit_code"]
        if (
            isinstance(exit_code, bool)
            or not isinstance(exit_code, int)
            or not isinstance(item["stdout"], str)
            or not isinstance(item["stderr"], str)
        ):
            return _failure_receipts(
                "executor_result_shape",
                f"rank {rank} result values have invalid types",
            )
        normalized[rank] = {
            "exit_code": exit_code,
            "stdout": item["stdout"],
            "stderr": item["stderr"],
            "failure_kind": None,
        }
    return normalized


def _failed(result: dict) -> bool:
    if not isinstance(result, dict) or set(result) != _EXPECTED_RANKS:
        return True
    for item in result.values():
        if not isinstance(item, dict) or set(item) != _RANK_RECEIPT_KEYS:
            return True
        if item["exit_code"] != 0:
            return True
    return False


def _mutating_phase(name: str) -> bool:
    return name in {
        "candidate_start",
        "candidate_restart",
        "candidate_remove",
        "checkpoint_full_attest",
        "checkpoint_helper_remove",
        "baseline_remove_engines",
        "baseline_remove_servers",
        "baseline_start_servers",
        "baseline_start_engines",
    }


def execute_lifecycle(
    command: str,
    phases: dict,
    executor: Callable = exl3.execute,
) -> tuple[int, dict]:
    results = {}

    def execute_rollback(*, automatic: bool) -> bool:
        failed = False
        quiescence_failed = False
        safe_after_failed_quiescence = {
            "checkpoint_helper_absent",
            "candidate_absent_final",
        }
        if automatic:
            results["automatic_rollback"] = []
        for rollback_name in rollback_lifecycle():
            if quiescence_failed and rollback_name not in safe_after_failed_quiescence:
                if automatic:
                    results["automatic_rollback"].append(
                        {"phase": rollback_name, "skipped": "quiescence_failed"}
                    )
                else:
                    results.setdefault(rollback_name, []).append(
                        {"skipped": "quiescence_failed"}
                    )
                continue
            rollback_result = _execute_phase(
                executor, phases[rollback_name], timeout=3900
            )
            if automatic:
                results["automatic_rollback"].append(
                    {"phase": rollback_name, "result": rollback_result}
                )
            else:
                results.setdefault(rollback_name, []).append(rollback_result)
            phase_failed = _failed(rollback_result)
            failed = failed or phase_failed
            # The wrapper/reclaim authority must be proved before removing a
            # candidate or canonical engine.  Continuing after this failure
            # would turn rollback into an avoidable serving outage.
            if rollback_name == "verified_start_prepare" and phase_failed:
                break
            if rollback_name == "checkpoint_quiescent" and phase_failed:
                quiescence_failed = True
        return failed

    if command == "rollback":
        return (1 if execute_rollback(automatic=False) else 0), results

    mutation_started = False
    for name in lifecycle(command):
        mutation_started = mutation_started or _mutating_phase(name)
        result = _execute_phase(executor, phases[name], timeout=3900)
        results.setdefault(name, []).append(result)
        if _failed(result):
            if mutation_started and command != "rollback":
                execute_rollback(automatic=True)
            return 1, results
    return 0, results


def main(argv: list[str] | None = None, *, executor: Callable = exl3.execute) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--site", required=True)
    parser.add_argument("--baseline-profile", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--output")
    parser.add_argument(
        "command",
        choices=("plan", "status", "cutover", "restart-engines", "restart-stack", "rollback"),
    )
    args = parser.parse_args(argv)
    try:
        site = load_site(args.site)
        baseline = exl3.load_profile(Path(args.baseline_profile))
        if baseline.profile_id != lmcache.PROFILE_ID:
            raise LauncherError("baseline must be canonical EXL3+LMCache CS512")
        candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
        state = validate_candidate(candidate, baseline)
        phases = build_phases(site, baseline, state)
        selected = "cutover" if args.command == "plan" else args.command
    except (
        OSError,
        KeyError,
        json.JSONDecodeError,
        SiteConfigError,
        exl3.ProfileError,
        config.SparkCacheProfileError,
        LauncherError,
    ) as error:
        parser.error(str(error))

    plan = {
        "schema": PLAN_SCHEMA,
        "lane": "public-functional",
        "maturity": "offline-validated",
        "configuration_status": "candidate",
        "command": args.command,
        "dry_run": not args.execute,
        "mutates_remote": selected != "status",
        "candidate_sha256": state["candidate_sha256"],
        "candidate_profile_id": state["profile"].profile_id,
        "baseline_profile_id": baseline.profile_id,
        "confirmation_required": CONFIRMATIONS.get(selected),
        "lifecycle": lifecycle(selected),
        "phases": {name: _render(actions) for name, actions in phases.items()},
        "rank_completeness": {
            "rank_ids": sorted(rank.id for rank in site.ranks),
            "all_phases_have_four_ranks": all(len(actions) == 4 for actions in phases.values()),
        },
        "attestations": [
            "exact image/container contract plus receipt inventory and byte sizes on every rank while live",
            "full checkpoint receipt identity in a GPU-less exact-image helper only after all-rank model-process quiescence",
            "hash-attested shared EXL3 outer-verified entrypoint installed before serving interruption",
            "one full model verifier per start path with fail-fast host page-cache reclaim before GPU initialization",
            "exclusive connector bundle inventory and identity on every rank",
            "async speculative-placeholder rollback patch semantics",
            "SparkContextCacheConnector-only VMM exemption semantics",
            "exact ownership labels and model/staging/cache mounts",
        ],
        "rollback_target": "canonical EXL3+LMCache CS512 four-engine/four-server stack",
        "disclaimer": "A successful plan is not live validation or acceptance.",
    }
    if args.command == "plan" or not args.execute:
        try:
            _emit(plan, args.output)
        except LauncherError as error:
            parser.error(str(error))
        return 0
    required = CONFIRMATIONS[selected]
    if args.confirmation != required:
        parser.error(f"execute requires --confirmation {required}")
    exit_code, results = execute_lifecycle(selected, phases, executor)
    try:
        _emit({"plan": plan, "results": results}, args.output)
    except LauncherError as error:
        parser.error(str(error))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
