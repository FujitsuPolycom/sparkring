"""Create stopped managed GLM containers from the shared structured specification."""
import argparse
from dataclasses import replace
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from runtime.common import candidate, glm_targets, glm_tp4, r35, glm_source_candidate  # noqa: E402
from runtime.common.container_spec import docker_create, expected_inspection  # noqa: E402

STRUCTURED_LABEL = "io.sparkring.container-spec"
STRUCTURED_SCHEMA = "glm-tp4/v1"


@lru_cache(maxsize=1)
def profile_owner():
    path = ROOT / "runtime/glm53-spark-mtp3-mesh/profile.py"
    loader = importlib.util.spec_from_file_location("structured_glm_profile", path)
    module = importlib.util.module_from_spec(loader)
    sys.modules[loader.name] = module
    loader.loader.exec_module(module)
    return module


def read_api_keys(path):
    if not path:
        return ()
    path = Path(path)
    if not path.is_file() or stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise ValueError("API_KEYS_FILE must be a readable regular file with mode 0600")
    keys = tuple(line for line in path.read_text().splitlines() if line.strip())
    if not keys or any(any(char.isspace() for char in key) for key in keys):
        raise ValueError("API_KEYS_FILE requires nonempty keys without whitespace")
    return keys


def resolve_spec(launch, image_receipt, rank, *, owner=None):
    """Regenerate supplied launch inputs before building the typed container."""
    launch, image_receipt = Path(launch), Path(image_receipt)
    if type(rank) is not int or rank not in range(4):
        raise ValueError("Managed GLM rank must be zero through three")
    profile = profile_owner() if owner is None else owner
    site, _, _ = profile.load_site(launch / "site.json")
    with tempfile.TemporaryDirectory(prefix="sparkring-glm-inputs-") as temporary:
        canonical = Path(temporary) / "launch"
        profile.render(launch / "site.json", Path(site["bundle_root"]), canonical, image_receipt)
        for name in ("launch-rank.sh", f"rank{rank}.env"):
            supplied = launch / name
            if not stat.S_ISREG(supplied.lstat().st_mode) or supplied.read_bytes() != (canonical / name).read_bytes():
                raise ValueError("Supplied launch input differs from the canonical source: " + name)
    resolved = profile.resolve_rank_environments(launch / "site.json", Path(site["bundle_root"]), image_receipt)
    record = resolved["image_record"]
    if record.get("schema") not in (r35.SCHEMA, candidate.SCHEMA, glm_source_candidate.SCHEMA):
        raise ValueError("Structured GLM creation supports R35 and registered candidate images; use the retained launcher for other releases")
    adapter = glm_source_candidate if record["schema"] == glm_source_candidate.SCHEMA else candidate if record["schema"] == candidate.SCHEMA else r35
    environment = resolved["environments"][rank]
    metadata = {}
    if environment["TARGET_MODEL_VARIANT"] != glm_targets.DEFAULT:
        root = Path(environment["TARGET_MODEL_HOST_PATH"])
        metadata = {"model_config": (root / "config.json").read_bytes(),
                    "model_index": (root / "model.safetensors.index.json").read_bytes()}
        resolved["model_metadata_sha256"] = {key: hashlib.sha256(value).hexdigest()
                                             for key, value in metadata.items()}
    spec = glm_tp4.build_spec(environment, image_record=record,
                             contract=(glm_source_candidate.contract_for_receipt(record) if record["schema"] == glm_source_candidate.SCHEMA
                                       else adapter.profile_contract(record["installed"])),
                             api_keys=read_api_keys(environment.get("API_KEYS_FILE")), **metadata)
    spec = replace(spec, labels={**spec.labels, STRUCTURED_LABEL: STRUCTURED_SCHEMA})
    return spec, record, resolved


def document(launch, image_receipt, rank, backend, spec, record, resolved):
    if backend not in ("docker", "compose"):
        raise ValueError("Select Docker or Compose creation")
    return {"schema": "sparkring-glm-container-plan/v1", "rank": rank, "backend": backend,
            "image_reference": record["image_reference"], "container": spec.document(),
            "site_sha256": hashlib.sha256((Path(launch) / "site.json").read_bytes()).hexdigest(),
            "topology_sha256": resolved["topology"].sha256,
            **({"model_metadata_sha256": resolved["model_metadata_sha256"]}
               if "model_metadata_sha256" in resolved else {}),
            "image_receipt_sha256": hashlib.sha256(Path(image_receipt).read_bytes()).hexdigest(),
            "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                              for name in ("runtime/common/glm_launch.py", "runtime/common/glm_tp4.py",
                                           "runtime/common/container_spec.py", "runtime/common/compose.py",
                                           "runtime/common/glm_targets.py", "runtime/common/glm_source_candidate.py",
                                           "runtime/common/source_candidate.py", "profiles/glm53-target-variants.json",
                                           "runtime/glm53-spark-mtp3-mesh/profile.py")}}


def plan_directory(launch, rank):
    """Keep generated control files beside the immutable authenticated launch.

    A launch at /deployment/launch uses /deployment/launch-containers/rankN.
    Resolving the launch first gives every consumer the same location without
    adding files to the preparation verifier's exact launch inventory.
    """
    if type(rank) is not int or rank not in range(4):
        raise ValueError("Managed GLM rank must be zero through three")
    launch = Path(launch).resolve()
    root = launch.with_name(launch.name + "-containers")
    result = root / f"rank{rank}"
    if root.resolve() != root or result.resolve() != result:
        raise ValueError("GLM container plan directories cannot be symlinks")
    return result


def check_plan(launch, image_receipt, rank, spec, record, resolved):
    path = plan_directory(launch, rank) / "plan.json"
    if not path.is_file():
        raise ValueError("Structured GLM creation plan is missing")
    saved = json.loads(path.read_bytes())
    expected = document(launch, image_receipt, rank, saved.get("backend"), spec, record, resolved)
    # asdict contains tuples before JSON serialization.
    if saved != json.loads(json.dumps(expected)):
        raise ValueError("GLM creation plan changed; regenerate a separate launch directory")
    return expected["backend"]


def run(argv, **kwargs):
    if argv[0] == "docker" and argv[1:3] != ["--context", "default"]:
        argv = ["docker", "--context", "default", *argv[1:]]
    kwargs.setdefault("check", True)
    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("text", True)
    kwargs.setdefault("timeout", 300)
    return subprocess.run(argv, **kwargs)


def validate_mounts(spec):
    for mount in spec.mounts:
        source = Path(mount.source)
        if mount.target == "/opt/sparkring/chat_template.jinja":
            if not source.is_file():
                raise ValueError("The selected chat template must be a regular file")
        elif not source.is_dir():
            raise ValueError("Create the selected model/cache directory before container creation")
        if mount.target == "/models/target" and not (source / "config.json").is_file():
            raise ValueError("The selected model directory must contain config.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "check", "create"))
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--image-receipt", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=range(4), required=True)
    parser.add_argument("--backend", choices=("docker", "compose"), default="docker")
    args = parser.parse_args(argv)
    try:
        from runtime.common import compose

        spec, record, resolved = resolve_spec(args.launch, args.image_receipt, args.rank)
        expected = document(args.launch, args.image_receipt, args.rank, args.backend, spec, record, resolved)
        output = plan_directory(args.launch, args.rank)
        yaml_text = compose.compose_text(spec, record["image_reference"])
        if args.action == "plan":
            target = output.resolve()
            if target.is_relative_to(ROOT) and not target.is_relative_to(ROOT / ".sparkring"):
                raise ValueError("Private plans inside the repository must be under .sparkring/")
            output.parent.mkdir(mode=0o700, exist_ok=True)
            output.mkdir(mode=0o700, exist_ok=False)
            for name, text in (("plan.json", compose.encoded(expected)), ("compose.yaml", yaml_text)):
                with (output / name).open("x", encoding="utf-8", newline="\n") as stream:
                    stream.write(text)
                (output / name).chmod(0o600)
        else:
            if not (output / "plan.json").is_file() or check_plan(args.launch, args.image_receipt, args.rank, spec, record, resolved) != args.backend:
                raise ValueError("Generate the matching GLM creation plan first")
            if (output / "compose.yaml").read_bytes() != yaml_text.encode():
                raise ValueError("Edited Compose settings differ from the canonical GLM specification")
            if args.backend == "compose":
                compose.check_equivalence(spec, record["image_reference"], yaml_text, run=run)
            if args.action == "create":
                if (output / "created.json").exists():
                    raise ValueError("Creation receipt already exists; inspect the deployment before recovery")
                validate_mounts(spec)
                adapter = glm_source_candidate if record["schema"] == glm_source_candidate.SCHEMA else candidate if record["schema"] == candidate.SCHEMA else r35
                adapter.verify_local_image(record, run=run)
                if args.backend == "compose":
                    compose.check_project_containers(spec.name, run=run)
                    run(compose.compose_command(spec.name, output / "compose.yaml") +
                        ["create", "--no-build", "--no-recreate", "--pull", "never", "model"])
                else:
                    run(docker_create(spec))
                observed = json.loads(run(["docker", "inspect", spec.name]).stdout)[0]
                image = json.loads(run(["docker", "image", "inspect", spec.image_id]).stdout)[0]
                # The managed installer applies the same effective-spec check.
                from importlib import util
                loader = util.spec_from_file_location("glm_managed_install", ROOT / "runtime/glm53-spark-mtp3-mesh/managed_install.py")
                module = util.module_from_spec(loader)
                sys.modules[loader.name] = module
                support = str(ROOT / "runtime/glm53-spark-mtp3-mesh")
                sys.path.insert(0, support)
                try:
                    loader.loader.exec_module(module)
                finally:
                    sys.path.remove(support)
                module.validate_container_spec(observed, expected_inspection(spec, image, backend=args.backend))
                if observed.get("State", {}).get("Running"):
                    raise ValueError("Creation unexpectedly started the model")
                with (output / "created.json").open("x", encoding="utf-8") as stream:
                    json.dump({"container_id": observed["Id"], "image_id": spec.image_id, "running": False}, stream)
                (output / "created.json").chmod(0o600)
        print(json.dumps({"action": args.action, "backend": args.backend, "plan": str(output / "plan.json"),
                          "name": spec.name, "image_id": spec.image_id, "model_started": False}))
        return 0
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        reason = type(error).__name__ if isinstance(error, subprocess.SubprocessError) else str(error)
        print("GLM creation failed: " + reason + "; inspect the private plan and any existing container before retrying", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
