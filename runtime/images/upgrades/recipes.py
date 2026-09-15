"""Create a private, source-pinned upgrade recipe from the retained R37 composition."""

import json
from pathlib import Path
import shutil

from .contracts import load_policy, sha, require
from .io import write_json

ROOT = Path(__file__).resolve().parents[3]


def initialize(output, *, endpoint=None, model=None, native=False):
    output = Path(output).resolve()
    require(not output.exists(), "Recipe directory already exists")
    output.mkdir(parents=True)
    recipe = ROOT / "runtime/images/compositions/lil-r37-glm-spark"
    lock = json.loads((recipe / "source-lock.json").read_text())
    for name in ("pytest_gate.py", "image_gate.py"):
        shutil.copyfile(Path(__file__).with_name(name), output / name)
    image = "ghcr.io/fujitsupolycom/sparkring@sha256:aef597a5ee70f7b4e0807901e43456b6ac8d2234247ab4df6a6cfe031e5169c6"
    sources, gates = [], []
    tests = {
        "vllm": [
            "tests/v1/core/test_recurrent_prefill_checkpoint.py",
            "tests/v1/kv_connector/unit/test_hybrid_recovery_source.py",
        ],
        "b12x": ["tests/sequence/test_kda_prefill_two_checkpoints_cpu.py"],
    }
    for name, source in lock["components"].items():
        patch = output / source["patch"]
        shutil.copyfile(recipe / source["patch"], patch)
        sources.append(
            {
                "id": name,
                "repository": "https://github.com/local-inference-lab/" + name + ".git",
                "ref": "refs/heads/dev/jovian-judgement"
                if name == "vllm"
                else "refs/heads/master",
                "baseline": source["base_commit"],
                "patch": patch.name,
                "patch_sha256": sha(patch.read_bytes()),
                "editable_paths": [name],
                "native_paths": source["native_comparison"]["paths"],
                "excluded_paths": [".claude", ".agents"],
                "contracts": [
                    {
                        "id": name + "-checkpoint-invariants",
                        "kind": "optimization",
                        "oracle": name + "-checkpoint",
                        "invariant": "Retain recurrent checkpoint export, planned-boundary propagation and fail-closed external-cache recovery. CPU checks do not establish GPU correctness or performance; do not retire optimization behavior without a performance oracle.",
                    }
                ],
            }
        )
        gates.append(
            {
                "id": name + "-checkpoint",
                "stage": "oracle",
                "executor": "docker",
                "image": image,
                "argv": [
                    "/opt/venv/bin/python",
                    "/oracles/pytest_gate.py",
                    "--source",
                    "/source",
                    "--baseline",
                    "/baseline",
                    "--component",
                    name,
                    "--result",
                    "/out/result.json",
                    *tests[name],
                ],
                "inputs": [
                    {
                        "path": "pytest_gate.py",
                        "sha256": sha((output / "pytest_gate.py").read_bytes()),
                    }
                ],
            }
        )
    gates.append(
        {
            "id": "installed-contracts",
            "stage": "image",
            "executor": "docker",
            "image": "candidate",
            "argv": ["/opt/venv/bin/python", "/oracles/image_gate.py"],
            "inputs": [
                {
                    "path": "image_gate.py",
                    "sha256": sha((output / "image_gate.py").read_bytes()),
                }
            ],
        }
    )
    policy = {
        "schema": "sparkring-image-upgrade/v1",
        "name": "lil-arm64",
        "platform": "linux/arm64",
        "foundation": {
            "image": image,
            "image_id": "sha256:2540686d726a28eb07784f9d2db5dc1f795404c7874fc1d6c11f018cd789adc2",
        },
        "required_features": ["qwen-collectives", "qwen-prefill"],
        "sources": sources,
        "gates": gates,
        "budgets": dict(
            run_seconds=7200,
            command_seconds=1800,
            source_bytes=1024**3,
            output_bytes=4 * 1024**2,
            agent_attempts=2,
            state_bytes=20 * 1024**3,
            min_free_bytes=10 * 1024**3,
        ),
        "permissions": {"candidate_publication": False},
        "build": {
            "argv": [
                "python",
                "{controller_root}/runtime/images/upgrades/build_candidate.py",
                "--policy",
                "{policy}",
                "--bundle",
                "{bundle}",
                "--output",
                "{run_root}/image-context",
                "--result",
                "{result}",
            ],
            "supports_native_rebuild": False,
        },
    }
    if native:
        policy["native"] = {
            "schema": "sparkring-native-recipe/v1",
            "architecture": "12.1a",
            "jobs": 8,
            "cpus": 12,
            "memory_bytes": 80 * 1024**3,
            "build_seconds": 21600,
            "torch_version": "2.13.0",
            "network": "bridge",
        }
        policy["build"]["argv"][1] = (
            "{controller_root}/runtime/images/upgrades/build_native.py"
        )
        policy["build"]["supports_native_rebuild"] = True
        policy["budgets"].update(
            run_seconds=43200,
            command_seconds=21600,
            output_bytes=64 * 1024**2,
            state_bytes=64 * 1024**3,
        )
        for gate in gates:
            gate["timeout_seconds"] = 900 if gate["stage"] == "oracle" else 1800
    if endpoint or model:
        require(endpoint and model, "Provide both agent endpoint and model")
        policy["agent"] = {
            "endpoint": endpoint,
            "model": model,
            "max_tokens": 8192,
            "context_bytes": 1000000,
            "timeout_seconds": 120,
        }
    write_json(output / "policy.json", policy)
    result = load_policy(output / "policy.json")
    return {
        "policy": str(output / "policy.json"),
        "policy_sha256": result["_digest"],
        "scope": (
            "Native GB10 wheel candidate; serving qualification is not configured."
            if native
            else "Pinned source-overlay candidate recipe. Native-input changes and changed feature/connector bindings block adoption; hardware qualification is not configured."
        ),
    }
