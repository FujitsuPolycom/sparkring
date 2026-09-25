"""GLM container lifecycle on a verified, independently retained native mesh."""
import contextlib
import json
import re
import sys
import time
import urllib.error
import urllib.request

from runtime.common import glm_launch, installer, qwen_mesh
from scripts import deploy_selection, deploy_stage


def resolve(lock, rank, state):
    from scripts import installer_host as host
    binding = installer.read(host.plain(state / "glm-preparation.json"))
    if binding.get("deployment") != lock["id"]:
        raise ValueError("GLM staging belongs to a different installer deployment")
    workspace = host.plain(installer.managed_workspace(lock["site"]["name"]))
    deploy_stage.verify_host(workspace, binding["preparation_sha256"], quiet=True)
    prepared = installer.read(workspace / "preparation.json")
    receipt = deploy_selection.receipt_path(prepared["spec"], workspace)
    spec, image_record, resolved = glm_launch.resolve_spec(workspace / "launch", receipt, rank)
    row = lock["site"]["ranks"][rank]
    models = [m.source for m in spec.mounts if m.target == "/models/target"]
    if (spec.name != lock["site"]["name"] + f"-r{rank}" or spec.image_id != lock["selection"]["image_id"]
            or models != [row["model"]] or ("--kv-transfer-config" in spec.command) != lock["selection"]["sparkcache"]
            or "--enable-prefix-caching" not in spec.command
            or resolved["environments"][rank]["HOST_IP"] != row["host_ip"]):
        raise ValueError("Staged GLM settings differ from the requested model/image/cache policy")
    return spec, image_record, workspace / "launch", receipt


def perform(operation, lock, rank, state):
    from scripts import installer_host as host
    row = lock["site"]["ranks"][rank]
    if operation in ("existing-glm-bind", "existing-glm-bound"):
        supplied = json.load(sys.stdin)
        digest = supplied.get("preparation_sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("GLM preparation needs a complete controller digest")
        expected = {"deployment": lock["id"], "preparation_sha256": digest}
        workspace = host.plain(installer.managed_workspace(lock["site"]["name"]))
        deploy_stage.verify_host(workspace, digest, quiet=True)
        path = host.plain(state / "glm-preparation.json")
        if not path.exists() and operation == "existing-glm-bind":
            installer.write(path, expected)
        if not path.exists() or installer.read(path) != expected:
            raise ValueError("Host GLM preparation is not bound to this controller")
        return {"ok": True}
    spec, record, launch, receipt = resolve(lock, rank, state)
    image = host.image_info(lock)
    info = host.container(spec)
    if info:
        host.owned(spec, info, image)
    if operation == "status":
        return host.model_observation(lock, row, info)
    if operation == "owned":
        return {"ok": True}
    if operation in ("stop", "stopped"):
        if info and info["State"].get("Running"):
            if operation == "stopped":
                raise ValueError("GLM rank is still running")
            host.run(["docker", "stop", "--time", "60", info["Id"]])
        return {"ok": True}
    if operation in ("running", "created"):
        host.owned(spec, info, image)
        if operation == "running" and not info["State"].get("Running"):
            raise ValueError("GLM rank is not running")
        return {"ok": True}
    if operation in ("preflight", "create", "start"):
        qwen_mesh.check(row["fabric"], rank, row["hcas"], row["gid"], row["host_ip"])
        host.admit_image(lock)
        host.verify_model(lock, row, state / "model.json")
        if not (info and info["State"].get("Running")):
            host.require_idle()
        if operation == "create" and info is None:
            options = ["--launch", str(launch), "--image-receipt", str(receipt), "--rank", str(rank), "--backend", "compose"]
            with contextlib.redirect_stdout(sys.stderr):
                if not (glm_launch.plan_directory(launch, rank) / "plan.json").exists():
                    if glm_launch.main(["plan", *options]):
                        raise ValueError("GLM container planning failed")
                if glm_launch.main(["create", *options]):
                    raise ValueError("GLM stopped-container creation failed")
            host.owned(spec, host.container(spec), image)
        if operation == "start":
            host.owned(spec, info, image)
            if not info["State"].get("Running"):
                host.run(["docker", "start", info["Id"]])
        return {"ok": True}
    connection = installer.connection(lock)
    if operation == "ready":
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            current = host.owned(spec, host.container(spec), image)
            if not current["State"].get("Running"):
                raise ValueError("GLM rank exited during startup")
            if rank != 0:
                return {"ok": True}
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{connection['port']}/health", timeout=5) as response:
                    if response.status == 200:
                        return {"ok": True}
            except (urllib.error.URLError, TimeoutError):
                pass
            time.sleep(2)
        raise ValueError("GLM readiness exceeded 30 minutes")
    if operation == "smoke":
        listed = host.http_json(connection["port"], "/v1/models")
        if connection["model"] not in [m["id"] for m in listed["data"]]:
            raise ValueError("GLM API serves a different model")
        response = host.http_json(connection["port"], "/v1/chat/completions", {
            "model": connection["model"], "messages": [{"role": "user", "content": "Reply only READY"}],
            "max_tokens": 64, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
        if not response.get("choices") or not response["choices"][0]["message"].get("content", "").strip():
            raise ValueError("GLM smoke request returned no answer")
        return {"ok": True, "model": connection["model"]}
    raise ValueError("Unsupported existing-mesh GLM operation")
