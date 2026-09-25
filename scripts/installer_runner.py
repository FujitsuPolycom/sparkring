"""SSH execution boundary for the profile installer; imported planning never calls it."""
from __future__ import annotations

import base64
import contextlib
import inspect
import json
from pathlib import Path
import shlex
import subprocess
import sys

from runtime.common import distribution, installer
from scripts import deploy_engine


def _probe():
    import ipaddress
    import json
    import os
    import platform
    import re
    import shutil
    import subprocess
    from pathlib import Path

    def run(args):
        value = subprocess.run(args, capture_output=True, text=True, timeout=30)
        if value.returncode:
            raise ValueError("Failed prerequisite: " + " ".join(args) + ": " + value.stderr.strip())
        return value.stdout

    tools = {name: shutil.which(name) is not None for name in ("git", "python3", "docker", "nvidia-smi", "nvidia-ctk", "ip", "ibv_devinfo")}
    addresses = json.loads(run(["ip", "-j", "address"]))
    ipv4 = {row["ifname"]: [item["local"] for item in row.get("addr_info", []) if item["family"] == "inet"] for row in addresses}
    management = os.environ.get("SSH_CONNECTION", "").split()
    management_ip = management[2] if len(management) == 4 else ""
    rdma = []
    for path in sorted(Path("/sys/class/infiniband").glob("*")):
        port = path / "ports/1"
        try:
            netdev = (port / "gid_attrs/ndevs/3").read_text().strip()
            gid = ipaddress.IPv6Address((port / "gids/3").read_text().strip())
            verbs = run(["ibv_devinfo", "-d", path.name, "-i", "1"])
            active_mtu = re.search(r"active_mtu:\s+(\d+)\s+\(\d+\)", verbs)
            if active_mtu is None:
                raise ValueError("RDMA active MTU is unavailable")
            rdma.append({"device": path.name, "netdev": netdev, "gid_ip": str(gid.ipv4_mapped),
                         "type": (port / "gid_attrs/types/3").read_text().strip(),
                         "active": "ACTIVE" in (port / "state").read_text(), "ips": ipv4.get(netdev, []),
                         "rdma_mtu": int(active_mtu.group(1)),
                         "mtu": next(row["mtu"] for row in addresses if row["ifname"] == netdev)})
        except (OSError, ValueError, StopIteration):
            rdma.append({"device": path.name, "error": "Missing or invalid GID index 3/interface mapping"})
    primary = next((row for row in rdma if row["device"] == "rocep1s0f0" and "error" not in row), {})
    try:
        import yaml  # noqa: F401
        tools["PyYAML"] = True
    except ImportError:
        tools["PyYAML"] = False
    ids = run(["docker", "--context", "default", "ps", "--quiet"]).split()
    containers = json.loads(run(["docker", "--context", "default", "inspect", *ids])) if ids else []
    gpu_containers = [{"name": c["Name"].lstrip("/"), "pid": c["State"]["Pid"], "image": c["Image"],
                       "labels": c["Config"].get("Labels", {})} for c in containers if c["HostConfig"].get("DeviceRequests")]
    processes = []
    for text in run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]).splitlines():
        pid = int(text.strip())
        chain, current = [pid], pid
        for _ in range(256):
            if current <= 1:
                break
            try:
                current = int((Path("/proc") / str(current) / "stat").read_text().rsplit(")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                break
            chain.append(current)
        processes.append(chain)
    return {"platform": platform.system(), "architecture": platform.machine(), "tools": tools,
            "management_ip": management_ip, "interface": primary.get("netdev"),
            "fabric_ip": primary.get("ips", [None])[0] if len(primary.get("ips", [])) == 1 else None,
            "rdma": rdma, "ipv4": ipv4, "docker": run(["docker", "info", "--format", "{{.ServerVersion}}"]),
            "compose": run(["docker", "compose", "version", "--short"]),
            "gpu_containers": gpu_containers, "gpu_process_ancestors": processes,
            "gpu": run(["nvidia-smi", "-L"])}


PROBE = inspect.getsource(_probe) + "\nimport json\nprint(json.dumps(_probe()))\n"

SOURCE = r'''import hashlib,json,pathlib,subprocess,sys
workspace,source,revision,identity,digest,mode=sys.argv[1:]
root=pathlib.Path(workspace)
target=pathlib.Path(source)
if not target.is_relative_to(root) or any(p.is_symlink() for p in (root,*root.parents,target,*target.parents)):
 raise SystemExit('Unsafe source/workspace path')
owner=root/'.installer-owner.json'
expected={'deployment':identity}
if mode=='install':
 if not root.exists(): root.mkdir(mode=0o700)
 if owner.exists():
  if json.loads(owner.read_text())!=expected: raise SystemExit('Workspace belongs to another deployment')
 else:
  if any(root.iterdir()): raise SystemExit('Workspace is not empty; choose a dedicated deployment name/path')
  with owner.open('x') as f: json.dump(expected,f)
  owner.chmod(0o600)
 data=sys.stdin.buffer.read()
 if hashlib.sha256(data).hexdigest()!=digest: raise SystemExit('Source bundle checksum differs')
 bundle=root/'source.bundle'
 if bundle.exists():
  if hashlib.sha256(bundle.read_bytes()).hexdigest()!=digest: raise SystemExit('Existing bundle differs')
 else:
  with bundle.open('xb') as f: f.write(data)
  bundle.chmod(0o600)
 if not target.exists(): subprocess.run(['git','clone','--quiet',str(bundle),str(target)],check=True)
if json.loads(owner.read_text())!=expected: raise SystemExit('Workspace owner differs')
actual=subprocess.check_output(['git','-C',str(target),'rev-parse','HEAD'],text=True).strip()
dirty=subprocess.check_output(['git','-C',str(target),'status','--porcelain','--untracked-files=all'],text=True).strip()
if actual!=revision or dirty: raise SystemExit('Host source changed; restore the exact checkout before proceeding')
print('ok')
'''

HOST = r'''import base64,json,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1])
lock=json.loads(base64.b64decode(sys.argv[2]))
actual=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
dirty=subprocess.check_output(['git','-C',str(root),'status','--porcelain','--untracked-files=all'],text=True).strip()
if actual!=lock['source_revision'] or dirty: raise SystemExit('Host source differs from the deployment lock')
sys.path.insert(0,str(root))
from scripts.installer_host import perform
print(json.dumps(perform(sys.argv[3],lock,int(sys.argv[4]))))
'''

STATUS = r'''import json,subprocess,sys,time
name,image,key,value,rank=sys.argv[1:]
def run(args): return subprocess.check_output(['docker','--context','default',*args],text=True)
names=run(['container','ls','--all','--format','{{.Names}}']).splitlines()
result={'schema':'sparkring-unverified-container-observation/v1','source':'docker-name-lookup',
        'observed_at':time.time(),'spec_verified':False,'rank':int(rank),'name':name,'present':name in names,'running':False}
if name in names:
 info=json.loads(run(['inspect',name]))[0]
 result.update(running=info['State'].get('Running',False),health=info['State'].get('Health',{}).get('Status'),
               image_matches=info['Image']==image,ownership_label_matches=info['Config'].get('Labels',{}).get(key)==value)
print(json.dumps(result))
'''


def ssh(target, argv, *, data=None, timeout=7200):
    installer.host(target)
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", target,
                             shlex.join(argv)], input=data, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace")[-4000:] or "Remote operation failed")
    return result.stdout.decode()


def discover(target):
    facts = json.loads(ssh(target, ["python3", "-I", "-B", "-c", PROBE], timeout=120))
    return {"host": target, "management_ip": facts["management_ip"], "fabric_ip": facts["fabric_ip"], "interface": facts["interface"]}


def check_facts(facts, row):
    if facts["platform"] != "Linux" or facts["architecture"] not in ("aarch64", "arm64"):
        raise ValueError("Serving hosts must be Linux ARM64 GB10 machines")
    if len(facts["gpu"].strip().splitlines()) != 1 or "GB10" not in facts["gpu"]:
        raise ValueError("Each rank must expose one GB10 GPU")
    missing = [name for name, found in facts["tools"].items() if not found]
    if missing:
        raise ValueError("Missing host tools: " + ", ".join(missing))
    if facts["management_ip"] != row["management_ip"]:
        raise ValueError("Discovered management/fabric mapping differs from the saved site; inspect before reinitializing")
    if "fabric" in row:
        # TP4 bootstraps over the admitted mesh's management address. Its RDMA
        # endpoints are checked independently below and by qwen_mesh admission.
        matching = [name for name, values in facts.get("ipv4", {}).items() if row["host_ip"] in values]
        if matching != [row["interface"]] or row["interface"] in {d.get("netdev") for d in facts["rdma"]}:
            raise ValueError("TP4 bootstrap address must identify its independent management interface")
    elif facts["fabric_ip"] != row["host_ip"] or facts["interface"] != row["interface"]:
        raise ValueError("Discovered management/fabric mapping differs from the saved site; inspect before reinitializing")
    devices = {entry["device"]: entry for entry in facts["rdma"]}
    for name in row["hcas"]:
        item = devices.get(name, {})
        if (not item.get("active") or item.get("type") != "RoCE v2" or item.get("mtu", 0) < 9000 or item.get("rdma_mtu") != 4096
                or len(item.get("ips", [])) != 1 or item.get("gid_ip") not in item.get("ips", [])
                or row["management_ip"] in item.get("ips", [])):
            raise ValueError(name + ": prepare the expected link, MTU and IPv4 RoCE-v2 GID index 3 before deployment")


def check_workloads(facts, lock, number, *, managed_prepared=False):
    containers, processes = facts["gpu_containers"], facts["gpu_process_ancestors"]
    if not containers and not processes:
        return
    expected = (lock["site"]["name"] if lock["backend"] == "glm-managed" else "sr-" + lock["site"]["name"]) + f"-r{number}"
    if len(containers) == 1:
        entry = containers[0]
        identity = (entry["labels"].get("io.sparkring.deployment") == lock["id"] if lock["backend"] == "compose"
                    else managed_prepared and entry["labels"].get("io.sparkring.container-spec") == "glm-tp4/v1")
        if (entry["name"] == expected and entry["image"] == lock["selection"]["image_id"] and identity
                and all(entry["pid"] in chain for chain in processes)):
            return
    raise ValueError("Another GPU workload is running. Stop only the intended workload in the approved test window, then rerun up; no assets were downloaded")


class Runner:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.lock = installer.load(self.directory)
        revision = distribution.identity(installer.ROOT)
        if revision != self.lock["source_revision"]:
            raise ValueError("Use the clean controller checkout recorded in deployment.lock.json; the source bundle preserves it")

    def remote(self, number, operation, *, data=None):
        row = self.lock["site"]["ranks"][number]
        payload = base64.b64encode(json.dumps(self.lock).encode()).decode()
        prefix = ["sudo", "-n"] if "fabric" in row and (operation in ("preflight", "create", "start") or operation.startswith("mesh-")) else []
        return json.loads(ssh(row["host"], prefix + ["python3", "-I", "-B", "-c", HOST,
                          row["repository"], payload, operation, str(number)], data=data))

    def native_mesh(self, operation):
        from scripts.deploy_stage import prepare_secrets
        directory = self.directory / "native-mesh"
        record = directory / "installation.json"
        if operation == "mesh-installed":
            if not record.exists():
                raise ValueError("Native mesh installation is incomplete")
            for number in range(4):
                self.remote(number, "mesh-installed-local")
            return {"ok": True}
        if record.exists():
            return self.native_mesh("mesh-installed")
        epoch = prepare_secrets(directory)
        containers = [self.remote(rank, "container-record") for rank in range(4)]
        payload = json.dumps({"epoch": epoch, "key": base64.b64encode((directory / "health.key").read_bytes()).decode(),
                              "containers": containers}).encode()
        for rank in range(4):
            self.remote(rank, "mesh-install-local", data=payload)
        installer.write(record, {"deployment": self.lock["id"], "container_ids": [c["Id"] for c in containers]})
        return {"ok": True}

    def managed(self, operation):
        from scripts import deploy_runtime, deploy_stage, deploy_suite
        lock, directory = self.lock, self.directory / "managed"
        directory.mkdir(mode=0o700, exist_ok=True)
        prepared_path = directory / "runtime/prepared.json"
        if operation in ("managed-prepare", "managed-prepared"):
            if prepared_path.exists():
                deploy_suite.check_network(installer.read(prepared_path))
                return {"ok": True}
            if operation == "managed-prepared":
                raise ValueError("Managed staging is incomplete; inspect its stage receipt")
            receipt = self.remote(0, "receipt")
            from runtime.common import glm_native_candidate
            glm_native_candidate.validate_receipt(receipt)
            image_path = directory / "image.json"
            if not image_path.exists():
                installer.write(image_path, receipt)
            rows = lock["site"]["ranks"]
            inventory = deploy_suite.discover([r["host"] + "=" + r["management_ip"] for r in rows], lock["site"]["controller_address"])
            spec = deploy_suite.create_spec(inventory, lock["site"]["name"], installer.managed_workspace(lock["site"]["name"]),
                                            "198.18.0.0/21", image_path, reuse_existing_image=True,
                                            existing_model_roots=[r["model"] for r in rows],
                                            runtime_profile="tp4-dcp1-sparkcache" if lock["selection"]["sparkcache"] else "tp4-dcp1",
                                            target_model_variant=lock["selection"]["target_variant"], preserve_existing_network=True)
            prep = {"schema": "sparkring-deploy-preparation/v1", "spec": spec,
                    "network_plan": deploy_suite.plan_network(spec, inventory["hosts"]),
                    "model_started": False, "lifecycle_capabilities": deploy_suite.lifecycle_capabilities()}
            verified = deploy_suite.check_network(prep)
            deploy_stage.stage(verified, directory / "runtime")
            return {"ok": True}
        mapping = {"managed-created": "create", "managed-installed": "install", "managed-up-check": "up",
                   "managed-native-checked": "native-check", "managed-running": "start", "managed-stopped": "stop"}
        action = mapping.get(operation, operation.removeprefix("managed-"))
        preparation = installer.read(prepared_path)
        plan = deploy_runtime.build_runtime_plan(preparation, action, container_backend="compose" if action == "create" else "docker")
        generation = installer.read(self.directory / "state.json")["generation"]
        # Retain one initialization receipt; subsequent sessions recheck rather
        # than recreate containers or install over existing managed services.
        key = "initial" if action in ("create", "install", "up", "native-check") else str(generation)
        path = directory / f"{key}-{action}.json"
        deploy_engine.execute_plan(plan, path, plan["sha256"], resume=path.exists(),
                                   allow_model_actions=True, allow_hardware_tests=True)
        return {"ok": True}

    def __call__(self, target, argv, timeout):
        from runtime.host import progress
        labels = {"prerequisites": "Check host and GPU availability", "prepare-prerequisites": "Check host before preparing assets", "source": "Copy installer source",
                  "source-check": "Verify installer source", "image": "Prepare pinned image", "image-check": "Verify image",
                  "model": "Prepare checkpoint and verify all shards", "model-check": "Verify checkpoint receipt",
                  "preflight": "Check model and fabric", "create": "Create stopped model container", "created": "Verify model container",
                  "start": "Start model", "running": "Check model process", "ready": "Wait for API readiness",
                  "smoke": "Test a short model response", "mesh-prepare": "Prepare native fabric helper",
                  "mesh-install": "Install supervised native fabric", "mesh-up": "Start native fabric",
                  "mesh-gate": "Verify all four fabric ranks", "stop": "Stop model", "stopped": "Confirm model stopped"}
        operation = argv[1] if len(argv) > 1 else "operation"
        rank = argv[2] if len(argv) > 2 else "?"
        with progress.step(f"Node {rank}: {labels.get(operation, operation.replace('-', ' '))}") as outcome:
            result = self._call(target, argv, timeout)
            if result["returncode"]:
                outcome["failed"] = True
                progress.failure(result["stderr"])
            return result

    def _call(self, target, argv, timeout):
        try:
            if len(argv) != 3 or argv[0] != "installer":
                raise ValueError("Installer runner refuses arbitrary commands")
            operation, number = argv[1], int(argv[2])
            row = self.lock["site"]["ranks"][number]
            if target != row["host"]:
                raise ValueError("Plan host differs from its locked rank")
            if operation in ("prerequisites", "prepare-prerequisites"):
                facts = json.loads(ssh(target, ["python3", "-I", "-B", "-c", PROBE], timeout=120))
                check_facts(facts, row)
                if operation == "prerequisites":
                    check_workloads(facts, self.lock, number,
                                    managed_prepared=(self.directory / "managed/runtime/prepared.json").is_file())
                result = {"ok": True}
            elif operation in ("source", "source-check"):
                ssh(target, ["python3", "-I", "-B", "-c", SOURCE, self.lock["site"]["workspace"],
                             row["repository"], self.lock["source_revision"], self.lock["id"],
                             self.lock["bundle_sha256"], "install" if operation == "source" else "check"],
                    data=(self.directory / "source.bundle").read_bytes() if operation == "source" else None)
                result = {"ok": True}
            elif operation.startswith("managed-"):
                with contextlib.redirect_stdout(sys.stderr):
                    result = self.managed(operation)
            elif operation in ("mesh-install", "mesh-installed"):
                with contextlib.redirect_stdout(sys.stderr):
                    result = self.native_mesh(operation)
            elif operation == "status":
                managed = self.lock["backend"] == "glm-managed"
                staged = ssh(target, ["python3", "-I", "-c", "from pathlib import Path; import sys; print(Path(sys.argv[1]).is_dir())", row["repository"]]).strip() == "True"
                if not managed and staged:
                    result = self.remote(number, "status")
                    return {"returncode": 0, "stdout": json.dumps(result), "stderr": "", "uncertain": False}
                name = ("" if managed else "sr-") + self.lock["site"]["name"] + f"-r{number}"
                key = "io.sparkring.container-spec" if managed else "io.sparkring.deployment"
                value = "glm-tp4/v1" if managed else self.lock["id"]
                result = json.loads(ssh(target, ["python3", "-I", "-B", "-c", STATUS, name,
                                                self.lock["selection"]["image_id"], key, value, str(number)], timeout=120))
            else:
                result = self.remote(number, operation)
            return {"returncode": 0, "stdout": "ok" if result.get("ok") is True else json.dumps(result),
                    "stderr": "", "uncertain": False}
        except subprocess.TimeoutExpired:
            return {"returncode": 124, "stdout": "", "stderr": "Remote operation timed out; inspect host state before recovery", "uncertain": True}
        except (ValueError, RuntimeError, OSError, KeyError, subprocess.SubprocessError) as error:
            return {"returncode": 1, "stdout": "", "stderr": str(error), "uncertain": False}
