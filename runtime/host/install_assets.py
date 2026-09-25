"""Prepare head/worker packages and cached images through verified fabric paths."""
import concurrent.futures
import inspect
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time

from runtime.common import distribution
from runtime.host import node, packages, progress
from runtime.host.install_errors import NeedsInput


def image_probe(image):
    import json
    import shutil
    import subprocess
    docker = ["docker", "--context", "default"]
    root = subprocess.check_output([*docker, "info", "--format", "{{.DockerRootDir}}"], text=True).strip()
    found = subprocess.run([*docker, "image", "inspect", image], capture_output=True, text=True)
    record = json.loads(found.stdout)[0] if found.returncode == 0 else None
    return {"present": record is not None, "image_id": record["Id"] if record else None,
            "size_bytes": record["Size"] if record else None,
            "free_bytes": shutil.disk_usage(root).free}


def storage_probe():
    import shutil
    import subprocess
    root = subprocess.check_output(["docker", "--context", "default", "info", "--format", "{{.DockerRootDir}}"], text=True).strip()
    return shutil.disk_usage(root).free


def receive_image(image, reserve):
    """Import authenticated stdin directly, without creating a second archive copy."""
    import json
    import shutil
    import subprocess
    import sys
    import tempfile
    docker = ["docker", "--context", "default"]
    root = subprocess.check_output([*docker, "info", "--format", "{{.DockerRootDir}}"], text=True).strip()
    if shutil.disk_usage(root).free < reserve:
        raise ValueError("Image import space changed; existing model remains untouched")
    with tempfile.TemporaryFile() as details:
        result = subprocess.run([*docker, "image", "load"], stdin=sys.stdin.buffer, stdout=details, stderr=details)
        if result.returncode:
            details.seek(0)
            raise ValueError("Image import failed: " + details.read()[-4000:].decode(errors="replace"))
    actual = json.loads(subprocess.check_output([*docker, "image", "inspect", image], text=True))[0]
    if actual["Id"] != image or actual["Architecture"] != "arm64" or actual["Os"] != "linux":
        raise ValueError("Imported image does not match the selected ARM64 identity")
    return {"image_id": image, "free_bytes": shutil.disk_usage(root).free, "model_started": False}


def worker_revision():
    import json
    from pathlib import Path
    path = Path("/usr/lib/sparkring/distribution.json")
    return json.loads(path.read_text())["revision"] if path.exists() else None


class Assets:
    def __init__(self, transport, directory, *, run=subprocess.run, popen=subprocess.Popen):
        self.transport = transport
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.run, self.popen = run, popen

    def command(self, rank, argv):
        return self.transport.command(rank, argv if rank == 0 else ["sudo", "-n", *argv])

    def remote(self, rank, function, *args, **kwargs):
        code = inspect.getsource(function) + "\nimport json\nprint(json.dumps(" + function.__name__ + "(*" + repr(args) + ", **" + repr(kwargs) + ")))\n"
        result = self.run(self.command(rank, ["python3", "-I", "-c", code]), capture_output=True, text=True, timeout=7200, check=True)
        return json.loads(result.stdout)

    def sync_packages(self):
        current = distribution.identity(node.ROOT)
        outdated = [rank for rank in range(1, len(self.transport.hosts)) if self.remote(rank, worker_revision) != current]
        if not outdated:
            return {"updated": [], "revision": current}
        # Existing enrolled access is retained; this bundle does not invoke seed.
        archive = packages.build(self.directory / ("worker-" + str(time.time_ns())), "")
        for rank in outdated:
            target = "/var/tmp/sparkring-enroll-update-" + current[:12] + "-" + str(time.time_ns())
            with progress.step(f"Node {rank}: Update SparkRing from Node A's package"):
                packages.transfer(self.transport, rank, archive, target)
                progress.command(self.command(rank, ["python3", "-I", target + "/install.py", "--apply"]),
                                 title=f"Node {rank}: Install verified package bundle", invoke=self.run, check=True)
                if self.remote(rank, worker_revision) != current:
                    raise ValueError(f"Node {rank}: installed package revision differs")
        return {"updated": outdated, "revision": current}

    def images(self, card):
        count = len(self.transport.hosts)
        observations = [self.remote(rank, image_probe, card["image_id"]) for rank in range(count)]
        present = [rank for rank, value in enumerate(observations) if value["present"]]
        if not present:
            if card["image_reference"] == card["image_id"]:
                raise NeedsInput("The selected private image is not cached on any enrolled node. Supply a registry-backed image lock or load its archive on Node A.",
                                 field="image_lock", details={"image_id": card["image_id"]})
            from runtime.common import profiles
            policy = profiles.read_json(node.ROOT / "profiles/storage-planning.json")
            if "image_bytes" in card:
                # Compressed layers and the unpacked image coexist during a pull.
                reserve = card["image_bytes"] + card["download_bytes"] + 8 * 1024**3
            else:
                reserve = (policy["image_allowance_gib"] + policy["cache_and_jit_allowance_gib"]) * 1024**3
            if self.remote(0, storage_probe) < reserve:
                raise NeedsInput("Node A needs more Docker storage before the pinned image can be downloaded.",
                                 field="storage", details={"rank": 0, "required_bytes": reserve})
            with progress.step("Node 0: Download pinned image once for the cluster"):
                progress.command(["docker", "--context", "default", "pull", "--platform", "linux/arm64", card["image_reference"]],
                                 title="Fetch selected image", invoke=self.run, check=True)
            observations[0] = self.remote(0, image_probe, card["image_id"])
            if not observations[0]["present"]:
                raise ValueError("Downloaded image differs from the selected configuration ID")
            present = [0]
        donor = present[0]
        size = observations[donor]["size_bytes"]
        required = max(2 * size + 4 * 1024**3, size + 32 * 1024**3)
        missing = [rank for rank, value in enumerate(observations) if not value["present"]]
        for rank in missing:
            if observations[rank]["free_bytes"] < required:
                free = self.remote(rank, storage_probe)
                if free < required:
                    raise NeedsInput(f"Node {rank}: insufficient image-import space. Free space or choose a larger Docker data volume, then repeat the command. Current model has not been stopped.",
                                     field="storage", details={"rank": rank, "required_bytes": required, "free_bytes": free})
        def transfer(rank):
            code = inspect.getsource(receive_image) + "\nimport json\nprint(json.dumps(receive_image(" + repr(card["image_id"]) + "," + str(required) + ")))\n"
            with progress.step(f"Node {rank}: Transfer pinned image over {self.transport.mode}"):
                with tempfile.TemporaryFile() as errors:
                    source = self.popen(self.command(donor, ["docker", "--context", "default", "image", "save", card["image_id"]]), stdout=subprocess.PIPE, stderr=errors)
                    try:
                        target = self.run(self.command(rank, ["python3", "-I", "-c", code]), stdin=source.stdout,
                                          capture_output=True, timeout=7200)
                        source.stdout.close()
                        source.wait(timeout=30)
                        if target.returncode or source.returncode:
                            errors.seek(0)
                            raise ValueError((target.stderr or errors.read()).decode(errors="replace")[-4000:])
                        return {"rank": rank, **json.loads(target.stdout)}
                    finally:
                        if source.stdout and not source.stdout.closed:
                            source.stdout.close()
                        if source.poll() is None:
                            source.terminate()
                            try:
                                source.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                source.kill()
                                source.wait()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(missing))) as pool:
            imported = list(pool.map(transfer, missing))
        result = {"image_id": card["image_id"], "donor_rank": donor, "reused_ranks": present, "imported": imported}
        node.save(self.directory, "images.json", result, mode=0o600)
        return result

    def models(self, lock, runner, previous=None):
        """Verify one cached/downloaded checkpoint, then fill missing peers via rsync."""
        import shlex
        rows = lock["site"]["ranks"]
        if previous:
            from runtime.common import installer
            saved = installer.read(Path(previous) / "deployment.lock.json")
            if [r["host"] for r in saved["site"]["ranks"]] == [r["host"] for r in rows]:
                payload = json.dumps({"workspace": saved["site"]["workspace"], "deployment": saved["id"]}).encode()
                for row in rows:
                    runner.remote(row["rank"], "model-reuse-receipt", data=payload)
        cached = [r["rank"] for r in rows if runner.remote(r["rank"], "model-present")["present"]]
        donor = cached[0] if cached else 0
        runner.remote(donor, "model")
        manifest = runner.remote(donor, "model-transfer-manifest")
        names = sorted(manifest["files"])
        listing = self.directory / "checkpoint-files.txt"
        listing.write_text("\n".join(names) + "\n", encoding="utf-8")
        missing = [r["rank"] for r in rows if r["rank"] not in cached and r["rank"] != donor]

        def copy(source, target):
            runner.remote(target, "model-transfer-prepare", data=json.dumps(manifest).encode())
            remote = source if source != 0 else target
            ssh = shlex.join(self.transport.argv(remote)[:-1])
            alias = self.transport.argv(remote)[-1]
            origins = (alias + ":" if source != 0 else "") + rows[source]["model"] + "/"
            destination = (alias + ":" if target != 0 else "") + rows[target]["model"] + "/"
            with progress.step(f"Node {target}: Copy verified checkpoint over {self.transport.mode}"):
                progress.command(["rsync", "-rlt", "--partial", "--checksum", "--protect-args", "--rsync-path=sudo -n rsync",
                                  "--files-from=" + str(listing), "-e", ssh, origins, destination],
                                 title=f"Node {target}: Transfer checkpoint files", invoke=self.run, check=True)
            runner.remote(target, "model-transfer-complete", data=json.dumps(manifest).encode())

        # The head is the relay for the opposite side of the ring, never the PC.
        if 0 not in cached and donor != 0:
            copy(donor, 0)
            missing.remove(0)
        for rank in missing:
            copy(0, rank)
        return {"donor_rank": donor, "cached_ranks": cached, "copied_ranks": [r for r in range(len(rows)) if r not in cached and r != donor]}

    def runner(self, directory, previous=None):
        from scripts.installer_runner import Runner
        assets = self
        class PreparedRunner(Runner):
            def __init__(self, directory):
                super().__init__(directory)
                self.model_lock = threading.Lock()
                self.models_prepared = False

            def _call(self, target, argv, timeout):
                if argv[1] == "model":
                    with self.model_lock:
                        if not self.models_prepared:
                            assets.models(self.lock, self, previous)
                            self.models_prepared = True
                return super()._call(target, argv, timeout)
        return PreparedRunner(directory)
