"""Prepare head/worker packages and cached images through verified fabric paths."""
import concurrent.futures
import inspect
import json
from pathlib import Path
import random
import re
import secrets
import subprocess
import tempfile
import threading
import time

from runtime.common import distribution
from runtime.host import fabric_stream, node, packages, progress, registry_relay
from runtime.host.install_errors import NeedsInput

DOCKER = ["docker", "--context", "default"]


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


def layer_prefix(diff_ids):
    """Count the image's leading layers that this node's Docker already holds.

    The graph-driver image store identifies a layer by the chain of
    uncompressed layer digests (diff IDs) beneath it, so any local image whose
    layers begin with the same diff IDs proves that chain is present, however
    it arrived. The containerd image store loads only complete archives and
    reports zero.
    """
    import json
    import subprocess
    docker = ["docker", "--context", "default"]
    status = subprocess.check_output([*docker, "info", "--format", "{{json .DriverStatus}}"], text=True)
    if "io.containerd.snapshotter" in status:
        return 0
    images = sorted(set(subprocess.check_output([*docker, "image", "ls", "-a", "-q", "--no-trunc"], text=True).split()))
    shared = 0
    for record in json.loads(subprocess.check_output([*docker, "image", "inspect", *images], text=True)) if images else []:
        count = 0
        for held, wanted in zip(record.get("RootFS", {}).get("Layers") or [], diff_ids):
            if held != wanted:
                break
            count += 1
        shared = max(shared, count)
    return shared


def write_layer_archive(stream, image_id, config, layers, present, blob, tag=None):
    """Write a ``docker load`` archive holding only the layers after ``present``.

    The manifest lists every layer. Docker's loader looks up each leading chain
    in the local store and opens a layer file only when that chain is absent;
    it decompresses each loaded registry blob and rejects any whose content
    differs from the configuration's diff ID.
    """
    import io
    import tarfile

    def add(name, handle, size):
        info = tarfile.TarInfo(name)
        info.size, info.mode = size, 0o644
        archive.addfile(info, handle)

    names = [diff.removeprefix("sha256:") + "/layer.tar" for diff, _, _ in layers]
    manifest = [{"Config": image_id.removeprefix("sha256:") + ".json", "RepoTags": [tag] if tag else None,
                 "Layers": names}]
    with tarfile.open(fileobj=stream, mode="w|") as archive:
        add(manifest[0]["Config"], io.BytesIO(config), len(config))
        encoded = json.dumps(manifest).encode()
        add("manifest.json", io.BytesIO(encoded), len(encoded))
        for name, (_, digest, _) in list(zip(names, layers))[present:]:
            path = Path(blob(digest))
            with open(path, "rb") as handle:
                add(name, handle, path.stat().st_size)


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

    @staticmethod
    def code(function, *args, **kwargs):
        """Python source that runs a self-contained function and prints its JSON result."""
        return inspect.getsource(function) + "\nimport json\nprint(json.dumps(" + function.__name__ + "(*" + repr(args) + ", **" + repr(kwargs) + ")))\n"

    def remote(self, rank, function, *args, **kwargs):
        result = self.run(self.command(rank, ["python3", "-I", "-c", self.code(function, *args, **kwargs)]),
                          capture_output=True, text=True, timeout=7200, check=True)
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

    def relay(self, card, missing, observations):
        """Pull on every node that lacks the image, through one upstream download on Node A."""
        from runtime.common import profiles
        policy = profiles.read_json(node.ROOT / "profiles/storage-planning.json")
        if "image_bytes" in card:
            # Compressed layers and the unpacked image coexist during a pull.
            pull = card["image_bytes"] + card["download_bytes"] + 8 * 1024**3
            cache = card["download_bytes"]
        else:
            pull = (policy["image_allowance_gib"] + policy["cache_and_jit_allowance_gib"]) * 1024**3
            cache = policy["image_allowance_gib"] * 1024**3
        for rank in sorted({0, *missing}):
            required = (pull if rank in missing else 0) + (cache if rank == 0 else 0)
            if observations[rank]["free_bytes"] < required:
                raise NeedsInput(f"Node {rank} needs more Docker storage before the pinned image can be downloaded. Free space, then repeat sudo sparkring install. A running model has not been stopped.",
                                 field="storage", details={"rank": rank, "required_bytes": required,
                                                           "free_bytes": observations[rank]["free_bytes"]})
        relay = registry_relay.Relay(card["image_reference"], self.directory / "relay")
        try:
            layout = relay.image(card["image_id"])
        except (OSError, ValueError, KeyError) as error:
            # Every node then pulls; a pull reports its own registry failure.
            progress.say(f"Pinned layer list unavailable ({error}); nodes pull the whole image.")
            layout = None
        except BaseException:
            relay.close()
            raise

        def pull_on(rank):
            present = self.remote(rank, layer_prefix, [diff for diff, _, _ in layout[1]]) if layout else 0
            if present:
                # Docker's pull re-downloads a held layer unless it recorded that
                # layer's registry digest, which layers imported by `docker load`
                # or built locally lack; the node instead loads only what it lacks.
                missing = len(layout[1]) - present
                title = (f"Node {rank}: Load {missing} missing of {len(layout[1])} image layers through Node A's registry relay"
                         if missing else f"Node {rank}: Register the pinned image from layers the node already holds")
                with progress.step(title):
                    self.load_layers(rank, relay, card, layout, present)
                    if not self.remote(rank, image_probe, card["image_id"])["present"]:
                        raise ValueError(f"Node {rank}: layer load produced a different image")
                return rank
            with progress.step(f"Node {rank}: Pull pinned image through Node A's registry relay"):
                if rank == 0:
                    argv = [*DOCKER, "pull", "-q", "--platform", "linux/arm64", relay.reference()]
                    self.run(argv, capture_output=True, text=True, timeout=7200, check=True)
                else:
                    # The worker's Docker reaches the relay on its own loopback
                    # address; another service holding that port selects a free one.
                    for port in (relay.port, *random.sample(range(20000, 60000), 3)):
                        pull = ["sudo", "-n", *DOCKER, "pull", "-q", "--platform", "linux/arm64", relay.reference(port)]
                        result = self.run(self.transport.forwarded(rank, port, relay.port, pull),
                                          capture_output=True, text=True, timeout=7200)
                        if result.returncode == 0 or "forwarding failed" not in result.stderr:
                            break
                    if result.returncode:
                        raise ValueError(f"Node {rank}: relay pull failed: " + result.stderr.strip()[-2000:])
                if not self.remote(rank, image_probe, card["image_id"])["present"]:
                    raise ValueError(f"Node {rank}: relay pull produced a different image")
            return rank

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(missing)) as pool:
                return list(pool.map(pull_on, missing))
        finally:
            relay.close()

    def load_layers(self, rank, relay, card, layout, present):
        """Stream the image configuration and the layers after ``present`` into one node's ``docker load``."""
        config, layers = layout
        # Loaded blobs, their unpacked layers and the loader's extracted copy coexist.
        reserve = 4 * sum(size for _, _, size in layers[present:]) + 4 * 1024**3
        release = card.get("release", "")
        tag = (f"127.0.0.1:{registry_relay.PORT}/{relay.repository}:{release}"
               if re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", release) else None)
        with tempfile.TemporaryFile() as errors:
            child = self.popen(self.command(rank, ["python3", "-I", "-c", self.code(receive_image, card["image_id"], reserve)]),
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors)
            try:
                write_layer_archive(child.stdin, card["image_id"], config, layers, present, relay.blob, tag)
                child.stdin.close()
            except BrokenPipeError:
                pass  # The receiver stopped early; its own error is reported below.
            finally:
                if not child.stdin.closed:
                    try:
                        child.stdin.close()
                    except BrokenPipeError:
                        pass
            child.stdout.read()
            if child.wait(timeout=7200):
                errors.seek(0)
                raise ValueError(f"Node {rank}: layer load failed: " + errors.read()[-4000:].decode(errors="replace"))

    def images(self, card):
        count = len(self.transport.hosts)
        observations = [self.remote(rank, image_probe, card["image_id"]) for rank in range(count)]
        missing = [rank for rank, value in enumerate(observations) if not value["present"]]
        if missing and card["image_reference"] != card["image_id"]:
            try:
                pulled = self.relay(card, missing, observations)
                result = {"image_id": card["image_id"], "relayed_ranks": pulled,
                          "reused_ranks": [r for r in range(count) if r not in missing]}
                node.save(self.directory, "images.json", result, mode=0o600)
                return result
            except NeedsInput:
                raise
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                # Pulls fail when the registry needs credentials or is
                # unreachable; nodes holding the image then serve the others.
                progress.say(f"Registry relay unavailable ({error}); copying the image between nodes.")
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
                raise NeedsInput("Node A needs more Docker storage before the pinned image can be downloaded. Free space, then repeat sudo sparkring install. A running model has not been stopped.",
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
                    raise NeedsInput(f"Node {rank}: insufficient image-import space. Free space or choose a larger Docker data volume, then repeat sudo sparkring install. A running model has not been stopped.",
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

    def stream_checkpoint(self, runner, rows, manifest, source, target):
        """Copy the verified checkpoint from ``source`` to cable-adjacent ``target`` over the fabric."""
        pairs = fabric_stream.links(self.transport.hosts, source, target)
        if not pairs:
            raise ValueError(f"Node {source} and Node {target} share no fabric subnet")
        runner.remote(target, "model-transfer-prepare", data=json.dumps(manifest).encode())
        files = {name: [manifest["sizes"][name], digest] for name, digest in manifest["files"].items()}
        mine, theirs = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
        token = secrets.token_bytes(32)
        receive = self.code(fabric_stream.receive, theirs, mine, rows[target]["model"], files)
        with progress.step(f"Node {target}: Copy verified checkpoint from Node {source} over the fabric"), \
                tempfile.TemporaryFile() as errors:
            receiver = self.popen(self.command(target, ["python3", "-I", "-c", receive]),
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors)
            try:
                receiver.stdin.write(token)
                receiver.stdin.close()
                line = receiver.stdout.readline()
                offer = json.loads(line) if line else None
                if offer and offer["needed"]:
                    groups = fabric_stream.balance(offer["needed"], manifest["sizes"], len(pairs))
                    send = self.code(fabric_stream.send, mine, theirs, offer["ports"], rows[source]["model"], groups)
                    sender = self.run(self.command(source, ["python3", "-I", "-c", send]), input=token,
                                      capture_output=True, timeout=7200)
                    if sender.returncode:
                        raise ValueError("sender: " + sender.stderr.decode(errors="replace").strip()[-1000:])
                receiver.stdout.read()
                if receiver.wait(timeout=600) or offer is None:
                    errors.seek(0)
                    raise ValueError("receiver: " + errors.read().decode(errors="replace").strip()[-1000:])
            finally:
                if receiver.poll() is None:
                    receiver.kill()
                    receiver.wait()
        runner.remote(target, "model-transfer-complete", data=json.dumps(manifest).encode())
        return target

    def models(self, lock, runner, previous=None):
        """Verify one cached/downloaded checkpoint, then fill missing peers.

        Copies use direct fabric streams along the cables; ranks that a stream
        could not fill are copied with rsync over the administration SSH path.
        """
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
        have = {donor, *cached}
        if missing:
            # Copies follow the cables outward from the donor, each level in
            # parallel; a failed direct copy leaves the rest to the SSH path.
            try:
                for level in fabric_stream.tree(len(rows), donor):
                    level = [(s, t) for s, t in level if t not in have]
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(level))) as pool:
                        for target in pool.map(lambda edge: self.stream_checkpoint(runner, rows, manifest, *edge), level):
                            have.add(target)
            except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
                progress.say(f"Direct fabric checkpoint copy unavailable ({error}); copying over {self.transport.mode}.")

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
        if 0 not in have:
            copy(donor, 0)
            have.add(0)
        for rank in missing:
            if rank not in have:
                copy(0, rank)
        return {"donor_rank": donor, "cached_ranks": cached, "copied_ranks": [r for r in range(len(rows)) if r not in cached and r != donor]}

    def runner(self, directory, previous=None, images=None):
        """Runner whose checkpoint and image phases wait for ``images``, a pending fan-out.

        Checkpoint work needs the serving image: a Hugging Face download, and a
        repair of a reused copy, run the image's own client with ``--pull
        never``. The checkpoint phase therefore starts only after every node
        holds the image.
        """
        from scripts.installer_runner import Runner
        assets = self

        def failed(message):
            return {"returncode": 1, "stdout": "", "stderr": message, "uncertain": False}

        class PreparedRunner(Runner):
            def __init__(self, directory):
                super().__init__(directory)
                self.model_lock = threading.Lock()
                self.models_prepared = False
                self.models_error = None

            def _call(self, target, argv, timeout):
                if argv[1] in ("model", "image") and images is not None and images.exception() is not None:
                    # The caller re-raises the fan-out error itself, so a request
                    # for input keeps its type.
                    return failed("Image distribution failed: " + str(images.exception()))
                if argv[1] == "model":
                    with self.model_lock:
                        if not self.models_prepared and self.models_error is None:
                            try:
                                assets.models(self.lock, self, previous)
                                self.models_prepared = True
                            except Exception as error:  # noqa: BLE001 - reported as the phase's failure
                                # A returned failure is recorded in the operation
                                # receipt, where a raised error would leave every
                                # rank's action recorded as running. Other ranks
                                # report the same failure instead of repeating it.
                                self.models_error = "Checkpoint preparation failed: " + str(error)
                        if self.models_error is not None:
                            return failed(self.models_error)
                return super()._call(target, argv, timeout)
        return PreparedRunner(directory)
