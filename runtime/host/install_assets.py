"""Prepare head/worker packages, cached images and the pinned checkpoint through verified fabric paths."""
import concurrent.futures
import inspect
import json
from pathlib import Path
import posixpath
import random
import re
import secrets
import shlex
import subprocess
import tempfile
import threading
import time

from runtime.common import distribution, installer
from runtime.host import checkpoint_plan, fabric_stream, node, packages, progress, registry_relay
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

    def stream_checkpoint(self, runner, rows, manifest, source, target, names=None):
        """Copy ``names`` (default: every name of ``manifest``) from ``source`` to cable-adjacent ``target``.

        ``manifest`` holds the pinned ``files`` (SHA-256) and ``sizes`` of the
        names; its subset for ``names`` is the transfer manifest of the target's
        ``model-transfer-prepare`` and ``model-transfer-complete``, so the sender
        needs no receipt and the same call serves pooling and ring receives. The
        target runs ``fabric_stream.receiver_source``, which places each file
        only after its SHA-256 equals the pin. Returns the result of
        ``model-transfer-complete``.
        """
        subset = transfer_manifest(manifest, names)
        pairs = fabric_stream.links(self.transport.hosts, source, target)
        if not pairs:
            raise ValueError(f"Node {source} and Node {target} share no fabric subnet")
        data = json.dumps(subset).encode()
        files = {name: [subset["sizes"][name], digest] for name, digest in subset["files"].items()}
        mine, theirs = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
        token = secrets.token_bytes(32)
        receive = fabric_stream.receiver_source(theirs, mine, rows[target]["model"], subset["repository"],
                                                subset["revision"], files)
        with progress.step(f"Node {target}: Copy checkpoint files from Node {source} over the fabric"), \
                tempfile.TemporaryFile() as errors:
            operation(runner, target, "model-transfer-prepare", data)
            receiver = self.popen(self.command(target, ["python3", "-I", "-c", receive]),
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors)
            try:
                receiver.stdin.write(token)
                receiver.stdin.close()
                line = receiver.stdout.readline()
                offer = json.loads(line) if line else None
                if offer and not set(offer["needed"]) <= set(files):
                    raise ValueError(f"Node {target}: the checkpoint receiver asked for unplanned files")
                if offer and offer["needed"]:
                    groups = fabric_stream.balance(offer["needed"], subset["sizes"], len(pairs))
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
            return operation(runner, target, "model-transfer-complete", data)

    def copy(self, runner, rows, manifest, source, target, names=None):
        """Copy ``names`` from ``source`` to ``target`` with rsync over the administration path.

        One end is Node A, which runs rsync; the other is reached through the
        transport's SSH command. rsync writes only into the target's empty
        staging directory ``receive/rsync/`` (``rsync_staging``), never into the
        checkpoint directory, and transfers only the names that
        ``model-transfer-prepare`` reports as needed, which it reads on stdin. It
        runs without ``-t``, ``-a``, ``--inplace``, ``--append``, ``--partial``
        and ``--checksum``; ``model-transfer-complete`` then hashes each staged
        file and places it. Returns the result of ``model-transfer-complete``.
        """
        if 0 not in (source, target) or source == target:
            raise ValueError("rsync copies run between Node A and one other Spark")
        subset = transfer_manifest(manifest, names)
        data = json.dumps(subset).encode()
        with progress.step(f"Node {target}: Copy checkpoint files from Node {source} over {self.transport.mode}"):
            prepared = operation(runner, target, "model-transfer-prepare", data)
            needed = sorted(subset["files"])
            if isinstance(prepared, dict) and isinstance(prepared.get("needed"), list):
                needed = sorted(set(prepared["needed"]) & set(subset["files"]))
            if needed:
                remote = source if source != 0 else target
                ssh = shlex.join(self.transport.argv(remote)[:-1])
                alias = self.transport.argv(remote)[-1]
                origin = (alias + ":" if source != 0 else "") + rows[source]["model"] + "/"
                destination = (alias + ":" if target != 0 else "") + rsync_staging(rows[target]["model"]) + "/"
                progress.command([*RSYNC, "--files-from=-", "-e", ssh, origin, destination],
                                 title=f"Node {target}: Transfer checkpoint files", invoke=self.run, check=True,
                                 input=("\n".join(needed) + "\n").encode())
            return operation(runner, target, "model-transfer-complete", data)

    def assemble(self, runner, rows, manifest, item):
        """Pool ``item["names"]`` into Node A from ``item["source"]``: fabric when cable-adjacent, else rsync.

        A failed fabric stream falls back to rsync. Returns the result of
        ``model-transfer-complete`` and the transport used.
        """
        if item["transport"] == "fabric":
            try:
                return self.stream_checkpoint(runner, rows, manifest, item["source"], 0, item["names"]), "fabric"
            except NeedsInput:
                raise
            except FALLBACK as error:
                progress.say(f"Direct fabric checkpoint copy from Node {item['source']} unavailable ({error}); "
                             f"copying over {self.transport.mode}.")
        return self.copy(runner, rows, manifest, item["source"], 0, item["names"]), "rsync"

    def distribute(self, runner, rows, manifest, donor, receives, complete):
        """Send each receive of ``receives`` along the cables, level by level, then rsync what remains.

        Streams of one level run in parallel. After a failed stream the fabric
        is not used again; the remaining Sparks are copied with rsync through
        Node A, which first receives from ``donor`` when it lacks files.
        ``complete`` gains every Spark that completes. Returns what was sent.
        """
        def stream(item):
            try:
                return self.stream_checkpoint(runner, rows, manifest, item["source"], item["target"], item["names"])
            except NeedsInput:
                raise
            except FALLBACK as error:
                return error

        pending, sent = sorted(receives, key=lambda item: (item["level"], item["target"])), []
        for level in sorted({item["level"] for item in pending}):
            batch = [item for item in pending if item["level"] == level]
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as pool:
                outcomes = list(pool.map(stream, batch))
            failure = None
            for item, outcome in zip(batch, outcomes):
                if isinstance(outcome, BaseException):
                    failure = failure or outcome
                    continue
                finished(outcome, item["target"], f"receiving from Node {item['source']}")
                complete.add(item["target"])
                pending.remove(item)
                sent.append({**item, "transport": "fabric"})
            if failure is not None:
                progress.say(f"Direct fabric checkpoint copy unavailable ({failure}); copying over {self.transport.mode}.")
                break
        # The head is the relay for the opposite side of the ring, never the PC.
        for item in sorted(pending, key=lambda item: (item["target"] != 0, item["level"], item["target"])):
            source = donor if item["target"] == 0 else 0
            if source not in complete:
                raise ValueError(f"Node {source} holds no complete checkpoint to copy to Node {item['target']}")
            outcome = self.copy(runner, rows, manifest, source, item["target"], item["names"])
            finished(outcome, item["target"], f"receiving from Node {source}")
            complete.add(item["target"])
            sent.append({**item, "source": source, "transport": "rsync"})
        return sent

    def models(self, lock, runner, previous=None, plan=None, receipts=None):
        """Place the pinned checkpoint on every Spark within the approved checkpoint plan.

        ``plan`` is the approved ``sparkring-checkpoint-plan/v1`` document (the
        plan reviewed with ``--plan`` when one bounds this run). Without one, the
        approval is ``standing_plan``: adoption uses no source outside SparkRing's
        directories, nothing is downloaded, and each Spark may write at most the
        guard's tolerance. ``receipts`` lists, for every Spark or per rank, the
        other deployments' model receipts that adoption refreshes.

        1. The previous deployment's receipts are reused where the hosts and
           paths are equal.
        2. ``model-adopt`` runs on every Spark with an owned row, in parallel,
           with its local actions from the plan. Rows served in place are
           verified later by the ``model`` operation itself.
        3. The guard (``checkpoint_plan.unplanned``) compares what adoption left
           with the plan before any download, pooling or receive; beyond it the
           install stops with ``NeedsInput(field="checkpoint")``.
        4. Without a complete Spark, Node A pools the names other Sparks verified
           and downloads only the names no Spark holds (``model-fetch``).
        5. The lowest-numbered complete Spark is the donor; the others receive
           their missing names along the cables, with rsync through Node A as the
           fallback.

        Every Spark with an owned row ends with a receipt, so the ``model``
        operation that follows only verifies. The result is saved as
        ``checkpoint-result.json``.
        """
        rows = lock["site"]["ranks"]
        pins = installer.checkpoint_pins(lock["selection"])
        approved = plan if plan is not None else standing_plan(lock, pins)
        check_plan(approved, pins, rows)
        problems = approved.get("problems") or []
        if problems:
            # The workflow stops on these before approval; a plan carrying one approves nothing.
            raise NeedsInput(problems[0]["message"], field=problems[0]["field"], details={"problems": problems})
        count = len(rows)
        in_place = [rank for rank, row in enumerate(rows) if row.get("reuse_verified_model")]
        owned = [rank for rank in range(count) if rank not in in_place]
        if previous:
            saved = installer.read(Path(previous) / "deployment.lock.json")
            if [r["host"] for r in saved["site"]["ranks"]] == [r["host"] for r in rows]:
                payload = json.dumps({"workspace": saved["site"]["workspace"], "deployment": saved["id"]}).encode()
                for row in rows:
                    operation(runner, row["rank"], "model-reuse-receipt", payload)

        def adopt(rank):
            data = json.dumps(checkpoint_plan.adoption(approved, rank, receipts_for(receipts, rank))).encode()
            with progress.step(f"Node {rank}: " + adoption_label(approved["nodes"][rank])):
                return operation(runner, rank, "model-adopt", data)
        results = [None] * count
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(owned))) as pool:
            futures = [(rank, pool.submit(adopt, rank)) for rank in owned]
        for rank, future in futures:
            results[rank] = future.result()
        items = checkpoint_plan.unplanned(approved, results)
        if items:
            raise NeedsInput(checkpoint_plan.unplanned_message(items, approved.get("command") or checkpoint_plan.COMMAND),
                             field="checkpoint", details={"items": items})
        after = checkpoint_plan.redistribute(approved, results)
        complete = set(after["complete"])
        manifest = transfer_manifest({"repository": pins["repository"], "revision": pins["revision"],
                                      "files": {n: pins["files"][n]["sha256"] for n in approved["required"]["sizes"]},
                                      "sizes": approved["required"]["sizes"]})
        pooled = []
        if after["pool"] or after["hub"]:
            outcome = None
            for item in after["pool"]:
                outcome, transport = self.assemble(runner, rows, manifest, item)
                pooled.append({**item, "transport": transport})
            if after["hub"]:
                with progress.step("Node 0: Download missing checkpoint files from huggingface.co"):
                    outcome = operation(runner, 0, "model-fetch", json.dumps({"names": after["hub"]}).encode())
            finished(outcome, 0, "assembling it from the other Sparks and huggingface.co")
            complete.add(0)
        received = self.distribute(runner, rows, manifest, after["donor"], after["receive"], complete)
        incomplete = [rank for rank in owned if rank not in complete]
        if incomplete:
            raise ValueError("The checkpoint is incomplete on " + ", ".join(f"Node {rank}" for rank in incomplete)
                             + " after adoption and transfers; repeat sudo sparkring install")
        result = {"schema": RESULT_SCHEMA, "repository": pins["repository"], "revision": pins["revision"],
                  "approval": approved.get("approval"), "donor_rank": after["donor"], "in_place_ranks": in_place,
                  "complete_after_adoption": sorted(after["complete"]),
                  "adopted": [{"rank": rank, "complete_after_adoption": results[rank].get("complete"),
                               **{key: results[rank].get(key) for key in ("linked", "copied", "bytes_written",
                                                                          "refreshed")}}
                              for rank in owned],
                  "pooled": pooled, "downloaded": list(after["hub"]), "received": received}
        node.save(self.directory, "checkpoint-result.json", result, mode=0o600)
        return result

    def runner(self, directory, previous=None, images=None, plan=None, receipts=None):
        """Runner whose checkpoint and image phases wait for ``images``, a pending fan-out.

        Every checkpoint write (a copy, a fabric receive, rsync or a download)
        needs the serving image on its Spark, and a download runs the image's own
        Hugging Face client with ``--pull never``. The first ``model`` action
        therefore waits for the image future, then runs ``models`` once with the
        approved ``plan`` and ``receipts``; the other ranks' ``model`` actions
        report its outcome. A ``NeedsInput`` raised there is kept, with its type,
        in ``needs_input``, and every rank's action reports only that the plan
        needs a decision, so the request itself is printed once, by the caller,
        which raises ``needs_input`` when the preparation fails. Any other
        failure is kept as ``models_error``, whose text names the Spark and the
        cause. ``checkpoint`` holds the result of ``models``.
        """
        from scripts.installer_runner import Runner
        assets = self
        # ``models`` receives ``plan`` and ``receipts`` only when they are given, so
        # a replacement taking (lock, runner, previous) also serves as ``models``.
        extra = {key: value for key, value in (("plan", plan), ("receipts", receipts)) if value is not None}

        def failed(message):
            return {"returncode": 1, "stdout": "", "stderr": message, "uncertain": False}

        class PreparedRunner(Runner):
            def __init__(self, directory):
                super().__init__(directory)
                self.model_lock = threading.Lock()
                self.models_prepared = False
                self.models_error = None
                self.needs_input = None
                self.checkpoint = None

            def _call(self, target, argv, timeout):
                if argv[1] in ("model", "image") and images is not None and images.exception() is not None:
                    # The caller re-raises the fan-out error itself, so a request
                    # for input keeps its type.
                    return failed("Image distribution failed: " + str(images.exception()))
                if argv[1] == "model":
                    with self.model_lock:
                        if not self.models_prepared and self.models_error is None:
                            try:
                                self.checkpoint = assets.models(self.lock, self, previous, **extra)
                                self.models_prepared = True
                            except NeedsInput as error:
                                self.needs_input = error
                                self.models_error = STOPPED
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


# Transfer errors after which the rsync path is tried; NeedsInput is re-raised before.
FALLBACK = (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError)
RESULT_SCHEMA = "sparkring-checkpoint-result/v1"
# What each Spark's checkpoint action reports when the plan needs a decision; the request is printed once.
STOPPED = "Stopped: the checkpoint plan needs a decision; the request is printed at the end."
# rsync writes new files into an empty staging directory. Without -t, -a,
# --inplace, --append, --partial or --checksum it neither sets times nor writes
# into existing files. --perms with --chmod gives every received file mode
# 0644, like every file SparkRing writes into a checkpoint directory, whatever
# the receiving rsync's umask. --open-noatime keeps the sender's reads from
# changing access times of files SparkRing did not create; it needs rsync 3.2.3
# or later on both ends.
RSYNC = ["rsync", "-r", "--perms", "--no-owner", "--no-group", "--chmod=D0755,F0644", "--protect-args",
         "--open-noatime", "--rsync-path=sudo -n rsync"]


def cause(error):
    """What went wrong, as the last line of an error's text without its exception name.

    A rank operation's failure arrives as the remote traceback; its last line
    names the cause, such as ``ValueError: Downloading config.json from
    huggingface.co failed: ...``.
    """
    lines = [line.strip() for line in str(error).splitlines() if line.strip()]
    text = lines[-1] if lines else type(error).__name__
    return re.sub(r"^(?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*(?:Error|Exception|Exit|Interrupt|Expired):\s*", "", text,
                  count=1)[:500]


def operation(runner, rank, name, data=None):
    """Run checkpoint rank operation ``name`` on ``rank``; a failure names the Spark and its cause."""
    try:
        return runner.remote(rank, name, data=data)
    except NeedsInput:
        raise
    except Exception as error:  # noqa: BLE001 - reported with the Spark it came from
        raise RuntimeError(f"Node {rank}: {cause(error)}") from error


def transfer_manifest(manifest, names=None):
    """``{"repository", "revision", "files": {name: sha256}, "sizes": {name: size}}`` restricted to ``names``."""
    names = sorted(manifest["files"] if names is None else names)
    unknown = [name for name in names if name not in manifest["files"] or name not in manifest["sizes"]]
    if unknown:
        raise ValueError("Checkpoint transfer names unpinned files: " + ", ".join(unknown[:5]))
    return {"repository": manifest["repository"], "revision": manifest["revision"],
            "files": {name: manifest["files"][name] for name in names},
            "sizes": {name: int(manifest["sizes"][name]) for name in names}}


def rsync_staging(model):
    """rsync's destination for checkpoint directory ``model``: ``receive/rsync`` in its state directory.

    The state directory is ``.<name of model>.sparkring`` beside ``model``
    (``checkpoint_place.Claim.state``).
    """
    parent, name = posixpath.split(posixpath.normpath(model))
    return posixpath.join(parent, "." + name + ".sparkring", "receive", "rsync")


def receipts_for(receipts, rank):
    """The receipts adoption refreshes on ``rank``: one list for every Spark, or a mapping from rank."""
    if not receipts:
        return []
    if isinstance(receipts, dict):
        return list(receipts.get(rank, receipts.get(str(rank), ())))
    return list(receipts)


def adoption_label(node):
    """Progress label of one Spark's ``model-adopt``.

    A Spark whose plan holds no present, linked or copied file only claims and
    settles SparkRing's directory before it receives or downloads.
    """
    actions = {entry["action"] for entry in node["files"].values()}
    if not actions & {"present", "link", "copy"}:
        return "Prepare SparkRing's checkpoint directory"
    sources = {source["path"]: source for source in node.get("sources") or []}
    other_disks = any(entry["action"] == "copy" and (sources.get(entry.get("candidate")) or {}).get("other_filesystem")
                      for entry in node["files"].values())
    if other_disks and not actions & {"link", "present"}:
        return "Copy checkpoint files from other disks"
    return "Link and verify local checkpoint files"


def finished(outcome, rank, doing):
    """Refuse a rank-operation result that leaves ``rank`` without its complete checkpoint.

    ``model-adopt``, ``model-fetch`` and ``model-transfer-complete`` report
    ``complete``; a result without that key reports success with ``ok``.
    """
    complete = outcome.get("complete", outcome.get("ok") is True) if isinstance(outcome, dict) else False
    if complete is not True:
        missing = outcome.get("missing") if isinstance(outcome, dict) else None
        names = [item.get("name") if isinstance(item, dict) else item for item in missing or []]
        raise ValueError(f"Node {rank}: the checkpoint is incomplete after {doing}"
                         + (" (missing " + ", ".join(str(n) for n in names[:5]) + ")" if names else ""))


def standing_plan(lock, pins):
    """The approval that holds without a checkpoint plan.

    Rows keep their locked mode; owned rows plan no local source, no download
    and no write, so the guard permits only the tolerance of unplanned writes
    per Spark.
    """
    sizes = checkpoint_plan.required_files(pins)
    nodes = []
    for rank, row in enumerate(lock["site"]["ranks"]):
        in_place = bool(row.get("reuse_verified_model"))
        nodes.append({"rank": rank, "host": row["host"], "hostname": row["host"],
                      "mode": "in-place" if in_place else "owned", "path": row["model"],
                      "files": {name: {"action": "in-place", "size": size} for name, size in sizes.items()}
                      if in_place else {},
                      "bytes": {key: 0 for key in checkpoint_plan.BYTE_KEYS}, "write_bytes": 0, "sources": []})
    return {"schema": checkpoint_plan.SCHEMA, "repository": pins["repository"], "revision": pins["revision"],
            "pins_sha256": checkpoint_plan.pins_digest(pins), "approval": None,
            "required": {"files": len(sizes), "bytes": sum(sizes.values()), "sizes": sizes},
            "hub_files": [], "hub_bytes": 0, "nodes": nodes}


def check_plan(plan, pins, rows):
    """Refuse an approved plan made for another checkpoint, other Sparks, other paths or other modes."""
    if (plan.get("schema") != checkpoint_plan.SCHEMA
            or (plan.get("repository"), plan.get("revision")) != (pins["repository"], pins["revision"])
            or plan.get("pins_sha256") != checkpoint_plan.pins_digest(pins)
            or plan["required"]["sizes"] != checkpoint_plan.required_files(pins)):
        raise ValueError("The approved checkpoint plan is for another checkpoint revision or pin manifest")
    nodes = plan["nodes"]
    if len(nodes) != len(rows) or any(
            (node["host"], node["path"], node["mode"] == "in-place") != (row["host"], row["model"],
                                                                       bool(row.get("reuse_verified_model")))
            for node, row in zip(nodes, rows)):
        raise ValueError("The approved checkpoint plan names other Sparks, checkpoint paths or modes than the "
                         "deployment lock")
