"""Write a ``docker load`` archive of an image without the layers that its parent image holds.

A derived installer image keeps its parent's layers and adds its own on top
(runtime/images/derived_layer.py, runtime/images/sircl_layer.py). A Spark that
holds the parent needs only the added layers. ``docker save IMAGE`` writes
every layer, about 32 GB for a kraken-line image, so this program filters its
output:

- it reads the ``docker save`` archive of ``--image`` (Docker 25 or later,
  whose archive names each layer ``blobs/sha256/<diff ID>``, the SHA-256 of
  the uncompressed layer) and writes the same archive without the layer blobs
  of ``--parent``;
- it requires the image's layers to begin with all of the parent's, so every
  omitted layer is one the parent provides;
- it holds back ``manifest.json``, ``index.json`` and the other metadata until
  the whole archive has passed its checks: the manifest lists the image's
  configuration and every layer, each omitted blob is a parent layer and each
  added layer was written. A failed check therefore leaves an archive without
  ``manifest.json``, which ``docker load`` refuses.

Docker's loader with the overlay2 (graph-driver) image store opens a layer
file only when its store lacks that layer's chain, so ``docker load`` of the
delta on a host that holds the parent recreates the image with the same ID
(``runtime/host/install_assets.py`` ``write_layer_archive`` relies on the same
loader behavior). A host without the parent, or Docker's containerd image
store, needs the whole ``docker save`` stream instead.

``docker save`` exports every layer into Docker's temporary directory before
it streams, so the build host needs free space for the whole image while this
program runs; the delta itself holds only the added layers.

    python3 runtime/images/layer_delta.py --image IMAGE --parent PARENT --output DELTA.tar
    ssh HOST 'docker image inspect PARENT >/dev/null && docker load' < DELTA.tar
"""
import argparse
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile

SCHEMA = "sparkring-layer-delta/v1"
BLOB = re.compile(r"blobs/sha256/([0-9a-f]{64})")
# Metadata members are held in memory until the archive has passed its checks.
HELD_LIMIT = 64 * 1024 * 1024
CHUNK = 1 << 20


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inspect(image, run=subprocess.run):
    """``(image ID, [diff ID, ...])`` of a local image."""
    done = run(["docker", "image", "inspect", image], capture_output=True, text=True, check=True)
    record = json.loads(done.stdout)[0]
    return record["Id"], list(record["RootFS"]["Layers"])


def filter_archive(source, output, image_id, image_layers, parent_layers):
    """Copy the ``docker save`` archive ``source`` to ``output`` without the parent's layer blobs.

    Returns a summary; raises ValueError before writing the held metadata when
    a check fails.
    """
    count = len(parent_layers)
    require(count and image_layers[:count] == list(parent_layers) and len(image_layers) > count,
            "The image's layers do not begin with all of the parent's, or it adds none")
    added = set(image_layers[count:])
    omit = set(parent_layers) - added
    held, held_bytes, written, omitted, written_bytes = [], 0, set(), set(), 0
    with tarfile.open(fileobj=source, mode="r|", bufsize=CHUNK) as archive, \
            tarfile.open(fileobj=output, mode="w|", format=tarfile.PAX_FORMAT, bufsize=CHUNK) as delta:
        for member in archive:
            name = member.name.removeprefix("./")
            found = BLOB.fullmatch(name)
            digest = "sha256:" + found.group(1) if found else None
            if digest in omit:
                omitted.add(digest)
                continue
            if digest in added and member.isfile():
                delta.addfile(member, archive.extractfile(member))
                written.add(digest)
                written_bytes += member.size
                continue
            if member.isdir():
                delta.addfile(member)
                continue
            data = archive.extractfile(member).read() if member.isfile() else None
            held_bytes += len(data or b"")
            require(held_bytes <= HELD_LIMIT, "The archive holds more metadata than a docker save archive does")
            held.append((member, data))
        files = {member.name.removeprefix("./"): data for member, data in held if data is not None}
        require("manifest.json" in files, "The archive has no manifest.json; it is not a docker save archive")
        manifest = json.loads(files["manifest.json"])
        require(isinstance(manifest, list) and len(manifest) == 1, "The archive holds more than one image")
        entry = manifest[0]
        require(entry.get("Config") == "blobs/sha256/" + image_id.removeprefix("sha256:"),
                "The archive's configuration is not the image's")
        config = json.loads(files[entry["Config"]])
        require(config.get("rootfs", {}).get("diff_ids") == list(image_layers),
                "The archive's configuration lists other layers than the image's")
        paths = entry.get("Layers") or []
        require(paths == ["blobs/sha256/" + layer.removeprefix("sha256:") for layer in image_layers],
                "docker save did not name each layer by its diff ID; stream the whole image instead")
        require(omitted == omit, "The archive lacks some of the parent's layers: "
                + ", ".join(sorted(omit - omitted)))
        require(written == added, "The archive lacks some of the image's added layers: "
                + ", ".join(sorted(added - written)))
        for member, data in held:
            delta.addfile(member, io.BytesIO(data) if data is not None else None)
    return {"schema": SCHEMA, "image_id": image_id, "parent_layers": count, "added_layers": len(added),
            "omitted_layers": len(omitted), "added_layer_bytes": written_bytes}


def write(image, parent, output, *, run=subprocess.run, popen=subprocess.Popen):
    """Stream ``docker save image`` through ``filter_archive`` into ``output`` (a path, or ``-`` for stdout)."""
    image_id, image_layers = inspect(image, run)
    _, parent_layers = inspect(parent, run)
    target = None if output == "-" else Path(output)
    require(target is None or not target.exists(), f"{output} exists")
    saver = popen(["docker", "save", image], stdout=subprocess.PIPE)
    try:
        with (sys.stdout.buffer if target is None else target.open("xb")) as stream:
            try:
                result = filter_archive(saver.stdout, stream, image_id, image_layers, parent_layers)
            except BaseException:
                if target is not None:
                    stream.close()
                    target.unlink(missing_ok=True)
                raise
    finally:
        saver.stdout.close()
        code = saver.wait()
    if code:
        if target is not None:
            target.unlink(missing_ok=True)
        raise ValueError(f"docker save {image} exited with {code}")
    return dict(result, output=output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--image", required=True, help="the derived image: a local tag or image ID")
    parser.add_argument("--parent", required=True, help="the parent image every receiving host holds")
    parser.add_argument("--output", required=True, help="the delta archive to create, or - for standard output")
    args = parser.parse_args(argv)
    try:
        result = write(args.image, args.parent, args.output)
    except (ValueError, OSError, KeyError, tarfile.TarError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2), file=sys.stderr if args.output == "-" else sys.stdout)


if __name__ == "__main__":
    main()
