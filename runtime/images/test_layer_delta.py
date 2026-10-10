"""The docker load archive of a derived image without its parent's layers; offline.

The archives below are laid out as ``docker save`` of Docker 25 or later
writes them: layer blobs named by their diff IDs, the configuration blob named
by the image ID, then ``index.json``, ``manifest.json``, ``oci-layout`` and
``repositories``.
"""
import hashlib
import io
import json
import subprocess
import tarfile

import pytest

from runtime.images import layer_delta


def digest(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


PARENT_LAYERS = [b"base layer", b"toolchain layer"]
ADDED_LAYERS = [b"csf layer", b"sircl layer"]


def saved(parent=PARENT_LAYERS, added=ADDED_LAYERS, *, legacy=False):
    """``(archive bytes, image ID, image diff IDs, parent diff IDs)`` of a saved derived image."""
    layers = [digest(data) for data in parent + added]
    config = json.dumps({"rootfs": {"type": "layers", "diff_ids": layers}}).encode()
    image_id = digest(config)
    members = [("blobs/", None)]
    names = []
    for data in parent + added:
        name = (digest(data)[7:19] + "/layer.tar") if legacy else "blobs/sha256/" + digest(data)[7:]
        names.append(name)
        members.append((name, data))
    members.append(("blobs/sha256/" + image_id[7:], config))
    members += [("index.json", b"{}"),
                ("manifest.json", json.dumps([{"Config": "blobs/sha256/" + image_id[7:],
                                               "RepoTags": ["sparkring-dev/kraken:csf-sircl"],
                                               "Layers": names}]).encode()),
                ("oci-layout", b'{"imageLayoutVersion": "1.0.0"}'), ("repositories", b"{}")]
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, data in members:
            entry = tarfile.TarInfo(name)
            if data is None:
                entry.type = tarfile.DIRTYPE
                archive.addfile(entry)
            else:
                entry.size = len(data)
                archive.addfile(entry, io.BytesIO(data))
    return buffer.getvalue(), image_id, layers, layers[:len(parent)]


def members(data):
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        return {entry.name: (archive.extractfile(entry).read() if entry.isfile() else None) for entry in archive}


def test_the_delta_keeps_the_added_layers_and_every_metadata_file_and_omits_the_parents():
    archive, image_id, layers, parent = saved()
    output = io.BytesIO()
    result = layer_delta.filter_archive(io.BytesIO(archive), output, image_id, layers, parent)
    assert result["omitted_layers"] == 2 and result["added_layers"] == 2
    found = members(output.getvalue())
    for data in PARENT_LAYERS:
        assert "blobs/sha256/" + digest(data)[7:] not in found
    for data in ADDED_LAYERS:
        assert found["blobs/sha256/" + digest(data)[7:]] == data
    assert {"manifest.json", "index.json", "oci-layout", "repositories", "blobs/sha256/" + image_id[7:]} <= set(found)
    # The metadata follows every layer, so an archive cut short has no manifest.
    names = list(found)
    assert names.index("manifest.json") > max(names.index("blobs/sha256/" + digest(d)[7:]) for d in ADDED_LAYERS)


@pytest.mark.parametrize("case, message", [
    ("unrelated parent", "do not begin with all of the parent's"),
    ("no added layer", "do not begin with all of the parent's"),
    ("legacy layout", "did not name each layer by its diff ID"),
    ("other configuration", "configuration is not the image's"),
])
def test_an_archive_that_fails_a_check_gets_no_manifest(case, message):
    archive, image_id, layers, parent = saved(legacy=case == "legacy layout")
    if case == "unrelated parent":
        parent = [digest(b"other")] + parent[1:]
    elif case == "no added layer":
        layers = parent
    elif case == "other configuration":
        image_id = digest(b"another configuration")
    output = io.BytesIO()
    with pytest.raises(ValueError, match=message):
        layer_delta.filter_archive(io.BytesIO(archive), output, image_id, layers, parent)
    assert "manifest.json" not in output.getvalue().decode("latin-1")


def test_a_parent_layer_that_the_image_repeats_above_the_parent_is_kept():
    archive, image_id, layers, parent = saved(added=[b"csf layer", b"base layer"])
    output = io.BytesIO()
    result = layer_delta.filter_archive(io.BytesIO(archive), output, image_id, layers, parent)
    found = members(output.getvalue())
    assert found["blobs/sha256/" + digest(b"base layer")[7:]] == b"base layer"
    assert "blobs/sha256/" + digest(b"toolchain layer")[7:] not in found and result["omitted_layers"] == 1


def test_write_streams_docker_save_into_the_delta_file(tmp_path):
    archive, image_id, layers, parent = saved()
    images = {"derived": {"Id": image_id, "RootFS": {"Layers": layers}},
              "parent": {"Id": "sha256:" + "p" * 64, "RootFS": {"Layers": parent}}}
    calls = []

    def run(argv, **_):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, json.dumps([images[argv[-1]]]), "")

    class Saver:
        def __init__(self, argv, stdout):
            calls.append(argv)
            self.stdout = io.BytesIO(archive)

        def wait(self):
            return 0

    result = layer_delta.write("derived", "parent", tmp_path / "delta.tar", run=run, popen=Saver)
    assert calls[-1] == ["docker", "save", "derived"] and result["added_layers"] == 2
    assert "manifest.json" in members((tmp_path / "delta.tar").read_bytes())
    with pytest.raises(ValueError, match="exists"):
        layer_delta.write("derived", "parent", tmp_path / "delta.tar", run=run, popen=Saver)

    class Failed(Saver):
        def wait(self):
            return 1
    with pytest.raises(ValueError, match="exited with 1"):
        layer_delta.write("derived", "parent", tmp_path / "other.tar", run=run, popen=Failed)
    assert not (tmp_path / "other.tar").exists()
