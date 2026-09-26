"""Pin manifest generation against a fake Hub, and the committed manifests.

No test contacts the network: the Hub is a local fake, and the HTTP client is
checked against a patched ``urlopen``.
"""
import copy
import hashlib
import io
import json

import pytest

from runtime.common import installer, profiles, setup
from scripts import pin_checkpoint

REPOSITORY = "example-owner/Example-Model"
REVISION = "0123456789abcdef0123456789abcdef01234567"
WEIGHTS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
COMMITTED = {
    ("local-inference-lab/Qwen3.8-Flash-Next-NVFP4", "60215d26cf5e42c2db6128774032d57fc62678da"):
        ["qwen38-flash-next-qad-tp4", "qwen38-flash-next-tp2"],
    ("local-inference-lab/GLM-5.3-Flash-NVFP4-Spark", "a608241037e4c2565356bff7ca293f2133888f88"):
        ["glm53-flash-nvfp4-spark-tp2", "glm53-flash-nvfp4-spark-tp4"],
    ("XiaomiMiMo/MiMo-V2.6-Flash-RL", "5711b268169967567844e1e560e8a3966da959b1"):
        ["mimo-v26-flash-rl-tp2", "mimo-v26-flash-rl-tp4"],
}
# Revisions pinned only by profiles outside the installer: the Qwen SparkCache
# profiles run the step-4000 checkpoint and keep their own SHA256SUMS.
RETAINED = {
    ("local-inference-lab/Qwen3.8-Flash-Next-NVFP4", "629bc3218833a38b475b719f34aa571666f4a03e"):
        ["qwen38-flash-next-qad-tp4-sparkcache", "qwen38-flash-next-tp2-sparkcache"],
}


class FakeHub:
    """A revision whose small files are served and whose LFS bodies only define hashes.

    ``served`` replaces the bytes returned for a name, to model a Hub or network
    fault. Downloading a weight file fails the test.
    """

    def __init__(self, small, lfs, *, commit=REVISION, served=None, listing=None):
        self.small, self.lfs, self.commit_id = small, lfs, commit
        self.served, self.listing = served or {}, listing
        self.downloads = []

    def commit(self, repository, revision):
        return self.commit_id

    def tree(self, repository, revision):
        if self.listing is not None:
            return self.listing
        items = [{"type": "directory", "path": "extra", "oid": "d" * 40, "size": 0}]
        for name, data in self.small.items():
            items.append({"type": "file", "path": name, "size": len(data), "oid": pin_checkpoint.git_blob_id(data)})
        for name, data in self.lfs.items():
            digest = hashlib.sha256(data).hexdigest()
            pointer = pin_checkpoint.LFS_POINTER.format(oid=digest, size=len(data)).encode()
            items.append({"type": "file", "path": name, "size": len(data), "oid": pin_checkpoint.git_blob_id(pointer),
                          "lfs": {"oid": digest, "size": len(data), "pointerSize": len(pointer)}, "xetHash": "e" * 64})
        return items

    def download(self, repository, revision, name, size):
        assert name not in WEIGHTS, "the generator must never download a weight file"
        self.downloads.append(name)
        data = self.served.get(name, self.small.get(name, self.lfs.get(name)))
        return data[:size + 1]


def revision_files():
    index = json.dumps({"metadata": {"total_size": 7},
                        "weight_map": {"a.weight": WEIGHTS[1], "b.weight": WEIGHTS[0], "c.weight": WEIGHTS[1]}}).encode()
    small = {"config.json": b'{"model_type": "example"}\n', "tokenizer_config.json": b"{}\n",
             "README.md": b"# Example\n", ".gitattributes": b"*.safetensors filter=lfs\n",
             "extra/chat_template.jinja": b"{{ messages }}\n"}
    lfs = {"model.safetensors.index.json": index, WEIGHTS[0]: b"weights-one", WEIGHTS[1]: b"weights-two",
           "tokenizer.json": b'{"version": "1.0"}'}
    return small, lfs


def test_generator_checks_git_blob_ids_of_downloaded_small_files():
    small, lfs = revision_files()
    hub = FakeHub(small, lfs)
    pins = pin_checkpoint.manifest(hub, REPOSITORY, REVISION)
    assert set(pins) == {"schema", "repository", "revision", "index", "weights", "optional", "files"}
    assert (pins["schema"], pins["index"], pins["weights"]) == (pin_checkpoint.SCHEMA, "model.safetensors.index.json", list(WEIGHTS))
    for name, data in small.items():
        assert pins["files"][name] == {"size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                                       "git_blob": pin_checkpoint.git_blob_id(data)}
    for name, data in lfs.items():
        entry = pins["files"][name]
        assert (entry["sha256"], entry["size"], entry["lfs"], entry["xet_hash"]) == (hashlib.sha256(data).hexdigest(), len(data), True, "e" * 64)
    # Small files and the index are downloaded; LFS weights and tokenizer never are.
    assert sorted(hub.downloads) == sorted([*small, "model.safetensors.index.json"])
    assert "extra" not in pins["files"]

    faults = {
        "same length, other bytes": {"served": {"config.json": b'{"model_type": "exampl3"}\n'}},
        "longer body": {"served": {"config.json": small["config.json"] + b" "}},
        "shorter body": {"served": {"tokenizer_config.json": b"{}"}},
        "index body differs from its LFS SHA-256": {"served": {"model.safetensors.index.json": lfs["model.safetensors.index.json"].replace(b"a.weight", b"z.weight")}},
        "revision resolves elsewhere": {"commit": "f" * 40},
    }
    for label, fault in faults.items():
        with pytest.raises(ValueError):
            pin_checkpoint.manifest(FakeHub(small, lfs, **fault), REPOSITORY, REVISION)
    with pytest.raises(ValueError, match="Git blob id"):
        pin_checkpoint.manifest(FakeHub(small, lfs, served={"config.json": b'{"model_type": "exampl3"}\n'}), REPOSITORY, REVISION)

    listing = FakeHub(small, lfs).tree(REPOSITORY, REVISION)
    pointer = next(item for item in listing if item["path"] == WEIGHTS[0])
    for label, change in {
        "LFS blob id is not the pointer of its SHA-256": lambda items, item: item.update(oid="0" * 40),
        "LFS size disagrees": lambda items, item: item["lfs"].update(size=item["size"] + 1),
        "zero-byte file": lambda items, item: items.append({"type": "file", "path": "empty.txt", "size": 0, "oid": pin_checkpoint.git_blob_id(b"")}),
        "unsafe name": lambda items, item: items.append({**copy.deepcopy(item), "path": "../escape.safetensors"}),
        "cache name": lambda items, item: items.append({**copy.deepcopy(item), "path": ".cache/huggingface/x"}),
        "listed twice": lambda items, item: items.append(copy.deepcopy(item)),
        "weight map names an unlisted file": lambda items, item: items.remove(item),
    }.items():
        items = copy.deepcopy(listing)
        change(items, next(entry for entry in items if entry["path"] == pointer["path"]))
        with pytest.raises(ValueError):
            pin_checkpoint.manifest(FakeHub(small, lfs, listing=items), REPOSITORY, REVISION)

    for repository, revision in ((REPOSITORY, "main"), (REPOSITORY, REVISION[:12]), ("example-owner/../x", REVISION),
                                 ("owner--x/model", REVISION), ("no-owner", REVISION)):
        with pytest.raises(ValueError):
            pin_checkpoint.manifest(FakeHub(small, lfs), repository, revision)


def test_generator_downloads_are_bounded(monkeypatch):
    small, lfs = revision_files()
    # Every small file fits the limit; the index does not and is never requested.
    monkeypatch.setattr(pin_checkpoint, "DOWNLOAD_LIMIT", 64)
    assert max(map(len, small.values())) <= 64 < len(lfs["model.safetensors.index.json"])
    hub = FakeHub(small, lfs)
    with pytest.raises(ValueError, match="download limit"):
        pin_checkpoint.manifest(hub, REPOSITORY, REVISION)
    assert sorted(hub.downloads) == sorted(small)


def test_optional_files_are_only_documentation_and_repository_metadata():
    documentation = [".gitattributes", "README.md", "README", "LICENSE", "LICENSE.txt", "NOTICE", "USAGE.md",
                     "MiMo_V2_6_technical_report.pdf", "assets/architecture.png", "figure.JPG", "docs/README.md"]
    served = ["config.json", "generation_config.json", "chat_template.jinja", "tokenizer.json", "tokenizer_config.json",
              "vocab.json", "merges.txt", "preprocessor_config.json", "hf_quant_config.json", "export-manifest.json",
              "configuration_mimo_v2.py", "dflash/mask_embedding.pt", "dflash/model.safetensors.index.json",
              "model.safetensors.index.json", "model-00001-of-00036.safetensors", "audio_tokenizer/model.safetensors",
              "docs/notes.md", "docs/manual.pdf", "license_config.json"]
    assert [name for name in documentation if not pin_checkpoint.optional(name)] == []
    assert [name for name in served if pin_checkpoint.optional(name)] == []

    small, lfs = revision_files()
    pins = pin_checkpoint.manifest(FakeHub(small, lfs), REPOSITORY, REVISION)
    assert pins["optional"] == [".gitattributes", "README.md"]

    for (repository, revision), _ in COMMITTED.items():
        card = {"model_repository": repository, "model_revision": revision}
        pins = json.loads(pin_checkpoint.manifest_path(pin_checkpoint.ROOT, repository, revision).read_text(encoding="utf-8"))
        assert pins["optional"] == sorted(name for name in pins["files"] if pin_checkpoint.optional(name)), card
        required = set(pins["files"]) - set(pins["optional"])
        assert {"config.json", pins["index"], *pins["weights"]} <= required
        assert not [name for name in pins["optional"] if name.endswith((".json", ".jinja", ".py", ".safetensors", ".txt", ".pt"))]


def test_committed_manifests_and_sums_are_generator_output(tmp_path):
    for (repository, revision), profile_ids in {**COMMITTED, **RETAINED}.items():
        path = pin_checkpoint.manifest_path(pin_checkpoint.ROOT, repository, revision)
        text = path.read_bytes().decode("utf-8")
        pins = json.loads(text)
        assert text.replace("\r\n", "\n") == pin_checkpoint.encode(pins)
        assert pin_checkpoint.pinning_profiles(pin_checkpoint.ROOT, repository, revision) == profile_ids
        pin_checkpoint.check_contracts(pin_checkpoint.ROOT, pins, profile_ids)
        for profile_id in profile_ids:
            sums = (pin_checkpoint.ROOT / "profiles" / profile_id / "SHA256SUMS").read_bytes().decode("utf-8")
            assert sums.replace("\r\n", "\n") == pin_checkpoint.checksums(pins)
            assert installer.checkpoint_pins(setup.selection(profile_id)) == pins

    pins = json.loads(pin_checkpoint.manifest_path(pin_checkpoint.ROOT, *next(iter(COMMITTED))).read_text(encoding="utf-8"))
    written = pin_checkpoint.write(tmp_path, pins, ["example-profile"])
    assert [path.relative_to(tmp_path).as_posix() for path in written] == [
        "profiles/checkpoints/local-inference-lab--Qwen3.8-Flash-Next-NVFP4/60215d26cf5e42c2db6128774032d57fc62678da.json",
        "profiles/example-profile/SHA256SUMS"]
    assert all(b"\r" not in path.read_bytes() for path in written)
    assert sorted(p.name for p in written[0].parent.iterdir()) == [written[0].name]
    assert json.loads(written[0].read_text(encoding="utf-8")) == pins

    changed = copy.deepcopy(pins)
    changed["files"]["config.json"]["sha256"] = "b1ecb2697178111fb10b73354a681d10884c19ecd830af1073ea93d2a09a5358"
    with pytest.raises(ValueError, match="config.json"):
        pin_checkpoint.check_contracts(pin_checkpoint.ROOT, changed, ["qwen38-flash-next-tp2"])


class Response(io.BytesIO):
    def __init__(self, body, link=""):
        super().__init__(json.dumps(body).encode() if not isinstance(body, bytes) else body)
        self.headers = {"Link": link} if link else {}


def test_hub_requests_are_anonymous_and_follow_pagination(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_example_value_that_must_not_be_sent")
    base = "https://hub.example"
    pages = {
        f"{base}/api/models/{REPOSITORY}/tree/{REVISION}?recursive=true&expand=true":
            Response([{"type": "file", "path": "a"}], f'<{base}/api/models/{REPOSITORY}/tree/{REVISION}?cursor=2>; rel="next"'),
        f"{base}/api/models/{REPOSITORY}/tree/{REVISION}?cursor=2": Response([{"type": "file", "path": "b"}]),
        f"{base}/api/models/{REPOSITORY}/revision/{REVISION}": Response({"sha": REVISION}),
        f"{base}/{REPOSITORY}/resolve/{REVISION}/sub%20dir/config.json": Response(b"0123456789"),
    }
    requests = []

    def urlopen(request, timeout):
        requests.append(request)
        return pages[request.full_url]

    monkeypatch.setattr(pin_checkpoint.urllib.request, "urlopen", urlopen)
    hub = pin_checkpoint.Hub(base + "/")
    assert [item["path"] for item in hub.tree(REPOSITORY, REVISION)] == ["a", "b"]
    assert hub.commit(REPOSITORY, REVISION) == REVISION
    assert hub.download(REPOSITORY, REVISION, "sub dir/config.json", 4) == b"01234"
    assert requests and all(set(request.headers) == {"User-agent"} for request in requests)

    pages[f"{base}/api/models/{REPOSITORY}/tree/{REVISION}?recursive=true&expand=true"] = Response(
        [], '<https://elsewhere.example/api/next>; rel="next"')
    with pytest.raises(ValueError, match="pagination"):
        hub.tree(REPOSITORY, REVISION)


def test_every_installer_profile_revision_has_a_committed_manifest():
    pinned = {(installer.setup.selection(profile)["model_repository"], installer.setup.selection(profile)["model_revision"])
              for profile in installer.INSTALLABLE}
    assert pinned == set(COMMITTED)
    assert sorted(profile for ids in COMMITTED.values() for profile in ids) == sorted(installer.INSTALLABLE)
    assert all(profiles.local_path(pin_checkpoint.manifest_path(pin_checkpoint.ROOT, *key).relative_to(pin_checkpoint.ROOT).as_posix())
               for key in COMMITTED)
