"""Image and compile cache needs of the install plan, from release locks and Spark inspections, without hosts."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from runtime.common import installer_image
from runtime.host import install_space

GIB = 1024 ** 3
ALLOWANCE = 68 * GIB
RELEASES = installer_image.ROOT / "runtime/releases"
PLAINSTATUS = "dev-20260928-plainstatus-cuda1342-nccl2323-status033"
SPINWAIT = "dev-20260930-spinwait-cuda1342-nccl2323-status033"
PORTGID = "dev-20261001-portgid-cuda1342-nccl2323-status033"
STATUSROWS = "dev-20261001-statusrows-cuda1342-nccl2323-status034"


def release(name):
    return json.loads((RELEASES / name / "installer-image.json").read_text(encoding="utf-8"))


def holding(*ids, containerd=False):
    return {"images": list(ids), "containerd": containerd, "caches": {}}


def test_lineage_follows_each_release_publication_to_its_parent():
    chain = install_space.lineage(release(STATUSROWS))
    assert [entry["name"] for entry in chain[:4]] == [STATUSROWS, PORTGID, SPINWAIT, PLAINSTATUS]
    assert chain[2] == {"name": SPINWAIT, "image_id": release(SPINWAIT)["image_id"],
                        "image_bytes": 31654261946, "download_bytes": 15232947091}
    # The chain ends at the first installer image, whose publication names no parent.
    assert chain[-1]["name"] == "dev-20260924-cuda1342-nccl2323-status031" and len(chain) == 12
    # A lock is found by its image ID when its name is not a release directory.
    assert install_space.lineage({**release(SPINWAIT), "name": "renamed-lock"})[1]["name"] == PLAINSTATUS


def write_release(root, name, image_id, parent=None, parent_id=None, **lock):
    directory = root / "runtime/releases" / name
    directory.mkdir(parents=True)
    value = {"schema": installer_image.SCHEMA, "name": name, "image_id": image_id, "image_bytes": 3000,
             "download_bytes": 1000, **lock}
    (directory / "installer-image.json").write_text(json.dumps(value))
    publication = {"release": name, "image_id": image_id}
    if parent:
        publication["derivation"] = {"parent_release": parent, "parent_image_id": parent_id, "layers": []}
    (directory / "publication.json").write_text(json.dumps(publication))
    return value


def test_lineage_ends_where_a_derivation_cannot_be_confirmed(tmp_path):
    ids = {name: "sha256:" + name * 64 for name in "abcdef"}
    write_release(tmp_path, "base", ids["a"])
    write_release(tmp_path, "middle", ids["b"], "base", ids["a"])
    top = write_release(tmp_path, "top", ids["c"], "middle", ids["b"])
    assert [entry["name"] for entry in install_space.lineage(top, tmp_path)] == ["top", "middle", "base"]
    # The parent release records another image than the derivation names.
    wrong = write_release(tmp_path, "wrong", ids["d"], "base", ids["e"])
    assert [entry["name"] for entry in install_space.lineage(wrong, tmp_path)] == ["wrong"]
    # A parent that derives from its own child ends the chain at the repeated image.
    write_release(tmp_path, "loop-a", ids["e"], "loop-b", ids["f"])
    loop = write_release(tmp_path, "loop-b", ids["f"], "loop-a", ids["e"])
    assert [entry["name"] for entry in install_space.lineage(loop, tmp_path)] == ["loop-b", "loop-a"]
    # A development lock outside runtime/releases has no recorded parent; a v1 lock records no sizes.
    assert install_space.lineage({"schema": installer_image.SCHEMA_V1, "name": "local", "image_id": "sha256:" + "9" * 64},
                                   tmp_path) == [{"name": "local", "image_id": "sha256:" + "9" * 64,
                                                  "image_bytes": None, "download_bytes": None}]


def test_a_derived_image_with_its_parent_present_needs_only_the_missing_layers():
    # The pair's case: spinwait adds one layer to plainstatus. Its recorded download is smaller than the
    # parent's, whose lock kept the upper bound, so the unpacked size bounds the missing layer's download.
    chain = install_space.lineage(release(SPINWAIT))
    need = install_space.image_need(chain, holding(release(PLAINSTATUS)["image_id"]), ALLOWANCE)
    assert need == {"basis": "layers", "bytes": 4 * 1700475 + 4 * GIB, "relay_bytes": 1700475,
                    "ancestor": PLAINSTATUS, "unpacked_bytes": 1700475, "download_bytes": 1700475}
    # The ring's case: statusrows adds two layers to spinwait through portgid.
    chain = install_space.lineage(release(STATUSROWS))
    need = install_space.image_need(chain, holding(release(SPINWAIT)["image_id"]), ALLOWANCE)
    assert (need["ancestor"], need["unpacked_bytes"], need["bytes"]) == (SPINWAIT, 3614816, 4 * 3614816 + 4 * GIB)
    # The nearest held ancestor counts, and an image whose download grew more than it unpacks counts that growth.
    lineage = [{"name": "t", "image_id": "t", "image_bytes": 1000, "download_bytes": 900},
               {"name": "p", "image_id": "p", "image_bytes": 990, "download_bytes": 860},
               {"name": "g", "image_id": "g", "image_bytes": 500, "download_bytes": 400}]
    assert install_space.image_need(lineage, holding("p"), ALLOWANCE)["download_bytes"] == 40
    assert install_space.image_need(lineage, holding("g"), ALLOWANCE)["download_bytes"] == 500


def test_without_a_held_ancestor_the_whole_image_is_needed():
    target = release(STATUSROWS)
    chain = install_space.lineage(target)
    whole = target["image_bytes"] + target["download_bytes"] + 8 * GIB
    assert f"{whole / GIB:.1f}" == "51.7"
    expected = {"basis": "whole", "bytes": whole, "relay_bytes": target["download_bytes"], "reason": "none"}
    assert install_space.image_need(chain, holding(), ALLOWANCE) == expected
    assert install_space.image_need(chain, holding(target["image_id"]), ALLOWANCE) == \
        {"basis": "present", "bytes": 0, "relay_bytes": 0}
    # Docker's containerd store loads no single layers; a failed inspection or an unknown store lists nothing.
    containerd = install_space.image_need(chain, holding(release(SPINWAIT)["image_id"], containerd=True), ALLOWANCE)
    assert containerd["bytes"] == whole and (containerd["reason"], containerd["ancestor"]) == ("containerd", SPINWAIT)
    unknown = install_space.image_need(chain, holding(release(SPINWAIT)["image_id"], containerd=None), ALLOWANCE)
    assert unknown["bytes"] == whole and unknown["reason"] == "unchecked"
    failed = install_space.image_need(chain, {"error": "Permission denied"}, ALLOWANCE)
    assert failed["bytes"] == whole and failed["reason"] == "unchecked" and failed["error"] == "Permission denied"
    # An ancestor whose lock records no sizes, or one larger than the image, gives no layer figure.
    unsized = [chain[0], {**chain[1], "image_bytes": None, "download_bytes": None}]
    assert install_space.image_need(unsized, holding(chain[1]["image_id"]), ALLOWANCE)["basis"] == "whole"
    larger = [chain[0], {**chain[1], "image_bytes": chain[0]["image_bytes"] + 1}]
    assert install_space.image_need(larger, holding(chain[1]["image_id"]), ALLOWANCE)["basis"] == "whole"


def test_a_lock_without_sizes_needs_the_allowance():
    chain = [{"name": "local", "image_id": "sha256:" + "9" * 64, "image_bytes": None, "download_bytes": None}]
    assert install_space.image_need(chain, holding(), ALLOWANCE) == \
        {"basis": "unsized", "bytes": ALLOWANCE, "relay_bytes": 0}
    assert install_space.image_need(chain, {"error": "timeout"}, ALLOWANCE)["bytes"] == ALLOWANCE
    # The relay's pull check keeps its allowance for such a lock: the image allowance and 32 GiB of layers.
    assert install_space.relay_check({"image_id": "x"}, [1], {1: 0}, None, ALLOWANCE) == \
        {0: ALLOWANCE, 1: ALLOWANCE + 32 * GIB}


def test_relay_needs_one_copy_of_the_layers_the_sparks_lack():
    needs = [{"basis": "present", "bytes": 0, "relay_bytes": 0},
             {"basis": "layers", "bytes": 4 * GIB, "relay_bytes": 3614816},
             {"basis": "whole", "bytes": 52 * GIB, "relay_bytes": 15 * GIB}, None]
    assert install_space.relay_need(needs) == 15 * GIB and install_space.relay_need(needs[:2]) == 3614816
    card = {"image_id": "x", "image_bytes": 30 * GIB, "download_bytes": 15 * GIB}
    assert install_space.relay_check(card, [0, 2], {0: 0, 2: 0}, None, ALLOWANCE) == \
        {0: 53 * GIB + 15 * GIB, 2: 53 * GIB}


def test_compile_cache_needs_nothing_once_every_directory_holds_files():
    built = {"files": 640, "bytes": 380 * 10 ** 6, "complete": True}
    assert install_space.cache_need({"/c/a": built, "/c/b": built}, 4 * GIB) == \
        {"basis": "built", "bytes": 0, "present_bytes": 760 * 10 ** 6}
    for states in ({"/c/a": built, "/c/b": None}, {"/c/a": {"files": 0, "bytes": 0, "complete": True}}, {}):
        assert install_space.cache_need(states, 4 * GIB)["bytes"] == 4 * GIB


def test_directory_use_counts_regular_files_without_following_links(tmp_path):
    assert install_space.directory_use(str(tmp_path / "absent")) is None
    (tmp_path / "file").write_bytes(b"x")
    assert install_space.directory_use(str(tmp_path / "file")) is None
    cache = tmp_path / "cache"
    (cache / "triton" / "kernel").mkdir(parents=True)
    (cache / "triton" / "kernel" / "a.cubin").write_bytes(b"\0" * 5000)
    (cache / "inductor").mkdir()
    use = install_space.directory_use(str(cache))
    assert use["files"] == 1 and use["bytes"] >= 5000 and use["complete"]
    assert install_space.directory_use(str(cache), entries=1)["complete"] is False


def test_inspection_reports_the_nearest_held_image_the_store_and_the_caches(tmp_path, monkeypatch, capsys):
    held = {"sha256:p", "sha256:g"}
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if argv[3:4] == ["info"]:
            return SimpleNamespace(returncode=0, stdout='[["Backing Filesystem","extfs"]]\n')
        image = argv[-1]
        return SimpleNamespace(returncode=0 if image in held else 1, stdout=image + "\n" if image in held else "")
    monkeypatch.setattr(subprocess, "run", run)
    (tmp_path / "cache").mkdir()
    paths = [str(tmp_path / "cache"), str(tmp_path / "missing")]
    result = install_space.inspect_spark(["sha256:t", "sha256:p", "sha256:g"], paths)
    assert result == {"images": ["sha256:p"], "containerd": False,
                      "caches": {paths[0]: {"files": 0, "bytes": 0, "complete": True}, paths[1]: None}}
    # The search stops at the nearest held image.
    assert [argv[-1] for argv in calls if argv[3:5] == ["image", "inspect"]] == ["sha256:t", "sha256:p"]
    # The shipped source runs on its own and prints the same document.
    exec(compile(install_space.probe_source(["sha256:t", "sha256:p"], paths), "inspection", "exec"), {})
    assert json.loads(capsys.readouterr().out) == result
    monkeypatch.setattr(subprocess, "run", lambda argv, **kwargs: SimpleNamespace(
        returncode=0, stdout='[["driver-type","io.containerd.snapshotter.v1"]]' if argv[3:4] == ["info"] else ""))
    assert install_space.inspect_spark(["sha256:t"], [])["containerd"] is True

    def unavailable(argv, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "docker")
    monkeypatch.setattr(subprocess, "run", unavailable)
    assert install_space.inspect_spark(["sha256:t"], [])["error"].startswith("[Errno 2]")


def test_cache_names_are_the_installer_containers_cache_directories():
    from runtime.common import setup
    card = setup.selection("qwen38-flash-next-tp2")
    assert install_space.cache_names(card, release(STATUSROWS)["image_id"]) == [
        "qwen-flash-next-490a668978e1-60215d26cf5e", "qwen-flash-next-cuda13.4.2-60215d26cf5e"]
    assert Path(card["configuration"]).name == "config.json"
