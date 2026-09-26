"""Checkpoint plans built from pin manifests and survey documents, without hosts.

The fixtures model the Qwen revision 629bc3218833 (48 required files, 36 weight
files) on the two-Spark pair and the four-Spark ring, with the file sizes of the
Hub tree at that revision. Hashes are stand-ins; the plan never reads them.
"""
import datetime
import hashlib
import json

import pytest

from runtime.host import checkpoint_plan as cp

GIB = 1024 ** 3
REPO = "local-inference-lab/Qwen3.8-Flash-Next-NVFP4"
REV = "629bc3218833a38b475b719f34aa571666f4a03e"
MAIN = "7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd"
SHARD_SIZES = [4248628952, 1800011248, 1716167248, 4248633272, 1800011248, 4248633272, 1800011248, 4248645368,
               1800011248] + [4248647096, 1800011248] * 12 + [4548275968, 2786893032, 9669968]
FILES = {
    ".gitattributes": 1635, "README.md": 3461, "chat_template.jinja": 8952, "config.json": 109897,
    "export-manifest.json": 78375, "generation_config.json": 2097, "hf_quant_config.json": 99088,
    "merges.txt": 3353259, "model.safetensors.index.json": 33295470, "preprocessor_config.json": 390,
    "tokenizer.json": 12809320, "tokenizer_config.json": 17928, "video_preprocessor_config.json": 385,
    "vocab.json": 6722759,
    **{f"model-{i:05d}-of-00036.safetensors": size for i, size in enumerate(SHARD_SIZES, 1)},
}
WEIGHTS = sorted(name for name in FILES if name.endswith(".safetensors"))
PINS = {
    "schema": "sparkring-checkpoint-pins/v1", "repository": REPO, "revision": REV,
    "index": "model.safetensors.index.json", "weights": WEIGHTS, "optional": [".gitattributes", "README.md"],
    "files": {name: {"size": size, "sha256": hashlib.sha256(name.encode()).hexdigest(),
                     "git_blob": hashlib.sha1(name.encode()).hexdigest()} for name, size in FILES.items()},
}
SIZES = cp.required_files(PINS)
REQUIRED = sorted(SIZES)
SMALL = sorted(name for name in REQUIRED if name not in WEIGHTS)
WEIGHT_BYTES = sum(SIZES[name] for name in WEIGHTS)
SMALL_BYTES = sum(SIZES[name] for name in SMALL)
TOTAL_BYTES = WEIGHT_BYTES + SMALL_BYTES
POLICY = {"cache_bytes": 32 * GIB, "image_bytes": 68 * GIB}
NOW = datetime.datetime(2026, 9, 26, 2, 0, tzinfo=datetime.timezone.utc)
DEV, MOUNT = 66306, 29
HOSTNAMES = {2: ["spark-aa42", "spark-931e"], 4: ["spark-edfd", "spark-ebb8", "spark-ebee", "spark-4a87"]}
# SSH targets use documentation addresses (RFC 5737).
SSH = {2: ["root@192.0.2.1", "root@192.0.2.2"],
       4: ["root@198.51.100.1", "root@198.51.100.2", "root@198.51.100.3", "root@198.51.100.4"]}
FOLDER = "/var/tmp/models/Qwen3.8-Flash-Next-NVFP4-QAD/" + REV
NEAR_MISS = ("/var/tmp/models/local-inference-lab--Qwen3.8-Flash-Next-NVFP4-4p89/"
             "b184bb5650367c3e934c7849407be9da3671e7f5")
MAIN_CACHE = "/home/code/.cache/huggingface/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4"
CODY_CACHE = "/home/cody/.cache/huggingface/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4"
OPERATOR = {"account": "code", "kind": "operator"}


def cluster(count):
    return "tp2" if count == 2 else "tp4"


def owned_path(count):
    return f"/srv/sparkring/{cluster(count)}/checkpoints/local-inference-lab--Qwen3.8-Flash-Next-NVFP4/{REV}"


def rows(count, **changes):
    result = [{"rank": rank, "host": SSH[count][rank], "model": owned_path(count)} for rank in range(count)]
    for rank, change in changes.items():
        result[int(rank.removeprefix("r"))].update(change)
    return result


def identity(path, name, device):
    return [device, int(hashlib.sha256(f"{path}/{name}".encode()).hexdigest()[:8], 16)]


def candidate(path, names=None, *, layout="local-dir", evidence="recorded", state="match", device=DEV,
              mount_id=MOUNT, home=None, owner="code", commit=REV, branches=(), sparkring=False,
              found_by=("folder",), mode=0o644, blobs=False, overrides=None, **extra):
    names = REQUIRED if names is None else names
    files = {}
    for name in names:
        source = f"{path}/blobs/{PINS['files'][name]['sha256']}" if blobs else f"{path}/{name}"
        files[name] = {"state": state, "evidence": evidence, "size": SIZES[name], "source": source,
                       "kind": "blob" if blobs else "file", "identity": identity(path, name, device),
                       "owner": owner, "mode": mode, "mount_id": mount_id}
    for name, change in (overrides or {}).items():
        files[name] = {**files.get(name, {"size": SIZES[name], "source": f"{path}/{name}",
                                          "identity": identity(path, name, device), "owner": owner, "mode": mode}),
                       **change}
    counts = {key: sum(1 for entry in files.values() if entry["state"] == key)
              for key in ("match", "differs", "size-only", "missing", "incomplete")}
    return {"path": path, "layout": layout, "found_by": list(found_by), "commit": commit, "branches": list(branches),
            "home": home, "sparkring": sparkring, "mount_id": mount_id, "device": device, "rotational": False,
            "files": files, "counts": counts, **extra}


def survey(host, count=2, *, candidates=(), free_gib=291, not_used=(), network=("/mnt/synologytwo",),
           complete=True, seconds=1.2, entries=81234, owned_files=None, userns=False, stopped=None,
           unvisited=None, named=None, operator="code", free_bytes=None):
    return {"schema": "sparkring-checkpoint-survey/v1", "host": host, "repository": REPO, "revision": REV,
            "operator": operator, "docker": {"userns": userns, "driver": "overlay2"},
            "owned": {"path": owned_path(count), "probe_path": f"/srv/sparkring/{cluster(count)}", "mount_point": "/",
                      "mount_id": MOUNT, "device": DEV, "fstype": "ext4",
                      "free_bytes": free_bytes if free_bytes is not None else round(free_gib * GIB),
                      "files": owned_files or {}},
            "search": {"complete": complete, "passes": 1 if complete else 2, "seconds": seconds, "entries": entries,
                       "stopped": stopped, "unvisited": unvisited or {},
                       "skipped_mounts": [{"path": path, "type": "autofs", "network": True} for path in network],
                       "large_directories": 0, "unreadable": 0, "errors": []},
            "candidates": list(candidates), "not_used": list(not_used), "named": named or []}


def make(surveys, count=None, *, rows_=None, **options):
    count = count or len(surveys)
    options.setdefault("policy", POLICY)
    options.setdefault("now", NOW)
    return cp.plan(PINS, surveys, rows_ or rows(count), **options)


def actions(node, action):
    return sorted(name for name, entry in node["files"].items() if entry["action"] == action)


def owner_copy_surveys(count, changes=None):
    hosts, changes = HOSTNAMES[count], changes or {}
    return [survey(hosts[rank], count, candidates=[candidate(FOLDER, found_by=("folder", "docker-mount", "record"))],
                   **changes.get(rank, {})) for rank in range(count)]


@pytest.mark.parametrize("count", [2, 4])
def test_every_rank_links_its_own_copy_and_nothing_is_downloaded(count):
    result = make(owner_copy_surveys(count))
    assert result["hub_files"] == [] and result["hub_bytes"] == 0
    assert result["distribution"] == {"donor": 0, "pool": [], "hub": [], "receive": []}
    for node in result["nodes"]:
        assert node["mode"] == "owned" and node["path"] == owned_path(count)
        assert actions(node, "link") == WEIGHTS and actions(node, "copy") == SMALL
        assert {entry["candidate"] for entry in node["files"].values()} == {FOLDER}
        assert node["bytes"] == {"present": 0, "link": WEIGHT_BYTES, "copy": SMALL_BYTES, "pool": 0, "receive": 0,
                                 "hub": 0}
        assert node["write_bytes"] == SMALL_BYTES
        assert node["required_bytes"] == SMALL_BYTES + SIZES["model.safetensors.index.json"] + 32 * GIB
    assert cp.describe(result)[-1] == "Nothing is downloaded."
    assert cp.attention(result) == []
    assert result["problems"] == []
    json.dumps(result)


@pytest.mark.parametrize("count", [2, 4])
def test_main_copies_download_config_once_on_node_a_and_stream_it(count):
    main = candidate(MAIN_CACHE, [n for n in REQUIRED if n != "config.json"], layout="hf-cache", evidence="hub-named",
                     blobs=True, commit=MAIN, branches=["main"], home=OPERATOR,
                     overrides={"config.json": {"state": "differs", "evidence": "hashed"}})
    hosts = HOSTNAMES[count]
    result = make([survey(hosts[rank], count, candidates=[main], free_bytes=312403148800) for rank in range(count)])
    assert result["hub_files"] == ["config.json"] and result["hub_bytes"] == 109897
    assert result["nodes"][0]["files"]["config.json"]["action"] == "hub"
    expected = {2: {1: 0}, 4: {1: 0, 3: 0, 2: 1}}[count]
    for rank, source in expected.items():
        assert result["nodes"][rank]["files"]["config.json"] == {"action": "receive", "size": 109897, "from": source,
                                                                 "transport": "fabric"}
        assert actions(result["nodes"][rank], "receive") == ["config.json"]
    for node in result["nodes"]:
        assert actions(node, "link") == WEIGHTS and len(actions(node, "copy")) == 11
    # Node 0's entry in the JSON result of sparkring install --json.
    assert cp.summary(result)["nodes"][0] == {
        "rank": 0, "host": SSH[count][0], "hostname": hosts[0], "mode": "owned", "path": owned_path(count),
        "sources": [{"path": MAIN_CACHE, "layout": "hf-cache", "commit": MAIN, "branches": ["main"],
                     "home": OPERATOR, "files": 47}],
        "bytes": {"present": 0, "link": 105839492200, "copy": 56388023, "pool": 0, "receive": 0, "hub": 109897},
        "free_bytes": 312403148800, "required_bytes": 34449531758,
        "search": {"complete": True, "seconds": 1.2, "not_searched": ["/mnt/synologytwo"], "unvisited": {}},
        "not_used": []}
    assert cp.summary(result)["refreshed_receipts"] == 0
    assert cp.summary(result)["reviewed"] is False and cp.summary(result)["command"] == "sudo sparkring install"
    assert cp.describe(result)[-1] == "Downloads 110 KB from huggingface.co on Node 0."
    assert cp.attention(result) == []


def test_node_a_pools_from_peers_before_downloading_what_no_spark_holds():
    first, near, far = set(WEIGHTS[:10]) | set(SMALL), set(WEIGHTS[5:20]), set(WEIGHTS[15:30])
    hosts = HOSTNAMES[4]
    surveys = [survey(hosts[0], 4),
               survey(hosts[1], 4, candidates=[candidate("/data/one", sorted(first))]),
               survey(hosts[2], 4, candidates=[candidate("/data/two", sorted(far))]),
               survey(hosts[3], 4, candidates=[candidate("/data/three", sorted(near))])]
    result = make(surveys)
    missing = sorted(set(WEIGHTS[30:]))
    assert result["distribution"]["pool"] == [
        {"source": 1, "names": sorted(first), "transport": "fabric"},
        {"source": 3, "names": sorted(near - first), "transport": "fabric"},
        {"source": 2, "names": sorted(far - near - first), "transport": "rsync"},
    ]
    assert result["hub_files"] == missing
    node0 = result["nodes"][0]
    assert {node0["files"][n]["from"] for n in first} == {1}
    assert node0["files"][WEIGHTS[18]] == {"action": "pool", "size": SIZES[WEIGHTS[18]], "from": 3,
                                           "transport": "fabric"}
    assert node0["files"][WEIGHTS[25]]["transport"] == "rsync"
    assert actions(node0, "hub") == missing
    # Node A is then the donor; the others receive along the ring.
    assert [(r["source"], r["target"]) for r in result["distribution"]["receive"]] == [(0, 1), (0, 3), (1, 2)]
    for rank, held in ((1, first), (2, far), (3, near)):
        assert actions(result["nodes"][rank], "receive") == sorted(set(REQUIRED) - held)
    text = cp.describe(result)
    assert any("from Node 2 over SSH (rsync)" in line for line in text)
    assert any(line.strip().startswith(f"downloads 6 files ({cp._human(sum(SIZES[n] for n in missing))}) from huggingface.co")
               for line in text)
    assert text[-1].startswith("Downloads ") and text[-1].endswith(" from huggingface.co on Node 0.")


def test_a_complete_worker_is_the_donor_and_nothing_is_downloaded():
    hosts = HOSTNAMES[4]
    half = sorted(set(WEIGHTS[:18]) | set(SMALL))
    surveys = [survey(hosts[0], 4, candidates=[candidate("/data/half", half)]),
               survey(hosts[1], 4),
               survey(hosts[2], 4, candidates=[candidate(FOLDER)]),
               survey(hosts[3], 4, candidates=[candidate("/data/some", WEIGHTS[:3])])]
    result = make(surveys)
    assert result["hub_files"] == [] and result["distribution"]["pool"] == []
    assert result["distribution"]["donor"] == 2
    assert [(r["source"], r["target"]) for r in result["distribution"]["receive"]] == [(2, 3), (2, 1), (3, 0)]
    assert actions(result["nodes"][0], "receive") == sorted(set(REQUIRED) - set(half))
    assert {e["from"] for e in result["nodes"][0]["files"].values() if e["action"] == "receive"} == {3}
    assert actions(result["nodes"][1], "receive") == REQUIRED
    assert actions(result["nodes"][2], "receive") == []
    assert cp.describe(result)[-1] == "Nothing is downloaded."


def test_weight_files_are_linked_and_other_files_copied():
    def node(*candidates):
        return make([survey("a", candidates=list(candidates)), survey("b")])["nodes"][0]

    same_mount = node(candidate("/data/qwen"))
    assert actions(same_mount, "link") == WEIGHTS and actions(same_mount, "copy") == SMALL
    other_directory = f"/srv/sparkring/lab/checkpoints/local-inference-lab--Qwen3.8-Flash-Next-NVFP4/{REV}"
    sparkring = node(candidate(other_directory, layout="sparkring", sparkring=True, owner="root"))
    assert actions(sparkring, "link") == REQUIRED
    # A bind mount of the same filesystem has another mount ID: link(2) fails with EXDEV.
    bind = node(candidate("/mnt/bind/qwen", mount_id=MOUNT + 7))
    assert actions(bind, "copy") == REQUIRED and bind["bytes"]["copy"] == TOTAL_BYTES
    other_device = node(candidate("/mnt/usb/qwen", device=2049, mount_id=51))
    assert actions(other_device, "copy") == REQUIRED


def order_cases():
    weight_file, small_file = WEIGHTS[0], "tokenizer.json"
    return {
        "named": (candidate("/z/named", state="size-only", evidence="size"), candidate("/a/recorded"), weight_file,
                  ["/z/named"]),
        "sparkring": (candidate("/z/sparkring", layout="sparkring", sparkring=True), candidate("/a/user"),
                      weight_file, []),
        "linkable": (candidate("/z/same-mount"), candidate("/a/other-disk", device=2049, mount_id=51), weight_file, []),
        "evidence": (candidate("/z/recorded"), candidate("/a/cache", layout="hf-cache", evidence="hub-named",
                                                         blobs=True), weight_file, []),
        "blob": (candidate("/z/cache", layout="hf-cache", evidence="hub-named", blobs=True),
                 candidate("/a/download", evidence="hub-metadata"), weight_file, []),
        "completeness": (candidate("/z/complete"), candidate("/a/partial", [weight_file, *WEIGHTS[5:8]]), weight_file,
                         []),
        "home": (candidate("/z/data"), candidate("/a/home/code/qwen", home=OPERATOR), weight_file, []),
        "slow-copies": (candidate("/z/nvme"), candidate("/a/usb", rotational=True), small_file, []),
        "path": (candidate("/a/one"), candidate("/b/two"), weight_file, []),
    }


@pytest.mark.parametrize("case", sorted(order_cases()))
def test_source_order_is_deterministic(case):
    preferred, other, name, named = order_cases()[case]
    row = rows(2)[0]
    for candidates in ([preferred, other], [other, preferred]):
        node = cp.choose_sources(PINS, survey("a", candidates=candidates), row, named)
        assert node["files"][name]["candidate"] == preferred["path"], case
    assert cp.choose_sources(PINS, survey("a", candidates=[preferred, other]), row, named) == \
        cp.choose_sources(PINS, survey("a", candidates=[other, preferred]), row, named)


def test_a_complete_spark_is_preferred_to_copying_from_a_slow_disk():
    usb = candidate("/media/usb/qwen", device=2065, mount_id=88, usb=True)
    result = make([survey("a", candidates=[candidate(FOLDER)]), survey("b", candidates=[usb])])
    node = result["nodes"][1]
    assert actions(node, "receive") == REQUIRED and node["sources"] == []
    assert node["not_used"][0]["path"] == "/media/usb/qwen"
    # Without a complete Spark, the slow disk is still the best source.
    alone = make([survey("a"), survey("b", candidates=[usb])])
    assert actions(alone["nodes"][1], "copy") == REQUIRED and alone["distribution"]["donor"] == 1


def test_copies_in_other_accounts_homes_are_listed_but_used_only_when_named():
    hosts = HOSTNAMES[4]
    cody = candidate(CODY_CACHE, layout="hf-cache", evidence="hub-named", blobs=True, owner="cody",
                     home={"account": "cody", "kind": "private"})
    surveys = [survey(hosts[0], 4, operator="dooner"), survey(hosts[1], 4, candidates=[candidate(FOLDER)]),
               survey(hosts[2], 4, candidates=[cody], operator="dooner"), survey(hosts[3], 4)]
    unnamed = make(surveys, operator="dooner")
    node = unnamed["nodes"][2]
    assert actions(node, "receive") == REQUIRED
    assert node["not_used"] == [{"path": CODY_CACHE, "reason": "in cody's home, another account",
                                 "option": f"--model-path 2={CODY_CACHE}"}]
    # The JSON summary names the copy it did not use and the option that would use it.
    assert cp.summary(unnamed)["nodes"][2]["not_used"] == node["not_used"]
    named = make(surveys, operator="dooner", named=[f"2={CODY_CACHE}"])
    node = named["nodes"][2]
    assert actions(node, "link") == WEIGHTS and actions(node, "copy") == SMALL and node["not_used"] == []
    assert any(line.strip() == "Hugging Face cache, snapshot 629bc3218833 (the pinned revision), in cody's home "
                               "(another account)" for line in cp.describe(named))


def test_named_paths_are_per_node_and_additive():
    assert cp.named_paths(["/mnt/a/", "1=/data/qwen", {"rank": 0, "path": "/x"}], 2) == [
        {"rank": None, "path": "/mnt/a"}, {"rank": 1, "path": "/data/qwen"}, {"rank": 0, "path": "/x"}]
    assert cp.named_paths({1: ["/b"], None: "/c"}) == [{"rank": 1, "path": "/b"}, {"rank": None, "path": "/c"}]
    for bad in (["relative/path"], ["5=/data"], ["1=relative"]):
        with pytest.raises(ValueError):
            cp.named_paths(bad, 2)
    named_folder = candidate("/data/qwen", [n for n in REQUIRED if n != "config.json"], layout="folder",
                             state="size-only", evidence="size")
    surveys = [survey("a", candidates=[candidate(FOLDER), named_folder]),
               survey("b", candidates=[candidate(FOLDER), named_folder])]
    result = make(surveys, named=["1=/data/qwen"])
    node0, node1 = result["nodes"]
    assert {e["candidate"] for e in node0["files"].values()} == {FOLDER}
    # Named first on Node 1; the search still supplies what the named copy lacks.
    assert {n for n, e in node1["files"].items() if e["candidate"] == "/data/qwen"} == set(REQUIRED) - {"config.json"}
    assert node1["files"]["config.json"]["candidate"] == FOLDER
    assert result["named"] == [{"rank": 1, "path": "/data/qwen"}]
    absent = [survey("a", named=[{"path": "/mnt/usb/qwen", "state": "absent"}]),
              survey("b", named=[{"path": "/mnt/usb/qwen", "state": "absent"}])]
    missing = make(absent, named=["/mnt/usb/qwen"])
    assert missing["problems"][0] == {"field": "model_path", "rank": None,
                                      "message": "--model-path /mnt/usb/qwen exists on no Spark. Nothing has been changed."}
    one = make([survey("a", candidates=[candidate("/mnt/usb/qwen", device=2049, mount_id=51)]),
                survey("b", named=[{"path": "/mnt/usb/qwen", "state": "absent"}])], named=["/mnt/usb/qwen"])
    assert one["problems"] == [] and one["nodes"][0]["mode"] == "in-place"


def test_a_candidate_reached_through_a_named_path_belongs_to_that_path():
    # The survey reached /mnt/fast/qwen through the second named path; its report gives no resolved form.
    reached = candidate("/mnt/fast/qwen", found_by=("named",), named_as="/data/second")
    reports = [{"path": "/data/first", "state": "absent"}, {"path": "/data/second", "state": "partial"}]
    result = make([survey("a", candidates=[reached], named=reports),
                   survey("b", named=[{"path": path, "state": "absent"} for path in ("/data/first", "/data/second")])],
                  named=["/data/first", "/data/second"])
    assert result["nodes"][0]["sources"][0]["path"] == "/mnt/fast/qwen" and result["nodes"][0]["sources"][0]["named"]
    assert result["problems"] == [{"field": "model_path", "rank": None,
                                   "message": "--model-path /data/first exists on no Spark. Nothing has been changed."}]


def test_named_copy_on_another_filesystem_is_in_place_only_if_exact():
    hosts = HOSTNAMES[4]
    usb = "/mnt/usb-models/Qwen3.8-Flash-Next-NVFP4"

    def surveys(copy):
        return [survey(hosts[rank], 4, candidates=[candidate(FOLDER)]) for rank in range(3)] + [
            survey(hosts[3], 4, candidates=[copy], free_gib=137)]

    # As the survey reports a named folder: its requested path, and exact when it
    # holds only the pinned files.
    exact = candidate(usb, device=2049, mount_id=51, found_by=("named",), named_as=usb, exact=True, extra=[],
                      symlinks=[])
    result = make(surveys(exact), named=[f"3={usb}"])
    node = result["nodes"][3]
    assert node["mode"] == "in-place" and node["path"] == usb
    assert actions(node, "in-place") == REQUIRED and node["write_bytes"] == 0
    assert cp.summary(result)["nodes"][3]["bytes"]["in_place"] == TOTAL_BYTES
    text = cp.describe(result)
    start = text.index(f"Node 3 spark-4a87: serve {usb} in place, read-only")
    assert text[start:start + 3] == [
        f"Node 3 spark-4a87: serve {usb} in place, read-only",
        "    named with --model-path 3=...; on another filesystem than /srv/sparkring, and it holds exactly the "
        "pinned files",
        "    The model will not start while that folder is changed or missing."]
    inexact = candidate(usb, device=2049, mount_id=51, overrides={"config.json": {"state": "differs"}})
    copied = make(surveys(inexact), named=[f"3={usb}"])["nodes"][3]
    assert copied["mode"] == "owned"
    assert actions(copied, "copy") == sorted(set(REQUIRED) - {"config.json"})
    assert copied["files"]["config.json"]["action"] == "receive"
    same_mount = make(surveys(candidate(usb, named_as=usb, exact=True)), named=[f"3={usb}"])["nodes"][3]
    assert same_mount["mode"] == "owned" and actions(same_mount, "link") == WEIGHTS
    assert make(surveys(exact), named=[f"3={usb}"], locked=True)["nodes"][3]["mode"] == "owned"
    # Without the survey's classification, every pinned file at its own name makes a folder exact.
    unclassified = candidate(usb, device=2049, mount_id=51)
    assert make(surveys(unclassified), named=[f"3={usb}"])["nodes"][3]["mode"] == "in-place"
    # A classification from the survey decides over the file states.
    extra = candidate(usb, device=2049, mount_id=51, exact=False)
    assert make(surveys(extra), named=[f"3={usb}"])["nodes"][3]["mode"] == "owned"
    # A named path with symlink components matches through the survey's resolved form.
    resolved = candidate("/srv/spark-models/qwen", device=2049, mount_id=51)
    aliased = surveys(resolved)
    aliased[3]["named"] = [{"path": "/models/qwen", "state": "exact", "resolved": "/srv/spark-models/qwen"}]
    node = make(aliased, named=["3=/models/qwen"])["nodes"][3]
    assert node["mode"] == "in-place" and node["path"] == "/srv/spark-models/qwen"
    # A Hugging Face cache holds its files through symlinks, so it is never served in place.
    cache = candidate("/mnt/usb/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4", device=2049,
                      mount_id=51, layout="hf-cache", evidence="hub-named", blobs=True)
    node = make(surveys(cache), named=["3=/mnt/usb/hub"])["nodes"][3]
    assert node["mode"] == "owned" and actions(node, "copy") == REQUIRED
    assert cp.summary(result)["nodes"][3]["sources"] == [{"path": usb, "layout": "local-dir", "commit": REV,
                                                          "branches": [], "home": None, "files": 48}]


def test_in_place_copy_that_changed_after_planning_is_refused():
    hosts = HOSTNAMES[4]
    changed = candidate("/mnt/usb/qwen", device=2049, mount_id=51, extra=["added_tokens.json"],
                        overrides={"config.json": {"state": "differs"}})
    surveys = [survey(hosts[rank], 4, candidates=[candidate(FOLDER)]) for rank in range(3)] + [
        survey(hosts[3], 4, candidates=[changed])]
    in_place = rows(4, r3={"model": "/mnt/usb/qwen", "reuse_verified_model": True})
    request = {"profile": "qwen38-flash-next-qad-tp4", "cache_path": "/mnt/fast/cache"}
    result = make(surveys, rows_=in_place, named=["3=/mnt/usb/qwen", "/data/other"], locked=True, request=request)
    assert result["nodes"][3]["mode"] == "in-place" and result["nodes"][3]["in_place"]["state"] == "changed"
    # The suggested command repeats the request without the named copy, so SparkRing assembles its own.
    assert result["problems"] == [{"field": "model_path", "rank": 3, "message": (
        "Node 3 spark-4a87: /mnt/usb/qwen differs from the pinned revision in config.json and also holds "
        "added_tokens.json, which the serving engine would load. SparkRing never changes your copy and serves a "
        "named folder in place only when it holds exactly the pinned files. Change the folder yourself, or install "
        "without naming it on Node 3 so that SparkRing assembles its own copy: sudo sparkring install --profile "
        "qwen38-flash-next-qad-tp4 --model-path /data/other --cache-path /mnt/fast/cache --plan.")}]
    # The plan line above that message no longer says the copy is exact.
    text = cp.describe(result)
    start = text.index("Node 3 spark-4a87: serve /mnt/usb/qwen in place, read-only")
    assert text[start + 1] == ("    named with --model-path 3=...; on another filesystem than /srv/sparkring; it no "
                               "longer holds exactly the pinned files (see below)")
    # A Spark whose search failed was not checked; the plan says so instead of calling the copy exact.
    failed = make(surveys[:3] + [TimeoutError("ssh: timed out after 150 s")], rows_=in_place,
                  named=["3=/mnt/usb/qwen"], locked=True)
    assert failed["nodes"][3]["in_place"]["state"] == "unchecked" and failed["problems"] == []
    text = cp.describe(failed)
    assert text[text.index("Node 3 root@198.51.100.4: serve /mnt/usb/qwen in place, read-only") + 1] == (
        "    named with --model-path 3=...; not checked, because the search on this Spark failed; SparkRing "
        "verifies every file before the model starts")
    assert cp.summary(failed)["nodes"][3]["in_place"] == "unchecked"


@pytest.mark.parametrize("state,text", [
    ("foreign", "is not empty and was not created by SparkRing; SparkRing does not adopt a directory it did not "
                "create. Move it away or choose another path."),
    ("replaced", "was replaced after SparkRing created it."),
    ("symlink", "is a symlink;"),
    ("mount-point", "is a mount point;"),
])
def test_checkpoint_directories_the_rank_operations_refuse_stop_the_plan(state, text):
    refused = survey("spark-aa42", candidates=[candidate(FOLDER)])
    refused["owned"]["state"] = state
    result = make([refused, survey("spark-931e", candidates=[candidate(FOLDER)])])
    [problem] = result["problems"]
    assert (problem["field"], problem["rank"]) == ("storage", 0)
    assert problem["message"].startswith(f"Node 0 spark-aa42: {owned_path(2)} {text}")
    assert problem["message"].endswith(" Nothing has been changed.")
    fine = survey("spark-aa42", candidates=[candidate(FOLDER)])
    fine["owned"]["state"] = "owned"
    assert make([fine, survey("spark-931e", candidates=[candidate(FOLDER)])])["problems"] == []


def test_named_paths_the_survey_excludes_are_listed_not_reported_missing():
    ignored = [survey("a", named=[{"path": "/data/qwen", "resolved": "/data/qwen", "state": "ignored"}]),
               survey("b", named=[{"path": "/data/qwen", "resolved": "/data/qwen", "state": "ignored"}])]
    result = make(ignored, named=["/data/qwen"])
    assert result["problems"] == []
    assert result["nodes"][0]["not_used"] == [{"path": "/data/qwen", "option": None,
                                               "reason": "named, but a .sparkring-ignore file excludes it"}]


def test_ignore_local_copies_keeps_sparkring_directories_and_named_paths():
    directory = f"/srv/sparkring/lab/checkpoints/local-inference-lab--Qwen3.8-Flash-Next-NVFP4/{REV}"
    theirs = candidate(directory, WEIGHTS[:10], layout="sparkring", sparkring=True, owner="root")
    named = candidate("/data/named", WEIGHTS[10:20], layout="folder", state="size-only", evidence="size")
    user = candidate(FOLDER)
    surveys = [survey("a", candidates=[theirs, named, user]), survey("b", candidates=[user])]
    result = make(surveys, ignore_local=True, named=["0=/data/named"])
    node = result["nodes"][0]
    assert {e["candidate"] for e in node["files"].values() if e["action"] in ("link", "copy")} == {directory, "/data/named"}
    assert actions(node, "hub") == sorted(set(REQUIRED) - set(WEIGHTS[:20]))
    assert result["ignore_local_copies"] is True
    assert make(surveys, named=["0=/data/named"])["hub_files"] == []


def test_network_named_copy_is_read_on_one_node_only():
    hosts = HOSTNAMES[4]
    share = "/mnt/synologytwo/qwen"
    named = [{"path": share, "state": "network"}]
    result = make([survey(hosts[rank], 4, named=named) for rank in range(4)], named=[share])
    assert actions(result["nodes"][0], "copy") == REQUIRED
    assert {e["candidate"] for e in result["nodes"][0]["files"].values()} == {share}
    for rank in (1, 2, 3):
        assert actions(result["nodes"][rank], "receive") == REQUIRED
    assert result["hub_files"] == [] and result["problems"] == []
    later = make([survey(hosts[0], 4, named=[{"path": share, "state": "absent"}])]
                 + [survey(hosts[rank], 4, named=named) for rank in (1, 2, 3)], named=[share])
    assert actions(later["nodes"][1], "copy") == REQUIRED and later["distribution"]["donor"] == 1
    assert all(actions(later["nodes"][rank], "copy") == [] for rank in (0, 2, 3))
    # Network storage found by the search, not named, is never read.
    found = candidate(share, network=True)
    assert make([survey("a", candidates=[found]), survey("b")])["hub_files"] == REQUIRED


def test_storage_formula_is_the_same_at_plan_and_run_time():
    assert cp.storage_policy() == POLICY
    hosts = HOSTNAMES[4]
    surveys = [survey(hosts[0], 4, free_gib=44), survey(hosts[1], 4, candidates=[candidate(FOLDER)], free_gib=356),
               survey(hosts[2], 4, candidates=[candidate(FOLDER)], free_gib=623), survey(hosts[3], 4, free_gib=137)]
    result = make(surveys)
    node = result["nodes"][0]
    written = [entry["size"] for entry in node["files"].values() if entry["action"] in cp.WRITE_ACTIONS]
    # model-transfer-prepare checks the needed names' sizes with the same function
    # and no image term, because the image is present by then.
    assert node["required_bytes"] == cp.required_space(written, cache_bytes=POLICY["cache_bytes"]) == 144804004456
    assert f"{node['required_bytes'] / GIB:.1f}" == "134.9"
    linked = result["nodes"][1]
    assert linked["required_bytes"] == cp.required_space(
        [SIZES[n] for n in SMALL], cache_bytes=POLICY["cache_bytes"]) == 34449531758
    absent = make(surveys, images_present=[False, True, True, True])
    assert absent["nodes"][0]["required_bytes"] == node["required_bytes"] + 68 * GIB
    assert absent["nodes"][1]["required_bytes"] == linked["required_bytes"]
    elsewhere = [dict(s, docker={"userns": False, "driver": "overlay2", "device": 2049}) for s in surveys]
    assert make(elsewhere, images_present=[False] * 4)["nodes"][0]["required_bytes"] == node["required_bytes"]
    # A compile cache on another filesystem (--cache-path) needs no allowance on the checkpoint's filesystem.
    fast = [dict(s, owned={**s["owned"], "cache_path": "/mnt/fast/cache", "cache_device": 2065}) for s in surveys]
    moved = make(fast)["nodes"][0]
    assert moved["storage"]["cache_bytes"] == 0 and moved["required_bytes"] == node["required_bytes"] - 32 * GIB
    assert cp.required_space([]) == 0 and cp.required_space([5, 7], cache_bytes=1, image_bytes=2) == 22


def test_describe_matches_the_documented_lines():
    near = {"path": NEAR_MISS, "reason": "another checkpoint: its index differs from the pinned revision"}
    surveys = owner_copy_surveys(2, {0: {"not_used": [near], "free_bytes": 312403148800},
                                     1: {"not_used": [near], "free_gib": 308, "seconds": 0.9}})
    result = make(surveys, retained={"qwen38-flash-next-tp2-i0123456789ab": [FOLDER, FOLDER],
                                     "qwen38-flash-next-tp2-ifedcba987654": ["/elsewhere", "/elsewhere"]})
    assert cp.announce(PINS, 2) == ("Looking for existing copies of local-inference-lab/Qwen3.8-Flash-Next-NVFP4 at "
                                    "629bc3218833 on 2 Sparks (up to 20 s each; up to 75 s on a Spark whose search "
                                    "needs a second pass).")
    assert cp.describe(result) == [
        "Checkpoint local-inference-lab/Qwen3.8-Flash-Next-NVFP4 at 629bc3218833: 48 files, 98.6 GiB",
        "Searched spark-aa42 in 1.2 s and spark-931e in 0.9 s. Not searched: /mnt/synologytwo (network storage; "
        "name a copy there with --model-path N=PATH).",
        "",
        f"Node 0 spark-aa42 -> {owned_path(2)}",
        f"    from {FOLDER}",
        "        Hugging Face download folder, commit 629bc3218833 (the pinned revision), files owned by code",
        "        48 of 48 files identified by SparkRing's earlier checksums",
        "    hard-link 36 weight files (no copy, no extra space); copy 12 other files (56.5 MB)",
        "    needs 32.1 GiB free on / (89.8 MB for checkpoint files, 32 GiB for the compile cache; 291 GiB free)",
        f"    not used: {NEAR_MISS} (another checkpoint: its index differs)",
        "Node 1 spark-931e: as Node 0 (308 GiB free)",
        "",
        "Also updates the recorded file identities of the retained deployments that serve the linked folder: "
        "qwen38-flash-next-tp2-i0123456789ab",
        "SparkRing checks the SHA-256 of every file before it links or copies it (about 20 s to 5 minutes per Spark); "
        "later starts compare recorded file identities and re-hash any file that changed.",
        "SparkRing only reads your copies and adds hard links to their weight files; it never writes, moves or "
        "deletes them.",
        "While this checkpoint directory exists, deleting a linked copy does not free its disk space (see sudo "
        "sparkring checkpoints).",
        "Nothing is downloaded.",
    ]
    assert result["refreshed_receipts"] == 1

    def stripped(value):
        return [line.strip() for line in cp.describe(value)]

    main = candidate(MAIN_CACHE, [n for n in REQUIRED if n != "config.json"], layout="hf-cache", evidence="hub-named",
                     blobs=True, commit=MAIN, branches=["main"], home=OPERATOR,
                     overrides={"config.json": {"state": "differs"}})
    lines = stripped(make([survey("spark-aa42", candidates=[main]), survey("spark-931e", candidates=[main])]))
    for line in ("Hugging Face cache, snapshot 7c4f1bc1a2d6 (branch main), in code's home (the operator's)",
                 "47 of 48 files identified by their blob names; config.json differs from the pinned revision",
                 "hard-link 36 weight files (no copy, no extra space); copy 11 other files (56.4 MB)",
                 "config.json (110 KB) is downloaded once on Node 0 from huggingface.co and copied over the fabric"):
        assert line in lines

    plain = candidate("/data/qwen-copy", layout="folder", evidence="size", state="size-only", commit=None,
                      overrides={name: {"state": "match", "evidence": "hashed"} for name in SMALL})
    lines = stripped(make([survey("spark-aa42", candidates=[plain]), survey("spark-931e")]))
    for line in ("from /data/qwen-copy", "plain folder, files owned by code",
                 "12 small files checked by SHA-256; 36 weight files match by name and size only",
                 "SparkRing hashes them before linking. If any differs, SparkRing stops before downloading a "
                 "replacement.",
                 "hard-link 36 weight files (no copy, no extra space); copy 12 other files (56.5 MB)"):
        assert line in lines

    hosts = HOSTNAMES[4]
    cody = candidate(CODY_CACHE, layout="hf-cache", evidence="hub-named", blobs=True, owner="cody",
                     home={"account": "cody", "kind": "private"})
    ring = make([survey(hosts[0], 4), survey(hosts[1], 4, candidates=[candidate(FOLDER)]),
                 survey(hosts[2], 4, candidates=[cody], free_gib=623, entries=81402, seconds=2.4), survey(hosts[3], 4)],
                operator="dooner")
    lines = stripped(ring)
    for line in ("Node 2 spark-ebee: no copy found (searched 81,402 folder entries in 2.4 s)",
                 "receives 48 files (98.6 GiB) from Node 1 over the fabric; needs 134.9 GiB free on / (102.9 GiB for "
                 "checkpoint files, 32 GiB for the compile cache; 623 GiB free)",
                 f"not used: {CODY_CACHE} (in cody's home, another account); use it with --model-path 2={CODY_CACHE}"):
        assert line in lines

    nothing = make([survey("spark-aa42"), survey("spark-931e")])
    lines = cp.describe(nothing)
    assert ("No Spark holds the checkpoint: Node 0 downloads it once from huggingface.co (98.6 GiB) and copies it to "
            "the other Sparks over the fabric.") in lines
    assert lines[-1] == "Downloads 98.6 GiB from huggingface.co on Node 0."


def envelope_case(name):
    hf = "/var/tmp/sparkring-test/hf/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4"
    cache = candidate(hf, layout="hf-cache", evidence="hub-named", blobs=True)
    without = candidate(hf, [n for n in REQUIRED if n != WEIGHTS[1]], layout="hf-cache", evidence="hub-named",
                        blobs=True)
    usb = "/mnt/usb/qwen"
    exact = candidate(usb, device=2049, mount_id=51)
    reviewed = [survey("spark-aa42", candidates=[cache]), survey("spark-931e")]
    options = {}
    if name == "equal":
        fresh = reviewed
    elif name == "smaller":
        reviewed = [survey("spark-aa42", candidates=[cache]),
                    survey("spark-931e", candidates=[candidate(FOLDER, [n for n in REQUIRED if n != WEIGHTS[0]])])]
        fresh = [survey("spark-aa42", candidates=[cache]), survey("spark-931e", candidates=[candidate(FOLDER)])]
    elif name == "added-source":
        fresh = [survey("spark-aa42", candidates=[cache]), survey("spark-931e", candidates=[candidate(FOLDER)])]
    elif name == "growth-within-tolerance":
        reviewed = [survey("spark-aa42", candidates=[cache]),
                    survey("spark-931e", candidates=[candidate(FOLDER, [n for n in REQUIRED if n != "vocab.json"])])]
        fresh = [survey("spark-aa42", candidates=[cache]),
                 survey("spark-931e", candidates=[candidate(FOLDER, [n for n in REQUIRED if n not in (
                     "vocab.json", "tokenizer.json")])])]
    elif name == "download":
        fresh = [survey("spark-aa42", candidates=[without]), survey("spark-931e")]
    elif name == "writes":
        reviewed = [survey("spark-aa42", candidates=[cache]), survey("spark-931e", candidates=[candidate(FOLDER)])]
        fresh = [survey("spark-aa42", candidates=[cache]), survey("spark-931e")]
    elif name == "mode":
        options = {"named": [f"1={usb}"]}
        reviewed = [survey("spark-aa42", candidates=[cache]), survey("spark-931e", candidates=[exact])]
        fresh = [survey("spark-aa42", candidates=[cache]), survey("spark-931e", candidates=[
            candidate(usb, device=2049, mount_id=51, extra=["added_tokens.json"])])]
    elif name == "sources":
        directory = f"/srv/sparkring/lab/checkpoints/local-inference-lab--Qwen3.8-Flash-Next-NVFP4/{REV}"
        fresh = [survey("spark-aa42", candidates=[cache, candidate(directory, WEIGHTS[:1], sparkring=True,
                                                                   layout="sparkring")]), survey("spark-931e")]
    return make(reviewed, **options), make(fresh, **options)


@pytest.mark.parametrize("name,kinds", [
    ("equal", []), ("smaller", []), ("growth-within-tolerance", []), ("download", ["download"]),
    ("writes", ["writes"]), ("mode", ["writes", "mode"]), ("sources", ["sources"]), ("added-source", ["sources"]),
])
def test_envelope_accepts_equal_or_smaller_plans_and_rejects_growth(name, kinds):
    reviewed, fresh = envelope_case(name)
    # The reviewed plan is read back from checkpoint-plan.json.
    reviewed = json.loads(json.dumps(reviewed))
    items = cp.envelope(reviewed, fresh)
    assert [item["kind"] for item in items] == kinds
    if name == "mode":
        assert any("would use SparkRing's checkpoint directory instead of serving /mnt/usb/qwen in place" in i["text"]
                   for i in items)
    if name == "download":
        assert cp.envelope_message(items) == (
            "The checkpoint plan differs from the plan reviewed with --plan: Node 0 spark-aa42 would download 1.68 GiB "
            "from huggingface.co (model-00002-of-00036.safetensors), which the reviewed plan took from "
            "/var/tmp/sparkring-test/hf/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4, where that file is "
            "absent. Nothing was changed. Review the plan again with sudo sparkring install --plan, then repeat sudo "
            "sparkring install --yes.")
    other = dict(fresh, revision="0" * 40)
    assert [item["kind"] for item in cp.envelope(reviewed, other)] == ["checkpoint"]


def test_attention_items():
    nothing = make([survey("spark-aa42"), survey("spark-931e", complete=False, stopped="time",
                                                   unvisited={"/home": 2}, seconds=40.3)])
    items = cp.attention(nothing)
    assert [item["kind"] for item in items] == ["download", "search-incomplete"]
    assert cp.attention_message(items) == (
        "The checkpoint plan downloads 98.6 GiB from huggingface.co on Node 0 spark-aa42, and the search on Node 1 "
        "spark-931e stopped at its time limit (not searched: 2 folders under /home). Setup's approval did not show "
        "this plan. Review it with sudo sparkring install --plan, or approve it with sudo sparkring install --yes. "
        "Nothing has been changed.")
    assert cp.summary(nothing)["nodes"][1]["search"]["unvisited"] == {"/home": 2}
    assert "The search on spark-931e stopped at its time limit (not searched: 2 folders under /home)." in \
        cp.describe(nothing)[1]
    failed = make([survey("spark-aa42", candidates=[candidate(FOLDER)]), TimeoutError("ssh: timed out after 150 s")])
    items = cp.attention(failed)
    assert items == [{"kind": "search-failed", "rank": 1,
                      "text": "the search on Node 1 root@192.0.2.2 failed (ssh: timed out after 150 s)"}]
    assert cp.attention_message(items).startswith("The search on Node 1 root@192.0.2.2 failed")
    assert actions(failed["nodes"][1], "receive") == REQUIRED
    assert "Node 1 root@192.0.2.2: search failed: ssh: timed out after 150 s" in cp.describe(failed)
    small = candidate(FOLDER, [n for n in REQUIRED if n != WEIGHTS[-1]])
    assert cp.attention(make([survey("a", candidates=[small]), survey("b")])) == []
    entries = make([survey("a", complete=False, stopped="entries"), survey("b", candidates=[candidate(FOLDER)])])
    assert cp.attention(entries)[0]["text"] == "the search on Node 0 a stopped at its entry limit"


def test_userns_remap_requires_world_readable_link_sources():
    private = candidate(FOLDER, mode=0o640, overrides={WEIGHTS[0]: {"mode": 0o644}})
    remapped = make([survey("a", candidates=[private], userns=True), survey("b")])["nodes"][0]
    assert actions(remapped, "link") == [WEIGHTS[0]]
    assert actions(remapped, "copy") == sorted(set(REQUIRED) - {WEIGHTS[0]})
    plain = make([survey("a", candidates=[private]), survey("b")])["nodes"][0]
    assert actions(plain, "link") == WEIGHTS


def test_storage_messages_match_the_documented_text():
    hosts = HOSTNAMES[4]
    usb = candidate("/mnt/usb/qwen", device=2049, mount_id=51, usb=True)
    ring = make([survey(hosts[0], 4, candidates=[usb], free_gib=44)]
                + [survey(hosts[rank], 4, candidates=[candidate(FOLDER)]) for rank in (1, 2, 3)])
    assert actions(ring["nodes"][0], "receive") == REQUIRED
    assert ring["problems"] == [{"field": "storage", "rank": 0, "message": (
        "Node 0 spark-edfd needs 134.9 GiB free on / to receive the checkpoint (98.6 GiB of checkpoint files, "
        "4.2 GiB to stage the largest file, 32 GiB for the compile cache); 44.0 GiB is free. Free another 90.9 GiB "
        "there, or put a copy of the pinned checkpoint on that filesystem: SparkRing hard-links its weight files, so "
        "it would need 32.1 GiB. Then repeat sudo sparkring install --plan. An exact copy was found on another "
        "filesystem at /mnt/usb/qwen; to serve it in place instead, review sudo sparkring install --model-path "
        "0=/mnt/usb/qwen --plan. The running model has not been stopped.")}]
    # The figure a linked copy needs counts the other files, the cache and, without the image, the image.
    bare = make([survey(hosts[0], 4, free_gib=44)] + [survey(hosts[rank], 4, candidates=[candidate(FOLDER)])
                                                      for rank in (1, 2, 3)],
                images_present=[False, True, True, True], request={"profile": "qwen38-flash-next-qad-tp4"})
    message = bare["problems"][0]["message"]
    assert "(98.6 GiB of checkpoint files, 4.2 GiB to stage the largest file, 32 GiB for the compile cache, " \
           "68 GiB for the image); 44.0 GiB is free. Free another 158.9 GiB there" in message
    assert "so it would need 100.1 GiB. Then repeat sudo sparkring install --profile qwen38-flash-next-qad-tp4 " \
           "--plan. The running model has not been stopped." in message
    main_copy = candidate("/mnt/usb/qwen", device=2049, mount_id=51, commit=MAIN, branches=["main"],
                          overrides={"config.json": {"state": "differs"}})
    pair = make([survey("spark-aa42", candidates=[candidate(FOLDER)]),
                 survey("spark-931e", candidates=[main_copy], free_gib=44)], named=["1=/mnt/usb/qwen"])
    assert pair["problems"] == [{"field": "storage", "rank": 1, "message": (
        "Node 1 spark-931e: /mnt/usb/qwen is on another filesystem and holds the main branch's config.json, so "
        "SparkRing cannot serve it in place, and copying it needs 134.9 GiB free on / (44.0 GiB free). To make that "
        "folder an exact copy yourself: hf download local-inference-lab/Qwen3.8-Flash-Next-NVFP4 config.json "
        "--revision 629bc3218833a38b475b719f34aa571666f4a03e --local-dir /mnt/usb/qwen (this changes your folder; "
        "if config.json there is a hard link to another copy, remove it first). Then repeat sudo sparkring install "
        "--model-path 1=/mnt/usb/qwen --plan. The running model has not been stopped.")}]
    nfs = survey("a", candidates=[candidate(FOLDER)])
    nfs["owned"]["fstype"] = "nfs4"
    assert make([nfs, survey("b")])["problems"][0]["message"].startswith(
        "Node 0 a: the checkpoint directory's filesystem at /srv/sparkring/tp2 is nfs4, not a local filesystem")


def lookalike_plan():
    folder = "/var/tmp/sparkring-test/lookalike"
    plain = candidate(folder, layout="folder", evidence="size", state="size-only",
                      overrides={name: {"state": "match", "evidence": "hashed"} for name in SMALL})
    return make([survey("spark-aa42", candidates=[plain]), survey("spark-931e")]), folder


def test_guard_reports_unplanned_downloads_and_writes():
    approved, folder = lookalike_plan()
    assert approved["hub_files"] == [] and approved["distribution"]["donor"] == 0
    wrong = [WEIGHTS[1], WEIGHTS[4]]
    results = [{"complete": False, "verified": {n: [] for n in REQUIRED if n not in wrong}, "missing": wrong,
                "differs": [{"name": n, "source": f"{folder}/{n}"} for n in wrong], "bytes_written": SMALL_BYTES},
               {"complete": False, "verified": {}, "missing": REQUIRED, "differs": [], "bytes_written": 0}]
    items = cp.unplanned(approved, results)
    assert [item["kind"] for item in items] == ["download"]
    assert cp.unplanned_message(items) == (
        "Node 0 spark-aa42: 2 files in /var/tmp/sparkring-test/lookalike are not the pinned model's "
        "(model-00002-of-00036.safetensors, model-00005-of-00036.safetensors); that folder holds another fine-tune "
        "or damaged files. No Spark holds the pinned files, so they need 3.35 GiB from huggingface.co, which the "
        "approved plan did not include. Nothing was downloaded and the running model was not changed. Review the "
        "resulting plan with sudo sparkring install --plan.")
    within = [{"verified": {n: [] for n in REQUIRED}, "bytes_written": SMALL_BYTES}, None]
    assert cp.unplanned(approved, within) == []
    assert cp.redistribute(approved, within)["receive"][0]["names"] == REQUIRED

    linked = make([survey("spark-aa42", candidates=[candidate(FOLDER)]),
                   survey("spark-931e", candidates=[candidate("/data/models/qwen")])])
    results = [{"verified": {n: [] for n in REQUIRED}, "bytes_written": SMALL_BYTES},
               {"verified": {n: [] for n in SMALL}, "bytes_written": SMALL_BYTES,
                "missing": [{"name": n, "reason": "EXDEV", "source": f"/data/models/qwen/{n}"} for n in WEIGHTS]}]
    items = cp.unplanned(linked, results)
    assert cp.unplanned_message(items) == (
        f"Node 1 spark-931e: hard links from /data/models/qwen into {owned_path(2)} failed (they are different mounts "
        "of one filesystem), so 98.6 GiB must be received over the fabric instead, which the approved plan did not "
        "include. Nothing more was written; the running model was not changed. Review the resulting plan with sudo "
        "sparkring install --plan.")
    # Immutable files fail the same links on every plan, so the message names the ways out instead of a new plan.
    immutable = [results[0], {**results[1], "missing": [{**item, "reason": "EPERM"} for item in results[1]["missing"]]}]
    reviewed = {**linked, "request": {"profile": "qwen38-flash-next-tp2"}, "command": "sudo sparkring install "
                "--profile qwen38-flash-next-tp2"}
    items = cp.unplanned(reviewed, immutable)
    assert cp.unplanned_message(items, reviewed["command"]) == (
        f"Node 1 spark-931e: hard links from /data/models/qwen into {owned_path(2)} failed (the files are immutable "
        "or append-only), so 98.6 GiB must be received over the fabric instead, which the approved plan did not "
        "include. SparkRing does not change those files: remove the attribute yourself (lsattr shows it), name "
        "another copy with --model-path 1=PATH, or repeat sudo sparkring install --profile qwen38-flash-next-tp2 "
        "--ignore-local-copies --plan. Nothing more was written; the running model was not changed.")
    small_growth = [results[0], {**results[1], "verified": {n: [] for n in REQUIRED if n != WEIGHTS[-1]},
                                 "missing": [WEIGHTS[-1]]}]
    assert cp.unplanned(linked, small_growth) == []


def test_install_command_repeats_the_deployment_request():
    assert cp.install_command() == "sudo sparkring install"
    request = {"profile": "qwen38-flash-next-tp2", "cache_path": "/mnt/fast cache", "image_lock": "locks/dev.json"}
    assert cp.install_command(request, ["/data/qwen", "1=/mnt/usb/qwen"], ignore_local=True) == (
        "sudo sparkring install --profile qwen38-flash-next-tp2 --model-path /data/qwen --model-path 1=/mnt/usb/qwen "
        "--cache-path '/mnt/fast cache' --image-lock locks/dev.json --ignore-local-copies")
    result = make(owner_copy_surveys(2), named=["1=" + FOLDER], ignore_local=True, request=request)
    assert result["command"] == cp.install_command(request, ["1=" + FOLDER], ignore_local=True)
    assert result["request"] == request and result["profile"] == "qwen38-flash-next-tp2"


def test_adoption_input_and_saved_plan():
    result = make(owner_copy_surveys(2), operator="code", named=["/unused"])
    entry = cp.adoption(result, 1, receipts=["/srv/sparkring/tp2/x/installer/model.json"])
    assert entry["tolerance_bytes"] == GIB and entry["receipts"] == ["/srv/sparkring/tp2/x/installer/model.json"]
    assert sorted(entry["files"]) == REQUIRED
    assert entry["files"][WEIGHTS[0]] == {"action": "link", "source": f"{FOLDER}/{WEIGHTS[0]}",
                                          "identity": identity(FOLDER, WEIGHTS[0], DEV), "size": SIZES[WEIGHTS[0]]}
    assert result["schema"] == "sparkring-checkpoint-plan/v1" and result["reviewed"] is False
    assert result["pins_sha256"] == cp.pins_digest(PINS) and result["created_at"] == "2026-09-26T02:00:00+00:00"
    assert result["operator"] == "code" and result["named"] == [{"rank": None, "path": "/unused"}]
    assert result["problems"][0]["message"] == "--model-path /unused exists on no Spark. Nothing has been changed."
    again = json.loads(json.dumps(result))
    assert cp.describe(again) == cp.describe(result) and cp.summary(again) == cp.summary(result)
    assert cp.envelope(again, result) == []
