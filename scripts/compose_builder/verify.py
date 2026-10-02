"""Compare the Compose builder's engine with compose.build.

cases() draws random sites, checkpoints and serving settings for every listed
profile, invalid sites, and sites that compose.build accepts but the engine
refuses because it limits their characters (paths with spaces, IPv6
addresses). run() renders every case with engine.js under Node (run_engine.js)
and with compose.build, and reports each difference in the per-rank files,
deployment.json, site.yaml, deployment ID, serving-setting lines and refusal
messages. For a sample of valid cases it also opens the engine's deployment
archive with zipfile and requires compose.load_deployment to accept the
unzipped directory.
"""
import base64
import copy
import io
import json
from pathlib import Path
import random
import subprocess
import tempfile
import zipfile

import yaml

from runtime.common import compose
from runtime.common import serving as serving_settings

HERE = Path(__file__).resolve().parent
LOWER = "abcdefghijklmnopqrstuvwxyz"
DIGITS = "0123456789"
ALNUM = LOWER + LOWER.upper() + DIGITS
# Values whose YAML encoding differs from their plain spelling, and other edge spellings.
TRICKY_NAMES = ("on", "no", "yes", "null", "true", "a", "e5", "x-1")
TRICKY_HOSTS = ("spark0", "yes", "123", "1.5", "2026-10-01", "user@spark1", "a_b", "x.y-z", "0x10", "089", "NULL", "e5", "Off", "1_000")
TRICKY_INTERFACES = ("eth0", "enP7s7", "123", "on", "1.5", "_x", "a.b-c", "0x1", "enp1s0f0np0", "Yes", "2026-10-01")
TRICKY_HCAS = ("rocep1s0f0", "roceP2p1s0f0", "0", "1", "mlx5_0", "on", "123", "a_b", "089")
TRICKY_SEGMENTS = ("on", "123", "2026-10-01", "srv", "models", "v1.2", "~x", "a=b", "x@y", "-z", "+p")


class Sites:
    """Random sites whose values compose.build and the engine both accept."""

    def __init__(self, seed):
        self.rng = random.Random(seed)

    def chars(self, alphabet, low, high):
        return "".join(self.rng.choice(alphabet) for _ in range(self.rng.randint(low, high)))

    def name(self):
        if self.rng.random() < 0.3:
            return self.rng.choice(TRICKY_NAMES)
        return self.rng.choice(LOWER) + self.chars(LOWER + DIGITS + "-", 0, 39)

    def host(self, used):
        while True:
            value = (self.rng.choice(TRICKY_HOSTS) if self.rng.random() < 0.4
                     else self.rng.choice(ALNUM) + self.chars(ALNUM + "_.@-", 0, 20))
            if value != "controller" and value not in used:
                used.add(value)
                return value

    def address(self, used):
        while True:
            value = ".".join(str(n) for n in (self.rng.randint(1, 254), self.rng.randint(0, 255),
                                               self.rng.randint(0, 255), self.rng.randint(1, 254)))
            if value not in used:
                used.add(value)
                return value

    def interface(self):
        if self.rng.random() < 0.4:
            return self.rng.choice(TRICKY_INTERFACES)
        return self.rng.choice(ALNUM + "_") + self.chars(ALNUM + "_.-", 0, 14)

    def hcas(self, count):
        found = []
        while len(found) < count:
            value = self.rng.choice(TRICKY_HCAS) if self.rng.random() < 0.4 else self.chars(ALNUM + "_", 1, 12)
            if value not in found:
                found.append(value)
        return found

    def segment(self):
        while True:
            value = (self.rng.choice(TRICKY_SEGMENTS) if self.rng.random() < 0.3
                     else self.chars(ALNUM + "._+=@~-", 1, 14))
            if value not in (".", ".."):
                return value

    def path(self, top):
        return "/" + top + "".join("/" + self.segment() for _ in range(self.rng.randint(0, 3)))

    def digest(self, digits_only):
        return self.chars(DIGITS if digits_only else DIGITS + "abcdef", 64, 64)

    def site(self, nodes):
        hosts, addresses = set(), set()
        shared = self.rng.random() < 0.5
        base = {"interface": self.interface(), "hcas": self.hcas(nodes), "gid": self.rng.randint(0, 255)}
        tops = ["m" + self.segment(), "c" + self.segment(), "r" + self.segment(), "d" + self.segment()]
        shared_paths = [self.path(top) for top in tops]
        ranks = []
        for number in range(nodes):
            paths = shared_paths if shared else [self.path(top) for top in tops]
            row = {"rank": number, "host": self.host(hosts), "host_ip": self.address(addresses),
                   "interface": base["interface"] if shared else self.interface(),
                   "hcas": list(base["hcas"]) if shared else self.hcas(nodes),
                   "gid": base["gid"] if shared else self.rng.randint(0, 255),
                   "model": paths[0], "cache": paths[1], "repository": paths[2], "deployment_root": paths[3]}
            if nodes == 4:
                row["fabric"] = {"site_path": self.path("f" + self.segment()),
                                 "site_sha256": self.digest(self.rng.random() < 0.2),
                                 "plan_sha256": self.digest(self.rng.random() < 0.2)}
            ranks.append(row)
        return {"schema": "sparkring-compose-site/v1", "name": self.name(), "master": ranks[0]["host_ip"], "ranks": ranks}

    def settings(self, checkpoint):
        chosen = {}
        for row in checkpoint["settings"]:
            if self.rng.random() < 0.5:
                continue
            if row.get("switch"):
                chosen[row["name"]] = True
            elif row["name"] == "kv_cache_gib":
                chosen[row["name"]] = self.rng.randint(1, row["maximum"])
            elif row["name"] == "context_length":
                chosen[row["name"]] = self.rng.randint(1024, row["profile"])
            else:
                chosen[row["name"]] = self.rng.randint(row["minimum"], max(row["minimum"], row["profile"] * 2))
        return chosen

    def checkpoint_name(self, checkpoint):
        """The name the page sends: None, the name or an alias for the default, else the name."""
        if checkpoint["default"]:
            return self.rng.choice([None, checkpoint["name"], *checkpoint["aliases"]])
        return checkpoint["name"]


def _rank(site, number):
    return site["ranks"][number]


# Invalid sites: each edit breaks one rule of compose.site_settings or the container specification.
INVALID = (
    lambda s: s.update(name="Bad"),
    lambda s: s.update(name="x" * 41),
    lambda s: _rank(s, 1).update(host=_rank(s, 0)["host"]),
    lambda s: _rank(s, 1).update(host_ip=_rank(s, 0)["host_ip"]),
    lambda s: _rank(s, 1).update(host_ip="300.1.1.1"),
    lambda s: _rank(s, 1).update(host_ip="01.2.3.4"),
    lambda s: _rank(s, 0).update(interface="x" * 16),
    lambda s: _rank(s, 0).update(interface="a/b"),
    lambda s: _rank(s, 0)["hcas"].__setitem__(1, _rank(s, 0)["hcas"][0]),
    lambda s: _rank(s, 0)["hcas"].pop(),
    lambda s: _rank(s, 1).update(gid=256),
    lambda s: _rank(s, 1).update(gid="3"),
    lambda s: _rank(s, 0).update(gid=True),
    lambda s: _rank(s, 0).update(model="relative/path"),
    lambda s: _rank(s, 0).update(cache=_rank(s, 0)["cache"] + "/"),
    lambda s: _rank(s, 0).update(cache=_rank(s, 0)["model"] + "/inner"),
    lambda s: _rank(s, 0).update(repository="/a,b"),
    lambda s: s.update(master="192.0.2.250"),
    lambda s: _rank(s, 0).update(host="controller"),
    lambda s: _rank(s, 0).pop("deployment_root"),
    lambda s: s.update(extra=1),
    lambda s: _rank(s, 0).update(rank=1),
    lambda s: (_rank(s, 0)["fabric"].update(site_sha256="A" * 64) if "fabric" in _rank(s, 0) else _rank(s, 0).update(gid=-1)),
    lambda s: (_rank(s, 0)["fabric"].pop("plan_sha256") if "fabric" in _rank(s, 0) else _rank(s, 0).update(model="/")),
)
# Sites compose.build accepts and the engine refuses: it limits paths to characters whose
# encodings it reproduces and addresses to IPv4.
STRICTER = (
    lambda s: _rank(s, 0).update(model="/srv/my models"),
    lambda s: (_rank(s, 0).update(host_ip="fd00::10"), s.update(master="fd00::10")),
)


def cases(data, *, per_checkpoint, seed, archive_every=5):
    """Valid, invalid and stricter cases for every profile and checkpoint of ``data``."""
    sites = Sites(seed)
    found = []
    for profile in data["profiles"]:
        nodes = profile["nodes"]
        found.append({"profile": profile["id"], "site": profile["example_site"], "settings": {}, "checkpoint": None,
                      "kind": "valid", "archive": True})
        for checkpoint in profile["checkpoints"]:
            for _ in range(per_checkpoint):
                found.append({"profile": profile["id"], "site": sites.site(nodes), "settings": sites.settings(checkpoint),
                              "checkpoint": sites.checkpoint_name(checkpoint), "kind": "valid",
                              "archive": len(found) % archive_every == 0})
            ceiling = next((row for row in checkpoint["settings"] if row["name"] == "kv_cache_gib"), None)
            if ceiling:
                found.append({"profile": profile["id"], "site": sites.site(nodes),
                              "settings": {"kv_cache_gib": ceiling["maximum"] + 1},
                              "checkpoint": sites.checkpoint_name(checkpoint), "kind": "invalid"})
        for kind, edits in (("invalid", INVALID), ("stricter", STRICTER)):
            for edit in edits:
                site = sites.site(nodes)
                edit(site)
                checkpoint = sites.rng.choice(profile["checkpoints"])
                found.append({"profile": profile["id"], "site": site, "settings": {},
                              "checkpoint": sites.checkpoint_name(checkpoint), "kind": kind})
        found.append({"profile": profile["id"], "site": sites.site(nodes), "settings": {"max_concurrency": 0},
                      "checkpoint": None, "kind": "invalid"})
    return found


def expected(case):
    """compose.build's result for a case, in the engine's output shape."""
    site = copy.deepcopy(case["site"])
    try:
        manifest, files = compose.build(case["profile"], site, checkpoint=case["checkpoint"], serving=case["settings"])
        settings = manifest.get("serving") or {}
        options = {key: value for key, value in compose.selection_options(manifest).items() if key != "serving"}
        command = compose.specifications(manifest["profile"], manifest["site"], **options)[0][0].command
    except (ValueError, KeyError, TypeError) as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "id": manifest["id"], "files": files, "deployment": compose.encoded(manifest),
            "site_yaml": yaml.safe_dump(site, sort_keys=False),
            "serving_lines": serving_settings.describe(settings, command),
            "warnings": serving_settings.warnings(settings, command), "manifest": manifest}


def engine(data, found, *, node="node", work=None):
    """The engine's outputs for ``found``, from run_engine.js under Node."""
    with tempfile.TemporaryDirectory(dir=work) as directory:
        data_path, cases_path = Path(directory) / "data.json", Path(directory) / "cases.json"
        data_path.write_text(json.dumps(data), encoding="utf-8")
        cases_path.write_text(json.dumps(found), encoding="utf-8")
        result = subprocess.run([node, str(HERE / "run_engine.js"), str(data_path), str(cases_path)],
                                capture_output=True, text=True, encoding="utf-8", check=False)
    if result.returncode != 0:
        raise RuntimeError("run_engine.js failed: " + result.stderr.strip()[-2000:])
    return json.loads(result.stdout)


def check_archive(case, output, want, work=None):
    """Differences between the engine's archive and the deployment compose.build writes."""
    problems = []
    root = case["site"]["name"] + "/"
    entries = {root + "deployment.json": want["deployment"], root + case["site"]["name"] + ".site.yaml": want["site_yaml"],
               **{root + name: text for name, text in want["files"].items()}}
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(output["archive"]["base64"]))) as archive:
        if archive.testzip() is not None:
            return ["archive CRC check failed"]
        for info in archive.infolist():
            mode = info.external_attr >> 16
            if mode != (0o40700 if info.filename.endswith("/") else 0o100600):
                problems.append(f"{info.filename} has mode {oct(mode)}")
        names = {info.filename for info in archive.infolist() if not info.filename.endswith("/")}
        if names != set(entries) | {root + "README.txt"}:
            problems.append("archive entries differ: " + ", ".join(sorted(names ^ (set(entries) | {root + "README.txt"}))))
        problems.extend(f"{name} differs from compose.build" for name, text in entries.items()
                        if name in names and archive.read(name) != text.encode())
        with tempfile.TemporaryDirectory(dir=work) as directory:
            archive.extractall(directory)
            try:
                if compose.load_deployment(Path(directory) / case["site"]["name"])[0] != want["manifest"]:
                    problems.append("compose.load_deployment returned another manifest")
            except ValueError as exc:
                problems.append("compose.load_deployment refused the unzipped archive: " + str(exc))
    return problems


def run(data, *, per_checkpoint=6, seed=20261002, node="node", work=None):
    """Compare the engine with compose.build; the summary lists every failure."""
    found = cases(data, per_checkpoint=per_checkpoint, seed=seed)
    outputs = engine(data, found, node=node, work=work)
    summary = {"valid": 0, "invalid": 0, "stricter": 0, "archives": 0, "seed": seed, "failures": []}
    keys = ("id", "files", "deployment", "site_yaml", "serving_lines", "warnings")
    for number, (case, output) in enumerate(zip(found, outputs, strict=True)):
        want = expected(case)
        label = f"case {number} ({case['profile']}, {case['kind']})"
        if case["kind"] == "stricter":
            if want["ok"] and not output["ok"]:
                summary["stricter"] += 1
            else:
                summary["failures"].append(f"{label}: compose.build should accept and the engine refuse")
            continue
        if want["ok"] != output["ok"]:
            summary["failures"].append(f"{label}: compose.build {want.get('error', 'accepts')!r}, engine {output.get('error', 'accepts')!r}")
            continue
        if not want["ok"]:
            if want["error"] == output["error"]:
                summary["invalid"] += 1
            else:
                summary["failures"].append(f"{label}: refusal {want['error']!r}, engine {output['error']!r}")
            continue
        differing = [key for key in keys if want[key] != output[key]]
        if differing:
            summary["failures"].append(f"{label}: engine differs in {', '.join(differing)}")
            continue
        summary["valid"] += 1
        if case.get("archive"):
            problems = check_archive(case, output, want, work)
            summary["failures"].extend(f"{label}: {problem}" for problem in problems)
            summary["archives"] += not problems
    return summary
