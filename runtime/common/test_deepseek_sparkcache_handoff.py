"""The documented source render produces both four-rank SIRCL arms offline."""

import json
import re

from runtime.common import distribution, installer, installer_image
from runtime.common.test_transport import sircl_deployment
from runtime.images import derived_layer


def test_documented_render_keeps_transport_and_generates_both_arms(tmp_path, monkeypatch):
    root = installer_image.ROOT
    composition = root / "runtime/images/compositions/installer-sparkcache-4a17f1f5198e"
    record = json.loads((composition / "sources.json").read_text())
    build = json.loads((composition / "build-inputs.json").read_text())
    parent = json.loads((root / record["parent_lock"]).read_text())
    prior, _ = sircl_deployment("deepseek-v41-flash-tp4", "cycle", 8, [0, 1, 2, 3], image=parent)
    plan = {"parent_lock": parent, "payload_bytes": 1,
            "receipts": {derived_layer.BASE_RECEIPT: "7" * 64, derived_layer.TOOLCHAIN_RECEIPT: "8" * 64}}
    built = derived_layer.derived_lock(plan, {"Id": "sha256:" + "9" * 64, "Size": 1},
                                       build["name"], build["profiles"])
    (tmp_path / "dsv41-prior.json").write_text(json.dumps(prior))
    (tmp_path / "dsv41-image.json").write_text(json.dumps(built))
    monkeypatch.setattr(distribution, "identity", lambda root: "1" * 40)
    monkeypatch.setattr(distribution, "bundle", lambda root, output: output.write_bytes(b"offline bundle fixture"))
    guide = (root / "profiles/deepseek-v41-flash-sparkcache-tp4/HARDWARE.md").read_text()
    scripts = re.findall(r"python3 - <<'PY'\n(.*?)\nPY", guide, re.S)
    render = next(code for code in scripts if "installer.init(" in code)
    exec(compile(render.replace("/var/tmp", tmp_path.as_posix()), "HARDWARE.md render", "exec"), {})
    specs = {}
    for arm in ("on", "off"):
        directory = tmp_path / ("dsv41-cache-" + arm)
        lock = installer.load(directory)
        assert lock["transport"]["group"]["positions"] == [0, 1, 2, 3]
        assert lock["transport"]["nccl"] == "never"
        specs[arm] = installer.specifications(lock)
        assert len(specs[arm]) == 4
        assert len(list(directory.glob("rank*/compose.yaml"))) == 4
        for spec in specs[arm]:
            assert spec.image_id == built["image_id"]
            assert spec.environment["SPARKCACHE_ENABLED"] == ("1" if arm == "on" else "0")
            assert spec.memory == 115964116992
    for on, off in zip(specs["on"], specs["off"], strict=True):
        before, after = list(off.command), list(on.command)
        index = after.index("--kv-transfer-config")
        del after[index:index + 2]
        assert before == after


def test_deepseek_research_ring_requires_all_four_fabric_references():
    import pytest
    from runtime.common.test_sparkcache_variants import development_lock, ring_site
    raw = ring_site(4)
    del raw["hosts"][2]["fabric"]
    with pytest.raises(ValueError, match="mesh fabric reference"):
        installer.make_lock("deepseek-v41-flash-sparkcache-tp4", raw, "1" * 40, "2" * 64,
                            image_runtime=development_lock())
