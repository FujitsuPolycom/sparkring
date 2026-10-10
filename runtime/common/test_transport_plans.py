"""The plans of SIRCL and prepared-transport deployments stay those recorded in fixtures/transport-plans.json.

``render`` builds, for each deployment of ``CASES``, what the installer
plans for it: the deployment lock's ``transport`` section, the plan text,
the lock's identity and every rank's container specification. The fixture
holds the SHA-256 of each rendered part (the plan text in full), as revision
``generated_at`` of the fixture renders them. A change that is meant to alter
these plans rewrites the fixture in the same commit:

    python -m runtime.common.test_transport_plans --write
"""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

from runtime.common import image_lock, installer, transport
from runtime.common.test_image_lock import sircl_lock
from runtime.common.test_transport import TP2, TP4, document, install_site, sircl_deployment

FIXTURE = Path(__file__).with_name("fixtures") / "transport-plans.json"
SCHEMA = "sparkring-transport-plan-digests/v1"
TP8 = "glm53-flash-csf-tp8"
# (case, profile, fabric shape, fabric size, positions, NCCL mode); None as the mode: the prepared transport.
CASES = (
    ("sircl-pair", TP2, "pair", 2, [0, 1], "never"),
    ("sircl-pair-nccl-auto", TP2, "pair", 2, [0, 1], "auto"),
    ("sircl-cycle-4", TP4, "cycle", 4, [0, 1, 2, 3], "never"),
    ("sircl-path-4-of-cycle-8", TP4, "cycle", 8, [4, 5, 6, 7], "never"),
    ("sircl-cycle-8", TP8, "cycle", 8, list(range(8)), "never"),
    ("prepared-pair", TP2, "pair", 2, [0, 1], None),
    ("prepared-path-4", TP4, "path", 4, [0, 1, 2, 3], None),
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=list).encode()).hexdigest()


def _image(profile):
    if profile == TP8:
        profiles = sorted({*installer.installer_image.default_lock()["profiles"], *installer.installer_image.SIRCL_ONLY})
        return sircl_lock(profiles=profiles)
    return sircl_lock()


def _case(profile, shape, size, positions, nccl):
    image = _image(profile)
    if nccl is not None:
        lock, section = sircl_deployment(profile, shape, size, positions, nccl=nccl, image=image)
        lines = transport.plan_lines(section)
    else:
        placement = None if list(positions) == list(range(size)) else positions
        site = install_site(len(positions), placement=placement)
        if shape == "path":
            # The path's lock is the SIRCL one without its section, as the prepared transport plans the same site.
            lock, section = sircl_deployment(profile, shape, size, positions, image=image)
            lock = {key: value for key, value in lock.items() if key != "transport"}
        else:
            lock = installer.make_lock(profile, site, "1" * 40, "2" * 64, image_runtime=image_lock.v2_view(image))
        section = None
        lines = [transport.prepared_line(None, True), transport.prepared_line(None, False),
                 transport.prepared_line(f"image {image['name']} carries no SIRCL layer", False)]
    specs = [dataclasses.asdict(spec) for spec in installer.specifications(lock)]
    return {"plan_lines": lines, "lock_id": lock.get("id"), "section_sha256": _digest(section),
            "specs_sha256": [_digest(spec) for spec in specs]}


def _choices():
    """``transport.choose`` for the images and fabrics the cases use, without and with a requested transport."""
    rows = {}
    default = installer.installer_image.default_lock()
    for name, image, value in (("v2-no-fabric", default, None), ("v3-no-fabric", sircl_lock(), None),
                               ("v3-pair", sircl_lock(), document("pair", 2)),
                               ("v3-cycle-8", sircl_lock(), document("cycle", 8))):
        for backend in (None, "prepared"):
            rows[f"{name}/{backend}"] = list(transport.choose(image, value, backend=backend))
    return rows


def render():
    return {"cases": {case[0]: _case(*case[1:]) for case in CASES}, "choose": _choices()}


def test_sircl_and_prepared_plans_are_those_the_fixture_records():
    recorded = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert recorded["schema"] == SCHEMA
    rendered = render()
    assert sorted(rendered["cases"]) == sorted(recorded["cases"])
    for name, value in rendered["cases"].items():
        assert value == recorded["cases"][name], name
    assert rendered["choose"] == recorded["choose"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", action="store_true", help="rewrite the fixture from this checkout")
    parser.add_argument("--revision", help="the revision the fixture records as generated_at")
    args = parser.parse_args(argv)
    value = {"schema": SCHEMA, "generated_at": args.revision, **render()}
    text = json.dumps(value, indent=1, sort_keys=True) + "\n"
    if args.write:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(text, encoding="utf-8", newline="\n")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
