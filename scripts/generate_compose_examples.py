"""Keep public Compose examples synchronized with canonical serving profiles."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.common import compose, installer, profiles, setup, standalone_compose  # noqa: E402


def examples():
    for profile in standalone_compose.SUPPORTED:
        yield ROOT / "profiles" / profile / "compose/standalone.yaml", standalone_compose.render(profile)
    for profile in compose.SUPPORTED:
        # A SparkCache variant shares its base profile's site example.
        owner = profile.removesuffix("-sparkcache")
        source = ROOT / "profiles" / owner / "compose/site.example.yaml"
        site = compose.read_site(source)
        manifest, files = compose.build(profile, site)
        for rank in range(len(site["ranks"])):
            yield ROOT / "profiles" / profile / "compose" / f"compose.rank{rank}.yaml", files[
                f"rank{rank}/compose.yaml"
            ]
        if manifest.get("image_runtime") is not None:
            # Keep the selection beside the public files: profile.json's release
            # and config.json's default can predate what `sparkring install` runs.
            definition, _ = profiles.load(profile)
            config = profiles.read_json(ROOT / definition["configuration"]["path"])
            checkpoint = manifest.get("checkpoint") or config.get("checkpoint")
            selection = {"schema": "sparkring-compose-selection/v1", "profile": profile,
                         "checkpoint": checkpoint, "image_lock": compose.installer_image_lock_path(),
                         "deployment": manifest["id"]}
            directory = ROOT / "profiles" / profile / "compose"
            yield directory / "selection.json", json.dumps(selection, indent=2) + "\n"
            pins = installer.checkpoint_pins(setup.selection(profile, checkpoint))
            sums = "".join(f"{pins['files'][name]['sha256']}  {name}\n"
                           for name in sorted(set(pins["files"]) - set(pins["optional"])))
            yield directory / "SHA256SUMS", sums


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if any generated public example differs",
    )
    args = parser.parse_args(argv)
    stale = []
    for path, expected in examples():
        if args.check:
            if not path.is_file() or path.read_bytes() != expected.encode():
                stale.append(path.relative_to(ROOT).as_posix())
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(expected, encoding="utf-8", newline="\n")
    if stale:
        print(
            "Regenerate Compose examples with python scripts/generate_compose_examples.py:\n"
            + "\n".join(stale)
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
