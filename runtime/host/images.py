"""List the installer images this package can run, for `sudo sparkring install --image NAME`."""
import argparse
import json

from runtime.common import installer_image, profiles


def rows(profile=None):
    """One row per installer image, the default first; with ``profile``, only images that run it."""
    listed = []
    for row in installer_image.catalog():
        admitted = [name for name in installer_image.profiles_of(row["lock"]) if name not in profiles.REPLACED]
        if profile is not None and profile not in admitted:
            continue
        listed.append({"name": row["name"], "tags": row["tags"], "default": row["default"],
                       "image_reference": row["lock"]["image_reference"],
                       "runtime_status": row["lock"]["status_version"],
                       "download_bytes": row["lock"].get("download_bytes"), "profiles": admitted})
    return listed


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sparkring images",
                                     description="List the installer images this package can run, the default first.")
    parser.add_argument("--profile", help="only images that run this installer profile")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.profile is not None and args.profile not in installer_image.SUPPORTED:
        replaced = profiles.replacement_message(args.profile)
        parser.error(replaced if replaced else f"{args.profile} is not an installer profile; sparkring models lists them")
    listed = rows(args.profile)
    if args.json:
        print(json.dumps(listed, indent=2))
        return 0
    for row in listed:
        marks = [*row["tags"], *(["default"] if row["default"] else [])]
        print(row["name"] + (f"  ({', '.join(marks)})" if marks else ""))
        size = f"{row['download_bytes'] / 2**30:.1f} GiB download" if row["download_bytes"] else "download size not recorded"
        missing = [name for name in installer_image.SUPPORTED if name not in row["profiles"]]
        if not missing:
            runs = "every installer profile"
        elif len(missing) < len(row["profiles"]):
            runs = "every installer profile except " + ", ".join(missing)
        else:
            runs = ", ".join(row["profiles"])
        print(f"  {size}; runs {runs}")
    print("Install a profile on one of them: sudo sparkring install --profile PROFILE --image NAME")
    return 0
