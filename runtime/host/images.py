"""List the installer images this package can run, for `sudo sparkring install --image NAME`."""
import argparse
import json

from runtime.common import image_lock, installer_image, profiles


def rows(profile=None):
    """One row per installer image, the install default first; with ``profile``, only images that run it.

    Each row names the image's line and the collective transports it carries
    (``runtime/common/image_lock.py``); ``archived`` marks an archived image,
    which ``--image`` still selects.
    """
    listed = []
    for row in image_lock.catalog():
        admitted = [name for name in image_lock.profiles_of(row["lock"]) if name not in profiles.REPLACED]
        if profile is not None and profile not in admitted:
            continue
        listed.append({"name": row["name"], "tags": row["tags"], "default": row["default"],
                       "image_reference": row["lock"]["image_reference"],
                       "runtime_status": row["lock"]["status_version"],
                       "download_bytes": row["lock"].get("download_bytes"), "profiles": admitted,
                       "line": row["line"], "transports": row["transports"], "archived": row["archived"]})
    return listed


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sparkring images",
                                     description="List the installer images this package can run, the default first.")
    parser.add_argument("--profile", help="only images that run this installer profile")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.profile is not None:
        from runtime.host import models
        args.profile = models.canonical(args.profile)
    if args.profile is not None and args.profile not in installer_image.SUPPORTED:
        replaced = profiles.replacement_message(args.profile)
        parser.error(replaced if replaced else f"{args.profile} is not an installer profile; sparkring models lists them")
    listed = rows(args.profile)
    if args.json:
        print(json.dumps(listed, indent=2))
        return 0
    for row in listed:
        marks = [*row["tags"], *(["default"] if row["default"] else []), *(["archived"] if row["archived"] else [])]
        print(row["name"] + (f"  ({', '.join(marks)})" if marks else ""))
        size = f"{row['download_bytes'] / 2**30:.1f} GiB download" if row["download_bytes"] else "download size not recorded"
        missing = [name for name in installer_image.SUPPORTED if name not in row["profiles"]]
        if not missing:
            runs = "every installer profile"
        elif len(missing) < len(row["profiles"]):
            runs = "every installer profile except " + ", ".join(missing)
        else:
            runs = ", ".join(row["profiles"])
        carries = ", ".join(row["transports"]) + (f"; {row['line']} line" if row["line"] else "")
        print(f"  {size}; transports {carries}; runs {runs}")
    print("Install a profile on one of them: sudo sparkring install --profile PROFILE --image NAME")
    return 0
