"""Presentation-only ring name, stored separately from cluster and deployment identities."""
import argparse
import json
from pathlib import Path

from runtime.host import node, privilege

FILE = "display-name.json"


def validate(value):
    if not isinstance(value, str):
        raise ValueError("Display name must be text")
    value = value.strip()
    if not value or len(value) > 40 or not value.isprintable():
        raise ValueError("Display name must contain 1 to 40 printable Unicode characters after trimming whitespace")
    return value


def read(state_root):
    try:
        return validate(json.loads((Path(state_root) / FILE).read_text(encoding="utf-8"))["name"])
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def effective(state_root, cluster=None):
    if cluster is None:
        try:
            cluster = json.loads((Path(state_root) / "cluster.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cluster = {}
    return read(state_root) or cluster.get("name") or "sparkring"


def save(state_root, value):
    """Atomically set the name; None removes the override. Callers own authorization/publication."""
    if value is None:
        (Path(state_root) / FILE).unlink(missing_ok=True)
    else:
        node.save(state_root, FILE, {"name": validate(value)}, mode=0o600)


def main(argv=None):
    from runtime.host import controller, dashboard_admin, operator_view

    parser = argparse.ArgumentParser(prog="sparkring name", description="Read or set the ring's display name on Node A")
    parser.add_argument("text", nargs="?")
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.clear and args.text is not None:
            raise ValueError("Use TEXT or --clear, not both")
        changing = args.clear or args.text is not None
        if changing:
            if not privilege.running_as_root():
                raise ValueError("Run sudo sparkring name on Node A to change the display name")
            # Only Node A has this controller record. Workers never write a local override.
            if not (controller.STATE / "cluster.json").is_file():
                raise ValueError("No cluster recorded here; run sparkring name on Node A after setup")
            save(controller.STATE, None if args.clear else args.text)
            dashboard_admin.refresh_site(reload=False)
        try:
            # Check access before resolving the fallback; do not hide a private controller directory.
            cluster = json.loads((controller.STATE / "cluster.json").read_text(encoding="utf-8"))
            value = effective(controller.STATE, cluster)
        except PermissionError:
            document = operator_view.load() if privilege.member() else None
            value = ((document or {}).get("cluster") or {}).get("name")
            if not value:
                raise ValueError("Reading the ring name needs " + privilege.reader_request()) from None
        print(value)
        return 0
    except (OSError, ValueError) as error:
        import sys
        print("SparkRing: " + str(error), file=sys.stderr)
        return 2
