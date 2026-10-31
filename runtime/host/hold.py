"""Operator holds: one record on Node A that reserves the cluster for one operator's longer task.

``install.lock`` and the slice locks (``runtime.host.slices``) serialize single
operations. A hold covers the time between them, such as a checkpoint
experiment or an acceptance run during which a model must keep serving:

- ``sudo sparkring hold --reason TEXT [--until TIME]`` records the holder, the
  reason, the start and an optional expiry in ``hold.json`` beside the
  controller records (``sparkring-hold/v1``). It survives a reboot of Node A.
  ``sudo sparkring hold --release`` clears it; ``sparkring hold`` shows it.
- While another operator's hold stands, ``install``, ``setup``, ``up``,
  ``down``, ``restart``, ``recover on|off``, ``storage --release`` and
  ``checkpoints --release`` refuse and name the holder and the reason
  (``runtime.host.activity``, ``require``). ``--override-hold`` passes the hold
  and appends the override to ``hold-overrides.jsonl``
  (``sparkring-hold-override/v1``). The entry records that the command
  passed the hold; a command that then stops at a later check or prompt
  also leaves one. Automatic recovery never overrides: it
  observes, and acts after the hold ends.
- ``sparkring status`` shows a standing hold on its first line, and the
  operator view (``runtime.host.operator_view``) carries it to members of the
  ``sparkring`` group.

The holder is the operator's login: ``SUDO_USER`` under sudo, else the
process's user. Sessions that share one login, such as an operator and an
agent session under the same account, are one operator, so a hold does not
refuse them. A hold whose ``--until`` has passed no longer refuses anything.
Holds are separate from the dashboard's quiet holders
(``runtime.host.dashboard``), which pause the dashboard around one
measurement and only warn.
"""
from __future__ import annotations

import argparse
import datetime
import getpass
import json
import os
import re
import socket
import sys
import time
from pathlib import Path

from runtime.host import node, privilege
from runtime.host.install_errors import NeedsInput

SCHEMA = "sparkring-hold/v1"
OVERRIDE_SCHEMA = "sparkring-hold-override/v1"
NAME = "hold.json"
OVERRIDES = "hold-overrides.jsonl"
REASON_LIMIT = 200
DURATION_RE = re.compile(r"([1-9][0-9]{0,4})([mhd])")
DURATION_UNITS = {"m": 60, "h": 3600, "d": 86400}
OVERRIDE_FLAG = "--override-hold"


def operator() -> str:
    """The login of the operator running this process: ``SUDO_USER`` under sudo, else the process's user."""
    sudo_user = os.environ.get("SUDO_USER", "").strip()
    if sudo_user and sudo_user != "root":
        return sudo_user
    try:
        return getpass.getuser()
    except (KeyError, OSError):
        return "uid-" + str(os.getuid()) if hasattr(os, "getuid") else "unknown"


def parse_until(text: str, now: float) -> float:
    """Seconds since the epoch for ``--until``: a duration (``90m``, ``4h``, ``2d``) or an ISO 8601 time.

    An ISO time without a zone is Node A's local time. The time must be in the future.
    """
    value = (text or "").strip()
    duration = DURATION_RE.fullmatch(value)
    if duration:
        until = now + int(duration.group(1)) * DURATION_UNITS[duration.group(2)]
    else:
        try:
            moment = datetime.datetime.fromisoformat(value)
        except ValueError:
            raise ValueError(f"--until takes a duration such as 90m, 4h or 2d, or a time such as "
                             f"2026-11-02T18:00, not {text!r}") from None
        until = moment.timestamp() if moment.tzinfo else time.mktime(moment.timetuple())
    if until <= now:
        raise ValueError(f"--until {text} is not in the future")
    return float(until)


def _valid(record) -> bool:
    return (isinstance(record, dict) and record.get("schema") == SCHEMA
            and isinstance(record.get("holder"), str) and record["holder"]
            and isinstance(record.get("reason"), str) and isinstance(record.get("since"), (int, float))
            and (record.get("until") is None or isinstance(record.get("until"), (int, float))))


def current(state_root, now: float | None = None) -> dict | None:
    """The standing hold at ``state_root``, or None: no record, an unreadable one, or one whose ``until`` passed.

    A record root cannot read raises ``PermissionError``; callers without root read the operator view.
    """
    now = time.time() if now is None else now
    try:
        record = json.loads((Path(state_root) / NAME).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except ValueError:
        return None
    if not _valid(record):
        return None
    if record.get("until") is not None and record["until"] <= now:
        return None
    return record


def foreign(state_root, now: float | None = None, who: str | None = None) -> dict | None:
    """The standing hold when another operator holds it, else None."""
    record = current(state_root, now)
    return record if record is not None and record["holder"] != (who or operator()) else None


def _moment(value: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M %Z", time.localtime(value)).strip()


def text(record: dict) -> str:
    """One line: ``Held by NAME since TIME [until TIME]: REASON``."""
    line = f"Held by {record['holder']} since {_moment(record['since'])}"
    if record.get("until") is not None:
        line += f" until {_moment(record['until'])}"
    return line + ": " + (record.get("reason") or "no reason given")


def refusal(record: dict, operation: str) -> str:
    return (f"{text(record)}. {operation} would change what this hold reserves; ask {record['holder']} "
            f"or repeat with {OVERRIDE_FLAG} to proceed (the override is recorded).")


def record_override(path, record: dict, operation: str, *, positions=None, now: float | None = None,
                    who: str | None = None) -> dict:
    """Append one override to ``path`` (``hold-overrides.jsonl``); returns the appended entry."""
    entry = {"schema": OVERRIDE_SCHEMA, "at": time.time() if now is None else now, "operator": who or operator(),
             "operation": operation, "positions": list(positions) if positions is not None else None,
             "hold": {key: record.get(key) for key in ("holder", "reason", "since", "until")}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(entry, sort_keys=True) + "\n")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return entry


def require(state_root, operation: str, *, override: bool = False, plan: bool = False, positions=None,
            now: float | None = None, who: str | None = None) -> dict | None:
    """Refuse ``operation`` while another operator's hold stands, unless ``override``; returns that hold.

    For commands outside ``runtime.host.activity``'s assessment, such as
    ``storage --release``. A plan prints the hold and proceeds; an override is
    recorded when the hold is passed, before the command's later checks.
    """
    record = foreign(state_root, now, who)
    if record is None:
        return None
    if plan:
        print(text(record), file=sys.stderr)
        return record
    if not override:
        raise NeedsInput(refusal(record, operation), field="hold", details={"hold": record})
    record_override(Path(state_root) / OVERRIDES, record, operation, positions=positions, now=now, who=who)
    print(f"{text(record)}. Proceeding with {OVERRIDE_FLAG}; the override is recorded.", file=sys.stderr)
    return record


def take(state_root, reason: str, until: float | None = None, *, override: bool = False,
         now: float | None = None, who: str | None = None) -> dict:
    """Record a hold for ``who``; replaces this operator's own hold, and another's only with ``override``."""
    now = time.time() if now is None else now
    who = who or operator()
    reason = " ".join((reason or "").split())
    if not reason or len(reason) > REASON_LIMIT or not reason.isprintable():
        raise ValueError(f"--reason is 1 to {REASON_LIMIT} printable characters")
    standing = foreign(state_root, now, who)
    if standing is not None:
        if not override:
            raise NeedsInput(refusal(standing, "A new hold"), field="hold", details={"hold": standing})
        record_override(Path(state_root) / OVERRIDES, standing, "hold", now=now, who=who)
    record = {"schema": SCHEMA, "holder": who, "reason": reason, "since": now, "until": until,
              "host": socket.gethostname()}
    node.save(state_root, NAME, record, mode=0o644)
    return record


def release(state_root, *, override: bool = False, now: float | None = None, who: str | None = None) -> dict | None:
    """Clear the standing hold; another operator's only with ``override``. Returns the cleared hold, if any."""
    who = who or operator()
    record = current(state_root, now)
    if record is not None and record["holder"] != who:
        if not override:
            raise NeedsInput(refusal(record, "Releasing it"), field="hold", details={"hold": record})
        record_override(Path(state_root) / OVERRIDES, record, "hold --release", now=now, who=who)
    (Path(state_root) / NAME).unlink(missing_ok=True)
    return record


def add_override_argument(parser) -> None:
    parser.add_argument(OVERRIDE_FLAG, dest="override_hold", action="store_true",
                        help="proceed although another operator holds the cluster (sparkring hold); "
                             "the override is recorded")


def main(argv=None) -> int:
    from runtime.host import controller, dashboard_admin, operator_view

    parser = argparse.ArgumentParser(
        prog="sparkring hold",
        description="Show, take or release a hold on the cluster: other operators' install, setup, up, down, "
                    "restart, recover on|off and storage or checkpoint releases refuse while it stands.")
    parser.add_argument("--reason", help=f"why the cluster is held, at most {REASON_LIMIT} characters")
    parser.add_argument("--until", help="when the hold ends by itself: a duration such as 4h or 2d, or a time "
                                        "such as 2026-11-02T18:00 (Node A's local time)")
    parser.add_argument("--release", action="store_true", help="clear the hold")
    add_override_argument(parser)
    parser.add_argument("--json", action="store_true", help="print one JSON document")
    args = parser.parse_args(argv)
    state_root = controller.STATE
    try:
        if args.release and (args.reason or args.until):
            raise ValueError("--release takes no --reason or --until")
        if args.until and not args.reason:
            raise ValueError("--until needs --reason")
        if args.release or args.reason:
            if not privilege.running_as_root():
                raise ValueError("Run sudo sparkring hold on Node A")
            if not (state_root / "cluster.json").is_file():
                raise ValueError("No cluster is recorded here; run sparkring hold on Node A after setup")
            now = time.time()
            if args.release:
                cleared = release(state_root, override=args.override_hold, now=now)
                result = {"state": "released" if cleared else "none", "hold": None, "released": cleared}
                message = "Hold released." if cleared else "No hold stands."
            else:
                until = parse_until(args.until, now) if args.until else None
                record = take(state_root, args.reason, until, override=args.override_hold, now=now)
                result = {"state": "held", "hold": record}
                message = text(record)
            dashboard_admin.refresh_site(reload=False)
        else:
            try:
                record = current(state_root)
            except PermissionError:
                document = operator_view.load() if privilege.member() else None
                if document is None:
                    raise ValueError("To read the hold, " + privilege.reader_request()) from None
                record = document.get("hold")
                if record and record.get("until") is not None and record["until"] <= time.time():
                    record = None
            result = {"state": "held" if record else "none", "hold": record}
            message = text(record) if record else "No hold stands."
        print(json.dumps({"schema": SCHEMA, **result}, indent=2) if args.json else message)
        return 0
    except NeedsInput as error:
        print(json.dumps(error.document()) if args.json else "SparkRing: " + str(error),
              file=sys.stdout if args.json else sys.stderr)
        return 3
    except (OSError, ValueError) as error:
        print("SparkRing: " + str(error), file=sys.stderr)
        return 2
