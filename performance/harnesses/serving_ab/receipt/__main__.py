"""``python -m performance.harnesses.serving_ab.receipt COMMAND``: run receipts of serving benchmarks.

Commands (README.md in this package's parent directory, "Run receipts"):

- ``collect --node-a TARGET --positions P,Q --output DIR``: read what the deployment on those positions
  serves and write ``DIR/resolved.json`` (READ-ONLY REMOTE);
- ``diff A B``: every difference of two receipts (OFFLINE);
- ``check ARM CONTROL --declare FILE``: fail unless the arm differs from its control in exactly the declared
  settings (OFFLINE);
- ``method RESULT_JSON [--expect NAME]``: the measurement method a benchmark result records (OFFLINE);
- ``preflight --node-a TARGET --positions P,Q``: refuse while the deployment serves requests or another
  owner reserves the Sparks (READ-ONLY REMOTE);
- ``window --node-a TARGET --positions P,Q --since T0 --until T1 --client ADDRESS``: record the request
  sources on rank 0's access log in the window and fail when another client sent requests (READ-ONLY REMOTE;
  ``--log FILE`` reads a saved ``docker logs --timestamps`` text instead).

Exit status: 0 when the command's verdict holds, 1 when it does not, 2 on a usage or read error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from performance.harnesses.serving_ab.receipt import collect, compare, method, window  # noqa: E402


def positions_of(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        if "-" in part:
            first, last = part.split("-", 1)
            out.extend(range(int(first), int(last) + 1))
        elif part.strip():
            out.append(int(part))
    return out


def pairs(items: list[str]) -> dict[int, str]:
    return {int(k): v for k, v in (item.split("=", 1) for item in items)}


def write_json(path: str | None, document) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(document, indent=1, default=str) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m performance.harnesses.serving_ab.receipt",
                                     description="Run receipts: what a measured arm served, and whether it compares.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("collect", help="write DIR/resolved.json for the deployment on --positions (read-only)")
    p.add_argument("--node-a", required=True, help="SSH target of Node A, e.g. user@host")
    p.add_argument("--positions", required=True, help="fabric positions in rank order, e.g. 6,7 or 0-3")
    p.add_argument("--output", required=True)
    p.add_argument("--spark", action="append", default=[], metavar="POSITION=HOST",
                   help="SSH target of a position as Node A reaches it (default: from sparkring status --json)")
    p.add_argument("--container", action="append", default=[], metavar="POSITION=NAME",
                   help="container of a position (default: from the status, else the one ending -r<rank>)")
    p.add_argument("--api", help="the deployment's API URL (default: from the status)")

    p = sub.add_parser("diff", help="every difference of two receipts")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--ignore", action="append", default=[], metavar="PATTERN", help="setting keys to leave out (fnmatch)")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("check", help="an arm against its control and its declared change")
    p.add_argument("arm", help="the arm's resolved.json or its directory")
    p.add_argument("control", help="the control's resolved.json or its directory")
    p.add_argument("--declare", required=True, help="serving-ab-arm/v1 declaration (JSON)")
    p.add_argument("--ignore", action="append", default=[], metavar="PATTERN")
    p.add_argument("--output", help="write the verdict as JSON (a report reads check.json in a start's directory)")

    p = sub.add_parser("method", help="the measurement method a benchmark result records")
    p.add_argument("result", help="llm-inference-bench result JSON")
    p.add_argument("--expect", choices=tuple(method.METHODS), help="fail unless the result was measured by this method")

    p = sub.add_parser("preflight", help="refuse while the deployment serves or another owner reserves the Sparks")
    p.add_argument("--node-a", required=True)
    p.add_argument("--positions", required=True)
    p.add_argument("--api", help="the deployment's API URL (default: from sparkring status --json)")
    p.add_argument("--owner", help="the campaign's own reservation name; its reservations do not block")
    p.add_argument("--reservations", default="hwlock,setlock",
                   help="reservation sources: hwlock, setlock, command:CMD (default hwlock,setlock)")
    p.add_argument("--samples", type=int, default=5, help="/metrics readings, all of which must be idle (default 5)")
    p.add_argument("--interval", type=float, default=2.0, help="seconds between readings (default 2)")
    p.add_argument("--output")

    p = sub.add_parser("window", help="request sources on rank 0's access log in a measurement window")
    p.add_argument("--since", required=True, help="window start, ISO-8601 (UTC unless an offset is given)")
    p.add_argument("--until", help="window end (default: now)")
    p.add_argument("--client", action="append", required=True, help="address of a declared client (repeatable)")
    p.add_argument("--strict", action="store_true", help="also fail on monitoring reads by other sources")
    p.add_argument("--max-concurrency", type=int,
                   help="the benchmark's largest stream count: fail when the engine held more requests in the window")
    p.add_argument("--log", help="a saved docker logs --timestamps text instead of reading the ring")
    p.add_argument("--node-a")
    p.add_argument("--positions")
    p.add_argument("--container", help="rank 0's container (default: from sparkring status --json)")
    p.add_argument("--output")

    args = parser.parse_args(argv)
    try:
        return run(args)
    except (collect.CollectError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _rank0(node_a: collect.NodeA, positions: list[int], container: str | None = None) -> tuple[str, str, dict]:
    status = json.loads(node_a.run("sudo -n sparkring status --json"))
    deployment = collect.slot_for(status, positions).get("deployment") or {}
    first = next((c for c in deployment.get("containers") or [] if c["rank"] == 0), None)
    if first is None:
        raise ValueError(f"the status lists no rank-0 container on positions {positions}")
    return first["host"], container or first["name"], deployment


def run(args) -> int:
    if args.command == "collect":
        node_a = collect.NodeA(args.node_a)
        document, texts = collect.collect(node_a, positions_of(args.positions), sparks=pairs(args.spark),
                                          containers=pairs(args.container), api=args.api)
        path = collect.write(document, texts, Path(args.output))
        settings = document["settings"]
        print(f"{path}: {document['deployment'].get('profile')} on {args.positions}, digest {document['digest']}")
        for key in ("target.kv_cache_dtype", "target.block_size", "speculative.kv_cache_dtype",
                    "scheduler.async_scheduling", "compilation.mode"):
            print(f"  {key} = {compare.show(settings.get(key, compare.MISSING))} ({document['sources'].get(key, '-')})")
        for mismatch in document["rank_mismatches"]:
            print(f"  WARN rank {mismatch['rank']} {mismatch['key']} = {compare.show(mismatch['value'])}, "
                  f"rank 0 {compare.show(mismatch['rank0'])}")
        return 0
    if args.command == "diff":
        a, b = compare.load(args.a), compare.load(args.b)
        found = compare.diff(a, b, ignore=args.ignore)
        print(json.dumps({k: {key: [compare.show(x), compare.show(y)] for key, (x, y) in v.items()}
                          for k, v in found.items()}, indent=1) if args.json
              else compare.format_diff(found, labels=(args.a, args.b)))
        return 1 if found["settings"] else 0
    if args.command == "check":
        result = compare.check(compare.load(args.arm), compare.load(args.control),
                               compare.load_declaration(args.declare), ignore=args.ignore)
        write_json(args.output, result)
        print(compare.format_check(result))
        return 0 if result["passed"] else 1
    if args.command == "method":
        text, ok = method.format_check(args.result, method.load(args.result), args.expect)
        print(text)
        return 0 if ok else 1
    if args.command == "preflight":
        positions = positions_of(args.positions)
        node_a = collect.NodeA(args.node_a)
        api = args.api
        if not api:
            api = _rank0(node_a, positions)[2].get("api_url")
        if not api:
            raise ValueError("no API URL: give --api")
        result = window.preflight(
            positions=positions, read_metrics=lambda: collect.http_get(collect.api_base(api) + "/metrics", node_a),
            run_on_node_a=lambda command: node_a.run(command, check=False),
            sources=window.sources_from([s for s in args.reservations.split(",") if s]), owner=args.owner,
            samples=args.samples, interval=args.interval)
        write_json(args.output, result)
        print(window.format_preflight(result))
        return 0 if result["clear"] else 1
    if args.command == "window":
        until = args.until or window.dt.datetime.now(window.dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        if args.log:
            text = Path(args.log).read_text(encoding="utf-8", errors="replace")
            source = args.log
        else:
            if not (args.node_a and args.positions):
                raise ValueError("give --log, or --node-a and --positions")
            node_a = collect.NodeA(args.node_a)
            host, container, _ = _rank0(node_a, positions_of(args.positions), args.container)
            text = node_a.spark(host, window.access_log_script(container, args.since, until))
            source = f"{host} {container}"
        record = window.summarize(window.parse_access_log(text), clients=args.client, since=args.since, until=until,
                                  strict=args.strict, engine_stats=window.parse_engine_stats(text),
                                  max_concurrency=args.max_concurrency)
        record["log"] = source
        write_json(args.output, record)
        print(window.format_window(record))
        return 0 if record["exclusive"] else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
