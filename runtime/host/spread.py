"""The self-spreading install: Node A places the serving image and the checkpoint on every Spark along the cables.

Status: implemented and tested offline (``runtime/host/test_spread.py``);
not yet qualified on Spark hardware.

Three parts, all derived from the fabric document (``sparkring-fabric/v1``,
``runtime/common/fabric_document.py``):

- **Bootstrap order** (``bootstrap_order``). Node A reaches each Spark by the
  fewest cables, both ways round a cycle and the only way along a path, with
  the Spark before it as its SSH jump host; no route is longer than
  ``size - 1`` cables (``hop_limit``). Workers receive SparkRing's package in
  this order.
- **Pipelines** (``chains``). An asset (the serving image's registry blobs or
  the checkpoint's files) flows from its source Spark along each direction:
  ``next`` leaves the source through its port 0, ``previous`` through its port
  1. On a cycle each Spark is reached the shorter way, ties going ``next``; on
  a path, or a cycle with a down cable, the way that remains. Each chain
  starts at the last Spark before the first Spark that lacks something and
  that holds the whole asset, and every Spark on it forwards each chunk to the
  next while it writes (``runtime/host/fabric_stream.py``, ``hop``), so the
  farthest Spark finishes about one chunk time per extra cable after the
  first. A Spark that needs nothing but holds no source copy (an image already
  in Docker) forwards without writing.
- **Execution** (``Spread``). Node A starts one hop program per Spark of a
  chain and the source program on its first Spark, collects every placed file
  and repeats the planning with what each Spark now holds: a file that
  arrived with another SHA-256 is sent again from the Spark before, which
  verified its copy; a file a slow writer deferred is sent again; Sparks after
  a Spark that stopped answering are reached from the other direction of a
  cycle. A Spark that stopped answering ends the install with ``needs_input``
  (``stopped_message``); placed files stay, so the same command resumes. A
  path with a down cable sends the Sparks beyond it over the administration
  network (``cable_message``).

The spread plan, ``sparkring-spread-plan/v1``, is printed with the install
plan and saved with the checkpoint plan, whose ``envelope`` bounds a later
``--yes`` (``envelope`` here): ``fabric_id``, ``layout``, ``hop_limit``,
``bootstrap_order`` (``position``, ``via``, ``hops``), ``pipelines``
(``asset``, ``source_position``, ``direction``, ``hops`` of ``from``, ``to``,
``cable`` and ``functions``, ``bytes_written_per_position``,
``forwarded_only_positions``), ``fallbacks`` (``asset``, ``position``,
``path``: ``admin``, ``reason``: ``cable`` or ``layers``, ``cable``,
``bytes``) and ``sha256`` of the rest.
"""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import queue
import secrets
import threading

from runtime.common import fabric_document, fabric_layout
from runtime.host import bootstrap, fabric_stream
from runtime.host.install_errors import NeedsInput

SCHEMA = "sparkring-spread-plan/v1"
NEXT, PREVIOUS = "next", "previous"
DIRECTIONS = (NEXT, PREVIOUS)
GIB = 1024 ** 3
# Growth of one Spark's writes that a reviewed plan still covers, as for the checkpoint plan.
TOLERANCE_BYTES = GIB
# The fabric document's health state of a cable that carries no traffic.
DOWN_STATE = "failed"
# Each Spark's private directory of the serving image's verified registry blobs, one per image ID.
IMAGE_ROOT = "/var/lib/sparkring/spread/images"
# Planning rounds of one asset: the first pass, then repairs after mismatches, deferrals and lost Sparks.
ROUNDS = 6
ANNOUNCE_SECONDS = 180
FINISH_SECONDS = 3600
NOUNS = {"image": ("the image", "image layers"), "checkpoint": ("the checkpoint", "files"),
         "check": ("the test files", "test files")}


# Geometry

def hop_limit(layout):
    """The most cables between Node A and any Spark: ``size - 1``, the far end of a path (``bootstrap.hop_limit``)."""
    return bootstrap.hop_limit(layout["size"])


def bootstrap_order(document):
    """The order in which Node A reaches the other Sparks: ``[{"position", "via", "hops"}]``.

    Breadth first from position 0 along the cables, port 0's neighbour first,
    as ``fabric_ssh.routes`` reaches workers: on a cycle both directions
    alternate and the far Spark of an even cycle is reached through port 0's
    side. ``via`` is the Spark before it, its SSH jump host. A route longer
    than ``hop_limit`` is refused.
    """
    layout = fabric_document.layout(document)
    reached, rows, frontier = {0}, [], [(0, 0)]
    while frontier:
        following = []
        for here, hops in frontier:
            for there in fabric_layout.neighbors(layout, here):
                if there not in reached:
                    reached.add(there)
                    rows.append({"position": there, "via": here, "hops": hops + 1})
                    following.append((there, hops + 1))
        frontier = following
    limit = hop_limit(layout)
    if any(row["hops"] > limit for row in rows) or len(reached) != layout["size"]:
        raise ValueError(f"A {fabric_layout.name(layout)} must reach every Spark in at most {limit} cables")
    return rows


def down_cables(document):
    """The cables whose recorded health is ``failed``; the spread does not use them."""
    return {cable["cable"] for cable in document["cables"] if (cable.get("health") or {}).get("state") == DOWN_STATE}


def walk(layout, source, direction, *, members=None, down=(), dead=()):
    """``[(position, cable into it)]`` from ``source`` leaving through port 0 (``next``) or port 1 (``previous``).

    The walk stops before a down cable, a Spark outside ``members``, a Spark in
    ``dead``, at the end of a path, or when it comes back to ``source``.
    """
    members = set(range(layout["size"])) if members is None else set(members)
    found, here, port = [], source, 0 if direction == NEXT else 1
    while True:
        far = fabric_layout.peer(layout, here, port)
        if far is None:
            break
        cable, there = fabric_layout.cable_of(layout, here, port), far[0]
        if cable in down or there == source or there not in members or there in dead:
            break
        found.append((there, cable))
        here, port = there, 1 - far[1]
    return found


def chains(layout, source, needs, *, members=None, holders=(), relays=None, down=(), dead=()):
    """``(pipelines, unreached)``: the chains that bring each position of ``needs`` what it lacks from ``source``.

    ``needs`` maps a position to what it lacks (anything true: names or
    bytes). ``holders`` hold the whole asset and may start a chain; ``source``
    is one. ``relays`` may forward without writing (default: every member).
    Each pipeline is ``{"direction", "start", "hops", "cables"}``: the
    positions after ``start`` in order and the cable into each. ``unreached``
    lists the positions with needs that no chain reaches.
    """
    members = set(range(layout["size"])) if members is None else set(members)
    if source not in members or source in dead:
        raise ValueError(f"The spread's source, position {source}, is not an answering member of the fabric")
    relays = members if relays is None else set(relays)
    holders = (set(holders) | {source}) - set(dead)
    wanting = {position for position, value in needs.items() if value and position != source}
    walks = {direction: walk(layout, source, direction, members=members, down=down, dead=dead)
             for direction in DIRECTIONS}
    nearest = {}
    for direction in DIRECTIONS:
        for index, (position, _) in enumerate(walks[direction]):
            if position not in nearest or index < nearest[position][0]:
                nearest[position] = (index, direction)
    found = []
    for direction in DIRECTIONS:
        start, pending, current = source, [], None
        for position, cable in walks[direction]:
            if nearest[position][1] != direction:
                break
            if position in wanting:
                if current is None:
                    current = {"direction": direction, "start": start, "hops": [], "cables": []}
                for hop_position, hop_cable in [*pending, (position, cable)]:
                    current["hops"].append(hop_position)
                    current["cables"].append(hop_cable)
                pending = []
            elif position in holders:
                if current is not None:
                    found.append(current)
                    current = None
                start, pending = position, []
            elif position in relays:
                pending.append((position, cable))
            else:
                break
        if current is not None:
            found.append(current)
    reached = {position for chain in found for position in chain["hops"]}
    return found, sorted(wanting - reached)


def blocking(layout, source, position, *, members=None, down=(), dead=()):
    """What keeps ``position`` from ``source``: ``("cable", n)``, ``("spark", p)`` or None.

    The block reported is the first down cable or stopped Spark on the shorter
    way through ``members``; None when that way is open or no way exists.
    """
    ways = []
    for direction in DIRECTIONS:
        steps = walk(layout, source, direction, members=members)
        for index, (there, _) in enumerate(steps):
            if there == position:
                ways.append(steps[:index + 1])
                break
    for steps in sorted(ways, key=len)[:1]:
        for there, cable in steps:
            if cable in down:
                return ("cable", cable)
            if there in dead and there != position:
                return ("spark", there)
    return None


# The plan document

def _digest(value):
    core = {key: item for key, item in value.items() if key != "sha256"}
    return hashlib.sha256(fabric_document.encoded(core).encode()).hexdigest()


def plan(document, assets):
    """``sparkring-spread-plan/v1`` of ``assets`` on the fabric of ``document``.

    Each asset is ``{"asset", "source", "writes", "members", "holders",
    "relays", "admin"}``: ``writes`` maps each position that lacks the asset to
    the bytes it writes, ``admin`` lists ``{"position", "reason", "bytes"}``
    that receive over the administration network by choice (``layers``: a
    Spark whose Docker holds leading layers of the image loads only the rest
    through Node A's registry relay). Cables recorded as ``failed`` carry
    nothing; a position they cut off receives over the administration network.
    """
    layout = fabric_document.layout(document)
    down = down_cables(document)
    pipelines, fallbacks = [], []
    for asset in assets:
        writes = {int(position): int(size or 0) for position, size in asset["writes"].items()}
        found, unreached = chains(layout, asset["source"], {p: True for p in writes}, members=asset.get("members"),
                                  holders=asset.get("holders") or (), relays=asset.get("relays"), down=down)
        for chain in found:
            path = [chain["start"], *chain["hops"]]
            pipelines.append({
                "asset": asset["asset"], "source_position": chain["start"], "direction": chain["direction"],
                "hops": [{"from": a, "to": b, "cable": cable, "functions": list(fabric_layout.FUNCTIONS)}
                         for a, b, cable in zip(path, path[1:], chain["cables"])],
                "bytes_written_per_position": {str(p): writes[p] for p in chain["hops"] if p in writes},
                "forwarded_only_positions": [p for p in chain["hops"] if p not in writes]})
        for position in unreached:
            block = blocking(layout, asset["source"], position, members=asset.get("members"), down=down)
            fallbacks.append({"asset": asset["asset"], "position": position, "path": "admin", "reason": "cable",
                              "cable": block[1] if block and block[0] == "cable" else None,
                              "bytes": writes[position]})
        for row in asset.get("admin") or ():
            fallbacks.append({"asset": asset["asset"], "position": row["position"], "path": "admin",
                              "reason": row["reason"], "cable": None, "bytes": row.get("bytes")})
    value = {"schema": SCHEMA, "fabric_id": document["id"], "layout": fabric_layout.name(layout),
             "hop_limit": hop_limit(layout), "bootstrap_order": bootstrap_order(document),
             "pipelines": pipelines, "fallbacks": fallbacks}
    value["sha256"] = _digest(value)
    return value


def writes(value):
    """Bytes each position writes over the fabric and the administration path, all assets together."""
    total = {}
    for pipeline in value["pipelines"]:
        for position, size in pipeline["bytes_written_per_position"].items():
            total[int(position)] = total.get(int(position), 0) + size
    for row in value["fallbacks"]:
        total[row["position"]] = total.get(row["position"], 0) + (row.get("bytes") or 0)
    return total


def assets(document, checkpoint, *, positions, card):
    """The spread assets of ``sparkring install``: the image to every Spark, the checkpoint to the deployment's.

    ``checkpoint`` is the checkpoint plan (``sparkring-checkpoint-plan/v1``)
    and ``positions`` the fabric position of each of its ranks; ``card`` is
    the selection, whose ``download_bytes`` sizes the image's registry blobs.
    A Spark the plan did not survey is planned as lacking the image, the most
    it can write. The checkpoint's source is the plan's donor; Sparks whose
    plan holds every file locally may start a chain, and Sparks that serve a
    named copy in place hold it but cannot forward.
    """
    size = document["size"]
    nodes = checkpoint["nodes"]
    by_position = {positions[rank]: node for rank, node in enumerate(nodes)}
    blob_bytes = card.get("download_bytes") if isinstance(card.get("download_bytes"), int) else 0
    image_writes, admin = {}, []
    for position in range(1, size):
        node = by_position.get(position)
        basis = (((node or {}).get("storage") or {}).get("image") or {}).get("basis")
        if basis == "present":
            continue
        if basis == "layers":
            admin.append({"position": position, "reason": "layers",
                          "bytes": node["storage"]["image"].get("download_bytes")})
            continue
        image_writes[position] = blob_bytes
    received = {}
    for rank, node in enumerate(nodes):
        total = sum(entry["size"] for entry in node["files"].values() if entry["action"] == "receive")
        if total:
            received[positions[rank]] = total
    holders = [positions[rank] for rank, node in enumerate(nodes)
               if not any(entry["action"] in ("receive", "pool", "hub") for entry in node["files"].values())]
    return [{"asset": "image", "source": 0, "writes": image_writes, "admin": admin},
            {"asset": "checkpoint", "source": positions[checkpoint["distribution"]["donor"]], "writes": received,
             "members": list(positions), "holders": holders,
             "relays": [positions[rank] for rank, node in enumerate(nodes) if node["mode"] != "in-place"]}]


def recorded(state, cluster):
    """Node A's fabric document when it describes the cluster's Sparks in rank order, else None.

    A cluster set up before the fabric document existed, or whose document
    names other Sparks, spreads as before: checkpoints level by level and the
    image through the registry relay.
    """
    path = Path(state) / "fabric.json"
    try:
        document = fabric_document.validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None
    hosts = cluster["plan"]["spec"]["hosts"]
    if [row["node_id"] for row in document["positions"]] != [host.get("node_id") for host in hosts]:
        return None
    return document


def deployment_layout(layout, positions):
    """The layout of a deployment's Sparks: the fabric's when it uses all of them, else a pair or a path along the cables."""
    if len(positions) == layout["size"]:
        return layout
    if len(positions) == 2:
        return fabric_layout.layout(fabric_layout.PAIR, 2)
    return fabric_layout.layout(fabric_layout.PATH, len(positions))


# Text

def ranges(positions):
    """``0-3 and 5-7`` for positions 0, 1, 2, 3, 5, 6, 7."""
    runs = []
    for position in sorted(positions):
        if runs and position == runs[-1][1] + 1:
            runs[-1][1] = position
        else:
            runs.append([position, position])
    parts = [str(a) if a == b else f"{a}-{b}" for a, b in runs]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def positions_text(positions):
    positions = sorted(positions)
    return ("position " if len(positions) == 1 else "positions ") + ranges(positions)


def hostname(document, position):
    """The hostname the fabric document records for ``position``."""
    return document["positions"][position].get("hostname") or f"position {position}"


def _human(value):
    if value >= GIB:
        return f"{value / GIB:.1f} GiB"
    return f"{value / 1024 ** 2:.1f} MiB" if value >= 1024 ** 2 else f"{value} bytes"


def cable_text(document, cable):
    """``cable 3 (spark-d port 0 ↔ spark-e port 1)``."""
    ends = document["cables"][cable]["ends"]
    return (f"cable {cable} (" + " ↔ ".join(f"{hostname(document, end['position'])} port {end['port']}"
                                           for end in ends) + ")")


def describe(value, document):
    """The plan's lines for the install plan."""
    lines = [f"Spread over the fabric ({value['layout']}, fabric {value['fabric_id'][:19]}):"]
    order = ", ".join(f"{row['position']} via {row['via']}" for row in value["bootstrap_order"])
    reach = max((row["hops"] for row in value["bootstrap_order"]), default=0)
    limit = value["hop_limit"]
    lines.append(f"  Node A reaches the Sparks in cable order ({order}); the farthest is {reach} of at most "
                 f"{limit} {'cable' if limit == 1 else 'cables'} away.")
    for asset in NOUNS:
        rows = [pipeline for pipeline in value["pipelines"] if pipeline["asset"] == asset]
        if not rows:
            continue
        sources = sorted({pipeline["source_position"] for pipeline in rows})
        paths = " and ".join(" → ".join(str(p) for p in [pipeline["source_position"],
                                                           *[hop["to"] for hop in pipeline["hops"]]])
                             for pipeline in rows)
        written = {}
        for pipeline in rows:
            written.update({int(p): size for p, size in pipeline["bytes_written_per_position"].items()})
        sizes = set(written.values())
        if len(sizes) == 1:
            amount = next(iter(sizes))
            each = (f"{positions_text(written)} each write {_human(amount)}" if len(written) > 1 else
                    f"{positions_text(written)} writes {_human(amount)}") if amount else (
                f"{positions_text(written)} write{'s' if len(written) == 1 else ''} it")
        else:
            each = "; ".join(f"position {p} writes {_human(size)}" for p, size in sorted(written.items()))
        forwarding = sorted({p for pipeline in rows for p in pipeline["forwarded_only_positions"]})
        text = (f"  {NOUNS[asset][0].capitalize()} from {positions_text(sources)}, each Spark forwarding while it "
                f"writes: {paths}; {each}")
        if forwarding:
            text += f"; {positions_text(forwarding)} forward{'s' if len(forwarding) == 1 else ''} without writing"
        lines.append(text + ".")
    for asset in NOUNS:
        layers = [row["position"] for row in value["fallbacks"] if row["asset"] == asset and row["reason"] == "layers"]
        if layers:
            lines.append(f"  {positions_text(layers).capitalize()} already hold{'s' if len(layers) == 1 else ''} "
                         "leading image layers and load the rest through Node A's registry relay.")
        cut = {}
        for row in value["fallbacks"]:
            if row["asset"] == asset and row["reason"] == "cable":
                cut.setdefault(row["cable"], []).append(row["position"])
        for cable, positions in sorted(cut.items(), key=lambda item: (item[0] is None, item[0] or 0)):
            where = cable_text(document, cable) + " is down" if cable is not None else "No fabric path"
            lines.append(f"  {where}: {positions_text(positions)} receive{'s' if len(positions) == 1 else ''} "
                         f"{NOUNS[asset][0]} over the administration network instead of the fabric (slower).")
    return lines


def envelope(reviewed, fresh, tolerance=TOLERANCE_BYTES):
    """Differences by which the ``fresh`` spread plan leaves the ``reviewed`` one; empty when within it.

    The fresh plan stays within the reviewed one on the same fabric when no
    Spark writes more than ``tolerance`` beyond the reviewed plan and no Spark
    receives an asset over the administration network that the reviewed plan
    sent over the fabric.
    """
    if not reviewed or not fresh:
        return []
    if reviewed.get("fabric_id") != fresh.get("fabric_id"):
        return [{"kind": "spread", "rank": None, "text": "the fabric changed since the plan was reviewed "
                                                         "(sparkring fabric show prints it)"}]
    items = []
    before, after = writes(reviewed), writes(fresh)
    for position in sorted(after):
        if after[position] > before.get(position, 0) + tolerance:
            items.append({"kind": "spread", "rank": None, "position": position,
                          "text": f"position {position} would write {_human(after[position])} during the spread "
                                  f"instead of {_human(before.get(position, 0))}"})
    known = {(row["asset"], row["position"]) for row in reviewed["fallbacks"]}
    for row in fresh["fallbacks"]:
        if (row["asset"], row["position"]) not in known:
            items.append({"kind": "spread", "rank": None, "position": row["position"],
                          "text": f"position {row['position']} would receive {NOUNS[row['asset']][0]} over the "
                                  "administration network instead of the fabric"})
    return items


def stopped_message(document, asset, dead, remaining, members, total):
    """The ``needs_input`` text when Sparks stopped answering during an asset's spread."""
    unit = NOUNS[asset][1]
    dead = sorted(dead)
    names = " and ".join(f"{hostname(document, p)} (position {p})" for p in dead)
    complete = sorted(set(members) - set(dead) - set(remaining))
    waiting = sorted(set(remaining) - set(dead))
    held = ", ".join(f"position {p} holds {total - len(remaining.get(p, ()))} of {total} {unit}" for p in dead)
    text = f"{names} stopped answering during the {asset} spread; "
    if complete:
        text += f"{positions_text(complete)} hold{'s' if len(complete) == 1 else ''} the complete {asset}, "
    text += held
    if waiting:
        text += f", and {positions_text(waiting)} wait{'s' if len(waiting) == 1 else ''} for it"
    pronoun = "it" if len(dead) == 1 else "them"
    return text + f". Power {pronoun} on and repeat the same command; the spread resumes."


def cable_message(document, cable, positions):
    """The progress line when a down cable sends Sparks to the administration network."""
    return (f"{cable_text(document, cable)} is down; {positions_text(positions)} receive over the admin network "
            "instead of the fabric (slower). Reseat the cable and run sudo sparkring fabric verify.")


def unreached_message(document, position, done):
    """The ``needs_input`` text when Node A could not reach a Spark in bootstrap order."""
    via = next(row["via"] for row in bootstrap_order(document) if row["position"] == position)
    layout = fabric_document.layout(document)
    cable = next(fabric_layout.cable_of(layout, via, port) for port in (0, 1)
                 if (fabric_layout.peer(layout, via, port) or (None,))[0] == position)
    prefix = f"{positions_text(done)} {'is' if len(done) == 1 else 'are'} set up; " if done else ""
    return (f"{prefix}{hostname(document, position)} (position {position}) was not reached over cable {cable}; "
            "check the cable and repeat the same command.")


def first_unreached(document, alive):
    """``needs_input`` for the first Spark in bootstrap order for which ``alive(position)`` is false, or None."""
    done = [0]
    for row in bootstrap_order(document):
        if not alive(row["position"]):
            return NeedsInput(unreached_message(document, row["position"], done), field="spark",
                              details={"position": row["position"], "set_up": done})
        done.append(row["position"])
    return None


# Execution

class _Lines:
    """The JSON lines of one hop process: its announcement, the files it placed and its result."""

    def __init__(self, process):
        self.process, self.placed, self.result = process, [], None
        self.first = queue.Queue(1)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        announced = False
        try:
            for raw in self.process.stdout:
                try:
                    value = json.loads(raw)
                except ValueError:
                    continue
                if not announced:
                    announced = True
                    self.first.put(value)
                elif isinstance(value, dict) and "result" in value:
                    self.result = value["result"]
                elif isinstance(value, dict) and isinstance(value.get("placed"), str):
                    self.placed.append(value["placed"])
        except (OSError, ValueError):
            pass
        if not announced:
            self.first.put(None)

    def announcement(self, timeout):
        try:
            value = self.first.get(timeout=timeout)
        except queue.Empty:
            return None
        return value if isinstance(value, dict) and "ports" in value else None

    def finish(self, timeout):
        self.thread.join(timeout)
        try:
            self.process.wait(timeout=60)
        except Exception:  # noqa: BLE001 - a process that does not end is killed by the caller
            pass
        return self.result

    def error(self):
        errors = getattr(self.process, "errors", None)
        text = ""
        if errors is not None:
            try:
                errors.seek(0)
                data = errors.read()
                text = (data.decode(errors="replace") if isinstance(data, bytes) else data).strip()
            except (OSError, ValueError):
                text = ""
        lines = [line for line in text.splitlines() if line.strip()]
        status = getattr(self.process, "returncode", None)
        return (lines[-1][:300] if lines else "") or f"the hop program ended with status {status}"


class Spread:
    """Run one asset's pipelines from Node A, round after round, until every position that needs it holds it.

    ``start(position, listen, peers, files, want)`` starts the hop program on
    ``position`` (``fabric_stream.hop_source`` under ``fabric_stream.BOOT``)
    and returns its process, whose stdin already carries the program and
    stays open. ``send(position, sources, addresses, ports, groups, token)``
    runs the source program on a chain's first Spark and returns its result.
    ``alive(position)`` says whether a Spark answers over the administration
    network. ``say`` receives progress lines.
    """

    def __init__(self, document, *, start, send, alive, say=None, rounds=ROUNDS,
                 announce_seconds=ANNOUNCE_SECONDS, finish_seconds=FINISH_SECONDS):
        self.document, self.layout = document, fabric_document.layout(document)
        self.start, self.send, self.alive = start, send, alive
        self.say = say or (lambda text: None)
        self.rounds, self.announce_seconds, self.finish_seconds = rounds, announce_seconds, finish_seconds

    def run(self, asset, files, source, needs, *, members=None, holders=(), relays=None):
        """Spread ``files`` (``{name: [size, sha256]}``) from ``source`` to each position of ``needs``.

        ``needs`` maps a position to the names it lacks and ``holders`` lists
        positions that hold every file and may start a chain (``source`` is
        one); a position that receives everything becomes one. Returns ``{"remaining",
        "dead", "down", "failures", "received", "rounds", "fallbacks"}``:
        ``remaining`` maps each position that still lacks names to them,
        ``dead`` the Sparks that stopped answering, ``down`` the cables found
        or recorded down, ``failures`` a hop program's error per position,
        ``received`` one row per hop that placed files, ``rounds`` the passes
        that ran, and ``fallbacks`` the
        remaining positions that a down cable keeps from the fabric (to be
        sent over the administration network), by cable.
        """
        names = set(files)
        needs = {position: set(value) & names for position, value in needs.items() if set(value) & names}
        holders = set(holders) | {source}
        dead, down, failures, received = set(), set(down_cables(self.document)), {}, []
        rounds = 0
        for number in range(1, self.rounds + 1):
            pipelines, _ = chains(self.layout, source, needs, members=members, holders=holders, relays=relays,
                                  down=down, dead=dead)
            if not pipelines:
                break
            rounds = number
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(pipelines)) as pool:
                outcomes = list(pool.map(lambda chain: self.chain(chain, files, needs), pipelines))
            # A file that arrived with another SHA-256, that a slow writer deferred or that a lost
            # stream left incomplete is sent again in the next round, up to ``rounds`` rounds.
            changed = False
            for pipeline, outcome in zip(pipelines, outcomes):
                path = [pipeline["start"], *pipeline["hops"]]
                for index, position in enumerate(pipeline["hops"]):
                    hop = outcome["hops"].get(position)
                    if hop is None:
                        continue
                    gained = (set(hop.get("placed") or ()) | set(hop.get("held") or ())) & names
                    before = len(needs.get(position, ()))
                    if position in needs:
                        needs[position] -= gained
                        if not needs[position]:
                            del needs[position]
                    if position not in needs and gained >= names:
                        holders.add(position)
                    changed |= len(needs.get(position, ())) < before
                    changed |= bool(hop.get("mismatched") or hop.get("deferred") or hop.get("incomplete"))
                    if hop.get("placed"):
                        received.append({"asset": asset, "round": rounds, "direction": pipeline["direction"],
                                         "start": pipeline["start"], "source": path[index], "target": position,
                                         "hop": index + 1, "names": sorted(hop["placed"]),
                                         "bytes": sum(files[name][0] for name in hop["placed"]),
                                         "completed_at": hop.get("completed_at")})
                for position in outcome["lost"]:
                    if self.alive(position):
                        failures[position] = outcome["errors"].get(position) or "the hop program failed"
                    elif position not in dead:
                        dead.add(position)
                        changed = True
                        self.say(f"{hostname(self.document, position)} (position {position}) stopped answering "
                                 f"during the {asset} spread.")
                for cable in outcome["cables"]:
                    if cable not in down:
                        down.add(cable)
                        changed = True
            if not changed:
                break
        fallbacks = {}
        for position in sorted(needs):
            if position in dead:
                continue
            block = blocking(self.layout, source, position, members=members, down=down, dead=dead)
            if block is not None and block[0] == "cable":
                fallbacks.setdefault(block[1], []).append(position)
        for cable, positions in sorted(fallbacks.items()):
            self.say(cable_message(self.document, cable, positions))
        return {"remaining": needs, "dead": dead, "down": down, "failures": failures, "received": received,
                "rounds": rounds, "fallbacks": fallbacks}

    def chain(self, pipeline, files, needs):
        """Run one pipeline once; returns ``{"hops", "lost", "errors", "cables", "sent", "send_error"}``."""
        start, hops = pipeline["start"], pipeline["hops"]
        path = [start, *hops]
        links = [fabric_stream.cable_links(self.document, a, b) for a, b in zip(path, path[1:])]
        token = secrets.token_bytes(32)
        outcome = {"hops": {}, "lost": [], "errors": {}, "cables": [], "sent": None, "send_error": None}
        processes, lines = {}, {}
        try:
            for index, position in enumerate(hops):
                listen, peers = [theirs for _, theirs in links[index]], [mine for mine, _ in links[index]]
                process = self.start(position, listen, peers, files, sorted(needs.get(position, ())))
                processes[position] = process
                lines[position] = _Lines(process)
                try:
                    process.stdin.write(token)
                    process.stdin.flush()
                except OSError:
                    pass
            announced = {position: lines[position].announcement(self.announce_seconds) for position in hops}
            silent = [position for position in hops if announced[position] is None]
            if silent:
                for position in silent:
                    lines[position].finish(5)
                    outcome["lost"].append(position)
                    outcome["errors"][position] = lines[position].error()
                return outcome
            reach = {}
            for name in files:
                last = max((index + 1 for index, position in enumerate(hops)
                            if name in announced[position]["needed"]), default=0)
                if last:
                    reach[name] = last
            for index, position in enumerate(hops):
                line = None
                if index + 1 < len(hops):
                    pairs = links[index + 1]
                    line = {"sources": [mine for mine, _ in pairs], "addresses": [theirs for _, theirs in pairs],
                            "ports": announced[hops[index + 1]]["ports"]}
                try:
                    processes[position].stdin.write((json.dumps(line) + "\n").encode())
                    processes[position].stdin.close()
                except OSError:
                    pass
            sizes = {name: files[name][0] for name in reach}
            groups = [[[name, sizes[name], reach[name]] for name in group]
                      for group in fabric_stream.balance(sorted(reach), sizes, len(links[0]))]
            try:
                outcome["sent"] = self.send(start, [mine for mine, _ in links[0]], [theirs for _, theirs in links[0]],
                                            announced[hops[0]]["ports"], groups, token)
            except Exception as error:  # noqa: BLE001 - the hops' own results say where the chain broke
                outcome["send_error"] = str(error)[:300] or type(error).__name__
            for position in hops:
                result = lines[position].finish(self.finish_seconds)
                if result is None:
                    outcome["lost"].append(position)
                    outcome["errors"][position] = lines[position].error()
                    # Files the hop reported before it stopped are placed and stay.
                    outcome["hops"][position] = {"placed": list(lines[position].placed),
                                                 "held": announced[position]["held"]}
                else:
                    outcome["hops"][position] = result
            for index, position in enumerate(hops):
                result = outcome["hops"].get(position)
                if position in outcome["lost"] or result is None or result.get("connected"):
                    continue
                # This Spark answered but its previous Spark's streams never arrived: the cable into it is down
                # when the previous Spark (or the source) could not open them.
                before = path[index]
                previous = outcome["hops"].get(before) if index else None
                failed = (outcome["send_error"] is not None if index == 0 else
                          before not in outcome["lost"] and previous is not None and previous.get("forward_error"))
                if failed:
                    outcome["cables"].append(pipeline["cables"][index])
            return outcome
        finally:
            for process in processes.values():
                if process.poll() is None:
                    process.kill()
                    try:
                        process.wait(timeout=30)
                    except Exception:  # noqa: BLE001 - nothing more can be done for a process that ignores kill
                        pass


def raise_stopped(document, asset, outcome, members, total):
    """Raise ``needs_input`` when Sparks stopped answering; their placed files stay for the repeated command."""
    if not outcome["dead"]:
        return
    remaining = {position: names for position, names in outcome["remaining"].items()}
    for position in outcome["dead"]:
        remaining.setdefault(position, set())
    message = stopped_message(document, asset, outcome["dead"], remaining, members, total)
    raise NeedsInput(message, field="spark", details={
        "asset": asset, "stopped": sorted(outcome["dead"]),
        "remaining": {str(position): len(names) for position, names in sorted(remaining.items())}})


# The serving image's registry blobs on each Spark

def image_files(image_id, config, layers):
    """The pipeline files of an image: its configuration and each layer's registry blob, named by digest."""
    image = image_id.removeprefix("sha256:")
    files = {image: [len(config), image]}
    for _, digest, size in layers:
        files[digest.removeprefix("sha256:")] = [int(size), digest.removeprefix("sha256:")]
    return files


def image_root(image_id):
    """This image's directory of verified registry blobs on every Spark (``IMAGE_ROOT``)."""
    image = image_id.removeprefix("sha256:")
    if len(image) != 64 or any(character not in "0123456789abcdef" for character in image):
        raise ValueError("An image ID is sha256: and 64 lowercase hexadecimal digits")
    return IMAGE_ROOT + "/" + image


def load_staged_image(root, image_id, layers, tag, reserve, docker=None):
    """Import the image from its verified registry blobs in ``root`` with ``docker image load``.

    Runs on a Spark as root (``install_assets.Assets.remote``). Every blob is
    hashed again first; one whose SHA-256 differs from its digest, or that is
    missing, is removed and named in ``differs`` and nothing is imported, so
    Docker loads only a complete, verified archive. The archive holds the
    configuration, the manifest and every layer, as
    ``install_assets.write_layer_archive`` writes it, and the imported image
    must have ``image_id`` and be ARM64 Linux. ``docker`` replaces the Docker
    command, for tests.
    """
    import hashlib
    import io
    import json
    import os
    import shutil
    import subprocess
    import tarfile
    import tempfile
    docker = list(docker or ["docker", "--context", "default"])
    image = image_id.removeprefix("sha256:")
    names = [image] + [digest.removeprefix("sha256:") for _, digest, _ in layers]
    differs = []
    for name in dict.fromkeys(names):
        path = os.path.join(root, name)
        digest = hashlib.sha256()
        try:
            with open(path, "rb") as stream:
                while block := stream.read(16 << 20):
                    digest.update(block)
        except FileNotFoundError:
            differs.append(name)
            continue
        if digest.hexdigest() != name:
            os.unlink(path)
            differs.append(name)
    if differs:
        return {"loaded": False, "differs": differs}
    data_root = subprocess.check_output([*docker, "info", "--format", "{{.DockerRootDir}}"], text=True).strip()
    if shutil.disk_usage(data_root).free < reserve:
        raise ValueError("Image import space changed; existing model remains untouched")
    with open(os.path.join(root, image), "rb") as stream:
        config = stream.read()
    entries = [diff.removeprefix("sha256:") + "/layer.tar" for diff, _, _ in layers]
    manifest = json.dumps([{"Config": image + ".json", "RepoTags": [tag] if tag else None,
                            "Layers": entries}]).encode()
    with tempfile.TemporaryFile() as details:
        child = subprocess.Popen([*docker, "image", "load"], stdin=subprocess.PIPE, stdout=details, stderr=details)
        try:
            with tarfile.open(fileobj=child.stdin, mode="w|") as archive:
                def add(name, handle, size):
                    info = tarfile.TarInfo(name)
                    info.size, info.mode = size, 0o644
                    archive.addfile(info, handle)
                add(image + ".json", io.BytesIO(config), len(config))
                add("manifest.json", io.BytesIO(manifest), len(manifest))
                for entry, (_, digest, size) in zip(entries, layers):
                    with open(os.path.join(root, digest.removeprefix("sha256:")), "rb") as handle:
                        add(entry, handle, size)
            child.stdin.close()
        except BrokenPipeError:
            pass  # The loader stopped early; its own error is reported below.
        finally:
            if not child.stdin.closed:
                try:
                    child.stdin.close()
                except BrokenPipeError:
                    pass
        if child.wait():
            details.seek(0)
            raise ValueError("Image import failed: " + details.read()[-4000:].decode(errors="replace"))
    actual = json.loads(subprocess.check_output([*docker, "image", "inspect", image_id], text=True))[0]
    if actual["Id"] != image_id or actual["Architecture"] != "arm64" or actual["Os"] != "linux":
        raise ValueError("Imported image does not match the selected ARM64 identity")
    return {"loaded": True, "image_id": image_id, "free_bytes": shutil.disk_usage(data_root).free}


def discard_staged(root, base="/var/lib/sparkring/spread/images"):
    """Remove an image's directory of registry blobs below ``base`` once the image is in Docker.

    ``root`` must be ``base`` followed by one 64-digit image digest; returns
    whether it existed. ``base`` is ``IMAGE_ROOT`` on a Spark.
    """
    import os
    import re
    import shutil
    if os.path.dirname(root) != base or not re.fullmatch(r"[0-9a-f]{64}", os.path.basename(root)):
        raise ValueError("Only SparkRing's spread directories of image blobs are removed")
    if not os.path.lexists(root):
        return False
    shutil.rmtree(root)
    return True
