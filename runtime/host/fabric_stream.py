"""Copy pinned files between cabled Sparks over their fabric links: one cable, or a pipeline along many.

Administration traffic uses SSH, which reaches workers through the WireGuard
control network or, without one, SSH over the fabric; both carry one encrypted
stream per connection (0.62 GB/s through WireGuard on TP2). A checkpoint copy
instead uses plain TCP between the two ends of one fabric cable, one stream per
fabric function the cable's ports share (2.7-2.9 GB/s per stream on TP2).

The receiver binds only its own fabric addresses, accepts only the sender's
fabric address on each subnet, and requires a random token that the controller
delivers to both ends over their authenticated administration sessions. The
bytes are public model weights and image layers; the path provides neither
confidentiality nor integrity by itself, so the receiver checks every file
against the pinned SHA-256 that the controller sends it.

The receiver writes only into SparkRing's checkpoint directory ``D`` that the
rank operation ``model-transfer-prepare`` claimed, and only through the
placement protocol of ``runtime.host.checkpoint_place``: each needed file is
created as ``<name>.part`` in the marked staging directory ``receive/``
(``O_EXCL``), hashed while it is written, and hard-linked into ``D`` onto a name
that does not exist when its SHA-256 equals the pin. It never opens a name that
already exists in ``D``, so it cannot write through a hard link to a file
SparkRing did not create. The receiver runs on the target as the source text of
``checkpoint_place`` followed by ``receive`` and ``receive_checkpoint``
(``receiver_source``), under ``python3 -I``.

Pipelines. ``hop`` extends the receiver to a chain of Sparks along the cables
(``runtime/host/spread.py`` plans the chains). The chain's first Spark runs
``push``; every other Spark runs ``hop``, which accepts one stream per cable
function from the Spark before it and opens the same streams to the Spark after
it. Each stream carries files as a JSON header line ``[name, size, reach]``
followed by ``size`` bytes; ``reach`` counts the Sparks the file still passes,
this one included. A hop reads each file in fixed-size chunks (``CHUNK_BYTES``)
and sends every chunk to the next Spark before it reads the next one, so the
last Spark of a chain of ``h`` cables finishes about ``h - 1`` chunks after the
first. It hands the same chunk to its own writer through a queue of at most
``BUFFER_CHUNKS`` chunks per stream. The writer hashes and writes the file to a
part file and places it only when its SHA-256 equals the pin. A writer that
leaves the queue full for ``STALL_SECONDS`` loses that file (it is ``deferred``
and sent again in a later pass) while forwarding continues, so one slow disk
delays the chain by at most that long per file and holds at most the buffer in
memory. Once forwarding ends, a hop ends its stream to the next Spark first and
then lets its writer finish, so the Sparks after it never wait for its disk. A
Spark that already holds a file, or that the controller does not ask
to write it, forwards it without writing. A Spark whose next Spark stops
answering keeps receiving and writing; one whose previous Spark stops reports
the file it was writing as ``incomplete``. The two sinks are SparkRing's claimed
checkpoint directory (``CheckpointSink``) and a private directory of verified
files named by their digests (``DirectorySink``), which holds the registry blobs
of the serving image until ``docker load`` imports them.
"""
import hashlib
import hmac
import inspect
import ipaddress
import json
import os
import posixpath
import queue
import re
import socket
import stat
import sys
import threading
import time

from runtime.common import fabric_document, fabric_layout
from runtime.host import checkpoint_place
# The receiver's placement primitives. The shipped receiver defines the same
# names by running checkpoint_place's source ahead of ``receive``.
from runtime.host.checkpoint_place import (claim, create_part, discard_part, journal_load, listing,  # noqa: F401
                                           place_staged, safe_name, staging)


def links(hosts, source, target):
    """(source address, target address) for every fabric subnet both ranks share."""
    pairs = []
    for mine in hosts[source]["data_interfaces"]:
        for theirs in hosts[target]["data_interfaces"]:
            a, b = ipaddress.IPv4Interface(mine["address"]), ipaddress.IPv4Interface(theirs["address"])
            if a.network == b.network and a.ip != b.ip:
                pairs.append((str(a.ip), str(b.ip)))
    return pairs


def neighbours(document, position):
    """The Sparks one cable away from ``position`` in fabric document ``document``, port 0's first.

    Returns ``[{"position", "port", "cable", "links"}]``: ``port`` is this
    Spark's port, ``cable`` the cable's number and ``links`` the ``(this
    Spark's address, the neighbour's address)`` of each function the cable
    carries, primary first. A free port contributes nothing.
    """
    rows = {}
    for device in fabric_document.devices(document, position).values():
        far = device["neighbor"]
        if far is None:
            continue
        row = rows.setdefault(device["port"], {"position": far["position"], "port": device["port"],
                                               "cable": device["cable"], "links": []})
        row["links"].append((device["function"], device["address"], far["address"]))
    result = []
    for port in sorted(rows):
        row = rows[port]
        row["links"] = [(mine, theirs) for _, mine, theirs in
                        sorted(row["links"], key=lambda item: fabric_layout.FUNCTIONS.index(item[0]))]
        result.append(row)
    return result


def cable_links(document, a, b):
    """``[(a's address, b's address)]`` of each function of the cable between neighbours ``a`` and ``b``."""
    for row in neighbours(document, a):
        if row["position"] == b:
            return list(row["links"])
    raise ValueError(f"Positions {a} and {b} share no fabric cable")


def tree(count, donor, layout=None, *, line=False):
    """Cable-adjacent copy order from ``donor``: a list of levels of (source, target).

    ``layout`` (``fabric_layout.layout``) names the cables. ``line`` says
    that the ranks are consecutive Sparks of a line, whose ends share no
    cable, such as four Sparks of an eight-Spark ring. Without either the
    ranks form a pair or a ring, in which the last rank's cable also reaches
    rank 0: the layouts of records that name only their Spark count.
    """
    if line and layout is not None:
        raise ValueError("A copy tree follows a layout or a line, not both")
    if line:
        neighbors = {rank: [other for other in (rank + 1, rank - 1) if 0 <= other < count] for rank in range(count)}
    elif layout is None:
        neighbors = {rank: ([1 - rank] if count == 2 else [(rank + 1) % count, (rank - 1) % count])
                     for rank in range(count)}
    else:
        if layout["size"] != count:
            raise ValueError(f"A {fabric_layout.name(layout)} has {layout['size']} Sparks, not {count}")
        neighbors = {rank: fabric_layout.neighbors(layout, rank) for rank in range(count)}
    reached, levels, frontier = {donor}, [], [donor]
    while frontier:
        level = []
        for source in frontier:
            for target in neighbors[source]:
                if target not in reached:
                    reached.add(target)
                    level.append((source, target))
        if level:
            levels.append(level)
        frontier = [target for _, target in level]
    return levels


def balance(names, sizes, streams):
    """Assign files to streams, largest first, to the stream with the fewest bytes."""
    groups, loads = [[] for _ in range(streams)], [0] * streams
    for name in sorted(names, key=lambda n: (-sizes[n], n)):
        index = loads.index(min(loads))
        groups[index].append(name)
        loads[index] += sizes[name]
    return groups


def receive(addresses, peers, root, files, *, staging, journal, token=None, announce=None):
    """Receive the missing files of ``files`` into claimed checkpoint directory ``root``.

    Runs on the target as root while the caller holds the claim of ``root``
    (``journal.claim``). ``files`` maps each name to its pinned
    ``[size, sha256]``; ``staging`` is the descriptor of the marked staging
    directory ``receive/`` and ``journal`` the directory's placement journal.

    Names that already exist in ``root`` are not requested and never opened;
    ``model-transfer-prepare`` removed SparkRing's own names whose content
    differed, and ``model-transfer-complete`` verifies the rest. Each needed file
    is written to ``<name>.part`` in staging, created with ``O_EXCL`` after a
    stale part is unlinked, hashed while written and synced. A part whose
    SHA-256 equals the pin is linked into ``root`` by ``place_staged`` (journal
    first, never replacing a name) and its staging name removed; a mismatch
    removes the part and fails the transfer.

    The first stdout line announces the listening ports and the needed names;
    the sender then connects. Returns ``{"received", "bytes", "placed"}``.
    """
    import concurrent.futures
    import hashlib
    import hmac
    import json
    import os
    import socket
    import sys
    import threading

    claimed = journal.claim
    if root != claimed.path:
        raise ValueError("The checkpoint receiver holds the claim of another directory than " + str(root)[:200])
    claimed.check()
    for name, value in files.items():
        safe_name(name)
        if (not isinstance(value, (list, tuple)) or len(value) != 2 or type(value[0]) is not int or value[0] <= 0
                or not isinstance(value[1], str) or len(value[1]) != 64
                or any(character not in "0123456789abcdef" for character in value[1])):
            raise ValueError("Invalid pinned size or SHA-256 for checkpoint file " + name)
    token = token if token is not None else sys.stdin.buffer.read(32)
    announce = announce or (lambda value: print(json.dumps(value), flush=True))
    present = listing(claimed.dir_fd)[0]
    needed = sorted(name for name in files if name not in present)
    listeners = [socket.create_server((address, 0)) for address in addresses]
    for listener in listeners:
        listener.settimeout(120)
    announce({"ports": [listener.getsockname()[1] for listener in listeners], "needed": needed})
    expected, started, placed, guard = set(needed), set(), set(), threading.Lock()

    def store(stream, name, size):
        """Write one file from ``stream`` into staging while hashing it; place it when it matches its pin."""
        part = name + ".part"
        fd = create_part(staging, part)
        try:
            digest, remaining = hashlib.sha256(), size
            while remaining:
                block = stream.read(min(remaining, 16 << 20))
                if not block:
                    raise ValueError("Checkpoint stream ended inside " + name)
                digest.update(block)
                view = memoryview(block)
                while view:
                    view = view[os.write(fd, view):]
                remaining -= len(block)
            os.fsync(fd)
            if digest.hexdigest() != files[name][1]:
                raise ValueError(f"The fabric stream delivered different bytes for {name} than its pinned SHA-256; "
                                 "it was not placed")
            place_staged(claimed.dir_fd, staging, part, name, journal, fd=fd, sha256=digest.hexdigest(),
                         origin="fabric")
        except BaseException:
            discard_part(staging, part, fd)
            raise
        os.close(fd)

    def serve(index):
        with listeners[index] as listener:
            connection, (host, _) = listener.accept()
        # The reader holds its own reference to the socket, so both close
        # together; a rejected sender then sees the connection end at once.
        with connection, connection.makefile("rb", buffering=16 << 20) as stream:
            if host != peers[index]:
                raise ValueError("Checkpoint stream came from an unexpected fabric address")
            connection.settimeout(300)
            if not hmac.compare_digest(stream.read(32), token):
                raise ValueError("Checkpoint stream token differs")
            total = 0
            while line := stream.readline():
                name, size = json.loads(line)
                with guard:
                    if name not in expected or name in started or size != files[name][0]:
                        raise ValueError("Checkpoint stream sent an unplanned file: " + str(name)[:200])
                    started.add(name)
                store(stream, name, size)
                with guard:
                    placed.add(name)
                total += size
            return total

    if not needed:
        for listener in listeners:
            listener.close()
        return {"received": 0, "bytes": 0, "placed": []}
    with concurrent.futures.ThreadPoolExecutor(len(listeners)) as pool:
        total = sum(pool.map(serve, range(len(listeners))))
    if placed != expected:
        raise ValueError("Checkpoint stream omitted planned files")
    return {"received": len(placed), "bytes": total, "placed": sorted(placed)}


def receive_checkpoint(addresses, peers, root, repository, revision, files, token=None, announce=None):
    """Claim prepared checkpoint directory ``root`` and receive ``files`` into it with ``receive``.

    ``root`` must already be SparkRing's checkpoint directory of ``repository``
    at ``revision``, claimed by ``model-transfer-prepare``: the receiver never
    creates a checkpoint directory. It holds the directory's lock while it
    receives.
    """
    import os
    import posixpath
    import stat

    state = posixpath.join(posixpath.dirname(root), "." + posixpath.basename(root) + ".sparkring")
    try:
        prepared = (stat.S_ISDIR(os.lstat(root).st_mode)
                    and stat.S_ISREG(os.lstat(posixpath.join(state, "owner.json")).st_mode))
    except OSError:
        prepared = False
    if not prepared:
        raise ValueError(f"{root} was not prepared for receiving checkpoint files; SparkRing receives only into a "
                         "checkpoint directory it created")
    with claim(root, repository, revision) as claimed:
        if claimed.action != "verified":
            raise ValueError(f"{root} was not prepared for receiving checkpoint files")
        journal = journal_load(claimed)
        directory = staging(claimed, "receive")
        try:
            return receive(addresses, peers, root, files, staging=directory, journal=journal, token=token,
                           announce=announce)
        finally:
            os.close(directory)


def receiver_source(addresses, peers, root, repository, revision, files):
    """Python source of the target's receiver: ``checkpoint_place``, ``receive`` and ``receive_checkpoint``.

    The program sets the umask to 0022, so received files get mode 0644
    whatever the umask of the session that started it; it reads the 32-byte
    token from stdin, prints the announcement line, and prints the receiver's
    result as the last line.
    """
    call = "receive_checkpoint(*" + repr((addresses, peers, root, repository, revision, files)) + ")"
    return (inspect.getsource(checkpoint_place) + "\n\n" + inspect.getsource(receive) + "\n\n"
            + inspect.getsource(receive_checkpoint) + "\n\nos.umask(0o022)\nprint(json.dumps(" + call + "))\n")


def send(sources, addresses, ports, root, groups, token=None):
    """Run on the source. Stream ``groups[i]`` from ``sources[i]`` to ``addresses[i]:ports[i]``.

    Files are opened read-only with ``O_NOFOLLOW`` and ``O_NONBLOCK`` and, when
    this process may use it, ``O_NOATIME``, so streaming leaves their access
    times unchanged. Only regular files are sent.
    """
    import concurrent.futures
    import json
    import os
    import socket
    import stat
    import sys

    token = token if token is not None else sys.stdin.buffer.read(32)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)

    def opened(name):
        path = os.path.join(root, name)
        try:
            fd = os.open(path, flags | getattr(os, "O_NOATIME", 0))
        except PermissionError:
            # O_NOATIME needs the file's owner or CAP_FOWNER.
            fd = os.open(path, flags)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise ValueError("Checkpoint source is not a regular file: " + name)
        return os.fdopen(fd, "rb")

    def push(index):
        total = 0
        with socket.create_connection((addresses[index], ports[index]), timeout=60,
                                      source_address=(sources[index], 0)) as connection:
            connection.settimeout(300)
            connection.sendall(token)
            for name in groups[index]:
                with opened(name) as stream:
                    size = os.fstat(stream.fileno()).st_size
                    connection.sendall((json.dumps([name, size]) + "\n").encode())
                    if connection.sendfile(stream) != size:
                        raise ValueError("Checkpoint file changed while streaming: " + name)
                total += size
            connection.shutdown(socket.SHUT_WR)
            connection.recv(1)
        return total

    with concurrent.futures.ThreadPoolExecutor(len(groups)) as pool:
        return {"sent": sum(pool.map(push, range(len(groups))))}


# Pipelines along a chain of Sparks ---------------------------------------------

# A hop forwards each chunk before it reads the next: the last Spark of a chain
# of h cables finishes about h - 1 chunk times after the first Spark.
CHUNK_BYTES = 8 << 20
# Chunks a hop's writer may lag behind its forwarding, per stream.
BUFFER_CHUNKS = 16
# How long forwarding waits for a full writer queue before that file is deferred.
STALL_SECONDS = 60
# How long a hop waits for the previous Spark's streams, which open once every hop has announced.
ACCEPT_SECONDS = 180
CONNECT_SECONDS = 60
READ_SECONDS = 300
# The modules the shipped pipeline programs import; their functions use nothing else.
PROGRAM_IMPORTS = ("hashlib", "hmac", "json", "os", "posixpath", "queue", "re", "socket", "stat", "sys",
                   "threading", "time")
PROGRAM_CONSTANTS = ("CHUNK_BYTES", "BUFFER_CHUNKS", "STALL_SECONDS", "ACCEPT_SECONDS", "CONNECT_SECONDS",
                     "READ_SECONDS")
# The settings of ``hop`` and ``push`` that a caller may pass to the shipped programs.
HOP_OPTIONS = ("chunk", "buffer", "stall", "limit", "accept_seconds", "read_seconds")
PUSH_OPTIONS = ("chunk", "limit", "read_seconds")


class Throttle:
    """Pace one stream as a link of ``limit`` bytes per second; ``None`` sends at once.

    Each call takes ``count / limit`` seconds after the link is free, so a
    chunk reaches the next Spark one chunk time after it left this one. The
    installer does not pace fabric streams; tests pace them to model links of
    a known speed.
    """

    def __init__(self, limit):
        self.limit, self.free = limit, time.monotonic()

    def __call__(self, count):
        if not self.limit:
            return
        now = time.monotonic()
        self.free = max(now, self.free) + count / self.limit
        if self.free > now:
            time.sleep(self.free - now)


class DirectorySink:
    """Verified files in the private directory ``root``, which the first write creates (mode 0700).

    A file is written to ``<name>.part`` (created exclusively, mode 0600),
    synced and, once its SHA-256 equals the pin, linked to ``<name>``, which it
    never replaces, and its part removed. A name is held when it is a regular
    file of its pinned size; only a verified part becomes one. ``root`` must lie
    in a directory that only this account can change, such as
    ``/var/lib/sparkring``.

    Both sinks offer ``held()``, ``open(name, size)`` (a handle),
    ``write(handle, data)``, ``commit(handle, name, sha256)``, which places the
    file or removes its part before it raises, and ``discard(handle, name)``.
    """

    def __init__(self, root, files):
        self.root, self.files = root, files
        self.guard = threading.Lock()

    def path(self, name):
        return os.path.join(self.root, *safe_name(name).split("/"))

    def held(self):
        """The names held; a part left beside one by a commit that was interrupted after its link is removed."""
        found = set()
        for name, (size, _) in self.files.items():
            try:
                info = os.lstat(self.path(name))
            except OSError:
                continue
            if stat.S_ISREG(info.st_mode) and info.st_size == size:
                found.add(name)
                try:
                    os.unlink(self.path(name) + ".part")
                except FileNotFoundError:
                    pass
        return found

    def open(self, name, size):
        part = self.path(name) + ".part"
        with self.guard:
            os.makedirs(os.path.dirname(part), mode=0o700, exist_ok=True)
        try:
            os.unlink(part)
        except FileNotFoundError:
            pass
        flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
                 | getattr(os, "O_BINARY", 0))
        return os.open(part, flags, 0o600)

    def write(self, handle, data):
        view = memoryview(data)
        while view:
            view = view[os.write(handle, view):]

    def commit(self, handle, name, sha256):
        final = self.path(name)
        try:
            os.fsync(handle)
        finally:
            os.close(handle)
        try:
            os.link(final + ".part", final)
        finally:
            os.unlink(final + ".part")

    def discard(self, handle, name):
        if handle is not None:
            os.close(handle)
        try:
            os.unlink(self.path(name) + ".part")
        except FileNotFoundError:
            pass


class CheckpointSink:
    """SparkRing's claimed checkpoint directory: parts in staging ``receive/``, placed by ``place_staged``.

    ``claimed`` is the directory's claim, ``staging_fd`` its ``receive``
    staging directory and ``journal`` its placement journal; names already in
    the directory are held and never opened.
    """

    def __init__(self, claimed, staging_fd, journal, files):
        self.claimed, self.staging, self.journal, self.files = claimed, staging_fd, journal, files

    def held(self):
        return set(listing(self.claimed.dir_fd)[0]) & set(self.files)

    def open(self, name, size):
        return create_part(self.staging, name + ".part")

    def write(self, handle, data):
        view = memoryview(data)
        while view:
            view = view[os.write(handle, view):]

    def commit(self, handle, name, sha256):
        try:
            os.fsync(handle)
            place_staged(self.claimed.dir_fd, self.staging, name + ".part", name, self.journal, fd=handle,
                         sha256=sha256, origin="fabric")
        except BaseException:
            discard_part(self.staging, name + ".part", handle)
            raise
        os.close(handle)

    def discard(self, handle, name):
        discard_part(self.staging, name + ".part", handle)


def hop(listen, peers, files, sink, *, token, want=None, downstream=None, announce=None, progress=None,
        chunk=CHUNK_BYTES, buffer=BUFFER_CHUNKS, stall=STALL_SECONDS, limit=None, connect=None,
        accept_seconds=ACCEPT_SECONDS, read_seconds=READ_SECONDS):
    """One Spark of a pipeline: receive ``files`` from the Spark before it, write what it lacks, forward each chunk.

    ``listen`` lists this Spark's addresses on the cable to the previous
    Spark, one per stream, and ``peers`` that Spark's address on each.
    ``files`` maps every name the pass may carry to its pinned ``[size,
    sha256]``; ``want`` lists the names this Spark writes when ``sink`` does
    not hold them (default: all). ``announce`` receives ``{"ports", "needed",
    "held"}`` once the listeners are open. ``downstream`` (a value or a
    callable called after the announcement) is None for the chain's last
    Spark, else ``{"sources", "addresses", "ports"}``: this Spark's addresses
    on the cable to the next Spark and that Spark's addresses and announced
    ports, one per stream. ``progress`` receives ``{"placed": name, "at":
    time}`` for every placed file; ``connect`` opens the next Spark's streams
    (``socket.create_connection``); ``limit`` paces forwarding (``Throttle``).

    The token, the previous Spark's address and every header are checked; an
    unplanned name or size raises ValueError. A stream that ends or fails
    mid-file leaves that file ``incomplete``; a next Spark that stops answering
    ends forwarding (``forward_error``) while this Spark keeps writing. Returns
    ``{"placed", "placed_at", "completed_at", "held", "needed", "mismatched",
    "deferred", "incomplete", "failed", "missing", "written_bytes",
    "forwarded_bytes", "connected", "forward_error", "upstream_error"}``;
    ``missing`` lists needed names that never arrived.
    """
    connect = connect or socket.create_connection
    names = set(files)
    for name, value in files.items():
        safe_name(name)
        if (not isinstance(value, (list, tuple)) or len(value) != 2 or type(value[0]) is not int or value[0] < 0
                or not isinstance(value[1], str) or not re.fullmatch("[0-9a-f]{64}", value[1])):
            raise ValueError("Invalid pinned size or SHA-256 for " + name)
    if not listen or len(listen) != len(peers):
        raise ValueError("A pipeline hop needs one listening address and one peer address per stream")
    if type(chunk) is not int or chunk <= 0 or type(buffer) is not int or buffer <= 0:
        raise ValueError("A pipeline hop needs a positive chunk size and buffer")
    held = set(sink.held()) & names
    needed = (names if want is None else set(want) & names) - held
    listeners = [socket.create_server((address, 0)) for address in listen]
    try:
        for listener in listeners:
            listener.settimeout(accept_seconds)
        (announce or print)({"ports": [listener.getsockname()[1] for listener in listeners],
                             "needed": sorted(needed), "held": sorted(held)})
        target = downstream() if callable(downstream) else downstream
        if target is not None and not (isinstance(target, dict) and all(
                isinstance(target.get(key), list) and len(target[key]) == len(listen)
                for key in ("sources", "addresses", "ports"))):
            raise ValueError("The next Spark of a pipeline needs one source, address and port per stream")
    except BaseException:
        for listener in listeners:
            listener.close()
        raise
    guard = threading.Lock()
    outcome = {"placed": [], "placed_at": {}, "mismatched": [], "deferred": [], "incomplete": [], "failed": {},
               "written_bytes": 0, "forwarded_bytes": 0, "connected": [False] * len(listen),
               "forward_error": None, "upstream_error": None}
    started, errors = set(), []

    class Refused(Exception):
        """The previous Spark broke the pipeline protocol; the hop stops."""

    def record(key, name):
        with guard:
            outcome[key].append(name)

    def fail(name, error):
        with guard:
            outcome["failed"][name] = (str(error) or type(error).__name__)[:300]

    def first_error(key, error):
        with guard:
            outcome[key] = outcome[key] or (str(error) or type(error).__name__)[:300]

    def write_files(feed, dropped):
        current = None

        def drop(entry):
            try:
                sink.discard(entry[1], entry[0])
            except Exception:  # noqa: BLE001 - a part that cannot be removed is replaced by the next pass
                pass

        while True:
            kind, value = feed.get()
            if kind == "end":
                if current is not None:
                    drop(current)
                return
            if kind == "open":
                if current is not None:
                    drop(current)
                try:
                    current = [value, sink.open(value, files[value][0]), hashlib.sha256()]
                except Exception as error:  # noqa: BLE001 - reported per file; forwarding continues
                    fail(value, error)
                    current = None
                continue
            if current is None:
                continue
            name, handle, digest = current
            if kind == "abort" or name in dropped:
                drop(current)
                current = None
            elif kind == "data":
                try:
                    digest.update(value)
                    sink.write(handle, value)
                except Exception as error:  # noqa: BLE001 - reported per file; forwarding continues
                    fail(name, error)
                    drop(current)
                    current = None
                    continue
                with guard:
                    outcome["written_bytes"] += len(value)
            elif kind == "close":
                entry, current = current, None
                if digest.hexdigest() != files[name][1]:
                    drop(entry)
                    record("mismatched", name)
                    continue
                try:
                    # A failed commit has already removed its part.
                    sink.commit(handle, name, digest.hexdigest())
                except Exception as error:  # noqa: BLE001 - reported per file
                    fail(name, error)
                    continue
                at = time.time()
                with guard:
                    outcome["placed"].append(name)
                    outcome["placed_at"][name] = at
                if progress is not None:
                    progress({"placed": name, "at": at})

    def stream_files(index):
        feed, dropped, state = queue.Queue(buffer), set(), {"down": None}
        writer = threading.Thread(target=write_files, args=(feed, dropped), daemon=True)
        writer.start()
        throttle = Throttle(limit)

        def hand(item):
            # The writer may lag by ``buffer`` chunks; one that stays behind for
            # ``stall`` seconds loses the file instead of stalling the chain.
            try:
                feed.put(item, timeout=stall)
                return True
            except queue.Full:
                return False

        def lose_downstream(error):
            first_error("forward_error", error)
            down, state["down"] = state["down"], None
            if down is not None:
                down.close()

        def send(data):
            if state["down"] is None:
                return False
            try:
                throttle(len(data))
                state["down"].sendall(data)
                return True
            except OSError as error:
                lose_downstream(error)
                return False

        if target is not None:
            try:
                state["down"] = connect((target["addresses"][index], target["ports"][index]), CONNECT_SECONDS,
                                        (target["sources"][index], 0))
                state["down"].settimeout(read_seconds)
                state["down"].sendall(token)
            except OSError as error:
                lose_downstream(error)
        connection, source, current = None, None, None
        try:
            with listeners[index] as listener:
                connection, (host, _) = listener.accept()
            connection.settimeout(read_seconds)
            source = connection.makefile("rb", buffering=max(chunk, 1 << 20))
            if host != peers[index]:
                raise Refused("Pipeline stream came from an unexpected fabric address")
            if not hmac.compare_digest(source.read(32), token):
                raise Refused("Pipeline stream token differs")
            outcome["connected"][index] = True
            while True:
                line = source.readline(65536)
                if not line:
                    break
                if not line.endswith(b"\n"):
                    raise EOFError("The previous Spark's stream ended inside a file header")
                try:
                    name, size, reach = json.loads(line)
                except (ValueError, TypeError):
                    raise Refused("Pipeline stream sent an invalid file header") from None
                with guard:
                    if (not isinstance(name, str) or name not in names or name in started or size != files[name][0]
                            or type(reach) is not int or reach < 1):
                        raise Refused("Pipeline stream sent an unplanned file: " + str(name)[:200])
                    started.add(name)
                current = name
                forward = reach > 1 and send((json.dumps([name, size, reach - 1]) + "\n").encode())
                writing = name in needed
                if writing and not hand(("open", name)):
                    writing = False
                    record("deferred", name)
                remaining = size
                while remaining:
                    block = source.read(min(chunk, remaining))
                    if not block:
                        raise EOFError(f"The previous Spark's stream ended inside {name}")
                    remaining -= len(block)
                    if forward and send(block):
                        with guard:
                            outcome["forwarded_bytes"] += len(block)
                    if writing and not hand(("data", block)):
                        writing = False
                        dropped.add(name)
                        record("deferred", name)
                if writing and not hand(("close", name)):
                    dropped.add(name)
                    record("deferred", name)
                current = None
        except Refused as error:
            errors.append(ValueError(str(error)))
        except (OSError, EOFError) as error:
            first_error("upstream_error", error)
            if current is not None:
                if current in needed:
                    record("incomplete", current)
                hand(("abort", None))
        finally:
            down = state["down"]
            if down is not None:
                # Forwarding is over, so the Sparks after this one never wait for its writer; the next
                # Spark closes once it and the Sparks after it are done.
                try:
                    down.shutdown(socket.SHUT_WR)
                    while down.recv(65536):
                        pass
                except OSError as error:
                    first_error("forward_error", error)
                down.close()
            # The writer finishes what it holds within the read timeout; a file it does not finish is
            # left unplaced (``missing``) and its part is replaced by the next pass.
            try:
                feed.put(("end", None), timeout=read_seconds)
                writer.join(read_seconds)
            except queue.Full:
                pass
            # Closing both the reader and the socket ends the connection, which
            # tells the previous Spark that this one and the Sparks after it are done.
            if source is not None:
                source.close()
            if connection is not None:
                connection.close()

    threads = [threading.Thread(target=stream_files, args=(index,)) for index in range(len(listen))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    placed = set(outcome["placed"])
    accounted = (placed | set(outcome["mismatched"]) | set(outcome["deferred"]) | set(outcome["incomplete"])
                 | set(outcome["failed"]))
    return {**outcome, "placed": sorted(placed), "held": sorted(held), "needed": sorted(needed),
            "mismatched": sorted(set(outcome["mismatched"])), "deferred": sorted(set(outcome["deferred"])),
            "incomplete": sorted(set(outcome["incomplete"])), "missing": sorted(needed - accounted),
            "connected": all(outcome["connected"]),
            "completed_at": max(outcome["placed_at"].values(), default=None)}


def hop_checkpoint(root, repository, revision, listen, peers, files, **options):
    """``hop`` into the prepared checkpoint directory ``root``, holding its claim while the pass runs.

    As for ``receive_checkpoint``, ``root`` must be SparkRing's directory of
    ``repository`` at ``revision`` that ``model-transfer-prepare`` claimed.
    """
    state = posixpath.join(posixpath.dirname(root), "." + posixpath.basename(root) + ".sparkring")
    try:
        prepared = (stat.S_ISDIR(os.lstat(root).st_mode)
                    and stat.S_ISREG(os.lstat(posixpath.join(state, "owner.json")).st_mode))
    except OSError:
        prepared = False
    if not prepared:
        raise ValueError(f"{root} was not prepared for receiving checkpoint files; SparkRing receives only into a "
                         "checkpoint directory it created")
    with claim(root, repository, revision) as claimed:
        if claimed.action != "verified":
            raise ValueError(f"{root} was not prepared for receiving checkpoint files")
        journal = journal_load(claimed)
        directory = staging(claimed, "receive")
        try:
            return hop(listen, peers, files, CheckpointSink(claimed, directory, journal, files), **options)
        finally:
            os.close(directory)


def files_in(root):
    """An opener for ``push``: regular files below ``root``, read-only, without following a final link."""
    flags = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
             | getattr(os, "O_BINARY", 0))

    def opened(name):
        path = os.path.join(root, *safe_name(name).split("/"))
        try:
            fd = os.open(path, flags | getattr(os, "O_NOATIME", 0))
        except PermissionError:
            # O_NOATIME needs the file's owner or CAP_FOWNER.
            fd = os.open(path, flags)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise ValueError("Pipeline source is not a regular file: " + name)
        return os.fdopen(fd, "rb")
    return opened


def push(sources, addresses, ports, groups, opener, *, token, chunk=CHUNK_BYTES, limit=None, connect=None,
         read_seconds=READ_SECONDS):
    """The chain's first Spark: send ``groups[i]`` on stream ``i`` to ``addresses[i]:ports[i]`` from ``sources[i]``.

    Each group entry is ``[name, size, reach]`` and ``opener(name)`` opens the
    file for reading; a file whose size differs from its entry is refused.
    Returns ``{"sent"}`` once the next Spark closed every stream, which it does
    when the Sparks after it are done.
    """
    connect = connect or socket.create_connection
    totals, errors = [0] * len(groups), []

    def stream_files(index):
        throttle = Throttle(limit)
        try:
            with connect((addresses[index], ports[index]), CONNECT_SECONDS, (sources[index], 0)) as connection:
                connection.settimeout(read_seconds)
                connection.sendall(token)
                for name, size, reach in groups[index]:
                    with opener(name) as stream:
                        if os.fstat(stream.fileno()).st_size != size:
                            raise ValueError("Pipeline source differs from its pinned size: " + name)
                        connection.sendall((json.dumps([name, size, reach]) + "\n").encode())
                        if limit:
                            remaining = size
                            while remaining:
                                block = stream.read(min(chunk, remaining))
                                if not block:
                                    raise ValueError("Pipeline source changed while streaming: " + name)
                                throttle(len(block))
                                connection.sendall(block)
                                remaining -= len(block)
                        elif size and connection.sendfile(stream, 0, size) != size:
                            raise ValueError("Pipeline source changed while streaming: " + name)
                    totals[index] += size
                connection.shutdown(socket.SHUT_WR)
                while connection.recv(65536):
                    pass
        except (OSError, ValueError) as error:
            errors.append(error)

    threads = [threading.Thread(target=stream_files, args=(index,)) for index in range(len(groups))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return {"sent": sum(totals)}


def run_hop(kind, place, listen, peers, files, want, options, stdin=None, stdout=None):
    """Entry of the shipped hop program (``hop_source``).

    ``kind`` is ``checkpoint`` (``place``: ``root``, ``repository``,
    ``revision``; ``hop_checkpoint``) or ``directory`` (``place``: ``root``;
    ``DirectorySink``). Reads the 32-byte token from ``stdin``, then, after the
    announcement, one JSON line naming the next Spark (``null`` for the last).
    Writes the announcement, one line per placed file and ``{"result"}`` to
    ``stdout`` as JSON lines.
    """
    stdin = stdin or sys.stdin.buffer
    stdout = stdout or sys.stdout
    guard = threading.Lock()

    def emit(value):
        with guard:
            stdout.write(json.dumps(value) + "\n")
            stdout.flush()

    token = stdin.read(32)
    if len(token) != 32:
        raise ValueError("The pipeline token is missing")

    def downstream():
        line = stdin.readline()
        return json.loads(line) if line.strip() else None

    arguments = dict(options, token=token, want=want, downstream=downstream, announce=emit, progress=emit)
    if kind == "checkpoint":
        result = hop_checkpoint(place["root"], place["repository"], place["revision"], listen, peers, files,
                                **arguments)
    elif kind == "directory":
        result = hop(listen, peers, files, DirectorySink(place["root"], files), **arguments)
    else:
        raise ValueError("Unknown pipeline sink: " + str(kind)[:100])
    emit({"result": result})
    return result


def run_push(root, sources, addresses, ports, groups, options, stdin=None, stdout=None):
    """Entry of the shipped source program (``push_source``): send files below ``root``; prints ``{"result"}``."""
    stdin = stdin or sys.stdin.buffer
    stdout = stdout or sys.stdout
    token = stdin.read(32)
    if len(token) != 32:
        raise ValueError("The pipeline token is missing")
    result = push(sources, addresses, ports, groups, files_in(root), token=token, **options)
    stdout.write(json.dumps({"result": result}) + "\n")
    stdout.flush()
    return result


# A pipeline program reaches its Spark on stdin, behind its length, so the SSH
# command line stays short whatever the size of the program and its manifest;
# the program then reads its own input from the same stream.
BOOT = ["python3", "-I", "-c", "import sys; exec(compile(sys.stdin.buffer.read(int(sys.stdin.buffer.readline())), "
        "'<sparkring-pipeline>', 'exec'))"]


def boot_input(program):
    """The first bytes of a pipeline process's stdin under ``BOOT``: the program's length, then the program."""
    data = program.encode()
    return b"%d\n" % len(data) + data


def _program(parts, entry, arguments):
    """A pipeline program: the stdlib imports, the constants, the source of ``parts`` and the call of ``entry``."""
    module = sys.modules[__name__]
    lines = [f"import {name}" for name in PROGRAM_IMPORTS]
    lines += [f"{name} = {getattr(module, name)!r}" for name in PROGRAM_CONSTANTS]
    text = "\n".join(lines) + "\n\n\n" + "\n\n\n".join(inspect.getsource(part) for part in parts)
    return text + f"\n\n\n{entry.__name__}(*json.loads({json.dumps(arguments)!r}))\n"


def hop_source(kind, place, listen, peers, files, want=None, options=None):
    """The hop program of one Spark of a pipeline (``run_hop``), self-contained; it runs under ``BOOT``."""
    parts = ([checkpoint_place, Throttle, CheckpointSink, hop, hop_checkpoint] if kind == "checkpoint"
             else [safe_name, Throttle, DirectorySink, hop])
    arguments = [kind, place, list(listen), list(peers), files, None if want is None else sorted(want),
                 {key: value for key, value in (options or {}).items() if key in HOP_OPTIONS}]
    return _program(parts + [run_hop], run_hop, arguments)


def push_source(root, sources, addresses, ports, groups, options=None):
    """The source program of a pipeline's first Spark (``run_push``), self-contained; it runs under ``BOOT``."""
    arguments = [root, list(sources), list(addresses), list(ports), groups,
                 {key: value for key, value in (options or {}).items() if key in PUSH_OPTIONS}]
    return _program([safe_name, Throttle, files_in, push, run_push], run_push, arguments)
