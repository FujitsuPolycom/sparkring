"""Copy pinned checkpoint files between two cabled Sparks over their fabric link.

Administration traffic uses SSH, which reaches workers through the WireGuard
control network or, without one, SSH over the fabric; both carry one encrypted
stream per connection (0.62 GB/s through WireGuard on TP2). A checkpoint copy
instead uses plain TCP between the two ends of one fabric cable, one stream per
fabric function the cable's ports share (2.7-2.9 GB/s per stream on TP2).

The receiver binds only its own fabric addresses, accepts only the sender's
fabric address on each subnet, and requires a random token that the controller
delivers to both ends over their authenticated administration sessions. The
bytes are public model weights; the path provides neither confidentiality nor
integrity by itself, so the receiver checks every file against the pinned
SHA-256 that the controller sends it.

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
"""
import inspect
import ipaddress

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


def tree(count, donor):
    """Cable-adjacent copy order from ``donor``: a list of levels of (source, target)."""
    neighbors = {rank: ([1 - rank] if count == 2 else [(rank + 1) % count, (rank - 1) % count]) for rank in range(count)}
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
