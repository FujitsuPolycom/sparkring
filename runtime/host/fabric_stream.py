"""Copy verified checkpoint files between two cabled Sparks over their fabric link.

Administration traffic uses SSH, which reaches workers through the WireGuard
control network or, without one, SSH over the fabric; both carry one encrypted
stream per connection (0.62 GB/s through WireGuard on TP2). A checkpoint copy
instead uses plain TCP between the two ends of one fabric cable, one stream per
fabric function the cable's ports share (2.7-2.9 GB/s per stream on TP2).

The receiver binds only its own fabric addresses, accepts only the sender's
fabric address on each subnet, and requires a random token that the controller
delivers to both ends over their authenticated administration sessions. It
writes only the files named in the verified transfer manifest, at their
manifest sizes, and the checkpoint receipt then re-hashes every file against
that manifest. The bytes are public model weights; the path provides neither
confidentiality nor integrity by itself.
"""
import ipaddress


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


def receive(addresses, peers, root, files, token=None, announce=None):
    """Run on the target as root. ``files`` maps name to [size, sha256].

    Files that already match are kept. The first stdout line announces the
    listening ports and the files still needed; the sender then connects.
    """
    import concurrent.futures
    import hashlib
    import hmac
    import json
    import socket
    import sys
    import threading
    from pathlib import Path, PurePosixPath

    token = token if token is not None else sys.stdin.buffer.read(32)
    announce = announce or (lambda value: print(json.dumps(value), flush=True))
    root = Path(root)
    for name in files:
        parts = PurePosixPath(name).parts
        if PurePosixPath(name).is_absolute() or ".." in parts or not parts:
            raise ValueError("Unsafe checkpoint file name")

    def current(name):
        path = root / name
        if not path.is_file() or path.stat().st_size != files[name][0]:
            return False
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            while block := stream.read(16 << 20):
                digest.update(block)
        return digest.hexdigest() == files[name][1]

    with concurrent.futures.ThreadPoolExecutor(16) as pool:
        needed = sorted(name for name, ok in zip(files, pool.map(current, files)) if not ok)
    listeners = [socket.create_server((address, 0)) for address in addresses]
    for listener in listeners:
        listener.settimeout(120)
    announce({"ports": [listener.getsockname()[1] for listener in listeners], "needed": needed})
    expected, done, guard = set(needed), set(), threading.Lock()

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
                    if name not in expected or name in done or size != files[name][0]:
                        raise ValueError("Checkpoint stream sent an unplanned file: " + str(name)[:200])
                    done.add(name)
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "wb") as output:
                    remaining = size
                    while remaining:
                        block = stream.read(min(remaining, 16 << 20))
                        if not block:
                            raise ValueError("Checkpoint stream ended inside " + name)
                        output.write(block)
                        remaining -= len(block)
                total += size
            return total

    if not needed:
        for listener in listeners:
            listener.close()
        return {"received": 0, "bytes": 0}
    with concurrent.futures.ThreadPoolExecutor(len(listeners)) as pool:
        total = sum(pool.map(serve, range(len(listeners))))
    if done != expected:
        raise ValueError("Checkpoint stream omitted planned files")
    return {"received": len(done), "bytes": total}


def send(sources, addresses, ports, root, groups, token=None):
    """Run on the source. Stream ``groups[i]`` from ``sources[i]`` to ``addresses[i]:ports[i]``."""
    import concurrent.futures
    import json
    import socket
    import sys
    from pathlib import Path

    token = token if token is not None else sys.stdin.buffer.read(32)
    root = Path(root)

    def push(index):
        total = 0
        with socket.create_connection((addresses[index], ports[index]), timeout=60,
                                      source_address=(sources[index], 0)) as connection:
            connection.settimeout(300)
            connection.sendall(token)
            for name in groups[index]:
                with open(root / name, "rb") as stream:
                    size = stream.seek(0, 2)
                    stream.seek(0)
                    connection.sendall((json.dumps([name, size]) + "\n").encode())
                    if connection.sendfile(stream) != size:
                        raise ValueError("Checkpoint file changed while streaming: " + name)
                total += size
            connection.shutdown(socket.SHUT_WR)
            connection.recv(1)
        return total

    with concurrent.futures.ThreadPoolExecutor(len(groups)) as pool:
        return {"sent": sum(pool.map(push, range(len(groups))))}
