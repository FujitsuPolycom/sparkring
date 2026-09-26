"""Read-only survey of one Spark for copies of a pinned checkpoint revision.

Node A sends the text returned by ``probe_source`` to every rank and runs it as
root with ``python3 -I -B -``. The functions in ``SHIPPED`` use only the header
imports of that text, builtins and each other, so a rank whose installed
package differs surveys with Node A's pins and rules.

The survey never writes. It lists directories and reads ``lstat`` results,
symlink targets, small text files, Hugging Face (HF) download metadata and
SparkRing records, and it hashes files of at most ``small_file_bytes`` within a
per-node budget. Every system call on a host path goes through ``guard``, which
refuses paths on network or automount filesystems and paths below a directory
holding ``.sparkring-ignore``. Symlinks are followed only by ``safe_resolve``,
one hop at a time, with the guard applied to every hop; a path is used for a
system call only after its directories are known to be real (resolved, or
listed as non-symlink directories), so the kernel never follows a symlink on
the survey's behalf.

Host paths are POSIX strings as the Spark sees them. ``options["root"]``
prefixes every host path when a system call is made, so tests run the survey
against a fixture tree; paths in the result never carry that prefix.

The result (``sparkring-checkpoint-survey/v1``) holds:

- ``owned``: the SparkRing checkpoint directory this installation uses, its
  mount, filesystem type, free space, claim state and verified files, and the
  device of the deployment's compile cache;
- ``candidates``: folders, HF repository folders, snapshot folders and blob
  stores holding pinned files, with per-file ``state`` (``match``,
  ``differs``, ``size-only``, ``missing``, ``incomplete``, ``outside`` or
  ``network``) and ``evidence`` (``recorded``, ``hashed``, ``hub-named``,
  ``hub-metadata`` or ``size``);
- ``not_used``: near misses and copies in other accounts' homes, at most five;
- ``named``: the classification of every ``--model-path`` on this Spark;
- ``docker``: user-namespace remapping, storage driver and root device;
- ``search``: completeness, budgets used, skipped mounts and errors.
"""
import hashlib
import inspect
import json
import os
import stat
import subprocess
import sys  # noqa: F401  (part of the shipped header; kept importable in-process)
import time


def mount_table(root="/"):
    """Mounts listed in ``<root>/proc/self/mountinfo``, in the kernel's order.

    Each entry holds ``id``, ``parent``, ``device`` ("major:minor"), ``root``
    (the mounted subtree of its filesystem), ``path`` (the mount point, octal
    escapes decoded), ``type``, ``source`` and ``rotational``. A later entry at
    the same path is stacked over an earlier one. ``rotational`` is true for a
    rotational or USB-attached block device, read from
    ``/sys/dev/block/<major>:<minor>`` for local filesystem types only. A
    missing mount table raises ``OSError``: without it no path can be checked.
    """
    prefix = "" if root in ("", "/") else root.rstrip("/")
    local = ("ext2", "ext3", "ext4", "xfs", "btrfs", "f2fs", "zfs", "exfat", "vfat", "ntfs3", "fuseblk")
    speeds = {}

    def read(path):
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        try:
            chunks = []
            while True:
                chunk = os.read(descriptor, 1 << 16)
                if not chunk:
                    return b"".join(chunks)
                chunks.append(chunk)
        finally:
            os.close(descriptor)

    def unescape(raw):
        out = bytearray()
        position = 0
        while position < len(raw):
            code = raw[position + 1:position + 4]
            if raw[position] == 92 and len(code) == 3 and all(48 <= c <= 55 for c in code):
                out.append(int(code, 8))
                position += 4
            else:
                out.append(raw[position])
                position += 1
        return out.decode("utf-8", "surrogateescape")

    def rotational(device):
        # Sysfs is kernel state, not storage the guard protects, so it is read directly.
        if device not in speeds:
            base = prefix + "/sys/dev/block/" + device
            try:
                value = "/usb" in os.readlink(base)
            except OSError:
                value = False
            if not value:
                for name in (base + "/queue/rotational", base + "/../queue/rotational"):
                    try:
                        value = read(name).strip() == b"1"
                        break
                    except OSError:
                        continue
            speeds[device] = value
        return speeds[device]

    table = []
    for line in read(prefix + "/proc/self/mountinfo").splitlines():
        fields = line.split(b" ")
        try:
            separator = fields.index(b"-", 6)
            entry = {"id": int(fields[0]), "parent": int(fields[1]), "device": fields[2].decode("ascii"),
                     "root": unescape(fields[3]), "path": unescape(fields[4]),
                     "type": fields[separator + 1].decode("utf-8", "replace"),
                     "source": unescape(fields[separator + 2])}
        except (ValueError, IndexError, UnicodeDecodeError):
            continue
        entry["rotational"] = rotational(entry["device"]) if entry["type"] in local else False
        table.append(entry)
    return table


def guard(ctx, path, op=None, arg=None):
    """Check a host path, then make at most one system call on it.

    The check refuses a path whose deepest covering mount, found from its
    lexical path, is not a local filesystem type, and records that mount under
    ``ctx["skipped"]``. It also refuses a path below a directory holding
    ``.sparkring-ignore``; a marker affects the paths below its directory, so
    a marked directory is still listed but nothing in it is used.
    ``ctx["refusal"]`` then names the reason: ``network``, ``unsupported`` or
    ``ignored``.

    ``op`` selects the call. None returns the normalized path. ``mount``
    returns the covering mount entry without any check or system call.
    ``lstat``, ``readlink`` and ``statvfs`` return the call's result.
    ``scandir`` returns ``(entries, truncated)`` with at most ``arg`` entries.
    ``read`` returns the content of a regular file of at most ``arg`` bytes.
    ``hash`` returns the SHA-256 of a regular file whose device, inode and size
    equal those of the ``lstat`` result ``arg``. Files are opened with
    ``O_NOFOLLOW``, ``O_NONBLOCK`` and, when permitted, ``O_NOATIME``, so a
    symlink, FIFO or device is never followed or waited on.

    A refused path or a failed call returns None. A missing path is not an
    error; a permission error is counted in ``ctx["unreadable"]``, and any
    other ``OSError`` is recorded in ``ctx["errors"]``.
    """
    local = ("ext2", "ext3", "ext4", "xfs", "btrfs", "f2fs", "zfs", "exfat", "vfat", "ntfs3", "fuseblk")
    network = ("autofs", "nfs", "nfs4", "cifs", "smb3", "smbfs", "fuse.sshfs", "fuse.rclone",
               "9p", "ceph", "glusterfs", "lustre")
    parts = []
    for part in path.split("/"):
        if part == "..":
            if parts:
                parts.pop()
        elif part and part != ".":
            parts.append(part)
    prefixes = ["/"]
    for part in parts:
        prefixes.append(prefixes[-1].rstrip("/") + "/" + part)
    lexical = prefixes[-1]
    table = ctx.get("mount_index")
    if table is None:
        table = ctx["mount_index"] = {entry["path"]: entry for entry in ctx["mounts"]}
    memo = ctx.setdefault("mount_memo", {})

    def covering(depth):
        here = prefixes[depth]
        if here in table:
            return table[here]
        if depth == 0:
            return None
        parent = prefixes[depth - 1]
        if parent not in memo:
            memo[parent] = covering(depth - 1)
        return memo[parent]

    mount = covering(len(parts))
    if op == "mount":
        return mount
    kind = mount["type"] if mount is not None else None
    if kind not in local:
        ctx["refusal"] = "network" if kind in network else "unsupported"
        if mount is not None:
            ctx.setdefault("skipped", {}).setdefault(
                mount["path"], {"path": mount["path"], "type": kind, "network": kind in network})
        return None
    prefix = ctx["root"]
    marks = ctx.setdefault("ignore", {})
    for depth in range(len(parts)):
        directory = prefixes[depth]
        marked = marks.get(directory)
        if marked is None:
            holder = covering(depth)
            marked = False
            if holder is not None and holder["type"] in local:
                try:
                    os.lstat(prefix + directory.rstrip("/") + "/.sparkring-ignore")
                    marked = True
                except OSError:
                    marked = False
            marks[directory] = marked
        if marked:
            ctx["refusal"] = "ignored"
            return None
    if op is None:
        return lexical
    full = prefix + lexical
    try:
        if op == "lstat":
            return os.lstat(full)
        if op == "readlink":
            return os.readlink(full)
        if op == "statvfs":
            return os.statvfs(full)
        if op == "scandir":
            entries = []
            with os.scandir(full) as listing:
                for entry in listing:
                    if len(entries) >= arg:
                        return entries, True
                    entries.append(entry)
            return entries, False
        flags = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
                 | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_BINARY", 0))
        try:
            descriptor = os.open(full, flags | getattr(os, "O_NOATIME", 0))
        except PermissionError:
            # O_NOATIME needs the file's owner or CAP_FOWNER; retry without it.
            descriptor = os.open(full, flags)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                return None
            if op == "read":
                if info.st_size > arg:
                    return None
                chunks, total = [], 0
                while total <= arg:
                    chunk = os.read(descriptor, 1 << 16)
                    if not chunk:
                        return b"".join(chunks)
                    chunks.append(chunk)
                    total += len(chunk)
                return None
            if (info.st_dev, info.st_ino, info.st_size) != (arg.st_dev, arg.st_ino, arg.st_size):
                return None
            digest, left = hashlib.sha256(), info.st_size
            while left > 0:
                chunk = os.read(descriptor, min(left, 1 << 20))
                if not chunk:
                    return None
                digest.update(chunk)
                left -= len(chunk)
            if os.read(descriptor, 1):
                return None
            return digest.hexdigest()
        finally:
            os.close(descriptor)
    except (FileNotFoundError, NotADirectoryError):
        return None
    except PermissionError:
        ctx["unreadable"] = ctx.get("unreadable", 0) + 1
        return None
    except OSError as error:
        errors = ctx.setdefault("errors", [])
        if len(errors) < 20:
            errors.append(f"{op} {lexical}: {error.strerror or error}")
        return None


def safe_resolve(ctx, path, inside=None, hops=8):
    """Resolve symlinks in ``path`` one hop at a time, guarding every step.

    Components are examined from the root down; directories already known to
    be real are not examined again. A symlink's target is computed lexically
    against the directory holding the link; ``..`` is applied to resolved
    components only. When ``inside`` (a resolved directory) is given, ``path``
    must start with it, every hop target must stay inside it, and so must the
    result.

    Returns ``(state, path, lstat)``: ``ok`` with the resolved path and its
    ``lstat``; ``missing`` with the first missing component; ``outside`` with
    the hop target or result that left ``inside``; ``loop`` after more than
    ``hops`` links; or the guard's refusal (``network``, ``unsupported`` or
    ``ignored``) with the refused path.
    """
    def split(text):
        return [part for part in text.split("/") if part and part != "."]

    def lexical(parts):
        result = []
        for part in parts:
            if part == "..":
                if result:
                    result.pop()
            else:
                result.append(part)
        return result

    def within(parts):
        return limit is None or parts[:len(limit)] == limit

    real = ctx.setdefault("real", set())
    pending, done, limit = split(path), [], None
    if inside is not None:
        limit = lexical(split(inside))
        if pending[:len(limit)] != limit:
            return ("outside", "/" + "/".join(lexical(pending)), None)
        done, pending = pending[:len(limit)], pending[len(limit):]
    count, info = 0, None
    while pending:
        name = pending.pop(0)
        if name == "..":
            if done:
                done.pop()
            info = None
            if not within(done):
                return ("outside", "/" + "/".join(done), None)
            continue
        current = done + [name]
        text = "/" + "/".join(current)
        if pending and text in real:
            done, info = current, None
            continue
        if guard(ctx, text) is None:
            return (ctx.get("refusal", "unsupported"), text, None)
        info = guard(ctx, text, "lstat")
        if info is None:
            return ("missing", text, None)
        if stat.S_ISLNK(info.st_mode):
            count += 1
            if count > hops:
                return ("loop", text, None)
            target = guard(ctx, text, "readlink")
            if target is None:
                return ("missing", text, None)
            hop = lexical(([] if target.startswith("/") else done) + split(target))
            if not within(hop):
                return ("outside", "/" + "/".join(hop), None)
            if target.startswith("/"):
                done = []
            pending = split(target) + pending
            info = None
            continue
        if pending:
            if not stat.S_ISDIR(info.st_mode):
                return ("missing", text, None)
            real.add(text)
        done = current
    final = "/" + "/".join(done)
    if not within(done):
        return ("outside", final, None)
    if info is None:
        # The last step was "..", a symlink or a known directory: ``final`` is real.
        if guard(ctx, final) is None:
            return (ctx.get("refusal", "unsupported"), final, None)
        info = guard(ctx, final, "lstat")
        if info is None:
            return ("missing", final, None)
    return ("ok", final, info)


def accounts(ctx):
    """Accounts whose home directory exists, from ``/etc/passwd`` read as text.

    The ``pwd`` module is not used, because name services can query network
    directories. System homes (``/``, ``/nonexistent``, ``/bin``, ``/sbin``,
    ``/proc``, ``/sys``, ``/dev`` and anything in ``/usr``, ``/run`` or
    ``/var/run``) are skipped. ``kind`` is ``root`` for uid 0, ``operator`` for
    the operator's account, ``service`` below uid 1000 and ``private``
    otherwise. Each row holds ``account``, ``uid``, ``home`` (as written) and
    ``resolved``. Also sets ``ctx["owners"]``, uid to account name.
    """
    operator = ctx["options"]["operator"]
    state, path, info = safe_resolve(ctx, "/etc/passwd")
    data = (guard(ctx, path, "read", 16 << 20) if state == "ok" else None) or b""
    system = ("", "/nonexistent", "/bin", "/sbin", "/proc", "/sys", "/dev", "/usr", "/run", "/var/run")
    below = ("/usr/", "/run/", "/var/run/", "/proc/", "/sys/", "/dev/")
    owners, rows = {}, []
    for line in data.decode("utf-8", "replace").splitlines():
        fields = line.split(":")
        if len(fields) < 7:
            continue
        try:
            uid = int(fields[2])
        except ValueError:
            continue
        name, home = fields[0], fields[5].rstrip("/")
        owners.setdefault(uid, name)
        if not home.startswith("/") or home in system or home.startswith(below):
            continue
        state, resolved, info = safe_resolve(ctx, home)
        if state != "ok" or not stat.S_ISDIR(info.st_mode):
            continue
        kind = "root" if uid == 0 else "operator" if name == operator else "service" if uid < 1000 else "private"
        rows.append({"account": name, "uid": uid, "home": home, "resolved": resolved, "kind": kind})
    ctx["owners"] = owners
    return rows


def declared_hub_roots(ctx, homes):
    """Hub roots named by HF cache variables, read as text and never executed.

    Files: ``/etc/environment``, ``/etc/profile.d/*.sh``, ``/etc/bash.bashrc``,
    ``Environment=`` lines of ``/etc/systemd/system/*.service`` and
    ``/etc/systemd/system/*.service.d/*.conf``, and, for root and the operator
    only, their shell, fish and ``environment.d`` start-up files. Forms:
    ``[export ]NAME=VALUE``, fish ``set -x|-gx NAME VALUE`` and systemd
    ``Environment=NAME=VALUE``. Expansion covers ``~``, ``$HOME``, ``${HOME}``,
    ``$USER`` and variables assigned earlier in the same file; system-wide
    shell files are expanded once for root and once for the operator, and
    systemd values are taken literally. A value holding a backquote, ``$(`` or
    any other ``$`` is ignored. ``HF_HUB_CACHE``, ``HUGGINGFACE_HUB_CACHE`` and
    ``TRANSFORMERS_CACHE`` name a hub root, ``HF_HOME`` gives ``<value>/hub``
    and ``XDG_CACHE_HOME`` gives ``<value>/huggingface/hub``. Only these derived
    roots are returned; nothing else read from the files is kept.
    """
    variables = ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME", "TRANSFORMERS_CACHE")
    people = [row for row in homes if row["kind"] in ("root", "operator")]
    everyone = people or [{"account": "root", "home": "/root", "resolved": "/root", "kind": "root"}]
    roots = set()

    def text(path):
        state, resolved, info = safe_resolve(ctx, path)
        if state != "ok" or not stat.S_ISREG(info.st_mode):
            return None
        data = guard(ctx, resolved, "read", 1 << 20)
        return None if data is None else data.decode("utf-8", "replace")

    def matching(directory, suffix, directories=False):
        state, resolved, info = safe_resolve(ctx, directory)
        if state != "ok" or not stat.S_ISDIR(info.st_mode):
            return []
        listing = guard(ctx, resolved, "scandir", 1000)
        names = []
        for entry in (listing or ([], False))[0]:
            if entry.name.startswith(".") or not entry.name.endswith(suffix):
                continue
            try:
                if directories and (entry.is_symlink() or not entry.is_dir(follow_symlinks=False)):
                    continue
            except OSError:
                continue
            names.append(resolved + "/" + entry.name)
        return sorted(names)

    def tokens(line):
        # Whitespace-separated words; quotes group characters. A word is literal
        # when every character of it was single-quoted.
        words, word, quote, literal, started = [], [], None, True, False
        for character in line:
            if quote:
                if character == quote:
                    quote = None
                else:
                    word.append(character)
            elif character in "'\"":
                quote, started = character, True
                literal = literal and character == "'"
            elif character.isspace():
                if started:
                    words.append(("".join(word), literal))
                word, literal, started = [], True, False
            elif character == "#" and not started:
                break
            else:
                word.append(character)
                literal, started = False, True
        if started:
            words.append(("".join(word), literal))
        return words

    def assignments(content, systemd):
        found = []
        for raw in content.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if systemd:
                if line.startswith("Environment="):
                    for word, _ in tokens(line[len("Environment="):]):
                        name, separator, value = word.partition("=")
                        if separator:
                            found.append((name, value, True))
                continue
            if line.startswith("set "):
                words = tokens(line)
                if len(words) >= 4 and words[1][0] in ("-x", "-gx"):
                    found.append((words[2][0], words[3][0], words[3][1]))
                continue
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            name, separator, rest = line.partition("=")
            if separator and name and (name[0].isalpha() or name[0] == "_") and all(
                    c.isalnum() or c == "_" for c in name):
                words = tokens(rest)
                found.append((name, words[0][0] if words else "", words[0][1] if words else True))
        return found

    def expand(value, literal, person, assigned):
        if "`" in value or "$(" in value:
            return None
        if not literal:
            if value == "~" or value.startswith("~/"):
                value = person["home"] + value[1:]
            known = {"HOME": person["home"], "USER": person["account"], **assigned}
            out, position = [], 0
            while position < len(value):
                if value[position] != "$":
                    out.append(value[position])
                    position += 1
                    continue
                if value[position + 1:position + 2] == "{":
                    end = value.find("}", position)
                    if end < 0:
                        return None
                    name, position = value[position + 2:end], end + 1
                else:
                    end = position + 1
                    while end < len(value) and (value[end].isalnum() or value[end] == "_"):
                        end += 1
                    name, position = value[position + 1:end], end
                if name not in known:
                    return None
                out.append(known[name])
            value = "".join(out)
        if "$" in value or not value.startswith("/"):
            return None
        return value

    def scan(path, systemd, persons):
        content = text(path)
        if content is None:
            return
        found = assignments(content, systemd)
        for person in persons:
            assigned = {}
            for name, value, literal in found:
                result = expand(value, literal, person, assigned)
                if result is None:
                    continue
                assigned[name] = result
                if name in variables:
                    base = result.rstrip("/")
                    if name == "HF_HOME":
                        roots.add(base + "/hub")
                    elif name == "XDG_CACHE_HOME":
                        roots.add(base + "/huggingface/hub")
                    else:
                        roots.add(base or "/")

    for path in ["/etc/environment", "/etc/bash.bashrc", *matching("/etc/profile.d", ".sh")]:
        scan(path, False, everyone)
    for path in matching("/etc/systemd/system", ".service"):
        scan(path, True, everyone[:1])
    for directory in matching("/etc/systemd/system", ".service.d", directories=True):
        for path in matching(directory, ".conf"):
            scan(path, True, everyone[:1])
    for person in people:
        home = person["resolved"]
        names = [home + "/" + name for name in (".bashrc", ".bash_profile", ".bash_aliases", ".profile", ".zshrc",
                                                 ".zshenv", ".zprofile", ".config/fish/config.fish")]
        names += matching(home + "/.bashrc.d", "") + matching(home + "/.config/environment.d", ".conf")
        for path in names:
            scan(path, False, [person])
    return sorted(roots)


def docker_sources(ctx):
    """Walk roots and hub roots named by Docker containers and volumes.

    Five calls (``ps -aq``, one ``inspect``, ``volume ls -q``, ``volume
    inspect`` and ``info``) share ``options["docker_seconds"]``. After a
    timeout, or when ``docker`` cannot run, the remaining calls are skipped and
    ``docker: unavailable: <message>`` is recorded; a call that fails is
    recorded and the others still run. With ``options["ignore_local"]`` only
    ``info`` runs: the search ends after SparkRing's own directories and named
    paths, and user-namespace remapping still decides which files can be
    linked.

    Returns ``roots`` (``(path, priority, tag)``: each bind or volume mount
    source, priority 1 when its container destination suggests model storage,
    and each named volume's mount point at priority 1), ``hubs``
    (``(path, tag)``: container hub paths mapped back to host paths through the
    mounts, or into an ``overlay2`` writable layer), and ``userns``, ``driver``
    and ``root`` from ``docker info``.
    """
    allowance = ctx["options"]["docker_seconds"]
    started = time.monotonic()
    errors = ctx.setdefault("errors", [])
    status = {"dead": False}

    def call(arguments):
        if status["dead"]:
            return None
        left = allowance - (time.monotonic() - started)
        if left <= 0:
            status["dead"] = True
            errors.append(f"docker: unavailable: its {allowance} s allowance ran out before docker {arguments[0]}")
            return None
        try:
            done = subprocess.run(["docker", "--context", "default", *arguments], capture_output=True, text=True,
                                  timeout=left, check=False, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            status["dead"] = True
            errors.append(f"docker: unavailable: docker {' '.join(arguments[:2])} did not answer within "
                          f"the {allowance} s allowance")
            return None
        except OSError as error:
            status["dead"] = True
            errors.append(f"docker: unavailable: {error.strerror or error}")
            return None
        if done.returncode != 0:
            message = (done.stderr or "").strip().splitlines()
            errors.append(f"docker: {' '.join(arguments[:2])} failed: {message[0] if message else done.returncode}")
            return None
        return done.stdout

    def load(text, default):
        if text is None:
            return default
        try:
            value = json.loads(text)
        except ValueError:
            errors.append("docker: unreadable output")
            return default
        return value if isinstance(value, type(default)) else default

    containers, volumes = [], []
    if not ctx["options"].get("ignore_local"):
        identifiers = (call(["ps", "-aq"]) or "").split()
        containers = load(call(["inspect", *identifiers]), []) if identifiers else []
        names = (call(["volume", "ls", "-q"]) or "").split()
        volumes = load(call(["volume", "inspect", *names]), []) if names else []
    info = load(call(["info", "--format", "{{json .}}"]), {})

    words = ("model", "hf", "hugging", "checkpoint")
    roots, hubs = [], []
    for container in containers:
        if not isinstance(container, dict):
            continue
        mounts = []
        for mount in container.get("Mounts") or []:
            if not isinstance(mount, dict) or mount.get("Type") not in ("bind", "volume"):
                continue
            source, destination = mount.get("Source"), mount.get("Destination")
            if not isinstance(source, str) or not source.startswith("/") or not isinstance(destination, str):
                continue
            mounts.append((destination.rstrip("/") or "/", source.rstrip("/") or "/"))
            priority = 1 if any(word in destination.lower() for word in words) else 2
            roots.append((source, priority, "docker-mount"))
        config = container.get("Config") or {}
        environment = {}
        for item in config.get("Env") or []:
            if isinstance(item, str) and "=" in item:
                key, _, value = item.partition("=")
                environment[key] = value
        paths = [environment[key] for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE")
                 if environment.get(key)]
        if environment.get("HF_HOME"):
            paths.append(environment["HF_HOME"].rstrip("/") + "/hub")
        if not paths:
            paths.append("/root/.cache/huggingface/hub")
        arguments = [item for item in (container.get("Args") or []) if isinstance(item, str)]
        for position, item in enumerate(arguments):
            if item == "--download-dir" and position + 1 < len(arguments):
                paths.append(arguments[position + 1])
            elif item.startswith("--download-dir="):
                paths.append(item.partition("=")[2])
        driver = container.get("GraphDriver") or {}
        upper = (driver.get("Data") or {}).get("UpperDir") if driver.get("Name") == "overlay2" else None
        for path in paths:
            path = path.rstrip("/")
            if not path.startswith("/"):
                continue
            covering = [(destination, source) for destination, source in mounts
                        if path == destination or path.startswith(destination.rstrip("/") + "/")]
            if covering:
                destination, source = max(covering, key=lambda item: len(item[0]))
                hubs.append((source.rstrip("/") + path[len(destination.rstrip("/")):], "docker-mount"))
            elif isinstance(upper, str) and upper.startswith("/"):
                hubs.append((upper.rstrip("/") + path, "docker-layer"))
    for volume in volumes:
        if isinstance(volume, dict) and isinstance(volume.get("Mountpoint"), str):
            roots.append((volume["Mountpoint"], 1, "docker-volume"))
    security = info.get("SecurityOptions") or []
    return {"roots": roots, "hubs": hubs,
            "userns": any(isinstance(item, str) and "name=userns" in item for item in security),
            "driver": info.get("Driver") if isinstance(info.get("Driver"), str) else None,
            "root": info.get("DockerRootDir") if isinstance(info.get("DockerRootDir"), str) else None}


def hub_repository_files(ctx, hub, only=None):
    """Pinned files in the Hugging Face hub root ``hub`` (a resolved path).

    Every repository folder ``models--*`` is examined (or only the folder named
    ``only``), because blob names are content addresses: pinned files are found
    under a repository's former name and inside forks. A file matches
    (``hub-named``) when ``blobs/<key>`` resolves inside ``hub`` to a regular
    file of the pinned size; the key is the pinned SHA-256 for LFS files and
    the git blob id otherwise. A snapshot entry naming another blob marks the
    file ``differs``; one naming the pinned blob while that blob is absent
    marks it ``missing``. Snapshot folders holding regular files (caches
    written without symlinks) are classified as folders. Pinned blobs of the
    shared store ``<hub>/blobs/<xet[:2]>/<xet>`` that no repository folder
    reaches form a result of their own. A SparkRing record valid for a blob's
    inode outranks its name. Once the survey's overall deadline
    (``ctx["deadline"]``) has passed, the remaining repository folders and the
    shared store are not examined; their paths go to ``ctx["late"]``. Returns
    results shaped like ``classify_folder``'s, plus ``branches`` and
    ``snapshots``.
    """
    pins = ctx["pins"]
    files = pins["files"]
    optional = set(pins.get("optional") or ())
    required = sorted(name for name in files if name not in optional)
    records = ctx.get("records", {})
    owners = ctx.get("owners", {})
    limit = ctx["options"]["directory_entries"]

    def key(pin):
        return pin["sha256"] if pin.get("lfs") else pin["git_blob"]

    def entry(state, evidence, source, info, kind):
        result = {"state": state, "evidence": evidence, "source": source, "kind": kind}
        if info is not None:
            mount = guard(ctx, source, "mount")
            result.update(size=info.st_size, identity=[info.st_dev, info.st_ino],
                          owner=owners.get(info.st_uid, str(info.st_uid)), mode=stat.S_IMODE(info.st_mode),
                          mount_id=mount["id"] if mount else None)
        return result

    def judge(name, source, info, kind):
        pin = files[name]
        if info.st_size != pin["size"]:
            return entry("differs", "size", source, info, kind)
        recorded = records.get((info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns))
        if recorded and pin["sha256"] not in recorded:
            return entry("differs", "recorded", source, info, kind)
        if recorded and len(recorded) == 1:
            return entry("match", "recorded", source, info, kind)
        return entry("match", "hub-named", source, info, kind)

    def counts(found):
        result = dict.fromkeys(("match", "differs", "size-only", "missing", "incomplete", "outside", "network"), 0)
        for name in required:
            result[(found.get(name) or {}).get("state", "missing")] += 1
        return result

    def listing(path, cap):
        info = guard(ctx, path, "lstat")
        if info is None or not stat.S_ISDIR(info.st_mode):
            return []
        return (guard(ctx, path, "scandir", cap) or ([], False))[0]

    def references(directory, prefix, depth, refs):
        for item in listing(directory, 1000):
            path = directory + "/" + item.name
            try:
                if item.is_symlink():
                    continue
                if item.is_dir(follow_symlinks=False):
                    if depth < 2:
                        references(path, prefix + item.name + "/", depth + 1, refs)
                    continue
            except OSError:
                continue
            data = guard(ctx, path, "read", 256)
            commit = data.decode("ascii", "replace").strip() if data else ""
            if len(commit) == 40 and all(c in "0123456789abcdef" for c in commit):
                refs[prefix + item.name] = commit

    def late(path):
        if time.monotonic() < ctx.get("deadline", float("inf")):
            return False
        ctx.setdefault("late", []).append(path)
        return True

    results = []
    for item in sorted(listing(hub, limit), key=lambda value: value.name):
        if not item.name.startswith("models--") or (only is not None and item.name != only):
            continue
        repository = hub + "/" + item.name
        if late(repository) or guard(ctx, repository) is None:
            continue
        try:
            if item.is_symlink() or not item.is_dir(follow_symlinks=False):
                continue
        except OSError:
            continue
        found = {}
        for name in required:
            state, source, info = safe_resolve(ctx, repository + "/blobs/" + key(files[name]), inside=hub)
            if state == "ok" and stat.S_ISREG(info.st_mode):
                kind = "blob" if source.startswith(repository + "/") else "shared-blob"
                found[name] = judge(name, source, info, kind)
            elif state in ("outside", "loop", "unsupported", "ignored"):
                found[name] = {"state": "outside", "evidence": None, "source": source}
            elif state == "network":
                found[name] = {"state": "network", "evidence": None, "source": source}
        refs = {}
        references(repository + "/refs", "", 0, refs)
        commits = []
        for snapshot in listing(repository + "/snapshots", limit):
            try:
                if not snapshot.is_symlink() and snapshot.is_dir(follow_symlinks=False):
                    commits.append(snapshot.name)
            except OSError:
                continue
        commits.sort()
        plain = []
        for commit in commits:
            folder = repository + "/snapshots/" + commit
            regular = False
            for name in required:
                head, _, tail = name.rpartition("/")
                parent = folder
                if head:
                    state, parent, info = safe_resolve(ctx, folder + "/" + head, inside=folder)
                    if state != "ok" or not stat.S_ISDIR(info.st_mode):
                        continue
                info = guard(ctx, parent + "/" + tail, "lstat")
                if info is None:
                    continue
                if stat.S_ISREG(info.st_mode):
                    regular = True
                elif stat.S_ISLNK(info.st_mode) and name not in found:
                    target = guard(ctx, parent + "/" + tail, "readlink") or ""
                    if target.rsplit("/", 1)[-1] != key(files[name]):
                        found[name] = {"state": "differs", "evidence": "hub-named", "source": None}
                    else:
                        found[name] = {"state": "missing", "evidence": None, "source": None}
            if regular:
                plain.append(commit)
        if pins["revision"] in commits:
            commit = pins["revision"]
        elif refs.get("main") in commits:
            commit = refs["main"]
        else:
            commit = commits[0] if commits else None
        if found:
            results.append({"path": repository, "layout": "hf-cache", "commit": commit,
                            "branches": sorted(name for name, value in refs.items() if value == commit),
                            "snapshots": commits, "files": dict(sorted(found.items())), "counts": counts(found)})
        for commit in plain:
            result = classify_folder(ctx, repository + "/snapshots/" + commit, "hf-snapshot", commit)
            result["branches"] = sorted(name for name, value in refs.items() if value == commit)
            results.append(result)
    if only is None and not late(hub + "/blobs"):
        seen = {tuple(value["identity"]) for result in results for value in result["files"].values()
                if value.get("identity")}
        store = {}
        for name in required:
            xet = files[name].get("xet_hash")
            if not xet:
                continue
            state, path, info = safe_resolve(ctx, hub + "/blobs/" + xet[:2] + "/" + xet, inside=hub)
            if (state == "ok" and stat.S_ISREG(info.st_mode) and info.st_size == files[name]["size"]
                    and (info.st_dev, info.st_ino) not in seen):
                store[name] = judge(name, path, info, "shared-blob")
        if store:
            results.append({"path": hub + "/blobs", "layout": "hf-blob-store", "commit": None, "branches": [],
                            "files": store, "counts": counts(store)})
    return results


def download_metadata(ctx, folder, name, info, pin):
    """Evidence from a Hugging Face ``--local-dir`` download record for ``folder/name``.

    The record is ``.cache/huggingface/download/<name>.metadata`` or, from
    huggingface_hub 0.23, ``.huggingface/download/<name>.metadata``: line 1 is
    the commit, line 2 the ETag, line 3 the time the record was written. It
    applies only while ``st_mtime - 1 <= time``, the client's own rule; a newer
    file gives ``size`` evidence. An ETag equal to the pinned key (SHA-256 for
    LFS files, git blob id otherwise) or to the pinned git blob id (whose LFS
    pointer names the SHA-256) gives ``match``; another ETag gives ``differs``.
    Returns ``{"state", "evidence", "commit"}``, or None without a readable
    record; a corrupt record is ignored, as the client ignores it.
    """
    for base in ("/.cache/huggingface/download/", "/.huggingface/download/"):
        state, path, record = safe_resolve(ctx, folder + base + name + ".metadata", inside=folder)
        data = guard(ctx, path, "read", 4096) if state == "ok" else None
        if data is None:
            continue
        try:
            lines = data.decode("utf-8").splitlines()
            commit, etag, written = lines[0].strip(), lines[1].strip(), float(lines[2].strip())
        except (UnicodeDecodeError, IndexError, ValueError):
            return None
        if etag.startswith("W/"):
            etag = etag[2:]
        etag = etag.strip('"')
        if info.st_mtime - 1 > written:
            return {"state": "size-only", "evidence": "size", "commit": commit}
        key = pin["sha256"] if pin.get("lfs") else pin["git_blob"]
        return {"state": "match" if etag in (key, pin["git_blob"]) else "differs", "evidence": "hub-metadata",
                "commit": commit}
    return None


def sparkring_evidence(ctx):
    """SparkRing's own checkpoint directories and the records it wrote.

    Reads receipts ``/srv/sparkring/*/installer/model.json`` and
    ``/srv/sparkring/*/*/installer/model.json``, path records
    ``/var/lib/sparkring/checkpoints/*.json``, per-inode records
    ``/var/lib/sparkring/checkpoints/files/*.json`` and the journals of
    SparkRing checkpoint directories. Every recorded
    ``(st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns)`` with its SHA-256
    goes to ``ctx["records"]``; a file counts as ``recorded`` only while all
    five values equal its ``lstat``.

    A directory is SparkRing's by records, never by its path:
    ``/srv/sparkring/*/checkpoints/*/<revision>`` when its state directory's
    ``owner.json`` names that path, the revision and the directory's identity;
    ``/srv/sparkring/*[/*]/models/<revision>`` when the enclosing workspace
    holds ``.installer-owner.json`` and its receipt names that path with origin
    ``pinned-hub-download`` or ``verified-fabric-copy``. When ``/srv/sparkring``
    is reached through a symlink, a record may name either the written path or
    the resolved one. Returns ``sparkring`` (those directories, resolved) and
    ``folders`` (``(path, tag)``: other directories at those paths, tagged
    ``folder``, and folders that records name, tagged ``record``).
    """
    revision = ctx["pins"]["revision"]
    records = ctx.setdefault("records", {})
    owned_dirs, folders = [], []

    def directory(path):
        state, resolved, info = safe_resolve(ctx, path)
        return (resolved, info) if state == "ok" and stat.S_ISDIR(info.st_mode) else (None, None)

    def load(path, inside):
        state, resolved, info = safe_resolve(ctx, path, inside=inside)
        data = guard(ctx, resolved, "read", 16 << 20) if state == "ok" else None
        try:
            value = json.loads(data) if data is not None else None
        except ValueError:
            return None
        return value if isinstance(value, dict) else None

    def remember(stats, digest):
        if (isinstance(stats, list) and len(stats) == 5 and all(type(value) is int for value in stats)
                and isinstance(digest, str) and len(digest) == 64):
            records.setdefault(tuple(stats), set()).add(digest)

    def receipt(document):
        hashes, stats = document.get("files"), document.get("file_stats")
        if isinstance(hashes, dict) and isinstance(stats, dict):
            for name, digest in hashes.items():
                remember(stats.get(name), digest)
        path = document.get("path")
        return path if isinstance(path, str) and path.startswith("/") else None

    def children(path):
        listing = guard(ctx, path, "scandir", 10000)
        names = []
        for item in (listing or ([], False))[0]:
            try:
                if not item.is_symlink() and item.is_dir(follow_symlinks=False):
                    names.append(item.name)
            except OSError:
                continue
        return sorted(names)

    def workspace(real, named):
        document = load(real + "/installer/model.json", real)
        recorded = receipt(document) if document else None
        legacy, legacy_name = real + "/models/" + revision, named + "/models/" + revision
        state, resolved, info = safe_resolve(ctx, legacy, inside=real)
        if state == "ok" and stat.S_ISDIR(info.st_mode) and resolved == legacy:
            marker = guard(ctx, real + "/.installer-owner.json", "lstat")
            if (marker is not None and stat.S_ISREG(marker.st_mode) and recorded in (legacy, legacy_name)
                    and document.get("origin") in ("pinned-hub-download", "verified-fabric-copy")):
                owned_dirs.append(legacy)
            else:
                folders.append((legacy, "folder"))
        if recorded and recorded not in (legacy, legacy_name):
            folders.append((recorded, "record"))

    base, _ = directory("/srv/sparkring")
    for cluster in children(base) if base else []:
        real, named = base + "/" + cluster, "/srv/sparkring/" + cluster
        workspace(real, named)
        for name in children(real):
            workspace(real + "/" + name, named + "/" + name)
        checkpoints = real + "/checkpoints"
        info = guard(ctx, checkpoints, "lstat")
        if info is None or not stat.S_ISDIR(info.st_mode):
            continue
        for repository in children(checkpoints):
            path = checkpoints + "/" + repository + "/" + revision
            path_name = named + "/checkpoints/" + repository + "/" + revision
            info = guard(ctx, path, "lstat")
            if info is None or not stat.S_ISDIR(info.st_mode):
                continue
            state = checkpoints + "/" + repository + "/." + revision + ".sparkring"
            owner = load(state + "/owner.json", checkpoints + "/" + repository)
            if (owner and owner.get("schema") == "sparkring-checkpoint-owner/v1"
                    and owner.get("path") in (path, path_name) and owner.get("revision") == revision
                    and owner.get("directory") == [info.st_dev, info.st_ino]):
                owned_dirs.append(path)
                journal = load(state + "/journal.json", checkpoints + "/" + repository) or {}
                entries = journal.get("files") if isinstance(journal.get("files"), dict) else {}
                for value in entries.values():
                    if isinstance(value, dict) and value.get("state") == "placed":
                        remember(value.get("stats"), value.get("sha256"))
            else:
                folders.append((path, "folder"))
    base, _ = directory("/var/lib/sparkring/checkpoints")
    if base:
        listing = guard(ctx, base, "scandir", 10000)
        for item in sorted((listing or ([], False))[0], key=lambda value: value.name):
            if item.name.endswith(".json"):
                document = load(base + "/" + item.name, base)
                path = receipt(document) if document else None
                if path:
                    folders.append((path, "record"))
        info = guard(ctx, base + "/files", "lstat")
        if info is not None and stat.S_ISDIR(info.st_mode):
            listing = guard(ctx, base + "/files", "scandir", 1000000)
            for item in (listing or ([], False))[0]:
                if item.name.endswith(".json"):
                    document = load(base + "/files/" + item.name, base + "/files") or {}
                    remember(document.get("stats"), document.get("sha256"))
    return {"sparkring": owned_dirs, "folders": folders}


def classify_folder(ctx, path, layout=None, commit=None, exact=False):
    """Per-file evidence for the required names in the resolved folder ``path``.

    ``layout`` is detected when not given: ``local-dir`` with HF download
    records, ``git-lfs`` with ``.git/lfs/objects``, else ``folder``. For each
    required name, strongest first: a git-lfs object of the pinned size
    (``hub-named``; the object is the source), a SparkRing record whose five
    stat values equal the file's (``recorded``), a download record
    (``hub-metadata``), the SHA-256 of a file of at most ``small_file_bytes``
    within the node's hash budget (``hashed``), else name and size (``size``).
    Only regular files at their final name with the pinned size are
    considered: another size is ``differs``, a ``<name>.aria2`` sibling makes
    the file ``incomplete``, and a symlink leaving the folder is ``outside``.
    Files in another account's private home (``ctx["private"]``) are not
    hashed unless the folder is below a named path (``ctx["named"]``): such
    copies are listed, not used, so their content is not read.

    With ``exact``, every other regular file and every symlink or special file
    outside ``.cache/huggingface``, ``.huggingface`` and ``.git`` is listed
    under ``extra`` and ``symlinks``, as serving a named folder in place needs.
    """
    pins = ctx["pins"]
    files = pins["files"]
    optional = set(pins.get("optional") or ())
    required = sorted(name for name in files if name not in optional)
    options = ctx["options"]
    records = ctx.get("records", {})
    owners = ctx.get("owners", {})

    def below(bases):
        return any(path == base or path.startswith(base.rstrip("/") + "/") for base in bases)

    hashing = exact or below(ctx.get("named", ())) or not below(ctx.get("private", ()))

    def present(relative):
        state, resolved, info = safe_resolve(ctx, path + "/" + relative, inside=path)
        return state == "ok" and stat.S_ISDIR(info.st_mode)

    if layout is None:
        if present(".cache/huggingface/download") or present(".huggingface/download"):
            layout = "local-dir"
        elif present(".git/lfs/objects"):
            layout = "git-lfs"
        else:
            layout = "folder"

    def entry(state, evidence, source, info, kind):
        result = {"state": state, "evidence": evidence, "source": source, "kind": kind}
        if info is not None:
            mount = guard(ctx, source, "mount")
            result.update(size=info.st_size, identity=[info.st_dev, info.st_ino],
                          owner=owners.get(info.st_uid, str(info.st_uid)), mode=stat.S_IMODE(info.st_mode),
                          mount_id=mount["id"] if mount else None)
        return result

    found, commits = {}, {}
    for name in required:
        pin = files[name]
        if layout == "git-lfs" and pin.get("lfs"):
            digest = pin["sha256"]
            state, source, info = safe_resolve(
                ctx, path + "/.git/lfs/objects/" + digest[:2] + "/" + digest[2:4] + "/" + digest, inside=path)
            if state == "ok" and stat.S_ISREG(info.st_mode) and info.st_size == pin["size"]:
                found[name] = entry("match", "hub-named", source, info, "lfs-object")
                continue
        head, _, tail = name.rpartition("/")
        parent = path
        if head:
            state, parent, info = safe_resolve(ctx, path + "/" + head, inside=path)
            if state in ("outside", "loop", "unsupported", "ignored", "network"):
                found[name] = {"state": "network" if state == "network" else "outside", "evidence": None,
                               "source": parent}
                continue
            if state != "ok" or not stat.S_ISDIR(info.st_mode):
                continue
        state, source, info = safe_resolve(ctx, parent + "/" + tail, inside=path)
        if state in ("outside", "loop", "unsupported", "ignored", "network"):
            found[name] = {"state": "network" if state == "network" else "outside", "evidence": None,
                           "source": source}
            continue
        if state != "ok" or not stat.S_ISREG(info.st_mode):
            continue
        if guard(ctx, parent + "/" + tail + ".aria2", "lstat") is not None:
            found[name] = entry("incomplete", None, source, info, "file")
            continue
        if info.st_size != pin["size"]:
            found[name] = entry("differs", "size", source, info, "file")
            continue
        verdict = None
        recorded = records.get((info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns))
        if recorded and pin["sha256"] not in recorded:
            verdict = ("differs", "recorded")
        elif recorded and len(recorded) == 1:
            verdict = ("match", "recorded")
        if verdict is None and layout == "local-dir":
            metadata = download_metadata(ctx, path, name, info, pin)
            if metadata is not None:
                commits[metadata["commit"]] = commits.get(metadata["commit"], 0) + 1
                if metadata["evidence"] != "size":
                    verdict = (metadata["state"], metadata["evidence"])
        if (verdict is None and hashing and info.st_size <= options["small_file_bytes"]
                and ctx.get("hash_left", 0) >= info.st_size):
            ctx["hash_left"] -= info.st_size
            digest = guard(ctx, source, "hash", info)
            if digest is not None:
                verdict = ("match" if digest == pin["sha256"] else "differs", "hashed")
        found[name] = entry(*(verdict or ("size-only", "size")), source, info, "file")

    counts = dict.fromkeys(("match", "differs", "size-only", "missing", "incomplete", "outside", "network"), 0)
    for name in required:
        counts[(found.get(name) or {}).get("state", "missing")] += 1
    if commit is None and commits:
        commit = max(sorted(commits), key=lambda value: commits[value])
    result = {"path": path, "layout": layout, "commit": commit, "branches": [], "files": found, "counts": counts}
    if exact:
        allowed = set(required) | optional
        extra, links = [], []
        pending = [("", path)]
        while pending and len(extra) + len(links) < 50:
            relative, directory = pending.pop()
            listing = guard(ctx, directory, "scandir", options["directory_entries"])
            for item in sorted((listing or ([], False))[0], key=lambda value: value.name):
                name = relative + item.name
                if name in (".cache/huggingface", ".huggingface", ".git"):
                    continue
                try:
                    if item.is_symlink():
                        links.append(name)
                    elif item.is_dir(follow_symlinks=False):
                        pending.append((name + "/", directory + "/" + item.name))
                    elif not item.is_file(follow_symlinks=False):
                        links.append(name)
                    elif name not in allowed:
                        extra.append(name)
                except OSError:
                    continue
        result["extra"], result["symlinks"] = sorted(extra), sorted(links)
    return result


def walk(ctx, tiers, found, seconds, entries):
    """Bounded breadth-first walk for checkpoint folders and hub roots.

    ``tiers`` is a list of tiers, each a list of roots ``(path, depth, tag)``
    with resolved paths. Tiers are walked in order, breadth-first across the
    roots of a tier. A root's directories at depth <= ``depth`` below it are
    listed with ``os.scandir``; a child is descended only when it passes the
    guard, is a directory and not a symlink (by ``d_type``), and is not
    pruned. Each directory, by device and inode, is listed once. Within a
    directory, names that suggest model storage or contain a token of the
    repository name are queued first. After ``directory_entries`` entries a
    listing stops, and the index and pinned weight names are probed with
    ``lstat``.

    ``found(kind, path, tag)`` receives ``hub`` for a directory holding
    ``models--*`` entries, ``repo`` for a ``models--*`` root, ``folder`` for a
    directory holding a pinned weight name at its pinned size, the index at the
    pinned index size, or pinned names beside ``.git``, ``near`` for a
    directory with the index at another size beside a pinned weight name, the
    pinned ``config.json`` or the repository name in its path, and ``merge``
    when a directory already reported is reached from another root. It returns
    True when the directory must not be descended. A directory holding
    ``.sparkring-ignore`` is neither reported nor descended.

    The budget is ``seconds`` of wall clock, cut to the survey's overall
    deadline ``ctx["deadline"]``, and ``entries`` directory entries, checked
    every 256 entries; a walk that starts after the deadline visits nothing.
    Returns ``complete``, ``stopped`` (``time``, ``entries`` or None),
    ``entries``, ``unvisited`` (per root: a count, and the paths that are
    outside other accounts' homes) and ``large_directories``.
    """
    options = ctx["options"]
    pins = ctx["pins"]
    files = pins["files"]
    index = pins["index"]
    weights = [name for name in pins["weights"] if "/" not in name]
    config = files.get("config.json")
    limit = options["directory_entries"]
    owner, _, repository = pins["repository"].partition("/")
    tokens = ["model", "hf", "hugging", "checkpoint", "weight", owner.lower()]
    tokens += [part for part in repository.lower().replace("_", "-").split("-") if len(part) >= 5]
    pruned_names = {".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", "site-packages",
                    "dist-packages", ".npm", ".cargo", ".rustup", ".conda", "anaconda3", "miniconda3", "miniforge3",
                    "mambaforge", "micromamba", ".pyenv", ".nvm", ".vscode-server", ".cursor-server", "snap",
                    ".ollama", ".lmstudio", "xet", "chunk_cache", "snapshots", "blobs", ".no_exist", ".locks",
                    "lost+found", ".Trash", ".docker", ".nv", ".triton", ".nccl", ".m2", ".gradle"}
    pruned_tails = ("/.cache/pip", "/.cache/uv", "/.cache/torch", "/.cache/triton", "/.cache/vllm",
                    "/.cache/flashinfer", "/.cache/torch_extensions", "/.cache/huggingface/datasets",
                    "/.cache/huggingface/xet", "/.local/share/Trash", "/.local/share/uv",
                    "/.local/share/containers", "/.local/lib", "/go/pkg")
    pruned_paths = {"/srv/sparkring", "/var/lib/containerd", "/snap", ctx.get("docker_root") or "/var/lib/docker"}
    homes = ctx.get("homes") or []
    budget = {"entries": 0, "next": 256, "deadline": min(time.monotonic() + seconds, ctx.get("deadline", float("inf"))),
              "stopped": None, "large": 0}
    if time.monotonic() >= budget["deadline"]:
        budget["stopped"] = "time"
    marks = ctx.setdefault("ignore", {})
    visited = {}

    def spend(count):
        budget["entries"] += count
        if budget["entries"] >= budget["next"]:
            budget["next"] = (budget["entries"] // 256 + 1) * 256
            if budget["entries"] >= entries:
                budget["stopped"] = "entries"
            elif time.monotonic() >= budget["deadline"]:
                budget["stopped"] = "time"

    def pruned(name, child):
        return (name in pruned_names or name.startswith(".Trash-")
                or (name.startswith(".") and name.endswith(".sparkring"))
                or child in pruned_paths or child.endswith(pruned_tails))

    def shown(path):
        for row in homes:
            for base in (row["resolved"], row["home"]):
                if base not in ("", "/") and (path == base or path.startswith(base + "/")):
                    return row["kind"] == "operator"
        return True

    def size(path):
        info = guard(ctx, path, "lstat")
        return info.st_size if info is not None and stat.S_ISREG(info.st_mode) else None

    left = []
    for number, tier in enumerate(tiers):
        queue = [(path, 0, depth, tag, path) for path, depth, tag in tier]
        position = 0
        while position < len(queue) and not budget["stopped"]:
            path, level, depth, tag, origin = queue[position]
            position += 1
            here = tag if level == 0 else "folder"
            info = guard(ctx, path, "lstat")
            if info is None or not stat.S_ISDIR(info.st_mode):
                continue
            identity = (info.st_dev, info.st_ino)
            if identity in visited:
                if visited[identity]:
                    found("merge", visited[identity], here)
                continue
            visited[identity] = None
            base = path.rstrip("/")
            if base.rsplit("/", 1)[-1].startswith("models--"):
                if found("repo", path, here):
                    visited[identity] = path
                continue
            listing = guard(ctx, path, "scandir", limit)
            if listing is None:
                continue
            children, truncated = listing
            spend(len(children))
            names = {child.name: child for child in children}
            if truncated:
                budget["large"] += 1
                if guard(ctx, base + "/.sparkring-ignore", "lstat") is not None:
                    continue
                for name in [index, *weights]:
                    if name not in names and guard(ctx, base + "/" + name, "lstat") is not None:
                        names[name] = None
            else:
                marks[path] = ".sparkring-ignore" in names
            if ".sparkring-ignore" in names:
                continue
            if any(name.startswith("models--") for name in names):
                if found("hub", path, here):
                    visited[identity] = path
                continue
            sizes = {name: size(base + "/" + name) for name in [index, *weights] if name in names}
            hit = (sizes.get(index) == files[index]["size"]
                   or any(sizes.get(name) == files[name]["size"] for name in weights))
            if hit or (".git" in names and sizes):
                if found("folder", path, here):
                    visited[identity] = path
                    continue
            elif index in sizes:
                near = any(name in names for name in weights) or repository.lower() in path.lower()
                private = any(path == base or path.startswith(base + "/") for base in ctx.get("private", ()))
                if not near and not private and config is not None and "config.json" in names:
                    config_info = guard(ctx, base + "/config.json", "lstat")
                    if (config_info is not None and stat.S_ISREG(config_info.st_mode)
                            and config_info.st_size == config["size"]
                            and config["size"] <= options["small_file_bytes"]
                            and ctx.get("hash_left", 0) >= config["size"]):
                        ctx["hash_left"] -= config["size"]
                        near = guard(ctx, base + "/config.json", "hash", config_info) == config["sha256"]
                if near:
                    found("near", path, here)
            if level >= depth:
                continue
            # This directory and its ancestors passed the guard, and a directory
            # holding a marker is not descended, so a child needs its own guard
            # check only when it is a mount point. The check precedes is_dir,
            # which calls lstat when the file system reports no entry type.
            table = ctx["mount_index"]
            directories = []
            for name, child in names.items():
                path_of_child = base + "/" + name
                if child is None or (path_of_child in table and guard(ctx, path_of_child) is None):
                    continue
                try:
                    if child.is_symlink() or not child.is_dir(follow_symlinks=False):
                        continue
                except OSError:
                    continue
                if not pruned(name, path_of_child):
                    directories.append(name)
            for name in sorted(directories, key=lambda name: (not any(token in name.lower() for token in tokens),
                                                              name)):
                queue.append((base + "/" + name, level + 1, depth, tag, origin))
        if budget["stopped"]:
            left = queue[position:] + [(path, 0, depth, tag, path)
                                       for later in tiers[number + 1:] for path, depth, tag in later]
            break
    unvisited = {}
    for path, level, depth, tag, origin in left:
        slot = unvisited.setdefault(origin, {"count": 0})
        slot["count"] += 1
        if shown(path) and len(slot.setdefault("paths", [])) < 20:
            slot["paths"].append(path)
    for slot in unvisited.values():
        if not slot.get("paths"):
            slot.pop("paths", None)
    return {"complete": budget["stopped"] is None, "stopped": budget["stopped"], "entries": budget["entries"],
            "unvisited": unvisited, "large_directories": budget["large"]}


def survey(pins, options):
    """Survey this host for copies of the pinned checkpoint; never writes.

    ``pins`` is the pin manifest or its compact form; ``options`` is the
    document ``options()`` builds. Invalid inputs raise ``ValueError``; every
    other failure on a path, a record or Docker is recorded and the survey
    continues. Groups run in order: SparkRing's directories and records, and
    named paths; Docker, whose user-namespace setting decides linkability even
    when ``ignore_local`` ends the search after the first group; HF caches in
    every account's home; hub roots declared in environment and shell files;
    Docker's mounts, volumes and writable layers; then the bounded walk (tier
    B, tier C, then other mount points), repeated once with twice the budget
    when a budget runs out and ``retry`` is set.

    The whole survey has a deadline of three walk budgets plus the Docker
    allowance (75 s by default), so it ends well within the caller's SSH
    timeout. SparkRing's own directories and named paths are always examined;
    folders named in records, home caches, declared and container hub roots,
    repository folders and walk roots reached after the deadline are listed
    under ``search.unvisited`` and the search is reported incomplete, stopped
    by ``time``. Returns a ``sparkring-checkpoint-survey/v1`` document.
    """
    started = time.monotonic()

    def digest(value, length):
        return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdef" for c in value)

    def safe(name):
        return (isinstance(name, str) and not any(c in name for c in "\0\r\n\\")
                and all(part not in ("", ".", "..", ".cache", ".git") for part in name.split("/")))

    if not isinstance(pins, dict) or pins.get("schema") != "sparkring-checkpoint-pins/v1":
        raise ValueError("pins: expected sparkring-checkpoint-pins/v1")
    files = pins.get("files")
    if (not isinstance(pins.get("repository"), str) or pins["repository"].count("/") != 1
            or not digest(pins.get("revision"), 40) or not isinstance(files, dict) or not files):
        raise ValueError("pins: repository, revision or files are malformed")
    for name, entry in files.items():
        if (not safe(name) or not isinstance(entry, dict) or type(entry.get("size")) is not int
                or entry["size"] <= 0 or not digest(entry.get("sha256"), 64) or not digest(entry.get("git_blob"), 40)
                or ("xet_hash" in entry and not digest(entry["xet_hash"], 64))):
            raise ValueError(f"pins: unsafe or malformed entry {name!r}")
    # The compact form keeps optional names but drops their entries from ``files``.
    optional = pins.get("optional") or []
    if (not isinstance(optional, list) or not all(safe(name) for name in optional)
            or pins.get("index") not in files or pins["index"] in optional
            or not isinstance(pins.get("weights"), list) or not pins["weights"]
            or not set(pins["weights"]) <= set(files) - set(optional)):
        raise ValueError("pins: index, weights or optional names are malformed")
    defaults = {"root": "/", "operator": "root", "named": [], "ignore_local": False, "walk_seconds": 20,
                "walk_entries": 400000, "retry": True, "docker_seconds": 15, "directory_entries": 10000,
                "small_file_bytes": 64 << 20, "hash_bytes": 256 << 20, "cache": None}
    if not isinstance(options, dict) or set(options) - set(defaults) - {"owned"}:
        raise ValueError("options: unknown keys")
    options = {**defaults, **options}
    if (not isinstance(options.get("owned"), str) or not options["owned"].startswith("/")
            or not (options["cache"] is None or (isinstance(options["cache"], str) and options["cache"].startswith("/")))
            or not isinstance(options["root"], str) or not options["root"]
            or not isinstance(options["operator"], str) or not options["operator"] or ":" in options["operator"]
            or not isinstance(options["named"], list)
            or not all(isinstance(path, str) and path.startswith("/") for path in options["named"])
            or not isinstance(options["ignore_local"], bool) or not isinstance(options["retry"], bool)
            or any(type(options[key]) not in (int, float) or options[key] < 0
                   for key in ("walk_seconds", "docker_seconds"))
            or any(type(options[key]) is not int or options[key] <= 0
                   for key in ("walk_entries", "directory_entries", "small_file_bytes", "hash_bytes"))):
        raise ValueError("options: malformed values")

    root = options["root"]
    ctx = {"root": "" if root == "/" else root.rstrip("/"), "mounts": mount_table(root), "pins": pins,
           "options": options, "hash_left": options["hash_bytes"], "errors": [], "unreadable": 0,
           "deadline": started + 3 * options["walk_seconds"] + options["docker_seconds"]}
    required = sorted(name for name in files if name not in set(optional))
    weights = set(pins["weights"])
    index = pins["index"]
    repository_name = pins["repository"].partition("/")[2].lower()
    local = ("ext2", "ext3", "ext4", "xfs", "btrfs", "f2fs", "zfs", "exfat", "vfat", "ntfs3", "fuseblk")
    strong = ("recorded", "hashed", "hub-named", "hub-metadata")
    homes = accounts(ctx)
    ctx["homes"] = homes
    ctx["private"] = [row[key] for row in homes if row["kind"] == "private" for key in ("resolved", "home")]
    ctx["named"] = []
    spots = sorted([(row[key], row) for row in homes for key in ("resolved", "home")],
                   key=lambda item: -len(item[0]))
    candidates, notes, named, hubs, decided = {}, {}, [], {}, {}

    def home_of(path):
        for base, row in spots:
            if base not in ("", "/") and (path == base or path.startswith(base + "/")):
                return {"account": row["account"], "kind": row["kind"]}
        return None

    late = {}

    def expired(path):
        """Whether the overall deadline has passed; ``path`` is then listed as unvisited, as the walk lists roots."""
        if time.monotonic() < ctx["deadline"]:
            return False
        top = "/" + path.strip("/").split("/", 1)[0]
        slot = late.setdefault(top, {"count": 0})
        slot["count"] += 1
        home = home_of(path)
        if (home is None or home["kind"] == "operator") and len(slot.setdefault("paths", [])) < 20:
            slot["paths"].append(path)
        return True

    def verdict(result, lenient):
        # The candidate rule of the equivalence section: the index or a weight
        # file identified by content or client, or pinned weight names at their
        # pinned sizes. SparkRing directories and named paths are candidates
        # whenever any file matches.
        found = result["files"]
        if lenient:
            if any(value.get("state") in ("match", "size-only") for value in found.values()):
                return "candidate"
        entry = found.get(index) or {}
        if entry.get("state") == "match" and entry.get("evidence") in strong:
            return "candidate"
        for name in weights:
            value = found.get(name) or {}
            if value.get("state") == "size-only" or (value.get("state") == "match"
                                                      and value.get("evidence") in strong):
                return "candidate"
        if entry.get("state") == "differs" and (any(name in found for name in weights)
                                                or (found.get("config.json") or {}).get("state") == "match"
                                                or repository_name in result["path"].lower()):
            return "near"
        return None

    def note(path, reason, home, extra=None):
        if path not in notes and path not in candidates:
            notes[path] = {"path": path, "reason": reason, "home": home, **(extra or {})}

    def add(result, tag, named_path=None, sparkring=False):
        path = result["path"]
        if path == owned["resolved"]:
            return None
        if path in candidates:
            known = candidates[path]
            known["found_by"].add(tag)
            if named_path is not None:
                known.setdefault("named_as", named_path)
                for key in ("extra", "symlinks", "exact"):
                    if key in result:
                        known[key] = result[key]
            return "candidate"
        decision = verdict(result, sparkring or named_path is not None)
        home = home_of(path)
        if decision == "near":
            note(path, "another checkpoint: its index differs from the pinned revision", home)
            return "near"
        if decision is None:
            return None
        if home is not None and home["kind"] == "private" and named_path is None:
            note(path, f"in {home['account']}'s home, another account", home,
                 {"layout": result["layout"], "counts": result["counts"]})
            return "private"
        notes.pop(path, None)
        mount = guard(ctx, path, "mount")
        info = guard(ctx, path, "lstat")
        result.update(found_by={tag}, home=home, sparkring=sparkring, mount_id=mount["id"] if mount else None,
                      device=info.st_dev if info is not None else None,
                      rotational=bool(mount and mount["rotational"]))
        if named_path is not None:
            result["named_as"] = named_path
        candidates[path] = result
        return "candidate"

    def examine_hub(path, tag, only=None):
        key = (path, only)
        if key in hubs:
            for member in hubs[key]:
                candidates[member]["found_by"].add(tag)
            return True
        hubs[key] = [result["path"] for result in hub_repository_files(ctx, path, only)
                     if add(result, tag) == "candidate"]
        return True

    def examine_folder(path, tag, layout=None, commit=None, sparkring=False):
        if path in candidates:
            candidates[path]["found_by"].add(tag)
            return True
        if path not in decided:
            outcome = add(classify_folder(ctx, path, layout, commit), tag, sparkring=sparkring)
            decided[path] = outcome in ("candidate", "private")
        return decided[path]

    def resolved_directory(path):
        state, resolved, info = safe_resolve(ctx, path)
        return resolved if state == "ok" and stat.S_ISDIR(info.st_mode) else None

    # Group 1: SparkRing's own directories and records.
    evidence = sparkring_evidence(ctx)
    owned = describe = {"path": "/" + "/".join(part for part in options["owned"].split("/") if part),
                        "resolved": None, "probe_path": None, "mount_point": None, "mount_id": None, "device": None,
                        "fstype": None, "free_bytes": None, "state": "absent", "files": {}}
    mount = guard(ctx, describe["path"], "mount")
    head, _, tail = describe["path"].rpartition("/")
    state, where, info = safe_resolve(ctx, head or "/")
    probe = None
    if state == "missing":
        probe = where.rpartition("/")[0] or "/"
        info = guard(ctx, probe, "lstat")
    elif state == "ok" and stat.S_ISDIR(info.st_mode):
        probe = describe["resolved"] = where.rstrip("/") + "/" + tail
        target = guard(ctx, probe, "lstat")
        if target is None:
            probe = where
        else:
            info = target
    if probe is not None and info is not None:
        mount = guard(ctx, probe, "mount")
        # statvfs follows a symlink, so a symlinked D is measured at its resolved parent.
        space = (guard(ctx, where if stat.S_ISLNK(info.st_mode) else probe, "statvfs")
                 if hasattr(os, "statvfs") else None)
        describe.update(probe_path=probe, device=info.st_dev,
                        free_bytes=space.f_bavail * space.f_frsize if space is not None else None)
        if probe == describe["resolved"]:
            if stat.S_ISLNK(info.st_mode):
                describe["state"] = "symlink"
            elif not stat.S_ISDIR(info.st_mode):
                describe["state"] = "foreign"
            elif mount is not None and mount["path"] == probe:
                describe["state"] = "mount-point"
            elif probe in evidence["sparkring"]:
                describe["state"] = "owned"
                describe["files"] = classify_folder(ctx, probe, "sparkring", pins["revision"])["files"]
            elif safe_resolve(ctx, where.rstrip("/") + "/." + tail + ".sparkring/owner.json",
                              inside=where)[0] == "ok":
                describe["state"] = "replaced"
            else:
                listing = guard(ctx, probe, "scandir", 1)
                describe["state"] = "empty" if listing is not None and not listing[0] else "foreign"
    if mount is not None:
        describe.update(mount_point=mount["path"], mount_id=mount["id"], fstype=mount["type"])
    # The deployment's compile cache: the one it names, or the cluster cache beside SparkRing's directories.
    parts = describe["path"].split("/")
    cache = options["cache"]
    if cache is None and len(parts) > 3 and parts[1:3] == ["srv", "sparkring"]:
        cache = "/srv/sparkring/" + parts[3] + "/cache"
    if cache is not None:
        state, where, info = safe_resolve(ctx, cache)
        if state == "missing":
            info = guard(ctx, where.rpartition("/")[0] or "/", "lstat")
        describe["cache_path"] = cache
        describe["cache_device"] = info.st_dev if state in ("ok", "missing") and info is not None else None
    for path in evidence["sparkring"]:
        if path != describe["resolved"]:
            examine_folder(path, "sparkring", "sparkring", pins["revision"], True)

    # Named paths: classified per Spark; a named copy is used wherever it is.
    for given in options["named"]:
        state, resolved, info = safe_resolve(ctx, given)
        report = {"path": given, "resolved": resolved if state == "ok" else None}
        named.append(report)
        if state in ("network", "ignored", "unsupported"):
            report["state"] = state
            continue
        if state != "ok" or not stat.S_ISDIR(info.st_mode):
            report["state"] = "absent"
            continue
        if guard(ctx, resolved + "/.sparkring-ignore") is None and ctx.get("refusal") == "ignored":
            report["state"] = "ignored"
            continue
        ctx["named"].append(resolved)
        head, _, tail = resolved.rpartition("/")
        above, _, parent = head.rpartition("/")
        if tail.startswith("models--"):
            results = hub_repository_files(ctx, head or "/", tail)
        elif parent == "snapshots" and above.rpartition("/")[2].startswith("models--"):
            flat = False
            for name in required:
                item = guard(ctx, resolved + "/" + name, "lstat") if "/" not in name else None
                flat = flat or (item is not None and stat.S_ISREG(item.st_mode))
            if flat:
                results = [classify_folder(ctx, resolved, "hf-snapshot", tail, exact=True)]
            else:
                repository_head, _, repository_tail = above.rpartition("/")
                results = hub_repository_files(ctx, repository_head or "/", repository_tail)
        else:
            listing = guard(ctx, resolved, "scandir", options["directory_entries"])
            if listing is not None and any(item.name.startswith("models--") for item in listing[0]):
                results = hub_repository_files(ctx, resolved)
            else:
                results = [classify_folder(ctx, resolved, exact=True)]
        states = []
        for result in results:
            if "extra" in result:
                result["exact"] = (not result["extra"] and not result["symlinks"] and all(
                    (result["files"].get(name) or {}).get("state") in ("match", "size-only") for name in required))
                report["extra"], report["symlinks"] = result["extra"], result["symlinks"]
            add(result, "named", given)
            states += [value.get("state") for value in result["files"].values()]
        if len(results) == 1 and results[0].get("exact"):
            report["state"] = "exact"
        elif "differs" in states:
            report["state"] = "differs"
        elif any(value in ("match", "size-only") for value in states):
            report["state"] = "partial"
        else:
            report["state"] = "absent"

    sources = docker_sources(ctx)
    docker = {"userns": sources["userns"], "driver": sources["driver"], "root": sources["root"], "device": None}
    if sources["root"]:
        state, where, info = safe_resolve(ctx, sources["root"])
        if state == "ok":
            ctx["docker_root"] = where
            docker["device"] = info.st_dev
    search = {"complete": True, "stopped": None, "passes": 0, "entries": 0, "unvisited": {}, "large_directories": 0}
    if not options["ignore_local"]:
        for path, tag in evidence["folders"]:
            if expired(path):
                continue
            resolved = resolved_directory(path)
            if resolved is not None:
                examine_folder(resolved, tag)
        # Group 2: Hugging Face caches and download folders in every account's home.
        excluded = {"hub", "xet", "datasets", "token", "stored_tokens", "assets", "modules", "accelerate"}
        extra_roots = []
        for row in homes:
            for suffix in ("/.cache/huggingface/hub", "/.cache/nim/huggingface/hub"):
                if expired(row["resolved"] + suffix):
                    continue
                resolved = resolved_directory(row["resolved"] + suffix)
                if resolved is not None:
                    examine_hub(resolved, "hub")
            if expired(row["resolved"] + "/.cache/huggingface"):
                continue
            cache = resolved_directory(row["resolved"] + "/.cache/huggingface")
            listing = guard(ctx, cache, "scandir", 1000) if cache else None
            for item in sorted((listing or ([], False))[0], key=lambda value: value.name):
                if item.name not in excluded and not item.name.startswith("."):
                    extra_roots.append((cache + "/" + item.name, 4, "folder"))
        # Group 3: hub roots declared in environment, shell and systemd files.
        for path in declared_hub_roots(ctx, homes):
            if expired(path):
                continue
            resolved = resolved_directory(path)
            if resolved is not None:
                examine_hub(resolved, "declared")
        # Group 4: Docker containers' hub paths; their mounts and volumes are walk roots below.
        for path, tag in sources["hubs"]:
            if expired(path):
                continue
            resolved = resolved_directory(path)
            if resolved is not None:
                examine_hub(resolved, tag)
        # Group 5: folders, walked in priority tiers.
        system = ("/bin", "/boot", "/dev", "/etc", "/lib", "/lib32", "/lib64", "/libx32", "/proc", "/run", "/sbin",
                  "/sys", "/usr", "/var/run", "/snap", "/tmp", "/var")
        tier_b = [(path, 4, "folder") for path in ("/var/tmp/models", "/models", "/srv/models", "/data/models",
                                                  "/opt/models", "/workspace/models", "/mnt/models")]
        tier_b += [(row["resolved"] + "/models", 4, "folder") for row in homes] + extra_roots
        tier_c = [(path, 6, "folder") for path in ("/home", "/root", "/data", "/srv", "/mnt", "/media",
                                                  "/workspace", "/scratch")]
        tier_c += [("/opt", 4, "folder"), ("/var/tmp", 4, "folder")]
        top = {"bin", "boot", "cdrom", "dev", "etc", "lib", "lib32", "lib64", "libx32", "lost+found", "proc", "run",
               "sbin", "snap", "sys", "tmp", "usr", "var", "home", "root", "data", "srv", "mnt", "media",
               "workspace", "scratch", "opt", "models"}
        listing = guard(ctx, "/", "scandir", 1000)
        for item in sorted((listing or ([], False))[0], key=lambda value: value.name):
            if item.name not in top and not item.name.startswith("."):
                tier_c.append(("/" + item.name, 6, "folder"))
        for path, priority, tag in sources["roots"]:
            clean = "/" + "/".join(part for part in path.split("/") if part)
            if clean == "/" or clean in system or clean.startswith(tuple(value + "/" for value in system[:-2])):
                continue
            (tier_b if priority == 1 else tier_c).append((clean, 4, tag))
        mounted = [(entry["path"], 6, "folder") for entry in ctx["mounts"]
                   if entry["type"] in local and entry["path"] != "/"
                   and not entry["path"].startswith(("/boot", "/var/lib/docker", "/snap/", "/srv/sparkring/"))
                   and entry["path"] not in ("/efi", "/var/lib/containerd", "/snap", "/srv/sparkring")]
        tiers, seen = [], set()
        for tier in (tier_b, tier_c, mounted):
            roots = []
            for path, depth, tag in tier:
                resolved = resolved_directory(path)
                if resolved is not None and resolved not in seen:
                    seen.add(resolved)
                    roots.append((resolved, depth, tag))
            tiers.append(roots)

        def found(kind, path, tag):
            if kind == "merge":
                if path in candidates:
                    candidates[path]["found_by"].add(tag)
                for key, members in hubs.items():
                    if key[0] == path or key == (path.rpartition("/")[0] or "/", path.rpartition("/")[2]):
                        for member in members:
                            candidates[member]["found_by"].add(tag)
                return True
            if kind == "hub":
                return examine_hub(path, tag)
            if kind == "repo":
                head, _, tail = path.rpartition("/")
                return examine_hub(head or "/", tag, tail)
            if kind == "near":
                note(path, "another checkpoint: its index differs from the pinned revision", home_of(path))
                return False
            return examine_folder(path, tag)

        seconds, limit = options["walk_seconds"], options["walk_entries"]
        while True:
            search["passes"] += 1
            outcome = walk(ctx, tiers, found, seconds, limit)
            search["entries"] += outcome["entries"]
            search["large_directories"] += outcome["large_directories"]
            search.update(complete=outcome["complete"], stopped=outcome["stopped"], unvisited=outcome["unvisited"])
            if (outcome["complete"] or not options["retry"] or search["passes"] == 2
                    or time.monotonic() >= ctx["deadline"]):
                break
            seconds, limit = seconds * 2, limit * 2

    for path in ctx.get("late", []):
        expired(path)
    if late:
        # Roots the deadline left unexamined, beside what the walk left.
        for top, slot in late.items():
            known = search["unvisited"].setdefault(top, {"count": 0})
            known["count"] += slot["count"]
            if slot.get("paths"):
                known["paths"] = (known.get("paths", []) + slot["paths"])[:20]
        search.update(complete=False, stopped="time")
    listed = []
    for path in sorted(candidates):
        value = candidates[path]
        value["found_by"] = sorted(value["found_by"])
        listed.append(value)
    ordered = sorted(notes.values(), key=lambda item: (not item["reason"].startswith("in "), item["path"]))
    search.update(seconds=round(time.monotonic() - started, 1),
                  skipped_mounts=sorted(ctx.get("skipped", {}).values(), key=lambda item: item["path"]),
                  unreadable=ctx.get("unreadable", 0), errors=list(ctx.get("errors", [])))
    return {"schema": "sparkring-checkpoint-survey/v1", "host": os.uname().nodename if hasattr(os, "uname") else "",
            "repository": pins["repository"], "revision": pins["revision"], "operator": options["operator"],
            "docker": docker, "owned": owned, "search": search, "candidates": listed, "not_used": ordered[:5],
            "named": named}


SHIPPED = (mount_table, guard, safe_resolve, accounts, declared_hub_roots, docker_sources, hub_repository_files,
           download_metadata, sparkring_evidence, classify_folder, walk, survey)


def compact(pins):
    """The pin manifest reduced to what the survey reads, as canonical JSON text.

    Optional files keep only their names; required files keep their size,
    SHA-256, git blob id and LFS and Xet identities. For the Qwen manifest the
    text is about 12 KB.
    """
    optional = sorted(pins.get("optional") or [])
    files = {name: {key: value for key, value in entry.items()
                    if key in ("size", "sha256", "git_blob", "lfs", "xet_hash")}
             for name, entry in pins["files"].items() if name not in optional}
    document = {"schema": pins["schema"], "repository": pins["repository"], "revision": pins["revision"],
                "index": pins["index"], "weights": list(pins["weights"]), "optional": optional, "files": files}
    return json.dumps(document, sort_keys=True, separators=(",", ":"))


def options(*, owned, operator=None, named=(), ignore_local=False, root="/", walk_seconds=20, walk_entries=400000,
            retry=True, docker_seconds=15, directory_entries=10000, small_file_bytes=64 << 20,
            hash_bytes=256 << 20, cache=None):
    """The survey's options document.

    ``owned`` is the SparkRing checkpoint directory this installation uses.
    ``operator`` defaults to ``SUDO_USER``, or ``root`` when that is unset.
    ``named`` holds the ``--model-path`` values that apply to this Spark.
    ``cache`` is the deployment's compile cache, whose filesystem decides where
    the plan counts the cache allowance (default: the cluster cache
    ``/srv/sparkring/<cluster>/cache`` beside ``owned``). ``root`` prefixes
    every host path, for tests against a fixture tree. The budgets bound the
    walk (``walk_seconds`` and ``walk_entries``, doubled once on ``retry``),
    Docker (``docker_seconds`` for all five calls; 0 skips Docker), the whole
    survey (three walk budgets plus the Docker allowance), directory listings
    (``directory_entries``) and hashing (files of at most ``small_file_bytes``,
    ``hash_bytes`` in total).
    """
    return {"root": str(root), "operator": operator or os.environ.get("SUDO_USER") or "root",
            "named": [str(path) for path in named], "ignore_local": bool(ignore_local), "owned": str(owned),
            "walk_seconds": walk_seconds, "walk_entries": walk_entries, "retry": bool(retry),
            "docker_seconds": docker_seconds, "directory_entries": directory_entries,
            "small_file_bytes": small_file_bytes, "hash_bytes": hash_bytes,
            "cache": str(cache) if cache else None}


def probe_source(pins, options):
    """Self-contained Python source that prints this host's survey as JSON.

    The text is the header imports, the source of every function in
    ``SHIPPED`` and a call of ``survey`` with the compact pins and ``options``
    as JSON literals. Node A runs it on each rank as root with
    ``python3 -I -B -``, sending it on stdin.
    """
    header = "import hashlib, json, os, stat, subprocess, sys, time\n"
    body = "\n\n".join(inspect.getsource(function) for function in SHIPPED)
    call = (f"print(json.dumps(survey(json.loads({compact(pins)!r}), "
            f"json.loads({json.dumps(options, sort_keys=True)!r}))))\n")
    return header + "\n\n" + body + "\n\n" + call
