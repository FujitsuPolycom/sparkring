"""Checkpoint placement primitives against real files, hard links and a synthetic mount table."""
import ast
import errno
import hashlib
import inspect
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import threading
import time

import pytest

from runtime.host import checkpoint_place as place

REPOSITORY = "local-inference-lab/Qwen3.8-Flash-Next-NVFP4"
REVISION = "629bc3218833a38b475b719f34aa571666f4a03e"
WEIGHT = "model-00001-of-00002.safetensors"
SECOND = "model-00002-of-00002.safetensors"
linux = pytest.mark.skipif(not sys.platform.startswith("linux"),
                           reason="placement uses Linux hard links, /proc/self/fd, O_NOFOLLOW, O_NOATIME and flock")


def escape(path):
    return "".join(f"\\{ord(character):03o}" if character in " \t\n\\" else character for character in path)


def mount_line(mount_id, parent, point, kind, device="259:2"):
    return f"{mount_id} {parent} {device} / {escape(point)} rw,relatime shared:1 - {kind} /dev/nvme0n1p2 rw\n"


@pytest.fixture
def mount_table(tmp_path, monkeypatch):
    """A synthetic mount table whose root filesystem is ext4; tests append entries."""
    path = tmp_path / "mountinfo"
    path.write_text(mount_line(29, 1, "/", "ext4"))
    monkeypatch.setattr(place, "MOUNTINFO", str(path))
    return path


@pytest.fixture(autouse=True)
def records(tmp_path, monkeypatch):
    directory = tmp_path / "records"
    monkeypatch.setattr(place, "RECORDS", str(directory))
    return directory


def tree_state(root):
    """Every entry at and below ``root`` with its metadata and content, excluding access and change times."""
    root = str(root)
    state = {}
    paths = [root]
    for directory, directories, files in os.walk(root):
        paths.extend(os.path.join(directory, name) for name in directories + files)
    for path in paths:
        info = os.lstat(path)
        if stat.S_ISREG(info.st_mode):
            content = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        elif stat.S_ISLNK(info.st_mode):
            content = os.readlink(path)
        else:
            content = None
        state[os.path.relpath(path, root)] = (info.st_mode, info.st_uid, info.st_size, info.st_ino, info.st_nlink,
                                              info.st_mtime_ns, content)
    return state


def digest(content):
    return hashlib.sha256(content).hexdigest()


def checkpoint(tmp_path):
    return str(tmp_path / "srv/tp2/checkpoints/local-inference-lab--Qwen3.8-Flash-Next-NVFP4" / REVISION)


def user_file(tmp_path, name, content):
    path = tmp_path / "user" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return str(path)


def verified_link(claimed, journal, path, name, content):
    """Open, hash and link ``path`` the way model adoption does."""
    info = os.lstat(path)
    fd, before = place.open_source(path, [info.st_dev, info.st_ino], len(content))
    try:
        measured = place.hash_descriptor(fd, len(content))
        assert measured == digest(content)
        return place.place_link(claimed.dir_fd, fd, name, journal, sha256=measured, before=before, source=path)
    finally:
        os.close(fd)


def refused_without_blocking(call):
    outcome = {}

    def attempt():
        try:
            call()
        except ValueError as error:
            outcome["refused"] = str(error)
        except BaseException as error:  # reported by the assertion below
            outcome["other"] = error

    thread = threading.Thread(target=attempt, daemon=True)
    thread.start()
    thread.join(5)
    assert not thread.is_alive(), "opening the source blocked"
    assert "refused" in outcome, outcome
    return outcome["refused"]


CLAIM_CASES = {
    "absent": None,
    "empty": None,
    "state directory without owner.json": None,
    "non-empty without owner record": "is not empty and was not created by SparkRing",
    "state directory without owner.json, non-empty": "is not empty and was not created by SparkRing",
    "replaced identity": "was replaced after SparkRing created it",
    "symlink": "symlink",
    "mount point": "is a mount point",
    # Docker and rsync name the staging directories by path, so no other account may rename or replace them.
    "empty, writable by other accounts": "belongs to another account or other accounts can write to it",
    "parent writable by other accounts": "can be changed by other accounts",
    "sticky parent writable by other accounts": None,
}


@linux
@pytest.mark.parametrize("case", CLAIM_CASES)
def test_claim_only_absent_or_empty_directories(case, tmp_path, mount_table):
    path = checkpoint(tmp_path)
    parent = os.path.dirname(path)
    state = os.path.join(parent, "." + REVISION + ".sparkring")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    if case != "absent":
        os.makedirs(parent)
    if case in ("empty", "mount point", "empty, writable by other accounts"):
        os.mkdir(path)
    if case == "empty, writable by other accounts":
        os.chmod(path, 0o777)
    if case == "parent writable by other accounts":
        os.chmod(parent, 0o777)
    if case == "sticky parent writable by other accounts":
        os.chmod(parent, 0o1777)
    if case.startswith("state directory"):
        os.mkdir(state, 0o700)
    if case in ("non-empty without owner record", "state directory without owner.json, non-empty"):
        os.mkdir(path)
        Path(path, "config.json").write_text("{}")
    if case == "replaced identity":
        with place.claim(path, REPOSITORY, REVISION):
            pass
        os.rename(path, str(elsewhere / "moved"))
        os.mkdir(path)
    if case == "symlink":
        os.symlink(elsewhere, path)
    if case == "mount point":
        with open(mount_table, "a") as stream:
            stream.write(mount_line(40, 29, path, "ext4", "259:3"))

    refusal = CLAIM_CASES[case]
    if refusal is not None:
        before = tree_state(tmp_path / "srv"), tree_state(elsewhere)
        with pytest.raises(ValueError, match=refusal):
            place.claim(path, REPOSITORY, REVISION)
        assert (tree_state(tmp_path / "srv"), tree_state(elsewhere)) == before
        return

    existing = os.lstat(path) if os.path.exists(path) else None
    with place.claim(path, REPOSITORY, REVISION) as claimed:
        assert claimed.action == ("claimed" if case == "empty" else "created")
        info = os.lstat(path)
        if existing is not None:
            assert (info.st_dev, info.st_ino) == (existing.st_dev, existing.st_ino)
        assert claimed.identity == [info.st_dev, info.st_ino] and claimed.state == state
        assert json.loads(Path(state, "owner.json").read_text()) == {
            "schema": "sparkring-checkpoint-owner/v1", "repository": REPOSITORY, "revision": REVISION,
            "path": path, "directory": [info.st_dev, info.st_ino]}
        assert json.loads(Path(state, "journal.json").read_text()) == {
            "schema": "sparkring-checkpoint-journal/v1", "files": {}, "directories": []}
        assert stat.S_IMODE(os.lstat(state).st_mode) == 0o700 and os.listdir(path) == []
        with pytest.raises(ValueError, match="Another SparkRing operation is changing"):
            place.claim(path, REPOSITORY, REVISION)
    with place.claim(path, REPOSITORY, REVISION) as again:
        assert again.action == "verified" and again.identity == [info.st_dev, info.st_ino]
    with pytest.raises(ValueError, match="not local-inference-lab/Qwen3.8-Flash-Next-NVFP4 at 0000"):
        place.claim(path, REPOSITORY, "0" * 40)


@linux
def test_links_come_from_the_hashed_descriptor(tmp_path, mount_table, monkeypatch):
    original = b"pinned weights " * 4096
    path = user_file(tmp_path, WEIGHT, original)
    info = os.lstat(path)
    with place.claim(checkpoint(tmp_path), REPOSITORY, REVISION) as claimed:
        journal = place.journal_load(claimed)
        fd, before = place.open_source(path, [info.st_dev, info.st_ino], len(original))
        try:
            # The owner renames the file away and writes other bytes at the path after SparkRing opened it.
            os.rename(path, path + ".previous")
            Path(path).write_bytes(b"x" * len(original))
            measured = place.hash_descriptor(fd, len(original))
            assert measured == digest(original)
            placed = place.place_link(claimed.dir_fd, fd, WEIGHT, journal, sha256=measured, before=before, source=path)
        finally:
            os.close(fd)
        target = os.path.join(claimed.path, WEIGHT)
        assert Path(target).read_bytes() == original and Path(path).read_bytes() == b"x" * len(original)
        linked = os.lstat(target)
        assert [linked.st_dev, linked.st_ino] == [info.st_dev, info.st_ino] and placed == place.stats(linked)
        entry = place.journal_load(claimed).files[WEIGHT]
        assert (entry["state"], entry["identity"], entry["origin"], entry["source"], entry["sha256"]) == (
            "placed", [info.st_dev, info.st_ino], "link", path, digest(original))

        # A file unlinked after it was opened has no name left to link from, and nothing is placed.
        other = user_file(tmp_path, SECOND, original)
        other_info = os.lstat(other)
        fd, before = place.open_source(other, [other_info.st_dev, other_info.st_ino], len(original))
        try:
            os.unlink(other)
            with pytest.raises(FileNotFoundError):
                place.place_link(claimed.dir_fd, fd, SECOND, journal,
                                 sha256=place.hash_descriptor(fd, len(original)), before=before, source=other)
        finally:
            os.close(fd)
        assert not os.path.lexists(os.path.join(claimed.path, SECOND))
        assert SECOND not in place.journal_load(claimed).files

    # A FIFO, a symlink to /dev/zero and a device are refused without blocking.
    fifo = str(tmp_path / "user/fifo.safetensors")
    os.mkfifo(fifo)
    zero = str(tmp_path / "user/zero.safetensors")
    os.symlink("/dev/zero", zero)
    for candidate in (fifo, zero, "/dev/null"):
        assert "not a regular file" in refused_without_blocking(lambda: place.open_source(candidate, None, 4096))
    # The descriptor check still refuses a FIFO or device that replaced a regular file after the path check.
    lstat, regular = os.lstat, os.lstat(target)
    monkeypatch.setattr(place.os, "lstat", lambda name, *args, **kwargs: regular if name in (fifo, "/dev/null")
                        else lstat(name, *args, **kwargs))
    for candidate in (fifo, "/dev/null"):
        refused_without_blocking(lambda: place.open_source(candidate, None, regular.st_size))
    monkeypatch.setattr(place.os, "lstat", lstat)
    assert Path(zero).is_symlink() and stat.S_ISFIFO(os.lstat(fifo).st_mode)


@linux
def test_a_file_rewritten_after_hashing_is_not_placed_even_with_its_old_modification_time(tmp_path, mount_table):
    original = b"pinned weights " * 4096
    path = user_file(tmp_path, WEIGHT, original)
    info = os.lstat(path)
    with place.claim(checkpoint(tmp_path), REPOSITORY, REVISION) as claimed:
        journal = place.journal_load(claimed)
        fd, before = place.open_source(path, [info.st_dev, info.st_ino], len(original))
        try:
            measured = place.hash_descriptor(fd, len(original))
            # Another writer changes bytes in place after they were hashed and restores the
            # modification time; only the change time shows it, so the descriptor is read again.
            time.sleep(0.05)
            with open(path, "r+b") as stream:
                stream.write(b"X")
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            with pytest.raises(ValueError, match="changed after SparkRing opened it; it was not placed"):
                place.place_link(claimed.dir_fd, fd, WEIGHT, journal, sha256=measured, before=before, source=path)
        finally:
            os.close(fd)
        assert not os.path.lexists(os.path.join(claimed.path, WEIGHT))
        assert place.journal_load(claimed).files == {}


@linux
def test_directories_for_nested_names_are_journaled_before_they_are_created(tmp_path, mount_table, monkeypatch):
    content = b"nested weights " * 64
    path = user_file(tmp_path, "audio/" + WEIGHT, content)
    target = checkpoint(tmp_path)
    journal_file = Path(os.path.dirname(target), "." + REVISION + ".sparkring", "journal.json")
    seen, mkdir = [], os.mkdir

    def recording(name, *args, **kwargs):
        if name == "audio" and kwargs.get("dir_fd") is not None:
            seen.append(json.loads(journal_file.read_text())["directories"])
        return mkdir(name, *args, **kwargs)
    with place.claim(target, REPOSITORY, REVISION) as claimed:
        journal = place.journal_load(claimed)
        monkeypatch.setattr(os, "mkdir", recording)
        verified_link(claimed, journal, path, "audio/" + WEIGHT, content)
        monkeypatch.setattr(os, "mkdir", mkdir)
    assert seen == [["audio"]]
    with place.claim(target, REPOSITORY, REVISION) as claimed:
        assert place.journal_load(claimed).directories == {"audio"}


@linux
def test_a_placed_file_that_changes_while_it_is_checked_is_removed(tmp_path, mount_table, monkeypatch):
    content = b"pinned weights " * 4096
    path = user_file(tmp_path, WEIGHT, content)
    with place.claim(checkpoint(tmp_path), REPOSITORY, REVISION) as claimed:
        journal = place.journal_load(claimed)
        verified_link(claimed, journal, path, WEIGHT, content)
        time.sleep(0.05)
        os.utime(path)  # the stats no longer equal the journal's, so the check reads the file again
        real = place.hash_descriptor

        def rewritten(fd, size):
            value = real(fd, size)
            time.sleep(0.05)
            info = os.stat(path)
            with open(path, "r+b") as stream:
                stream.write(b"X")
            os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))
            return value
        monkeypatch.setattr(place, "hash_descriptor", rewritten)
        # The bytes read matched the pin, but the change time moved while they were read.
        assert place.check_placed(claimed.dir_fd, WEIGHT, journal, len(content), digest(content)) == "removed"
        assert not os.path.lexists(os.path.join(claimed.path, WEIGHT)) and os.path.exists(path)


@linux
def test_final_names_are_never_replaced(tmp_path, mount_table):
    weights = b"a" * 8192
    config = b'{"model_type": "qwen3_8"}\n'
    first = user_file(tmp_path, "one/" + WEIGHT, weights)
    second = user_file(tmp_path, "two/" + WEIGHT, weights)
    source = user_file(tmp_path, "one/config.json", config)
    with place.claim(checkpoint(tmp_path), REPOSITORY, REVISION) as claimed:
        journal = place.journal_load(claimed)
        foreign = Path(claimed.path, "config.json")
        foreign.write_bytes(b"{}")
        kept = os.lstat(foreign).st_ino
        with pytest.raises(ValueError, match="never replaces a name"):
            verified_link(claimed, journal, source, "config.json", config)
        assert "config.json" not in place.journal_load(claimed).files

        verified_link(claimed, journal, first, WEIGHT, weights)
        with pytest.raises(ValueError, match="already recorded for another file"):
            verified_link(claimed, journal, second, WEIGHT, weights)
        assert os.lstat(os.path.join(claimed.path, WEIGHT)).st_ino == os.lstat(first).st_ino
        assert os.lstat(second).st_nlink == 1
        # Placing the same inode again is accepted and adds no link.
        verified_link(claimed, journal, first, WEIGHT, weights)
        assert os.lstat(first).st_nlink == 2

        receive = place.staging(claimed, "receive")
        try:
            source_fd, _ = place.open_source(source, None, len(config))
            try:
                part, measured = place.copy_part(source_fd, len(config), receive, "config.json.part")
            finally:
                os.close(source_fd)
            try:
                with pytest.raises(ValueError, match="never replaces a name"):
                    place.place_staged(claimed.dir_fd, receive, "config.json.part", "config.json", journal,
                                       fd=part, sha256=measured, origin="copy", source=source)
            finally:
                place.discard_part(receive, "config.json.part", part)
        finally:
            os.close(receive)
        assert foreign.read_bytes() == b"{}" and os.lstat(foreign).st_ino == kept
        assert "config.json" not in place.journal_load(claimed).files
        assert place.unexpected_names(claimed.dir_fd, journal, [WEIGHT, "config.json"]) == ["config.json"]


@linux
def test_journal_write_ahead_resumes_after_an_interruption(tmp_path, mount_table, monkeypatch):
    class Interrupted(BaseException):
        pass

    first, second = b"w" * 8192, b"v" * 4096
    first_path = user_file(tmp_path, WEIGHT, first)
    second_path = user_file(tmp_path, SECOND, second)
    path = checkpoint(tmp_path)
    journal_file = Path(os.path.dirname(path), "." + REVISION + ".sparkring", "journal.json")
    seen = {}
    link, journal_place = os.link, place.journal_place

    def stop_before_linking(*args, **kwargs):
        seen["journal"] = json.loads(journal_file.read_text())
        raise Interrupted

    with place.claim(path, REPOSITORY, REVISION) as claimed:
        monkeypatch.setattr(place.os, "link", stop_before_linking)
        with pytest.raises(Interrupted):
            verified_link(claimed, place.journal_load(claimed), first_path, WEIGHT, first)
        monkeypatch.setattr(place.os, "link", link)
    # The entry reached the disk before the link was attempted.
    assert seen["journal"]["files"][WEIGHT]["state"] == "placing"

    def stop_after_linking(*args, **kwargs):
        raise Interrupted

    with place.claim(path, REPOSITORY, REVISION) as claimed:
        journal = place.journal_load(claimed)
        assert journal.files[WEIGHT]["state"] == "placing"
        assert place.journal_recover(claimed.dir_fd, journal) == {
            "completed": [], "dropped": [WEIGHT], "missing": [], "foreign": []}
        assert place.journal_load(claimed).files == {}
        monkeypatch.setattr(place, "journal_place", stop_after_linking)
        with pytest.raises(Interrupted):
            verified_link(claimed, journal, second_path, SECOND, second)
        monkeypatch.setattr(place, "journal_place", journal_place)

    with place.claim(path, REPOSITORY, REVISION) as claimed:
        journal = place.journal_load(claimed)
        assert journal.files[SECOND]["state"] == "placing"
        assert place.journal_recover(claimed.dir_fd, journal)["completed"] == [SECOND]
        entry = place.journal_load(claimed).files[SECOND]
        assert entry["state"] == "placed"
        assert entry["stats"] == place.stats(os.lstat(os.path.join(path, SECOND)))
        assert place.check_placed(claimed.dir_fd, SECOND, journal, len(second), digest(second)) == "verified"
        verified_link(claimed, journal, first_path, WEIGHT, first)
        assert place.unexpected_names(claimed.dir_fd, journal, [WEIGHT, SECOND]) == []
        assert {name: value["state"] for name, value in place.journal_load(claimed).files.items()} == {
            WEIGHT: "placed", SECOND: "placed"}


@linux
def test_only_journaled_inodes_are_removed(tmp_path, mount_table):
    weights = b"u" * 8192
    source = user_file(tmp_path, WEIGHT, weights)
    user_before = tree_state(tmp_path / "user")
    with place.claim(checkpoint(tmp_path), REPOSITORY, REVISION) as claimed:
        journal = place.journal_load(claimed)
        target = Path(claimed.path, WEIGHT)
        added = Path(claimed.path, "added_tokens.json")
        added.write_text("{}")
        with pytest.raises(ValueError, match="not in SparkRing's journal"):
            place.unlink_placed(claimed.dir_fd, "added_tokens.json", journal)
        assert added.read_text() == "{}"

        verified_link(claimed, journal, source, WEIGHT, weights)
        # Another inode now holds SparkRing's name.
        replacement = Path(claimed.path, "replacement")
        replacement.write_bytes(b"r" * 8192)
        os.replace(replacement, target)
        with pytest.raises(ValueError, match="not the file SparkRing placed"):
            place.unlink_placed(claimed.dir_fd, WEIGHT, journal)
        with pytest.raises(ValueError, match="did not place"):
            place.check_placed(claimed.dir_fd, WEIGHT, journal, len(weights), digest(weights))
        assert target.read_bytes() == b"r" * 8192
        assert place.journal_recover(claimed.dir_fd, journal)["foreign"] == [WEIGHT]
        assert place.unexpected_names(claimed.dir_fd, journal, [WEIGHT]) == ["added_tokens.json", WEIGHT]

        # A placed name that disappeared is forgotten; the name SparkRing placed is removed only by inode.
        os.unlink(target)
        assert place.journal_recover(claimed.dir_fd, journal)["missing"] == [WEIGHT]
        verified_link(claimed, journal, source, WEIGHT, weights)
        assert place.unlink_placed(claimed.dir_fd, WEIGHT, journal) is True
        assert not target.exists() and WEIGHT not in place.journal_load(claimed).files
        assert added.read_text() == "{}"

        # Staging directories are emptied and removed only with their marker.
        fetch = place.staging(claimed, "fetch")
        try:
            os.mkdir("nested", dir_fd=fetch)
            Path(claimed.state, "fetch/nested/partial").write_bytes(b"p")
            os.symlink(source, os.path.join(claimed.state, "fetch", "link"))
        finally:
            os.close(fetch)
        marker = Path(claimed.state, "fetch", place.STAGING_MARKER)
        marker.write_text(json.dumps({"purpose": "fetch", "path": "/somewhere/else"}))
        with pytest.raises(ValueError, match="marked for another use"):
            place.remove_staging(claimed, "fetch")
        with pytest.raises(ValueError, match="marked for another use"):
            place.staging(claimed, "fetch")
        marker.write_text(json.dumps({"purpose": "fetch", "path": claimed.path}))
        assert place.remove_staging(claimed, "fetch") is True
        assert not Path(claimed.state, "fetch").exists()
        unmarked = Path(claimed.state, "receive")
        unmarked.mkdir(mode=0o700)
        (unmarked / "file").write_text("kept")
        with pytest.raises(ValueError, match="without SparkRing's staging marker"):
            place.staging(claimed, "receive", empty=True)
        with pytest.raises(ValueError, match="without SparkRing's staging marker"):
            place.remove_staging(claimed, "receive")
        assert (unmarked / "file").read_text() == "kept"
    assert tree_state(tmp_path / "user") == user_before


@linux
def test_copies_are_exact_length_and_verified_before_placement(tmp_path, mount_table):
    config = b'{"model_type": "qwen3_8", "hidden_size": 4096}\n'
    source = user_file(tmp_path, "config.json", config)
    source_info = os.lstat(source)
    with place.claim(checkpoint(tmp_path), REPOSITORY, REVISION) as claimed:
        journal = place.journal_load(claimed)
        receive = place.staging(claimed, "receive", empty=True)
        staged = Path(claimed.state, "receive")
        try:
            (staged / "config.json.part").write_bytes(b"stale bytes from an interrupted copy")
            source_fd, _ = place.open_source(source, [source_info.st_dev, source_info.st_ino], len(config))
            try:
                with pytest.raises(ValueError, match="more than its pinned"):
                    place.copy_part(source_fd, len(config) - 1, receive, "short.part")
                with pytest.raises(ValueError, match="ended after"):
                    place.copy_part(source_fd, len(config) + 1, receive, "long.part")
                assert sorted(os.listdir(staged)) == [place.STAGING_MARKER, "config.json.part"]
                part, measured = place.copy_part(source_fd, len(config), receive, "config.json.part")
            finally:
                os.close(source_fd)
            assert measured == digest(config) and (staged / "config.json.part").read_bytes() == config
            assert not Path(claimed.path, "config.json").exists()
            try:
                place.place_staged(claimed.dir_fd, receive, "config.json.part", "config.json", journal,
                                   fd=part, sha256=measured, origin="copy", source=source)
            finally:
                os.close(part)
            assert os.listdir(staged) == [place.STAGING_MARKER]

            # Bytes that differ from the pin are discarded, never placed.
            other = user_file(tmp_path, "other/generation_config.json", b'{"top_k": 20}\n')
            source_fd, _ = place.open_source(other, None, 14)
            try:
                part, measured = place.copy_part(source_fd, 14, receive, "generation_config.json.part")
            finally:
                os.close(source_fd)
            assert measured != digest(b'{"top_k": 40}\n')
            place.discard_part(receive, "generation_config.json.part", part)
            assert os.listdir(staged) == [place.STAGING_MARKER]

            # A file written into staging by another program (rsync, the Hub client) is re-opened and hashed.
            tokenizer = b'{"version": "1.0"}\n'
            (staged / "tokenizer.json").write_bytes(tokenizer)
            fd, before = place.open_source("tokenizer.json", None, len(tokenizer), dir_fd=receive)
            try:
                place.place_staged(claimed.dir_fd, receive, "tokenizer.json", "tokenizer.json", journal, fd=fd,
                                   sha256=place.hash_descriptor(fd, len(tokenizer)), origin="rsync", before=before)
            finally:
                os.close(fd)

            # A staged file that another name links, or that others may write, is not placed.
            shared = user_file(tmp_path, "shared/vocab.json", b"{}")
            os.link(shared, staged / "vocab.json")
            (staged / "merges.txt").write_bytes(b"a b\n")
            os.chmod(staged / "merges.txt", 0o666)
            for name in ("vocab.json", "merges.txt"):
                size = os.lstat(staged / name).st_size
                fd, before = place.open_source(name, None, size, dir_fd=receive)
                try:
                    with pytest.raises(ValueError, match="not a private regular file"):
                        place.place_staged(claimed.dir_fd, receive, name, name, journal, fd=fd,
                                           sha256=place.hash_descriptor(fd, size), origin="rsync", before=before)
                finally:
                    os.close(fd)
            assert Path(shared).read_bytes() == b"{}" and os.lstat(shared).st_nlink == 2
        finally:
            os.close(receive)
        placed = os.lstat(Path(claimed.path, "config.json"))
        assert Path(claimed.path, "config.json").read_bytes() == config
        assert placed.st_ino != source_info.st_ino and placed.st_nlink == 1 and placed.st_uid == os.geteuid()
        assert stat.S_IMODE(placed.st_mode) & 0o022 == 0
        entries = place.journal_load(claimed).files
        assert {name: (entry["state"], entry["origin"]) for name, entry in entries.items()} == {
            "config.json": ("placed", "copy"), "tokenizer.json": ("placed", "rsync")}
        assert sorted(os.listdir(claimed.path)) == ["config.json", "tokenizer.json"]
    assert Path(source).read_bytes() == config and os.lstat(source).st_nlink == 1


@linux
def test_per_inode_records_use_the_hashed_descriptors_stats(tmp_path, mount_table, records):
    content = b"r" * 5000
    path = user_file(tmp_path, WEIGHT, content)
    fd, before = place.open_source(path, None, len(content))
    try:
        measured = place.hash_descriptor(fd, len(content))
        # The owner moves the file and puts other bytes at the path after it was hashed.
        os.rename(path, path + ".previous")
        Path(path).write_bytes(b"s" * len(content))
        record = place.record_inode(before, measured, path, now=lambda: 1234.5)
    finally:
        os.close(fd)
    saved = json.loads((records / f"{before.st_dev}-{before.st_ino}.json").read_text())
    assert saved == record == {"identity": [before.st_dev, before.st_ino], "stats": place.stats(before),
                               "sha256": measured, "observed_at": 1234.5}
    replaced = os.lstat(path)
    assert not (records / f"{replaced.st_dev}-{replaced.st_ino}.json").exists()
    assert place.record_inode(os.lstat(path + ".previous"), measured, path + ".previous")["seen_as"] == path + ".previous"

    with place.claim(checkpoint(tmp_path), REPOSITORY, REVISION) as claimed:
        placed = verified_link(claimed, place.journal_load(claimed), path + ".previous", WEIGHT, content)
    saved = json.loads((records / f"{before.st_dev}-{before.st_ino}.json").read_text())
    assert saved["stats"] == placed == place.stats(os.lstat(path + ".previous"))
    assert saved["stats"][:4] == place.stats(before)[:4] and saved["seen_as"] == path + ".previous"


@linux
@pytest.mark.parametrize("kind, where, refusal", [
    ("nfs4", "srv", "is nfs4, not a local filesystem"),
    ("autofs", "srv", "is autofs, not a local filesystem"),
    ("cifs", "srv", "is cifs, not a local filesystem"),
    ("fuse.sshfs", "srv", "is fuse.sshfs, not a local filesystem"),
    ("tmpfs", "srv", "is tmpfs, not a local filesystem"),
    ("ext4", "checkpoint", "is a mount point"),
    ("xfs", "srv", None),
    ("btrfs", "srv", None),
])
def test_non_local_or_mount_point_directories_are_refused(kind, where, refusal, tmp_path, mount_table):
    path = checkpoint(tmp_path)
    os.makedirs(os.path.dirname(path))
    if where == "checkpoint":
        os.mkdir(path)
    point = str(tmp_path / "srv") if where == "srv" else path
    with open(mount_table, "a") as stream:
        stream.write(mount_line(41, 29, point, kind, "0:61"))
    if refusal is None:
        entry = place.local_filesystem(path)
        assert (entry["id"], entry["type"], entry["point"], entry["device"]) == (41, kind, point, 61)
        with place.claim(path, REPOSITORY, REVISION) as claimed:
            assert claimed.action == "created"
        return
    before = tree_state(tmp_path / "srv")
    with pytest.raises(ValueError, match=refusal):
        place.local_filesystem(path)
    with pytest.raises(ValueError, match=refusal):
        place.claim(path, REPOSITORY, REVISION)
    assert tree_state(tmp_path / "srv") == before


def test_mount_table_entries_are_unescaped_and_the_deepest_covering_mount_wins():
    table = place.parse_mounts(
        mount_line(29, 1, "/", "ext4", "259:2") + mount_line(35, 29, "/srv/spark share", "autofs", "0:52")
        + mount_line(36, 29, "/srv", "nfs4", "0:60") + mount_line(37, 36, "/srv", "ext4", "259:3")
        + "not a mountinfo line\n")
    assert [entry["id"] for entry in table] == [29, 35, 36, 37]
    assert table[0]["device"] == 66306 and table[1]["point"] == "/srv/spark share"
    assert place.mount_of("/srv/spark share/models", table)["type"] == "autofs"
    assert place.mount_of("/srv/spark shared", table)["id"] == 37
    assert place.mount_of("/srv/sparkring/tp2", table)["id"] == 37
    assert place.mount_of("/srv", table)["id"] == 37
    assert place.mount_of("/var/tmp", table)["id"] == 29
    assert place.mount_of("/srv", []) is None


@linux
@pytest.mark.parametrize("code", [errno.EXDEV, errno.EPERM])
def test_refused_links_leave_neither_a_name_nor_a_journal_entry(code, tmp_path, mount_table, monkeypatch):
    weights = b"x" * 4096
    source = user_file(tmp_path, WEIGHT, weights)

    def refuse(*args, **kwargs):
        raise OSError(code, os.strerror(code))

    link = os.link
    with place.claim(checkpoint(tmp_path), REPOSITORY, REVISION) as claimed:
        journal = place.journal_load(claimed)
        monkeypatch.setattr(place.os, "link", refuse)
        with pytest.raises(OSError) as raised:
            verified_link(claimed, journal, source, WEIGHT, weights)
        monkeypatch.setattr(place.os, "link", link)
        assert raised.value.errno == code
        assert os.listdir(claimed.path) == [] and place.journal_load(claimed).files == {}
    assert os.lstat(source).st_nlink == 1


@linux
def test_staged_names_are_reached_without_following_symlinks(tmp_path, mount_table):
    weights = b"t" * 4096
    user_file(tmp_path, "audio/model.safetensors", weights)
    with place.claim(checkpoint(tmp_path), REPOSITORY, REVISION) as claimed:
        fetch = place.staging(claimed, "fetch")
        try:
            os.symlink(tmp_path / "user/audio", os.path.join(claimed.state, "fetch", "audio_tokenizer"))
            with pytest.raises(ValueError, match="symlink"):
                place.open_source("audio_tokenizer/model.safetensors", None, len(weights), dir_fd=fetch)
            for unsafe in ("../owner.json", "/etc/passwd", ".cache/huggingface/x", "a//b"):
                with pytest.raises(ValueError, match="Unsafe checkpoint file name"):
                    place.open_source(unsafe, None, 1, dir_fd=fetch)
        finally:
            os.close(fetch)


@linux
def test_placed_files_are_checked_by_stats_then_by_content(tmp_path, mount_table):
    config = b'{"architectures": ["Qwen3_8ForCausalLM"]}\n'
    source = user_file(tmp_path, "config.json", config)
    with place.claim(checkpoint(tmp_path), REPOSITORY, REVISION) as claimed:
        journal = place.journal_load(claimed)
        receive = place.staging(claimed, "receive", empty=True)
        try:
            source_fd, _ = place.open_source(source, None, len(config))
            try:
                part, measured = place.copy_part(source_fd, len(config), receive, "config.json.part")
            finally:
                os.close(source_fd)
            try:
                place.place_staged(claimed.dir_fd, receive, "config.json.part", "config.json", journal,
                                   fd=part, sha256=measured, origin="copy", source=source)
            finally:
                os.close(part)
        finally:
            place.remove_staging(claimed, "receive")
            os.close(receive)
        target = Path(claimed.path, "config.json")
        pin = digest(config)
        assert place.check_placed(claimed.dir_fd, "config.json", journal, len(config), pin) == "verified"
        # Same content with other stats: hashed again, and the journal keeps the new stats.
        os.utime(target, ns=(0, 10**18))
        assert place.check_placed(claimed.dir_fd, "config.json", journal, len(config), pin) == "rehashed"
        assert place.journal_load(claimed).files["config.json"]["stats"] == place.stats(os.lstat(target))
        # Other content: SparkRing removes its own name.
        with open(target, "r+b") as stream:
            stream.write(b"[")
        assert place.check_placed(claimed.dir_fd, "config.json", journal, len(config), pin) == "removed"
        assert not target.exists() and place.journal_load(claimed).files == {}
    assert Path(source).read_bytes() == config


def test_module_source_runs_standalone_with_only_standard_library_imports(tmp_path):
    source = inspect.getsource(place)
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0 and node.module != "__future__"
            imported.add(node.module.split(".")[0])
    assert imported <= set(sys.stdlib_module_names)
    program = source + "\nprint(json.dumps(parse_mounts(" + repr(mount_line(29, 1, "/", "ext4")) + ")))\n"
    environment = {"SYSTEMROOT": os.environ["SYSTEMROOT"]} if os.name == "nt" else {}
    result = subprocess.run([sys.executable, "-I", "-B", "-"], input=program, capture_output=True, text=True,
                            env=environment, cwd=tmp_path, timeout=60)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)[0]["type"] == "ext4"
