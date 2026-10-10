"""Environment variables of SIRCL's point-to-point channels (torch-free).

Every variable is read when a group's point-to-point context is built; the
values that shape the wire protocol (slots, slot bytes, chunk, kernel
geometry, wait limits) are part of the setup agreement, so ranks with
different values fail together. The context also reads these variables of
SIRCL's collective sessions with the same meaning: ``SIRCL_STARTUP_WAIT_S``,
``SIRCL_SERVING_WAIT_S``, ``SIRCL_HAIRPIN_QUEUE_BYTES``, ``SIRCL_MAX_RELAYS``,
``SIRCL_GID_INDEX`` (fallback ``NCCL_IB_GID_INDEX``), ``SIRCL_TRAFFIC_CLASS``
(fallback ``NCCL_IB_TC``) and ``SIRCL_BUILD_CACHE_DIR``.
``python -m sparkring_sircl.p2p.settings`` prints the table.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping

from . import protocol

DEFAULT_STARTUP_WAIT_S = 600.0
DEFAULT_SERVING_WAIT_S = 20.0
MAX_WAIT_S = 0xFFFFFFFF / 1e6
DEFAULT_HAIRPIN_QUEUE_BYTES = 524288
DEFAULT_LANE_CHECK_MS = 2000


@dataclasses.dataclass(frozen=True)
class Variable:
    name: str
    meaning: str
    default: str


VARIABLES = (
    Variable("SIRCL_P2P_SLOTS", "slots of every point-to-point channel, a power of two from 2 to 32",
             str(protocol.DEFAULT_SLOTS)),
    Variable("SIRCL_P2P_SLOT_BYTES", "bytes of a channel slot, the largest item (multiple of 4096 up to 512 MiB)",
             str(protocol.DEFAULT_SLOT_BYTES)),
    Variable("SIRCL_P2P_BLOCKS", "thread blocks of a send or receive launch (1-64), each taking every B-th item",
             str(protocol.DEFAULT_BLOCKS)),
    Variable("SIRCL_P2P_THREADS", "threads per block of the point-to-point kernels (multiple of 32, 64-1024)",
             str(protocol.DEFAULT_THREADS)),
    Variable("SIRCL_P2P_UNROLL", "16-byte packs each thread copies per pass (1-8)", str(protocol.DEFAULT_UNROLL)),
    Variable("SIRCL_P2P_WINDOW_BYTES", "largest forward window of a point-to-point lane through relays; 0 refuses "
             "relayed channels", str(protocol.DEFAULT_WINDOW_BYTES)),
    Variable("SIRCL_P2P_CHUNK_BYTES", "largest write of a windowed lane's stripe (multiple of 16)",
             str(protocol.DEFAULT_CHUNK_BYTES)),
    Variable("SIRCL_P2P_PROGRESS_CPU", "CPU list of the point-to-point progress thread, e.g. 9 or 5-9,15-19",
             "unpinned"),
    Variable("SIRCL_P2P_NATIVE_LIBRARY", "a prebuilt point-to-point native library to load instead of the build "
             "cache", "unset"),
)


class SettingError(ValueError):
    """A point-to-point setting has a value the channels cannot use."""


def _int(environ: Mapping[str, str], name: str, default: int) -> int:
    raw = environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw, 0)
    except ValueError:
        raise SettingError(f"{name}={raw} is not an integer") from None


def _seconds(environ: Mapping[str, str], name: str, default: float) -> float:
    raw = environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise SettingError(f"{name}={raw} is not a number of seconds") from None
    if not 1e-6 <= value <= MAX_WAIT_S:
        raise SettingError(f"{name} {value} must be between 1e-6 and {MAX_WAIT_S:.0f} seconds")
    return value


@dataclasses.dataclass(frozen=True)
class P2PSettings:
    slots: int = protocol.DEFAULT_SLOTS
    slot_bytes: int = protocol.DEFAULT_SLOT_BYTES
    blocks: int = protocol.DEFAULT_BLOCKS
    threads: int = protocol.DEFAULT_THREADS
    unroll: int = protocol.DEFAULT_UNROLL
    window_bytes: int = protocol.DEFAULT_WINDOW_BYTES
    chunk_bytes: int = protocol.DEFAULT_CHUNK_BYTES
    hairpin_queue_bytes: int = DEFAULT_HAIRPIN_QUEUE_BYTES
    startup_wait_s: float = DEFAULT_STARTUP_WAIT_S
    serving_wait_s: float = DEFAULT_SERVING_WAIT_S
    max_relays: int = 3

    def __post_init__(self) -> None:
        protocol.P2PLayout(2, 1, self.slots, self.slot_bytes)   # validates slots and slot bytes
        if not 1 <= self.blocks <= 64:
            raise SettingError(f"SIRCL_P2P_BLOCKS {self.blocks} must be 1 to 64")
        if self.threads % 32 or not 64 <= self.threads <= 1024:
            raise SettingError(f"SIRCL_P2P_THREADS {self.threads} must be a multiple of 32 from 64 to 1024")
        if not 1 <= self.unroll <= 8:
            raise SettingError(f"SIRCL_P2P_UNROLL {self.unroll} must be 1 to 8")
        if self.chunk_bytes < 16 or self.chunk_bytes % 16:
            raise SettingError(f"SIRCL_P2P_CHUNK_BYTES {self.chunk_bytes} must be a positive multiple of 16")
        if self.window_bytes < 0 or (self.window_bytes and self.window_bytes < self.chunk_bytes):
            raise SettingError(f"SIRCL_P2P_WINDOW_BYTES {self.window_bytes} must be 0 or at least one chunk "
                               f"({self.chunk_bytes} bytes)")
        if self.hairpin_queue_bytes < 4096:
            raise SettingError("SIRCL_HAIRPIN_QUEUE_BYTES must be at least 4096")
        if not 0 < self.serving_wait_s <= MAX_WAIT_S or not 0 < self.startup_wait_s <= MAX_WAIT_S:
            raise SettingError("wait limits are positive seconds")
        if self.max_relays < 0:
            raise SettingError("SIRCL_MAX_RELAYS must not be negative")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None, **overrides) -> "P2PSettings":
        env = os.environ if environ is None else environ
        values = dict(
            slots=_int(env, "SIRCL_P2P_SLOTS", protocol.DEFAULT_SLOTS),
            slot_bytes=_int(env, "SIRCL_P2P_SLOT_BYTES", protocol.DEFAULT_SLOT_BYTES),
            blocks=_int(env, "SIRCL_P2P_BLOCKS", protocol.DEFAULT_BLOCKS),
            threads=_int(env, "SIRCL_P2P_THREADS", protocol.DEFAULT_THREADS),
            unroll=_int(env, "SIRCL_P2P_UNROLL", protocol.DEFAULT_UNROLL),
            window_bytes=_int(env, "SIRCL_P2P_WINDOW_BYTES", protocol.DEFAULT_WINDOW_BYTES),
            chunk_bytes=_int(env, "SIRCL_P2P_CHUNK_BYTES", protocol.DEFAULT_CHUNK_BYTES),
            hairpin_queue_bytes=_int(env, "SIRCL_HAIRPIN_QUEUE_BYTES", DEFAULT_HAIRPIN_QUEUE_BYTES),
            startup_wait_s=_seconds(env, "SIRCL_STARTUP_WAIT_S", DEFAULT_STARTUP_WAIT_S),
            serving_wait_s=_seconds(env, "SIRCL_SERVING_WAIT_S", DEFAULT_SERVING_WAIT_S),
            max_relays=_int(env, "SIRCL_MAX_RELAYS", 3),
        )
        values.update({key: value for key, value in overrides.items() if value is not None})
        try:
            return cls(**values)
        except protocol.ProtocolError as error:
            raise SettingError(str(error)) from None

    def record(self) -> dict[str, object]:
        """The fields every rank of a group must share (part of the setup agreement)."""
        return dataclasses.asdict(self)


def explicit_gid_index(environ: Mapping[str, str] | None = None) -> int | None:
    """``SIRCL_GID_INDEX``, else ``NCCL_IB_GID_INDEX``, else None (resolve per device)."""
    env = os.environ if environ is None else environ
    for name in ("SIRCL_GID_INDEX", "NCCL_IB_GID_INDEX"):
        raw = env.get(name, "").strip()
        if raw:
            value = _int(env, name, 3)
            if not 0 <= value <= 255:
                raise SettingError(f"{name}={raw} is outside 0-255")
            return value
    return None


if __name__ == "__main__":  # pragma: no cover
    for variable in VARIABLES:
        print(f"{variable.name:28} {variable.default:10} {variable.meaning}")
