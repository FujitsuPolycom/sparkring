"""Bounded, fail-open SparkCache buddy replication primitives."""

from .protocol import (
    Frame,
    FrameStreamDecoder,
    FrameType,
    ProtocolError,
    decode_commit_record,
    encode_commit_record,
)
from .state import BuddyReceiver, BuddySender, CommittedReplica, ReplicaChunk

__all__ = [
    "BuddyReceiver",
    "BuddySender",
    "CommittedReplica",
    "ReplicaChunk",
    "Frame",
    "FrameStreamDecoder",
    "FrameType",
    "ProtocolError",
    "decode_commit_record",
    "encode_commit_record",
]
