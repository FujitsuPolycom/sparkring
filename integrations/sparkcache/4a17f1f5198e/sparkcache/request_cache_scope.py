"""Opaque, versioned request cache namespaces; raw salts never enter metadata."""

import hashlib
import json
import re


def fingerprint(value: str | None) -> str:
    if value is not None and type(value) is not str:
        raise ValueError("Request cache salt must be a string or None")
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return (
        "sparkcache-request-scope-v1:"
        + hashlib.sha256(b"sparkcache-request-scope-v1\x00" + encoded).hexdigest()
    )


UNSALTED_SCOPE = fingerprint(None)


def metadata_binding(scope: object, digest: object) -> str:
    """Detect mismatched fields on trusted connector metadata, not authenticate it."""
    if not valid_scope(scope) or type(digest) is not str:
        return ""
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        return ""
    return hashlib.sha256((scope + ":" + digest).encode("ascii")).hexdigest()


def valid_metadata(plan: object) -> bool:
    expected = metadata_binding(
        getattr(plan, "request_scope", None), getattr(plan, "digest", None)
    )
    return bool(expected) and getattr(plan, "scope_binding", None) == expected


def valid_scope(value: object) -> bool:
    return (
        type(value) is str
        and re.fullmatch(r"sparkcache-request-scope-v1:[0-9a-f]{64}", value) is not None
    )
