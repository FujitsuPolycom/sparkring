"""Derive a versioned checkpoint identity from explicit, immutable inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any

SCHEMA = "sparkcache-checkpoint-identity/v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _digest(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _canonical(value: Any) -> bytes:
    """Sorted ASCII JSON with no whitespace; nonfinite numbers are invalid."""
    if isinstance(value, dict):
        _require(all(isinstance(key, str) for key in value), "JSON keys must be strings")
        for item in value.values():
            _canonical(item)
    elif isinstance(value, list):
        for item in value:
            _canonical(item)
    else:
        _require(value is None or type(value) in (str, bool, int, float), "Inputs must be JSON values")
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def derive_checkpoint_identity(inputs: dict[str, Any]) -> str:
    """Return SHA-256 of the canonical v1 envelope, without changing recorded IDs.

    Inputs bind a repository revision, artifact manifest or file digests,
    quantization/encoding, model configuration and draft policy. This function
    hashes supplied pins; it does not attest that local weights match them.
    """
    fields = {"repository", "revision", "artifacts", "quantization", "model_config", "draft"}
    _require(isinstance(inputs, dict) and set(inputs) == fields,
             "Identity inputs require exactly: " + ", ".join(sorted(fields)))
    repository, revision = inputs["repository"], inputs["revision"]
    _require(isinstance(repository, str) and bool(repository.strip()) and repository == repository.strip(),
             "repository must be a nonempty identifier without surrounding whitespace")
    _require(isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision) is not None,
             "revision must be a full lowercase 40- or 64-hex commit")
    artifacts = inputs["artifacts"]
    _require(isinstance(artifacts, dict), "artifacts must be an object")
    if set(artifacts) == {"manifest_sha256"}:
        _require(_digest(artifacts["manifest_sha256"]), "manifest_sha256 must be SHA-256 of the manifest bytes")
    else:
        _require(set(artifacts) == {"files"} and isinstance(artifacts["files"], dict)
                 and bool(artifacts["files"]), "artifacts requires manifest_sha256 or a nonempty files map")
        for name, sha in artifacts["files"].items():
            _require(isinstance(name, str) and bool(name) and "\\" not in name and ":" not in name,
                     "artifact paths must be relative POSIX file names")
            path = PurePosixPath(name)
            _require(not path.is_absolute() and ".." not in path.parts and path.as_posix() == name
                     and name != ".", "artifact paths must be normalized and contained")
            _require(_digest(sha), "artifact file digests must be lowercase SHA-256")
    quantization = inputs["quantization"]
    _require(isinstance(quantization, dict) and {"method", "encoding"} <= set(quantization),
             "quantization requires explicit method and encoding (use none when absent)")
    for key in ("method", "encoding"):
        _require(isinstance(quantization[key], str) and bool(quantization[key].strip()),
                 "quantization " + key + " must be nonempty")
    config = inputs["model_config"]
    _require(isinstance(config, dict) and set(config) in ({"sha256"}, {"values"}),
             "model_config requires sha256 or values")
    if "sha256" in config:
        _require(_digest(config["sha256"]), "model_config sha256 must be lowercase SHA-256")
    else:
        _require(isinstance(config["values"], dict) and bool(config["values"]),
                 "model_config values must be a nonempty object")
    draft = inputs["draft"]
    _require(isinstance(draft, dict), "draft must be an object")
    mode = draft.get("mode")
    if mode == "none":
        _require(set(draft) == {"mode"}, "draft mode none has no other fields")
    else:
        fields = {"mode", "config"} | ({"checkpoint_sha256"} if mode == "separate" else set())
        _require(mode in ("embedded", "separate") and set(draft) == fields,
                 "draft requires none, embedded with config, or separate with config and checkpoint_sha256")
        _require(isinstance(draft["config"], dict) and bool(draft["config"]),
                 "draft config must be a nonempty object")
        if mode == "separate":
            _require(_digest(draft["checkpoint_sha256"]), "separate draft requires its checkpoint SHA-256")
    return hashlib.sha256(_canonical({"schema": SCHEMA, "inputs": inputs})).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="JSON containing the six pinned input fields")
    parser.add_argument("--output", type=Path, help="write a receipt exclusively; default: stdout")
    args = parser.parse_args(argv)
    try:
        inputs = json.loads(args.input.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
        identity = derive_checkpoint_identity(inputs)
        receipt = json.dumps({"schema": SCHEMA, "inputs": inputs, "checkpoint_sha256": identity},
                             indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(receipt)
        else:
            print(receipt, end="")
    except (ValueError, OSError, TypeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
