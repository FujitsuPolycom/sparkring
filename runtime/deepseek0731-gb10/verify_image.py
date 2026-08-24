#!/usr/bin/env python3
"""Verify the final GB10 DeepSeek image before a serving launch."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from apply_runtime_overlay import (
    DEFAULT_CONTRACT,
    OverlayError,
    attest_noop_files,
    load_contract,
    sha256_file,
)
from native_artifact_receipt import verify_installed
from parser_replay import ReplayError, run as replay_parser


REQUIRED_LD_PRELOAD = (
    "/usr/local/cuda/compat/libcuda.so.1:/opt/sparkring/nccl/libnccl.so.2"
)


def _verify_patch_results(
    root: Path, contract: dict[str, Any]
) -> list[dict[str, str]]:
    result = []
    for record in contract["runtime_patch"]["files"]:
        path = (root / record["path"]).resolve(strict=True)
        observed = sha256_file(path)
        if observed != record["result_sha256"]:
            raise OverlayError(f"runtime patch result differs: {path}")
        result.append({"path": str(path), "sha256": observed})
    return result


def verify(
    site_root: Path,
    source_root: Path,
    contract_path: Path,
    *,
    expect_native: bool,
    require_launch_env: bool,
) -> dict[str, Any]:
    contract = load_contract(contract_path)
    if require_launch_env and os.environ.get("LD_PRELOAD") != REQUIRED_LD_PRELOAD:
        raise OverlayError(
            "DeepSeek launch LD_PRELOAD differs: expected "
            f"{REQUIRED_LD_PRELOAD!r}, got {os.environ.get('LD_PRELOAD')!r}"
        )
    installed_noops = attest_noop_files(site_root, contract["attested_noop"])
    installed_results = _verify_patch_results(site_root, contract)
    source_results = _verify_patch_results(source_root, contract)
    parser_result = replay_parser(contract_path)

    native_result = None
    if expect_native:
        native = contract["native_patch"]
        native_result = verify_installed(
            Path(native["installed_path"]).resolve(strict=True),
            source_root,
            Path("/opt/sparkring-deepseek-gb10/native-artifact-receipt.json").resolve(
                strict=True
            ),
            contract_path,
        )
    return {
        "schema": "sparkring-deepseek-v4-gb10-final-image/v1",
        "status": "pass",
        "base_image": contract["base_image"],
        "variant": "native" if expect_native else "thin",
        "launch_environment_checked": require_launch_env,
        "installed_noops": installed_noops,
        "installed_results": installed_results,
        "retained_source_results": source_results,
        "parser": parser_result,
        "native": native_result,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--site-root",
        type=Path,
        default=Path("/opt/venv/lib/python3.12/site-packages"),
    )
    parser.add_argument("--source-root", type=Path, default=Path("/opt/r7-src/vllm"))
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--expect-native", action="store_true")
    parser.add_argument("--require-launch-env", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify(
            args.site_root.resolve(strict=True),
            args.source_root.resolve(strict=True),
            args.contract.resolve(strict=True),
            expect_native=args.expect_native,
            require_launch_env=args.require_launch_env,
        )
        payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output is None:
            print(payload, end="")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)
    except (
        FileExistsError,
        OSError,
        OverlayError,
        ReplayError,
        json.JSONDecodeError,
    ) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
