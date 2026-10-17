"""Deterministic semantic check for DeepSeek-V4.1 cache store and restore.

Requests pin ``thinking: false`` so the served answer is the final content
channel, matching how the model is qualified. The miss and hit checks reuse
the model-neutral qualification mechanics, including the canary that proves
a publication inventory exists before a restore is judged.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from typing import Any

from deploy.deployment_contract import (
    SemanticGateInconclusive,
    assistant_content as _content,
    build_long_prompt,
    request_chat,
    run_semantic_hit,
    run_semantic_miss,
)

__all__ = ("build_long_prompt", "run_hit", "run_miss")

_REQUEST = partial(request_chat, extra_body={"thinking": False})


def run_miss(
    endpoint: str,
    model: str,
    reference: Path,
    *,
    long_max_tokens: int = 32,
    records: int = 384,
) -> dict[str, Any]:
    """Run the shared miss check through this module's injectable HTTP seam."""

    return run_semantic_miss(
        endpoint,
        model,
        reference,
        long_max_tokens=long_max_tokens,
        records=records,
        request=_REQUEST,
        content_reader=_content,
    )


def run_hit(
    endpoint: str,
    model: str,
    reference: Path,
    *,
    long_max_tokens: int = 32,
    short_max_tokens: int = 32,
    records: int | None = None,
) -> dict[str, Any]:
    """Run the shared hit check through this module's injectable HTTP seam."""

    return run_semantic_hit(
        endpoint,
        model,
        reference,
        long_max_tokens=long_max_tokens,
        short_max_tokens=short_max_tokens,
        records=records,
        request=_REQUEST,
        content_reader=_content,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("miss", "hit"))
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="DeepSeek-V4.1-Flash-TP4")
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument(
        "--records",
        type=int,
        help="archive records; miss defaults to 384 and hit reads the reference",
    )
    args = parser.parse_args()
    try:
        result = (
            run_miss(
                args.endpoint,
                args.model,
                args.reference,
                records=args.records or 384,
            )
            if args.phase == "miss"
            else run_hit(
                args.endpoint,
                args.model,
                args.reference,
                records=args.records,
            )
        )
    except SemanticGateInconclusive as error:
        print(json.dumps(error.as_result(), sort_keys=True))
        raise SystemExit(2) from None
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
