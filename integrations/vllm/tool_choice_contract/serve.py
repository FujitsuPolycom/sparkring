"""Launch a single Python vLLM API frontend with an optional tool-result policy."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import contract


# These complete serving modules have the reviewed generator/event interfaces.
ADMITTED = {
    "6c3e80dd7d2671049eed9fc8ede5038074b8dddac905652ff17a404965a02066": "DeepSeek published native parent",
    "4c129414f20ecbb44ea8613d774614e8a66152164a039def7c5c086587a1cc99": "DeepSeek API source upgrade",
    "9982953285e9df469032a82fffa4095d0e9d86278bede6e2b91d03d02373d182": "SparkRing LIL R37",
    # Installer images dev-20260924-cuda1342-nccl2323-status031 through
    # dev-20260927-mimovision-cuda1342-nccl2323-status032 inherit this module
    # from their software layer. An image derived with
    # runtime/images/derive_tool_choice_contract.py installs the policy from
    # that module itself and needs no entrypoint wrapper.
    "ea1f76074a9587c8054f54d30a6ba748b6a5fc4d90dd82c801504505c1da1f92": "Installer software layer, vLLM 1794dcf18",
}


def validate_frontend(args, environ):
    if (getattr(args, "api_server_count", None) not in (None, 0, 1)
            or getattr(args, "data_parallel_size", 1) != 1
            or getattr(args, "data_parallel_multi_port_external_lb", False)
            or getattr(args, "grpc", False)
            or environ.get("VLLM_USE_RUST_FRONTEND", "0") != "0"):
        raise ValueError("Tool-choice contract bootstrap supports one Python API frontend only.")


def install(serving):
    digest = hashlib.sha256(Path(serving.__file__).read_bytes()).hexdigest()
    if digest not in ADMITTED:
        raise RuntimeError(f"Tool-choice contract serving source is not admitted: {digest}")
    contract.install(serving.OpenAIServingChat)


def main():
    if contract.enabled(os.environ):
        from vllm.entrypoints.openai.chat_completion import serving
        from vllm.entrypoints.cli.serve import ServeSubcommand

        install(serving)
        original = ServeSubcommand.cmd

        def checked(args):
            validate_frontend(args, os.environ)
            return original(args)

        ServeSubcommand.cmd = staticmethod(checked)
    from vllm.entrypoints.cli.main import main as vllm_main

    vllm_main()


if __name__ == "__main__":
    main()
