"""Reject unsupported reasoning suppression before GLM-5.3 prompt generation."""

import ast
import hashlib
from pathlib import Path

ANCHOR = (
    "        chat_template_kwargs = self._effective_chat_template_kwargs(request)\n"
)
GUARD = """        # GLM-5.3's template always opens a reasoning segment. Disabling
        # its parser would expose reasoning as answer text without changing
        # model generation. Reject the unsupported mode before rendering.
        if getattr(self.model_config.hf_config, "model_type", None) == "glm5_next":
            disabled = any(
                chat_template_kwargs.get(key) is False
                for key in ("enable_thinking", "thinking")
            )
            if disabled or request.reasoning_effort == "none":
                return self.create_error_response(
                    "GLM-5.3-Flash does not support disabling thinking with its "
                    "chat template. Use reasoning_effort='low', 'high', or 'max'."
                )
"""


def patched(source):
    assert source.count(ANCHOR) == 1
    result = source.replace(ANCHOR, ANCHOR + GUARD, 1)
    ast.parse(result)
    return result


def apply(site_packages, warmup_path):
    path = Path(site_packages) / "vllm/entrypoints/openai/chat_completion/serving.py"
    source = path.read_text(encoding="utf-8")
    if (
        hashlib.sha256(source.encode()).hexdigest()
        != "9982953285e9df469032a82fffa4095d0e9d86278bede6e2b91d03d02373d182"
    ):
        raise ValueError("GLM chat serving source differs from the attested preimage")
    result = patched(source).encode()
    warmup = Path(warmup_path)
    data = warmup.read_text(encoding="utf-8")
    needle = '"chat_template_kwargs": {"enable_thinking": False},'
    if data.count(needle) != 1:
        raise ValueError("GLM warmup request differs from the supported source")
    data = data.replace(
        needle,
        '"chat_template_kwargs": {"enable_thinking": True}, "reasoning_effort": "low",',
        1,
    )
    compile(data, str(warmup), "exec")
    path.write_bytes(result)
    warmup.write_bytes(data.encode())
    return {
        str(path): hashlib.sha256(result).hexdigest(),
        str(warmup): hashlib.sha256(data.encode()).hexdigest(),
    }
