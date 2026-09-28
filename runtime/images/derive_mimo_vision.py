"""Apply MiMo's vision-encoder attention sinks in the softmax denominator.

Most blocks of the MiMo-V2.6 vision encoder use windowed attention with a
learned per-head sink. The model was trained with the sink as an extra logit
that only enlarges the softmax denominator. vLLM's `mimo_v2_omni.py` in the
parent passes `sinks_bias_key0=True` to `context_attention_fwd`, which instead
adds the sink to the score of each image's first key. On regions of uniform
color every key scores the same, so those patches encode wrongly: a picture
with a red left half and a blue right half reads as black and white. This layer
passes `sinks_bias_key0=False`, the kernel's denominator form, as Local
Inference Lab's vLLM branches and vllm-project/vllm#58235 do. Text inputs never
reach the vision encoder and are unaffected.

The parent's mimo_v2_omni.py must have SHA-256 7cecf9be...; the result has
SHA-256 2b5bae54... (both pinned below).
/opt/sparkring/receipts/derived-mimo-vision.json records the replaced file.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runtime.images.derived_layer import Layer, main, swap  # noqa: E402

MODEL = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2_omni.py"
INHERITED = "7cecf9be2e17daba7cd1c0dae640f6c47ae84dc686fedc47cdb0a4d5ee9c1ca3"
RESULT = "2b5bae543d98f7ea23a134f18750912a7c59b1beb9dcf8cec82eaf124e995a6f"
KEY_ZERO = "            sinks_bias_key0=True,\n"
DENOMINATOR = "            sinks_bias_key0=False,\n"


def replace(read, receipt):
    return {MODEL: swap(read(MODEL).decode("utf-8"), KEY_ZERO, DENOMINATOR).encode("utf-8")}


LAYER = Layer(
    name="mimo-vision",
    purpose=("The MiMo-V2.6 vision encoder applies its attention sinks in the softmax denominator, as "
             "the model was trained, so images with regions of uniform color encode correctly"),
    replace=replace,
    provenance="/opt/sparkring/receipts/derived-mimo-vision.json",
    pins={MODEL: (INHERITED, RESULT)},
)

if __name__ == "__main__":
    main(LAYER)
