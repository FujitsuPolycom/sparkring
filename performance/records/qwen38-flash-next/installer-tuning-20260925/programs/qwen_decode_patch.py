#!/usr/bin/env python3
"""Write the Qwen4Exp decode patch for GB10 (SM121): GEMM plans and MXFP8 mixing projections.

Usage: qwen_decode_patch.py IMAGE_ID OUTPUT_DIR RESULTS_JSON [RESULTS_JSON ...]
Reads vllm/envs.py and the Qwen4Exp low_latency_gemm.py, model.py and mtp.py
from the image and writes patched copies to OUTPUT_DIR:
- envs.py registers VLLM_QWEN4_EXP_MXFP8_HC (default off). vLLM hashes every
  registered variable into its compile-cache key, so the two settings never
  share compiled graphs.
- low_latency_gemm.py gains QWEN4_EXP_SM121_GEMM_PLANS: for every measured
  (N, K, M), the fastest configuration whose output error against an FP32
  reference stayed within twice torch linear's BF16 error, kept when it was at
  least 5% faster than torch linear. `_gemm_plans()` selects it on SM121, and
  `applies_with_b12x()` reports whether the hook may run beside B12X.
- low_latency_gemm.py also gains `quantize_hyperconnection_mixing()`: with
  VLLM_QWEN4_EXP_MXFP8_HC=1 it switches BF16 hyper-connection down/injection
  projections whose input width is a multiple of 128 to an online MXFP8
  method. It quantizes the loaded weights once, runs batches of at most 16
  rows on B12X with BF16 activations, and runs larger batches on the retained
  BF16 weights.
- model.py and mtp.py call the mixing hook, then the low-latency hook, which
  runs beside B12X when `applies_with_b12x()` holds. On SM121 the low-latency
  hook replaces only BF16 projections, which B12X leaves to torch; other
  platforms keep their existing behavior.
Later results files override earlier ones for the same (N, K, M).
"""
import json
from pathlib import Path
import subprocess
import sys

IMAGE, OUTPUT = sys.argv[1], Path(sys.argv[2])
VLLM = "/usr/local/lib/python3.12/dist-packages/vllm/"
DIRECTORY = VLLM + "models/qwen4_exp/nvidia/"
MINIMUM_SPEEDUP = 1.05
NEWLINE = chr(10)

HELPER = '''

# Largest row count that uses the MXFP8 hyper-connection mixing weights. The
# B12X MXFP8 kernel with BF16 activations runs 16-row M tiles; one tile beats
# the BF16 GEMM, while larger batches (from four concurrent speculative
# requests up to prefill chunks) run as fast or faster on the retained BF16
# weight.
HC_MIXING_MXFP8_MAX_ROWS = 16


def _hc_mixing_linear(
    x: torch.Tensor,
    bf16_weight: torch.Tensor,
    out_features: int,
    layer_name: LayerNameType,
) -> torch.Tensor:
    if x.shape[0] <= HC_MIXING_MXFP8_MAX_ROWS:
        return torch.ops.vllm.b12x_blockscaled_linear(
            x, None, out_features, layer_name
        )
    return torch.nn.functional.linear(x, bf16_weight)


def _hc_mixing_linear_fake(
    x: torch.Tensor,
    bf16_weight: torch.Tensor,
    out_features: int,
    layer_name: LayerNameType,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], out_features))


# The row-count choice happens inside an opaque op, so the compiled graph has
# no shape guard and one capture serves every batch size.
direct_register_custom_op(
    op_name="qwen4_exp_hc_mixing_linear",
    op_func=_hc_mixing_linear,
    fake_impl=_hc_mixing_linear_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


@functools.cache
def _hc_mixing_method_class() -> type:
    from vllm.model_executor.layers.quantization.online.mxfp8 import (
        Mxfp8OnlineLinearMethod,
    )

    class HyperConnectionMixingMxfp8Method(Mxfp8OnlineLinearMethod):
        """MXFP8 weights for decode-sized batches, BF16 for larger ones.

        The loaded BF16 weight stays on the layer as a non-persistent buffer;
        the MXFP8 copy adds about half of its bytes.
        """

        def process_weights_after_loading(self, layer: nn.Module) -> None:
            if getattr(layer, "_already_called_process_weights_after_loading", False):
                return
            layer.register_buffer(
                "hc_mixing_bf16_weight", layer.weight.data, persistent=False
            )
            super().process_weights_after_loading(layer)

        def apply(
            self,
            layer: nn.Module,
            x: torch.Tensor,
            bias: torch.Tensor | None = None,
        ) -> torch.Tensor:
            source = x.reshape(-1, x.shape[-1]).contiguous()
            out_features = int(layer.b12x_mxfp8_packed_weight.out_features)
            output = torch.ops.vllm.qwen4_exp_hc_mixing_linear(
                source,
                layer.hc_mixing_bf16_weight,
                out_features,
                layer.b12x_layer_name,
            )
            if bias is not None:
                output = output + bias
            return output.view(*x.shape[:-1], out_features)

    return HyperConnectionMixingMxfp8Method


def quantize_hyperconnection_mixing(module: nn.Module) -> int:
    """Quantize BF16 hyper-connection down/injection projections to MXFP8.

    Enabled by VLLM_QWEN4_EXP_MXFP8_HC. These replicated projections are read
    in full on every rank at each decode step; MXFP8 halves those bytes for
    batches of at most HC_MIXING_MXFP8_MAX_ROWS rows. A projection qualifies
    when its input width is a multiple of 128, so the B12X MXFP8 kernel
    accepts BF16 activations. Weights are quantized once after loading.
    Returns the number of switched projections.
    """
    if not envs.VLLM_QWEN4_EXP_MXFP8_HC:
        return 0
    method = _hc_mixing_method_class()

    switched = 0
    for name, child in module.named_modules():
        weight = getattr(child, "weight", None)
        if (
            isinstance(child, LinearBase)
            and type(child.quant_method) is UnquantizedLinearMethod
            and name.rsplit(".", 1)[-1]
            in ("input_mix_weight_down_block_inject", "input_mix_weight_down")
            and weight is not None
            and weight.dim() == 2
            and weight.shape[1] % 128 == 0
        ):
            child.quant_method = method(use_a16=True)
            switched += 1
    return switched
'''

ENVS_ANCHOR_TYPE = "    VLLM_MXFP8_LM_HEAD: bool = False" + NEWLINE
ENVS_ANCHOR_VALUE = '    "VLLM_MXFP8_LM_HEAD": lambda: bool(int(os.getenv("VLLM_MXFP8_LM_HEAD", "0"))),' + NEWLINE
ENVS_VALUE = NEWLINE.join([
    "    # Quantize Qwen4Exp hyper-connection down/injection projections to MXFP8",
    "    # at load; they are otherwise BF16 and read in full on every rank.",
    '    "VLLM_QWEN4_EXP_MXFP8_HC": lambda: bool(',
    '        int(os.getenv("VLLM_QWEN4_EXP_MXFP8_HC", "0"))',
    "    ),",
]) + NEWLINE
IMPORT_OLD = "from .low_latency_gemm import enable_qwen4_exp_low_latency_gemm" + NEWLINE
IMPORT_NEW = NEWLINE.join([
    "from .low_latency_gemm import (",
    "    applies_with_b12x,",
    "    enable_qwen4_exp_low_latency_gemm,",
    "    quantize_hyperconnection_mixing,",
    ")",
]) + NEWLINE


def image_file(path):
    return subprocess.run(["docker", "run", "--rm", "--pull", "never", "--network", "none",
                           "--entrypoint", "cat", IMAGE, path], capture_output=True, check=True).stdout.decode()


def swap(text, old, new):
    assert text.count(old) == 1, old[:80]
    return text.replace(old, new)


plans = {}
for path in sys.argv[3:]:
    for name, shape in json.loads(Path(path).read_text())["shapes"].items():
        for m, row in shape["by_m"].items():
            if not m.isdigit():
                continue
            key = (shape["n"], shape["k"])
            if row.get("speedup", 0) >= MINIMUM_SPEEDUP:
                plans.setdefault(key, {})[int(m)] = (row["config"], name, row["torch_us"], row["best_us"])
            else:
                plans.get(key, {}).pop(int(m), None)
lines = [
    "# GB10 (SM121) plans selected by CUDA graph replay measurements with weights",
    "# read from memory. Every shape was measured at M = 1, 2, 4 and 8, and most",
    "# also at M = 3 and 16. A point is retained when its output error stays within",
    "# twice the standard linear implementation's BF16 error and it is at least 5%",
    "# faster; any other M uses the standard linear implementation. Comments name",
    "# the measured projection and give both times in microseconds.",
    "QWEN4_EXP_SM121_GEMM_PLANS: dict[tuple[int, int], dict[int, SkinnyGemmConfig]] = {",
]
for (n, k), by_m in sorted(plans.items()):
    if not by_m:
        continue
    lines.append(f"    # {', '.join(sorted({entry[1] for entry in by_m.values()}))}.")
    lines.append(f"    ({n}, {k}): {{")
    for m, (config, _, torch_us, best_us) in sorted(by_m.items()):
        rows, block, outputs, unroll, width = config
        lines.append(f"        {m}: SkinnyGemmConfig({rows}, {block}, {outputs}, k_unroll={unroll}, "
                     f"vector_width={width}),  # {torch_us} -> {best_us}")
    lines.append("    },")
lines.append("}")
SELECTORS = NEWLINE.join([
    "def _is_sm121() -> bool:",
    "    return current_platform.is_device_capability((12, 1))",
    "",
    "",
    "def applies_with_b12x() -> bool:",
    '    """Whether the low-latency hook may run beside B12X.',
    "",
    "    On SM121 the plans cover BF16 projections that B12X leaves to torch.",
    '    """',
    "    return _is_sm121()",
    "",
    "",
    "def _is_sm103() -> bool:",
])

gemm = image_file(DIRECTORY + "low_latency_gemm.py")
gemm = swap(gemm, "import torch" + NEWLINE, "import functools" + NEWLINE + NEWLINE + "import torch" + NEWLINE)
gemm = swap(gemm, "from vllm.utils.torch_utils import direct_register_custom_op" + NEWLINE,
            "from vllm.utils.torch_utils import LayerNameType, direct_register_custom_op" + NEWLINE)
gemm = swap(gemm, "def _is_sm103() -> bool:", NEWLINE.join(lines) + NEWLINE * 3 + SELECTORS)
sm90 = "    if _is_sm90():" + NEWLINE + "        return QWEN4_EXP_SM90_GEMM_PLANS" + NEWLINE
gemm = swap(gemm, sm90, sm90 + "    if _is_sm121():" + NEWLINE + "        return QWEN4_EXP_SM121_GEMM_PLANS" + NEWLINE)
gemm = gemm.rstrip(NEWLINE) + NEWLINE + HELPER
OUTPUT.mkdir(parents=True, exist_ok=True)
(OUTPUT / "low_latency_gemm.py").write_text(gemm)

environment = image_file(VLLM + "envs.py")
environment = swap(environment, ENVS_ANCHOR_TYPE, ENVS_ANCHOR_TYPE + "    VLLM_QWEN4_EXP_MXFP8_HC: bool = False" + NEWLINE)
environment = swap(environment, ENVS_ANCHOR_VALUE, ENVS_ANCHOR_VALUE + ENVS_VALUE)
(OUTPUT / "envs.py").write_text(environment)

for name, call in (("model.py", "enable_qwen4_exp_low_latency_gemm(self, self.model_config.dtype)"),
                   ("mtp.py", "enable_qwen4_exp_low_latency_gemm(self, vllm_config.model_config.dtype)")):
    text = image_file(DIRECTORY + name)
    text = swap(text, IMPORT_OLD, IMPORT_NEW)
    guarded = "        if not uses_b12x(vllm_config):" + NEWLINE + "            " + call + NEWLINE
    text = swap(text, guarded, "        quantize_hyperconnection_mixing(self)" + NEWLINE
                + "        if not uses_b12x(vllm_config) or applies_with_b12x():" + NEWLINE + "            " + call + NEWLINE)
    (OUTPUT / name).write_text(text)
print(json.dumps({"plans": sum(len(v) for v in plans.values()), "shapes": sum(1 for v in plans.values() if v)}))
