#!/usr/bin/env python3
"""Route the hybrid NVFP4 config's MXFP4 draft experts through the CUTLASS MoE method on SM90/SM120.

SGLang's HybridFp8NvFp4Config (nvidia/DeepSeek-V4*-NVFP4 checkpoints) sends FusedMoE layers that
stay MXFP4 -- the bundled DSpark draft experts -- to the TRT-LLM MoE method unconditionally. The
stock FP8 path (fp8.py) already selects the CUTLASS MXFP8xMXFP4 method on SM90/SM120, where no
TRT-LLM MoE kernel exists. This mirrors that choice for the hybrid config only; the NVFP4 routed
experts and every non-MoE layer are untouched.

Run with python -S in a CPU-only container; stdout is the patched module source.
"""
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text()
old = """            # Fall back to MXFP4 for MTP MoE layers
            if self.is_fp4_experts:
                from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
                from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
                    Mxfp4FlashinferTrtllmMoEMethod,
                )

                return Mxfp4FlashinferTrtllmMoEMethod(Fp8MoEMethod(self), prefix=prefix)
"""
new = """            # Fall back to MXFP4 for MTP MoE layers
            if self.is_fp4_experts:
                from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod

                # SparkRing: SM90/SM120 have no TRT-LLM MoE kernel; use the CUTLASS
                # MXFP8xMXFP4 method that fp8.py selects for the same platforms.
                if get_platform().is_sm90 or get_platform().is_sm120:
                    from sglang.srt.layers.quantization.mxfp4_flashinfer_cutlass_moe import (
                        Mxfp4FlashinferCutlassMoEMethod,
                    )

                    return Mxfp4FlashinferCutlassMoEMethod(Fp8MoEMethod(self), prefix=prefix)
                from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
                    Mxfp4FlashinferTrtllmMoEMethod,
                )

                return Mxfp4FlashinferTrtllmMoEMethod(Fp8MoEMethod(self), prefix=prefix)
"""
if source.count(old) != 1 or "get_platform" not in source:
    raise SystemExit("modelopt_quant source changed; refusing an unverified patch")
print(source.replace(old, new, 1), end="")
