"""Resolve the hunks of vllm#955 and b12x#449 that do not apply to the image's sources.

Run from a directory holding the parent image's site-packages ``vllm/`` and
``b12x/`` files after ``git apply --reject`` of the two pull-request diffs
(vllm#955 at 9b4f167a, b12x#449 at b8e242af, runtime files only). Eight hunks
are rejected there because the image's sources rewrote their context lines:
b12x master a7d7d29b generalized IQ2_XS handling to every block codec and added
IQ2_XXS and Q8_0, and the image's vLLM wraps one error message differently.
Each edit below places the pull request's added lines next to the rewritten
line, changing nothing else; every anchor must occur exactly once.
"""
from pathlib import Path


def edit(path, old, new):
    p = Path(path)
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one anchor, found {count}: {old[:70]!r}")
    p.write_text(text.replace(old, new))
    print("resolved", path)


# vllm#955 b12x.py hunk 4: error message names MXFP8 (image wraps the string differently).
edit("vllm/model_executor/layers/fused_moe/b12x.py",
     '                "b12x MoE requires MXFP4, NVFP4, EXL3, IQ2_XS, IQ2_XXS "\n'
     '                "or Q8_0 weights, got "\n',
     '                "b12x MoE requires MXFP4, MXFP8, NVFP4, EXL3, IQ2_XS, IQ2_XXS "\n'
     '                "or Q8_0 weights, got "\n')

# b12x#449 source.py hunk 1: new packed source format after MXFP6.
edit("b12x/moe/fused_moe/source.py",
     '    MXFP6_E8M0_K32 = "mxfp6_e2m3"\n',
     '    MXFP6_E8M0_K32 = "mxfp6_e2m3"\n    MXFP8_E8M0_K32 = "mxfp8_e8m0_k32"\n')

# b12x#449 weights.py hunk 1: new weight encoding after FP6.
edit("b12x/moe/fused_moe/weights.py",
     '    FP6_E2M3 = "fp6_e2m3"\n    TRELLIS = "trellis"\n',
     '    FP6_E2M3 = "fp6_e2m3"\n    FP8_E4M3 = "fp8_e4m3"\n    TRELLIS = "trellis"\n')

# b12x#449 _tuning.py hunk 1: docstring only.
edit("b12x/moe/fused_moe/_tuning.py",
     '    _device: DeviceIdentity | None,\n) -> None:\n    _validate_block_moe_launch(query, config)\n',
     '    _device: DeviceIdentity | None,\n) -> None:\n'
     '    """Reject a decode config the query\'s recipe cannot execute."""\n'
     '    _validate_block_moe_launch(query, config)\n')

# b12x#449 planning.py hunk 4: FP8 weight encoding for the MXFP8 source.
edit("b12x/moe/fused_moe/planning.py",
     '            WeightEncoding.FP6_E2M3\n'
     '            if source.format.value == "mxfp6_e2m3"\n'
     '            else WeightEncoding.FP4_E2M1\n',
     '            WeightEncoding.FP6_E2M3\n'
     '            if source.format.value == "mxfp6_e2m3"\n'
     '            else WeightEncoding.FP8_E4M3\n'
     '            if source.format.value == "mxfp8_e8m0_k32"\n'
     '            else WeightEncoding.FP4_E2M1\n')

# b12x#449 execution.py hunk 2: quant mode and source format sets.
edit("b12x/moe/_shared/execution.py",
     '_QUANT_MODES = {"nvfp4", "w4a16", "w4a8_mx", "w4a8_nvfp4", "w6a8_mx"}\n'
     '_SOURCE_FORMATS = {\n'
     '    "modelopt_nvfp4",\n'
     '    "fp4_e8m0_k32",\n'
     '    "compressed_tensors",\n'
     '    "mxfp6_e2m3",\n',
     '_QUANT_MODES = {"nvfp4", "w4a16", "w4a8_mx", "w4a8_nvfp4", "w6a8_mx", "w8a8_mx"}\n'
     '_SOURCE_FORMATS = {\n'
     '    "modelopt_nvfp4",\n'
     '    "fp4_e8m0_k32",\n'
     '    "compressed_tensors",\n'
     '    "mxfp6_e2m3",\n'
     '    "mxfp8_e8m0_k32",\n')

# b12x#449 execution.py hunk 7: MXFP8 weight operand encoding for w8a8_mx.
edit("b12x/moe/_shared/execution.py",
     '            OperandEncoding.FP6_E2M3\n'
     '            if quant_mode == "w6a8_mx"\n'
     '            else OperandEncoding.FP4_E2M1\n',
     '            OperandEncoding.FP6_E2M3\n'
     '            if quant_mode == "w6a8_mx"\n'
     '            else OperandEncoding.MXFP8_E4M3\n'
     '            if quant_mode == "w8a8_mx"\n'
     '            else OperandEncoding.FP4_E2M1\n')

# b12x#449 execution.py hunk 8: w8a8_mx preparation plan, before the w4a16 branch.
W8A8_BLOCK = '''        if spec.quant_mode == "w8a8_mx":
            if spec.activation != "silu":
                raise ValueError("W8A8-MXFP8 preparation requires silu")
            # FC1 streams 128-wide K tiles over hidden (one E4M3 byte per
            # element), so hidden must be 128-aligned.  The intermediate is
            # the FC2 K extent and the FC1 N extent: FC2 needs whole UE8M0
            # K/32 blocks, and non-128 FC1 halves use independent up/gate
            # TMA descriptors whose tail tile is zero-filled by TMA.
            if hidden_size % 128 != 0 or intermediate_size % 32 != 0:
                raise ValueError(
                    "W8A8-MXFP8 preparation requires hidden_size % 128 == 0 "
                    "and intermediate_size % 32 == 0"
                )
            # Weight bytes are preserved losslessly (no FP4/FP6 requantization);
            # UE8M0 scales are swizzled into the MMA layout and per-expert
            # alphas are derived from the checkpoint's *_weight_scale_2 globals.
            transforms.add(WeightPreparationTransform.W8A8_MXFP8)
            weight_layouts.add(PreparedWeightLayout.SOURCE_NATIVE)
            scale_layouts.update(
                {
                    PreparedScaleLayout.MMA_PACKED,
                    PreparedScaleLayout.RUNTIME_ALPHA,
                }
            )
            continue
'''
edit("b12x/moe/_shared/execution.py",
     '            continue\n'
     '        if spec.quant_mode == "w4a16":\n'
     '            if source_format in BLOCK_CODECS:\n',
     '            continue\n' + W8A8_BLOCK +
     '        if spec.quant_mode == "w4a16":\n'
     '            if source_format in BLOCK_CODECS:\n')
