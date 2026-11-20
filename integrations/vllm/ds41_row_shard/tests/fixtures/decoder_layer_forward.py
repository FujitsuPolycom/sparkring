# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Verbatim DeepseekV4DecoderLayer.forward from vllm/models/deepseek_v4_1/nvidia/model.py, lines 263-330,
# vLLM d51b4181acc06622357c81dc8e0be46ad96a0779 (file SHA-256 dafb69b36e68c36ba8878c6033e1bc94439526c3d1d9c83edd7c153ad2ba88c8).
# The tests execute it as the reference; test_image_sources.py checks it
# against an extracted image tree when SPARKRING_DS41_IMAGE_SOURCES is set.
# ruff: noqa: F821
def forward(
    self,
    x: torch.Tensor,
    positions: torch.Tensor,
    input_ids: torch.Tensor | None,
    pre_mix: torch.Tensor | None = None,
    post_mix: torch.Tensor | None = None,
    res_mix: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    engram_hashes: torch.Tensor | None = None,
    engram_mask: torch.Tensor | None = None,
    ced_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    previous_output = x if residual is not None else None
    apply_engram = self.engram is not None and engram_hashes is not None
    if residual is None:
        residual = x
    elif apply_engram:
        # Engram mutates the reconstructed residual; do not fuse across it.
        residual = self._b12x_mhc.post(x, residual, post_mix, res_mix)
        previous_output = None
    if apply_engram:
        if residual.ndim == 2:
            residual = (
                residual[:, None, :].expand(-1, self.hc_mult, -1).contiguous()
            )
        residual = self.engram(
            residual, engram_hashes[:, self.engram.layer_hash_index], engram_mask
        )
    fn = self.hc_attn_fn_broadcast if residual.ndim == 2 else self.hc_attn_fn
    residual, post_mix, res_mix, x, attn_pre = self._b12x_mhc.pre(
        residual,
        fn,
        self.hc_attn_scale,
        self.hc_attn_base,
        self.attn_norm.weight,
        pre_mix,
        previous_output=previous_output,
        previous_post=post_mix if previous_output is not None else None,
        previous_comb=res_mix if previous_output is not None else None,
    )
    global_kv_ready = None
    if ced_indices is not None:
        # Decoder global KV needs every encoder row, but decoder queries,
        # residual updates and experts only need the selected replay rows.
        global_kv_ready = self.attn.prepare_global_kv(positions, x)
        x = gather_rows(x, ced_indices)
        residual = gather_rows(residual, ced_indices)
        post_mix = gather_rows(post_mix, ced_indices)
        res_mix = gather_rows(res_mix, ced_indices)
        attn_pre = gather_rows(attn_pre, ced_indices)
        positions = gather_rows(positions, ced_indices)
        if input_ids is not None:
            input_ids = gather_rows(input_ids, ced_indices)
    x = self.attn(positions, x, None, global_kv_ready=global_kv_ready)
    residual, post_mix, res_mix, x, ffn_pre = self._b12x_mhc.post_pre(
        x,
        residual,
        post_mix,
        res_mix,
        self.hc_ffn_fn,
        self.hc_ffn_scale,
        self.hc_ffn_base,
        self.ffn_norm.weight,
        attn_pre,
    )
    x = self.ffn(x, input_ids)
    return x, residual, post_mix, res_mix, ffn_pre
