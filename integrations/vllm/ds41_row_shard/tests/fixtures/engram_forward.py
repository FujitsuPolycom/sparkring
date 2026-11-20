# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Verbatim Engram.forward from vllm/models/deepseek_v4_1/common/engram.py, lines 868-891,
# vLLM d51b4181acc06622357c81dc8e0be46ad96a0779 (file SHA-256 efda07bdd54d947852cdf75095905033038cf2a4425d7b234d092a9253c7c811).
# The tests execute it as the reference; test_image_sources.py checks it
# against an extracted image tree when SPARKRING_DS41_IMAGE_SOURCES is set.
# ruff: noqa: F821
def forward(self, hidden_states, hash_ids, token_mask=None):
    if (
        self.embed_tokens.table_memory == "disk"
        and not torch.compiler.is_compiling()
        and (
            not self._disk_prepared
            or self._disk_prepared_tokens < hash_ids.shape[0]
        )
    ):
        raise RuntimeError("Disk Engram output is not prepared")
    rows = tensor_model_parallel_all_reduce(self.staged_rows[: hash_ids.shape[0]])
    kv = self.wkv(rows)
    state = hidden_states.flatten(1)
    out = torch.empty_like(state)
    hyperconnection.run_engram_mix(
        state,
        kv,
        self.norm_weights,
        eps=self.eps,
        plan=self.mix_plans[token_mask is not None],
        out=out,
        token_mask=token_mask,
    )
    return out.view_as(hidden_states)
