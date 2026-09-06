# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-runner state for the GLM5Next pooled sparse-attention selector."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.config.compilation import CUDAGraphMode
from vllm.triton_utils import tl, triton
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.mamba_hybrid import (
    _checkpoint_plan_rows,
    MambaHybridAttnMetadata,
    MambaHybridModelState,
)
from vllm.v1.worker.utils import AttentionGroup


logger = init_logger(__name__)


@dataclass
class Glm5NextAttnMetadata(MambaHybridAttnMetadata):
    """Per-request state consumed by GLM5Next's b12x selector builder."""

    selector_state_slot_ids: torch.Tensor | None = None
    selector_state_is_fresh: torch.Tensor | None = None
    selector_num_accepted_tokens: torch.Tensor | None = None
    selector_is_prefilling: torch.Tensor | None = None

    def get_extra_attn_kwargs(
        self,
        attn_metadata_builder: Any,
        num_reqs: int,
    ) -> dict[str, Any]:
        kwargs = super().get_extra_attn_kwargs(attn_metadata_builder, num_reqs)
        if not getattr(
            attn_metadata_builder,
            "requires_glm_next_selector_metadata",
            False,
        ):
            return kwargs
        assert self.selector_state_slot_ids is not None
        assert self.selector_state_is_fresh is not None
        assert self.selector_num_accepted_tokens is not None
        assert self.selector_is_prefilling is not None
        kwargs.update(
            selector_state_slot_ids=self.selector_state_slot_ids[:num_reqs],
            selector_state_is_fresh=self.selector_state_is_fresh[:num_reqs],
            selector_num_accepted_tokens=(self.selector_num_accepted_tokens[:num_reqs]),
            selector_is_prefilling=self.selector_is_prefilling[:num_reqs],
        )
        return kwargs


class Glm5NextModelState(MambaHybridModelState):
    """Add persistent request identity for GLM5Next pooled selector state."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        config = self.model_config.hf_text_config
        self.uses_pooled_selector = getattr(config, "index_topk", None) is not None
        self.selector_pool_size = int(getattr(config, "index_kpool", 1) or 1)

        # These are fixed-capacity staging buffers. Their addresses remain stable
        # across request reordering and CUDA-graph capture/replay.
        self.selector_state_slot_ids = torch.full(
            (self.max_num_reqs,),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self.selector_state_is_fresh = torch.ones(
            self.max_num_reqs,
            dtype=torch.bool,
            device=self.device,
        )
        self.selector_num_accepted_tokens = torch.ones(
            self.max_num_reqs,
            dtype=torch.int32,
            device=self.device,
        )
        self.mamba_num_accepted_tokens = torch.ones(
            self.max_num_reqs,
            dtype=torch.int32,
            device=self.device,
        )
        self.selector_committed_num_accepted_tokens_gpu = torch.ones(
            self.max_num_reqs,
            dtype=torch.int32,
            device=self.device,
        )
        self.selector_state_is_fresh_gpu = torch.ones(
            self.max_num_reqs,
            dtype=torch.bool,
            device=self.device,
        )
        self.selector_is_prefilling = CpuGpuBuffer(
            self.max_num_reqs,
            dtype=torch.bool,
            device=self.device,
        )
        self._selector_draft_is_prefilling = torch.zeros(
            self.max_num_reqs,
            dtype=torch.bool,
            device="cpu",
        )
        self._selector_draft_is_prefilling_gpu = torch.zeros(
            self.max_num_reqs,
            dtype=torch.bool,
            device=self.device,
        )

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        prefix_length = int(new_req_data.num_computed_tokens)
        if self.uses_pooled_selector and prefix_length % self.selector_pool_size != 0:
            raise ValueError(
                "GLM5Next pooled selector cannot resume a fresh request from "
                f"num_computed_tokens={prefix_length}; the prefix length must be "
                f"divisible by index_kpool={self.selector_pool_size}."
            )
        super().add_request(req_index, new_req_data)
        if self.uses_pooled_selector:
            # The scheduler may recycle this request-state slot while selector
            # raw-ring tags and its interval anchor still belong to the prior owner.
            self.selector_state_is_fresh_gpu[req_index].fill_(True)
            self.selector_committed_num_accepted_tokens_gpu[req_index].fill_(1)

    def reset_kv_cache_state(self) -> None:
        super().reset_kv_cache_state()
        if self.uses_pooled_selector:
            self.selector_state_is_fresh_gpu.fill_(True)
            self.selector_committed_num_accepted_tokens_gpu.fill_(1)

    def _prepare_selector_state(
        self,
        input_batch: InputBatch,
        num_reqs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        slots = self.selector_state_slot_ids[:num_reqs]
        fresh = self.selector_state_is_fresh[:num_reqs]
        accepted = self.selector_num_accepted_tokens[:num_reqs]
        slots.fill_(-1)
        fresh.fill_(True)
        accepted.fill_(1)

        num_actual_reqs = input_batch.num_reqs
        if num_actual_reqs:
            idx_mapping = input_batch.idx_mapping[:num_actual_reqs]
            slots[:num_actual_reqs].copy_(idx_mapping)
            torch.index_select(
                self.selector_state_is_fresh_gpu,
                0,
                idx_mapping,
                out=fresh[:num_actual_reqs],
            )
            torch.index_select(
                self.selector_committed_num_accepted_tokens_gpu,
                0,
                idx_mapping,
                out=accepted[:num_actual_reqs],
            )
        return slots, fresh, accepted

    def _prepare_mamba_acceptance(
        self,
        input_batch: InputBatch,
        num_reqs: int,
    ) -> torch.Tensor:
        accepted = self.mamba_num_accepted_tokens[:num_reqs]
        accepted.fill_(1)
        num_actual_reqs = input_batch.num_reqs
        if num_actual_reqs:
            torch.index_select(
                self.num_accepted_tokens_gpu,
                0,
                input_batch.idx_mapping[:num_actual_reqs],
                out=accepted[:num_actual_reqs],
            )
        return accepted

    def prepare_draft_attn_metadata(
        self,
        *,
        idx_mapping: torch.Tensor,
        num_reqs: int,
        num_reqs_padded: int,
        draft_index: int,
    ) -> Glm5NextAttnMetadata | None:
        if not self.uses_pooled_selector:
            return None
        if draft_index < 1:
            raise RuntimeError(
                "GLM5Next supports pooled-selector draft metadata only for "
                "autoregressive MTP lookahead (draft_index >= 1)"
            )
        if not 0 <= num_reqs <= num_reqs_padded <= self.max_num_reqs:
            raise ValueError(
                "draft request counts must satisfy "
                "0 <= num_reqs <= num_reqs_padded <= max_num_reqs"
            )
        if idx_mapping.numel() < num_reqs:
            raise ValueError("idx_mapping does not cover every active draft request")

        slots = self.selector_state_slot_ids[:num_reqs_padded]
        fresh = self.selector_state_is_fresh[:num_reqs_padded]
        accepted = self.selector_num_accepted_tokens[:num_reqs_padded]
        slots.fill_(-1)
        fresh.fill_(True)
        accepted.fill_(1)
        if num_reqs:
            active_slots = idx_mapping[:num_reqs]
            slots[:num_reqs].copy_(active_slots)
            # Draft prefill uses the target metadata immediately before the
            # lookahead loop and has initialized these MTP selector slots.
            fresh[:num_reqs].fill_(False)
            if draft_index == 1:
                torch.index_select(
                    self.selector_committed_num_accepted_tokens_gpu,
                    0,
                    active_slots,
                    out=accepted[:num_reqs],
                )

        return Glm5NextAttnMetadata(
            is_prefilling=self._selector_draft_is_prefilling[:num_reqs_padded],
            num_accepted_tokens=accepted,
            selector_state_slot_ids=slots,
            selector_state_is_fresh=fresh,
            selector_num_accepted_tokens=accepted,
            selector_is_prefilling=(
                self._selector_draft_is_prefilling_gpu[:num_reqs_padded]
            ),
        )

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        recurrent_prefill_checkpoint_plans: dict | None = None,
    ) -> dict[str, Any]:
        # This is the MambaHybridModelState construction with only the metadata
        # object specialized. Keeping it package-local avoids a GLM hook in the
        # generic model runner.
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens
        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        max_query_len = input_batch.num_scheduled_tokens.max().item()
        seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound
        if for_capture:
            max_seq_len = self.max_model_len
        else:
            max_seq_len = seq_lens_cpu_upper_bound[:num_reqs].max().item()

        self.selector_is_prefilling.np[:num_reqs] = False
        self.selector_is_prefilling.np[: input_batch.num_reqs] = (
            input_batch.is_prefilling_np
        )
        is_prefilling = self.selector_is_prefilling.cpu[:num_reqs]
        selector_is_prefilling = self.selector_is_prefilling.copy_to_gpu(num_reqs)
        (
            selector_state_slot_ids,
            selector_state_is_fresh,
            selector_num_accepted_tokens,
        ) = self._prepare_selector_state(input_batch, num_reqs)

        # During CUDA-graph capture the builders create their own neutral
        # speculative metadata. Runtime calls stage the persistent buffers.
        num_accepted_tokens = None
        num_decode_draft_tokens_cpu = None
        if not for_capture and self.vllm_config.num_speculative_tokens > 0:
            # Mamba page alignment can reset its accepted-token offset after
            # moving state. The selector advances an independent pooled interval.
            num_accepted_tokens = self._prepare_mamba_acceptance(
                input_batch,
                num_reqs,
            )
            num_decode_draft_tokens_np = np.full(num_reqs, -1, dtype=np.int32)
            num_draft_tokens_per_req = input_batch.num_draft_tokens_per_req
            if num_draft_tokens_per_req is not None:
                is_decode = (
                    input_batch.num_scheduled_tokens == num_draft_tokens_per_req + 1
                )
                spec_decode_mask = (num_draft_tokens_per_req > 0) & is_decode
                num_decode_draft_tokens_np[: input_batch.num_reqs] = np.where(
                    spec_decode_mask,
                    num_draft_tokens_per_req,
                    -1,
                )
            num_decode_draft_tokens_cpu = torch.from_numpy(num_decode_draft_tokens_np)

        if self._align_mode:
            self._prepare_aligned_state_indices(
                input_batch.seq_lens,
                num_reqs,
                attn_groups,
                kv_cache_config,
                block_tables,
            )

        checkpoint_plans_cpu = _checkpoint_plan_rows(
            input_batch, recurrent_prefill_checkpoint_plans, num_reqs,
            for_capture=for_capture,
        )
        model_metadata = Glm5NextAttnMetadata(
            recurrent_prefill_checkpoint_plans_cpu=checkpoint_plans_cpu,
            is_prefilling=is_prefilling,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
            selector_state_slot_ids=selector_state_slot_ids,
            selector_state_is_fresh=selector_state_is_fresh,
            selector_num_accepted_tokens=selector_num_accepted_tokens,
            selector_is_prefilling=selector_is_prefilling,
        )
        attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=max_seq_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            model_specific_attn_metadata=model_metadata,
            for_cudagraph_capture=for_capture,
            rswa_prefix_lens=input_batch.prompt_lens,
        )
        if checkpoint_plans_cpu is not None:
            gdn_metadata = [metadata for metadata in attn_metadata.values()
                            if hasattr(metadata, "prefill_checkpoint")]
            if not gdn_metadata or any(metadata.prefill_checkpoint is None
                                      or metadata.prefill_checkpoint.required_mask is None
                                      for metadata in gdn_metadata):
                raise RuntimeError("GLM V2 scheduled checkpoint plans without GDN export metadata")
            if not getattr(self, "_explicit_checkpoint_plan_logged", False):
                logger.info("GLM V2 explicit recurrent checkpoint plan active: rows=%s layers=%d",
                            checkpoint_plans_cpu, len(gdn_metadata))
                self._explicit_checkpoint_plan_logged = True
        if self.recoverssm is not None:
            self.recoverssm.record_step(
                attn_metadata,
                attn_groups,
                for_capture=for_capture,
            )
        return attn_metadata

    def postprocess_state(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor | int,
        num_computed_tokens: torch.Tensor | None = None,
    ) -> None:
        if self.uses_pooled_selector and idx_mapping.numel():
            if isinstance(num_sampled, int):
                _fill_selector_request_state_kernel[(idx_mapping.numel(),)](
                    idx_mapping,
                    self.selector_committed_num_accepted_tokens_gpu,
                    self.selector_state_is_fresh_gpu,
                    max(num_sampled, 1),
                )
            else:
                _commit_selector_request_state_kernel[(idx_mapping.numel(),)](
                    idx_mapping,
                    num_sampled,
                    self.selector_committed_num_accepted_tokens_gpu,
                    self.selector_state_is_fresh_gpu,
                )
        super().postprocess_state(idx_mapping, num_sampled, num_computed_tokens)


@triton.jit
def _commit_selector_request_state_kernel(
    idx_mapping_ptr,
    num_sampled_ptr,
    selector_num_accepted_ptr,
    state_is_fresh_ptr,
):
    row = tl.program_id(0)
    state_slot = tl.load(idx_mapping_ptr + row)
    if state_slot >= 0:
        num_sampled = tl.load(num_sampled_ptr + row)
        tl.store(
            selector_num_accepted_ptr + state_slot,
            tl.maximum(num_sampled, 1),
        )
        tl.store(state_is_fresh_ptr + state_slot, 0)


@triton.jit
def _fill_selector_request_state_kernel(
    idx_mapping_ptr,
    selector_num_accepted_ptr,
    state_is_fresh_ptr,
    num_sampled,
):
    row = tl.program_id(0)
    state_slot = tl.load(idx_mapping_ptr + row)
    if state_slot >= 0:
        tl.store(selector_num_accepted_ptr + state_slot, num_sampled)
        tl.store(state_is_fresh_ptr + state_slot, 0)


__all__ = ["Glm5NextAttnMetadata", "Glm5NextModelState"]
