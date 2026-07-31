# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from dataclasses import dataclass

import numpy as np
import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.compressor_utils import get_compressed_slot_mapping
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
from vllm.v1.kv_cache_interface import AttentionSpec, MLAAttentionSpec
from vllm.v1.worker.cp_utils import get_total_cp_world_size

logger = init_logger(__name__)

_B12X_PAGED_INDEX_SUPERTILE_K_DEFAULT = 32768
_B12X_PAGED_INDEX_TILE_BLOCK_K = 512


def _get_b12x_paged_indexer_supertile_k() -> int:
    raw = os.environ.get("B12X_PAGED_INDEX_SUPERTILE_K")
    tokens = _B12X_PAGED_INDEX_SUPERTILE_K_DEFAULT if raw is None else int(raw)
    tokens = max(tokens, _B12X_PAGED_INDEX_TILE_BLOCK_K)
    return (
        (tokens + _B12X_PAGED_INDEX_TILE_BLOCK_K - 1)
        // _B12X_PAGED_INDEX_TILE_BLOCK_K
        * _B12X_PAGED_INDEX_TILE_BLOCK_K
    )

@triton.jit
def _prepare_uniform_decode_kernel(
    seq_lens_ptr,
    decode_seq_lens_ptr,
    block_table_ptr,
    block_table_stride,
    expanded_block_table_ptr,
    expanded_bt_stride,
    decode_lens_ptr,
    max_decode_len,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0)
    req_id = idx // max_decode_len
    local_idx = idx % max_decode_len

    # Compute number of KVs attended to by this token.
    seq_len = tl.load(seq_lens_ptr + req_id)
    per_token_seq_len = seq_len - max_decode_len + local_idx + 1
    tl.store(decode_seq_lens_ptr + idx, per_token_seq_len)

    # Copy block table row.
    src = block_table_ptr + req_id * block_table_stride
    dst = expanded_block_table_ptr + idx * expanded_bt_stride
    for i in tl.range(0, expanded_bt_stride, BLOCK_SIZE):
        off = i + tl.arange(0, BLOCK_SIZE)
        mask = off < expanded_bt_stride
        src_block = tl.load(src + off, mask=mask)
        tl.store(dst + off, src_block, mask=mask)

    # All reqs now have decode_len = 1.
    tl.store(decode_lens_ptr + idx, 1)


def split_indexer_prefill_chunks(
    seq_lens_cpu: torch.Tensor | np.ndarray,
    query_lens_cpu: torch.Tensor | np.ndarray,
    workspace_size: int,
    max_logits_bytes: int,
    request_offset: int = 0,
) -> list[tuple[slice, slice]]:
    """
    Split prefill requests into chunks for the sparse indexer, respecting:
    - N constraint: total_seq_lens <= workspace_size (existing O(N) workspace)
    - Logits constraint: M * N * 4 <= max_logits_bytes

    When a single request-level chunk still exceeds the logits budget,
    sub-chunks on the query dimension (M) to bound peak memory.

    Returns list of (req_slice, query_slice) tuples.
    """
    if isinstance(seq_lens_cpu, torch.Tensor):
        seq_lens_cpu = seq_lens_cpu.numpy()
    if isinstance(query_lens_cpu, torch.Tensor):
        query_lens_cpu = query_lens_cpu.numpy()

    chunks: list[tuple[slice, slice]] = []
    n = len(seq_lens_cpu)
    max_logits_elems = max_logits_bytes // 4
    end = 0

    while end < n:
        start, chunk_m, chunk_n = end, 0, 0

        while end < n:
            q, s = int(query_lens_cpu[end]), int(seq_lens_cpu[end])
            new_m, new_n = chunk_m + q, chunk_n + s
            if new_n <= workspace_size and new_m * new_n <= max_logits_elems:
                chunk_m, chunk_n = new_m, new_n
                end += 1
            else:
                break

        # A single request can exceed the budget, requiring sub-chunking
        # on the query dimension.
        if end == start:
            chunk_m, chunk_n = int(query_lens_cpu[end]), int(seq_lens_cpu[end])
            end += 1

        req_slice = slice(start + request_offset, end + request_offset)
        max_q = max(1, max_logits_elems // chunk_n) if chunk_n > 0 else chunk_m
        for q_off in range(0, chunk_m, max_q):
            sub_m = min(max_q, chunk_m - q_off)
            chunks.append((req_slice, slice(q_off, q_off + sub_m)))

    return chunks


class DeepseekV32IndexerBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "DEEPSEEK_V32_INDEXER"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [1, 64] if current_platform.is_rocm() else [64]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [32, 64, 128]

    @staticmethod
    def get_builder_cls() -> type["DeepseekV32IndexerMetadataBuilder"]:
        return DeepseekV32IndexerMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        assert num_kv_heads == 1
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            # DeepseekV32Indexer kernels do not support cross-layer
            # KV cache layout. Identity permutation keeps num_layers
            # first, signaling incompatibility.
            return (0, 1, 2, 3)
        return (0, 1, 2)


class B12xNonCompressedIndexerBackend(DeepseekV32IndexerBackend):
    @staticmethod
    def get_name() -> str:
        return "B12X_NON_COMPRESSED_INDEXER"


class DeepseekV4IndexerBackend(DeepseekV32IndexerBackend):
    @staticmethod
    def get_name() -> str:
        return "DEEPSEEK_V4_INDEXER"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [256]


@dataclass
class DeepseekV32IndexerPrefillChunkMetadata:
    block_table: torch.Tensor
    cu_seqlen_ks: torch.Tensor
    cu_seqlen_ke: torch.Tensor
    cu_seq_lens: torch.Tensor
    token_to_seq: torch.Tensor
    total_seq_lens: int
    token_start: int
    token_end: int
    num_reqs: int
    skip_kv_gather: bool = False


@dataclass
class DeepseekV32IndexerPrefillMetadata:
    chunks: list[DeepseekV32IndexerPrefillChunkMetadata]


@dataclass
class DeepSeekV32IndexerDecodeMetadata:
    block_table: torch.Tensor
    # seq_lens: per-token effective context lengths.
    #   - flatten path / plain decode: 1D (batch_size,)
    #   - native MTP path: 2D (B, next_n) where [b,j] = L_b - next_n + j + 1
    # Both fp8_fp4_paged_mqa_logits and the topk kernels accept both shapes.
    seq_lens: torch.Tensor
    max_seq_len: int
    decode_lens: torch.Tensor
    requires_padding: bool
    schedule_metadata: torch.Tensor | None
    compress_ratio: int = 1
    # Live scorer window (max compressed context across the batch) in cache
    # tokens, computed host-side in build() — a metadata tensor read by the
    # captured indexer kernel, never an in-kernel reduction. None => b12x uses
    # the capacity cap.
    active_width: torch.Tensor | None = None


@dataclass
class DeepseekV32IndexerMetadata:
    # FIXME (zyongye)
    # hacky way to access the data now, need to be in chunked meta
    seq_lens: torch.Tensor
    max_seq_len: int
    slot_mapping: torch.Tensor

    # New for MLA (compared to FlashAttention)
    # For handling prefill decode split
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int

    decode: DeepSeekV32IndexerDecodeMetadata | None = None
    prefill: DeepseekV32IndexerPrefillMetadata | None = None
    dcp_world_size: int = 1
    dcp_rank: int = 0
    cp_interleave_size: int = 1


def get_max_prefill_buffer_size(vllm_config: VllmConfig):
    max_model_len = vllm_config.model_config.max_model_len
    # NOTE(Chen): 40 is a magic number for controlling the prefill buffer size.
    # Each entry is 128 fp8 bytes and 4 scale bytes for a total of 132 bytes.
    # The flashmla_sparse backend uses a workspace size of 5 * max_model_len.
    # The memory usage of the workspace there is 576 * 2 bytes; so we size this as
    # (576 * 2 // 132) * 5 = 40 to maximize this workspace size while still fitting
    # within the flashmla_sparse workspace.
    # For DeepSeek-V3.2, the max_model_len is 163840.
    #   40 * 163840 * 132 = 865075200 bytes = 825 MB
    return max_model_len * 40


class DeepseekV32IndexerMetadataBuilder(AttentionMetadataBuilder):
    reorder_batch_threshold: int = 1

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        return AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.storage_block_size = int(self.kv_cache_spec.storage_block_size)
        scheduler_config = self.vllm_config.scheduler_config
        parallel_config = self.vllm_config.parallel_config
        self.dcp_world_size = parallel_config.decode_context_parallel_size
        self.pcp_world_size = parallel_config.prefill_context_parallel_size
        self.cp_kv_cache_interleave_size = (
            parallel_config.cp_kv_cache_interleave_size
        )
        self.dcp_rank = 0
        if self.dcp_world_size > 1:
            assert self.pcp_world_size == 1, (
                "DeepseekV32IndexerMetadataBuilder supports DCP but not PCP."
            )
            from vllm.distributed.parallel_state import get_dcp_group

            self.dcp_rank = get_dcp_group().rank_in_group
        # NOTE(Chen):an estimated max size of flattened_kv. Need to double check.
        self.max_prefill_buffer_size = get_max_prefill_buffer_size(self.vllm_config)
        self.num_speculative_tokens = (
            self.vllm_config.speculative_config.num_speculative_tokens
            if self.vllm_config.speculative_config
            else 0
        )
        self.use_fp4_indexer_cache = (
            self.vllm_config.attention_config.use_fp4_indexer_cache
        )

        assert (
            current_platform.is_device_capability_family(100)
            or not self.use_fp4_indexer_cache
        ), (
            "use_fp4_indexer_cache requires Blackwell datacenter GPUs "
            "(sm_10x, e.g. B200/GB200); sm_120 (consumer Blackwell) and "
            "earlier architectures are not supported."
        )

        next_n = self.num_speculative_tokens + 1
        # Match sparse_swa/flashmla_sparse decode split under the upstream
        # DSpark impl (parallel drafting doubles the max decode query len).
        # The sm_120 flattening path below handles arbitrary decode lens.
        import os as _os

        _spec_mult = (
            2
            if (
                self.vllm_config.speculative_config is not None
                and self.vllm_config.speculative_config.parallel_drafting
                and _os.environ.get("VLLM_DSPARK_IMPL", "fork") == "upstream"
            )
            else 1
        )
        self.reorder_batch_threshold += _spec_mult * self.num_speculative_tokens
        # NOTE: SM100 datacenter GPUs support any next_n natively via the
        # multi-atom paged MQA logits kernels (FP8 and FP4 indexer
        # caches). Outside the SM100 family the FP8
        # paged MQA logits kernel only supports next_n in (1, 2)
        # (deepgemm smxx_fp8_fp4_paged_mqa_logits.hpp:233), so flatten there.
        # The B12X / sparkinfer sparse indexer handles native next_n>2 on SM120
        # directly (see sparse_attn_indexer warmup q_rows 1, 2, 4), so it must
        # NOT be forced onto the DeepGEMM next_n<=2 flatten fallback. Flattening
        # MTP-2/MTP-3 verification into rank-1 rows produced subtly wrong accepted
        # tokens (code-gen syntax errors); the native (B, next_n) path keeps MTP
        # correct. Use the canonical backend-aware predicate so this also holds
        # when B12X is selected via --attention-backend B12X_MLA_SPARSE with the
        # VLLM_USE_B12X_SPARSE_INDEXER env var unset (it also asserts SM120).
        from vllm.model_executor.layers.sparse_attn_indexer import (
            use_b12x_sparse_indexer,
        )

        self.use_flattening = (
            not current_platform.is_device_capability_family(100)
            and next_n not in (1, 2)
            and not use_b12x_sparse_indexer()
        )
        logger.info_once(
            "DSA indexer decode path: use_flattening=%s "
            "(next_n=%d, use_fp4_indexer_cache=%s)",
            self.use_flattening,
            next_n,
            self.use_fp4_indexer_cache,
        )

        sm_count = num_compute_units(self.device.index)
        self.num_sms = sm_count

        self.offsets_buffer = torch.arange(
            next_n, device=self.device, dtype=torch.int32
        )
        self.decode_lens_buffer = torch.zeros(
            (scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=self.device,
        )
        # Shared workspace for decode seq_lens. Native MTP views this as
        # (B, max_decode_len) at runtime, keeping context_lens contiguous even
        # when max_decode_len is smaller than next_n.
        self.decode_seq_lens_buffer = torch.zeros(
            (scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=self.device,
        )
        self.arange_buffer = torch.arange(
            max(
                scheduler_config.max_num_seqs * next_n,
                scheduler_config.max_num_batched_tokens,
            ),
            dtype=torch.int32,
            device=self.device,
        )
        max_num_blocks_per_req = cdiv(
            self.vllm_config.model_config.max_model_len,
            self.kv_cache_spec.block_size * get_total_cp_world_size(),
        )
        self.expanded_block_table_buffer = torch.zeros(
            (
                scheduler_config.max_num_batched_tokens,
                max_num_blocks_per_req,
            ),
            dtype=torch.int32,
            device=self.device,
        )

        # See: DeepGMM/csrc/apis/attention.hpp
        self.scheduler_metadata_buffer = torch.empty(
            (self.num_sms + 1, 2), dtype=torch.int32, device=self.device
        )

        # Persistent live-active-width buffer for the b12x indexer decode
        # scorer window. Filled host-side each build() (outside cudagraph
        # capture) and read by the captured kernel at a stable address.
        self.b12x_active_width_buffer = torch.zeros(
            (1,), dtype=torch.int32, device=self.device
        )

        # KV compression. Default to 1 for no compression.
        self.compress_ratio = 1
        # Get compress_ratio for DeepseekV4 support
        if isinstance(self.kv_cache_spec, MLAAttentionSpec):
            self.compress_ratio = self.kv_cache_spec.compress_ratio

        _prewarm_prefill_chunk_metadata_kernel(
            device=self.device,
            dcp_world_size=self.dcp_world_size,
            dcp_rank=self.dcp_rank,
            cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
            compress_ratio=self.compress_ratio,
        )

        # DCP writes the indexer KV cache through rank-local pages even when
        # compress_ratio == 1 (GLM/Kimi). Keep the mapped slots graph-stable.
        if self.compress_ratio > 1 or self.dcp_world_size > 1:
            self.compressed_slot_mapping_buffer = torch.zeros(
                (scheduler_config.max_num_batched_tokens,),
                dtype=torch.int64,
                device=self.device,
            )

        # Pre-allocate buffers for CUDA graph compatibility when
        if self.compress_ratio > 1:
            # compress_ratio > 1 (DeepseekV4)
            # Buffer for compressed seq_lens in decode path
            self.expanded_seq_lens_buffer = torch.zeros(
                (scheduler_config.max_num_batched_tokens,),
                dtype=torch.int32,
                device=self.device,
            )

    def _maybe_build_b12x_schedule_metadata(
        self,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        num_decode_tokens: int,
        requires_padding: bool,
    ) -> torch.Tensor | None:
        if not envs.VLLM_USE_B12X_SPARSE_INDEXER or requires_padding:
            return None

        schedule_seq_lens = seq_lens
        if schedule_seq_lens.dim() == 2:
            batch_size, next_n = schedule_seq_lens.shape
            if num_decode_tokens != int(batch_size * next_n):
                return None
            schedule_seq_lens = schedule_seq_lens.reshape(-1)
        if schedule_seq_lens.dim() != 1:
            return None

        from b12x.attention.indexer import (
            build_paged_mqa_schedule_metadata,
            uses_paged_mqa_schedule,
        )

        if not uses_paged_mqa_schedule(
            q_rows=int(schedule_seq_lens.shape[0]),
            max_pages=int(block_table.shape[1]),
        ):
            return None

        return build_paged_mqa_schedule_metadata(
            schedule_seq_lens.contiguous(),
            self.storage_block_size,
            self.num_sms,
            out=self.scheduler_metadata_buffer,
        )

    def _maybe_build_deep_gemm_schedule_metadata(
        self,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        if current_platform.is_cuda():
            from vllm.utils.deep_gemm import (
                get_paged_mqa_logits_metadata,
                has_deep_gemm,
            )

            if has_deep_gemm():
                self.scheduler_metadata_buffer[:] = get_paged_mqa_logits_metadata(
                    seq_lens,
                    self.storage_block_size,
                    self.num_sms,
                )
        return self.scheduler_metadata_buffer

    def _decode_topk_max_seq_len_from_cpu(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        num_decodes: int,
        decode_lens_np: np.ndarray,
        max_decode_len: int,
        use_native: bool,
        dcp_local_seq_lens_cpu: torch.Tensor | None,
    ) -> int | None:
        """Return the exact decode max seq-len without reading CUDA tensors.

        B12X needs a host scalar for metadata/scheduling.  Computing it with
        ``seq_lens.max().item()`` synchronizes every decode step.  The model
        runner already maintains CPU shadows for normal decode; use those
        directly and mirror only the shape transforms that can affect the max.
        If async-spec has made the CPU shadow non-authoritative, return None so
        the caller can fall back to a safe graph-stable bound.
        """
        cpu_seq_lens = (
            dcp_local_seq_lens_cpu
            if dcp_local_seq_lens_cpu is not None
            else common_attn_metadata._seq_lens_cpu
        )
        if cpu_seq_lens is None:
            return None
        if cpu_seq_lens.device.type != "cpu":
            return None

        seq_lens_np = cpu_seq_lens[:num_decodes].numpy()
        if seq_lens_np.size == 0:
            return 0

        # Flattened variable-length MTP writes only real decode tokens and
        # zero-fills the tail.  Ignore padded decode rows to match that max.
        if not use_native and max_decode_len > 1:
            valid_decode_rows = decode_lens_np[:num_decodes] > 0
            if not bool(np.any(valid_decode_rows)):
                return 0
            seq_lens_np = seq_lens_np[valid_decode_rows]

        max_seq_len = int(seq_lens_np.max())
        if self.compress_ratio > 1:
            max_seq_len //= self.compress_ratio
        return max_seq_len

    def _prepare_decode_tensors(
        self,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        decode_lens: torch.Tensor,
        decode_lens_np: np.ndarray,
        query_start_loc: torch.Tensor,
        num_decodes: int,
        num_decode_tokens: int,
        use_native: bool,
        next_n: int,
        max_decode_len: int,
        global_seq_lens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, bool]:
        """Expand seq_lens/block_table/decode_lens for the decode kernels.

        Flatten path (not use_native, max_decode_len > 1):
          Each multi-token decode request is expanded into individual
          single-token entries so the kernel always sees next_n=1.

        Native path (use_native or max_decode_len == 1):
          Plain decode or spec-decode with 2D per-token context lengths.

        Returns (seq_lens, block_table, decode_lens, batch_size, requires_padding).
        seq_lens is 1D (batch_size,) for flatten/plain, 2D (B, max_decode_len)
        for native MTP.
        """
        min_decode_len = int(decode_lens_np.min())
        if not use_native and max_decode_len > 1:
            assert self.decode_seq_lens_buffer.dim() == 1
            if min_decode_len == max_decode_len:
                # Uniform decode lengths.
                num_decode_tokens = num_decodes * max_decode_len
                _prepare_uniform_decode_kernel[(num_decode_tokens,)](
                    seq_lens,
                    self.decode_seq_lens_buffer,
                    block_table,
                    block_table.stride(0),
                    self.expanded_block_table_buffer,
                    self.expanded_block_table_buffer.stride(0),
                    self.decode_lens_buffer,
                    max_decode_len,
                    BLOCK_SIZE=1024,
                )
                if global_seq_lens is not None and self.dcp_world_size > 1:
                    expanded_global = (
                        global_seq_lens[:num_decodes].unsqueeze(1)
                        - max_decode_len
                        + 1
                        + self.offsets_buffer[:max_decode_len]
                    ).reshape(-1)
                    self.decode_seq_lens_buffer[:num_decode_tokens].copy_(
                        get_dcp_local_seq_lens(
                            expanded_global,
                            self.dcp_world_size,
                            self.dcp_rank,
                            self.cp_kv_cache_interleave_size,
                        ),
                        non_blocking=True,
                    )
                self.decode_seq_lens_buffer[num_decode_tokens:] = 0
                seq_lens = self.decode_seq_lens_buffer[:num_decode_tokens]
                block_table = self.expanded_block_table_buffer[:num_decode_tokens]
                decode_lens = self.decode_lens_buffer[:num_decode_tokens]
                return seq_lens, block_table, decode_lens, num_decode_tokens, False
            else:
                # Variable decode lengths.
                # Assume 4 requests with seq_lens [10, 7, 12, 0] (the final req is
                # padding) and decode_lens [3, 1, 4, 0] in the below example comments.
                # The context lengths are therefore
                # [10-3, 7-1, 12-4, 0-0] = [7, 6, 8, 0].

                # 3 + 1 + 4 + 0 = 8
                actual_expanded = int(decode_lens_np.sum())

                # Fuse expanded_base and expanded_starts into a single
                # repeat_interleave:
                # seq_len_i = (context_start[b] - query_start_loc[b]) + arange[i] + 1
                # where context_start[b] = seq_lens[b] - decode_lens[b].
                # Example: offsets = [7-0, 6-3, 8-4, 0-8] = [7, 3, 4, -8]
                # expanded_offsets  = [7, 7, 7, 3, 4, 4, 4, 4]
                # result            = [8, 9, 10, 7, 9, 10, 11, 12]
                seq_lens_for_expansion = (
                    global_seq_lens if global_seq_lens is not None else seq_lens
                )
                expanded_offsets = torch.repeat_interleave(
                    seq_lens_for_expansion - decode_lens - query_start_loc,
                    decode_lens,
                    output_size=actual_expanded,
                )

                # [8, 9, 10, 7, 9, 10, 11, 12, ...] where ... is unused buffer space
                expanded_seq_lens = (
                    expanded_offsets + self.arange_buffer[:actual_expanded] + 1
                )
                if global_seq_lens is not None and self.dcp_world_size > 1:
                    expanded_seq_lens = get_dcp_local_seq_lens(
                        expanded_seq_lens,
                        self.dcp_world_size,
                        self.dcp_rank,
                        self.cp_kv_cache_interleave_size,
                    )
                self.decode_seq_lens_buffer[:actual_expanded] = expanded_seq_lens
                self.decode_seq_lens_buffer[actual_expanded:] = 0
                seq_lens = self.decode_seq_lens_buffer[:num_decode_tokens]

                # Give each of the flattened entries the same block table row as the
                # original request.
                self.expanded_block_table_buffer[:actual_expanded] = (
                    torch.repeat_interleave(
                        block_table, decode_lens, dim=0, output_size=actual_expanded
                    )
                )
                if actual_expanded < num_decode_tokens:
                    self.expanded_block_table_buffer[
                        actual_expanded:num_decode_tokens, 0
                    ] = 0
                block_table = self.expanded_block_table_buffer[:num_decode_tokens]

                # All reqs now have decode_len=1
                self.decode_lens_buffer[:num_decode_tokens] = 1
                decode_lens = self.decode_lens_buffer[:num_decode_tokens]
                return seq_lens, block_table, decode_lens, num_decode_tokens, False
        else:
            # Native path: plain decode (next_n==1) or spec decode
            # with 2D per-token context lengths (next_n > 1).
            #
            # When decode_lens are not truly uniform (e.g. some requests have
            # decode_len < next_n due to padding or short prefills), the simple
            # reshape in sparse_attn_indexer won't work. Use pack_seq_triton
            # (requires_padding) instead.
            requires_padding = min_decode_len != max_decode_len
            if use_native and next_n > 1:
                assert self.decode_seq_lens_buffer.dim() == 1
                # (B, max_decode_len): token j attends to
                # L - max_decode_len + j + 1 KV tokens.
                seq_lens_buffer = self.decode_seq_lens_buffer[
                    : num_decodes * max_decode_len
                ].view(num_decodes, max_decode_len)
                seq_lens_for_expansion = (
                    global_seq_lens if global_seq_lens is not None else seq_lens
                )
                expanded_seq_lens = (
                    seq_lens_for_expansion.unsqueeze(1)
                    - max_decode_len
                    + 1
                    + self.offsets_buffer[:max_decode_len]
                )
                if global_seq_lens is not None and self.dcp_world_size > 1:
                    expanded_seq_lens = get_dcp_local_seq_lens(
                        expanded_seq_lens.reshape(-1),
                        self.dcp_world_size,
                        self.dcp_rank,
                        self.cp_kv_cache_interleave_size,
                    ).view(num_decodes, max_decode_len)
                seq_lens_buffer[:] = expanded_seq_lens
                seq_lens = seq_lens_buffer
            return seq_lens, block_table, decode_lens, num_decodes, requires_padding

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV32IndexerMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        if common_attn_metadata.batch_topology is not None:
            query_start_loc_np = common_attn_metadata.batch_topology.query_start_loc_np[
                : num_reqs + 1
            ]
            query_lens_np = common_attn_metadata.batch_topology.query_lens_np
        else:
            query_start_loc_np = query_start_loc_cpu.numpy()
            query_lens_np = np.diff(query_start_loc_np)
        seq_lens = common_attn_metadata.seq_lens
        slot_mapping = common_attn_metadata.slot_mapping
        block_table = common_attn_metadata.block_table_tensor
        use_dcp_local_kv = (
            self.dcp_world_size > 1
            and common_attn_metadata.dcp_local_seq_lens is not None
        )
        effective_dcp_world_size = self.dcp_world_size if use_dcp_local_kv else 1
        effective_dcp_rank = self.dcp_rank if use_dcp_local_kv else 0

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            common_attn_metadata.split_decodes_and_prefills(
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=not self.use_flattening,
            )
        )

        assert num_decodes + num_prefills == num_reqs
        assert num_decode_tokens + num_prefill_tokens == num_tokens

        compressed_slot_mapping = slot_mapping
        logical_compressed_seq_lens = seq_lens
        compressed_seq_lens = seq_lens
        if self.compress_ratio > 1 or use_dcp_local_kv:
            compressed_slot_mapping = get_compressed_slot_mapping(
                num_tokens,
                query_start_loc,
                seq_lens,
                block_table,
                self.storage_block_size,
                self.compress_ratio,
                out=self.compressed_slot_mapping_buffer,
                dcp_world_size=effective_dcp_world_size,
                dcp_rank=effective_dcp_rank,
                cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
            )
            logical_compressed_seq_lens = seq_lens // self.compress_ratio
            compressed_seq_lens = logical_compressed_seq_lens
        if use_dcp_local_kv:
            compressed_seq_lens = get_dcp_local_seq_lens(
                logical_compressed_seq_lens,
                self.dcp_world_size,
                self.dcp_rank,
                self.cp_kv_cache_interleave_size,
            )

        prefill_metadata = None
        if num_prefills > 0:
            # This CPU value is an upper bound for async-spec extend rows.  It
            # is safe for chunking/allocation because CUDA metadata below is
            # built from exact device seq_lens and gather ignores the tail.
            assert common_attn_metadata.seq_lens_cpu_upper_bound is not None
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
            logical_compressed_seq_lens_cpu = (
                seq_lens_cpu // self.compress_ratio
                if self.compress_ratio > 1
                else seq_lens_cpu
            )
            compressed_seq_lens_cpu = logical_compressed_seq_lens_cpu
            if use_dcp_local_kv:
                compressed_seq_lens_cpu = get_dcp_local_seq_lens(
                    logical_compressed_seq_lens_cpu,
                    self.dcp_world_size,
                    self.dcp_rank,
                    self.cp_kv_cache_interleave_size,
                )
            compressed_seq_lens_cpu_np = compressed_seq_lens_cpu.numpy()
            prefill_query_lens_cpu = query_lens_np[
                num_decodes : num_decodes + num_prefills
            ]
            max_logits_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
            if envs.VLLM_USE_B12X_SPARSE_INDEXER:
                chunk_specs = []
                b12x_budget_seq_lens = np.array(
                    [_get_b12x_paged_indexer_supertile_k()],
                    dtype=compressed_seq_lens_cpu_np.dtype,
                )
                for prefill_idx in range(num_prefills):
                    req_idx = num_decodes + prefill_idx
                    chunk_specs.extend(
                        split_indexer_prefill_chunks(
                            b12x_budget_seq_lens,
                            prefill_query_lens_cpu[
                                prefill_idx : prefill_idx + 1
                            ],
                            self.max_prefill_buffer_size,
                            max_logits_bytes,
                            request_offset=req_idx,
                        )
                    )
            else:
                chunk_specs = split_indexer_prefill_chunks(
                    compressed_seq_lens_cpu_np[num_decodes:],
                    prefill_query_lens_cpu,
                    self.max_prefill_buffer_size,
                    max_logits_bytes,
                    request_offset=num_decodes,
                )

            chunks = []
            for req_slice, query_slice in chunk_specs:
                metadata = build_prefill_chunk_metadata(
                    req_slice.start,
                    req_slice.stop,
                    query_start_loc,
                    query_start_loc_cpu,
                    seq_lens,
                    compressed_seq_lens,
                    compressed_seq_lens_cpu,
                    common_attn_metadata.block_table_tensor,
                    self.compress_ratio,
                    query_slice=query_slice,
                    query_start_loc_np=query_start_loc_np,
                    compressed_seq_lens_np=compressed_seq_lens_cpu_np,
                    skip_kv_gather=query_slice.start > 0,
                    dcp_world_size=effective_dcp_world_size,
                    dcp_rank=effective_dcp_rank,
                    cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
                )
                # Skip when total_seq_lens is 0 (i.e., no compressed token).
                if metadata is not None:
                    chunks.append(metadata)
            prefill_metadata = DeepseekV32IndexerPrefillMetadata(chunks)

        decode_metadata = None
        if num_decodes > 0:
            torch.diff(
                common_attn_metadata.query_start_loc[: num_decodes + 1],
                out=self.decode_lens_buffer[:num_decodes],
            )
            decode_lens = self.decode_lens_buffer[:num_decodes]
            decode_lens_np = query_lens_np[:num_decodes]

            global_decode_seq_lens = common_attn_metadata.seq_lens[:num_decodes]
            dcp_local_seq_lens = (
                common_attn_metadata.dcp_local_seq_lens[:num_decodes]
                if self.compress_ratio == 1
                and use_dcp_local_kv
                and common_attn_metadata.dcp_local_seq_lens is not None
                else None
            )
            dcp_local_seq_lens_cpu = (
                common_attn_metadata.dcp_local_seq_lens_cpu[:num_decodes]
                if dcp_local_seq_lens is not None
                and common_attn_metadata.dcp_local_seq_lens_cpu is not None
                else None
            )
            seq_lens = (
                dcp_local_seq_lens
                if dcp_local_seq_lens is not None
                else global_decode_seq_lens
            )
            block_table = common_attn_metadata.block_table_tensor[:num_decodes, ...]

            max_decode_len = int(decode_lens_np.max())
            next_n = 1 + self.num_speculative_tokens
            use_native = not self.use_flattening and max_decode_len <= next_n

            seq_lens, block_table, decode_lens, batch_size, requires_padding = (
                self._prepare_decode_tensors(
                    seq_lens=seq_lens,
                    block_table=block_table,
                    decode_lens=decode_lens,
                    decode_lens_np=decode_lens_np,
                    query_start_loc=common_attn_metadata.query_start_loc[:num_decodes],
                    num_decodes=num_decodes,
                    num_decode_tokens=num_decode_tokens,
                    use_native=use_native,
                    next_n=next_n,
                    max_decode_len=max_decode_len,
                    global_seq_lens=global_decode_seq_lens
                    if dcp_local_seq_lens is not None
                    else None,
                )
            )

            # For DeepseekV4 (compress_ratio > 1), the indexer KV cache stores
            # compressed tokens. Convert uncompressed seq_lens to compressed.
            if self.compress_ratio > 1:
                # True iff seq_lens aliases decode_seq_lens_buffer (flatten or
                # native wrote it); False iff it aliases common_attn_metadata.
                seq_lens_is_local_view = (use_native and next_n > 1) or (
                    not use_native and max_decode_len > 1
                )
                if seq_lens_is_local_view:
                    seq_lens //= self.compress_ratio
                    if use_dcp_local_kv:
                        if seq_lens.dim() == 1:
                            dcp_seq_lens = get_dcp_local_seq_lens(
                                seq_lens,
                                self.dcp_world_size,
                                self.dcp_rank,
                                self.cp_kv_cache_interleave_size,
                            )
                        else:
                            dcp_seq_lens = get_dcp_local_seq_lens(
                                seq_lens.reshape(-1),
                                self.dcp_world_size,
                                self.dcp_rank,
                                self.cp_kv_cache_interleave_size,
                            ).view(seq_lens.shape)
                        seq_lens.copy_(dcp_seq_lens)
                else:
                    # Copy to avoid mutating shared state; keeps CG address stable.
                    compressed_decode_seq_lens = seq_lens // self.compress_ratio
                    if use_dcp_local_kv:
                        compressed_decode_seq_lens = get_dcp_local_seq_lens(
                            compressed_decode_seq_lens,
                            self.dcp_world_size,
                            self.dcp_rank,
                            self.cp_kv_cache_interleave_size,
                        )
                    self.expanded_seq_lens_buffer[:num_decodes] = (
                        compressed_decode_seq_lens
                    )
                    self.expanded_seq_lens_buffer[num_decodes:num_decode_tokens] = 0
                    seq_lens = self.expanded_seq_lens_buffer[:num_decode_tokens]
            elif use_dcp_local_kv and dcp_local_seq_lens is None:
                if seq_lens.dim() == 1:
                    seq_lens = get_dcp_local_seq_lens(
                        seq_lens,
                        self.dcp_world_size,
                        self.dcp_rank,
                        self.cp_kv_cache_interleave_size,
                    )
                else:
                    seq_lens_shape = seq_lens.shape
                    seq_lens = get_dcp_local_seq_lens(
                        seq_lens.reshape(-1),
                        self.dcp_world_size,
                        self.dcp_rank,
                        self.cp_kv_cache_interleave_size,
                    ).view(seq_lens_shape)

            # Non-MTP: deep_gemm paged MQA logits requires 2D context_lens
            # (csrc/apis/attention.hpp). Unsqueeze to (B, 1) so downstream
            # kernels see the same (B, next_n) layout as the MTP path.
            if seq_lens.dim() == 1:
                seq_lens = seq_lens.unsqueeze(-1)

            active_width = None
            active_width_tokens = None
            if self.compress_ratio > 1:
                active_width_tokens = -(
                    -int(common_attn_metadata.max_seq_len) //
                    self.compress_ratio
                )
            if envs.VLLM_USE_B12X_SPARSE_INDEXER:
                # Live scorer window in cache tokens. ceil(max_seq_len /
                # compress_ratio) is an upper bound on the max compressed
                # context across the batch, so windowing to it is top-k-identical
                # to the capacity cap and only skips wasted k-tiles. Computed on
                # the host here (metadata-prep, outside cudagraph capture) and
                # filled into the persistent buffer the captured kernel reads.
                if active_width_tokens is None:
                    active_width_tokens = int(common_attn_metadata.max_seq_len)
                self.b12x_active_width_buffer.fill_(active_width_tokens)
                active_width = self.b12x_active_width_buffer
                if self.compress_ratio > 1:
                    # Compressed-MLA models already bound the B12X scorer with
                    # active_width. Avoiding a host reduction here removes a
                    # per-decode sync without changing the scorer window.
                    decode_topk_max_seq_len = active_width_tokens
                else:
                    # GLM/Kimi-style uncompressed indexer rows need the exact
                    # scalar for best throughput; use the CPU shadow maintained
                    # by the runner instead of synchronizing on seq_lens.
                    decode_topk_max_seq_len = (
                        self._decode_topk_max_seq_len_from_cpu(
                            common_attn_metadata,
                            num_decodes,
                            decode_lens_np,
                            max_decode_len,
                            use_native,
                            dcp_local_seq_lens_cpu,
                        )
                    )
                    if decode_topk_max_seq_len is None:
                        # Async-spec can make GPU seq_lens authoritative.  In
                        # that mode there is no exact host scalar without a
                        # sync, so use the same graph-stable live-window bound
                        # already consumed by the B12X scorer.
                        decode_topk_max_seq_len = active_width_tokens
            elif active_width_tokens is not None:
                # DeepGEMM still receives exact per-row seq_lens on device.
                # The scalar max_seq_len is only a host scorer bound; using the
                # metadata upper bound avoids a per-token CUDA scalar sync.
                decode_topk_max_seq_len = active_width_tokens
            else:
                decode_topk_max_seq_len = int(seq_lens.max().item())

            if envs.VLLM_USE_B12X_SPARSE_INDEXER:
                schedule_metadata = self._maybe_build_b12x_schedule_metadata(
                    seq_lens,
                    block_table,
                    num_decode_tokens,
                    requires_padding,
                )
            else:
                # DeepGEMM is required for paged MQA logits on CUDA devices.
                schedule_metadata = self._maybe_build_deep_gemm_schedule_metadata(
                    seq_lens
                )

            decode_metadata = DeepSeekV32IndexerDecodeMetadata(
                block_table=block_table,
                seq_lens=seq_lens,
                max_seq_len=decode_topk_max_seq_len,
                decode_lens=decode_lens,
                requires_padding=requires_padding,
                schedule_metadata=schedule_metadata,
                compress_ratio=self.compress_ratio,
                active_width=active_width,
            )

        attn_metadata = DeepseekV32IndexerMetadata(
            seq_lens=common_attn_metadata.seq_lens,
            max_seq_len=common_attn_metadata.max_seq_len,
            slot_mapping=compressed_slot_mapping,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            prefill=prefill_metadata,
            decode=decode_metadata,
            dcp_world_size=effective_dcp_world_size,
            dcp_rank=effective_dcp_rank,
            cp_interleave_size=self.cp_kv_cache_interleave_size,
        )

        return attn_metadata


def build_prefill_chunk_metadata(
    start_idx: int,
    end_idx: int,
    query_start_loc: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    uncompressed_seq_lens: torch.Tensor,
    compressed_seq_lens: torch.Tensor,
    compressed_seq_lens_cpu: torch.Tensor,
    block_table: torch.Tensor,
    compress_ratio: int,
    query_slice: slice | None = None,
    query_start_loc_np: np.ndarray | None = None,
    compressed_seq_lens_np: np.ndarray | None = None,
    skip_kv_gather: bool = False,
    dcp_world_size: int = 1,
    dcp_rank: int = 0,
    cp_kv_cache_interleave_size: int = 1,
) -> DeepseekV32IndexerPrefillChunkMetadata | None:
    if compressed_seq_lens_np is not None:
        total_seq_lens = int(compressed_seq_lens_np[start_idx:end_idx].sum())
    else:
        total_seq_lens = compressed_seq_lens_cpu[start_idx:end_idx].sum().item()
    if total_seq_lens == 0:
        return None

    num_reqs = end_idx - start_idx
    device = block_table.device
    token_to_seq = torch.empty(total_seq_lens, dtype=torch.int32, device=device)

    cu_seq_lens = torch.empty(num_reqs + 1, dtype=torch.int32, device=device)
    # Assigning to slice avoids cpu sync.
    cu_seq_lens[:1] = 0
    torch.cumsum(compressed_seq_lens[start_idx:end_idx], dim=0, out=cu_seq_lens[1:])

    query_start_loc = (
        query_start_loc[start_idx : end_idx + 1] - query_start_loc[start_idx]
    )

    if query_start_loc_np is not None:
        total_query_len = int(
            query_start_loc_np[end_idx] - query_start_loc_np[start_idx]
        )
    else:
        total_query_len = int(
            (query_start_loc_cpu[end_idx] - query_start_loc_cpu[start_idx]).item()
        )
    if query_slice is not None:
        qs_start = query_slice.start
        qs_stop = query_slice.stop
    else:
        qs_start = 0
        qs_stop = total_query_len
    output_query_len = qs_stop - qs_start

    cu_seq_len_ks = torch.empty(output_query_len, dtype=torch.int32, device=device)
    cu_seq_len_ke = torch.empty(output_query_len, dtype=torch.int32, device=device)

    _build_prefill_chunk_metadata_kernel[(num_reqs,)](
        query_start_loc,
        uncompressed_seq_lens[start_idx:end_idx],
        cu_seq_lens,
        token_to_seq,
        cu_seq_len_ks,
        cu_seq_len_ke,
        qs_start,
        qs_stop,
        DCP_WORLD_SIZE=dcp_world_size,
        DCP_RANK=dcp_rank,
        CP_KV_CACHE_INTERLEAVE_SIZE=cp_kv_cache_interleave_size,
        BLOCK_SIZE=1024,
        COMPRESS_RATIO=compress_ratio,
    )

    token_start = (
        int(query_start_loc_np[start_idx])
        if query_start_loc_np is not None
        else query_start_loc_cpu[start_idx].item()
    )
    if query_slice is not None:
        token_end = token_start + qs_stop
        token_start = token_start + qs_start
        skip_kv_gather = skip_kv_gather or qs_start > 0
    else:
        token_end = (
            int(query_start_loc_np[end_idx])
            if query_start_loc_np is not None
            else query_start_loc_cpu[end_idx].item()
        )

    return DeepseekV32IndexerPrefillChunkMetadata(
        cu_seqlen_ks=cu_seq_len_ks,
        cu_seqlen_ke=cu_seq_len_ke,
        cu_seq_lens=cu_seq_lens,
        token_to_seq=token_to_seq,
        total_seq_lens=total_seq_lens,
        block_table=block_table[start_idx:end_idx],
        token_start=token_start,
        token_end=token_end,
        num_reqs=num_reqs,
        skip_kv_gather=skip_kv_gather,
    )


@triton.jit(do_not_specialize=["query_slice_start", "query_slice_stop"])
def _build_prefill_chunk_metadata_kernel(
    # Inputs
    query_start_loc_ptr,
    uncompressed_seq_lens_ptr,
    cu_compressed_seq_lens_ptr,
    # Outputs
    token_to_seq_ptr,
    cu_compressed_seq_len_ks_ptr,
    cu_compressed_seq_len_ke_ptr,
    query_slice_start,
    query_slice_stop,
    DCP_WORLD_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
):
    batch_idx = tl.program_id(0)

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    seq_start = tl.load(cu_compressed_seq_lens_ptr + batch_idx)
    seq_end = tl.load(cu_compressed_seq_lens_ptr + batch_idx + 1)
    compressed_seq_len = seq_end - seq_start

    uncompressed_seq_len = tl.load(uncompressed_seq_lens_ptr + batch_idx)
    start_pos = uncompressed_seq_len - query_len

    for i in range(0, query_len, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        abs_pos = query_start + offset
        mask = (
            (offset < query_len)
            & (abs_pos >= query_slice_start)
            & (abs_pos < query_slice_stop)
        )
        out_pos = abs_pos - query_slice_start

        # Compute cu_seq_len_ks
        tl.store(cu_compressed_seq_len_ks_ptr + out_pos, seq_start, mask=mask)

        # Compute cu_seq_len_ke
        seq_len_per_token = (start_pos + 1 + offset) // COMPRESS_RATIO
        if DCP_WORLD_SIZE > 1:
            base = (
                seq_len_per_token
                // CP_KV_CACHE_INTERLEAVE_SIZE
                // DCP_WORLD_SIZE
                * CP_KV_CACHE_INTERLEAVE_SIZE
            )
            remainder = seq_len_per_token - base * DCP_WORLD_SIZE
            rank_remainder = tl.minimum(
                tl.maximum(
                    remainder - DCP_RANK * CP_KV_CACHE_INTERLEAVE_SIZE,
                    0,
                ),
                CP_KV_CACHE_INTERLEAVE_SIZE,
            )
            seq_len_per_token = base + rank_remainder
        tl.store(
            cu_compressed_seq_len_ke_ptr + out_pos,
            seq_start + seq_len_per_token,
            mask=mask,
        )

    # Compute token_to_seq
    for i in range(0, compressed_seq_len, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < compressed_seq_len
        tl.store(token_to_seq_ptr + seq_start + offset, batch_idx, mask=mask)


def _prewarm_prefill_chunk_metadata_kernel(
    *,
    device: torch.device,
    dcp_world_size: int,
    dcp_rank: int,
    cp_kv_cache_interleave_size: int,
    compress_ratio: int,
) -> None:
    if not current_platform.is_cuda():
        return

    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=device)
    uncompressed_seq_lens = torch.tensor([1], dtype=torch.int32, device=device)
    cu_seq_lens = torch.tensor([0, 1], dtype=torch.int32, device=device)
    token_to_seq = torch.empty((1,), dtype=torch.int32, device=device)
    cu_seq_len_ks = torch.empty((1,), dtype=torch.int32, device=device)
    cu_seq_len_ke = torch.empty((1,), dtype=torch.int32, device=device)

    _build_prefill_chunk_metadata_kernel[(1,)](
        query_start_loc,
        uncompressed_seq_lens,
        cu_seq_lens,
        token_to_seq,
        cu_seq_len_ks,
        cu_seq_len_ke,
        0,
        1,
        DCP_WORLD_SIZE=int(dcp_world_size),
        DCP_RANK=int(dcp_rank),
        CP_KV_CACHE_INTERLEAVE_SIZE=int(cp_kv_cache_interleave_size),
        BLOCK_SIZE=1024,
        COMPRESS_RATIO=int(compress_ratio),
    )
