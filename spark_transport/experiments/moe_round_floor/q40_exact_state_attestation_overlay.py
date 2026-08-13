"""Build the pre-graph runtime attestation for the target-only Q40 state.

The generated model-runner source inventories every loaded EXL3 routed layer
after the eager 4096-row profile call, which also exercises the fixed-MTP
draft, and before sampler profiling or CUDA graph capture.  It writes one
exclusive, atomic receipt per DCP rank only after proving the exact target and
draft runtime geometry and stable precompiled storage.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path


INPUT_SHA256 = "992486a1a70fd0cb54c54cf3f70af0f09b4bcc6ae3bbf433e66b1296415015b4"
OUTPUT_SHA256 = "0e2e0150702029b3c09bd117c33101d90d8197386d278dc6008973b314ae9997"
PATCHED_EXL3_SHA256 = (
    "8fad5330c88f55dc57e4d8e298f2af23e16390b97153b569a2e572e0fb5065c2"
)
IMAGE_ID = "sha256:02881d5229d4f4d1cbba0cf40537492a2a505b9d4e43bbfe9a0b2a7bd0584513"
CHECKPOINT_REVISION = "9ab9579774cc432df91567a36f6e9e863e0d4c9f"
_IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")

INIT_ANCHOR = """\
        self.q40_route_capture_initialized = False
"""

INIT_REPLACEMENT = """\
        self.q40_route_capture_initialized = False
        self.q40_exact_state_policy_attested = False
"""

CALL_ANCHOR = """\
        hidden_states, sample_hidden_states = self._dummy_run(
            self.max_num_tokens, skip_attn=True, is_profile=True
        )

        # Only run sampler/pooler on last PP rank (non-last ranks return None).
"""

CALL_REPLACEMENT = """\
        hidden_states, sample_hidden_states = self._dummy_run(
            self.max_num_tokens, skip_attn=True, is_profile=True
        )

        if os.getenv("SPARK_Q40_EXACT_STATE_ATTEST_PATH"):
            self._attest_q40_exact_state_policy()

        # Only run sampler/pooler on last PP rank (non-last ranks return None).
"""

METHOD_ANCHOR = """\
    def get_model(self) -> nn.Module:
        return self.model

    def _init_q40_route_capturer(self) -> None:
"""

METHOD_REPLACEMENT = '''\
    def get_model(self) -> nn.Module:
        return self.model

    def _attest_q40_exact_state_policy(self) -> None:
        """Prove the target-only Q40 state before CUDA graph capture."""

        if self.q40_exact_state_policy_attested:
            return
        import hashlib
        import inspect
        import json
        from pathlib import Path

        from vllm.model_executor.layers.quantization import exl3 as exl3_module

        expected_exl3_sha256 = "8fad5330c88f55dc57e4d8e298f2af23e16390b97153b569a2e572e0fb5065c2"
        if os.environ.get("SPARK_Q40_EXACT_STATE_EXPECTED_EXL3_SHA256") != expected_exl3_sha256:
            raise RuntimeError("Q40 exact-state EXL3 hash declaration is absent or incorrect")
        if "VLLM_EXL3_PREFILL_BLOCK_M" in os.environ:
            raise RuntimeError(
                "Q40 exact-state policy requires VLLM_EXL3_PREFILL_BLOCK_M must remain unset"
            )
        if "VLLM_EXL3_TRELLIS_MAX_M" in os.environ:
            raise RuntimeError(
                "Q40 exact-state policy requires VLLM_EXL3_TRELLIS_MAX_M to remain unset"
            )
        if os.environ.get("VLLM_EXL3_PREFILL_CAPACITY") != "4096":
            raise RuntimeError("Q40 exact-state policy requires prefill capacity 4096")
        if os.environ.get("VLLM_EXL3_PREFILL_TRELLIS", "1") != "1":
            raise RuntimeError("Q40 exact-state policy requires prefill Trellis")
        if self.max_num_tokens != 4096:
            raise RuntimeError(
                "Q40 exact-state policy requires max_num_batched_tokens=4096, "
                f"got {self.max_num_tokens}"
            )
        if self.model_config.dtype != torch.bfloat16:
            raise RuntimeError(
                "Q40 exact-state numerical gate requires BF16 model activations, "
                f"got {self.model_config.dtype}"
            )

        def sha256(path: Path) -> str:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        def routed_layers(model, role):
            found = []
            seen = set()
            for module in model.modules():
                routed = getattr(module, "routed_experts", module)
                if id(routed) in seen:
                    continue
                quant = getattr(routed, "quant_method", None)
                if quant is None or quant.__class__.__name__ != "Exl3MoEMethod":
                    continue
                mixed = getattr(routed, "exl3_mixed_trellis", None)
                uniform = getattr(routed, "exl3_trellis_weights", None)
                if not isinstance(mixed, dict) and uniform is None:
                    continue
                seen.add(id(routed))
                layer_id = getattr(module, "layer_id", None)
                if layer_id is None:
                    name = str(getattr(routed, "layer_name", ""))
                    pieces = [
                        item
                        for item in name.replace("[", ".").replace("]", ".").split(".")
                        if item.isdigit()
                    ]
                    layer_id = int(pieces[-1]) if pieces else None
                found.append((layer_id, routed, quant, role, isinstance(mixed, dict)))
            return found

        def launch_tile(launch) -> tuple[int, int, int, int]:
            return tuple(
                int(getattr(launch, name))
                for name in ("fc1_tile_k", "fc1_tile_n", "fc2_tile_k", "fc2_tile_n")
            )

        def launch_tiers(launch, count: int) -> tuple[tuple[int, int], ...]:
            return tuple(
                (
                    int(getattr(launch, f"tier{tier_id}_bits")),
                    int(getattr(launch, f"tier{tier_id}_num_experts")),
                )
                for tier_id in range(count)
            )

        def run_mixed_state(layer, runtime, state, x, weights, ids):
            mixed = layer.exl3_mixed_trellis
            tiers = tuple(mixed["prefill_tiers"])
            run_kwargs = {}
            gate_experts = mixed.get("tier_gate_experts")
            up_experts = mixed.get("tier_up_experts")
            if (gate_experts is None) != (up_experts is None):
                raise RuntimeError("projection-tight R7 tiers require paired gate/up counts")
            if gate_experts is not None:
                run_kwargs.update(
                    gate_experts=gate_experts,
                    up_experts=up_experts,
                )
            run_mixed = (
                runtime["mixed_api"].run_mixed_trellis3
                if len(tiers) == 3
                else runtime["mixed_api"].run_mixed_trellis
            )
            if run_mixed is None:
                raise RuntimeError(
                    f"installed B12X lacks {len(tiers)}-tier mixed run support"
                )
            return run_mixed(
                x,
                *tiers,
                weights,
                ids,
                mixed["global_to_combined"],
                mixed["descriptor_map"],
                mixed["rotations"],
                state["launch"],
                state["buffers"],
                **run_kwargs,
            ).to(x.dtype)

        def walk_tensors(value, seen):
            if torch.is_tensor(value):
                yield value
                return
            identity = id(value)
            if identity in seen:
                return
            seen.add(identity)
            if isinstance(value, dict):
                children = value.values()
            elif isinstance(value, (tuple, list, set)):
                children = value
            elif hasattr(value, "__dict__"):
                children = vars(value).values()
            else:
                return
            for child in children:
                yield from walk_tensors(child, seen)

        def storage_record(tensor) -> tuple[tuple[str, int, int], dict[str, object]]:
            storage = tensor.untyped_storage()
            key = (str(tensor.device), int(storage.data_ptr()), int(storage.nbytes()))
            return key, {
                "device": key[0],
                "data_ptr": key[1],
                "bytes": key[2],
            }

        def normalize_buffer_key(key) -> dict[str, object]:
            if not isinstance(key, tuple) or len(key) != 13:
                raise RuntimeError(f"unexpected mixed-buffer cache key: {key!r}")
            owner = key[0]
            if not isinstance(owner, tuple) or len(owner) != 2:
                raise RuntimeError(f"unexpected mixed-buffer owner key: {owner!r}")
            return {
                "owner_scope_id": int(owner[0]),
                "owner_is_draft": bool(owner[1]),
                "device_index": None if key[1] is None else int(key[1]),
                "capacity": int(key[2]),
                "block_m": int(key[3]),
                "tile": [int(value) for value in key[4]],
                "topk": int(key[5]),
                "hidden_size": int(key[6]),
                "intermediate_size": int(key[7]),
                "launch_size_m": int(key[8]),
                "sms": int(key[9]),
                "rotation_input_dtype": str(key[10]),
                "route_ids_dtype": str(key[11]),
                "tier_count": int(key[12]),
            }

        target = routed_layers(self.model, "target")
        draft_model = getattr(self.speculator, "model", None)
        if draft_model is None:
            raise RuntimeError("Q40 exact-state policy requires the fixed-MTP draft model")
        draft = routed_layers(draft_model, "draft")
        target_mixed = [item for item in target if item[4]]
        target_uniform = [item for item in target if not item[4]]
        draft_mixed = [item for item in draft if item[4]]
        draft_uniform = [item for item in draft if not item[4]]
        target_ids = sorted(item[0] for item in target_mixed)
        if target_ids != list(range(3, 78)) or target_uniform:
            raise RuntimeError(
                "Q40 exact-state target inventory must be exactly mixed layer IDs 3..77; "
                f"mixed={target_ids}, uniform={len(target_uniform)}"
            )
        if draft_mixed or len(draft_uniform) != 1:
            raise RuntimeError(
                "Q40 exact-state draft inventory must be exactly one uniform EXL3 layer; "
                f"mixed={len(draft_mixed)}, uniform={len(draft_uniform)}"
            )

        mixed_cache = exl3_module._MIXED_TRELLIS_RUNTIMES
        buffer_cache = exl3_module._MIXED_TRELLIS_BUFFERS
        uniform_cache = exl3_module._RANK_SLICED_RUNTIMES
        before_cache = (len(mixed_cache), len(buffer_cache), len(uniform_cache))
        target_records = []
        arena_by_key = {}
        key_by_arena = {}
        unique_storages = {}
        numerical_records = []

        for layer_id, layer, quant, role, _is_mixed in target_mixed:
            hidden = int(layer.exl3_hidden_size)
            topk = int(layer.top_k)
            x = torch.zeros((20, hidden), dtype=self.model_config.dtype, device=self.device)
            ids = torch.zeros((20, topk), dtype=torch.int64, device=self.device)
            runtime = quant._mixed_rank_sliced_runtime(layer, x, ids)
            runtime_again = quant._mixed_rank_sliced_runtime(layer, x, ids)
            if runtime_again is not runtime:
                raise RuntimeError(f"target layer {layer_id} mixed runtime identity changed")
            owner_token = exl3_module._runtime_owner_token(quant.quant_config, layer)
            if bool(owner_token[1]):
                raise RuntimeError(f"target layer {layer_id} was classified as draft")
            mixed = layer.exl3_mixed_trellis
            policy = mixed.get("runtime_policy")
            if not isinstance(policy, dict):
                raise RuntimeError(f"target layer {layer_id} lacks a mixed runtime policy")
            if (
                int(runtime.get("max_decode_m", -1)) != 32
                or int(runtime.get("max_batched_tokens", -1)) != 4096
                or int(runtime.get("prefill_capacity", -1)) != 4096
            ):
                raise RuntimeError(f"target layer {layer_id} runtime thresholds drifted")

            states = {
                "decode": runtime.get("decode"),
                "q40": runtime.get("q40"),
                "prefill": runtime.get("prefill"),
            }
            if any(not isinstance(state, dict) for state in states.values()):
                raise RuntimeError(f"target layer {layer_id} lacks decode/Q40/prefill states")
            if len({id(state) for state in states.values()}) != 3:
                raise RuntimeError(f"target layer {layer_id} aliases runtime state records")
            for name, state in states.items():
                state_again = runtime_again.get(name)
                if (
                    state_again is not state
                    or state_again.get("launch") is not state.get("launch")
                    or state_again.get("buffers") is not state.get("buffers")
                ):
                    raise RuntimeError(
                        f"target layer {layer_id} {name} state identity changed"
                    )
            if (
                states["q40"]["buffers"] is states["decode"]["buffers"]
                or states["q40"]["buffers"] is states["prefill"]["buffers"]
            ):
                raise RuntimeError(f"target layer {layer_id} Q40 arena aliases another state")

            expected_geometry = {"decode": (32, 32, 8), "q40": (40, 40, 8)}
            for name, expected in expected_geometry.items():
                state = states[name]
                launch = state["launch"]
                actual = (
                    int(state.get("capacity", -1)),
                    int(getattr(launch, "size_m", -1)),
                    int(getattr(launch, "moe_block_size", -1)),
                )
                if actual != expected:
                    raise RuntimeError(
                        f"target layer {layer_id} {name} geometry drifted: {actual}"
                    )

            prefill = states["prefill"]
            prefill_launch = prefill["launch"]
            prefill_block = int(getattr(prefill_launch, "moe_block_size", -1))
            if (
                int(prefill.get("capacity", -1)) != 4096
                or int(getattr(prefill_launch, "size_m", -1)) != 4096
                or prefill_block not in (32, 64)
                or prefill_block != int(policy.get("prefill_block_m", -1))
            ):
                raise RuntimeError(
                    f"target layer {layer_id} general prefill geometry drifted: "
                    f"capacity={prefill.get('capacity')} size={getattr(prefill_launch, 'size_m', None)} "
                    f"block={prefill_block} policy={policy.get('prefill_block_m')}"
                )

            decode_tile = launch_tile(states["decode"]["launch"])
            prefill_tile = tuple(int(value) for value in mixed["prefill_tile_config"])
            if decode_tile != tuple(int(value) for value in mixed["tile_config"]):
                raise RuntimeError(f"target layer {layer_id} decode tile drifted")
            if launch_tile(states["q40"]["launch"]) != prefill_tile:
                raise RuntimeError(f"target layer {layer_id} Q40 does not use prefill tile")
            if launch_tile(prefill_launch) != prefill_tile:
                raise RuntimeError(f"target layer {layer_id} general prefill tile drifted")

            tier_signature = tuple(
                (int(bits), int(experts))
                for bits, experts in policy.get("tier_signature", ())
            )
            tier_count = len(tier_signature)
            if tier_count not in (2, 3):
                raise RuntimeError(f"target layer {layer_id} has invalid tier count")
            if len(tuple(mixed["prefill_tiers"])) != tier_count:
                raise RuntimeError(f"target layer {layer_id} prefill tier tuple drifted")
            for name, state in states.items():
                if launch_tiers(state["launch"], tier_count) != tier_signature:
                    raise RuntimeError(
                        f"target layer {layer_id} {name} launch tiers drifted"
                    )

            q40_buffers = states["q40"]["buffers"]
            matching_keys = [
                key for key, value in buffer_cache.items() if value is q40_buffers
            ]
            if len(matching_keys) != 1:
                raise RuntimeError(
                    f"target layer {layer_id} Q40 arena has {len(matching_keys)} cache keys"
                )
            normalized_key = normalize_buffer_key(matching_keys[0])
            if (
                normalized_key["capacity"] != 40
                or normalized_key["block_m"] != 8
                or normalized_key["tile"] != list(prefill_tile)
                or normalized_key["owner_is_draft"]
            ):
                raise RuntimeError(f"target layer {layer_id} Q40 buffer key drifted")
            serialized_key = json.dumps(normalized_key, sort_keys=True)
            arena_identity = id(q40_buffers)
            previous_arena = arena_by_key.setdefault(serialized_key, arena_identity)
            if previous_arena != arena_identity:
                raise RuntimeError("one Q40 buffer key resolved to multiple arenas")
            previous_key = key_by_arena.setdefault(arena_identity, serialized_key)
            if previous_key != serialized_key:
                raise RuntimeError("one Q40 arena resolved from multiple buffer keys")
            tensors = tuple(walk_tensors(q40_buffers, set()))
            if not tensors:
                raise RuntimeError(f"target layer {layer_id} Q40 arena has no tensors")
            for tensor in tensors:
                storage_key, record = storage_record(tensor)
                unique_storages.setdefault(storage_key, record)

            generator = torch.Generator(device=self.device)
            generator.manual_seed(0x5140 + int(layer_id))
            parity_x = torch.randn(
                (40, hidden),
                dtype=self.model_config.dtype,
                device=self.device,
                generator=generator,
            ).contiguous()
            parity_weights = torch.full(
                (40, topk),
                1.0 / float(topk),
                dtype=torch.float32,
                device=self.device,
            )
            route_experts = int(layer.local_num_experts)
            parity_ids = (
                torch.arange(40 * topk, dtype=torch.int64, device=self.device)
                .reshape(40, topk)
                .add_(int(layer_id))
                .remainder_(route_experts)
                .contiguous()
            )
            q40_output = quant._apply_mixed_rank_sliced(
                layer,
                parity_x,
                parity_weights,
                parity_ids,
            ).clone()
            prefill_output = run_mixed_state(
                layer,
                runtime,
                states["prefill"],
                parity_x,
                parity_weights,
                parity_ids,
            ).clone()
            torch.cuda.synchronize(self.device)
            finite = bool(
                torch.isfinite(q40_output).all().item()
                and torch.isfinite(prefill_output).all().item()
            )
            nonzero = bool(
                torch.count_nonzero(q40_output).item()
                and torch.count_nonzero(prefill_output).item()
            )
            exact = bool(torch.equal(q40_output, prefill_output))
            exact_bf16 = (
                q40_output.dtype == torch.bfloat16
                and prefill_output.dtype == torch.bfloat16
                and exact
            )
            if not finite or not nonzero or not exact_bf16:
                mismatches = int(
                    torch.count_nonzero(q40_output != prefill_output).item()
                )
                raise RuntimeError(
                    f"target layer {layer_id} Q40/general-prefill numerical gate failed: "
                    f"finite={finite}, nonzero={nonzero}, "
                    f"dtype={q40_output.dtype}/{prefill_output.dtype}, "
                    f"mismatches={mismatches}"
                )
            if any(runtime.get(name) is not state for name, state in states.items()):
                raise RuntimeError(
                    f"target layer {layer_id} runtime state changed during numerical gate"
                )
            numerical_records.append(
                {
                    "layer_id": int(layer_id),
                    "seed": 0x5140 + int(layer_id),
                    "routes": "deterministic-round-robin-local-experts",
                    "router_weights": 1.0 / float(topk),
                    "q40_exact_bf16_equal_to_general_prefill": exact_bf16,
                    "both_outputs_finite": finite,
                    "both_outputs_nonzero": nonzero,
                }
            )

            target_records.append(
                {
                    "role": role,
                    "kind": "mixed",
                    "layer_id": int(layer_id),
                    "tier_signature": [list(item) for item in tier_signature],
                    "decode": {"capacity": 32, "block_m": 8, "tile": list(decode_tile)},
                    "q40": {"capacity": 40, "block_m": 8, "tile": list(prefill_tile)},
                    "prefill": {
                        "capacity": 4096,
                        "block_m": prefill_block,
                        "tile": list(prefill_tile),
                    },
                    "q40_buffer_cache_key": normalized_key,
                }
            )

        draft_records = []
        for layer_id, layer, quant, role, _is_mixed in draft_uniform:
            hidden = int(layer.exl3_hidden_size)
            topk = int(layer.top_k)
            x = torch.zeros((20, hidden), dtype=self.model_config.dtype, device=self.device)
            ids = torch.zeros((20, topk), dtype=torch.int64, device=self.device)
            runtime = quant._rank_sliced_runtime(layer, x, ids)
            runtime_again = quant._rank_sliced_runtime(layer, x, ids)
            if runtime_again is not runtime:
                raise RuntimeError("draft uniform runtime identity changed")
            if "q40" in runtime:
                raise RuntimeError("draft uniform runtime unexpectedly contains a mixed Q40 state")
            owner_token = exl3_module._runtime_owner_token(quant.quant_config, layer)
            if not bool(owner_token[1]):
                raise RuntimeError("draft uniform layer was classified as target")
            decode = runtime.get("trellis_plan")
            prefill = runtime.get("prefill_plan")
            if decode is None or prefill is None:
                raise RuntimeError("draft uniform runtime lacks decode/prefill plans")
            decode_geometry = (
                int(decode.caps.max_tokens),
                int(decode.caps.w4a16_block_size_m),
            )
            prefill_geometry = (
                int(prefill.caps.max_tokens),
                int(prefill.caps.w4a16_block_size_m),
            )
            if (
                decode_geometry != (32, 8)
                or prefill_geometry != (4096, 64)
                or int(runtime.get("max_trellis_m", -1)) != 32
                or int(runtime.get("max_batched_tokens", -1)) != 4096
                or int(runtime.get("prefill_capacity", -1)) != 4096
            ):
                raise RuntimeError(
                    "draft uniform runtime geometry drifted: "
                    f"decode={decode_geometry}, prefill={prefill_geometry}"
                )
            draft_records.append(
                {
                    "role": role,
                    "kind": "uniform",
                    "layer_id": None if layer_id is None else int(layer_id),
                    "decode": {"capacity": 32, "block_m": 8},
                    "prefill": {"capacity": 4096, "block_m": 64},
                    "q40_state": "absent",
                }
            )

        after_cache = (len(mixed_cache), len(buffer_cache), len(uniform_cache))
        if after_cache != before_cache:
            raise RuntimeError(
                "Q40 exact-state attestation compiled missing runtime storage after profile: "
                f"before={before_cache}, after={after_cache}"
            )

        exl3_path = Path(inspect.getsourcefile(exl3_module) or "")
        if not exl3_path.is_file() or sha256(exl3_path) != expected_exl3_sha256:
            raise RuntimeError("loaded EXL3 source differs from the Q40 exact-state contract")
        exl3_text = exl3_path.read_text(encoding="utf-8")
        required_source = (
            "_MIXED_TRELLIS_TARGET_Q40_ROWS = 40",
            'and runtime["q40"] is not None',
            'runtime["q40"],\\n                mixed["prefill_tiers"],',
        )
        if any(fragment not in exl3_text for fragment in required_source):
            raise RuntimeError("loaded EXL3 source lacks the exact Q40 dispatch contract")

        image_id = os.environ.get("SPARK_Q40_EXACT_STATE_IMAGE_ID")
        checkpoint = os.environ.get("SPARK_Q40_EXACT_STATE_CHECKPOINT")
        if image_id != "sha256:02881d5229d4f4d1cbba0cf40537492a2a505b9d4e43bbfe9a0b2a7bd0584513":
            raise RuntimeError("Q40 exact-state image identity differs from the contract")
        if checkpoint != "9ab9579774cc432df91567a36f6e9e863e0d4c9f":
            raise RuntimeError("Q40 exact-state checkpoint differs from the contract")

        q40_storage_records = sorted(
            unique_storages.values(),
            key=lambda item: (str(item["device"]), int(item["data_ptr"]), int(item["bytes"])),
        )
        q40_buffer_cache_keys = [
            json.loads(item) for item in sorted(arena_by_key)
        ]
        q40_unique_storage_bytes = sum(int(item["bytes"]) for item in q40_storage_records)
        if q40_unique_storage_bytes <= 0 or not q40_buffer_cache_keys:
            raise RuntimeError("Q40 exact-state storage inventory is empty")
        receipt_value = os.environ["SPARK_Q40_EXACT_STATE_ATTEST_PATH"]
        expected_receipt = (
            f"/cache/jit/q40-exact-state-serving-v1-rank{self.dcp_rank}.json"
        )
        if receipt_value != expected_receipt:
            raise RuntimeError(
                "Q40 exact-state receipt path does not match the executing DCP rank: "
                f"expected {expected_receipt}, got {receipt_value}"
            )
        output = Path(receipt_value)
        temporary = output.with_name(output.name + ".tmp")
        if output.exists() or temporary.exists():
            raise RuntimeError(f"refusing to replace Q40 exact-state receipt {output}")
        result = {
            "schema": "sparkring-q40-exact-state-runtime-attestation/v1",
            "status": "live-runtime-attested",
            "scope": "target-mixed-exact-q40-only",
            "dcp_rank": int(self.dcp_rank),
            "image_id": image_id,
            "checkpoint_revision": checkpoint,
            "sources": {"exl3": {"path": str(exl3_path), "sha256": expected_exl3_sha256}},
            "inventory": {
                "target_mixed_layers": len(target_mixed),
                "target_uniform_layers": len(target_uniform),
                "draft_mixed_layers": len(draft_mixed),
                "draft_uniform_layers": len(draft_uniform),
            },
            "cache_counts": {
                "mixed_runtimes": after_cache[0],
                "mixed_buffer_arenas": after_cache[1],
                "uniform_runtimes": after_cache[2],
                "unchanged_during_attestation": True,
            },
            "q40_unique_storage_bytes": q40_unique_storage_bytes,
            "q40_unique_storage_count": len(q40_storage_records),
            "q40_unique_buffer_arenas": len(q40_buffer_cache_keys),
            "q40_storage_records": q40_storage_records,
            "q40_buffer_cache_keys": q40_buffer_cache_keys,
            "gates": {
                "global_prefill_block_override_absent": "pass",
                "target_decode32_block8": "pass",
                "target_q40_capacity40_block8_prefill_tile_and_tiers": "pass",
                "target_general_prefill_capacity4096_block32_or64": "pass",
                "draft_uniform_runtime_unchanged": "pass",
                "runtime_state_buffer_identity_stable": "pass",
                "all_storage_precompiled_before_attestation": "pass",
                "q40_exact_bf16_equal_to_general_prefill": "pass",
                "q40_and_general_prefill_finite_nonzero": "pass",
            },
            "numerical_gate": {
                "activation": "deterministic-nonzero-bf16",
                "routes": "deterministic-round-robin-local-experts",
                "router_weights": "uniform-float32",
                "layers": numerical_records,
            },
            "target_layers": target_records,
            "draft_layers": draft_records,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2, sort_keys=True)
                stream.write("\\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, output)
            if hasattr(os, "O_DIRECTORY"):
                directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        self.q40_exact_state_policy_attested = True
        logger.warning(
            "Q40 exact-state runtime policy attested on DCP rank %d: %s",
            self.dcp_rank,
            output,
        )

    def _init_q40_route_capturer(self) -> None:
'''


class ExactQ40StateAttestationOverlayError(RuntimeError):
    """The pinned model-runner source contract was not satisfied."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def transform(
    source: bytes,
    *,
    image_id: str = IMAGE_ID,
    checkpoint_revision: str = CHECKPOINT_REVISION,
) -> bytes:
    if not _IMAGE_ID_RE.fullmatch(image_id):
        raise ExactQ40StateAttestationOverlayError(
            "image ID must be sha256 followed by 64 lowercase hex characters"
        )
    if not _REVISION_RE.fullmatch(checkpoint_revision):
        raise ExactQ40StateAttestationOverlayError(
            "checkpoint revision must be 40 lowercase hex characters"
        )
    actual = sha256_bytes(source)
    if actual != INPUT_SHA256:
        raise ExactQ40StateAttestationOverlayError(
            f"model-runner input hash mismatch: expected {INPUT_SHA256}, got {actual}"
        )
    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ExactQ40StateAttestationOverlayError(
            "model-runner source is not UTF-8"
        ) from error
    method_replacement = METHOD_REPLACEMENT.replace(IMAGE_ID, image_id).replace(
        CHECKPOINT_REVISION, checkpoint_revision
    )
    replacements = (
        (INIT_ANCHOR, INIT_REPLACEMENT),
        (CALL_ANCHOR, CALL_REPLACEMENT),
        (METHOD_ANCHOR, method_replacement),
    )
    for anchor, replacement in replacements:
        if text.count(anchor) != 1:
            raise ExactQ40StateAttestationOverlayError(
                "model-runner insertion anchor is absent or non-unique"
            )
        text = text.replace(anchor, replacement, 1)
    compile(text, "model_runner.py", "exec")
    return text.encode("utf-8")


def install(
    source: Path,
    output: Path,
    *,
    image_id: str = IMAGE_ID,
    checkpoint_revision: str = CHECKPOINT_REVISION,
) -> dict[str, str | int]:
    if output.exists():
        raise ExactQ40StateAttestationOverlayError(f"refusing to overwrite {output}")
    source_bytes = source.read_bytes()
    output_bytes = transform(
        source_bytes,
        image_id=image_id,
        checkpoint_revision=checkpoint_revision,
    )
    if (
        image_id == IMAGE_ID
        and checkpoint_revision == CHECKPOINT_REVISION
        and OUTPUT_SHA256
        and sha256_bytes(output_bytes) != OUTPUT_SHA256
    ):
        raise ExactQ40StateAttestationOverlayError(
            "generated model-runner output hash mismatch"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("xb") as stream:
            stream.write(output_bytes)
            stream.flush()
    except FileExistsError as error:
        raise ExactQ40StateAttestationOverlayError(
            f"refusing to overwrite {output}"
        ) from error
    return {
        "input_path": str(source.resolve()),
        "input_sha256": sha256_bytes(source_bytes),
        "output_path": str(output.resolve()),
        "output_sha256": sha256_bytes(output_bytes),
        "output_bytes": len(output_bytes),
        "image_id": image_id,
        "checkpoint_revision": checkpoint_revision,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--checkpoint-revision", default=CHECKPOINT_REVISION)
    args = parser.parse_args()
    for key, value in install(
        args.source.resolve(),
        args.output.resolve(),
        image_id=args.image_id,
        checkpoint_revision=args.checkpoint_revision,
    ).items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
