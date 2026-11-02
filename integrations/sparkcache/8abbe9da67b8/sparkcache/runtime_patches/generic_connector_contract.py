"""Source contract for connector-owned reads of immutable KV boundary pages."""

from __future__ import annotations

import json
from pathlib import Path

from .verify_lease_contract import ContractError, verify_contract


CONNECTOR_API = "jj-block-state-read-leases/v1"
REQUIRED_SEMANTICS = (
    "scheduler-binds-owning-block-pool",
    "block-state-and-boundary-offers-describe-one-scheduled-step",
    "offered-recurrent-pages-are-retained-hashed-or-copy-on-write-destinations",
    "full-attention-prefix-pages-remain-immutable-while-referenced",
    "connector-metadata-is-built-before-scheduler-dispatch",
    "worker-post-forward-follows-target-and-mtp-state-writes",
    "worker-metadata-aggregates-distinct-physical-ranks",
    "pending-connector-work-keeps-idle-engine-stepping",
    "no-forward-steps-do-not-call-wait-for-save",
)
REQUIRED_SYMBOLS = {
    "vllm/distributed/kv_transfer/kv_connector/v1/base.py": (
        "KVConnectorBase_V1.bind_gpu_block_pool",
        "KVConnectorBase_V1.build_connector_worker_meta",
        "KVConnectorBase_V1.has_pending_push_work",
        "KVConnectorWorkerMetadata.aggregate",
    ),
    "vllm/distributed/kv_transfer/kv_connector/utils.py": (
        "KVOutputAggregator.from_connector",
        "KVOutputAggregator.aggregate",
    ),
    "vllm/v1/core/sched/output.py": (
        "KVConnectorBlockState.block_ids",
        "KVConnectorBlockState.boundary_state_offloads",
        "SchedulerOutput.kv_connector_block_state",
    ),
    "vllm/v1/core/sched/scheduler.py": (
        "Scheduler.__init__",
        "Scheduler.schedule",
        "Scheduler.has_requests",
        "Scheduler.update_from_output",
        "Scheduler._free_request_blocks",
    ),
    "vllm/v1/core/kv_cache_manager.py": (
        "KVCacheManager.take_boundary_state_offloads",
        "KVCacheManager.free",
    ),
    "vllm/v1/core/single_type_kv_cache_manager.py": (
        "SingleTypeKVCacheManager.take_pending_boundary_state_offloads",
        "MambaManager.cache_blocks",
        "MambaManager.allocate_new_blocks",
        "MambaManager.pop_blocks_for_free",
    ),
    "vllm/v1/core/block_pool.py": ("BlockPool.touch", "BlockPool.free_blocks"),
    "vllm/v1/worker/gpu/model_runner.py": (
        "GPUModelRunner.update_requests",
        "GPUModelRunner.sample_tokens",
    ),
    "vllm/v1/worker/gpu/kv_connector.py": (
        "ActiveKVConnector.pre_forward",
        "ActiveKVConnector.post_forward",
        "ActiveKVConnector.no_forward",
    ),
    # Alignment determines which recurrent page remains a read-only source
    # when a following step chooses its running-state destination.
    "vllm/v1/worker/mamba_utils.py": (),
}


def verify_connector_job_contract(root: Path, contract_path: Path) -> list[Path]:
    """Require the reviewed semantic contract and exact source fingerprints.

    Capability names identify review obligations; they do not establish CUDA
    correctness. A changed source revision requires conformance review and
    updated fingerprints, even when its connector class names are unchanged.
    """
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ContractError(f"cannot read connector-job contract: {error}") from error
    if contract.get("connector_api") != CONNECTOR_API:
        raise ContractError(
            "source contract does not qualify connector-job read leases"
        )
    if not set(REQUIRED_SEMANTICS).issubset(contract.get("required_semantics", ())):
        raise ContractError("connector-job contract omits required ownership semantics")
    records = {record.get("path"): record for record in contract.get("files", ())}
    for path, required in REQUIRED_SYMBOLS.items():
        record = records.get(path)
        if record is None or not set(required).issubset(
            record.get("required_symbols", ())
        ):
            raise ContractError(
                f"connector-job contract omits required capabilities: {path}"
            )
    return verify_contract(root, contract_path)
