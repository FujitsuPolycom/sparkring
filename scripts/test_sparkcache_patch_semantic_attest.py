import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from sparkcache_patch_semantic_attest import (  # noqa: E402
    SemanticAttestationError,
    attest_kv_output_aggregator_source,
    attest_sources,
)


SCHEDULER = """
def _handle_invalid_blocks(self):
    if not total_failed_requests:
        return set()
    if sync_blocks_to_evict and not self.recompute_kv_load_failures:
        self.kv_cache_manager.evict_blocks(sync_blocks_to_evict)
    if should_fail:
        all_failed_req_ids = async_failed_req_ids | sync_failed_req_ids
        logger.error('failing')
        return all_failed_req_ids
    for spark_req_id in async_failed_req_ids | sync_failed_req_ids:
        spark_request = self.requests.get(spark_req_id)
        if spark_request is None:
            continue
        if spark_request.num_output_placeholders:
            spark_request.async_tokens_to_discard += spark_request.num_output_placeholders
            spark_request.num_output_placeholders = 0
    logger.warning('recovered')
    return sync_failed_req_ids
"""

VMM = """
def _validate_kv_transfer_vmm(self):
    if self.model_config is not None:
        return
    if self.kv_transfer_config is not None and self.kv_transfer_config.kv_connector == 'SparkContextCacheConnector':
        return
    raise ValueError('unsupported')
"""

AGGREGATOR = """
def aggregate(self, outputs, output_rank=0):
    finished_recving = set()
    invalid_block_ids = set()
    for model_runner_output in outputs:
        kv_output = model_runner_output.kv_connector_output
        update_finished_set(
            kv_output.finished_recving,
            self._recv_remaining_count,
            finished_recving,
        )
        invalid_block_ids |= kv_output.invalid_block_ids
    output = outputs[output_rank]
    output.kv_connector_output = KVConnectorOutput(
        finished_recving=finished_recving or None,
        invalid_block_ids=invalid_block_ids,
    )
    return output
"""


def test_exact_semantics_pass():
    attest_sources(SCHEDULER, VMM)


def test_exact_kv_output_aggregation_semantics_pass():
    attest_kv_output_aggregator_source(AGGREGATOR)


@pytest.mark.parametrize(
    "source",
    [
        AGGREGATOR.replace(
            "invalid_block_ids |= kv_output.invalid_block_ids",
            "invalid_block_ids = kv_output.invalid_block_ids",
        ),
        AGGREGATOR.replace(
            "kv_output.finished_recving",
            "kv_output.finished_sending",
        ),
        AGGREGATOR.replace(
            "self._recv_remaining_count",
            "self._send_remaining_count",
        ),
        AGGREGATOR.replace(
            "finished_recving=finished_recving or None",
            "finished_recving=None",
        ),
        AGGREGATOR.replace(
            "invalid_block_ids=invalid_block_ids",
            "invalid_block_ids=set()",
        ),
        AGGREGATOR.replace(
            "for model_runner_output in outputs:",
            "for model_runner_output in outputs[:1]:",
        ),
    ],
)
def test_kv_output_aggregation_near_misses_fail(source):
    with pytest.raises(SemanticAttestationError):
        attest_kv_output_aggregator_source(source)


def test_current_vllm_compat_method_name_passes_same_semantics():
    attest_sources(
        SCHEDULER,
        VMM.replace("_validate_kv_transfer_vmm", "_verify_kv_transfer_compat"),
    )


def test_unrecognized_vmm_method_name_fails():
    with pytest.raises(SemanticAttestationError):
        attest_sources(
            SCHEDULER,
            VMM.replace("_validate_kv_transfer_vmm", "_unrelated_validator"),
        )


@pytest.mark.parametrize(
    "scheduler",
    [
        SCHEDULER.replace("async_failed_req_ids | sync_failed_req_ids", "async_failed_req_ids"),
        SCHEDULER.replace("for spark_req_id in", "for request_id in"),
        SCHEDULER.replace(
            "self.requests.get(spark_req_id)", "self.requests.get('unrelated')"
        ),
        SCHEDULER.replace("if spark_request is None:", "if spark_request is False:"),
        SCHEDULER.replace("            continue", "            return all_failed_req_ids"),
        SCHEDULER.replace("+= spark_request.num_output_placeholders", "+= 1"),
        SCHEDULER.replace(
            "spark_request.async_tokens_to_discard += spark_request.num_output_placeholders\n            spark_request.num_output_placeholders = 0",
            "spark_request.num_output_placeholders = 0\n            spark_request.async_tokens_to_discard += spark_request.num_output_placeholders",
        ),
        SCHEDULER.replace("spark_request.num_output_placeholders = 0", "spark_request.num_output_placeholders = 1"),
        SCHEDULER.replace(
            "for spark_req_id in async_failed_req_ids | sync_failed_req_ids:",
            "return all_failed_req_ids\n    for spark_req_id in async_failed_req_ids | sync_failed_req_ids:",
        ),
        SCHEDULER.replace(
            "spark_request = self.requests.get(spark_req_id)",
            "continue\n        spark_request = self.requests.get(spark_req_id)",
        ),
        SCHEDULER.replace(
            "for spark_req_id in async_failed_req_ids | sync_failed_req_ids:",
            "while True:\n        return all_failed_req_ids\n"
            "    for spark_req_id in async_failed_req_ids | sync_failed_req_ids:",
        ),
        SCHEDULER.replace(
            "for spark_req_id in async_failed_req_ids | sync_failed_req_ids:",
            "try:\n        return all_failed_req_ids\n"
            "    except BaseException:\n        return all_failed_req_ids\n"
            "    for spark_req_id in async_failed_req_ids | sync_failed_req_ids:",
        ),
        SCHEDULER.replace(
            "for spark_req_id in async_failed_req_ids | sync_failed_req_ids:",
            "try:\n        pass\n"
            "    finally:\n        return all_failed_req_ids\n"
            "    for spark_req_id in async_failed_req_ids | sync_failed_req_ids:",
        ),
        SCHEDULER.replace(
            "for spark_req_id in async_failed_req_ids | sync_failed_req_ids:",
            "match 1:\n"
            "        case _:\n"
            "            return all_failed_req_ids\n"
            "    for spark_req_id in async_failed_req_ids | sync_failed_req_ids:",
        ),
        SCHEDULER.replace(
            "spark_request.num_output_placeholders = 0",
            "logger.info('repair')\n            spark_request.num_output_placeholders = 0",
        ),
    ],
)
def test_scheduler_near_misses_fail(scheduler):
    with pytest.raises(SemanticAttestationError):
        attest_sources(scheduler, VMM)


def test_unproved_fallthrough_try_before_repair_fails_closed():
    scheduler = SCHEDULER.replace(
        "for spark_req_id in async_failed_req_ids | sync_failed_req_ids:",
        "try:\n        logger.info('probe')\n"
        "    except Exception:\n        logger.warning('ignored')\n"
        "    else:\n        logger.info('ok')\n"
        "    finally:\n        logger.info('done')\n"
        "    for spark_req_id in async_failed_req_ids | sync_failed_req_ids:",
    )
    with pytest.raises(SemanticAttestationError):
        attest_sources(scheduler, VMM)


@pytest.mark.parametrize(
    "prefix",
    [
        "if maybe_skip:\n        return sync_failed_req_ids\n    ",
        "if maybe_raise:\n        raise RuntimeError('skip')\n    ",
        "for _ in (1,):\n        return sync_failed_req_ids\n    ",
        "for _ in (1,):\n        match 1:\n            case _:\n                return sync_failed_req_ids\n    ",
    ],
)
def test_scheduler_rejects_any_unapproved_bypass_before_repair(prefix):
    scheduler = SCHEDULER.replace(
        "for spark_req_id in async_failed_req_ids | sync_failed_req_ids:",
        prefix + "for spark_req_id in async_failed_req_ids | sync_failed_req_ids:",
    )
    with pytest.raises(SemanticAttestationError):
        attest_sources(scheduler, VMM)


@pytest.mark.parametrize(
    "vmm",
    [
        VMM.replace("== 'SparkContextCacheConnector'", "in ('SparkContextCacheConnector', 'Other')"),
        VMM.replace("'SparkContextCacheConnector'", "'OtherConnector'"),
        VMM.replace("self.kv_transfer_config is not None and ", ""),
        VMM.replace("        return\n    raise ValueError", "        pass\n    raise ValueError"),
        VMM.replace(
            "    if self.kv_transfer_config is not None",
            "    return\n    if self.kv_transfer_config is not None",
        ),
        VMM.replace(
            "    if self.kv_transfer_config is not None",
            "    if self.reject_early:\n        raise ValueError('early')\n"
            "    if self.kv_transfer_config is not None",
        ),
        VMM.replace(
            "    if self.kv_transfer_config is not None",
            "    match 1:\n"
            "        case _:\n"
            "            return\n"
            "    if self.kv_transfer_config is not None",
        ),
        VMM.replace(
            "        return\n    raise ValueError('unsupported')",
            "        return 1\n    raise ValueError('unsupported')",
        ),
    ],
)
def test_vmm_near_misses_fail(vmm):
    with pytest.raises(SemanticAttestationError):
        attest_sources(SCHEDULER, vmm)
