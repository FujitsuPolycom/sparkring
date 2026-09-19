"""Persistent graph views must not reuse request IDs from a longer batch."""

from types import SimpleNamespace

import pytest
import torch

from integrations.vllm.patches.qwen_qsa_padding import apply


SOURCE = """class Qwen4ExpQSAMetadataBuilder:
    def build(self, cm):
        request_ids = cm.token_to_req_indices(self._request_ids)
        num_mapped_tokens = int(cm.query_start_loc_cpu[-1])
        if num_mapped_tokens < cm.num_actual_tokens:
            request_ids[num_mapped_tokens:].fill_(-1)
        return request_ids
"""


def builder(source=SOURCE):
    namespace = {}
    exec(source, namespace)
    result = namespace["Qwen4ExpQSAMetadataBuilder"]()
    result._request_ids = torch.full((64,), 987, dtype=torch.int32)
    return result


def metadata(live, actual=None):
    cache = []
    actual = live if actual is None else actual

    def mapping(buffer):
        if not cache:
            buffer[: max(live, actual)].fill_(0)
            cache.append(buffer[:actual])
        return cache[0]

    return SimpleNamespace(
        token_to_req_indices=mapping,
        query_start_loc_cpu=torch.tensor([0, live]),
        num_actual_tokens=actual,
    )


def test_unpatched_builder_leaves_stale_ids_in_a_captured_view():
    owner = builder()
    captured = owner.build(metadata(28))
    owner.build(metadata(26))
    assert captured[26:].tolist() == [0, 0]


@pytest.mark.parametrize("live", [0, 1, 17, 18, 19, 25, 26, 27, 28])
def test_shorter_refresh_clears_all_inactive_owned_capacity(live):
    owner = builder(apply(SOURCE))
    captured = owner.build(metadata(28))
    owner.build(metadata(live))
    assert captured[:live].tolist() == [0] * live
    assert captured[live:].tolist() == [-1] * (28 - live)
    assert owner._request_ids[live:].eq(-1).all()


def test_reused_mapping_and_full_graph_padding_remain_valid():
    owner = builder(apply(SOURCE))
    common = metadata(26, actual=28)
    first = owner.build(common)
    second = owner.build(common)
    assert first.data_ptr() == second.data_ptr()
    assert second[:26].eq(0).all()
    assert owner._request_ids[26:].eq(-1).all()


def test_shared_mapping_is_not_overwritten_by_another_builder():
    first, second = builder(apply(SOURCE)), builder(apply(SOURCE))
    common = metadata(26)
    retained = first.build(common)
    shared = second.build(common)
    assert shared.data_ptr() == retained.data_ptr()
    assert retained.eq(0).all()
    assert first._request_ids[26:].eq(-1).all()
    assert second._request_ids[26:].eq(-1).all()


def test_mapping_longer_than_actual_view_remains_intact():
    owner = builder(apply(SOURCE))
    result = owner.build(metadata(28, actual=26))
    assert result.shape == (26,)
    assert owner._request_ids[:28].eq(0).all()
    assert owner._request_ids[28:].eq(-1).all()


def test_shorter_refresh_preserves_multiple_request_ids():
    owner = builder(apply(SOURCE))
    captured = owner.build(metadata(28))
    common = metadata(26)

    def mapping(buffer):
        buffer[:12].fill_(0)
        buffer[12:26].fill_(1)
        return buffer[:26]

    common.token_to_req_indices = mapping
    common.query_start_loc_cpu = torch.tensor([0, 12, 26])
    owner.build(common)
    assert captured.tolist() == [0] * 12 + [1] * 14 + [-1, -1]


@pytest.mark.parametrize(
    "source",
    [
        SOURCE.replace("build", "refresh"),
        SOURCE.replace("cm.token_to_req_indices", "cm.other"),
        apply(SOURCE),
    ],
)
def test_source_drift_and_duplicate_application_are_rejected(source):
    with pytest.raises(ValueError):
        apply(source)
