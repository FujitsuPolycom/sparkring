"""Exercise the patched dense-indexer control flow with CPU kernel substitutes."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import textwrap
from types import ModuleType, SimpleNamespace as NS

import pytest

PATCHES = Path(__file__).parent / "patches"
spec = importlib.util.spec_from_file_location("sglang_overlay_installer", PATCHES / "apply.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


def method_source(name, *, postimage=True):
    # The patch includes complete method context so this executes the shipped
    # implementation without importing CUDA, Triton or the full SGLang runtime.
    lines = []
    in_hunk = False
    for line in (PATCHES / "bounded-prefill.patch").read_text(encoding="utf-8").splitlines():
        if line.startswith("@@"):
            in_hunk = True
        elif in_hunk and line.startswith((" ", "+" if postimage else "-")):
            lines.append(line[1:])
    start = next(i for i, line in enumerate(lines) if re.match(rf"\s*def {name}\(", line))
    indent = len(lines[start]) - len(lines[start].lstrip())
    end = start + 1
    while end < len(lines):
        line = lines[end]
        if line.lstrip().startswith(("def ", "class ", "@")) and len(line) - len(line.lstrip()) <= indent:
            break
        end += 1
    return textwrap.dedent("\n".join(lines[start:end]))


class Harness:
    def __init__(self, monkeypatch, *, postimage=True, budget=96):
        self.torch = pytest.importorskip("torch")
        t = self.torch
        self.calls = []
        self.mask_calls = []
        module = ModuleType("sglang.kernels.ops.attention.dsv4.fp4_indexer")
        module.quantize_fp4_indexer_tensor = lambda q, rne: (q[:, :64], t.zeros(q.shape[0], dtype=t.int32))
        monkeypatch.setitem(sys.modules, module.__name__, module)

        def logits(q, k, weights, starts, ends, width):
            rows = q[0][:, 0, 0]
            columns = t.arange(width)[None, :]
            values = t.sin((rows[:, None] + 1) * (starts[:, None] + columns + 1) * 0.173) + columns * 0.001
            self.calls.append((len(rows), width, starts.clone(), ends.clone()))
            return values.float()

        def topk(scores, lengths, *, out_offsets, out_indices):
            for row, length in enumerate(lengths.tolist()):
                out_indices[row].fill_(-1)
                n = min(length, out_indices.shape[1])
                if n:
                    out_indices[row, :n] = scores[row, :length].topk(n).indices + out_offsets[row]

        def candidate_blocks(scores, lengths, *, topk_blocks, block_size):
            self.mask_calls.append(tuple(scores.shape))
            result = t.zeros_like(scores, dtype=t.bool)
            for row, length in enumerate(lengths.flatten().tolist()):
                block_scores = [scores[row, start:min(start + block_size, length)].max().item()
                                for start in range(0, length, block_size)]
                for block in sorted(range(len(block_scores)), key=lambda i: block_scores[i], reverse=True)[:topk_blocks]:
                    result[row, block * block_size:min((block + 1) * block_size, length)] = True
            return result

        env = dict(torch=t, Optional=__import__("typing").Optional,
                   ceil_align=lambda n, align: (n + align - 1) // align * align,
                   _as_int_list=lambda value: value.tolist(),
                   _dense_fp4_mqa_logits=logits, topk_transform_ragged_v2=topk,
                   select_candidate_blocks=candidate_blocks,
                   _TORCH_INDEXER_SCORE_BUDGET_BYTES=budget,
                   _DENSE_INDEXER_LOGITS_BUDGET_BYTES=budget)
        exec(method_source("_mask_topk_scores"), env)
        exec(method_source("_low_ratio_index_topk_dense", postimage=postimage), env)
        if not postimage:
            exec(method_source("_publish_or_consume_candidates", postimage=False), env)
        self.env = env

    def run(self, lengths=(7, 4, 1), contexts=(18, 10, 1), *, publish=False,
            consume=None, tail=None, cp=False, raw=True):
        t = self.torch
        rows = sum(lengths)
        positions = t.cat([t.arange(context - length, context) for length, context in zip(lengths, contexts)])
        x = t.cat([t.arange(context - length, context) + i * 100
                   for i, (length, context) in enumerate(zip(lengths, contexts))]).float()[:, None]
        pages = t.full((rows + 2, 5), 123, dtype=t.int32)
        indices = t.full_like(pages, 456) if raw else None
        core = NS(sparse_page_indices=lambda ratio: pages, sparse_raw_indices=lambda ratio: indices)
        backend = NS(
            req_to_token=t.stack([t.arange(128) + i * 128 for i in range(len(lengths))]),
            forward_metadata=NS(core_metadata=core), candidate_masks=consume,
            tail_forward_metadata=None if tail is None else NS(late_layer_tail=NS(
                extend_seq_lens_cpu=tail, cp_metadata=object() if cp else None)),
            token_to_kv_pool=NS(get_low_ratio_index_k_fp4=lambda layer, slots: (slots, slots)))
        if "_publish_or_consume_candidates" in self.env:
            backend._publish_or_consume_candidates = lambda *args: self.env["_publish_or_consume_candidates"](backend, *args)
        layer = NS(layer_id=0, compress_ratio=2, freqs_cis=t.zeros(128, 1), indexer=NS(
            index_topk=5, is_candidate_source=publish, uses_candidates=consume is not None,
            candidate_topk_blocks=1, candidate_block_size=2,
            queries=lambda q, freqs: q[:, None, :].expand(-1, 1, 128),
            head_weights=lambda value: t.ones(value.shape[0], 1)))
        batch = NS(seq_lens_cpu=t.tensor(contexts), req_pool_indices=t.arange(len(lengths)))
        self.env["_low_ratio_index_topk_dense"](backend, layer, x, x, positions, batch, t.tensor(lengths), list(lengths))
        return NS(pages=pages, raw=indices, masks=backend.candidate_masks)


@pytest.mark.parametrize("publish", [False, True])
@pytest.mark.parametrize("lengths,contexts", [((7, 4, 1), (18, 10, 1)), ((0, 4, 0), (12, 10, 2)), ((1, 0), (1, 0))])
def test_chunking_matches_full_logits(monkeypatch, publish, lengths, contexts):
    reference = Harness(monkeypatch, postimage=False).run(lengths, contexts, publish=publish)
    harness = Harness(monkeypatch)
    actual = harness.run(lengths, contexts, publish=publish)
    t = harness.torch
    assert t.equal(actual.pages, reference.pages)
    assert t.equal(actual.raw, reference.raw)
    if publish:
        assert len(actual.masks) == len(reference.masks)
        assert all(t.equal(a, b) for a, b in zip(actual.masks, reference.masks))
    assert all(rows * width * 4 <= 96 or rows == 1 for rows, width, *_ in harness.calls)
    assert all(width % 4 == 0 for _, width, *_ in harness.calls)


@pytest.mark.parametrize("raw", [False, True])
def test_consumers_keep_request_offsets_and_mask_underfilled_topk(monkeypatch, raw):
    publisher = Harness(monkeypatch, postimage=False).run(publish=True)
    reference = Harness(monkeypatch, postimage=False).run(consume=publisher.masks, raw=raw)
    harness = Harness(monkeypatch)
    actual = harness.run(consume=publisher.masks, raw=raw)
    assert harness.torch.equal(actual.pages, reference.pages)
    if raw:
        assert harness.torch.equal(actual.raw, reference.raw)
    assert (actual.pages[:11, 2:] == -1).all()
    assert len(harness.calls) > 2


@pytest.mark.parametrize("tail", [(3, 2, 0), (0, 0, 0), (99, 99, 99)])
def test_bounded_replay_publishes_only_tail_rows(monkeypatch, tail):
    reference = Harness(monkeypatch, postimage=False).run(publish=True)
    harness = Harness(monkeypatch)
    actual = harness.run(publish=True, tail=tail)
    assert harness.torch.equal(actual.pages, reference.pages)
    assert harness.torch.equal(actual.raw, reference.raw)
    for count, full, compact in zip(tail, reference.masks, actual.masks):
        count = min(count, full.shape[0])
        assert harness.torch.equal(compact, full[full.shape[0] - count:])
    assert sum(rows for rows, width in harness.mask_calls) == sum(mask.shape[0] for mask in actual.masks)


def test_context_parallelism_retains_full_candidate_masks(monkeypatch):
    reference = Harness(monkeypatch, postimage=False).run(publish=True)
    harness = Harness(monkeypatch)
    actual = harness.run(publish=True, tail=(3, 2, 0), cp=True)
    assert all(harness.torch.equal(a, b) for a, b in zip(actual.masks, reference.masks))


def test_tail_masks_feed_late_layer_consumers_without_row_shift(monkeypatch):
    tail = (3, 2, 0)
    full = Harness(monkeypatch, postimage=False).run(publish=True)
    expected_masks = [mask[mask.shape[0] - count:] for mask, count in zip(full.masks, tail)]
    compact = Harness(monkeypatch).run(publish=True, tail=tail)
    reference = Harness(monkeypatch, postimage=False).run(lengths=tail, consume=expected_masks)
    harness = Harness(monkeypatch)
    actual = harness.run(lengths=tail, consume=compact.masks)
    assert harness.torch.equal(actual.pages, reference.pages)
    assert harness.torch.equal(actual.raw, reference.raw)


def test_empty_batch_avoids_kernel_launches(monkeypatch):
    harness = Harness(monkeypatch)
    actual = harness.run(lengths=(0, 0), contexts=(18, 10), publish=True)
    assert harness.calls == []
    assert len(actual.masks) == 2
    assert all(mask.numel() == 0 for mask in actual.masks)
    assert (actual.pages == -1).all()


def test_single_row_larger_than_budget_remains_processable(monkeypatch):
    harness = Harness(monkeypatch, budget=1)
    actual = harness.run()
    reference = Harness(monkeypatch, postimage=False).run()
    assert harness.torch.equal(actual.pages, reference.pages)
    assert all(rows == 1 for rows, *_ in harness.calls)


def test_patch_receipts_match_shipped_bytes():
    manifest = json.loads((PATCHES / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["files"]) == 16
    for patch in manifest["patches"]:
        assert installer.sha256(PATCHES / patch["path"]) == patch["sha256"]


def fixture_overlay(tmp_path):
    source = tmp_path / "source"
    patches = tmp_path / "patches"
    source.mkdir()
    patches.mkdir()
    (source / "a.py").write_bytes(b"first\n")
    data = [("one.patch", "first", "second"), ("two.patch", "second", "third")]
    for name, before, after in data:
        (patches / name).write_text(f"--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-{before}\n+{after}\n", encoding="utf-8", newline="\n")
    digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
    manifest = dict(schema_version=1, source_revision="test-fixture", patches=[
        dict(path=name, sha256=installer.sha256(patches / name)) for name, *_ in data], files=[
        dict(path="a.py", input_sha256=digest("first\n"), verify_overlay_sha256=digest("second\n"), output_sha256=digest("third\n"))])
    (patches / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return source, patches


@pytest.mark.skipif(shutil.which("git") is None, reason="git is required to apply source overlays")
def test_installer_applies_exact_sources_and_verifies_idempotently(tmp_path):
    source, patches = fixture_overlay(tmp_path)
    receipt = installer.install(source, patches)
    assert (source / "a.py").read_bytes() == b"third\n"
    assert installer.install(source, patches, check=True) == receipt
    assert installer.install(source, patches) == receipt


@pytest.mark.parametrize("damage", ["source", "patch", "output_hash"])
def test_installer_rejects_mismatches_without_source_edits(tmp_path, damage):
    source, patches = fixture_overlay(tmp_path)
    if damage == "source":
        (source / "a.py").write_bytes(b"different\n")
    elif damage == "patch":
        (patches / "two.patch").write_bytes(b"invalid\n")
    else:
        manifest = json.loads((patches / "manifest.json").read_text())
        manifest["files"][0]["output_sha256"] = "0" * 64
        (patches / "manifest.json").write_text(json.dumps(manifest))
    before = (source / "a.py").read_bytes()
    with pytest.raises(ValueError, match="mismatch"):
        installer.install(source, patches)
    assert (source / "a.py").read_bytes() == before


@pytest.mark.parametrize("path", ["../escape", "/escape", "nested/../../escape", "nested\\escape"])
def test_installer_rejects_escaping_paths(tmp_path, path):
    with pytest.raises(ValueError, match="Invalid overlay path"):
        installer.source_path(tmp_path, path)
