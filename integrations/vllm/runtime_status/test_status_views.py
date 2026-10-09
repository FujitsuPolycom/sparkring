"""Presentation, packaging and authentication checks without a GPU/server restart."""

import asyncio
import base64
import hashlib
import json
import re
from pathlib import Path
import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from sparkring_runtime_status import collector, plugin, presentation
from test_runtime_status import Engine, NS, config, service, worker


def fixture():
    result = asyncio.run(service(Engine()).snapshot())
    result["effective"]["model_type"] = {"state": "known", "value": "mimo_v2", "source": "resolved_vllm_config"}
    result["provenance"] = {"state": "known", "startup_identity": {
        "sparkring": "shared-2026.09.4-rc.4", "eugr": "eugr/spark-vllm-b12x:nightly-20260923",
        "vllm_version": "0.1.dev21504+g57fdda71b.d20260923", "b12x_version": "1.3.0"},
        "sources": {"vllm": {"commit": "a" * 40}}, "composition_sha256": "b" * 64}
    return result


def observed_fixture():
    """Synthetic transport/resource facts; no machine inventory in test data."""
    doc = fixture()
    fact = lambda value: {'state': 'known', 'value': value, 'source': 'resident_test_fixture'}
    doc['configured']['environment']['VLLM_ENABLE_ROCE_ALLREDUCE'] = fact(True)
    doc['configured']['environment']['VLLM_ROCE_ALLREDUCE_MAX_SIZE'] = fact(2097152)
    doc['configured']['environment']['NCCL_ALGO'] = {'state': 'unknown', 'source': 'process_environment', 'reason': 'not_set_in_environment'}
    for rank in doc['workers']['ranks']:
        rank['effective'].update(tp_rocenante_enabled=fact(True), tp_roce_allreduce_max_bytes=fact(2097152),
                                 tp_nccl_version=fact(23203), tp_roce_hcas=fact('rdma0,rdma1'))
        rank['configured'] = {'environment': dict(doc['configured']['environment'])}
        rank['resources'] = {'node': 'example-worker', 'collected_at_unix_ns': 1790272800000000000,
            'memory': {'state': 'known', 'total_bytes': 128 * 1024**3, 'used_bytes': 96 * 1024**3,
                       'available_bytes': 32 * 1024**3, 'swap_used_bytes': 0},
            'container_memory': {'state': 'known', 'current_bytes': 80 * 1024**3, 'limit_bytes': None},
            'filesystems': [{'state': 'known', 'paths': ['/cache'], 'total_bytes': 1024**4,
                             'used_bytes': 768 * 1024**3, 'available_bytes': 256 * 1024**3}]}
        rank['versions'] = {'host_nvidia_driver': fact('580.173.02'), 'torch_build_cuda': fact('13.0'),
            'cuda_toolkit': fact('13.4.2'), 'cuda_compiler': fact('13.4.92'),
            'packages': {'torch': fact('2.13.0+cu130')}, 'mapped_libraries': [
                {'component': 'nccl', 'path': '/opt/example/libnccl.so.2.32.3', 'version': '2.32.3'}]}
        rank['transport'] = {'groups': {'TP': {'state': 'known', 'world_size': fact(2),
            'rocenante': {'enabled': fact(True), 'allreduce_max_bytes': fact(2097152),
                          'allgather_max_bytes': fact(16777216), 'hcas': fact('rdma0,rdma1')},
            'nccl': {'available': fact(True)}}}, 'nics': [{'hca': 'rdma0',
                'pci': {'bdf': fact('0002:01:00.0'), 'device_key': fact('0002:01:00'),
                        'current_link_speed': fact('16.0'), 'current_link_width': fact(4)},
                'netdevs': [{'name': 'eth0', 'speed_mbps': fact(200000), 'operstate': fact('up'),
                            'mtu': fact(9000), 'mac': fact('02:00:00:00:00:01')}],
                'ports': [{'port': 1, 'state': fact('ACTIVE'), 'counters': {
                    'port_xmit_data': fact(262144), 'port_rcv_data': fact(0), 'port_rcv_errors': fact(0)}}]}]}
    doc['observed']['speculative_acceptance'] = {'state': 'known', 'source': 'prometheus_speculation_counters',
        'recent': {'state': 'not_observed', 'reason': 'no_drafts_in_window'},
        'lifetime': {'state': 'known', 'rounds': 100, 'acceptance_rate': .6,
                     'estimated_tokens_per_verification': 2.8,
                     'positions': [{'position': 1, 'drafted': 100, 'accepted': 80, 'acceptance_rate': .8}]}}
    return doc


def test_resident_transport_resolves_without_inventing_nccl_defaults():
    doc = observed_fixture()
    view = presentation.summarize(doc)
    rows = {row['label']: row for _, group in view['groups'] for row in group}
    assert rows['RoCEnante available (TP)']['resolved'] == 'ON'
    assert rows['RoCE all-reduce size limit']['resolved'] == '2 MiB'
    assert rows['NCCL runtime version (TP)']['resolved'] == '2.32.3'
    assert rows['NCCL algorithm override']['configured'] == 'Not set in environment'
    assert rows['NCCL algorithm override']['resolved'] == 'Not checked at runtime'
    assert rows['NCCL algorithm override']['evidence'] != 'Running worker'
    doc['workers']['ranks'][1]['effective']['tp_roce_hcas']['value'] = 'rdma2,rdma3'
    row = next(row for _, group in presentation.summarize(doc)['groups'] for row in group
               if row['label'] == 'RoCEnante selected HCAs')
    assert row['severity'] != 'bad' and row['resolved'] == 'Differs by node'


def test_resource_versions_link_rates_and_acceptance_have_explicit_scopes():
    doc = observed_fixture()
    text = presentation.render_text(doc)
    html = presentation.render_report(doc)[1]
    for value in ('32.0 GiB', '256.0 GiB', '580.173.02', '13.0', '13.4.2', '2.32.3',
                  '200.0 Gb/s', '16.0 GT/s × 4', '0002:01:00.0', '60.0%', '2.80', '1.0 MiB'):
        assert value in text and value in html
    assert 'No drafts in this period' in text
    assert 'not measured throughput' in html
    assert 'Running totals for the whole host, not only this model' in html
    assert 'This page runs no bandwidth or latency test' in html
    assert 'Unlimited' in text and 'No data' in text


def test_unused_single_rank_groups_are_collapsed_without_hiding_active_groups():
    doc = observed_fixture()
    fact = lambda value: {'state': 'known', 'source': 'fixture', 'value': value}
    groups = doc['workers']['ranks'][0]['transport']['groups']
    groups['PP'] = {'state': 'known', 'world_size': fact(1), 'rocenante': {}, 'nccl': {}}
    groups['EP'] = {'state': 'known', 'world_size': fact(2),
                    'rocenante': {'enabled': fact(False)}, 'nccl': {'available': fact(True)}}
    sections = {s['id']: s for s in presentation.detail_sections(doc, presentation.summarize(doc))}
    assert sections['singleton-groups']['open'] is False
    assert sections['singleton-groups']['rows'][0][1] == 'PP'
    assert sections['singleton-groups']['rows'][0][3] == 'Not needed (one worker)'
    assert any(row[1] == 'EP' for row in sections['communicators']['rows'])
    assert not any(row[1] == 'PP' for row in sections['communicators']['rows'])
    groups['PP']['nccl']['available'] = fact(True)
    sections = {s['id']: s for s in presentation.detail_sections(doc, presentation.summarize(doc))}
    assert any(row[1] == 'PP' for row in sections['communicators']['rows'])


def test_unknown_group_size_is_not_labeled_single_rank():
    doc = observed_fixture()
    doc['workers']['ranks'][0]['transport']['groups']['DP'] = {'state': 'known'}
    sections = {s['id']: s for s in presentation.detail_sections(doc, presentation.summarize(doc))}
    assert any(row[1] == 'DP' for row in sections['communicators']['rows'])


def test_detail_tables_escape_nested_paths_and_identifiers():
    doc = observed_fixture()
    doc['workers']['ranks'][0]['versions']['mapped_libraries'][0]['path'] = '<script>evil()</script>\x1b[31m'
    doc['workers']['ranks'][0]['resources']['node'] = '<img src=x onerror=alert(1)>'
    html = presentation.render_report(doc)[1]
    assert '<script>evil()' not in html and '<img src=x' not in html
    assert '&lt;script&gt;evil()' in html and '&lt;img src=x' in html
    assert '\x1b' not in presentation.render_text(doc)


def test_extended_snapshot_preserves_the_json_schema():
    jsonschema = pytest.importorskip('jsonschema')
    schema = json.loads(Path(__file__).with_name('schema-v1.json').read_text())
    jsonschema.validate(observed_fixture(), schema)


def test_views_share_the_existing_snapshot_cache_and_preserve_json():
    app = FastAPI()
    plugin.StatusPlugin().attach_router(app)
    engine = Engine()
    app.state.sparkring_status_service = service(engine)
    with TestClient(app) as client:
        data = client.get("/v1/sparkring/status")
        text = client.get("/v1/sparkring/status.txt")
        html = client.get("/v1/sparkring/status/view")
        assert data.json()["schema"] == "sparkring-runtime-status/v1"
        assert text.status_code == html.status_code == 200
        assert text.headers["content-type"].startswith("text/plain")
        assert html.headers["content-type"].startswith("text/html")
        assert "Tensor parallel" in text.text and "Tensor parallel" in html.text
        assert "Unknown values do not mean OFF." in text.text and "Unknown values do not mean OFF." in html.text
        assert "execution evidence" not in text.text + html.text
        assert len(engine.calls) == 1
        for path in ("/v1/sparkring/status", "/v1/sparkring/status.txt", "/v1/sparkring/status/view"):
            assert client.post(path).status_code == 405
        assert all(r.headers["cache-control"] == "no-store" for r in (data, text, html))


@pytest.mark.parametrize("suffix", ["", ".txt", "/view"])
def test_initialization_preserves_503_with_correct_representation(suffix):
    app = FastAPI()
    plugin.StatusPlugin().attach_router(app)
    with TestClient(app) as client:
        response = client.get("/v1/sparkring/status" + suffix)
    assert response.status_code == 503
    # JSON keeps the schema's state value; the text and browser views name it plainly.
    assert ("initializing" in response.text) if suffix == "" else ("starting up" in response.text.lower())
    assert response.headers["cache-control"] == "no-store"


def test_html_escapes_runtime_values_and_text_strips_terminal_controls():
    doc = fixture()
    payload = '</title><script>alert("x")</script>\x1b[31m\nFAKE'
    doc["provenance"]["startup_identity"]["sparkring"] = payload
    doc["effective"]["model_type"]["value"] = payload
    page = plugin.dashboard_response(doc).body.decode()
    text = presentation.render_text(doc)
    assert '<script>alert("x")</script>' not in page
    assert "&lt;/title&gt;&lt;script&gt;" in page
    assert "\x1b" not in text and "\nFAKE" not in text
    assert "?FAKE" in text


def test_asset_hashes_match_csp_and_page_is_self_contained():
    response = plugin.dashboard_response(fixture())
    html = response.body.decode()
    csp = response.headers["content-security-policy"]
    for tag in ("style", "script"):
        body = re.search(rf"<{tag}>(.*?)</{tag}>", html, re.S).group(1)
        digest = base64.b64encode(hashlib.sha256(body.encode()).digest()).decode()
        assert f"{tag}-src 'sha256-{digest}'" in csp
    assert "unsafe-inline" not in csp
    assert 'src="http' not in html and 'href="http' not in html
    assert "connect-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp
    assert 'href="../status.txt"' in html and 'href="../status"' in html


def test_mismatch_partial_unknown_and_stale_are_not_presented_as_success():
    doc = fixture()
    doc["workers"]["ranks"][1]["effective"]["tensor_parallel_size"]["value"] = 4
    view = presentation.summarize(doc)
    row = next(row for _, rows in view["groups"] for row in rows if row["label"] == "Tensor parallel")
    assert row["severity"] == "bad" and row["agreement"] == "Workers report different values"
    assert "rank 0=2; rank 1=4" in presentation.render_text(doc)
    assert "Unknown on both workers" in presentation.render_text(doc)
    doc["workers"]["stale"] = True
    assert presentation.summarize(doc)["complete"] is False
    assert "out of date" in presentation.render_report(doc)[1]
    doc["workers"]["rank_identities_unique"] = False
    text = presentation.render_text(doc)
    assert "Same on both workers" not in text
    assert "2 of 2 reported, but rank numbers are missing or repeated" in text


def settings_row(doc, label):
    return next(row for _, rows in presentation.summarize(doc)["groups"] for row in rows if row["label"] == label)


def test_specific_cache_format_on_workers_agrees_with_its_family():
    doc = fixture()
    doc["configured"]["arguments"]["cache_dtype"] = {"state": "known", "value": "fp8",
                                                     "source": "parsed_server_arguments"}
    for rank in doc["workers"]["ranks"]:
        rank["effective"]["cache_dtype"]["value"] = "fp8_ds_mla"
    row = settings_row(doc, "KV cache dtype")
    assert (row["resolved"], row["agreement"], row["comparison"], row["severity"]) == (
        "fp8 (fp8_ds_mla on workers)", "Same on both workers", "", "neutral")
    html = presentation.render_report(doc)[1]
    assert '<details class="rank-detail neutral"><summary>Same on both workers</summary><ul><li>Rank 0: fp8_ds_mla</li>' in html
    assert "API != workers" not in html + presentation.render_text(doc)
    assert presentation.summarize(doc)["issue_count"] == presentation.summarize(fixture())["issue_count"]
    # An automatic cache dtype resolves to a concrete one.
    doc["effective"]["cache_dtype"]["value"] = "auto"
    assert settings_row(doc, "KV cache dtype")["resolved"] == "auto (fp8_ds_mla on workers)"
    # A different family is a real disagreement between the API process and the workers.
    doc["effective"]["cache_dtype"]["value"] = "bfloat16"
    row = settings_row(doc, "KV cache dtype")
    assert row["comparison"] == "vLLM changed the configured value. The API and the workers report different values"
    assert row["severity"] == row["comparison_severity"] == "bad"
    assert row["agreement_severity"] == "neutral"


def test_prefix_rule_applies_only_to_dtypes():
    fact = lambda value: {"state": "known", "value": value}
    assert presentation.refines(fact("fp8"), fact("fp8_e4m3"), "cache_dtype")
    assert not presentation.refines(fact("fp8_e4m3"), fact("fp8"), "cache_dtype")
    assert not presentation.refines(fact("fp8"), fact("fp8x"), "cache_dtype")
    assert not presentation.refines(fact("FULL"), fact("FULL_AND_PIECEWISE"), "cudagraph_mode")


def test_block_size_enlarged_by_vllm_is_footnoted_not_flagged():
    doc = fixture()
    # The shared fixture configures block_size 32; vLLM resolved 1440 in the API process and workers.
    row = settings_row(doc, "KV cache block size")
    assert (row["configured"], row["resolved"], row["agreement"], row["severity"]) == (
        "32", "1,440 *", "Same on both workers", "neutral")
    assert row["footnote"].startswith("* vLLM adjusts the KV cache block size")
    text = presentation.render_text(doc)
    html = presentation.render_report(doc)[1]
    assert "\n* vLLM adjusts the KV cache block size to suit the model" in text
    assert '<p class="footnote">* vLLM adjusts the KV cache block size' in html
    assert "Adjusted for alignment" not in text + html
    # A worker that differs is still a real problem.
    doc["workers"]["ranks"][1]["effective"]["block_size"]["value"] = 16
    row = settings_row(doc, "KV cache block size")
    assert row["agreement"] == "Workers report different values" and row["severity"] == "bad"


def test_changed_setting_marks_the_note_without_coloring_agreeing_workers():
    doc = fixture()
    doc["configured"]["arguments"]["max_num_seqs"] = {"state": "known", "value": 8,
                                                      "source": "parsed_server_arguments"}
    row = settings_row(doc, "Max sequences per batch")
    assert (row["comparison"], row["comparison_severity"]) == ("vLLM changed the configured value", "warn")
    assert (row["agreement"], row["agreement_severity"], row["severity"]) == ("Same on both workers", "neutral", "warn")
    html = presentation.render_report(doc)[1]
    assert ('<details class="rank-detail neutral"><summary>Same on both workers</summary><ul><li>Rank 0: 16</li>'
            '<li>Rank 1: 16</li></ul></details><small class="comparison warn">vLLM changed the configured value</small>') in html


def test_partial_worker_reports_are_worded_plainly():
    doc = fixture()
    del doc["workers"]["ranks"][1]["effective"]["max_num_seqs"]
    row = settings_row(doc, "Max sequences per batch")
    assert (row["agreement"], row["severity"]) == ("Only 1 of 2 workers reported", "warn")
    doc["workers"]["expected_count"] = None
    assert settings_row(doc, "Max sequences per batch")["agreement"] == "Reported by 1 worker; expected count unknown"


def test_header_names_the_served_model_and_keeps_the_architecture_line():
    doc = fixture()
    name = {"state": "known", "value": "Qwen3.8-Flash-Next-NVFP4-QAD-TP2", "source": "resolved_vllm_config"}
    doc["effective"]["served_model_name"] = dict(name)
    doc["effective"]["model_type"]["value"] = "qwen4_exp"
    for rank in doc["workers"]["ranks"]:
        rank["effective"]["served_model_name"] = dict(name)
    text = presentation.render_text(doc)
    assert text.splitlines()[1:3] == ["Model: Qwen3.8-Flash-Next-NVFP4-QAD-TP2 | TP 2 / DCP 1 / PP 1",
                                      "Model architecture: qwen4_exp"]
    html = presentation.render_report(doc)[1]
    assert '<p class="subtitle">Qwen3.8-Flash-Next-NVFP4-QAD-TP2 <span>·</span> TP 2 / DCP 1 / PP 1</p>' in html
    assert '<p class="subtitle">Model architecture: qwen4_exp</p></header>' in html
    rows = {row["label"]: row for row in dict(presentation.summarize(doc)["groups"])["Model and topology"]}
    served, architecture = rows["Served model name"], rows["Model architecture"]
    assert (served["configured"], served["resolved"], served["evidence"], served["agreement"]) == (
        "From launch arguments", "Qwen3.8-Flash-Next-NVFP4-QAD-TP2", "vLLM config", "Same on both workers")
    assert (architecture["configured"], architecture["resolved"]) == ("From checkpoint", "qwen4_exp")
    # Architecture-specific settings still follow model_type, never the name.
    assert "Hyper-connection prefill row ownership" in text
    doc["effective"]["model_type"]["value"] = "mimo_v2"
    assert "Hyper-connection prefill row ownership" not in presentation.render_text(doc)
    doc["workers"]["ranks"][1]["effective"]["served_model_name"]["value"] = "Other-Name"
    served = next(row for row in dict(presentation.summarize(doc)["groups"])["Model and topology"]
                  if row["label"] == "Served model name")
    assert served["severity"] == "bad" and served["agreement"] == "Workers report different values"


def test_unknown_served_name_is_not_replaced_by_the_architecture():
    doc = fixture()
    # The test configuration has no model_config, so the name was not collected.
    assert doc["effective"]["served_model_name"]["reason"] == "not_collected"
    text = presentation.render_text(doc)
    assert "Model: Not collected | TP 2 / DCP 1 / PP 1\nModel architecture: mimo_v2\n" in text
    assert '<p class="subtitle">Model architecture: mimo_v2</p>' in presentation.render_report(doc)[1]
    # A document produced without the field renders as unknown, not as mimo_v2.
    del doc["effective"]["served_model_name"]
    assert "Model: Unknown | TP 2" in presentation.render_text(doc)


def test_served_name_is_escaped_in_html_and_stripped_in_text():
    doc = fixture()
    doc["effective"]["served_model_name"] = {"state": "known", "source": "resolved_vllm_config",
                                             "value": '<img src=x onerror=alert(1)>\x1b[31m\nFAKE'}
    page = plugin.dashboard_response(doc).body.decode()
    text = presentation.render_text(doc)
    assert "<img src=x" not in page and "&lt;img src=x onerror=alert(1)&gt;" in page
    assert "\x1b" not in text and "\nFAKE" not in text


def test_identity_and_qwen_specific_fields():
    doc = fixture()
    text = presentation.render_text(doc)
    assert "SparkRing shared-2026.09.4-rc.4" in text and "eugr/spark-vllm-b12x:nightly-20260923" in text
    assert "Hyper-connection prefill row ownership" not in text
    doc["effective"]["model_type"]["value"] = "qwen4_exp"
    assert "Hyper-connection prefill row ownership" in presentation.render_text(doc)


def serving_doc(tp, reasoning, tool, template=None, environ=None):
    """A snapshot collected from parsed arguments, config and environment as a profile sets them."""
    environ = environ or {}
    cfg = config(tp)
    cfg.structured_outputs_config = NS(reasoning_parser=reasoning)
    ranks = []
    for rank in range(tp):
        obj = worker(rank)
        obj.vllm_config = cfg
        ranks.append(collector.worker_snapshot(obj, environ=environ, modules={}))
    args = NS(reasoning_parser=reasoning, tool_call_parser=tool, default_chat_template_kwargs=template)
    return asyncio.run(plugin.StatusService(cfg, args, Engine(result=ranks), receipt={"state": "unknown"},
                                            environ=environ).snapshot())


def rendered(doc, labels):
    rows = {row["label"]: row for _, group in presentation.summarize(doc)["groups"] for row in group}
    return {label: (rows[label]["configured"], rows[label]["resolved"], rows[label]["evidence"],
                    rows[label]["agreement"], rows[label]["severity"]) for label in labels}


SERVING_LABELS = ("Reasoning parser", "Tool-call parser", "Default chat template arguments",
                  "Shared-memory reader window")


def test_qwen_pair_serving_rows_with_save_cpu():
    # qwen38-flash-next-tp2: --reasoning-parser qwen3 --tool-call-parser qwen3_xml, installed with --save-cpu.
    doc = serving_doc(2, "qwen3", "qwen3_xml", environ={"SPARKRING_SHM_BUSY_LOOP_S": "0.002"})
    assert rendered(doc, SERVING_LABELS) == {
        "Reasoning parser": ("qwen3", "qwen3", "vLLM config", "Same on both workers", "neutral"),
        "Tool-call parser": ("qwen3_xml", "Not checked at runtime", "Launch setting only", "API server only",
                             "neutral"),
        "Default chat template arguments": ("Not set (the chat template decides)", "Not checked at runtime",
                                            "Launch setting only", "API server only", "neutral"),
        "Shared-memory reader window": ("2 ms (sparkring install --save-cpu)", "Not checked at runtime",
                                        "Launch setting only", "Same on both workers", "neutral"),
    }
    assert presentation.summarize(doc)["issue_count"] == presentation.summarize(serving_doc(2, "", None))["issue_count"]


def test_glm_ring_serving_rows_with_vllm_defaults():
    # glm53-flash-nvfp4-spark-tp4: --reasoning-parser glm45 --tool-call-parser glm47, no --save-cpu.
    doc = serving_doc(4, "glm45", "glm47", template={"enable_thinking": False})
    assert rendered(doc, SERVING_LABELS) == {
        "Reasoning parser": ("glm45", "glm45", "vLLM config", "Same on all 4 workers", "neutral"),
        "Tool-call parser": ("glm47", "Not checked at runtime", "Launch setting only", "API server only",
                             "neutral"),
        "Default chat template arguments": ('{"enable_thinking": false}', "Not checked at runtime",
                                            "Launch setting only", "API server only", "neutral"),
        "Shared-memory reader window": ("1 s (vLLM default)", "Not checked at runtime", "Launch setting only",
                                        "Not set on all 4 workers", "neutral"),
    }
    text = presentation.render_text(doc)
    html = presentation.render_report(doc)[1]
    note = presentation.GROUP_NOTES["Chat and tools"]
    assert "\nCHAT AND TOOLS\n" in text and "\n" + note + "\n" in text
    assert f'<p class="footnote">{note}</p>' in html
    assert '{&quot;enable_thinking&quot;: false}' in html
    # API-server-only rows have no per-worker list.
    assert '<span class="agreement neutral">API server only</span>' in html


def test_unselected_parsers_and_other_reader_windows_read_plainly():
    doc = serving_doc(2, "", None, environ={"SPARKRING_SHM_BUSY_LOOP_S": "0.05"})
    rows = rendered(doc, SERVING_LABELS)
    assert rows["Reasoning parser"][:2] == ("None", "None")
    assert rows["Tool-call parser"][0] == "None"
    assert rows["Shared-memory reader window"][0] == "0.05 s"
    doc = serving_doc(2, "qwen3", "qwen3_xml", environ={"SPARKRING_SHM_BUSY_LOOP_S": "2 ms"})
    assert rendered(doc, SERVING_LABELS)["Shared-memory reader window"][0] == "Invalid value"
    # Workers with different windows are a real difference.
    doc = serving_doc(2, "qwen3", "qwen3_xml", environ={"SPARKRING_SHM_BUSY_LOOP_S": "0.002"})
    environment = doc["workers"]["ranks"][1]["configured"]["environment"]
    environment["SPARKRING_SHM_BUSY_LOOP_S"] = dict(environment["SPARKRING_SHM_BUSY_LOOP_S"], value=0.05)
    row = settings_row(doc, "Shared-memory reader window")
    assert (row["agreement"], row["severity"]) == ("Workers report different values", "bad")
    assert row["rank_values"] == [("0", "2 ms (sparkring install --save-cpu)"), ("1", "0.05 s")]
    # A worker without the variable is flagged and listed with the default window.
    environment["SPARKRING_SHM_BUSY_LOOP_S"] = {
        "state": "unknown", "source": "process_environment", "reason": "not_set_in_environment"}
    row = settings_row(doc, "Shared-memory reader window")
    assert row["severity"] == "warn" and row["rank_values"][1] == ("1", "1 s (vLLM default)")


def test_sircl_is_the_collective_transport_and_the_bundle_is_the_images():
    doc = observed_fixture()
    fact = lambda value, source: {'state': 'known', 'value': value, 'source': source}
    for rank in doc['workers']['ranks']:
        rank['effective'].update(tp_collective_transport=fact('sircl', 'sircl_receipt'),
                                 tp_sircl_version=fact('0.3.2', 'sircl_layer_receipt'))
    doc['provenance']['b12x_comm_bundle'] = {
        'name': fact('tp2-rocenante-adaptive-prepared', 'installed_receipt'), 'manifest_sha256': 'e' * 64,
        'role': 'carried_by_image_not_the_active_transport'}
    view = presentation.summarize(doc)
    rows = {row['label']: row for _, group in view['groups'] for row in group}
    assert rows['Collective transport (TP)']['resolved'] == 'sircl'
    assert rows['SIRCL version (TP)']['resolved'] == '0.3.2'
    items = dict(presentation.identity_items(view))
    assert items['b12x communication bundle in the image'] == 'tp2-rocenante-adaptive-prepared'
    assert 'Transport profile' not in presentation.render_text(doc)
