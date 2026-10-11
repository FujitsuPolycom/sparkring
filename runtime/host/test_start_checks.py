"""Start checks: verdicts from a rank-0 log, the status document, metrics and the in-container probe."""
import json
import os
import pathlib
import subprocess
import sys

import pytest

from runtime.host import install_workflow, start_checks

ENVIRONMENT = {"VLLM_B12X_MOE_FP4_FORCE_A16": "1", "GLM53FULL_LATENT_SHARD": "1", "GLM53FULL_EH_PROJ_TP": "1"}
FLAGS = ("GLM53FULL_LATENT_SHARD", "GLM53FULL_EH_PROJ_TP")
# The lines glm53full_speedups' helpers log when rank 0 builds each split layer.
BUILT = {"GLM53FULL_LATENT_SHARD": "glm53full_speedups: model.layers.0.self_attn.fused_qkv_a_proj is column-parallel "
                                   "over TP8: rank 0 computes fused rows [0, 328) of 2624",
         "GLM53FULL_EH_PROJ_TP": "glm53full_speedups: model.layers.78.eh_proj is RowParallelLinear(12288 -> 6144) over "
                                 "TP8: 1536 input columns per rank"}
# Rank 0's glm53full_speedups, glm_dsa_indexer_split and glm_dcp_decode_comm lines and its engine configuration
# from the first glm53-nvfp4-csf-tp8 start (image d52737a109e0), copied verbatim from its container log.
CSF_TP8_LOG = pathlib.Path(__file__).parent / "fixtures" / "start_checks" / "glm53-nvfp4-csf-tp8-rank0.log"


def log(weights=51.08, layers=75, mode="W4A16 decode, A4 prefill", patched=FLAGS, built=None, extra=""):
    lines = [f"(Worker pid=1) INFO glm53full_speedups: vllm.models.x.F patched for {flag}" for flag in patched]
    lines += [f"(Worker_TP0_DCP0 pid=1) INFO {BUILT[flag]}" for flag in (patched if built is None else built)]
    lines += [f"(Worker pid=1) INFO NVFP4-CSF layer {layer} rank 0/8: native NVFP4 {mode}, shared scale scratch "
              f"123 bytes" for layer in range(3, 3 + layers)]
    if weights is not None:
        lines.append(f"(Worker pid=1) INFO Model loading took {weights} GiB memory and 301.2 seconds")
    return "\n".join(lines) + "\n" + extra


def status(head=True, draft="nvfp4"):
    effective = {"target_head_mxfp8": {"state": "known", "source": "resident_lm_head", "value": head},
                 "draft_head_quantization": {"state": "known", "source": "resident_lm_head", "value": draft}}
    return {"workers": {"ranks": [{"identity": {}, "effective": effective}]}}


METRICS = ('vllm:spec_decode_num_draft_tokens_total{engine="0"} 400.0\n'
           'vllm:spec_decode_num_accepted_tokens_total{engine="0"} 270.0\n')
FACTS = {"scale_prefetch": True, "barrier_readers": ["b12x/moe/_shared/kernels/w4a16/kernel.py"], "errors": []}


def verdicts(document):
    return {name: item["verdict"] for name, item in document["checks"].items()}


def test_every_setting_took_effect():
    document = start_checks.judge(log(), status(), METRICS, FACTS, ENVIRONMENT)
    assert document["schema"] == start_checks.SCHEMA and document["weights_gib"] == 51.08
    assert verdicts(document) == dict.fromkeys(start_checks.CHECKS, start_checks.TOOK)
    assert "draft tokens accepted 270.0 of 400.0" in document["checks"]["draft-quantization"]["evidence"]


@pytest.mark.parametrize("change, check, verdict", [
    ({"log": log(weights=55.3)}, "mxfp8-linears", "did not take effect"),
    ({"status": status(head=False)}, "mxfp8-linears", "did not take effect"),
    ({"log": log(weights=52.8)}, "draft-quantization", "did not take effect"),
    ({"log": log(weights=52.8)}, "mxfp8-linears", "took effect"),
    ({"metrics": ""}, "draft-quantization", "unknown"),
    ({"log": log(patched=("GLM53FULL_LATENT_SHARD",))}, "column-split-loading", "unknown"),
    ({"log": log(patched=())}, "column-split-loading", "did not take effect"),
    ({"log": log(extra="PatchRefused: glm53full_speedups: ...\n")}, "column-split-loading", "did not take effect"),
    ({"log": log(patched=(), extra="INFO glm53full_speedups: GLM53FULL_LATENT_SHARD, GLM53FULL_EH_PROJ_TP are 0; "
                                   "nothing patched\n")}, "column-split-loading", "did not take effect"),
    ({"log": log(built=(), extra="INFO glm53full_speedups: model.layers.0.self_attn.fused_qkv_a_proj keeps the "
                                 "image's replicated fused_qkv_a_proj: tensor-parallel size 4\n")},
     "column-split-loading", "did not take effect"),
    ({"log": log(built=("GLM53FULL_LATENT_SHARD",))}, "column-split-loading", "unknown"),
    ({"log": log(extra="INFO glm_dcp_decode_comm: no GLM_DCP_DECODE_* item flag is 1; nothing patched\n"
                       "glm_dsa_indexer_split.PatchRefused: glm_dsa_indexer_split: refusing to patch\n")},
     "column-split-loading", "took effect"),
    ({"log": log(mode="A16")}, "csf-w4a16-a4-prefill", "did not take effect"),
    ({"log": log(layers=0)}, "csf-w4a16-a4-prefill", "unknown"),
    ({"facts": {**FACTS, "scale_prefetch": False}}, "csf-scale-prefetch", "did not take effect"),
    ({"facts": None}, "csf-scale-prefetch", "unknown"),
    ({"facts": {**FACTS, "barrier_readers": ["b12x/attention/other.py"]}}, "w4a16-host-barrier-reset",
     "did not take effect"),
    ({"facts": {**FACTS, "barrier_readers": None}}, "w4a16-host-barrier-reset", "unknown"),
])
def test_each_verdict_follows_its_evidence(change, check, verdict):
    sources = {"log": log(), "status": status(), "metrics": METRICS, "facts": FACTS, **change}
    document = start_checks.judge(sources["log"], sources["status"], sources["metrics"], sources["facts"], ENVIRONMENT)
    assert document["checks"][check]["verdict"] == verdict


def test_the_csf_tp8_start_log_shows_both_splits_built_despite_another_plugins_nothing_patched_line():
    text = CSF_TP8_LOG.read_text(encoding="utf-8")
    assert "glm_dcp_decode_comm: no GLM_DCP_DECODE_* item flag is 1; nothing patched" in text
    check = start_checks.judge(text, status(), METRICS, FACTS, ENVIRONMENT)["checks"]["column-split-loading"]
    assert check["verdict"] == start_checks.TOOK
    assert check["evidence"] == ["patched for: GLM53FULL_EH_PROJ_TP, GLM53FULL_LATENT_SHARD",
                                 "split layers built for: GLM53FULL_EH_PROJ_TP, GLM53FULL_LATENT_SHARD"]


def test_unreadable_sources_give_unknown_and_never_raise():
    document = start_checks.judge(None, None, None, None, None)
    assert set(verdicts(document).values()) <= {start_checks.UNKNOWN, start_checks.NOT}
    assert document["weights_gib"] is None
    assert start_checks.judge("", {"workers": "x"}, "garbage", [], {}, checks=("mxfp8-linears",))["checks"].keys() == {
        "mxfp8-linears"}


def test_the_probe_reads_the_images_prefetch_default_and_barrier_readers_without_importing_vllm(tmp_path):
    root = tmp_path / "site"
    (root / "vllm").mkdir(parents=True)
    (root / "vllm/__init__.py").write_text("raise RuntimeError('vllm itself must not be imported')\n")
    (root / "vllm/envs.py").write_text("import os\nenvironment_variables = {'VLLM_B12X_CSF_SCALE_PREFETCH': "
                                       "lambda: bool(int(os.getenv('VLLM_B12X_CSF_SCALE_PREFETCH', '1')))}\n")
    kernels = root / "b12x/moe/_shared/kernels/w4a16"
    kernels.mkdir(parents=True)
    (root / "b12x/__init__.py").write_text("")
    (kernels / "kernel.py").write_text("import os\nRESET = os.environ.get('B12X_W4A16_SMALL_M_HOST_BARRIER_RESET')\n")
    environment = {**os.environ, "PYTHONPATH": str(root)}
    environment.pop("VLLM_B12X_CSF_SCALE_PREFETCH", None)
    output = subprocess.run([sys.executable, "-B", "-"], input=start_checks.probe_source(), text=True,
                            capture_output=True, check=True, env=environment).stdout
    facts = start_checks.parse_probe(output.strip().splitlines()[-1])
    assert facts == {"scale_prefetch": True, "barrier_readers": ["b12x/moe/_shared/kernels/w4a16/kernel.py"],
                     "errors": []}
    environment["VLLM_B12X_CSF_SCALE_PREFETCH"] = "0"
    output = subprocess.run([sys.executable, "-B", "-"], input=start_checks.probe_source(), text=True,
                            capture_output=True, check=True, env=environment).stdout
    assert start_checks.parse_probe(output.strip().splitlines()[-1])["scale_prefetch"] is False


def test_the_install_result_reads_the_smoke_actions_start_checks(tmp_path):
    document = start_checks.judge(log(), status(), METRICS, FACTS, ENVIRONMENT)
    (tmp_path / "state.json").write_text(json.dumps({"receipt": "receipt.json"}))
    (tmp_path / "receipt.json").write_text(json.dumps({"actions": {
        "smoke:root@192.0.2.1": {"state": "succeeded", "result": {"stdout": json.dumps(
            {"ok": True, "start_checks": document})}}}}))
    assert install_workflow.start_check_result(tmp_path) == document
    lines = install_workflow.summary_lines({"example_request": "curl", "start_checks": document})
    assert "  Start checks:" in lines and any(line.startswith("    mxfp8-linears: took effect") for line in lines)
    (tmp_path / "receipt.json").write_text(json.dumps({"actions": {}}))
    assert install_workflow.start_check_result(tmp_path) is None
