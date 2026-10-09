"""Offline admission checks for the isolated Qwen Flash-Next TP2 plan."""

import copy
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest
from runtime.common.qwen_flash_next import ROOT, render, publication
from runtime.common import qwen_flash_next as adapter
from runtime.common import installer, native_candidate


def shared_publication():
    return native_candidate.publication("shared-2026.09.3")


def installer_image_id():
    # The installer profile runs on the image named by the installer image lock.
    from runtime.common import installer_image
    return adapter.read(installer_image.DEFAULT_LOCK)["image_id"]

PROFILE = (
    Path(__file__).resolve().parents[2] / "profiles/qwen38-flash-next-tp2/config.json"
)


def plan(rank=0, **extra):
    return render(
        json.loads(PROFILE.read_text()),
        rank=rank,
        master="192.0.2.1",
        host_ip=f"192.0.2.{rank + 1}",
        interface="test0",
        image=installer_image_id(),
        model=str(ROOT / "fixture-model"),
        cache=str(ROOT / "fixture-cache"),
        **extra,
    )


def test_checkpoint_table_names_the_default_and_its_alternatives():
    profile = adapter.read(PROFILE)
    assert adapter.checkpoint_names(profile) == ("qad-step5500-ple1000", (
        "jmni-qad5500-hybrid", "qad-step-4000", "qad-step5500-mxfp8-attention", "qad-step5500-ple1000"))
    assert adapter.checkpoint_settings(profile, None) is profile
    assert adapter.checkpoint_settings(profile, "qad-step5500-ple1000") is profile
    # qad-step-5500 spells step 5500 like the step-4000 branch.
    assert adapter.checkpoint_name(profile, "qad-step-5500") == "qad-step5500-ple1000"
    assert adapter.checkpoint_settings(profile, "qad-step-5500") is profile
    tp4 = adapter.read(ROOT / "profiles/qwen38-flash-next-qad-tp4/config.json")
    # Both profiles list the same checkpoints; served names end with each profile's -TP<nodes>.
    assert tp4["checkpoint"] == profile["checkpoint"] and json.loads(json.dumps(tp4["checkpoints"]).replace(
        "-TP4", "-TP2")) == profile["checkpoints"]
    assert tp4["checkpoint_aliases"] == profile["checkpoint_aliases"]
    for aliases in ({"qad-step-5500": "main"}, {"qad-step-4000": "qad-step5500-ple1000"}, {"Step 5500": "qad-step-4000"}):
        with pytest.raises(ValueError, match="new name for a listed checkpoint"):
            adapter.checkpoint_names({**profile, "checkpoint_aliases": aliases})
    with pytest.raises(ValueError, match="lists: jmni-qad5500-hybrid, qad-step-4000, qad-step5500-mxfp8-attention, "
                                         "qad-step5500-ple1000"):
        adapter.checkpoint_settings(profile, "main")
    added = copy.deepcopy(profile)
    added["checkpoints"]["qad-step-4000"]["environment"]["VLLM_NEW_SETTING"] = "1"
    with pytest.raises(ValueError, match="existing environment"):
        adapter.checkpoint_settings(added, "qad-step-4000")
    # vLLM refuses an unknown --speculative-config key at startup, so an entry may add keys.
    added = copy.deepcopy(profile)
    added["checkpoints"]["qad-step-4000"]["speculative"]["draft_tensor_parallel_size"] = 2
    draft = json.loads(adapter.checkpoint_settings(added, "qad-step-4000")["vllm_args"][
        profile["vllm_args"].index("--speculative-config") + 1])
    assert (draft["draft_tensor_parallel_size"], draft["moe_backend"]) == (2, "b12x")
    unspeculative = copy.deepcopy(profile)
    at = unspeculative["vllm_args"].index("--speculative-config")
    del unspeculative["vllm_args"][at:at + 2]
    with pytest.raises(ValueError, match="without --speculative-config"):
        adapter.checkpoint_settings(unspeculative, "qad-step-4000")
    changed = copy.deepcopy(profile)
    changed["checkpoints"]["qad-step5500-ple1000"]["environment"] = {"VLLM_MXFP8_LM_HEAD": "1"}
    with pytest.raises(ValueError, match="default checkpoint"):
        adapter.checkpoint_names(changed)
    duplicate = copy.deepcopy(profile)
    duplicate["checkpoints"]["step-5500-copy"] = copy.deepcopy(duplicate["checkpoints"]["qad-step5500-ple1000"])
    with pytest.raises(ValueError, match="another revision"):
        adapter.checkpoint_names(duplicate)
    for field, value in (("revision", "qad-step-4000"), ("config_sha256", "0" * 63), ("repository", None)):
        malformed = copy.deepcopy(profile)
        malformed["checkpoints"]["qad-step-4000"]["model"][field] = value
        with pytest.raises(ValueError, match="Invalid checkpoint entry: qad-step-4000"):
            adapter.checkpoint_names(malformed)


def test_a_checkpoint_changes_only_values_of_options_the_profile_sets():
    profile = adapter.read(PROFILE)
    for arguments in ({"--hf-overrides": "{}"}, {"--enable-prefix-caching": "0"},
                      {"--speculative-config": "{}"}, {"--quantization": ""}, {"--quantization": "--x"},
                      {"--quantization": 4}, ["--quantization", "modelopt"]):
        changed = copy.deepcopy(profile)
        changed["checkpoints"]["qad-step-4000"]["arguments"] = arguments
        with pytest.raises(ValueError, match="qad-step-4000"):
            adapter.checkpoint_settings(changed, "qad-step-4000")
    changed = copy.deepcopy(profile)
    changed["checkpoints"]["qad-step-4000"]["arguments"] = {"--quantization": "modelopt_fp4"}
    args = adapter.checkpoint_settings(changed, "qad-step-4000")["vllm_args"]
    assert len(args) == len(profile["vllm_args"])
    assert [(a, b) for a, b in zip(args, profile["vllm_args"]) if a != b] == [
        ("modelopt_fp4", "modelopt_mixed"), (args[args.index("--speculative-config") + 1],
                                             profile["vllm_args"][args.index("--speculative-config") + 1])]
    for served in ("Qwen3.8-Flash-Next-NVFP4-QAD", "Qwen3.8-Flash-Next-NVFP4-QAD-TP4", "../TP2", 7):
        changed["checkpoints"]["qad-step-4000"]["served_model_name"] = served
        with pytest.raises(ValueError, match="ending -TP2"):
            adapter.checkpoint_settings(changed, "qad-step-4000")


def test_the_step_4000_checkpoint_runs_its_pinned_settings():
    command = plan(checkpoint="qad-step-4000")
    # Revision 629bc3218833 stores NVFP4 MTP routed experts, which B12X runs,
    # and quantizes its target LM head to MXFP8 at load.
    assert "VLLM_MXFP8_LM_HEAD=1" in command
    draft = json.loads(command[command.index("--speculative-config") + 1])
    assert draft["moe_backend"] == "b12x" and draft["draft_sample_method"] == "probabilistic"
    namespace = f"qwen-flash-next-{installer_image_id()[7:19]}-629bc3218833"
    assert f"VLLM_CACHE_ROOT=/cache/{namespace}/vllm" in command
    default = plan()
    assert command[command.index("--served-model-name") + 1] == default[default.index("--served-model-name") + 1]
    with pytest.raises(ValueError, match="lists"):
        plan(checkpoint="main")


def test_the_jmni_hybrid_runs_its_nvfp4_draft_experts_on_b12x():
    """The third-party hybrid keeps step 5500's LM head and stores the MTP experts as W4A16 NVFP4, as step 4000 does."""
    command, default = plan(checkpoint="jmni-qad5500-hybrid"), plan()
    draft = json.loads(command[command.index("--speculative-config") + 1])
    assert draft == {**json.loads(default[default.index("--speculative-config") + 1]), "moe_backend": "b12x"}
    assert "VLLM_MXFP8_LM_HEAD=0" in command
    assert command[command.index("--served-model-name") + 1] == "Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid-TP2"
    assert f"VLLM_CACHE_ROOT=/cache/qwen-flash-next-{installer_image_id()[7:19]}-87c8f2fb738b/vllm" in command
    pins = installer.checkpoint_pins(installer.setup.selection("qwen38-flash-next-tp2", "jmni-qad5500-hybrid"))
    assert (pins["repository"], len(pins["weights"]), len(set(pins["files"]) - set(pins["optional"]))) == (
        "JMNI-Labs/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid", 40, 56)
    # The shard holding its MXFP8 attention projections is step 4000's model-00035-of-00036.safetensors.
    step4000 = installer.checkpoint_pins(installer.setup.selection("qwen38-flash-next-tp2", "qad-step-4000"))
    assert (pins["files"]["hybrid-main-00002.safetensors"]["sha256"]
            == step4000["files"]["model-00035-of-00036.safetensors"]["sha256"])


GLM_TP4 = ROOT / "profiles/glm53-flash-nvfp4-spark-tp4/config.json"
GLM_TP2 = ROOT / "profiles/glm53-flash-nvfp4-spark-tp2/config.json"


def glm_command(checkpoint=None, rank=0, path=GLM_TP4):
    spec = adapter.container_spec(adapter.read(path), rank=rank, master="192.0.2.1", host_ip=f"192.0.2.{rank + 1}",
                                  interface="test0", image=installer_image_id(), model=str(ROOT / "fixture-model"),
                                  cache=str(ROOT / "fixture-cache"), checkpoint=checkpoint)
    return spec, list(spec.command)


def test_the_glm_ring_lists_four_checkpoints_of_four_repositories():
    profile = adapter.read(GLM_TP4)
    assert adapter.checkpoint_names(profile) == ("nvfp4-spark", ("csf", "nvfp4-qad", "nvfp4-spark", "nvidia-nvfp4"))
    assert adapter.checkpoint_settings(profile, "nvfp4-spark") is profile
    assert adapter.preferred_checkpoint(profile) == "csf"
    repositories = {name: entry["model"]["repository"] for name, entry in profile["checkpoints"].items()}
    assert repositories == {"nvfp4-spark": "local-inference-lab/GLM-5.3-Flash-NVFP4-Spark",
                            "nvfp4-qad": "local-inference-lab/GLM-5.3-Flash-NVFP4",
                            "nvidia-nvfp4": "nvidia/GLM-5.3-Flash-NVFP4",
                            "csf": "local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD"}
    # The QAD checkpoint is the target record's; NVIDIA's revision da920bb0
    # holds the weights and index of the record's 423acf37 and a config that
    # excludes the BF16 MTP layer from quantization.
    from runtime.common import glm_targets
    qad, nvidia = glm_targets.target("nvfp4-qad"), glm_targets.target("nvidia-nvfp4")
    assert profile["checkpoints"]["nvfp4-qad"]["model"] == {key: qad[key] for key in adapter.MODEL_KEYS}
    selected = profile["checkpoints"]["nvidia-nvfp4"]["model"]
    assert (selected["repository"], selected["index_sha256"]) == (nvidia["repository"], nvidia["index_sha256"])
    assert selected["revision"] != nvidia["revision"] and selected["config_sha256"] != nvidia["config_sha256"]


def test_a_glm_checkpoint_with_larger_weights_reserves_that_much_less_kv_cache():
    """Each Spark keeps the free memory the profile's measurements left it.

    A checkpoint whose indexed weights are larger than the default's takes a KV
    cache smaller than the profile's by that difference per Spark, rounded up
    to a whole GiB (performance/records/glm53-flash/installer-memory-20260929.md
    measured Node A's memory with the default checkpoint).
    """
    from runtime.common import installer, setup
    gib = 2**30

    def weights(profile_id, name):
        pins = installer.checkpoint_pins(setup.selection(profile_id, name))
        return sum(pins["files"][file]["size"] for file in pins["weights"])

    def kv(configuration):
        return int(configuration["vllm_args"][configuration["vllm_args"].index("--kv-cache-memory-bytes") + 1])

    extra = {}
    for path, name in ((GLM_TP4, "nvfp4-qad"), (GLM_TP4, "nvidia-nvfp4"), (GLM_TP2, "nvfp4-qad")):
        profile, profile_id = adapter.read(path), path.parent.name
        per_spark = (weights(profile_id, name) - weights(profile_id, "nvfp4-spark")) / adapter.node_count(profile)
        reserved = kv(profile) - kv(adapter.checkpoint_settings(profile, name))
        assert reserved % gib == 0 and 0 <= reserved - per_spark < gib, (profile_id, name)
        extra[profile_id, name] = round(per_spark / gib, 1)
    assert extra == {("glm53-flash-nvfp4-spark-tp4", "nvfp4-qad"): 2.4,
                     ("glm53-flash-nvfp4-spark-tp4", "nvidia-nvfp4"): 3.9,
                     ("glm53-flash-nvfp4-spark-tp2", "nvfp4-qad"): 4.9}


def test_the_glm_pair_serves_the_qad_checkpoint_with_a_smaller_kv_cache_and_context():
    profile = adapter.read(GLM_TP2)
    assert adapter.checkpoint_names(profile) == ("nvfp4-spark", ("csf", "nvfp4-qad", "nvfp4-spark"))
    assert profile["checkpoints"]["nvfp4-qad"]["model"] == adapter.read(GLM_TP4)["checkpoints"]["nvfp4-qad"]["model"]
    # NVIDIA's weights take 7.8 GiB more on each Spark of a pair, which the
    # pair's 10 GiB of KV cache cannot give up; the pair does not list it.
    with pytest.raises(ValueError, match="lists: csf, nvfp4-qad, nvfp4-spark"):
        adapter.checkpoint_settings(profile, "nvidia-nvfp4")
    for rank in (0, 1):
        _, default = glm_command(rank=rank, path=GLM_TP2)
        spec, command = glm_command("nvfp4-qad", rank=rank, path=GLM_TP2)
        assert command[command.index("--served-model-name") + 1] == "GLM-5.3-Flash-NVFP4-QAD-TP2"
        # 5 GiB, the pair's 10 GiB less the 4.9 GiB the QAD weights add on each
        # Spark. With the pair's measured 137,000 to 153,000 tokens per GiB in
        # 2,048-token pages, it holds 0.68 to 0.77 million tokens, more than
        # one request of the 524,288-token context window.
        assert command[command.index("--kv-cache-memory-bytes") + 1] == str(5 * 2**30)
        assert command[command.index("--max-model-len") + 1] == "524288"
        # The pair's draft already runs its MTP experts on Humming, which the QAD experts' MXFP8 needs.
        assert json.loads(command[command.index("--speculative-config") + 1])["moe_backend"] == "humming"
        assert (command[command.index("--quantization") + 1], command[command.index("--load-format") + 1]) == (
            "modelopt_mixed", "b12x")
        assert [index for index, (a, b) in enumerate(zip(command, default)) if a != b] == [
            command.index(flag) + 1 for flag in ("--served-model-name", "--max-model-len", "--kv-cache-memory-bytes")]
        assert len(command) == len(default) and command[-1] == default[-1]
        assert (command[-1] == "--headless") == (rank == 1)
        assert spec.name == f"glm53-flash-nvfp4-spark-tp2-r{rank}"
        assert spec.environment["XDG_CACHE_HOME"] == (
            f"/cache/glm53-flash-nvfp4-spark-{installer_image_id()[7:19]}-175ae8ce3b5a")


def test_the_glm_qad_checkpoint_runs_its_mxfp8_draft_experts_on_humming():
    _, default = glm_command()
    spec, command = glm_command("nvfp4-qad")
    assert command[command.index("--served-model-name") + 1] == "GLM-5.3-Flash-NVFP4-QAD-TP4"
    # NVFP4 routed experts load through the B12X loader and run on B12X, as with NVFP4-Spark.
    assert (command[command.index("--quantization") + 1], command[command.index("--load-format") + 1]) == (
        "modelopt_mixed", "b12x")
    assert spec.environment["VLLM_PLUGINS"] == "b12x_loader"
    assert command[command.index("--moe-backend") + 1] == "b12x"
    # The MTP layer's routed experts are MXFP8, which B12X does not implement.
    draft = json.loads(command[command.index("--speculative-config") + 1])
    assert draft == {**json.loads(default[default.index("--speculative-config") + 1]), "moe_backend": "humming"}
    assert command[command.index("--kv-cache-memory-bytes") + 1] == str(37 * 2**30)
    assert [index for index, (a, b) in enumerate(zip(command, default)) if a != b] == [
        command.index(flag) + 1 for flag in ("--served-model-name", "--kv-cache-memory-bytes", "--speculative-config")]
    assert spec.environment["XDG_CACHE_HOME"] == f"/cache/glm53-flash-nvfp4-spark-{installer_image_id()[7:19]}-175ae8ce3b5a"


def test_the_nvidia_checkpoint_runs_modelopt_nvfp4_with_the_safetensors_loader():
    _, default = glm_command()
    spec, command = glm_command("nvidia-nvfp4")
    assert command[command.index("--served-model-name") + 1] == "GLM-5.3-Flash-NVFP4-NVIDIA-TP4"
    # ModelOpt NVFP4, the method the Swift profiles use on this image; the
    # checkpoint's own config excludes its BF16 MTP layer from quantization.
    assert command[command.index("--quantization") + 1] == "modelopt_fp4"
    assert command[command.index("--load-format") + 1] == "safetensors"
    assert "--hf-overrides" not in command
    # The BF16 draft experts run on vLLM's unquantized MoE kernel: no draft MoE backend is named.
    assert "moe_backend" not in json.loads(command[command.index("--speculative-config") + 1])
    assert command[command.index("--kv-cache-memory-bytes") + 1] == str(36 * 2**30)
    assert [index for index, (a, b) in enumerate(zip(command, default)) if a != b] == [
        command.index(flag) + 1 for flag in ("--served-model-name", "--kv-cache-memory-bytes", "--quantization",
                                             "--load-format")]
    assert spec.environment["B12X_COMPILE_CACHE_DIR"] == "/cache/glm53-flash-nvfp4-spark-cuda13.4.2-da920bb0b9f4/b12x"
    worker, _ = glm_command("nvidia-nvfp4", rank=3)
    assert worker.command[-1] == "--headless" and worker.name == "glm53-flash-nvfp4-spark-tp4-r3"


CSF_MODEL = {"repository": "local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD",
             "revision": "dec48abd33efa73c3bb7c95b74eee10cad34f9be",
             "config_sha256": "d9d0b32d0fa38d0cfc7ac162670db17fe16fe02404e847e6b2aa07d71efd67f1",
             "index_sha256": "568c770e4a083ab53c3d74deab348d541724f23ec9637d83513ecb8fa4a7ef73"}
GLM_TP8 = ROOT / "profiles/glm53-flash-csf-tp8/config.json"
CACHE_VARIABLES = {"XDG_CACHE_HOME", "VLLM_CACHE_ROOT", "TRITON_CACHE_DIR", "B12X_COMPILE_CACHE_DIR",
                   "CUTE_DSL_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"}


@pytest.mark.parametrize("path, nodes, kv_gib", [(GLM_TP4, 4, 37), (GLM_TP2, 2, 10)])
def test_the_glm_csf_checkpoint_runs_the_eight_spark_profiles_quantization_loader_and_draft(path, nodes, kv_gib):
    """The CSF entries pin the eight-Spark CSF profile's checkpoint and change only what that checkpoint needs."""
    profile = adapter.read(path)
    eight = adapter.read(GLM_TP8)
    assert profile["checkpoints"]["csf"]["model"] == CSF_MODEL == eight["model"]
    for rank in range(nodes):
        default_spec, default = glm_command(rank=rank, path=path)
        spec, command = glm_command("csf", rank=rank, path=path)
        assert command[command.index("--served-model-name") + 1] == f"GLM-5.3-Flash-CSF-TP{nodes}"
        for flag in ("--quantization", "--load-format"):
            assert command[command.index(flag) + 1] == eight["vllm_args"][eight["vllm_args"].index(flag) + 1] == (
                "nvfp4_csf")
        # W4A16 decode, as the eight-Spark profile sets it; every other variable but the
        # revision-keyed cache paths is the profile's own.
        assert spec.environment["VLLM_B12X_MOE_FP4_FORCE_A16"] == eight["environment"]["VLLM_B12X_MOE_FP4_FORCE_A16"]
        assert {key for key, value in spec.environment.items() if default_spec.environment.get(key) != value} == {
            "VLLM_B12X_MOE_FP4_FORCE_A16", *CACHE_VARIABLES}
        assert set(spec.environment) == set(default_spec.environment)
        # The draft's experts run on Marlin, as in the eight-Spark profile.
        draft = json.loads(command[command.index("--speculative-config") + 1])
        assert draft == {**json.loads(default[default.index("--speculative-config") + 1]), "moe_backend": "marlin"}
        assert command[command.index("--kv-cache-memory-bytes") + 1] == str(kv_gib * 2**30)
        assert command[command.index("--max-model-len") + 1] == "1048576"
        changed = ["--served-model-name", "--speculative-config", "--quantization", "--load-format"]
        if nodes == 4:
            changed.append("--kv-cache-memory-bytes")
        assert [index for index, (a, b) in enumerate(zip(command, default)) if a != b] == sorted(
            command.index(flag) + 1 for flag in changed)
        assert len(command) == len(default) and (command[-1] == "--headless") == (rank != 0)
        assert spec.name == f"glm53-flash-nvfp4-spark-tp{nodes}-r{rank}"
        assert spec.environment["XDG_CACHE_HOME"] == (
            f"/cache/glm53-flash-nvfp4-spark-{installer_image_id()[7:19]}-dec48abd33ef")
    if nodes == 4:
        # The ring's entry keeps the 37 GiB of the QAD entry and of the eight-Spark profile.
        assert str(kv_gib * 2**30) == eight["vllm_args"][eight["vllm_args"].index("--kv-cache-memory-bytes") + 1] == (
            profile["checkpoints"]["nvfp4-qad"]["arguments"]["--kv-cache-memory-bytes"])


def test_the_csf_checkpoint_keeps_no_larger_kv_cache_than_nvfp4_spark_for_smaller_weights():
    """The CSF weights take less memory on each Spark than NVFP4-Spark's, whose memory the profiles measured."""
    from runtime.common import installer, setup

    def weights(profile_id, name):
        pins = installer.checkpoint_pins(setup.selection(profile_id, name))
        return sum(pins["files"][file]["size"] for file in pins["weights"])

    for path in (GLM_TP4, GLM_TP2):
        profile, profile_id = adapter.read(path), path.parent.name
        per_spark = (weights(profile_id, "nvfp4-spark") - weights(profile_id, "csf")) / adapter.node_count(profile)
        assert round(per_spark / 2**30, 1) == {4: 2.3, 2: 4.6}[adapter.node_count(profile)]
        args, csf = profile["vllm_args"], adapter.checkpoint_settings(profile, "csf")["vllm_args"]
        assert int(csf[csf.index("--kv-cache-memory-bytes") + 1]) <= int(args[args.index("--kv-cache-memory-bytes") + 1])


def test_a_preferred_checkpoint_is_another_listed_published_checkpoint():
    profile = adapter.read(GLM_TP2)
    assert adapter.preferred_checkpoint(profile) == "csf"
    assert adapter.preferred_checkpoint(adapter.read(PROFILE)) is None
    # The preferred checkpoint changes no setting of the profile's own command.
    assert adapter.checkpoint_settings(profile, None) is profile
    for preferred in ("nvfp4-spark", "missing", 1):
        with pytest.raises(ValueError, match="preferred checkpoint must name another listed"):
            adapter.checkpoint_names({**profile, "preferred_checkpoint": preferred})
    qwen = adapter.read(PROFILE)
    with pytest.raises(ValueError, match="preferred checkpoint must name another listed"):
        adapter.checkpoint_names({**qwen, "preferred_checkpoint": "qad-step5500-mxfp8-attention"})
    without = {key: value for key, value in profile.items() if key not in ("checkpoint", "checkpoints")}
    with pytest.raises(ValueError, match="requires a checkpoints table"):
        adapter.checkpoint_names(without)


def test_qwen_model_and_native_prefix_only():
    command = plan()
    assert (
        command[command.index("--served-model-name") + 1] == "Qwen3.8-Flash-Next-NVFP4-QAD-TP2"
    )
    assert "--kv-transfer-config" not in command
    assert "--device-ids" not in command
    assert "VLLM_SSM_CONV_STATE_LAYOUT=DS" in command
    assert "VLLM_USE_V2_MODEL_RUNNER=1" in command
    assert not any(value.startswith("VLLM_PLE_TABLE_MEMORY=") for value in command)
    assert "VLLM_PLE_CPU_OFFLOAD=0" in command
    # The step-5500 checkpoint serves with its BF16 target LM head.
    assert "VLLM_MXFP8_LM_HEAD=0" in command
    # The installer image registers this setting and quantizes the BF16
    # hyper-connection down/injection projections to MXFP8 at load.
    assert "VLLM_QWEN4_EXP_MXFP8_HC=1" in command
    # Every speculative decode all-reduce (up to 16 sequences x 4 rows x 2560
    # BF16 values) runs on RoCEnante instead of the feature default of 4 rows.
    assert "QWEN_DISPATCH_AR_BYTES=327680" in command
    # Drafts sample from the draft distribution, so sampled requests accept
    # by distribution overlap rather than by the target's probability of one token.
    assert json.loads(command[command.index("--speculative-config") + 1])["draft_sample_method"] == "probabilistic"
    # The qad-step5500-ple1000 MTP experts are MXFP8, which the B12X MoE
    # backend does not implement; the draft uses the MXFP8-capable backend.
    assert (
        json.loads(command[command.index("--speculative-config") + 1])["moe_backend"]
        == "humming"
    )
    assert command[command.index("--decode-context-parallel-size") + 1] == "1"


def test_direct_loader_avoids_fastsafetensors_staging():
    command = plan()
    assert command[command.index("--load-format") + 1] == "b12x"
    assert "VLLM_PLUGINS=b12x_loader" in command
    assert not any(value.startswith("SAFETENSORS_FAST_GPU=") for value in command)


def test_tp2_qad_pins_manifest_and_cache_identity_are_consistent():
    root = PROFILE.parent
    plain = adapter.read(root / 'config.json')
    cached = adapter.read(root / 'sparkcache.json')
    tp4 = adapter.read(ROOT / 'profiles/qwen38-flash-next-qad-tp4/config.json')
    # The installer profiles follow checkpoint branch qad-step5500-ple1000; the
    # qualified SparkCache profiles keep the release-qualified revision.
    tp4_cached = adapter.read(ROOT / 'profiles/qwen38-flash-next-qad-tp4/sparkcache.json')
    assert plain['model'] == tp4['model']
    assert cached['model'] == tp4_cached['model']
    assert cached['model']['revision'] == '629bc3218833a38b475b719f34aa571666f4a03e'
    assert plain['model']['revision'] == '60215d26cf5e42c2db6128774032d57fc62678da'
    assert plain['served_model_name'] == cached['served_model_name'] == 'Qwen3.8-Flash-Next-NVFP4-QAD-TP2'
    assert tp4['served_model_name'] == 'Qwen3.8-Flash-Next-NVFP4-QAD-TP4'
    assert tp4_cached['served_model_name'] == tp4['served_model_name']
    manifest = (root / 'SHA256SUMS').read_text().splitlines()
    hashes = {line.split(maxsplit=1)[1].strip(): line.split()[0] for line in manifest}
    assert hashes['config.json'] == plain['model']['config_sha256']
    assert hashes['model.safetensors.index.json'] == plain['model']['index_sha256']
    assert len([name for name in hashes if name.endswith('-of-00041.safetensors')]) == 41
    # The SparkCache profiles keep their own sums for revision 629bc3218833.
    cached_sums = (ROOT / 'profiles/qwen38-flash-next-tp2-sparkcache/SHA256SUMS').read_text().splitlines()
    cached_hashes = {line.split(maxsplit=1)[1].strip(): line.split()[0] for line in cached_sums}
    assert cached_hashes['config.json'] == cached['model']['config_sha256']
    assert len([name for name in cached_hashes if name.endswith('-of-00036.safetensors')]) == 36
    assert (root / 'SHA256SUMS').read_bytes() == (ROOT / 'profiles/qwen38-flash-next-qad-tp4/SHA256SUMS').read_bytes()
    args = cached['vllm_args']
    config = json.loads(args[args.index('--kv-transfer-config') + 1])
    extra = config['kv_connector_extra_config']
    assert extra['spark_cache_root'] == '/cache/persistent/qwen38-flash-next-qad-tp2-shared-2026093'
    assert extra['spark_cache_target_checkpoint_sha256'] == extra['spark_cache_draft_checkpoint_sha256'] == '036c2f7994466d32514130f0417ccd117705ca01e2038c1ce5745f84813829ba'
    assert extra['spark_cache_target_checkpoint_sha256'] != 'ada04299f0b223ab6e55ff16edaf88db094d3f09b7fd31fc9f46aa9d8d7a2c47'
    assert config['kv_connector'] == 'SparkContextCacheConnector'
    assert args[args.index('--recurrent-checkpoint-policy') + 1] == 'aligned'


def test_reciprocal_selected_hca_positions():
    for rank in (0, 1):
        command = plan(rank)
        assert f"B12X_ROCE_PEER_HCA_MAP={1 - rank}=0/1" in command
        assert "B12X_ROCE_HCA=rocep1s0f0,roceP2p1s0f0" in command
        assert ("--headless" in command) == bool(rank)


def test_invalid_rank_rejected():
    with pytest.raises(ValueError, match="rank0/1"):
        plan(2)


def test_expanded_capacity_preserves_native_context():
    command = plan()
    assert command[command.index('--max-model-len') + 1] == '262144'
    assert command[command.index('--max-num-seqs') + 1] == '16'
    assert command[command.index('--max-num-batched-tokens') + 1] == '8192'
    assert command[command.index('--kv-cache-memory-bytes') + 1] == '25769803776'
    assert '--hf-overrides' not in command
    assert not any(value.startswith('VLLM_ALLOW_LONG_MAX_MODEL_LEN=') for value in command)
    graphs = json.loads(command[command.index('--compilation-config') + 1])
    assert all(4 * concurrency in graphs['cudagraph_capture_sizes'] for concurrency in range(1, 17))
    # The fused rotary-embedding op replaces QSA's per-element PyTorch rope
    # kernels on the decode critical path.
    assert graphs['custom_ops'] == ['+rotary_embedding']


def options():
    return dict(rank=0, master='192.0.2.1', host_ip='192.0.2.1', interface='test0',
                image=installer_image_id(), model=str(ROOT / 'fixture-model'),
                cache=str(ROOT / 'fixture-cache'))


def test_unregistered_image_rejected():
    # The SparkCache profile binds its image to the selected shared release; the
    # installer profile's image is admitted only through the installer image lock.
    values = options()
    values['image'] = 'sha256:' + 'a' * 64
    with pytest.raises(ValueError, match='selected shared release'):
        render(adapter.read(adapter.CONFIG_ROOT / 'sparkcache.json'), **values)
    with pytest.raises(ValueError, match='installer image lock'):
        adapter.image_verification_options(json.loads(PROFILE.read_text()))


def test_toolchain_image_profiles_offer_only_the_plan_action():
    # check and create verify the image with image_verification_options, which
    # refuses toolchain images, so the catalog launcher offers only plan for them.
    from runtime.common import profiles
    toolchain = []
    for profile_id, path in profiles.catalog().items():
        record = profiles.read_json(path)
        if (record['launcher'].get('path') == 'runtime/common/qwen_flash_next.py'
                and adapter.read(ROOT / record['configuration']['path']).get('image_extension') == 'toolchain'):
            toolchain.append(profile_id)
            assert record['launcher']['actions'] == ['plan'], profile_id
    assert {'qwen38-flash-next-tp2', 'qwen38-flash-next-qad-tp4'} <= set(toolchain)


def test_profile_arguments_and_environment_cannot_bypass_canonical():
    for location in ('vllm_args', 'environment'):
        profile = json.loads(PROFILE.read_text())
        if location == 'vllm_args':
            profile[location].append('--kv-transfer-config={}')
        else:
            profile[location]['SPARKCACHE_ENABLED'] = '1'
        with pytest.raises(ValueError, match='canonical'):
            render(profile, **options())


@pytest.mark.parametrize('key,value', [('model', 'relative'), ('cache', '/tmp/cache,readonly'),
                                     ('interface', 'eth0,eth1'), ('host_ip', 'placeholder'),
                                     ('master', 'host --flag')])
def test_invalid_site_inputs(key, value):
    values = options()
    values[key] = value
    with pytest.raises(ValueError):
        render(json.loads(PROFILE.read_text()), **values)


def test_mount_overlap_rejected():
    values = options()
    values['cache'] = str(Path(values['model']) / 'cache')
    with pytest.raises(ValueError, match='disjoint'):
        render(json.loads(PROFILE.read_text()), **values)


def test_explicit_entrypoint_and_compile_identity():
    command = plan()
    # The installer toolchain image runs its system Python and toolchain entrypoint.
    assert command[command.index('--entrypoint') + 1] == 'python3'
    assert adapter.TOOLCHAIN_ENTRYPOINT in command
    namespace = f"qwen-flash-next-{installer_image_id()[7:19]}-60215d26cf5e"
    assert f'VLLM_CACHE_ROOT=/cache/{namespace}/vllm' in command
    assert 'VLLM_SPARK_TP4_MODE=' in command
    assert 'VLLM_SPARK_TP4_VOCAB_MODE=' in command


def test_image_check_binds_receipt_without_glm_admission(monkeypatch):
    from types import SimpleNamespace
    image = publication()['image_id']
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if command[1:3] == ['image', 'inspect']:
            return SimpleNamespace(stdout=json.dumps([{'Id':image,'Os':'linux','Architecture':'arm64',
                'Config':{'Entrypoint':['/opt/venv/bin/python',adapter.candidate.ENTRYPOINT]}}]))
        if '/bin/cat' in command:
            return SimpleNamespace(stdout=b'{"fixture":"raw receipt bytes"}\n')
        assert command[-1] == 'verify'
        return SimpleNamespace(stdout='{"fixture":"verification"}')
    def make_receipt(actual, raw, verification):
        assert actual == image and isinstance(raw, bytes)
        assert verification == {'fixture':'verification'}
        return {'admitted':True}
    monkeypatch.setattr(adapter.candidate, 'make_receipt', make_receipt)
    assert adapter.verify_image(image, run=run) == {'admitted':True}
    assert len(calls) == 3
    assert all('create' not in command and 'start' not in command for command in calls)


def test_wrong_architecture_rejected_before_verifier():
    from types import SimpleNamespace
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=json.dumps([{'Id':publication()['image_id'],'Os':'linux','Architecture':'amd64'}]))
    with pytest.raises(ValueError, match='platform'):
        adapter.verify_image(publication()['image_id'], run=run)
    assert len(calls) == 1


def test_capacity_overrides_rejected():
    for flag, value in (("--max-model-len", "65536"), ("--max-num-seqs", "8"),
                        ("--max-num-batched-tokens", "4096"), ("--kv-cache-memory-bytes", "12884901888")):
        profile = json.loads(PROFILE.read_text())
        profile["vllm_args"][profile["vllm_args"].index(flag) + 1] = value
        with pytest.raises(ValueError, match="canonical"):
            render(profile, **options())


def test_catalog_exposes_native_capacity_and_no_overrides():
    from runtime.common.profiles import load, resolve
    profile, _ = load('qwen38-flash-next-tp2')
    resolved = resolve('qwen38-flash-next-tp2')
    assert profile['overrides'] == []
    # Release qualification covers the native image, not the installer image.
    assert resolved['status'] == 'implemented'
    assert resolved['topology'] == 'direct-pair-2'
    expected = {'tensor_parallel_size': 2, 'decode_context_parallel_size': 1,
                'max_model_len': 262144, 'max_num_seqs': 16,
                'max_num_batched_tokens': 8192, 'kv_cache_memory_bytes': 25769803776}
    assert all(resolved['serving'][key] == value for key, value in expected.items())


def test_evidence_uses_one_native_context_profile():
    root = Path(__file__).resolve().parents[2]
    evidence = json.loads((root / 'performance/records/qwen38-flash-next/r37-tp2.json').read_text())
    assert 'baseline' not in evidence
    assert evidence['checks']['c16_exact_json_passed'] == 16
    assert evidence['checks']['long_context']['prompt_tokens'] == 257504
    assert evidence['profile_defaults']['kv_cache_memory_bytes'] == 25769803776
    assert evidence['measurements']['profile_max_model_len'] == 262144
    assert adapter.CONFIG_NAMES == ('config.json', 'sparkcache.json')
    for name in adapter.CONFIG_NAMES:
        config = adapter.read(adapter.CONFIG_ROOT / name)
        assert config['vllm_args'][config['vllm_args'].index('--max-model-len') + 1] == '262144'


def test_sparkcache_requires_native_release_and_preserves_capacity():
    profile = adapter.read(adapter.CONFIG_ROOT / 'sparkcache.json')
    values = options()
    values['image'] = publication()['image_id']
    with pytest.raises(ValueError, match='selected shared release'):
        adapter.render(profile, **values)
    values['image'] = shared_publication()['image_id']
    command = adapter.render(profile, **values)
    assert command[command.index('--name') + 1] == 'qwen-flash-next-sparkcache-tp2-r0'
    assert command[command.index('--kv-cache-memory-bytes') + 1] == '25769803776'
    connector = json.loads(command[command.index('--kv-transfer-config') + 1])
    assert connector['kv_load_failure_policy'] == 'recompute'
    assert connector['kv_connector_extra_config']['spark_cache_model_profile'] == 'qwen38-flash-next-hybrid'


def test_media_defaults_keep_fixed_capacity():
    for rank in (0, 1):
        command = plan(rank)
        assert json.loads(command[command.index('--limit-mm-per-prompt') + 1]) == {'image': 3, 'video': 1}
        assert json.loads(command[command.index('--media-io-kwargs') + 1]) == {'video': {'num_frames': 16}}
        for flag, value in (('--max-model-len', '262144'), ('--max-num-seqs', '16'),
                            ('--max-num-batched-tokens', '8192'), ('--kv-cache-memory-bytes', '25769803776')):
            assert command[command.index(flag) + 1] == value


def test_media_evidence_does_not_claim_strict_combined_json():
    path = Path(__file__).resolve().parents[2] / 'performance/records/qwen38-flash-next/r37-tp2.json'
    evidence = json.loads(path.read_text())['media_validation']
    assert evidence['concurrency'] == 1
    assert all(row['semantic_pass'] and row['http_status'] == 200 for row in evidence['results'])
    combined = next(row for row in evidence['results'] if row['test'] == 'combined')
    assert combined['strict_json_response'] is False


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="O_NOATIME and relatime access times are Linux behavior")
def test_start_checks_read_checkpoint_metadata_without_changing_access_times(tmp_path):
    folder, cache = tmp_path / "mnt/usb/qwen", tmp_path / "cache"
    folder.mkdir(parents=True)
    cache.mkdir()
    contents = {"config.json": b'{"model_type": "qwen"}', "model.safetensors.index.json": b'{"weight_map": {}}'}
    before = {}
    for name, value in contents.items():
        (folder / name).write_bytes(value)
        info = os.stat(folder / name)
        # An access time three days old would move on an ordinary read, even under relatime.
        os.utime(folder / name, ns=(info.st_atime_ns - 3 * 86400 * 10 ** 9, info.st_mtime_ns))
        before[name] = os.stat(folder / name).st_atime_ns
    profile = {"model": {"config_sha256": hashlib.sha256(contents["config.json"]).hexdigest(),
                         "index_sha256": hashlib.sha256(contents["model.safetensors.index.json"]).hexdigest()}}
    adapter.verify_model_paths(profile, folder, cache)
    # preflight, create and start run this check on copies served in place, which SparkRing did not create.
    assert {name: os.stat(folder / name).st_atime_ns for name in contents} == before
    profile["model"]["config_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Checkpoint metadata mismatch: config.json"):
        adapter.verify_model_paths(profile, folder, cache)


@pytest.mark.parametrize("swift_id, qwen_id, master_port", [
    ("swift15-qwen38-flash-next-tp2", "qwen38-flash-next-tp2", "29639"),
    ("swift15-qwen38-flash-next-tp4", "qwen38-flash-next-qad-tp4", "29780"),
])
def test_swift_follows_the_qwen_profile_except_its_checkpoint_format(swift_id, qwen_id, master_port):
    """Swift 1.5 keeps the Qwen3.8-Flash-Next architecture, so its profiles take the Qwen
    installer settings; only settings its checkpoint format or identity requires may differ.
    A Qwen setting change fails here until the Swift profile adopts or declines it."""
    swift = adapter.canonical(adapter.read(ROOT / "profiles" / swift_id / "config.json"))
    qwen = adapter.read(ROOT / "profiles" / qwen_id / "config.json")
    owned = {"model", "served_model_name", "cache_namespace", "status", "qualification",
             "checkpoint", "checkpoints", "checkpoint_aliases", "environment", "vllm_args"}
    assert {k: v for k, v in swift.items() if k not in owned} == {k: v for k, v in qwen.items() if k not in owned}
    expected = dict(qwen["environment"])
    if adapter.node_count(swift) == 2:
        # Two Sparks cannot keep the 95.4 GiB BF16 PLE table resident beside the other weights and KV.
        del expected["VLLM_PLE_CPU_OFFLOAD"]
        expected["VLLM_PLE_TABLE_MEMORY"] = "disk"
    assert swift["environment"] == expected
    args, expected_args = list(swift["vllm_args"]), list(qwen["vllm_args"])
    expected_args[expected_args.index("--master-port") + 1] = master_port
    # The checkpoint declares ModelOpt NVFP4 routed experts, not MIXED_PRECISION.
    expected_args[expected_args.index("--quantization") + 1] = "modelopt_fp4"
    index = expected_args.index("--speculative-config") + 1
    draft = json.loads(expected_args[index])
    # The draft's BF16 experts run on vLLM's unquantized MoE kernel, not an MXFP8 backend.
    del draft["moe_backend"]
    assert json.loads(args[index]) == draft
    args[index] = expected_args[index] = None
    assert args == expected_args
    nodes = adapter.node_count(swift)
    assert swift["served_model_name"] == f"Swift-1.5-Qwen3.8-Flash-Next-NVFP4-TP{nodes}"
    assert swift["model"]["repository"] == "ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4"
    spec = adapter.container_spec(swift, rank=0, master="192.0.2.1", host_ip="192.0.2.1", interface="test0",
                                  image=installer_image_id(), model=str(ROOT / "fixture-model"),
                                  cache=str(ROOT / "fixture-cache"))
    # Compile caches and container names stay separate from the Qwen checkpoint's.
    assert spec.name == f"swift15-qwen38-flash-next-tp{nodes}-r0"
    assert spec.environment["B12X_COMPILE_CACHE_DIR"].startswith("/cache/swift15-qwen38-flash-next-cuda")
