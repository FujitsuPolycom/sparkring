"""Target identity and BF16 MTP preservation without model weights or hardware."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from runtime.common import glm_targets

CONFIG = Path(__file__).with_name("fixtures") / "glm53-nvidia-nvfp4-config.json"


def test_nvidia_metadata_matches_contributor_pin_and_preserves_quantization():
    raw = CONFIG.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == glm_targets.target("nvidia-nvfp4")["config_sha256"]
    config = json.loads(raw)
    original = deepcopy(config)
    override = glm_targets.mtp_override(config)["quantization_config"]
    quant = config["quantization_config"]
    assert override == {**quant, "ignore": quant["ignore"] + [
        "model.language_model.layers.45*", "model.layers.45*",
    ]}
    assert config == original


def test_mtp_override_handles_multiple_predictors_and_keeps_existing_exclusions():
    config = json.loads(CONFIG.read_bytes())
    config["text_config"]["num_nextn_predict_layers"] = 2
    config["quantization_config"]["ignore"].append("model.layers.45*")
    override = glm_targets.mtp_override(config)["quantization_config"]
    assert override["ignore"].count("model.layers.45*") == 1
    assert override["ignore"][-2:] == ["model.language_model.layers.46*", "model.layers.46*"]


@pytest.mark.parametrize("field,value", [
    ("num_hidden_layers", True), ("num_hidden_layers", "45"), ("num_hidden_layers", 0),
    ("num_nextn_predict_layers", 0), ("num_nextn_predict_layers", 65),
    ("num_nextn_predict_layers", None), ("num_nextn_predict_layers", 1.5),
])
def test_invalid_predictor_geometry_rejected(field, value):
    config = json.loads(CONFIG.read_bytes())
    config["text_config"][field] = value
    with pytest.raises(ValueError, match="layer counts"):
        glm_targets.mtp_override(config)


@pytest.mark.parametrize("ignore", [None, {}, "lm_head", [1]])
def test_invalid_ignore_list_rejected(ignore):
    config = json.loads(CONFIG.read_bytes())
    config["quantization_config"]["ignore"] = ignore
    with pytest.raises(ValueError, match="ignore"):
        glm_targets.mtp_override(config)


@pytest.mark.parametrize("config,index", [(None, None), (b"{}", b"{}"), (CONFIG.read_bytes(), b"{}")])
def test_unverified_metadata_never_produces_executable_override(config, index):
    with pytest.raises(ValueError, match="pinned identity"):
        glm_targets.verified_override("nvidia-nvfp4", config, index)


def test_verified_metadata_produces_override(monkeypatch):
    target = glm_targets.target("nvidia-nvfp4")
    index = b'{"weight_map":{}}'
    target["index_sha256"] = hashlib.sha256(index).hexdigest()
    monkeypatch.setattr(glm_targets, "target", lambda variant: target)
    result = glm_targets.verified_override("nvidia-nvfp4", CONFIG.read_bytes(), index)
    assert result == glm_targets.mtp_override(json.loads(CONFIG.read_bytes()))


@pytest.mark.parametrize("schema", ["sparkring-r33-image-receipt/v1", "sparkring-r35-image-receipt/v1",
                                   "sparkring-source-image-receipt/v1", "unknown"])
def test_frozen_and_unknown_images_reject_nvidia(schema):
    with pytest.raises(ValueError, match="unsupported"):
        glm_targets.require_image("nvidia-nvfp4", {"schema": schema})


def test_candidate_requires_registered_composition():
    image = {"schema": "sparkring-candidate-image-receipt/v1", "installed": {"composition_id": "unknown"}}
    with pytest.raises(ValueError, match="unsupported"):
        glm_targets.require_image("nvidia-nvfp4", image)
    image["installed"]["composition_id"] = "lil-r37-glm-spark"
    glm_targets.require_image("nvidia-nvfp4", image)


def test_variant_preserves_default_and_isolates_cache_after_image_binding():
    values = {"TARGET_MODEL_VARIANT": "nvfp4-spark", "SPARKCACHE_CACHE_NAMESPACE": "image-cache",
              "LOAD_FORMAT": "instanttensor", "DFLASH_WARMUP_TIMEOUT_SECONDS": "600"}
    assert glm_targets.environment("nvfp4-spark", values, None) == values
    result = glm_targets.environment("nvidia-nvfp4", values, None)
    assert result["SPARKCACHE_CACHE_NAMESPACE"] == "image-cache-nvidia-nvfp4"
    assert result["LOAD_FORMAT"] == "safetensors"
    assert result["DFLASH_WARMUP_TIMEOUT_SECONDS"] == "1500"
    assert values["LOAD_FORMAT"] == "instanttensor"
    assert glm_targets.readiness_timeout() == 900
    assert glm_targets.readiness_timeout("nvidia-nvfp4") == 1500


def test_target_shard_manifest_is_revision_bound():
    record = json.loads(glm_targets.RECORD.read_bytes())["nvidia-nvfp4"]
    target = record["target"]
    assert hashlib.sha256(f"{target['repository']}@{target['revision']}".encode()).hexdigest() == target["checkpoint_identity"]
    shards = [value for path, value in record["source_files"].items() if path.endswith(".safetensors")]
    assert len(shards) == 33
    assert all(len(shard["sha256"]) == 64 and shard["size"] > 0 for shard in shards)


@pytest.mark.parametrize("corruption", ["shard", "missing", "metadata", "none"])
def test_nvidia_download_rejects_changed_or_incomplete_model(corruption):
    record = json.loads(glm_targets.RECORD.read_bytes())["nvidia-nvfp4"]
    files = {name: identity.get("sha256", "0" * 64) for name, identity in record["source_files"].items()}
    files["config.json"] = record["target"]["config_sha256"]
    files["model.safetensors.index.json"] = record["target"]["index_sha256"]
    shard = next(name for name in files if name.endswith(".safetensors"))
    if corruption == "shard":
        files[shard] = "0" * 64
    elif corruption == "missing":
        del files[shard]
    elif corruption == "metadata":
        files["config.json"] = "0" * 64
    if corruption == "none":
        glm_targets.verify_download("nvidia-nvfp4", files)
    else:
        with pytest.raises(ValueError, match="pinned"):
            glm_targets.verify_download("nvidia-nvfp4", files)
