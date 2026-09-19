"""Pinned GLM target selection, metadata verification and loader requirements."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECORD = ROOT / "profiles/glm53-target-variants.json"
DEFAULT = "nvfp4-spark"
VARIANTS = (DEFAULT, "nvfp4-qad", "nvidia-nvfp4")


def target(variant=DEFAULT):
    if variant not in VARIANTS:
        raise ValueError("target_model_variant must be nvfp4-spark, nvfp4-qad or nvidia-nvfp4")
    return json.loads(RECORD.read_text())[variant]["target"]


def target_for_image(variant=DEFAULT, image=None):
    """Select maintained host model pins while preserving frozen image recipes."""
    if variant == DEFAULT and (image or {}).get("schema") not in (
            "sparkring-r35-image-receipt/v1", "sparkring-candidate-image-receipt/v1", "sparkring-glm-source-image-receipt/v1", "sparkring-glm-native-image-receipt/v1"):
        return json.loads((ROOT / "runtime/glm53-spark-mtp3-mesh/pins.json").read_text())["target"]
    return target(variant)


def adapt_launcher(text, image):
    """Bind an emitted compatibility launcher without editing its frozen source."""
    recorded = target_for_image()["checkpoint_identity"]
    selected = target_for_image(image=image)["checkpoint_identity"]
    if selected == recorded:
        return text
    assignment = "TARGET_CHECKPOINT_FINGERPRINT=" + recorded
    if text.count(assignment) != 1:
        raise ValueError("Source launcher fingerprint changed; model adaptation requires review")
    return text.replace(assignment, "TARGET_CHECKPOINT_FINGERPRINT=" + selected)


def require_image(variant, image):
    """Admit host configuration only; image admission does not qualify serving."""
    target(variant)
    if variant == "nvfp4-qad":
        if (image or {}).get("schema") != "sparkring-glm-native-image-receipt/v1":
            raise ValueError("LIL NVFP4 QAD requires an explicit native GLM image receipt")
        return
    if variant == DEFAULT or image is None:
        return
    schema = image.get("schema")
    if schema == "sparkring-candidate-image-receipt/v1":
        allowed = json.loads(RECORD.read_text())[variant]["candidate_compositions"]
        if image.get("installed", {}).get("composition_id") in allowed:
            return
    elif schema in ("sparkring-mtp3-performance-public-image/v1", "sparkring-mtp3-mesh-image-receipt/v1"):
        return
    raise ValueError("nvidia-nvfp4 requires the retained compute launcher or the registered R37 composition; R33/R35 and other source-image contracts are unsupported")


def readiness_timeout(variant=DEFAULT, image=None):
    target(variant)
    if variant == "nvfp4-qad" or (image or {}).get("schema") == "sparkring-glm-native-image-receipt/v1":
        return 1800
    return 1500 if variant == "nvidia-nvfp4" else 900


def verify_download(variant, files, image=None):
    """Verify pinned metadata and the variant's shard manifest when available."""
    selected = target(variant)
    if variant == DEFAULT:
        if target_for_image(variant, image)["revision"] == selected["revision"]:
            expected = json.loads(RECORD.read_text())[variant]["metadata_sha256"]
            for name, identity in expected.items():
                if files.get(name) != identity:
                    raise ValueError("Spark model metadata differs from its pinned identity: " + name)
        return
    expected = json.loads(RECORD.read_text())[variant]["source_files"]
    label = "QAD" if variant == "nvfp4-qad" else "NVIDIA"
    if set(files) != set(expected):
        raise ValueError(label + " model file set differs from the pinned revision")
    for name, identity in expected.items():
        if "sha256" not in identity:
            raise ValueError(label + " manifest lacks a pinned SHA256 identity: " + name)
        if files[name] != identity["sha256"]:
            raise ValueError(label + " model file differs from its pinned identity: " + name)
    for name, field in (("config.json", "config_sha256"), ("model.safetensors.index.json", "index_sha256")):
        if files[name] != selected[field]:
            raise ValueError(label + " model metadata differs from its pinned identity: " + name)


def environment(variant, values, image):
    """Bind model-specific loader settings and persistent cache namespaces."""
    require_image(variant, image)
    result = dict(values)
    if variant == "nvfp4-qad":
        result.update(TARGET_MODEL_VARIANT=variant, LOAD_FORMAT="b12x", VLLM_PLUGINS="b12x_loader",
                      B12X_NVFP4_DYNAMIC_MATERIALIZED="0", DFLASH_WARMUP_TIMEOUT_SECONDS="1800")
        result["SPARKCACHE_CACHE_NAMESPACE"] += "-nvfp4-qad-" + target(variant)["revision"][:12]
    elif variant != DEFAULT:
        result.update(TARGET_MODEL_VARIANT=variant, LOAD_FORMAT="safetensors",
                      DFLASH_WARMUP_TIMEOUT_SECONDS="1500")
        result["SPARKCACHE_CACHE_NAMESPACE"] += "-" + variant
    elif target_for_image(variant, image)["revision"] == target(variant)["revision"]:
        result["SPARKCACHE_CACHE_NAMESPACE"] += "-" + target(variant)["revision"][:12]
    return result


def mtp_override(config):
    """Preserve quantization fields while excluding every BF16 MTP predictor."""
    if not isinstance(config, dict):
        raise ValueError("Target config.json must contain an object")
    text = config.get("text_config", config)
    if not isinstance(text, dict):
        raise ValueError("Target text_config must contain an object")
    quantization = config.get("quantization_config") or text.get("quantization_config")
    if (not isinstance(quantization, dict) or quantization.get("quant_method") != "modelopt"
            or quantization.get("quant_algo") != "NVFP4"):
        raise ValueError("NVIDIA target requires ModelOpt NVFP4 quantization_config")
    layers, predictors = text.get("num_hidden_layers"), text.get("num_nextn_predict_layers")
    if (type(layers) is not int or not 1 <= layers <= 1024
            or type(predictors) is not int or not 1 <= predictors <= 64):
        raise ValueError("Target must declare finite base and MTP predictor layer counts")
    ignore = quantization.get("ignore", [])
    if not isinstance(ignore, list) or any(not isinstance(item, str) for item in ignore):
        raise ValueError("Target quantization ignore must be a list of strings")
    result = deepcopy(quantization)
    result["ignore"] = list(ignore)
    for index in range(layers, layers + predictors):
        for prefix in ("model.language_model.layers", "model.layers"):
            pattern = f"{prefix}.{index}*"
            if pattern not in result["ignore"]:
                result["ignore"].append(pattern)
    return {"quantization_config": result}


def verified_override(variant, config_bytes, index_bytes):
    """Verify original metadata bytes before deriving any executable override."""
    selected = target(variant)
    for name, data, field in (("config.json", config_bytes, "config_sha256"),
                              ("model.safetensors.index.json", index_bytes, "index_sha256")):
        if not isinstance(data, bytes) or hashlib.sha256(data).hexdigest() != selected[field]:
            raise ValueError("Target metadata is missing or differs from its pinned identity: " + name)
    return mtp_override(json.loads(config_bytes)) if variant == "nvidia-nvfp4" else None
