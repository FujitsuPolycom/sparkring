"""Deployment settings for the SparkContextCacheConnector.

Budget changes replace only capacity settings, never cache identity fields.
Integrity mode selects the upstream publication schema and cache identity.
Debug logging reads a deployment-mounted configuration and preserves identity.
The admission floor is one configured capture slot, including for synchronous
profiles: this reserves room for one full entry without depending on a runtime
tensor inventory. Slot counts and restore arenas are memory, not disk quotas.
"""
import json
import re

SETTINGS = frozenset({"sparkcache_size", "sparkcache_integrity", "sparkcache_debug"})
DEBUG_SOURCE = "runtime/common/sparkcache-debug-logging.json"
DEBUG_TARGET = "/run/sparkring/sparkcache-debug-logging.json"


def apply_integrity(command, mode):
    """Select upstream's payload format; its publication schema owns cache identity."""
    value = connector(command)
    if value is None:
        raise ValueError("--sparkcache-integrity requires a SparkCache profile")
    value["kv_connector_extra_config"]["spark_cache_payload_integrity"] = mode
    result = list(command)
    result[result.index("--kv-transfer-config") + 1] = json.dumps(value, separators=(",", ":"))
    return result


def deployment_options(lock):
    """Status fields for SparkCache settings retained in the deployment record."""
    if not (lock.get("selection") or {}).get("sparkcache"):
        return {}
    settings = lock.get("serving") or {}
    return {"sparkcache_integrity": settings.get("sparkcache_integrity", "fast"),
            "sparkcache_debug": bool(settings.get("sparkcache_debug"))}


def option_lines(values):
    lines = []
    if values.get("sparkcache_integrity") is not None:
        lines.append("SparkCache integrity: " + values["sparkcache_integrity"])
    if values.get("sparkcache_debug"):
        lines.append("SparkCache logs: debug")
    return lines


def parse_size(value):
    """Whole bytes, or binary K/M/G/T sizes with optional B or iB suffixes."""
    if type(value) is int and value > 0:
        return value
    if isinstance(value, str):
        match = re.fullmatch(r"([0-9]+)([KMGT](?:I?B)?|B)?", value, re.IGNORECASE)
        if match:
            unit = (match[2] or "B")[0].upper()
            size = int(match[1]) * 1024 ** "BKMGT".index(unit)
            if size > 0:
                return size
    raise ValueError("--sparkcache-size requires a positive size, such as 32G, 64GiB or 500M")


def connector(command):
    if command.count("--kv-transfer-config") != 1:
        return None
    position = command.index("--kv-transfer-config") + 1
    if position >= len(command):
        return None
    value = json.loads(command[position])
    return value if value.get("kv_connector") == "SparkContextCacheConnector" else None


def budget(command):
    value = connector(command)
    return value["kv_connector_extra_config"]["spark_cache_max_bytes"] if value else None


def apply(command, size):
    value = connector(command)
    if value is None:
        raise ValueError("--sparkcache-size requires a SparkCache profile")
    size = parse_size(size)
    extra = value["kv_connector_extra_config"]
    minimum = extra["spark_cache_async_page_capture_slot_bytes"]
    if size < minimum:
        raise ValueError(f"--sparkcache-size must be at least {format_size(minimum)} for this profile")
    extra.update(spark_cache_max_bytes=size, spark_cache_low_watermark_bytes=size * 7 // 8)
    result = list(command)
    result[result.index("--kv-transfer-config") + 1] = json.dumps(value, separators=(",", ":"))
    return result


def format_size(size):
    for exponent, unit in ((4, "TiB"), (3, "GiB"), (2, "MiB"), (1, "KiB")):
        if size >= 1024 ** exponent and size % 1024 ** exponent == 0:
            return f"{size // 1024 ** exponent} {unit}"
    return f"{size} B"


def status_line(size):
    return f"SparkCache: {format_size(size)} per Spark"


def deployment_budget(lock, *, root=None):
    """Read a deployment's override or its selected profile's connector default."""
    from runtime.common import profiles
    explicit = (lock.get("serving") or {}).get("sparkcache_size")
    if explicit is not None:
        return explicit
    card = lock.get("selection") or {}
    if not card.get("sparkcache") or not card.get("configuration"):
        return None
    try:
        configuration = profiles.read_json((root or profiles.ROOT) / card["configuration"])
    except (OSError, ValueError):
        # Status may describe a retained deployment whose profile is no longer packaged.
        return None
    return budget(configuration.get("vllm_args", []))
