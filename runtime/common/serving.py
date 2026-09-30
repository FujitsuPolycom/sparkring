"""Per-deployment serving settings that replace a profile's vLLM argument values.

A profile's serving configuration stays the checked-in file that its evidence
describes (qwen_flash_next.canonical). A deployment may name other values for
a few settings; the deployment lock records them, so a deployment with other
settings is another deployment, and installer.specifications writes them into
every rank's container command. A setting that is not named keeps the
profile's value.
"""
import json

# name: (vLLM flag, key in the flag's JSON value or None, value multiplier, minimum, help)
SETTINGS = {
    "max_images": ("--limit-mm-per-prompt", "image", 1, 0, "images per request (0 accepts none)"),
    "max_videos": ("--limit-mm-per-prompt", "video", 1, 0, "videos per request (0 accepts none)"),
    "context_length": ("--max-model-len", None, 1, 1024, "context window in tokens"),
    "max_concurrency": ("--max-num-seqs", None, 1, 1, "requests served at the same time"),
    "kv_cache_gib": ("--kv-cache-memory-bytes", None, 2**30, 1, "KV cache per Spark, in GiB"),
}


def option(name):
    return "--" + name.replace("_", "-")


def add_arguments(parser):
    group = parser.add_argument_group("serving settings", "replace the profile's value; without a flag the profile's value applies")
    for name, (_, _, _, _, text) in SETTINGS.items():
        group.add_argument(option(name), dest="serving_" + name, type=int, metavar="N", help=text)


def from_arguments(args):
    return normalized({name: getattr(args, "serving_" + name, None) for name in SETTINGS})


def normalized(values):
    """The named settings, without unset ones. Unknown names and values below a setting's minimum are refused."""
    result = {}
    for name, value in sorted((values or {}).items()):
        if value is None:
            continue
        if name not in SETTINGS:
            raise ValueError("Unknown serving setting: " + str(name))
        minimum = SETTINGS[name][3]
        if type(value) is not int or value < minimum:
            raise ValueError(f"{option(name)} takes a whole number of at least {minimum}")
        result[name] = value
    return result


def _position(command, flag):
    positions = [index for index, item in enumerate(command) if item == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        return None
    return positions[0] + 1


def profile_value(command, name):
    """The value a command gives setting ``name``, in the setting's unit, or None when the command sets none."""
    flag, key, scale, _, _ = SETTINGS[name]
    position = _position(command, flag)
    if position is None:
        return None
    if key is None:
        return int(command[position]) // scale
    limits = json.loads(command[position])
    return limits.get(key) if isinstance(limits, dict) else None


def apply(command, settings):
    """``command`` with each setting's value in place of the profile's value for its vLLM flag."""
    command = list(command)
    for name in sorted(settings):
        flag, key, scale, _, _ = SETTINGS[name]
        position = _position(command, flag)
        if position is None:
            raise ValueError(f"{option(name)} does not apply to this profile: it sets no {flag}")
        if key is None:
            command[position] = str(settings[name] * scale)
            continue
        limits = json.loads(command[position])
        if not isinstance(limits, dict) or key not in limits:
            raise ValueError(f"{option(name)} does not apply to this profile: its {flag} sets no {key} limit")
        limits[key] = settings[name]
        command[position] = json.dumps(limits, separators=(",", ":"))
    return tuple(command)


def describe(settings, command):
    """One line per named setting: its value and the profile's."""
    return [f"{option(name)} {value} (profile: {profile_value(command, name)})" for name, value in sorted(settings.items())]
