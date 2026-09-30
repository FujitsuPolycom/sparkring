"""Per-deployment serving settings that replace a profile's vLLM argument values.

A profile's serving configuration stays the checked-in file that its evidence
describes (qwen_flash_next.canonical). A deployment may name other values for
a few settings; the deployment lock records them, so a deployment with other
settings is another deployment, and installer.specifications writes them into
every rank's container command. A setting that is not named keeps the
profile's value. SWITCHES are settings without a value that set a container
environment variable instead of a vLLM argument.
"""
import json

# name: (vLLM flag, key in the flag's JSON value or None, value multiplier, minimum, help)
SETTINGS = {
    "max_images": ("--limit-mm-per-prompt", "image", 1, 0, "images per request (0 accepts none)"),
    "max_videos": ("--limit-mm-per-prompt", "video", 1, 0, "videos per request (0 accepts none)"),
    "context_length": ("--max-model-len", None, 1, 1024, "context window in tokens"),
    "max_concurrency": ("--max-num-seqs", None, 1, 1, "requests served at the same time"),
    "kv_cache_gib": ("--kv-cache-memory-bytes", None, 2**30, 1, "KV cache per Spark in GiB, at most a tenth above the profile's"),
}
# vLLM allocates the KV cache when the model starts. On a Spark, whose GPU and
# CPU share one memory, a KV cache far above the profile's value exhausts that
# memory: the kernel then stops processes and the Spark stops answering SSH
# until it recovers, which also defeats the installer's restoration of the
# previous model (observed with 200 GiB on a profile of 24 GiB). The GLM
# profiles started with a tenth more than their values (11 GiB for 10, 44 GiB
# for 40; performance/records/glm53-flash/installer-memory-20260929.md), so a
# value up to a tenth above the profile's, and at least one unit above it, is
# accepted with a warning, and a larger one is refused.
ABOVE_PROFILE = frozenset({"kv_cache_gib"})
# name: (container environment variable, value, help). save_cpu shortens the
# time vLLM's shared-memory readers poll after a read from one second to 2 ms,
# so between decode steps they sleep until notified instead of keeping CPU
# cores busy; each step then waits for a reader to wake, which cost about 1%
# of decode steps per second with one request and 2% with eight
# (FujitsuPolycom/sparkring#189). Only an image derived with
# runtime/images/derive_spin_wait.py reads the variable.
SWITCHES = {
    "save_cpu": ("SPARKRING_SHM_BUSY_LOOP_S", "0.002",
                 "let vLLM's waiting processes sleep between decode steps: less CPU use, about 1-2% slower decode"),
}


def ceiling(profile):
    """The largest accepted value of an ABOVE_PROFILE setting whose profile value is ``profile``."""
    return profile + max(1, profile // 10)


def option(name):
    return "--" + name.replace("_", "-")


def add_arguments(parser):
    group = parser.add_argument_group("serving settings", "replace the profile's value; without a flag the profile's value applies")
    for name, (_, _, _, _, text) in SETTINGS.items():
        group.add_argument(option(name), dest="serving_" + name, type=int, metavar="N", help=text)
    for name, (_, _, text) in SWITCHES.items():
        group.add_argument(option(name), dest="serving_" + name, action="store_true", default=None, help=text)


def from_arguments(args):
    return normalized({name: getattr(args, "serving_" + name, None) for name in (*SETTINGS, *SWITCHES)})


def normalized(values):
    """The named settings, without unset ones. Unknown names and values below a setting's minimum are refused."""
    result = {}
    for name, value in sorted((values or {}).items()):
        if value is None:
            continue
        if name in SWITCHES:
            if value is not True:
                raise ValueError(f"{option(name)} is a switch without a value")
            result[name] = True
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
    for name in sorted(set(settings) - set(SWITCHES)):
        flag, key, scale, _, _ = SETTINGS[name]
        position = _position(command, flag)
        if position is None:
            raise ValueError(f"{option(name)} does not apply to this profile: it sets no {flag}")
        if key is None:
            profile = int(command[position]) // scale
            if name in ABOVE_PROFILE and settings[name] > ceiling(profile):
                raise ValueError(f"{option(name)} {settings[name]} is more than {ceiling(profile)}, a tenth above the profile's "
                                 f"{profile}. A larger value can exhaust a Spark's memory while the model starts.")
            command[position] = str(settings[name] * scale)
            continue
        limits = json.loads(command[position])
        if not isinstance(limits, dict) or key not in limits:
            raise ValueError(f"{option(name)} does not apply to this profile: its {flag} sets no {key} limit")
        limits[key] = settings[name]
        command[position] = json.dumps(limits, separators=(",", ":"))
    return tuple(command)


def environment(settings):
    """Container environment variables that the named switches set."""
    return {SWITCHES[name][0]: SWITCHES[name][1] for name in sorted(set(settings) & set(SWITCHES))}


def label(name, value):
    """A setting as its command-line form: ``--max-images 8``, or ``--save-cpu`` for a switch."""
    return option(name) if name in SWITCHES else f"{option(name)} {value}"


def describe(settings, command):
    """One line per named setting: its value and the profile's."""
    return [label(name, value) + (" (profile: off)" if name in SWITCHES else f" (profile: {profile_value(command, name)})")
            for name, value in sorted(settings.items())]


def warnings(settings, command):
    """One line per ABOVE_PROFILE setting whose value exceeds the profile's in ``command``."""
    lines = []
    for name in sorted(ABOVE_PROFILE & set(settings)):
        profile = profile_value(command, name)
        if profile is not None and settings[name] > profile:
            lines.append(f"{option(name)} {settings[name]} is above the profile's {profile}: each Spark keeps that much less "
                         "memory for images and long requests, and the profile's measurements do not cover it.")
    return lines
