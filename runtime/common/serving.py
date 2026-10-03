"""Per-deployment serving settings that replace a profile's vLLM argument values.

A profile's serving configuration stays the checked-in file that its evidence
describes (qwen_flash_next.canonical). A deployment may name other values for
a few settings; the deployment lock records them, so a deployment with other
settings is another deployment, and installer.specifications writes them into
every rank's container command. A setting that is not named keeps the
profile's value. SWITCHES are settings without a value that set a container
environment variable instead of a vLLM argument.

CHOICES are settings whose value is a word, or for some models a number: what
the model does with thinking when a request does not choose. The profile sets
no value for them; the chat template, or for DeepSeek vLLM's prompt encoder,
decides (runtime/common/thinking.py). A choice adds keys to the JSON object of
vLLM's --default-chat-template-kwargs on the API rank only, since the other
ranks run --headless and serve no API. The keys and the accepted values are
the model's, so applying a choice needs the profile and checkpoint
(``model``). A request's own values take precedence over these defaults.
"""
import json
import re

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
                 "let vLLM's waiting processes sleep between decode steps: less CPU use, about 1 to 2 percent slower decode"),
}
# name: (vLLM flag, accepted values or None for the model's own, metavar, help)
CHOICES = {
    "reasoning_effort": ("--default-chat-template-kwargs", None, "LEVEL",
                         "how hard the model thinks when a request does not say: one of its levels, "
                         "which sparkring models lists"),
    "thinking": ("--default-chat-template-kwargs", ("off",), "off",
                 "turn the model's thinking off when a request does not turn it on"),
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
    for name, (_, values, metavar, text) in CHOICES.items():
        group.add_argument(option(name), dest="serving_" + name, choices=values, metavar=metavar, help=text)


def from_arguments(args):
    return normalized({name: getattr(args, "serving_" + name, None) for name in (*SETTINGS, *SWITCHES, *CHOICES)})


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
        if name in CHOICES:
            result[name] = choice(name, value)
            continue
        if name not in SETTINGS:
            raise ValueError("Unknown serving setting: " + str(name))
        minimum = SETTINGS[name][3]
        if type(value) is not int or value < minimum:
            raise ValueError(f"{option(name)} takes a whole number of at least {minimum}")
        result[name] = value
    return result


def choice(name, value):
    """A CHOICES value as the deployment records it: a lowercase word, or a whole number of at least 1.

    A word of digits is the number. The model's own values are checked when
    the setting is applied (thinking.arguments).
    """
    values = CHOICES[name][1]
    if isinstance(value, str) and value.isascii() and value.isdigit() and values is None:
        value = int(value)
    if values is not None and value not in values:
        raise ValueError(f"{option(name)} takes " + " or ".join(values))
    if not (isinstance(value, str) and re.fullmatch(r"[a-z]+", value) or type(value) is int and value >= 1):
        raise ValueError(f"{option(name)} takes a level name, such as low, or a whole number of at least 1")
    return value


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


def apply(command, settings, *, model=None):
    """``command`` with each setting's value in place of the profile's value for its vLLM flag.

    ``model`` is ``(profile, checkpoint)``, which CHOICES settings require
    (chat_defaults).
    """
    command = list(command)
    for name in sorted(set(settings) - set(SWITCHES) - set(CHOICES)):
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
    return chat_defaults(command, settings, model)


def chat_defaults(command, settings, model):
    """``command`` with the CHOICES settings' chat template arguments in vLLM's --default-chat-template-kwargs.

    ``model`` is ``(profile, checkpoint)``: the profile ID and its checkpoint
    name, or None for the default checkpoint. thinking.arguments gives the
    model's keys and refuses a value or setting the model does not accept. The
    arguments go on the API rank's command, which has no --headless, as a JSON
    object with sorted keys; a command that already sets the flag keeps its
    other keys. Without CHOICES settings the command is unchanged.
    """
    chosen = {name: settings[name] for name in CHOICES if name in settings}
    if not chosen:
        return tuple(command)
    from runtime.common import thinking
    if model is None:
        raise ValueError(" and ".join(option(name) for name in sorted(chosen)) + " apply only with the profile and checkpoint")
    profile, checkpoint = model
    values = thinking.arguments(thinking.of(profile, checkpoint), chosen, profile)
    command = list(command)
    if "--headless" in command:
        return tuple(command)
    flag = CHOICES[next(iter(chosen))][0]
    position = _position(command, flag)
    if position is None:
        command += [flag, "{}"]
        position = len(command) - 1
    defaults = json.loads(command[position])
    if not isinstance(defaults, dict):
        raise ValueError(f"The profile's {flag} must be a JSON object")
    defaults.update(values)
    command[position] = json.dumps(defaults, sort_keys=True, separators=(",", ":"))
    return tuple(command)


def environment(settings):
    """Container environment variables that the named switches set."""
    return {SWITCHES[name][0]: SWITCHES[name][1] for name in sorted(set(settings) & set(SWITCHES))}


def label(name, value):
    """A setting as its command-line form: ``--max-images 8``, or ``--save-cpu`` for a switch."""
    return option(name) if name in SWITCHES else f"{option(name)} {value}"


def describe(settings, command, *, model=None):
    """One line per named setting: its value and the profile's, or for a CHOICES setting the model's default.

    ``model`` is ``(profile, checkpoint)`` as for apply; without it, or
    without a thinking record, a CHOICES setting's default reads ``not recorded``.
    """
    lines = []
    for name, value in sorted(settings.items()):
        if name in CHOICES:
            lines.append(label(name, value) + f" (model default: {model_default(name, model)})")
            continue
        lines.append(label(name, value) + (" (profile: off)" if name in SWITCHES else f" (profile: {profile_value(command, name)})"))
    return lines


def model_default(name, model):
    """What a model does without CHOICES setting ``name``: its default level, or its thinking default."""
    from runtime.common import thinking
    record = thinking.of(*model) if model is not None else None
    if record is None:
        return "not recorded"
    if name == "reasoning_effort":
        return record["level"] or "no levels"
    return record["default"]


def warnings(settings, command):
    """One line per ABOVE_PROFILE setting whose value exceeds the profile's in ``command``."""
    lines = []
    for name in sorted(ABOVE_PROFILE & set(settings)):
        profile = profile_value(command, name)
        if profile is not None and settings[name] > profile:
            lines.append(f"{option(name)} {settings[name]} is above the profile's {profile} GiB, leaving each Spark "
                         f"{settings[name] - profile} GiB less for images and long requests. "
                         "This value has not been validated as stable.")
    return lines
