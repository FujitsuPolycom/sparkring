"""Per-deployment serving settings that replace a profile's vLLM argument values.

A profile's serving configuration stays the checked-in file that its evidence
describes (toolchain_profiles.canonical). A deployment may name other values for
a few settings; the deployment lock records them, so a deployment with other
settings is another deployment, and installer.specifications writes them into
every rank's container command. A setting that is not named keeps the
profile's value. SWITCHES are settings without a value that set a container
environment variable instead of a vLLM argument.

Two settings set the endpoint of the API that vLLM serves on the deployment's
first rank (the API rank): ``api_port`` replaces ``--port``, and ``api_bind``,
an IPv4 address of the API rank's Spark, replaces ``--host`` on the API rank
only, so that the API listens on that address alone. The API rank's container
health check then asks the API at that address and port (``container``).

CHOICES are settings whose value is a word, or for some models a number: what
the model does with thinking when a request does not choose. The profile sets
no value for them; the chat template, or for DeepSeek vLLM's prompt encoder,
decides (runtime/common/thinking.py). A choice adds keys to the JSON object of
vLLM's --default-chat-template-kwargs on the API rank only, since the other
ranks run --headless and serve no API. The keys and the accepted values are
the model's, so applying a choice needs the profile and checkpoint
(``model``). A request's own values take precedence over these defaults.
"""
import dataclasses
import ipaddress
import json
import re

# name: (vLLM flag, key in the flag's JSON value or None, value multiplier, minimum, help)
SETTINGS = {
    "max_images": ("--limit-mm-per-prompt", "image", 1, 0, "images per request (0 accepts none)"),
    "max_videos": ("--limit-mm-per-prompt", "video", 1, 0, "videos per request (0 accepts none)"),
    "context_length": ("--max-model-len", None, 1, 1024, "context window in tokens"),
    "max_concurrency": ("--max-num-seqs", None, 1, 1, "requests served at the same time"),
    "kv_cache_gib": ("--kv-cache-memory-bytes", None, 2**30, 1, "KV cache per Spark in GiB, at most a tenth above the profile's"),
    "api_port": ("--port", None, 1, 1024, "TCP port of the model's API, at most 65535"),
    "api_bind": ("--host", None, 1, None, "IPv4 address of the API's Spark that the API listens on alone; "
                                          "default: every address"),
}
# Settings whose value is an IPv4 address instead of a whole number. Their
# minimum is None; a value must be an address a Spark can listen on.
ADDRESSES = frozenset({"api_bind"})
# The largest accepted value of a whole-number setting that has one.
MAXIMUM = {"api_port": 65535}
# Settings that only the API rank's command takes. The other ranks run
# headless, serve no API and keep the profile's value.
API_RANK = frozenset({"api_bind"})
# TCP ports that SparkRing's own services use on the Sparks, which the API port
# may not take. SSH's port 22 is below the API port's minimum. The ports that
# the profile's command sets for its ranks, such as --master-port, are refused
# as well (reserved_ports).
RESERVED_PORTS = {2222: "the port of SparkRing's administration SSH", 5255: "the port of SparkRing's image relay",
                  29500: "vLLM's default master port"}
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
# (FujitsuPolycom/sparkring#189). Only an image with the shared-memory reader
# window of runtime/images/derive_spin_wait.py reads the variable (NEEDS).
SWITCHES = {
    "save_cpu": ("SPARKRING_SHM_BUSY_LOOP_S", "0.002",
                 "let vLLM's waiting processes sleep between decode steps: less CPU use, about 1 to 2 percent slower decode"),
}
# The installer image capability (installer_image.capabilities) that each
# switch needs: on an image without it, the switch's variable would change nothing.
NEEDS = {"save_cpu": "shm_reader_window"}
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
        if name in ADDRESSES:
            group.add_argument(option(name), dest="serving_" + name, metavar="ADDRESS", help=text)
        else:
            group.add_argument(option(name), dest="serving_" + name, type=int, metavar="N", help=text)
    for name, (_, _, text) in SWITCHES.items():
        group.add_argument(option(name), dest="serving_" + name, action="store_true", default=None, help=text)
    for name, (_, values, metavar, text) in CHOICES.items():
        group.add_argument(option(name), dest="serving_" + name, choices=values, metavar=metavar, help=text)


def from_arguments(args):
    return normalized({name: getattr(args, "serving_" + name, None) for name in (*SETTINGS, *SWITCHES, *CHOICES)})


def listenable(value):
    """Whether ``value`` is an IPv4 address in dotted form that a Spark can listen on.

    Addresses in 0.0.0.0/8 and multicast or reserved addresses (224.0.0.0 and
    above) are not.
    """
    if not isinstance(value, str):
        return False
    try:
        parsed = ipaddress.IPv4Address(value)
    except ValueError:
        return False
    return str(parsed) == value and 0 < int(parsed) >> 24 < 224


def normalized(values):
    """The named settings, without unset ones.

    Unknown names, whole numbers outside a setting's range and ADDRESSES values
    that are not ``listenable`` are refused.
    """
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
        if name in ADDRESSES:
            if not listenable(value):
                raise ValueError(f"{option(name)} takes one IPv4 address of the Spark that serves the API, such as "
                                 "192.0.2.10; without it the API listens on every address")
            result[name] = value
            continue
        minimum, maximum = SETTINGS[name][3], MAXIMUM.get(name)
        if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
            raise ValueError(f"{option(name)} takes a whole number "
                             + (f"from {minimum} to {maximum}" if maximum is not None else f"of at least {minimum}"))
        result[name] = value
    return result


def check_image(settings, image, capabilities):
    """Refuse a switch that image release ``image``, whose capabilities are ``capabilities``, cannot apply (NEEDS)."""
    for name in sorted(set(settings) & set(NEEDS)):
        if NEEDS[name] not in capabilities:
            raise ValueError(f"{option(name)} needs an image whose vLLM reads {SWITCHES[name][0]}, and "
                             f"{image} does not. Leave out {option(name)} or choose another image.")


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
    if name in ADDRESSES:
        return command[position]
    if key is None:
        return int(command[position]) // scale
    limits = json.loads(command[position])
    return limits.get(key) if isinstance(limits, dict) else None


def reserved_ports(command):
    """``[(port, reason)]`` of the ports ``api_port`` may not take: RESERVED_PORTS, then the command's other ports.

    The command's other ports are the values of its flags named ``--*-port``
    other than ``--port``, such as ``--master-port``, in command order.
    """
    found = sorted(RESERVED_PORTS.items())
    for index, item in enumerate(command[:-1]):
        if item != "--port" and re.fullmatch(r"--[a-z][a-z-]*-port", item) and str(command[index + 1]).isdigit():
            found.append((int(command[index + 1]), f"the profile's {item}"))
    return found


def apply(command, settings, *, model=None, api=True):
    """``command`` with each setting's value in place of the profile's value for its vLLM flag.

    ``model`` is ``(profile, checkpoint)``, which CHOICES settings require
    (chat_defaults). ``api`` is false for a rank other than the API rank: its
    command keeps the profile's value of every API_RANK setting. ``api_port``
    is refused when it is one of the command's ``reserved_ports``.
    """
    command = list(command)
    for name in sorted(set(settings) - set(SWITCHES) - set(CHOICES)):
        if name in API_RANK and not api:
            continue
        flag, key, scale, _, _ = SETTINGS[name]
        position = _position(command, flag)
        if position is None:
            raise ValueError(f"{option(name)} does not apply to this profile: it sets no {flag}")
        if name == "api_port":
            for port, reason in reserved_ports(command):
                if settings[name] == port:
                    raise ValueError(f"{option(name)} {port} is {reason}. Choose another port.")
        if name in ADDRESSES:
            command[position] = settings[name]
            continue
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


def container(spec, settings, *, rank, model=None):
    """Rank ``rank``'s container specification ``spec`` with the settings applied.

    Each setting replaces its vLLM flag's value in the command (``apply``; only
    the API rank, rank 0, takes the API_RANK settings), and each switch sets
    its container variable (``environment``). The API rank's health check asks
    the API at 127.0.0.1 on the profile's port; with ``api_port`` or
    ``api_bind`` it asks at the bound address and the deployment's port, where
    the API then listens. ``model`` is ``(profile, checkpoint)``, which CHOICES
    settings require. Without settings ``spec`` is returned unchanged.
    """
    if not settings:
        return spec
    health = spec.health_command
    if health and {"api_port", "api_bind"} & set(settings):
        port = profile_value(spec.command, "api_port")
        before = f"//127.0.0.1:{port}/"
        after = f"//{settings.get('api_bind', '127.0.0.1')}:{settings.get('api_port', port)}/"
        if port is None or not any(before in part for part in health):
            raise ValueError("--api-port and --api-bind do not apply to this profile: its health check does not "
                             "ask the API at 127.0.0.1 on its --port")
        health = tuple(part.replace(before, after) for part in health)
    return dataclasses.replace(spec, command=apply(spec.command, settings, model=model, api=rank == 0),
                               environment={**spec.environment, **environment(settings)}, health_command=health)


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
