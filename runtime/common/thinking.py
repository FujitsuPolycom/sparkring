"""What an installer model does with thinking when a request does not choose, and the defaults that change it.

A reasoning model can think before it answers. Whether it thinks, and how
hard (its effort level), is decided when vLLM renders a request into a
prompt: the checkpoint's chat template, or for DeepSeek-V4.1-Flash vLLM's
own prompt encoder, reads named arguments such as ``enable_thinking`` and
``reasoning_effort``. vLLM merges three sources of these arguments, each
later one taking precedence: the server option
``--default-chat-template-kwargs`` (a JSON object), a request's
``chat_template_kwargs``, and a request's ``reasoning_effort`` field, which
also sets ``enable_thinking`` unless the request names it. Unset request
values do not replace a default.

profiles/thinking.json records each behaviour, which is one chat template's
or encoder's handling of these arguments:

- ``default``: ``on`` (a request can turn thinking off), ``off`` or
  ``always`` (no argument turns it off);
- ``level``: the effort level that applies when none is named, or null;
- ``levels``: the named levels it accepts, from least to most thinking,
  and optionally ``range``, the lowest and highest whole number it accepts
  as a level;
- ``effort``: the argument that names a level, or null without levels;
- ``off``: the arguments that turn thinking off, or null;
- ``source``: the evidence: the SHA-256 of every chat template whose text
  was read for it (``chat_templates``), or for an encoder the image and the
  digests of its modules, and what that text does (``reading``).

It also maps each checkpoint, ``<repository>@<revision>``, to a behaviour.
Each checkpoint's pin manifest under profiles/checkpoints gives the SHA-256
of its chat_template.jinja, which must be one that its behaviour lists
(runtime/common/test_thinking.py). A checkpoint the file does not map has no
recorded behaviour.

Records apply to profiles on the shared installer image (configuration
``image_extension`` ``toolchain``), whose vLLM was read for the merge above
and for the DeepSeek encoder. Other profiles have no record.

profiles/research-thinking.json holds, in the same form, the behaviours and
checkpoints that only research-only profiles (profiles.RESEARCH_CATALOG)
serve. Every Compose export's label hashes profiles/thinking.json and not
this file, so a research profile's records change no export. ``catalog``
reads both; a name in both is refused, and a research checkpoint may name a
behaviour of either file. A record moves to profiles/thinking.json when a
profile of the main catalog serves its checkpoint.
"""
import re

from runtime.common import profiles

ROOT = profiles.ROOT
CATALOG = "profiles/thinking.json"
RESEARCH = "profiles/research-thinking.json"
DEFAULTS = ("on", "off", "always")
FIELDS = frozenset({"default", "level", "levels", "effort", "off", "source"})
CHECKPOINT = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")


def _behaviour(name, value):
    """Refuse a behaviour with missing fields or fields that contradict each other."""
    if not isinstance(value, dict) or not FIELDS <= set(value) <= FIELDS | {"range"}:
        raise ValueError(f"Thinking behaviour {name}: expected the fields " + ", ".join(sorted(FIELDS)) + " and optional range")
    levels, level, effort, off = value["levels"], value["level"], value["effort"], value["off"]
    if value["default"] not in DEFAULTS:
        raise ValueError(f"Thinking behaviour {name}: default is one of " + ", ".join(DEFAULTS))
    if (not isinstance(levels, list) or len(set(levels)) != len(levels)
            or any(not isinstance(item, str) or not re.fullmatch(r"[a-z]+", item) for item in levels)):
        raise ValueError(f"Thinking behaviour {name}: levels are distinct lowercase words")
    numbers = value.get("range")
    if numbers is not None and (not isinstance(numbers, list) or len(numbers) != 2
                                or any(type(item) is not int for item in numbers) or not 1 <= numbers[0] <= numbers[1]):
        raise ValueError(f"Thinking behaviour {name}: range is the lowest and highest whole number, at least 1")
    if (effort is None) != (not levels and numbers is None) or (effort is not None and not isinstance(effort, str)):
        raise ValueError(f"Thinking behaviour {name}: effort names the argument of its levels, and is null without levels")
    if level is not None and level not in levels or (level is None) != (effort is None):
        raise ValueError(f"Thinking behaviour {name}: level is one of its levels, or null without levels")
    if off is not None and (not isinstance(off, dict) or not off or any(not isinstance(key, str) for key in off)):
        raise ValueError(f"Thinking behaviour {name}: off holds the arguments that turn thinking off, or is null")
    if (value["default"] == "always") != (off is None and value["default"] != "off"):
        raise ValueError(f"Thinking behaviour {name}: a model that thinks by default without an off switch always thinks")
    source = value["source"]
    if (not isinstance(source, dict) or not isinstance(source.get("chat_templates"), list)
            or any(not isinstance(item, str) or not SHA256.fullmatch(item) for item in source["chat_templates"])
            or not isinstance(source.get("reading"), str) or not source["reading"]):
        raise ValueError(f"Thinking behaviour {name}: source lists the chat templates read and what they do")


def _file(path, name):
    data = profiles.read_json(path)
    if set(data) != {"schema", "behaviours", "checkpoints"} or data["schema"] != "sparkring-thinking/v1":
        raise ValueError(name + ": expected sparkring-thinking/v1 with behaviours and checkpoints")
    return data


def catalog(root=ROOT):
    """profiles/thinking.json with profiles/research-thinking.json merged in, validated."""
    data = _file(root / CATALOG, CATALOG)
    if (root / RESEARCH).is_file():
        research = _file(root / RESEARCH, RESEARCH)
        for part in ("behaviours", "checkpoints"):
            both = sorted(set(data[part]) & set(research[part]))
            if both:
                raise ValueError(f"{', '.join(both)} appear in both {CATALOG} and {RESEARCH}")
        data = {**data, "behaviours": {**data["behaviours"], **research["behaviours"]},
                "checkpoints": {**data["checkpoints"], **research["checkpoints"]}}
    for name, value in data["behaviours"].items():
        _behaviour(name, value)
    for key, name in data["checkpoints"].items():
        if not CHECKPOINT.fullmatch(key) or name not in data["behaviours"]:
            raise ValueError(f"{CATALOG}: {key} must be <repository>@<full revision> and name a behaviour")
    return data


def model(profile_id, checkpoint=None, root=ROOT):
    """``<repository>@<revision>`` that ``profile_id`` serves with ``checkpoint``, or None for a profile without records.

    ``checkpoint`` is a name or alias of the profile's checkpoints table, or
    None for its default checkpoint. A derived checkpoint is the derived
    repository, whose files the containers serve.
    """
    entries = profiles.catalog(root)
    if profile_id not in entries:
        return None
    source = profiles.read_json(entries[profile_id]).get("configuration") or {}
    if source.get("format") != "serving-profile":
        return None
    configuration = profiles.read_json(profiles.local_path(source["path"], root))
    if configuration.get("image_extension") != "toolchain":
        return None
    from runtime.common import toolchain_profiles
    served = toolchain_profiles.checkpoint_settings(configuration, checkpoint)["model"]
    return f"{served['repository']}@{served['revision']}"


def of(profile_id, checkpoint=None, root=ROOT):
    """The thinking record of the checkpoint that ``profile_id`` serves with ``checkpoint``, or None when none is recorded.

    A record is the behaviour's fields with ``name``, the behaviour's name,
    and ``checkpoint``, ``<repository>@<revision>``.
    """
    key = model(profile_id, checkpoint, root)
    if key is None:
        return None
    data = catalog(root)
    name = data["checkpoints"].get(key)
    if name is None:
        return None
    return {"name": name, "checkpoint": key, **data["behaviours"][name]}


def summary(record):
    """A record's default in a few words: ``on · xhigh``, ``always · max``, ``on``; ``not recorded`` without one."""
    if record is None:
        return "not recorded"
    return " · ".join(str(part) for part in (record["default"], record["level"]) if part is not None)


def accepts(record, value):
    """Whether ``value`` is one of the record's named levels or a whole number in its range."""
    if isinstance(value, str):
        return value in record["levels"]
    numbers = record.get("range")
    return type(value) is int and numbers is not None and numbers[0] <= value <= numbers[1]


def levels_text(record):
    """The levels a record accepts, for a message: ``low, medium or xhigh``; empty without levels."""
    words = list(record["levels"])
    text = ", ".join(words[:-1]) + " or " + words[-1] if len(words) > 1 else "".join(words)
    numbers = record.get("range")
    if numbers is not None:
        text += (", or " if text else "") + f"a whole number from {numbers[0]} to {numbers[1]}"
    return text


def arguments(record, settings, profile_id):
    """The chat template arguments that the thinking serving settings in ``settings`` set for ``profile_id``.

    ``settings`` may hold ``reasoning_effort``, a level name or whole number,
    and ``thinking``, whose only value is ``off``. A value the record does not
    accept, a setting its model has no argument for, both settings together
    and any setting without a record are refused with the valid choices.
    """
    effort, thinking = settings.get("reasoning_effort"), settings.get("thinking")
    named = [f"{flag} {value}" for flag, value in (("--reasoning-effort", effort), ("--thinking", thinking))
             if value is not None]
    if not named:
        return {}
    if record is None:
        raise ValueError(f"{' and '.join(named)} do{'es' if len(named) == 1 else ''} not apply to {profile_id}: "
                         "no thinking behaviour is recorded for its checkpoint on its image")
    if effort is not None and thinking is not None:
        raise ValueError("--reasoning-effort sets how hard the model thinks and --thinking off turns thinking off; "
                         "choose one")
    if thinking is not None:
        if record["off"] is None:
            least = f" --reasoning-effort {record['levels'][0]} makes it think least." if record["levels"] else ""
            raise ValueError(f"--thinking off does not apply to {profile_id}: its model always thinks, and its chat "
                             "template has no switch that turns thinking off." + least)
        if record["default"] == "off":
            raise ValueError(f"--thinking off does not apply to {profile_id}: its model does not think by default")
        return dict(record["off"])
    if record["effort"] is None:
        raise ValueError(f"--reasoning-effort does not apply to {profile_id}: its model has no effort levels"
                         + ("; --thinking off turns its thinking off" if record["off"] else ""))
    if not accepts(record, effort):
        raise ValueError(f"--reasoning-effort {effort} is not a level that {profile_id} accepts; choose "
                         f"{levels_text(record)} (default: {record['level']})")
    return {record["effort"]: effort}


def deployment_default(record, settings):
    """What applies to a request that names no thinking argument, or None without a record.

    ``{"thinking", "level", "chosen", "model"}``: ``thinking`` and ``level``
    as the deployment's thinking settings make them, ``chosen`` whether it
    has such a setting, and ``model`` the record's own default and level.
    """
    if record is None:
        return None
    settings = settings or {}
    default = {"thinking": record["default"], "level": record["level"]}
    current = dict(default)
    if settings.get("thinking") == "off":
        current = {"thinking": "off", "level": None}
    elif settings.get("reasoning_effort") is not None:
        current["level"] = settings["reasoning_effort"]
    chosen = settings.get("thinking") is not None or settings.get("reasoning_effort") is not None
    return {**current, "chosen": chosen, "model": default}


def deployment_text(state):
    """``deployment_default`` in words: ``on · xhigh (model default)`` or ``off (deployment default; model default: on)``."""
    if state is None:
        return "not recorded for this model"
    model_text = summary({"default": state["model"]["thinking"], "level": state["model"]["level"]})
    if not state["chosen"]:
        return model_text + " (model default)"
    return summary({"default": state["thinking"], "level": state["level"]}) + f" (deployment default; model default: {model_text})"
