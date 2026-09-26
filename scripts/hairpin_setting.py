"""ConnectX hairpin setting for four-Spark rings: required values and the in-effect rule.

Status: implemented. Standard library only. The network planner, the node
service, the mesh start check and node status evaluate functions with this rule.

A ConnectX function has the hairpin setting *in effect* when all of these hold:

1. ``devlink -j dev param show pci/<bdf> name <parameter>`` reports the
   ``driverinit`` values ``hairpin_queue_size`` 8192 and ``hairpin_num_queues`` 4;
2. ``devlink -s -j dev show pci/<bdf>`` reports
   ``dev["pci/<bdf>"].stats.reload.driver_reinit.unspecified`` of at least 1,
   and ``reload_failed`` is absent or false;
3. ``ethtool -k <netdev>`` reports ``hw-tc-offload: on``.

``param show`` also reports a ``driverinit`` value that was set but not yet
applied. The kernel applies such a value only in a successful ``driver_reinit``
reload, counts only successful reloads, and resets the counter whenever the
driver probes the function. The probe default is at most 1024 on this hardware,
so 8192 shown with a counter of 0 is a pending value, and 8192 shown with a
counter of at least 1 is the value in use. The one exception needs manual
devlink use: setting another value, reloading, then setting 8192 again without
reloading. ``remote_reload`` counts reloads started by another function and is
ignored, because only a function's own ``driver_reinit`` applies its values.
"""

from __future__ import annotations

from collections.abc import Mapping

# mlx5 sizes each hairpin queue at hairpin_queue_size 64-byte strides and drops
# forwarded packets when it fills, without pausing the upstream link. The
# driver maximum (512 KiB per queue) holds the prepared RoCEnante forwarded-path
# send window; the 1024 default (64 KiB) does not.
HAIRPIN_QUEUE_SIZE = 8192
HAIRPIN_NUM_QUEUES = 4
# Parameter order is the order in which planners set them.
PARAMETERS = {
    "hairpin_num_queues": HAIRPIN_NUM_QUEUES,
    "hairpin_queue_size": HAIRPIN_QUEUE_SIZE,
}

# Function states. RESTART_STATES need a driver_reinit reload; only DEFAULT
# also needs `devlink dev param set` first.
IN_EFFECT = "in-effect"
OFFLOAD_OFF = "offload-off"
PENDING = "pending"
DEFAULT = "default"
FAILED = "failed"
UNKNOWN = "unknown"
STATES = (IN_EFFECT, OFFLOAD_OFF, PENDING, DEFAULT, FAILED, UNKNOWN)
RESTART_STATES = frozenset({DEFAULT, PENDING, FAILED})

# Hardware TC offload as `ethtool -k` shows it. OFFLOAD_FIXED cannot be turned
# on with `ethtool -K`; a function showing it cannot carry four-Spark forwarding.
OFFLOAD_ON = "on"
OFFLOAD_DISABLED = "off"
OFFLOAD_FIXED = "off [fixed]"
OFFLOAD_VALUES = (OFFLOAD_ON, OFFLOAD_DISABLED, OFFLOAD_FIXED)


def _count(value):
    """A non-negative integer, or None; JSON booleans are not counts."""
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    return None


def driverinit_value(document, device, name):
    """Return the ``driverinit`` value of one parameter, or None when absent.

    ``document`` is the parsed output of
    ``devlink -j dev param show <device> name <name>``, for example
    ``{"param": {"pci/0000:01:00.0": [{"name": "hairpin_queue_size",
    "values": [{"cmode": "driverinit", "value": 8192}]}]}}``.
    """
    entries = document.get("param") if isinstance(document, Mapping) else None
    entries = entries.get(device) if isinstance(entries, Mapping) else None
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if not isinstance(entry, Mapping) or entry.get("name") != name:
            continue
        settings = entry.get("values")
        for setting in settings if isinstance(settings, list) else []:
            if isinstance(setting, Mapping) and setting.get("cmode") == "driverinit":
                return _count(setting.get("value"))
    return None


def reload_statistics(document, device):
    """Return ``{"driver_reinit": int or None, "failed": bool or None}``.

    ``document`` is the parsed output of ``devlink -s -j dev show <device>``.
    ``driver_reinit`` is ``dev[device].stats.reload.driver_reinit.unspecified``
    and is None when any level of that path is missing. iproute2 prints
    ``reload_failed`` only when it is true, so its absence means False;
    ``failed`` is None only when the device entry itself is missing or when
    ``reload_failed`` is not a JSON boolean.
    """
    entry = document.get("dev") if isinstance(document, Mapping) else None
    entry = entry.get(device) if isinstance(entry, Mapping) else None
    if not isinstance(entry, Mapping):
        return {"driver_reinit": None, "failed": None}
    failed = entry.get("reload_failed", False)
    counter = entry
    for key in ("stats", "reload", "driver_reinit", "unspecified"):
        counter = counter.get(key) if isinstance(counter, Mapping) else None
    return {
        "driver_reinit": counter if type(counter) is int and counter >= 0 else None,
        "failed": failed if isinstance(failed, bool) else None,
    }


def ethtool_offload(text):
    """Return the ``hw-tc-offload`` value from ``ethtool -k`` text, or None."""
    for line in (text or "").splitlines():
        name, _, value = line.partition(":")
        if name.strip() != "hw-tc-offload":
            continue
        words = value.split()
        if not words or words[0] not in ("on", "off"):
            return None
        if words[0] == "on":
            return OFFLOAD_ON
        return OFFLOAD_FIXED if "[fixed]" in words[1:] else OFFLOAD_DISABLED
    return None


def offload_setting(enabled, fixed):
    """Convert inventory booleans (``hw_tc_offload``, ``hw_tc_offload_fixed``)."""
    if enabled is True:
        return OFFLOAD_ON
    if enabled is False and fixed is False:
        return OFFLOAD_DISABLED
    if enabled is False and fixed is True:
        return OFFLOAD_FIXED
    return None


def function_state(values, statistics, offload):
    """Return the state of one function.

    ``values`` maps ``hairpin_queue_size`` and ``hairpin_num_queues`` to their
    ``driverinit`` values; ``statistics`` is a ``reload_statistics`` result;
    ``offload`` is one of ``OFFLOAD_VALUES`` or None.

    - ``unknown``: a value, the reload counter, ``reload_failed`` or the
      offload setting cannot be read. Nothing may restart the function.
    - ``default``: a value differs from 8192/4. Remedy: ``param set``, then
      restart.
    - ``failed``: the last restart set ``reload_failed``. Remedy: restart.
    - ``pending``: 8192/4 are shown but no restart has applied them since the
      driver probed the function. Remedy: restart without ``param set``.
    - ``offload-off``: the values are in use and hardware TC offload is off.
      Remedy: ``ethtool -K <netdev> hw-tc-offload on``, no restart; with
      ``OFFLOAD_FIXED`` that remedy is impossible and callers report it.
    - ``in-effect``: all three conditions hold.
    """
    values = values if isinstance(values, Mapping) else {}
    statistics = statistics if isinstance(statistics, Mapping) else {}
    current = {name: _count(values.get(name)) for name in PARAMETERS}
    counter = statistics.get("driver_reinit")
    failed = statistics.get("failed")
    if (
        None in current.values()
        or type(counter) is not int
        or counter < 0
        or not isinstance(failed, bool)
        or offload not in OFFLOAD_VALUES
    ):
        return UNKNOWN
    if current != PARAMETERS:
        return DEFAULT
    if failed:
        return FAILED
    if counter < 1:
        return PENDING
    if offload != OFFLOAD_ON:
        return OFFLOAD_OFF
    return IN_EFFECT


def evaluate(values, statistics, offload):
    """Return one function's status row: values, counter, failure flag, offload, state.

    Callers add identity fields (role, netdev, PCI address). The planner's
    ``hairpin`` rows and node status share this shape.
    """
    values = values if isinstance(values, Mapping) else {}
    statistics = statistics if isinstance(statistics, Mapping) else {}
    counter = statistics.get("driver_reinit")
    failed = statistics.get("failed")
    return {
        "values": {name: _count(values.get(name)) for name in PARAMETERS},
        "driver_reinit": counter if type(counter) is int and counter >= 0 else None,
        "reload_failed": failed if isinstance(failed, bool) else None,
        "offload": offload if offload in OFFLOAD_VALUES else None,
        "state": function_state(values, statistics, offload),
    }


def shortfall(row):
    """Describe why an ``evaluate`` row is not in effect, for operator messages.

    Only the causes are named, for example ``hairpin_queue_size 1024,
    required 8192``: a value that differs from the required one, the required
    values set but not yet applied by a driver restart, a failed last restart,
    hardware TC offload that is off, and each reading that is unavailable. A
    row that is in effect gives an empty string.
    """
    values = row.get("values") or {}
    parts = []
    for name, required in PARAMETERS.items():
        value = values.get(name)
        if value is None:
            parts.append(f"{name} unavailable")
        elif value != required:
            parts.append(f"{name} {value}, required {required}")
    counter, failed = row.get("driver_reinit"), row.get("reload_failed")
    if counter is None:
        parts.append("driver restarts since boot unavailable")
    if failed is None:
        parts.append("restart failure flag unavailable")
    elif failed:
        parts.append("last driver restart failed")
    elif counter == 0 and not parts:
        parts.append(
            f"hairpin_queue_size {HAIRPIN_QUEUE_SIZE} set, applied only by a driver restart (none since boot)"
        )
    offload = row.get("offload")
    if offload != OFFLOAD_ON:
        parts.append("hw-tc-offload " + (offload or "unavailable"))
    return ", ".join(parts)


def grouped(rows, name=lambda row: row["netdev"], describe=shortfall):
    """Name the functions of ``rows`` that share a description once, in first-seen order.

    Example: ``enp1s0f0np0, enp1s0f1np1: hairpin_queue_size 1024, required
    8192; enP2p1s0f0np0: last driver restart failed``. ``name`` labels a row
    and ``describe`` gives its text (``shortfall`` by default).
    """
    groups = {}
    for row in rows:
        groups.setdefault(describe(row), []).append(name(row))
    return "; ".join(", ".join(names) + ": " + text for text, names in groups.items())
