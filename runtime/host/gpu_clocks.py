"""Each rank's GPU SM clock and active clock event reasons, which ``sparkring check`` reports.

A GB10 can keep running far below its clock after a fault: another team's GB10 notes (not reproduced here)
describe one degraded rank after a crash holding all four ranks of its group at 721 MHz, with
``nvidia-smi -lgc`` ignored until a clean reboot. Such a group serves, slowly, and passes functional checks,
so the check reads every rank's GPU with ``nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,
clocks_event_reasons.active`` (read-only) and reports a rank as needing attention when

- any clock event reason other than ``gpu_idle`` is active (NVML's reason bits, named below), or
- the GPU is not idle and its SM clock is below half of its maximum SM clock.

The half-of-maximum floor separates the reported 721 MHz from the 2,405-2,411 MHz that the eight Sparks of
this repository's ring read with clocks locked at 2,418 MHz; it is a heuristic, not a measured limit. An
idle GPU lowers its own clock, so its clock is not judged. A rank whose GPU cannot be read is reported as
unknown, not as needing attention.
"""

QUERY = ["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,clocks_event_reasons.active",
         "--format=csv,noheader,nounits"]
# NVML's clock event reason bits (nvmlClocksEventReason*).
REASONS = {0x1: "gpu_idle", 0x2: "applications_clocks_setting", 0x4: "sw_power_cap", 0x8: "hw_slowdown",
           0x10: "sync_boost", 0x20: "sw_thermal_slowdown", 0x40: "hw_thermal_slowdown",
           0x80: "hw_power_brake_slowdown", 0x100: "display_clock_setting"}
IDLE = 0x1
LOW_FRACTION = 0.5


def _number(text):
    text = text.strip()
    try:
        return int(float(text))
    except ValueError:
        return None


def parse(text):
    """``{"sm_mhz", "max_sm_mhz", "reasons"}`` of one GPU's query line; a field nvidia-smi does not report
    (``[N/A]``) is None, and ``reasons`` names every active bit (unknown bits as hexadecimal)."""
    fields = [field.strip() for field in text.strip().splitlines()[0].split(",")]
    if len(fields) != 3:
        raise ValueError(f"unexpected nvidia-smi output: {text.strip()[:120]}")
    mask = int(fields[2], 16) if fields[2].lower().startswith("0x") else None
    reasons = None
    if mask is not None:
        reasons = [name for bit, name in REASONS.items() if mask & bit]
        unknown = mask & ~sum(REASONS)
        if unknown:
            reasons.append(hex(unknown))
    return {"sm_mhz": _number(fields[0]), "max_sm_mhz": _number(fields[1]), "reasons": reasons}


def attention(reading):
    """Why a rank's reading needs attention: active event reasons other than idle, and an SM clock below
    ``LOW_FRACTION`` of the maximum while the GPU is not idle."""
    found = []
    reasons = reading.get("reasons") or []
    active = [name for name in reasons if name != "gpu_idle"]
    if active:
        found.append("clock event reasons " + ", ".join(active))
    sm, maximum = reading.get("sm_mhz"), reading.get("max_sm_mhz")
    if "gpu_idle" not in reasons and sm is not None and maximum and sm < LOW_FRACTION * maximum:
        found.append(f"SM clock {sm} MHz, below half of its {maximum} MHz maximum")
    return found


def check(ranks, *, run):
    """One row per rank of ``ranks`` (``{"rank", "host"}``): the reading, or ``error`` when the GPU could not
    be read, and ``attention``. ``run(host, argv)`` returns a command's standard output on that Spark."""
    rows = []
    for row in ranks:
        try:
            reading = parse(run(row["host"], QUERY))
        except (OSError, ValueError, RuntimeError) as error:
            rows.append({"rank": row["rank"], "error": str(error).strip().splitlines()[-1][:200] if str(error)
                         else type(error).__name__, "attention": []})
            continue
        rows.append({"rank": row["rank"], **reading, "attention": attention(reading)})
    return rows


def lines(rows):
    """What ``sparkring check`` prints about the ranks' GPU clocks."""
    out = []
    flagged = [row for row in rows if row["attention"]]
    read = [row for row in rows if "error" not in row]
    if read and not flagged:
        clocks = sorted(row["sm_mhz"] for row in read if row["sm_mhz"] is not None)
        span = (f"{clocks[0]}-{clocks[-1]} MHz" if clocks and clocks[0] != clocks[-1]
                else f"{clocks[0]} MHz" if clocks else "not reported")
        out.append(f"GPU clocks: SM {span} on {len(read)} ranks; no clock event reason other than idle")
    for row in flagged:
        out.append(f"GPU needs attention: rank {row['rank']}: " + "; ".join(row["attention"])
                   + f" (SM {row['sm_mhz']} MHz, maximum {row['max_sm_mhz']} MHz, reasons "
                   + (", ".join(row.get("reasons") or []) or "none") + ")")
    for row in rows:
        if "error" in row:
            out.append(f"GPU clocks unknown on rank {row['rank']}: {row['error']}")
    return out
