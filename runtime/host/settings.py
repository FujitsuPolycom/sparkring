"""Optional literal setup preferences; credentials are entered through SSH."""
import ipaddress
from pathlib import Path
import re

# An empty SPARKRING_RETAIN_DEPLOYMENTS means the file does not set it, so the
# setting saved on Node A applies (runtime/host/retention.py); a file value is never empty.
DEFAULTS = {"SPARKRING_NAME": "sparkring", "SPARKRING_SSH_USER": "root", "SPARKRING_SSH_PORT": "22",
            "SPARKRING_SHARE_INTERNET": "yes", "SPARKRING_CONTROL_CIDR": "10.253.255.0/29",
            "SPARKRING_FABRIC_CIDR": "198.18.0.0/21", "SPARKRING_LINK_POLICY": "keep",
            "SPARKRING_DOWNLOAD_LIMIT": "none", "SPARKRING_RETAIN_DEPLOYMENTS": ""}
# The slowest accepted download limit, in bits per second.
MIN_DOWNLOAD_BITS = 10 ** 6
DOWNLOAD_LIMIT_HELP = "use none, or a rate in megabits or gigabits per second such as 850Mbit or 2Gbit"
# The most recent deployments of each profile that automatic release may be set to keep.
MAX_RETAINED = 99
RETAIN_HELP = f"use off, or the number of recent deployments of each profile to keep, 0 to {MAX_RETAINED}"


def retain_deployments(text):
    """The number of recent deployments of each profile that automatic release keeps; None for ``off``.

    ``0`` keeps only the deployments that the policy always keeps (the active
    one, the rollback target and the others ``runtime/host/retention.py``
    lists); ``off`` turns automatic release off.
    """
    value = text.strip().lower()
    if value == "off":
        return None
    if not re.fullmatch(r"[0-9]{1,2}", value):
        raise ValueError(f"Retained deployments {text!r}: " + RETAIN_HELP)
    return int(value)


def download_limit(text):
    """Bytes per second for a download limit such as ``850Mbit`` or ``1.5Gbit``; None for ``none``.

    The unit is bits per second, as network links are rated; ``Mbit`` is 10^6
    and ``Gbit`` 10^9 bits per second, in any letter case. A limit below 1 Mbit
    per second is refused.
    """
    if text.strip().lower() == "none":
        return None
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([mg])bit", text.strip(), re.IGNORECASE)
    if not match:
        raise ValueError(f"Download limit {text!r}: " + DOWNLOAD_LIMIT_HELP)
    bits = float(match.group(1)) * (10 ** 6 if match.group(2).lower() == "m" else 10 ** 9)
    if bits < MIN_DOWNLOAD_BITS:
        raise ValueError(f"Download limit {text!r} is below 1Mbit; " + DOWNLOAD_LIMIT_HELP)
    return int(bits // 8)


def load(path=None):
    values = dict(DEFAULTS)
    seen = set()
    if path:
        for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, sep, value = line.partition("=")
            if not sep or key not in values or key in seen or not value or not re.fullmatch(r"[A-Za-z0-9_./-]+", value):
                raise ValueError(f"{path}:{number}: use one literal supported KEY=value; passwords and shell code are not settings")
            seen.add(key)
            values[key] = value
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,34}", values["SPARKRING_NAME"]):
        raise ValueError("Choose a lowercase cluster name of at most 35 characters")
    if not re.fullmatch(r"[a-z_][a-z0-9_-]{0,31}", values["SPARKRING_SSH_USER"]):
        raise ValueError("Invalid SSH user")
    if values["SPARKRING_SSH_PORT"] not in ("22", "2222") or values["SPARKRING_SHARE_INTERNET"] not in ("yes", "no") or values["SPARKRING_LINK_POLICY"] not in ("keep", "reset"):
        raise ValueError("Use port 22/2222, Internet yes/no, and link policy keep/reset")
    control = ipaddress.IPv4Network(values["SPARKRING_CONTROL_CIDR"])
    fabric = ipaddress.IPv4Network(values["SPARKRING_FABRIC_CIDR"])
    if control.prefixlen != 29 or fabric.prefixlen not in range(16, 22) or control.overlaps(fabric):
        raise ValueError("Control requires /29; fabric requires a separate /16 through /21")
    download_limit(values["SPARKRING_DOWNLOAD_LIMIT"])
    # SPARKRING_RETAIN_DEPLOYMENTS is checked where it is used
    # (install_workflow.retain_preference), so that a refusal names that key.
    return values
