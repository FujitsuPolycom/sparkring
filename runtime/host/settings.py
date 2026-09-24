"""Optional literal setup preferences; credentials are entered through SSH."""
import ipaddress
from pathlib import Path
import re

DEFAULTS = {"SPARKRING_NAME": "sparkring", "SPARKRING_SSH_USER": "root", "SPARKRING_SSH_PORT": "22",
            "SPARKRING_SHARE_INTERNET": "yes", "SPARKRING_CONTROL_CIDR": "10.253.255.0/29",
            "SPARKRING_FABRIC_CIDR": "198.18.0.0/21", "SPARKRING_LINK_POLICY": "keep"}


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
    return values
