"""Bounded management grammar shared by the dashboard and privileged helper.

No path to executable code, image lock, environment file, output file or
shell text is accepted. Values are passed as argv, never interpreted by a shell.
"""
import re

NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
ARC = re.compile(r"[0-7](?:-[0-7]|(?:,[0-7]){1,7})")
REVIEW = re.compile(r"[0-9a-f]{64}")
OPTIONS = {
    "checkpoint": lambda value: bool(NAME.fullmatch(value)),
    "context-length": lambda value: value.isdecimal() and 1024 <= int(value) <= 4_194_304,
    "kv-cache-gib": lambda value: value.isdecimal() and 1 <= int(value) <= 128,
    "download-limit": lambda value: bool(re.fullmatch(r"[1-9][0-9]{0,3}[MG]bit", value)),
    "transport": lambda value: value in ("sircl", "prepared", "nccl"),
}


def base(request):
    if not isinstance(request, dict) or set(request) - {"action", "on", "profile", "options"}:
        raise ValueError("Unknown action fields")
    action, on = request.get("action"), request.get("on")
    if not isinstance(on, str) or not ARC.fullmatch(on):
        raise ValueError("Choose a slice by its fabric positions")
    options = request.get("options") or {}
    if not isinstance(options, dict):
        raise ValueError("Options must be a mapping")
    if action == "install":
        profile = request.get("profile")
        if not isinstance(profile, str) or not NAME.fullmatch(profile):
            raise ValueError("Choose an exact profile")
        if set(options) - set(OPTIONS):
            raise ValueError("Unsupported installation option")
        argv = ["install", "--profile", profile, "--on", on]
        for key, valid in OPTIONS.items():
            if key in options:
                value = options[key]
                if not isinstance(value, str) or len(value) > 64 or not valid(value):
                    raise ValueError("Invalid " + key)
                argv += ["--" + key, value]
        return argv
    if options or request.get("profile"):
        raise ValueError("Profile and options apply only to install/swap")
    if action in ("down", "restart"):
        return [action, "--on", on]
    if action in ("recovery-on", "recovery-off"):
        return ["recover", action.split("-")[1], "--on", on]
    if action == "logs":
        return ["logs", "--on", on, "--lines", "80", "--plain"]
    raise ValueError("Unsupported dashboard action")


def command(request, *, review=None):
    argv = base(request)
    if argv[0] == "logs":
        return argv
    if review is None:
        return [*argv, "--plan", "--json"]
    if not isinstance(review, str) or not REVIEW.fullmatch(review):
        raise ValueError("A reviewed plan is required")
    return [*argv, *(["--execute"] if argv[0] in ("down", "restart") else []),
            "--yes", "--json", "--review", review]


def allowed(argv):
    """Decode, then re-encode: exact ordering, no repeats, no unknown flags."""
    try:
        if argv[0] == "install" and argv[1] == "--profile" and argv[3] == "--on":
            request = {"action": "install", "profile": argv[2], "on": argv[4], "options": {}}
            index = 5
            while index + 1 < len(argv) and argv[index].removeprefix("--") in OPTIONS:
                key = argv[index].removeprefix("--")
                if key in request["options"]:
                    return False
                request["options"][key] = argv[index + 1]
                index += 2
        elif argv[0] == "recover" and argv[2] == "--on":
            request = {"action": "recovery-" + argv[1], "on": argv[3]}
        elif argv[0] in ("down", "restart", "logs") and argv[1] == "--on":
            request = {"action": argv[0], "on": argv[2]}
        else:
            return False
        review = argv[-1] if len(argv) >= 2 and argv[-2] == "--review" else None
        return argv == command(request, review=review)
    except (ValueError, TypeError, IndexError, KeyError):
        return False
