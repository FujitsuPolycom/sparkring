"""Highlighting of runnable SparkRing commands in the CLI's terminal output.

Messages name the commands the reader can run next, for example
``sudo sparkring fabric verify`` inside a ``sparkring status`` line. A message
builder wraps each command with :func:`cmd`; the writer that finally puts the
text on a stream decides what those marks become:

- a terminal shows each marked span in bold cyan;
- a log file, event stream or JSON document receives the text without marks,
  byte-identical to the same message built without highlighting.

The marks are private sentinel characters that only this module and the
writers it documents (``runtime.host.terminal.Console``, ``runtime.host.progress``)
interpret. No message is scanned for command-like text after the fact: a span
exists only where a builder placed it.

Colour follows the stream, not the process: ``stdout`` and ``stderr`` decide
independently. ``--color=auto|always|never`` on a command's parser overrides
``SPARKRING_COLOR``, which overrides the automatic detection. ``auto`` colours
a stream only when it is a terminal and neither ``NO_COLOR`` (any value, per
no-color.org) nor ``TERM=dumb`` applies. ``always`` keeps the colour on a
non-terminal stream, for ``watch -c sudo sparkring status --color=always``.
``never`` removes every command colour and the terminal decoration's
line colours as well.
"""
import os
import re

MARK = "\x00"
STYLE = "\x1b[1;36m"
RESET = "\x1b[0m"
MODES = ("auto", "always", "never")
ENV = "SPARKRING_COLOR"

_flag = None

_SPAN = re.compile(re.escape(MARK) + r"([^\n" + re.escape(MARK) + r"]*)" + re.escape(MARK))
# One status field names the command to run next, such as ``sparkring models``
# or ``sudo sparkring status --refresh``; a value that is not a command alone,
# for example the recovery advice to reconnect a cable, stays plain.
_ACTION = re.compile(r"(sudo )?sparkring(?![\w-])(?: [^\s;|,]+)*")


def cmd(text):
    """Mark ``text`` as one runnable command span inside a message."""
    return MARK + text + MARK


def plain(text):
    """The message without marks: what every non-terminal output receives."""
    return text.replace(MARK, "") if MARK in text else text


def set_mode(value):
    """Record the ``--color`` value a command's parser accepted (None when absent)."""
    global _flag
    if value is not None and value not in MODES:
        raise ValueError("--color must be one of " + ", ".join(MODES))
    _flag = value


def mode():
    """The selected mode: the ``--color`` value, else ``SPARKRING_COLOR``, else ``auto``.

    An unrecognised ``SPARKRING_COLOR`` value is ignored, so a stale variable
    cannot force colour into pipes and log files.
    """
    if _flag is not None:
        return _flag
    value = os.environ.get(ENV)
    return value if value in MODES else "auto"


def enabled(stream):
    """Whether ``stream`` receives colour under the selected mode."""
    selected = mode()
    if selected == "never":
        return False
    if selected == "always":
        return True
    return stream.isatty() and "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb"


def render(text, stream):
    """The message for ``stream``: marked spans coloured, or marks removed."""
    if MARK not in text:
        return text
    if not enabled(stream):
        return plain(text)
    return _SPAN.sub(STYLE + r"\1" + RESET, text).replace(MARK, "")


def action(text):
    """The status field ``text`` names the command to run next, marked when it is one.

    ``sparkring status`` prints each state's ``next_action`` value verbatim.
    The value is a command slot, so this marks the leading
    ``[sudo ]sparkring ...`` run; values that name no SparkRing command, such
    as ``systemctl status sparkring-agent`` or recovery advice, stay plain.
    """
    if MARK in text:
        return text
    match = _ACTION.match(text)
    return cmd(match[0]) + text[match.end():] if match and match[0] else text


def parser():
    """An ``--color`` argument definition for a command's own ``ArgumentParser``.

    ``sparkring status --color=always`` and every other command that renders
    highlighted messages accepts it after the subcommand; pass this parser in
    ``parents=``.
    """
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--color", choices=MODES, metavar="WHEN", default=None,
                        help="colour the named commands: auto (a terminal, without NO_COLOR or TERM=dumb), "
                             "always, or never; overrides SPARKRING_COLOR")
    return parser
