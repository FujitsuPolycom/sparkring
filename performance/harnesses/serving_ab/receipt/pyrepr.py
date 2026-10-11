"""Parse the Python ``repr`` text that vLLM writes into its configuration log lines.

vLLM logs its explicit arguments as ``non-default args: {...}`` (a ``dict`` repr) and its resolved engine
configuration as ``key=value`` pairs whose values are reprs of dicts, lists, enums and dataclasses. These
texts are not JSON: they hold single-quoted strings, ``True``/``None``, tuples, enum members such as
``<CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>``, dataclass reprs such as ``SpeculativeConfig(method='mtp',
num_spec_tokens=3)`` and bare names such as ``torch.bfloat16``. :func:`parse` turns such text into JSON-ready
values without evaluating it:

- ``dict``, ``list``, ``tuple`` and ``set`` become dicts and lists;
- an enum member becomes ``{"enum": "CUDAGraphMode.FULL_AND_PIECEWISE", "value": [2, 1]}``;
- a call such as ``Name(a=1)`` becomes ``{"__type__": "Name", "a": 1}`` (positional arguments under ``args``);
- a bare or dotted name becomes its text (``"torch.bfloat16"``);
- any other ``<...>`` text becomes ``{"repr": "<...>"}``.

Captured lines are sometimes cut short (a ``cut -c1-4000`` in a capture script). With ``partial=True`` the
parser keeps every item completed before the end of the text and reports that the text was truncated.
"""

from __future__ import annotations

import ast
import re

_NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?|[-+]?(?:inf|nan)\b")
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_STRING_START = re.compile(r"(?:[rRbBuU]|[rR][bB]|[bB][rR])?['\"]")
_CONSTANTS = {"True": True, "False": False, "None": None}


class ReprError(ValueError):
    """The text is not a repr this parser reads."""


class _Truncated(Exception):
    """The text ended inside a value."""


class _Parser:
    def __init__(self, text: str):
        self.text, self.pos, self.truncated = text, 0, False

    def skip(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos] in " \t\r\n":
            self.pos += 1

    def peek(self) -> str:
        self.skip()
        if self.pos >= len(self.text):
            raise _Truncated
        return self.text[self.pos]

    def expect(self, char: str) -> None:
        if self.peek() != char:
            raise ReprError(f"expected {char!r} at {self.pos}: {self.text[self.pos:self.pos + 40]!r}")
        self.pos += 1

    def value(self):
        char = self.peek()
        if char == "{":
            return self.braces()
        if char == "[":
            return self.sequence("[", "]")
        if char == "(":
            items = self.sequence("(", ")")
            return items
        if char == "<":
            return self.angle()
        string = _STRING_START.match(self.text, self.pos)
        if string:
            return self.string(string.end() - 1)
        number = _NUMBER.match(self.text, self.pos)
        if number and not _NAME.match(self.text, self.pos):
            self.pos = number.end()
            token = number.group()
            if token.lstrip("+-") in ("inf", "nan"):
                return token
            return float(token) if any(c in token for c in ".eE") else int(token)
        if self.text.startswith("...", self.pos):
            self.pos += 3
            return "..."
        name = _NAME.match(self.text, self.pos)
        if name:
            self.pos = name.end()
            word = name.group()
            if word in ("inf", "nan"):
                return word
            if word in _CONSTANTS:
                return _CONSTANTS[word]
            if self.pos < len(self.text) and self.text[self.pos] == "(":
                return self.call(word)
            return word
        raise ReprError(f"unexpected {char!r} at {self.pos}: {self.text[self.pos:self.pos + 40]!r}")

    def string(self, quote_at: int):
        quote = self.text[quote_at]
        triple = self.text.startswith(quote * 3, quote_at)
        end_token = quote * 3 if triple else quote
        i = quote_at + len(end_token)
        while True:
            if i >= len(self.text):
                raise _Truncated
            if self.text[i] == "\\":
                i += 2
                continue
            if self.text.startswith(end_token, i):
                i += len(end_token)
                break
            i += 1
        token = self.text[self.pos:i]
        self.pos = i
        value = ast.literal_eval(token)
        return value.decode("latin-1") if isinstance(value, bytes) else value

    def items(self, close: str, item):
        out = []
        try:
            while True:
                if self.peek() == close:
                    self.pos += 1
                    return out
                out.append(item())
                char = self.peek()
                if char == ",":
                    self.pos += 1
                elif char != close:
                    raise ReprError(f"expected ',' or {close!r} at {self.pos}: {self.text[self.pos:self.pos + 40]!r}")
        except _Truncated:
            # The text ended inside the next item: keep the items before it, and let every enclosing
            # container end here too.
            self.truncated = True
            self.pos = len(self.text)
            return out

    def sequence(self, open_: str, close: str) -> list:
        self.expect(open_)
        return self.items(close, self.value)

    def braces(self):
        self.expect("{")
        if self.peek() == "}":
            self.pos += 1
            return {}
        start = self.pos
        self.value()  # the first item: a dict when a colon follows it, else a set
        if self.peek() != ":":
            self.pos = start
            return self.items("}", self.value)  # a set: kept as a list
        self.pos = start
        pairs = self.items("}", self.pair)
        return {self.key(k): v for k, v in pairs}

    def pair(self):
        key = self.value()
        self.expect(":")
        return key, self.value()

    @staticmethod
    def key(key) -> str:
        return key if isinstance(key, str) else repr(key)

    def call(self, name: str) -> dict:
        self.expect("(")
        out: dict = {"__type__": name}
        positional = []

        def argument():
            keyword = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=(?!=)").match(self.text, self.pos)
            if keyword:
                self.pos = keyword.end()
                out[keyword.group(1)] = self.value()
            else:
                positional.append(self.value())
            return None

        self.items(")", argument)
        if positional:
            out["args"] = positional
        return out

    def angle(self):
        self.expect("<")
        name = _NAME.match(self.text, self.pos)
        if name and self.text.startswith(":", name.end()):
            self.pos = name.end() + 1
            member = self.value()
            self.expect(">")
            return {"enum": name.group(), "value": member}
        depth, i = 1, self.pos
        while depth:
            if i >= len(self.text):
                raise _Truncated
            depth += {"<": 1, ">": -1}.get(self.text[i], 0)
            i += 1
        raw = "<" + self.text[self.pos:i]
        self.pos = i
        return {"repr": raw}


def parse(text: str, *, partial: bool = False):
    """The value of one repr ``text``; with ``partial``, ``(value, truncated)`` for text that may be cut short.

    Without ``partial`` a truncated or malformed text raises :class:`ReprError`; trailing text after the
    value also raises it.
    """
    parser = _Parser(text)
    try:
        value = parser.value()
    except _Truncated:
        if not partial:
            raise ReprError("the text ends inside a value") from None
        return None, True
    if parser.truncated and not partial:
        raise ReprError("the text ends inside a value")
    if not parser.truncated:
        parser.skip()
        if parser.pos != len(text):
            raise ReprError(f"trailing text at {parser.pos}: {text[parser.pos:parser.pos + 40]!r}")
    return (value, parser.truncated) if partial else value


def split_assignments(text: str) -> list[tuple[str, str]]:
    """Split ``a=1, b=Name(x=2, y=[3, 4]), c=text`` into ``[(a, '1'), (b, 'Name(...)'), (c, 'text')]``.

    A comma separates two assignments only outside brackets and quotes and only when the next text is
    ``name=``, so values that are plain text (``served_model_name=Qwen3.8-Flash``) survive. A value cut short
    by truncation is returned as far as it goes.
    """
    out: list[tuple[str, str]] = []
    start_name = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)=")
    match = start_name.match(text)
    if not match:
        return out
    key, i, value_start = match.group(1), match.end(), match.end()
    depth, quote = 0, ""
    while i < len(text):
        char = text[i]
        if quote:
            if char == "\\":
                i += 2
                continue
            if text.startswith(quote, i):
                i += len(quote)
                quote = ""
                continue
        elif char in "'\"":
            quote = char * 3 if text.startswith(char * 3, i) else char
            i += len(quote)
            continue
        elif char in "([{<":
            depth += 1
        elif char in ")]}>":
            depth = max(depth - 1, 0)
        elif char == "," and depth == 0:
            following = start_name.match(text, i + 1)
            if following:
                out.append((key, text[value_start:i].strip()))
                key, value_start = following.group(1), following.end()
                i = following.end()
                continue
        i += 1
    out.append((key, text[value_start:].strip()))
    return out


def value_or_text(text: str):
    """The parsed repr of ``text``, or ``text`` itself when it is plain text rather than a repr."""
    try:
        return parse(text)
    except ReprError:
        value, truncated = parse(text, partial=True) if text[:1] in "{[(<" else (None, False)
        return value if truncated and value is not None else text
