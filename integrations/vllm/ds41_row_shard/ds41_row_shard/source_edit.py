"""Replace one function's code with an exact, line-preserving edit of its source.

The edit is applied to the source text of the loaded module file, after that
file's SHA-256 has been checked against a pin. Compiling the unedited module
must reproduce the loaded function code, so the edit applies to the analysed
implementation in its original class and module context.
Tracebacks keep pointing at the image file's real line numbers.
"""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Mapping
from inspect import unwrap
from types import CodeType, FunctionType, ModuleType


class PatchRefused(RuntimeError):
    """The loaded source differs from the source this plugin was written against."""


def file_sha256(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _find(tree: ast.Module, qualname: str, decorator: str | None) -> ast.FunctionDef:
    parts = qualname.split(".")
    body = tree.body
    node = None
    for part in parts:
        node = next(
            (
                item
                for item in body
                if isinstance(item, (ast.FunctionDef, ast.ClassDef))
                and item.name == part
            ),
            None,
        )
        if node is None:
            raise PatchRefused(f"{qualname}: {part} not found")
        body = node.body
    if not isinstance(node, ast.FunctionDef):
        raise PatchRefused(f"{qualname} is not a function")
    if decorator is None and node.decorator_list:
        raise PatchRefused(f"{qualname} is decorated; edit refused")
    if decorator is not None and (
        len(node.decorator_list) != 1
        or not isinstance(node.decorator_list[0], ast.Name)
        or node.decorator_list[0].id != decorator
    ):
        raise PatchRefused(f"{qualname}: expected the {decorator} decorator")
    return node


def _compile_function(source: str, qualname: str, filename: str) -> CodeType:
    # Class scopes and future imports affect compilation even when we never
    # execute the surrounding module.
    code = compile(source, filename, "exec", dont_inherit=True)
    for name in qualname.split("."):
        candidates = [
            value for value in code.co_consts
            if isinstance(value, CodeType) and value.co_name == name
        ]
        if len(candidates) != 1:
            raise PatchRefused(f"{qualname}: expected one code object for {name}")
        code = candidates[0]
    return code


def _same_code(a: CodeType, b: CodeType) -> bool:
    if (
        a.co_code != b.co_code
        or a.co_names != b.co_names
        or a.co_varnames != b.co_varnames
        or a.co_freevars != b.co_freevars
        or a.co_firstlineno != b.co_firstlineno
        or len(a.co_consts) != len(b.co_consts)
    ):
        return False
    for x, y in zip(a.co_consts, b.co_consts):
        if isinstance(x, CodeType) and isinstance(y, CodeType):
            if not _same_code(x, y):
                return False
        elif type(x) is not type(y) or x != y:
            return False
    return True


def edit_function(
    module: ModuleType,
    qualname: str,
    sha256: str,
    replacements: tuple[tuple[str, str], ...],
    helpers: Mapping[str, object],
    marker: str,
    *,
    decorator: str | None = None,
) -> bool:
    """Install the edited code on ``module.<qualname>``; return False if already done.

    Each ``(old, new)`` must occur exactly once in the function source and
    keep its line count. ``helpers`` are placed in the module namespace; their
    names must not already exist there with a different value.
    """
    path = module.__file__
    if path is None:
        raise PatchRefused(f"{module.__name__} has no source file")
    actual = file_sha256(path)
    if actual != sha256:
        raise PatchRefused(
            f"{module.__name__} ({path}) has SHA-256 {actual}, expected {sha256}"
        )
    owner: object = module
    *outer, name = qualname.split(".")
    for part in outer:
        owner = getattr(owner, part)
    function = owner.__dict__[name]
    if decorator is not None:
        function = unwrap(function)
    if not isinstance(function, FunctionType):
        raise PatchRefused(f"{qualname} is not a plain function")
    if getattr(function, marker, False):
        return False
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    node = _find(ast.parse(text), qualname, decorator)
    lines = text.splitlines(keepends=True)
    start, end = node.lineno - 1, node.end_lineno
    source = "".join(lines[start:end])
    original = _compile_function(text, qualname, path)
    if not _same_code(original, function.__code__):
        raise PatchRefused(f"{qualname}: loaded code differs from its source file")
    edited = source
    for old, new in replacements:
        if edited.count(old) != 1:
            raise PatchRefused(f"{qualname}: edit anchor found {edited.count(old)} times")
        if old.count("\n") != new.count("\n"):
            raise PatchRefused(f"{qualname}: edit changes the line count")
        edited = edited.replace(old, new)
    for helper, value in helpers.items():
        existing = module.__dict__.get(helper, value)
        if existing is not value:
            raise PatchRefused(f"{module.__name__}.{helper} already exists")
    code = _compile_function("".join(lines[:start]) + edited + "".join(lines[end:]), qualname, path)
    for helper, value in helpers.items():
        setattr(module, helper, value)
    function.__code__ = code
    setattr(function, marker, True)
    return True
