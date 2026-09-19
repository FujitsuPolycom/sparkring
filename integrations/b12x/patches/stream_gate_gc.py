"""Defer automatic cyclic finalizers while B12X autotuning gates CUDA work."""

import ast


_HOLD = """    @contextmanager
    def hold(self, stream):
        self.streams[stream.cuda_stream] = stream
        self.sequence = (self.sequence + 1) & 0xFFFFFFFF
        target = self.sequence
        self._check(self.driver.cuStreamWaitValue32(
            stream.cuda_stream, self.device_pointer, target,
            int(self.driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ),
        ))
        try:
            yield
        finally:
            # A later release must also satisfy an earlier, still queued wait.
            self.flag.value = target
"""

_GUARD = '''# Automatic collection may unload a CUDA library and wait for queued work.
# Every gate must release its device wait before the last holder restores GC.
_gc_deferral_lock = threading.RLock()
_gc_deferral_count = 0
_gc_deferral_restore = False


@contextmanager
def _defer_automatic_gc():
    """Suspend automatic cyclic GC, not explicit collection or refcount finalizers."""
    global _gc_deferral_count, _gc_deferral_restore
    with _gc_deferral_lock:
        if _gc_deferral_count == 0:
            _gc_deferral_restore = gc.isenabled()
            gc.disable()
        _gc_deferral_count += 1
    try:
        yield
    finally:
        with _gc_deferral_lock:
            _gc_deferral_count -= 1
            if _gc_deferral_count == 0 and _gc_deferral_restore:
                gc.enable()


'''

_GUARDED_HOLD = """    @contextmanager
    def hold(self, stream):
        with _defer_automatic_gc():
            self.streams[stream.cuda_stream] = stream
            self.sequence = (self.sequence + 1) & 0xFFFFFFFF
            target = self.sequence
            try:
                self._check(self.driver.cuStreamWaitValue32(
                    stream.cuda_stream, self.device_pointer, target,
                    int(self.driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ),
                ))
                yield
            finally:
                # Release queued work before automatic finalizers may resume.
                self.flag.value = target
"""


def apply(source: str) -> str:
    """Patch the known stream-gate method; reject source drift and repeat application."""
    tree = ast.parse(source)
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_StreamGate"
    ]
    if (
        len(classes) != 1
        or classes[0].decorator_list
        or "_defer_automatic_gc" in source
    ):
        raise ValueError("Expected one unpatched B12X autotuning stream gate")
    gate = classes[0]
    methods = [
        node
        for node in gate.body
        if isinstance(node, ast.FunctionDef) and node.name == "hold"
    ]
    if len(methods) != 1 or source.count(_HOLD) != 1:
        raise ValueError("B12X autotuning stream-gate boundary changed")
    method = methods[0]
    lines = source.splitlines(keepends=True)
    start = min([method.lineno, *[item.lineno for item in method.decorator_list]]) - 1
    if "".join(lines[start : method.end_lineno]) != _HOLD:
        raise ValueError("B12X autotuning stream-gate method changed")
    # Keep module docstrings and __future__ imports before standard imports.
    import_start = 0
    for node in tree.body:
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            import_start = node.end_lineno
        elif isinstance(node, ast.ImportFrom) and node.module == "__future__":
            import_start = node.end_lineno
        else:
            break
    changes = [
        (start, method.end_lineno, _GUARDED_HOLD),
        (gate.lineno - 1, gate.lineno - 1, _GUARD),
        (import_start, import_start, "import gc\nimport threading\n"),
    ]
    for first, last, replacement in sorted(
        changes, key=lambda item: item[0], reverse=True
    ):
        lines[first:last] = [replacement]
    result = "".join(lines)
    ast.parse(result)
    return result
