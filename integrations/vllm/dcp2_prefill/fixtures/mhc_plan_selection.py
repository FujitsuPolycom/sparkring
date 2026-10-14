# SPDX-License-Identifier: Apache-2.0
# Unmodified method excerpt from image d52737a109e0; enclosing indentation removed.
# ruff: noqa: F821 -- Plan is a name of the image module this method comes from; test_dcp2_prefill_patch.py supplies it.
def _plan_for(self, operation: str, tokens: int) -> Plan:
    tokens = int(tokens)
    exact = self._plans.get((operation, tokens))
    if exact is not None:
        return exact
    capacity = max((rows for op, rows in self._plans if op == operation), default=0)
    if 0 <= tokens <= capacity and capacity:
        return self._plans[(operation, capacity)]
    raise RuntimeError(
        f"B12x mHC {operation} live M={tokens} exceeds prepared capacity {capacity}"
    )
