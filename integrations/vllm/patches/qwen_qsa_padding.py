"""Clear inactive Qwen sparse-attention request IDs retained by CUDA graphs."""

import ast


def apply(source: str) -> str:
    """Preserve live/shared mappings and reject an unfamiliar metadata builder."""
    tree = ast.parse(source)
    builders = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Qwen4ExpQSAMetadataBuilder"
    ]
    methods = [
        node
        for cls in builders
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "build"
    ]
    if len(builders) != 1 or len(methods) != 1:
        raise ValueError("Expected one Qwen sparse-attention metadata builder")
    method = methods[0]
    needle = (
        "        request_ids = cm.token_to_req_indices(self._request_ids)\n"
        "        num_mapped_tokens = int(cm.query_start_loc_cpu[-1])\n"
        "        if num_mapped_tokens < cm.num_actual_tokens:\n"
        "            request_ids[num_mapped_tokens:].fill_(-1)\n"
    )
    lines = source.splitlines(keepends=True)
    body = "".join(lines[method.lineno - 1 : method.end_lineno])
    if body.count(needle) != 1 or "inactive_capacity_start" in body:
        raise ValueError("Qwen request-index mapping boundary changed")
    insertion = (
        "        # Captured views can exceed the next batch's live metadata slice.\n"
        "        # Clear the owned inactive capacity without overwriting a cached\n"
        "        # live mapping shared by attention metadata builders.\n"
        "        inactive_capacity_start = max(num_mapped_tokens, cm.num_actual_tokens)\n"
        "        self._request_ids[inactive_capacity_start:].fill_(-1)\n"
    )
    replacement = needle.replace(
        "        if num_mapped_tokens < cm.num_actual_tokens:\n",
        insertion + "        if num_mapped_tokens < cm.num_actual_tokens:\n",
    )
    result = (
        "".join(lines[: method.lineno - 1])
        + body.replace(needle, replacement, 1)
        + "".join(lines[method.end_lineno :])
    )
    ast.parse(result)
    return result
