"""Route Qwen multimodal execution through the language-model HC dispatcher."""

import ast


def apply_api_audit(source: str) -> str:
    """Install the audit after engine initialization and before HTTP serving."""
    needle = (
        "    await init_app_state(engine_client, app.state, args, supported_tasks)\n"
    )
    if source.count(needle) != 1 or "sparkring_startup_audit" in source:
        raise ValueError("Unexpected API initialization boundary")
    insertion = (
        needle + "\n    from .sparkring_startup_audit import emit\n"
        "\n    emit(engine_client.vllm_config, logger)\n"
    )
    result = source.replace(needle, insertion, 1)
    ast.parse(result)
    return result


def apply(source: str) -> str:
    """Patch only the known multimodal inner-model call; reject structural drift."""
    tree = ast.parse(source)
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "Qwen4ExpForConditionalGeneration"
    ]
    if len(classes) != 1:
        raise ValueError("Expected one Qwen multimodal model class")
    forwards = [
        node
        for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    ]
    if len(forwards) != 1:
        raise ValueError("Expected one multimodal forward method")
    calls = [
        node
        for node in ast.walk(forwards[0])
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "self.language_model.model"
    ]
    if len(calls) != 1:
        raise ValueError("Expected one direct language-model bypass")
    call = calls[0]
    expected = {
        "input_ids",
        "positions",
        "intermediate_tensors",
        "inputs_embeds",
        "query_start_loc",
        "ngram_context",
        "deepstack_input_embeds",
    }
    if call.args or {arg.arg for arg in call.keywords} != expected:
        raise ValueError("Multimodal forwarding arguments changed")
    lines = source.splitlines(keepends=True)
    func = call.func
    if func.lineno != func.end_lineno:
        raise ValueError("Unexpected multiline call target")
    line = lines[func.lineno - 1].encode("utf-8")
    if line[func.col_offset : func.end_col_offset] != b"self.language_model.model":
        raise ValueError("Call source span differs")
    lines[func.lineno - 1] = (
        line[: func.col_offset] + b"self.language_model" + line[func.end_col_offset :]
    ).decode("utf-8")
    result = "".join(lines)
    ast.parse(result)
    return result
