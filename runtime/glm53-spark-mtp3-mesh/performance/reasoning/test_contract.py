import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from patch_contract import patched

import gzip

SOURCE = gzip.decompress(
    (Path(__file__).parent / "chat-serving-source.py.gz").read_bytes()
).decode()


def request_path(source):
    tree = ast.parse(source)
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "_create_chat_completion"
    )
    # Execute the actual serving prefix through parser construction. A renderer
    # sentinel identifies whether an unsupported request reaches model work.
    stop = next(
        i
        for i, node in enumerate(function.body)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Await)
    )
    body = function.body[:stop] + ast.parse("return 'rendering'").body
    wrapper = ast.parse("async def invoke(self, request):\n    pass").body[0]
    wrapper.body = body
    scope = {"Parser": object}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])),
            "<serving-prefix>",
            "exec",
        ),
        scope,
    )
    return scope["invoke"]


@pytest.mark.parametrize(
    "kwargs,effort",
    [({"enable_thinking": False}, None), ({"thinking": False}, None),
     ({"enable_thinking": False}, "low"), ({}, "none")],
)
def test_glm53_rejects_unsupported_mode_before_parser_or_rendering(
    kwargs, effort
):
    server = SimpleNamespace(
        renderer=SimpleNamespace(tokenizer=object()),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="glm5_next")),
        _effective_chat_template_kwargs=lambda request: kwargs,
        parser_cls=None,
        create_error_response=lambda message: {"error": message},
    )
    # The guard executes before stream-specific rendering; no stream attribute
    # is supplied so this test cannot silently depend on a selected stream mode.
    request = SimpleNamespace(reasoning_effort=effort)
    assert asyncio.run(request_path(SOURCE)(server, request)) == "rendering"
    result = asyncio.run(request_path(patched(SOURCE))(server, request))
    assert "does not support disabling thinking" in result["error"]
    assert "Remove false enable_thinking/thinking" in result["error"]


@pytest.mark.parametrize(
    "model,kwargs,effort",
    [
        ("glm5_next", {}, None),
        ("glm5_next", {}, "low"),
        ("glm5_next", {}, "high"),
        ("glm5_next", {}, "max"),
        ("glm5_next", {"enable_thinking": True}, None),
        ("glm4_moe", {"enable_thinking": False}, "none"),
    ],
)
def test_supported_modes_and_other_models_preserve_rendering(model, kwargs, effort):
    server = SimpleNamespace(
        renderer=SimpleNamespace(tokenizer=object()),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model)),
        _effective_chat_template_kwargs=lambda request: kwargs,
        parser_cls=None,
        create_error_response=lambda message: {"error": message},
    )
    assert (
        asyncio.run(
            request_path(patched(SOURCE))(
                server, SimpleNamespace(reasoning_effort=effort)
            )
        )
        == "rendering"
    )
