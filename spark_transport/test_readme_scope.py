from pathlib import Path


README = Path(__file__).with_name("README.md")


def test_transport_readme_describes_a_model_independent_subsystem() -> None:
    text = README.read_text(encoding="utf-8")
    assert "model-independent collective transport" in text
    assert "It does not define a model" in text
    assert "Model profiles" in text


def test_transport_identity_does_not_use_model_names_or_profile_geometry() -> None:
    text = README.read_text(encoding="utf-8")
    for hidden_context_term in (
        "GLM-5.2",
        "GLM-5.3",
        "DeepSeek",
        "Qwen",
        "exact-Q40",
        "width 6,144",
        "width-4,096",
    ):
        assert hidden_context_term not in text
