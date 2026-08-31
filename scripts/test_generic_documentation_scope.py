from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_IDENTITIES = ("GLM-5", "DeepSeek", "Qwen", "DFlash")


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_repository_landing_page_routes_models_to_profiles() -> None:
    readme = _text("README.md")
    assert "docs/profiles/README.md" in readme
    assert not any(identity in readme for identity in MODEL_IDENTITIES)


def test_generic_subsystem_pages_do_not_define_model_geometry() -> None:
    generic_pages = (
        "docs/ARCHITECTURE.md",
        "docs/PREREQUISITES.md",
        "docs/SIRCL.md",
        "spark_transport/CABLE_QUALIFICATION.md",
        "spark_transport/integrations/vllm/README.md",
        "spark_transport/nccl/README.md",
    )
    for path in generic_pages:
        text = _text(path)
        assert not any(identity in text for identity in MODEL_IDENTITIES), path


def test_indexes_keep_model_names_inside_labeled_routing_sections() -> None:
    runtime = _text("runtime/README.md")
    config = _text("scripts/config/README.md")
    assert not any(
        identity in runtime.split("## Runtime index", 1)[0]
        for identity in MODEL_IDENTITIES
    )
    assert not any(
        identity in config.split("## Profile template index", 1)[0]
        for identity in MODEL_IDENTITIES
    )
