"""GPU-free regression tests for repository release hardening."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = (
    ROOT / ".github" / "workflows" / "ci.yml",
    ROOT / ".github" / "workflows" / "publish.yml",
)
RELEASING = ROOT / "RELEASING.md"
ACTION_USE = re.compile(r"^\s*-\s+uses:\s+([^#\s]+)", re.MULTILINE)


def test_external_github_actions_are_pinned_to_commit_shas() -> None:
    for workflow_path in WORKFLOWS:
        workflow = workflow_path.read_text(encoding="utf-8")
        action_refs = ACTION_USE.findall(workflow)

        assert action_refs
        for action_ref in action_refs:
            if action_ref.startswith("./"):
                continue
            _, separator, revision = action_ref.rpartition("@")
            assert separator == "@", action_ref
            assert re.fullmatch(r"[0-9a-f]{40}", revision), action_ref


def test_post_publication_import_probe_uses_isolated_mode() -> None:
    instructions = RELEASING.read_text(encoding="utf-8")

    assert (
        'sparkcache-release-check/bin/python -I -c "import sparkcache; '
        'print(sparkcache.__version__, sparkcache.__file__)"'
    ) in instructions
