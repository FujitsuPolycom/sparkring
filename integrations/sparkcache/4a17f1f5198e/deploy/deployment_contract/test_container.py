from __future__ import annotations

import pytest

from deploy.deployment_contract import build_container_command


def _inspection(environment: list[str]) -> dict:
    return {
        "Config": {"Env": environment, "Cmd": ["serve", "/model"]},
        "HostConfig": {},
        "Mounts": [],
    }


def _environment_arguments(command: list[str]) -> list[str]:
    return [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--env"
    ]


def test_container_command_materializes_explicit_environment_removal() -> None:
    command = build_container_command(
        _inspection(
            [
                "MODEL_REPOSITORY=inherited",
                "SPARKRING_EXPLICITLY_UNSET=MODEL_REPOSITORY,MODEL_REVISION",
            ]
        ),
        "image",
        "name",
        "a" * 64,
        create_only=True,
    )

    environment = _environment_arguments(command)
    assert "MODEL_REPOSITORY=inherited" in environment
    assert "MODEL_REPOSITORY=" in environment
    assert "MODEL_REVISION=" in environment
    assert environment.index("MODEL_REPOSITORY=") > environment.index(
        "MODEL_REPOSITORY=inherited"
    )


def test_container_command_rejects_an_invalid_explicit_unset_name() -> None:
    with pytest.raises(ValueError, match="invalid environment names"):
        build_container_command(
            _inspection(["SPARKRING_EXPLICITLY_UNSET=INVALID-NAME"]),
            "image",
            "name",
            "a" * 64,
            create_only=True,
        )
