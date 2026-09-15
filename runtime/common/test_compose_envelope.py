"""Real Compose resolution preserves optional managed-serving envelope settings."""
import pytest

from runtime.common import compose
from runtime.common.container_spec import ContainerSpec, docker_create
from runtime.common.test_compose import compose_cli as compose_cli


@pytest.mark.parametrize("mode,command", [
    ("inherit", ()), ("disabled", ()),
    ("shell", ('python3 -S -c "print(1)"',)), ("exec", ("python3", "-S", "-c", "print(1)")),
])
def test_managed_envelope_resolves_without_qwen_memory_limits(compose_cli, mode, command):
    spec = ContainerSpec(name="glm-envelope", image_id="sha256:" + "a" * 64,
                         entrypoint=("/opt/venv/bin/python",), command=("/serve.py",),
                         environment={}, mounts=(), memory=None, memory_swap=None,
                         shm_size=32 * 1024**3, cap_add=("IPC_LOCK",),
                         security_opt=("label=disable",), user="1000:1000", working_dir="/srv/app",
                         health_mode=mode, health_command=command,
                         health_timeout=6, health_start_period=1800)
    image = "example.invalid/runtime@sha256:" + "a" * 64
    resolved = compose.check_equivalence(spec, image, compose.compose_text(spec, image))["services"]["model"]
    assert "mem_limit" not in resolved and "memswap_limit" not in resolved
    assert resolved["shm_size"] == 32 * 1024**3
    assert resolved["cap_add"] == ["IPC_LOCK"]
    assert resolved["security_opt"] == ["label:disable"]
    assert resolved["user"] == "1000:1000" and resolved["working_dir"] == "/srv/app"
    docker = docker_create(spec)
    assert "--memory" not in docker and "--memory-swap" not in docker
    if mode == "inherit":
        assert "healthcheck" not in resolved and "--no-healthcheck" not in docker
    elif mode == "disabled":
        assert resolved["healthcheck"] == {"disable": True}
    else:
        assert resolved["healthcheck"]["start_period"] == "1800s"
        assert resolved["healthcheck"]["test"] == ["CMD-SHELL" if mode == "shell" else "CMD", *command]
