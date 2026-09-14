"""Docker envelope omission and literal health-command rendering."""

import copy
from dataclasses import replace
import shlex

import pytest

from runtime.common.container_spec import (
    Bind,
    ContainerSpec,
    docker_create,
    expected_inspection,
)


@pytest.fixture
def spec():
    return ContainerSpec(
        name="fixture",
        image_id="sha256:" + "a" * 64,
        entrypoint=("/opt/venv/bin/python",),
        command=("serve.py", "serve"),
        environment={},
        mounts=(),
    )


def test_defaults_preserve_existing_docker_argument_order_and_values(spec):
    assert docker_create(spec) == [
        "docker",
        "create",
        "--name",
        "fixture",
        "--entrypoint",
        "/opt/venv/bin/python",
        "--platform",
        "linux/arm64",
        "--pull",
        "never",
        "--restart",
        "no",
        "--gpus",
        "all",
        "--network",
        "host",
        "--ipc",
        "host",
        "--ulimit",
        "memlock=-1:-1",
        "--memory",
        str(108 * 1024**3),
        "--memory-swap",
        str(112 * 1024**3),
        "--init",
        "--no-healthcheck",
        "--device",
        "/dev/infiniband",
        spec.image_id,
        "serve.py",
        "serve",
    ]
    assert spec.effective_health_mode == "disabled"


def test_unset_limits_and_health_are_omitted_without_losing_image_command(spec):
    spec = replace(spec, memory=None, memory_swap=None, health_mode="inherit")
    argv = docker_create(spec)
    for flag in (
        "--memory",
        "--memory-swap",
        "--shm-size",
        "--user",
        "--workdir",
        "--no-healthcheck",
    ):
        assert flag not in argv
    assert not any(flag.startswith("--health-") for flag in argv)
    assert argv[-3:] == [spec.image_id, "serve.py", "serve"]
    assert spec.document()["health_mode"] == "inherit"
    assert spec.document()["memory"] is None


@pytest.mark.parametrize("swap", [None, 0, -1])
def test_swap_can_be_omitted_implicit_or_unlimited(spec, swap):
    argv = docker_create(replace(spec, memory_swap=swap))
    assert ("--memory-swap" in argv) == (swap is not None)
    if swap is not None:
        assert argv[argv.index("--memory-swap") + 1] == str(swap)


def test_capabilities_security_and_identity_are_literal_arguments(spec):
    values = {
        "shm_size": 32 * 1024**3,
        "cap_add": ("IPC_LOCK", "SYS_NICE"),
        "security_opt": ("label=disable", "no-new-privileges=true"),
        "user": "1000:1000",
        "working_dir": "/srv/model files/$UNCHANGED",
    }
    spec = replace(spec, **values)
    argv = docker_create(spec)
    for flag, expected in (
        ("--shm-size", str(values["shm_size"])),
        ("--user", values["user"]),
        ("--workdir", values["working_dir"]),
    ):
        assert argv[argv.index(flag) + 1] == expected
    assert [
        argv[index + 1] for index, value in enumerate(argv) if value == "--cap-add"
    ] == list(values["cap_add"])
    assert [
        argv[index + 1] for index, value in enumerate(argv) if value == "--security-opt"
    ] == list(values["security_opt"])
    assert {name: spec.document()[name] for name in values} == values


@pytest.mark.parametrize("mode", ["auto", "exec"])
def test_exec_health_preserves_argument_boundaries_and_literal_shell_characters(
    spec, mode
):
    command = ("/opt/python", "-c", 'print("$VALUE; $(touch never)")', "", "a'b")
    spec = replace(spec, health_mode=mode, health_command=command)
    argv = docker_create(spec)
    assert shlex.split(argv[argv.index("--health-cmd") + 1]) == list(command)
    assert argv[argv.index("--health-cmd") + 1] == shlex.join(command)
    assert spec.effective_health_mode == "exec"
    assert "--no-healthcheck" not in argv
    assert argv[argv.index("--health-cmd") + 2 : argv.index("--device")] == [
        "--health-interval",
        "10s",
        "--health-timeout",
        "5s",
        "--health-start-period",
        "900s",
        "--health-retries",
        "3",
    ]


def test_shell_health_preserves_one_command_without_quoting_it_again(spec):
    command = 'test "$READY" = yes && curl -fsS "http://127.0.0.1:$PORT/health"'
    spec = replace(
        spec,
        health_mode="shell",
        health_command=(command,),
        health_timeout=6,
        health_start_period=1800,
    )
    argv = docker_create(spec)
    assert argv[argv.index("--health-cmd") + 1] == command
    assert argv[argv.index("--health-timeout") + 1] == "6s"
    assert argv[argv.index("--health-start-period") + 1] == "1800s"
    assert spec.effective_health_mode == "shell"


def test_disabled_health_is_explicit_and_omits_timing_options(spec):
    argv = docker_create(replace(spec, health_mode="disabled"))
    assert "--no-healthcheck" in argv
    assert not any(value.startswith("--health-") for value in argv)


@pytest.mark.parametrize(
    "changes",
    [
        {"health_mode": "unsupported"},
        {"health_mode": "exec"},
        {"health_mode": "shell"},
        {"health_mode": "shell", "health_command": ("curl", "/health")},
        {"health_mode": "inherit", "health_command": ("true",)},
        {"health_mode": "disabled", "health_command": ("true",)},
        {"health_command": ("",)},
        {"health_command": ("curl\0",)},
        {"health_command": ["true"]},
        {"health_interval": 0},
        {"health_timeout": True},
        {"health_start_period": -1},
        {"health_retries": 0},
    ],
)
def test_ambiguous_or_invalid_health_policy_is_rejected(spec, changes):
    with pytest.raises(ValueError):
        replace(spec, **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"memory": True},
        {"memory": -1},
        {"memory_swap": -2},
        {"memory": None},
        {"memory": 0},
        {"memory_swap": 1},
        {"shm_size": "32g"},
        {"shm_size": 0},
        {"shm_size": True},
        {"cap_add": ("IPC_LOCK", "IPC_LOCK")},
        {"cap_add": "IPC_LOCK"},
        {"security_opt": ("",)},
        {"security_opt": ("label=disable\0",)},
        {"user": ""},
        {"user": 1000},
        {"user": "user\0"},
        {"working_dir": "relative"},
        {"working_dir": "/srv/../model"},
        {"working_dir": "/srv//model"},
        {"working_dir": "/srv\\model"},
    ],
)
def test_invalid_envelope_settings_fail_before_rendering(spec, changes):
    with pytest.raises(ValueError):
        replace(spec, **changes)


@pytest.fixture
def image(spec):
    return {
        "Id": spec.image_id,
        "Os": "linux",
        "Architecture": "arm64",
        "Variant": "v8",
        "Config": {
            "Entrypoint": ["/image/entrypoint"],
            "Cmd": ["image-command"],
            "Env": ["PATH=/image/bin", "SETTING=image", "EMPTY=", "EQUALS=a=b"],
            "Labels": {"source": "image", "setting": "image"},
            "WorkingDir": "/image/work",
            "User": "image-user",
            "Healthcheck": {
                "Test": ["CMD", "image-health"],
                "Interval": 22_000_000_000,
                "Timeout": 8_000_000_000,
                "StartPeriod": 50_000_000_000,
                "StartInterval": 2_000_000_000,
                "Retries": 7,
            },
        },
    }


def test_inspection_merges_image_defaults_without_mutating_any_input(spec, image):
    spec = replace(
        spec,
        health_mode="inherit",
        environment={"SETTING": "explicit", "ADDED": "$literal"},
        labels={"setting": "explicit"},
        mounts=(
            Bind("/models/source", "/models/target", True),
            Bind("/cache/source", "/cache"),
        ),
    )
    before = copy.deepcopy(image)
    result = expected_inspection(spec, image)
    assert result["name"] == spec.name and result["image"] == spec.image_id
    assert result["cmd"] == list(spec.command)
    assert result["entrypoint"] == list(spec.entrypoint)
    assert result["env"] == {
        "PATH": "/image/bin",
        "SETTING": "explicit",
        "EMPTY": "",
        "EQUALS": "a=b",
        "ADDED": "$literal",
    }
    assert result["labels"] == {"source": "image", "setting": "explicit"}
    assert result["working_dir"] == "/image/work" and result["user"] == "image-user"
    assert result["mounts"] == {
        "/models/target": {"Source": "/models/source", "Type": "bind", "RW": False},
        "/cache": {"Source": "/cache/source", "Type": "bind", "RW": True},
    }
    assert result["healthcheck"] == image["Config"]["Healthcheck"]
    result["healthcheck"]["Test"].append("result-only")
    result["env"]["SETTING"] = "result-only"
    result["labels"]["setting"] = "result-only"
    assert image == before
    assert (
        spec.environment["SETTING"] == "explicit"
        and spec.labels["setting"] == "explicit"
    )


@pytest.mark.parametrize("backend", ["docker", "compose"])
def test_inspection_describes_the_selected_backend_command_and_labels(
    spec, image, backend
):
    spec = replace(
        spec,
        entrypoint=("/python", "-B", "wrapper.py"),
        health_mode="exec",
        health_command=("/python", "-c", "print('$literal')"),
    )
    result = expected_inspection(spec, image, backend=backend)
    if backend == "docker":
        assert result["entrypoint"] == ["/python"]
        assert result["cmd"] == ["-B", "wrapper.py", *spec.command]
        assert result["healthcheck"]["Test"] == [
            "CMD-SHELL",
            shlex.join(spec.health_command),
        ]
        assert "com.docker.compose.project" not in result["labels"]
    else:
        assert result["entrypoint"] == list(spec.entrypoint)
        assert result["cmd"] == list(spec.command)
        assert result["healthcheck"]["Test"] == ["CMD", *spec.health_command]
        assert result["labels"]["com.docker.compose.project"] == spec.name
        assert result["labels"]["com.docker.compose.service"] == "model"
    assert result["host_config"]["DeviceRequests"] == [
        {
            "Driver": "nvidia" if backend == "compose" else "",
            "Count": -1,
            "Capabilities": [["gpu"]],
        }
    ]


@pytest.mark.parametrize("backend", ["docker", "compose"])
def test_literal_shell_health_overrides_test_and_selected_timers(spec, image, backend):
    command = 'test "$READY" = yes && curl -fsS "$URL"'
    spec = replace(
        spec,
        health_mode="shell",
        health_command=(command,),
        health_timeout=6,
        health_start_period=1800,
        user="1000:1000",
        working_dir="/srv/work",
    )
    result = expected_inspection(spec, image, backend=backend)
    assert result["healthcheck"] == {
        "Test": ["CMD-SHELL", command],
        "Interval": 10_000_000_000,
        "Timeout": 6_000_000_000,
        "StartPeriod": 1_800_000_000_000,
        "StartInterval": 2_000_000_000,
        "Retries": 3,
    }
    assert result["user"] == "1000:1000" and result["working_dir"] == "/srv/work"


@pytest.mark.parametrize("mode", ["auto", "disabled"])
def test_disabled_health_retains_image_timers_but_never_the_image_test(
    spec, image, mode
):
    result = expected_inspection(replace(spec, health_mode=mode), image)
    assert result["healthcheck"] == dict(image["Config"]["Healthcheck"], Test=["NONE"])


def test_zero_start_period_retains_image_value_or_remains_absent(spec, image):
    spec = replace(
        spec, health_mode="exec", health_command=("true",), health_start_period=0
    )
    assert (
        expected_inspection(spec, image)["healthcheck"]["StartPeriod"] == 50_000_000_000
    )
    assert "StartPeriod" not in expected_inspection(spec, {})["healthcheck"]


def test_inspection_has_only_known_host_envelope_expectations(spec, image):
    spec = replace(
        spec,
        memory=None,
        memory_swap=None,
        shm_size=32 * 1024**3,
        cap_add=("IPC_LOCK",),
        security_opt=("label=disable",),
        gpu_count=1,
    )
    assert expected_inspection(spec, image)["host_config"] == {
        "NetworkMode": "host",
        "IpcMode": "host",
        "Memory": 0,
        "MemorySwap": 0,
        "RestartPolicy": {"Name": "no"},
        "Privileged": False,
        "Init": True,
        "ShmSize": 32 * 1024**3,
        "Ulimits": [{"Name": "memlock", "Soft": -1, "Hard": -1}],
        "CapAdd": ["IPC_LOCK"],
        "SecurityOpt": ["label=disable"],
        "Devices": [
            {
                "PathOnHost": "/dev/infiniband",
                "PathInContainer": "/dev/infiniband",
                "CgroupPermissions": "rwm",
            }
        ],
        "DeviceRequests": [{"Driver": "", "Count": 1, "Capabilities": [["gpu"]]}],
    }


def test_unspecified_shm_and_docker_init_do_not_claim_daemon_defaults(spec):
    spec = replace(spec, init=False)
    docker = expected_inspection(spec, {})["host_config"]
    compose = expected_inspection(spec, {}, backend="compose")["host_config"]
    assert "ShmSize" not in docker and "ShmSize" not in compose
    assert "Init" not in docker and compose["Init"] is False
    assert docker["Memory"] == spec.memory and docker["MemorySwap"] == spec.memory_swap


def test_missing_optional_image_metadata_has_explicit_empty_defaults(spec):
    result = expected_inspection(replace(spec, health_mode="inherit"), {})
    assert result["healthcheck"] is None
    assert result["env"] == {} and result["labels"] == {}
    assert result["user"] == "" and result["working_dir"] == ""


@pytest.mark.parametrize(
    "field,value",
    [
        ("Id", "sha256:" + "b" * 64),
        ("Os", "windows"),
        ("Architecture", "amd64"),
    ],
)
def test_inspection_rejects_image_identity_or_platform_drift(spec, image, field, value):
    image[field] = value
    with pytest.raises(ValueError, match="differs from the container specification"):
        expected_inspection(spec, image)


def test_explicit_platform_variant_is_checked_when_available(spec, image):
    spec = replace(spec, platform="linux/arm64/v9")
    with pytest.raises(ValueError, match="platform differs"):
        expected_inspection(spec, image)


@pytest.mark.parametrize(
    "change",
    [
        {"Volumes": {"/image/data": {}}},
        {"Env": ["INHERITED"]},
        {"Env": ["DUP=one", "DUP=two"]},
        {"Env": ["BAD NAME=value"]},
        {"Labels": []},
        {"Healthcheck": "curl health"},
        {"User": 1000},
    ],
)
def test_inspection_rejects_ambiguous_image_defaults(spec, image, change):
    image["Config"].update(change)
    with pytest.raises(ValueError):
        expected_inspection(spec, image)


def test_inspection_rejects_aliasing_mount_destinations(spec, image):
    spec = replace(
        spec,
        mounts=(Bind("/source/a", "/target"), Bind("/source/b", "/other/../target")),
    )
    with pytest.raises(ValueError, match="duplicate mount destinations"):
        expected_inspection(spec, image)


def test_compose_metadata_cannot_claim_a_different_explicit_project(spec, image):
    spec = replace(spec, labels={"com.docker.compose.project": "another-project"})
    with pytest.raises(ValueError, match="conflict with the Compose project"):
        expected_inspection(spec, image, backend="compose")


def test_inspection_requires_an_explicit_supported_backend(spec, image):
    with pytest.raises(ValueError, match="backend must be"):
        expected_inspection(spec, image, backend="unknown")
