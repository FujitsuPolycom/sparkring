"""CPU affinity survives Docker, Compose and inspection projection."""

from dataclasses import replace

import pytest

from runtime.common import compose
from runtime.common.container_spec import (
    ContainerSpec,
    docker_create,
    expected_inspection,
)


def test_cpu_set_is_preserved_by_both_renderers():
    spec = ContainerSpec(
        name="cpu-affinity",
        image_id="sha256:" + "a" * 64,
        entrypoint=("python",),
        command=(),
        environment={},
        mounts=(),
        cpuset_cpus="0-7,12",
    )
    argv = docker_create(spec)
    assert argv[argv.index("--cpuset-cpus") + 1] == "0-7,12"
    assert compose.service(spec, spec.image_id)["cpuset"] == "0-7,12"
    assert expected_inspection(spec, {})["host_config"]["CpusetCpus"] == "0-7,12"
    plain = replace(spec, cpuset_cpus=None)
    assert "--cpuset-cpus" not in docker_create(plain)
    assert "cpuset" not in compose.service(plain, plain.image_id)
    assert "cpuset_cpus" not in plain.document()


@pytest.mark.parametrize("value", ["", True, "7-0", "0;id", "0 - 3", "0,,1"])
def test_ambiguous_cpu_set_is_rejected(value):
    with pytest.raises(ValueError, match="cpuset_cpus"):
        ContainerSpec(
            name="cpu-affinity",
            image_id="sha256:" + "a" * 64,
            entrypoint=("python",),
            command=(),
            environment={},
            mounts=(),
            cpuset_cpus=value,
        )
