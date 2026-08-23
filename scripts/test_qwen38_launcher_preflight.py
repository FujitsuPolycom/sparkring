"""Offline contracts for the Qwen four-rank local preflight."""

from __future__ import annotations

import os
import shlex
import subprocess
import textwrap
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "qwen38_dgx4_serve.sh"

VLLM_COMMIT = "229effc810ee6b8112f661472f6aace4eb8c787d"
EXLLAMAV3_COMMIT = "5f3c537ca9d89893d771256f5c43c93656553fbb"

EXLLAMAV3_FILES = {
    "exllamav3/exllamav3_ext/avx2_target.cpp": (
        "b26342bc6cb300587e5ed4ff77d75c21debfe034ee40634635ec455280ae6e8c"
    ),
    "exllamav3/exllamav3_ext/avx512_target.cpp": (
        "9ba59543263693598713de192028627c8a249cc62f6311c682f64fbb0d69df8e"
    ),
    "exllamav3/exllamav3_ext/cpu/arm_stubs.cpp": (
        "4abb18d5b6a99c9ce0e0b0f118a33e77a33417c6b471a02dc06792c78a007f2f"
    ),
    "exllamav3/exllamav3_ext/cpu/moe_handoff.cu": (
        "71382fdc782877a6a0f8173615082964f70353a50e09f02cbe98ae0a0e7d8051"
    ),
    "exllamav3/exllamav3_ext/parallel/all_reduce_cpu.cu": (
        "18e72f0c39c3a2447ab1d7de0f87c13e69f9216defa1c49f8bb5e31a40a9f14c"
    ),
    "setup.py": "29e339ee9205df20715d2cb876452569751270fbf62f4f045b358ed9949fb308",
}


def _write(path: Path, content: str = "", *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    if executable:
        path.chmod(0o755)


def _bash_path(path: str | Path) -> str:
    value = str(path)
    if os.name != "nt":
        return value
    drive, tail = os.path.splitdrive(value)
    assert drive, value
    return f"/mnt/{drive[0].lower()}/{tail.lstrip('\\/').replace('\\', '/')}"


def _shell_assignment(name: str, value: str | Path) -> str:
    return f"{name}={shlex.quote(_bash_path(value))}"


def _run_launcher(env: dict[str, str], *arguments: str) -> subprocess.CompletedProcess[str]:
    if os.name != "nt":
        return subprocess.run(
            ["bash", str(LAUNCHER), *arguments],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    path_keys = {
        "QWEN_ENV_FILE",
        "QWEN_MODEL_PATH",
        "QWEN_CHAT_TEMPLATE",
        "QWEN_VENV",
        "QWEN_VLLM_SOURCE",
        "QWEN_EXLLAMAV3_SOURCE",
        "QWEN_INFINIBAND_DEV_ROOT",
        "QWEN_INFINIBAND_SYS_ROOT",
        "QWEN_TEST_EXEC_MARKER",
    }
    forwarded: dict[str, str] = {}
    for key, value in env.items():
        if key in ("RANK", "RANK0_RENDEZVOUS_ADDR") or key.startswith("QWEN_"):
            forwarded[key] = _bash_path(value) if key in path_keys else value

    mock_bin = _bash_path(env["PATH"].split(os.pathsep, 1)[0])
    assignments = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in sorted(forwarded.items())
    )
    launcher = _bash_path(LAUNCHER)
    rendered_arguments = " ".join(shlex.quote(value) for value in arguments)
    command = (
        f"export PATH={shlex.quote(mock_bin)}:$PATH; "
        f"exec env {assignments} bash {shlex.quote(launcher)} {rendered_arguments}"
    )
    return subprocess.run(
        ["bash", "-lc", command],
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.fixture
def prepared_rank(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    mock_bin = tmp_path / "mock-bin"
    model = tmp_path / "model"
    venv = tmp_path / "venv"
    vllm_source = tmp_path / "vllm-source"
    exllamav3_source = tmp_path / "exllamav3-source"
    dev_root = tmp_path / "dev-infiniband"
    sys_root = tmp_path / "sys-infiniband"
    template = tmp_path / "chat-template.jinja"
    nccl = tmp_path / "libnccl.so.2"
    env_file = tmp_path / "rank.env"
    marker = tmp_path / "vllm-executed.txt"

    for path in (
        model,
        venv / "bin",
        vllm_source / ".git",
        exllamav3_source / ".git",
        dev_root,
        sys_root,
        mock_bin,
    ):
        path.mkdir(parents=True, exist_ok=True)

    _write(dev_root / "uverbs0")
    _write(template, "fixture template\n")
    _write(nccl, "fixture nccl\n")
    _write(model / "config.json", "{}\n")
    _write(model / "model.safetensors.index.json", "{}\n")
    manifest = "".join(f"{'0' * 64}  file-{index}\n" for index in range(16))
    _write(model / "SHA256SUMS", manifest)

    for relative_path in EXLLAMAV3_FILES:
        _write(exllamav3_source / relative_path, f"fixture {relative_path}\n")

    for hca, ndev, gid in (
        ("hca0", "fabric0", "0000:0000:0000:0000:0000:ffff:0a2a:0101"),
        ("hca1", "fabric1", "0000:0000:0000:0000:0000:ffff:0a2a:0201"),
    ):
        port_root = sys_root / hca / "ports" / "1"
        _write(port_root / "gids" / "3", f"{gid}\n")
        _write(port_root / "gid_attrs" / "types" / "3", "RoCE v2\n")
        _write(port_root / "gid_attrs" / "ndevs" / "3", f"{ndev}\n")

    _write(venv / "bin" / "activate", "# fixture activate\n")
    _write(venv / "bin" / "python", "#!/usr/bin/env bash\nexit 0\n", executable=True)
    _write(
        venv / "bin" / "vllm",
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$QWEN_TEST_EXEC_MARKER\"\n",
        executable=True,
    )

    git_mock = f"""\
        #!/usr/bin/env bash
        set -eu
        args="$*"
        if [[ "$args" == *"vllm-source"* ]]; then
            if [[ "$args" == *"rev-parse HEAD"* ]]; then
                echo {VLLM_COMMIT}
            elif [[ "$args" == *"status --porcelain"* ]]; then
                :
            else
                exit 2
            fi
        elif [[ "$args" == *"exllamav3-source"* ]]; then
            if [[ "$args" == *"rev-parse HEAD"* ]]; then
                echo {EXLLAMAV3_COMMIT}
            elif [[ "$args" == *"status --porcelain"* ]]; then
                if [[ "${{QWEN_TEST_EXLLAMAV3_STAGED:-0}}" == 1 ]]; then
                    cat <<'EOF'
M  exllamav3/exllamav3_ext/avx2_target.cpp
M  exllamav3/exllamav3_ext/avx512_target.cpp
M  exllamav3/exllamav3_ext/cpu/moe_handoff.cu
M  exllamav3/exllamav3_ext/parallel/all_reduce_cpu.cu
M  setup.py
A  exllamav3/exllamav3_ext/cpu/arm_stubs.cpp
EOF
                else
                    cat <<'EOF'
 M exllamav3/exllamav3_ext/avx2_target.cpp
 M exllamav3/exllamav3_ext/avx512_target.cpp
 M exllamav3/exllamav3_ext/cpu/moe_handoff.cu
 M exllamav3/exllamav3_ext/parallel/all_reduce_cpu.cu
 M setup.py
?? exllamav3/exllamav3_ext/cpu/arm_stubs.cpp
EOF
                fi
                if [[ "${{QWEN_TEST_EXLLAMAV3_EXTRA:-0}}" == 1 ]]; then
                    echo '?? unexpected-file'
                fi
            else
                exit 2
            fi
        else
            exit 2
        fi
    """
    _write(mock_bin / "git", textwrap.dedent(git_mock).lstrip(), executable=True)

    sha_cases = "\n".join(
        f'  *{shlex.quote(path)}*) digest={digest} ;;'
        for path, digest in EXLLAMAV3_FILES.items()
    )
    sha_mock = f"""\
        #!/usr/bin/env bash
        set -eu
        args="$*"
        if [[ "$args" == *"--check"* ]]; then
            exit 0
        fi
        if [[ "${{QWEN_TEST_BAD_EXLLAMAV3_SHA:-0}}" == 1 && "$args" == *"arm_stubs.cpp"* ]]; then
            printf '%064d  %s\n' 0 "${{!#}}"
            exit 0
        fi
        case "$args" in
{sha_cases}
          *model/SHA256SUMS*) digest=7626d18481e7f995fd1d9ff211083b7fd57f044daba39e107fb29a48207f24c4 ;;
          *model/config.json*) digest=fbb105334da6554c10784ff1257fda5e3821d4d5426d64469cee2b2ad67ba2b3 ;;
          *model/model.safetensors.index.json*) digest=ea6e0e1064efbb72d89b4a6f9e0ee76c909a94b3f25047487a2ffb282896a26c ;;
          *chat-template.jinja*) digest=4f9201169f5bacd1a494c8824470a1ef899c7024d23a2b166e42493e7efd9ac9 ;;
          *libnccl.so.2*) digest=e69a8c240f45d10166bcd901d99db78bb63147adda66e586d8dd505c6d608b54 ;;
          *) exit 2 ;;
        esac
        target=${{!#}}
        printf '%s  %s\\n' "$digest" "$target"
    """
    _write(mock_bin / "sha256sum", textwrap.dedent(sha_mock).lstrip(), executable=True)

    ip_mock = """\
        #!/usr/bin/env bash
        set -eu
        if [[ "$1" == "link" && "$2" == "show" && "$3" == "dev" ]]; then
            printf '1: %s: <UP>\\n' "$4"
            exit 0
        fi
        if [[ "$1" == "-o" && "$2" == "-4" && "$3" == "addr" ]]; then
            iface=$6
            case "$iface" in
              mgmt0) ip=192.0.2.10 ;;
              fabric0) ip=10.42.1.1 ;;
              fabric1) ip=10.42.2.1 ;;
              *) exit 1 ;;
            esac
            printf '1: %s inet %s/24 scope global %s\\n' "$iface" "$ip" "$iface"
            exit 0
        fi
        exit 2
    """
    _write(mock_bin / "ip", textwrap.dedent(ip_mock).lstrip(), executable=True)
    _write(
        mock_bin / "pgrep",
        "#!/usr/bin/env bash\n[[ \"${QWEN_TEST_PGREP_BUSY:-0}\" == 1 ]]\n",
        executable=True,
    )

    env_lines = [
        _shell_assignment("LD_PRELOAD", nccl),
        _shell_assignment("VLLM_NCCL_SO_PATH", nccl),
        "NCCL_SOCKET_IFNAME=mgmt0",
        "GLOO_SOCKET_IFNAME=mgmt0",
        "VLLM_HOST_IP=192.0.2.10",
        "NCCL_NET=IB",
        "NCCL_NET_PLUGIN=none",
        "NCCL_IB_DISABLE=0",
        "NCCL_IB_HCA=hca0,hca1",
        "NCCL_IB_GID_INDEX=3",
        "NCCL_IB_SUBNET_PREFIX_LEN=24",
        "NCCL_IB_SUBNET_AWARE_ROUTING=1",
        "NCCL_IB_MERGE_NICS=0",
        "NCCL_ALGO=Ring",
        "NCCL_PROTO=LL,LL128,Simple",
        "NCCL_P2P_LEVEL=SYS",
        "NCCL_MIN_NCHANNELS=4",
        "NCCL_MAX_NCHANNELS=4",
        "NCCL_CROSS_NIC=1",
        "NCCL_CUMEM_ENABLE=0",
        "NCCL_SKIP_TREE_CONNECT=1",
        "NCCL_IGNORE_CPU_AFFINITY=1",
        "VLLM_EXL3_GRAPH_DECODE=1",
        "VLLM_EXL3_PREFILL_FP8=1",
        "VLLM_EXL3_PREFILL_RECONSTRUCT_M=256",
    ]
    _write(env_file, "\n".join(env_lines) + "\n")

    env = dict(os.environ)
    env.update(
        {
            "PATH": f"{mock_bin}{os.pathsep}{env.get('PATH', '')}",
            "RANK": "0",
            "RANK0_RENDEZVOUS_ADDR": "192.0.2.10",
            "QWEN_ENV_FILE": str(env_file),
            "QWEN_MODEL_PATH": str(model),
            "QWEN_CHAT_TEMPLATE": str(template),
            "QWEN_VENV": str(venv),
            "QWEN_VLLM_SOURCE": str(vllm_source),
            "QWEN_EXLLAMAV3_SOURCE": str(exllamav3_source),
            "QWEN_INFINIBAND_DEV_ROOT": str(dev_root),
            "QWEN_INFINIBAND_SYS_ROOT": str(sys_root),
            "QWEN_TEST_EXEC_MARKER": str(marker),
        }
    )
    return env, env_file, marker


def test_launcher_statically_covers_the_local_preflight_contract() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    for required in (
        'if [ "$1" = "--check" ]',
        'elif [ "$1" = "--run" ]',
        "rank environment file contains unresolved placeholders",
        "status --porcelain --untracked-files=all",
        "exllamav3/exllamav3_ext/cpu/arm_stubs.cpp",
        "sha256sum --check --strict --status SHA256SUMS",
        "from exllamav3_ext import exl3_gemm",
        'ctypes.CDLL("libibverbs.so.1")',
        "torch.cuda.is_available()",
        "torch.cuda.device_count() != 1",
        "torch.cuda.get_device_capability(0)",
        "NCCL_IB_HCA must name exactly two cycle-facing devices",
        "RoCE v2",
        "required rank-0 port is already bound",
        "resolved command:",
        'exec "${command[@]}"',
    ):
        assert required in text
    assert "git diff --binary" not in text


def test_check_mode_validates_and_does_not_execute_vllm(
    prepared_rank: tuple[dict[str, str], Path, Path],
) -> None:
    env, _, marker = prepared_rank
    result = _run_launcher(env, "--check")
    assert result.returncode == 0, result.stderr
    assert "qwen38 preflight passed for rank 0" in result.stdout
    assert "resolved command:" in result.stdout
    assert "--tensor-parallel-size 4" in result.stdout
    assert not marker.exists()


def test_check_mode_rejects_an_unresolved_placeholder(
    prepared_rank: tuple[dict[str, str], Path, Path],
) -> None:
    env, env_file, _ = prepared_rank
    with env_file.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write("UNRESOLVED=<RANK_VALUE>\n")
    result = _run_launcher(env, "--check")
    assert result.returncode == 20
    assert "contains unresolved placeholders" in result.stderr


def test_check_mode_rejects_extra_exllamav3_state(
    prepared_rank: tuple[dict[str, str], Path, Path],
) -> None:
    env, _, _ = prepared_rank
    env["QWEN_TEST_EXLLAMAV3_EXTRA"] = "1"
    result = _run_launcher(env, "--check")
    assert result.returncode == 20
    assert "unexpected change" in result.stderr


def test_check_mode_accepts_builder_staged_exllamav3_state(
    prepared_rank: tuple[dict[str, str], Path, Path],
) -> None:
    env, _, marker = prepared_rank
    env["QWEN_TEST_EXLLAMAV3_STAGED"] = "1"
    result = _run_launcher(env, "--check")
    assert result.returncode == 0, result.stderr
    assert "qwen38 preflight passed for rank 0" in result.stdout
    assert not marker.exists()


def test_check_mode_rejects_wrong_exllamav3_postimage(
    prepared_rank: tuple[dict[str, str], Path, Path],
) -> None:
    env, _, _ = prepared_rank
    env["QWEN_TEST_BAD_EXLLAMAV3_SHA"] = "1"
    result = _run_launcher(env, "--check")
    assert result.returncode == 20
    assert "ExLlamaV3 ARM patch file SHA-256 mismatch" in result.stderr


def test_check_mode_rejects_a_duplicate_vllm_process(
    prepared_rank: tuple[dict[str, str], Path, Path],
) -> None:
    env, _, _ = prepared_rank
    env["QWEN_TEST_PGREP_BUSY"] = "1"
    result = _run_launcher(env, "--check")
    assert result.returncode == 20
    assert "vLLM serving process is already running" in result.stderr


def test_normal_mode_executes_the_preflighted_command(
    prepared_rank: tuple[dict[str, str], Path, Path],
) -> None:
    env, _, marker = prepared_rank
    result = _run_launcher(env)
    assert result.returncode == 0, result.stderr
    assert marker.exists()
    arguments = marker.read_text(encoding="utf-8")
    assert arguments.startswith("serve\n")
    assert "--node-rank\n0\n" in arguments
    assert "--host\n0.0.0.0\n--port\n8000\n" in arguments


def test_explicit_run_mode_overrides_the_image_sleep_command(
    prepared_rank: tuple[dict[str, str], Path, Path],
) -> None:
    env, _, marker = prepared_rank
    result = _run_launcher(env, "--run")
    assert result.returncode == 0, result.stderr
    assert marker.exists()
    assert "--tensor-parallel-size\n4\n" in marker.read_text(encoding="utf-8")
