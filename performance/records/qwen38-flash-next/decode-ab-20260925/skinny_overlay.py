#!/usr/bin/env python3
"""Build an experiment overlay that enables Qwen4Exp skinny-GEMM decode plans on SM121.

Run as root on each node: skinny_overlay.py RESULTS_JSON OVERLAY_DIR
The image's low_latency_gemm.py selects measured skinny-GEMM plans only on SM90
and SM103. The overlay copy adds a plan table measured on SM121 (GB10), keeping
each (N, K, M) whose fastest correct configuration beat torch linear by at least
5%, and selects it on SM121. Receipts are rewritten to record the overlay file,
so the image's startup verification accepts it.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

IMAGE = "sha256:5ce6ce267d8069215e08da87be3a8092175e63806cf97730847405eb04a2c51f"
TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/low_latency_gemm.py"
RESULTS, OVERLAY = Path(sys.argv[1]), Path(sys.argv[2])
MINIMUM_SPEEDUP = 1.05


def sha(data):
    return hashlib.sha256(data).hexdigest()


def image_file(name):
    return subprocess.run(["docker", "run", "--rm", "--entrypoint", "cat", IMAGE, name],
                          capture_output=True, check=True).stdout


results = json.loads(RESULTS.read_text())
plans = {}
for name, shape in results["shapes"].items():
    for m, row in shape["by_m"].items():
        if not m.isdigit() or row.get("speedup", 0) < MINIMUM_SPEEDUP:
            continue
        rows, block, outputs, unroll, width = row["config"]
        plans.setdefault((shape["n"], shape["k"]), {})[int(m)] = (
            f"SkinnyGemmConfig({rows}, {block}, {outputs}, k_unroll={unroll}, vector_width={width})",
            name, row["torch_us"], row["best_us"])
lines = ["# GB10 (SM121) plans measured by CUDA graph replay with weights read from",
         "# memory, M={1, 2, 4, 8}. Only points at least 5% faster than the standard",
         "# linear implementation are retained.",
         "QWEN4_EXP_SM121_GEMM_PLANS: dict[tuple[int, int], dict[int, SkinnyGemmConfig]] = {"]
for (n, k), by_m in sorted(plans.items()):
    label = sorted({entry[1] for entry in by_m.values()})
    lines.append(f"    # {', '.join(label)}.")
    lines.append(f"    ({n}, {k}): {{")
    for m, (config, _, torch_us, best_us) in sorted(by_m.items()):
        lines.append(f"        {m}: {config},  # {torch_us} -> {best_us} us")
    lines.append("    },")
lines.append("}")
table = "\n".join(lines) + "\n\n\n"

source = image_file(TARGET).decode()
anchor = "def _is_sm103() -> bool:"
assert source.count(anchor) == 1
source = source.replace(anchor, table + "def _is_sm121() -> bool:\n    return current_platform.is_device_capability((12, 1))\n\n\n" + anchor)
old = "    if _is_sm90():\n        return QWEN4_EXP_SM90_GEMM_PLANS\n"
assert source.count(old) == 1
source = source.replace(old, old + "    if _is_sm121():\n        return QWEN4_EXP_SM121_GEMM_PLANS\n")
probe = "        plan = plans.get((weight.shape[0], weight.shape[1]))\n        if plan is None:\n            continue\n"
assert source.count(probe) == 1
source = source.replace(probe, "        plan = plans.get((weight.shape[0], weight.shape[1]))\n"
                        "        print(f\"QWEN4_SKINNY {'plan' if plan else 'none'} {type(child).__name__} {tuple(weight.shape)}\", flush=True)\n"
                        "        if plan is None:\n            continue\n")
OVERLAY.mkdir(parents=True, exist_ok=True)
(OVERLAY / "low_latency_gemm.py").write_text(source)
files = {TARGET: source}
# The model and its MTP draft call the low-latency hook only without B12X; on
# SM121 the hook replaces only BF16 linear layers, which B12X leaves to torch.
MODEL_DIR = "/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/"
for name, call in (("model.py", "enable_qwen4_exp_low_latency_gemm(self, self.model_config.dtype)"),
                   ("mtp.py", "enable_qwen4_exp_low_latency_gemm(self, vllm_config.model_config.dtype)")):
    text = image_file(MODEL_DIR + name).decode()
    guarded = "        if not uses_b12x(vllm_config):\n            " + call + "\n"
    assert text.count(guarded) == 1, name
    text = text.replace(guarded, "        " + call + "\n")
    (OVERLAY / name).write_text(text)
    files[MODEL_DIR + name] = text

base = json.loads(image_file("/opt/sparkring/receipts/external-base-installed.json"))
for path, text in files.items():
    if path not in base["files"]:
        raise SystemExit("The image does not record " + path)
    base["files"][path] = sha(text.encode())
payload = (json.dumps(base, indent=2, sort_keys=True) + "\n").encode()
(OVERLAY / "external-base-installed.json").write_bytes(payload)
toolchain = json.loads(image_file("/opt/sparkring/toolchain/installed.json"))
toolchain["parent_receipt_sha256"] = sha(payload)
(OVERLAY / "toolchain-installed.json").write_text(json.dumps(toolchain, indent=2) + "\n")
print(f"{sum(len(v) for v in plans.values())} plan points over {len(plans)} shapes; base receipt {sha(payload)[:16]}")
