#!/usr/bin/env python3
"""Generate the NCCL host ABI surface from its pinned public header only.

This parser is deliberately narrow: a new public declaration that it cannot
understand stops generation instead of silently omitting an ABI entry point.
No NCCL implementation source is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTED = frozenset({
    "ncclGetVersion", "ncclGetUniqueId", "ncclGetErrorString", "ncclGetLastError",
    "ncclCommInitRank", "ncclCommInitRankConfig", "ncclCommInitAll",
    "ncclCommFinalize", "ncclCommDestroy", "ncclCommAbort", "ncclCommRevoke",
    "ncclCommGetAsyncError", "ncclCommCount", "ncclCommCuDevice", "ncclCommUserRank",
    "ncclGroupStart", "ncclGroupEnd", "ncclAllReduce", "ncclAllGather", "ncclReduceScatter",
    "ncclReduce", "ncclBroadcast", "ncclBcast", "ncclAlltoAll", "ncclGather", "ncclScatter",
    "ncclSend", "ncclRecv", "ncclCommSplit", "ncclMemAlloc", "ncclMemFree", "ncclCommRegister",
    "ncclCommDeregister", "ncclCommWindowRegister", "ncclCommWindowDeregister", "ncclWinGetUserPtr",
})


def declarations(source: str) -> list[dict]:
    clean = re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.S)
    clean = re.sub(r'__attribute__\s*\(\(\s*deprecated\("[^"]*"\)\s*\)\)', "", clean)
    pattern = re.compile(r"\b(ncclResult_t|const\s+char\s*\*|void)\s+(p?nccl\w+)\s*\(([^;{}]*)\)\s*;", re.S)
    apis = []
    for result, name, arguments in pattern.findall(clean):
        if name.startswith("pnccl"):
            continue
        params = []
        for argument in arguments.split(","):
            argument = " ".join(argument.split())
            if argument in ("", "void"):
                continue
            match = re.fullmatch(r"(.+?)([A-Za-z_]\w*)", argument)
            if not match:
                raise ValueError(f"Unrecognized argument in {name}: {argument}")
            params.append({"type": match[1].strip(), "name": match[2]})
        apis.append({"name": name, "return": " ".join(result.split()), "arguments": params,
                     "implemented": name in IMPLEMENTED,
                     "linux_only": name == "ncclResetDebugInit"})
    names = [api["name"] for api in apis]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate public function declarations")
    declared = {name for _, name, _ in pattern.findall(clean)}
    if declared != set(names) | {"p" + name for name in names}:
        raise ValueError("Every nccl function must have a matching pnccl declaration")
    # Check every public function-looking declaration, including unfamiliar return types.
    public = set(re.findall(r"\b(p?nccl\w+)\s*\(", clean))
    if public != declared:
        raise ValueError(f"Unrecognized declarations: {sorted(public - declared)}")
    if not IMPLEMENTED <= set(names):
        raise ValueError(f"Missing implemented functions: {sorted(IMPLEMENTED - set(names))}")
    return apis


def signature(api: dict, name: str | None = None) -> str:
    arguments = ", ".join(f"{p['type']} {p['name']}" for p in api["arguments"]) or "void"
    return f"{api['return']} {name or api['name']}({arguments})"


# Outputs that an unimplemented function sets before it returns ncclInvalidUsage, by function: the argument
# and its statement. ncclRedOpCreatePreMulSum returns ncclNumOps, the first value that is not a built-in op,
# so a caller that ignores the refusal and passes the op to a collective gets ncclInvalidArgument from the
# collective, never a plain sum from a handle left at 0 (ncclSum).
REFUSED_OUTPUTS = {
    "ncclRedOpCreatePreMulSum": ("op", "  if (op) *op = (ncclRedOp_t)ncclNumOps;\n"),
}


def artifacts() -> dict[Path, str]:
    raw = (ROOT / "vendor/nccl.h.in").read_bytes()
    source = raw.decode("utf-8").replace("\r\n", "\n")
    apis = declarations(source)
    substitutions = {"Major": "2", "Minor": "32", "Patch": "3", "Suffix": "", "Version": "23203"}
    header = re.sub(r"\$\{nccl:(\w+)\}", lambda m: substitutions[m[1]], source)
    cuda_start = header.index("#include <cuda_runtime.h>")
    cuda_end = header.index("#define NCCL_MAJOR")
    cuda = header[cuda_start:cuda_end].rstrip()
    notice_end = header.index("*/") + 3
    header = header[:notice_end] + (
        "/* Modified by SparkRing contributors in 2026 for libsircl: an explicit host-only build path\n"
        " * (LIBSIRCL_CPU_ONLY), a default for NCCL_OS_LINUX, and the version template values filled in for\n"
        " * 2.32.3. Original: vendor/nccl.h.in (NVIDIA/nccl v2.32.3-1, src/nccl.h.in). */\n"
    ) + header[notice_end:]
    cuda_start = header.index("#include <cuda_runtime.h>")
    cuda_end = header.index("#define NCCL_MAJOR")
    cuda = header[cuda_start:cuda_end].rstrip()
    header = header[:cuda_start] + (
        "/* libsircl modification: explicit host-only ABI inspection/build mode.\n"
        " * This does not supply or emulate a CUDA runtime. Normal clients use CUDA. */\n"
        "#include <stddef.h>\n"
        "#ifdef LIBSIRCL_CPU_ONLY\n#include \"sccl_cuda_types.h\"\n#else\n"
        + cuda + "\n#endif\n\n"
        "/* Preserve the public Linux-only deprecated declaration. */\n"
        "#if defined(__linux__) && !defined(NCCL_OS_LINUX)\n"
        "#define NCCL_OS_LINUX 1\n#endif\n\n"
    ) + header[cuda_end:]
    generated = ("/* Generated by tools/generate_api.py from vendor/nccl.h.in (NVIDIA NCCL v2.32.3-1, Copyright (c)\n"
                 " * 2015-2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0; see vendor/NCCL-LICENSE.txt). Do not edit. */\n")
    stubs = generated + '#include "internal.h"\n\n'
    aliases = generated + '#include "internal.h"\n\n'
    for api in apis:
        if not api["implemented"]:
            stubs += signature(api) + " {\n"
            output = REFUSED_OUTPUTS.get(api["name"])
            for argument in api["arguments"]:
                if output and argument["name"] == output[0]:
                    stubs += output[1]
                else:
                    stubs += f"  (void){argument['name']};\n"
            if api["return"] == "void":
                stubs += f"  (void)sccl_unsupported(\"{api['name']}\");\n"
            else:
                stubs += f"  return sccl_unsupported(\"{api['name']}\");\n"
            stubs += "}\n\n"
        if api["linux_only"]:
            aliases += '#if defined(__GNUC__)\n#pragma GCC diagnostic push\n#pragma GCC diagnostic ignored "-Wdeprecated-declarations"\n#endif\n'
        aliases += signature(api, "p" + api["name"]) + " {\n"
        args = ", ".join(argument["name"] for argument in api["arguments"])
        prefix = "" if api["return"] == "void" else "return "
        aliases += f"  {prefix}{api['name']}({args});\n}}\n\n"
        if api["linux_only"]:
            aliases += '#if defined(__GNUC__)\n#pragma GCC diagnostic pop\n#endif\n\n'
    cuda_types = """/* SIRCL host-only ABI declarations; no CUDA implementation is provided. */
#ifndef SIRCL_CUDA_TYPES_H_
#define SIRCL_CUDA_TYPES_H_
#ifndef LIBSIRCL_CPU_ONLY
#error "sccl_cuda_types.h is only for an explicit LIBSIRCL_CPU_ONLY build"
#endif
/* Public CUDA runtime handles are opaque pointers of these shapes. */
typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;
#endif
"""
    manifest = {"upstream": "NVIDIA/nccl v2.32.3-1", "header_sha256": hashlib.sha256(raw).hexdigest(),
                "header_version": 23203, "functions": apis}
    return {ROOT / "include/nccl.h": header,
            ROOT / "include/sccl_cuda_types.h": cuda_types,
            ROOT / "src/stubs.c": stubs,
            ROOT / "src/aliases.c": aliases,
            ROOT / "tests/api_manifest.json": json.dumps(manifest, indent=2) + "\n"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if committed/generated outputs drift")
    args = parser.parse_args()
    changed = []
    for path, content in artifacts().items():
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            changed.append(str(path.relative_to(ROOT)))
            if not args.check:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8", newline="\n")
    count = len(declarations((ROOT / "vendor/nccl.h.in").read_text(encoding="utf-8")))
    if args.check and changed:
        print("Generated API drift: " + ", ".join(changed), file=sys.stderr)
        return 1
    print(f"{'Verified' if args.check else 'Generated'} {count} nccl host APIs and {count} pnccl twins")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
