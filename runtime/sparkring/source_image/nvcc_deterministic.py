#!/usr/bin/env python3
"""Assign stable, distinct NVCC random seeds to source and device-link units."""
import hashlib
import os
from pathlib import PurePosixPath
import posixpath
import subprocess
import sys

COMPILER = "/opt/cuda-13.3/bin/nvcc"
ROOTS = (("/work/src-lf", "source"), ("/work/build", "build"))
POLICY = "sparkring-nvcc-source-seed/v1"
PROBES = {"--version", "-V", "--help", "-h", "--list-gpu-code", "--list-gpu-arch"}
SOURCE_SUFFIXES = {".cu", ".cuh", ".c", ".cc", ".cpp", ".cxx", ".C"}
VALUE_OPTIONS = {
    "-I", "--include-path", "-isystem", "--system-include", "-include", "--pre-include",
    "-ccbin", "--compiler-bindir", "-gencode", "--generate-code", "-arch", "--gpu-architecture",
    "-code", "--gpu-code", "-std", "--std", "-x", "-D", "--define-macro", "-U", "--undefine-macro",
    "-Xcompiler", "--compiler-options", "-Xptxas", "--ptxas-options", "-Xfatbin", "--fatbin-options",
    "-Xnvlink", "--nvlink-options", "-MT", "-MF", "-odir", "--output-directory",
}


def path_identity(value, cwd):
    if (not value or any(c in value for c in ("\x00", "\n", "\r", "\\", ":"))
            or not cwd.startswith("/")):
        raise ValueError("NVCC seed path must be an unambiguous Linux path")
    absolute = posixpath.normpath(value if value.startswith("/") else posixpath.join(cwd, value))
    for root, name in ROOTS:
        if absolute.startswith(root + "/"):
            return name + "/" + absolute[len(root) + 1:]
    raise ValueError("NVCC input/output path is outside the locked source and build roots")


def seeded_arguments(arguments, cwd):
    """Return unchanged probe argv or argv with one source-derived seed."""
    arguments = list(arguments)
    if len(arguments) == 1 and arguments[0] in PROBES:
        return arguments
    if any("frandom-seed" in value for value in arguments):
        raise ValueError("NVCC random seed is controlled by the source-image recipe")
    if any(value.startswith("@") or value.split("=", 1)[0] in ("-optf", "--options-file")
           for value in arguments):
        raise ValueError("NVCC response files hide seed input identity")
    sources, objects, outputs = [], [], []
    index = 0
    while index < len(arguments):
        value = arguments[index]
        if value in ("-o", "--output-file"):
            index += 1
            if index == len(arguments):
                raise ValueError("NVCC output option has no path")
            outputs.append(path_identity(arguments[index], cwd))
        elif value.startswith("--output-file=") or value.startswith("-o="):
            outputs.append(path_identity(value.split("=", 1)[1], cwd))
        elif value in VALUE_OPTIONS:
            index += 1
            if index == len(arguments):
                raise ValueError("NVCC option has no value")
        elif not value.startswith("-"):
            suffix = PurePosixPath(value).suffix
            if suffix in SOURCE_SUFFIXES:
                sources.append(path_identity(value, cwd))
            elif suffix in (".o", ".a"):
                objects.append(path_identity(value, cwd))
            else:
                raise ValueError("NVCC positional input has an unsupported file type")
        index += 1
    if len(outputs) > 1:
        raise ValueError("NVCC invocation has ambiguous output paths")
    device_link = any(value in ("-dlink", "--device-link") for value in arguments)
    if device_link:
        if sources or not objects or len(outputs) != 1:
            raise ValueError("Device link requires object inputs and one output path")
        identity = "device-link/" + outputs[0]
    else:
        if len(sources) != 1 or objects:
            raise ValueError("NVCC compilation requires exactly one source input")
        identity = "compile/" + sources[0]
    seed = hashlib.sha256((POLICY + "\n" + identity).encode()).hexdigest()
    return ["--frandom-seed=" + seed, *arguments]


def main():
    try:
        arguments = seeded_arguments(sys.argv[1:], os.getcwd())
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    return subprocess.call([COMPILER, *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
