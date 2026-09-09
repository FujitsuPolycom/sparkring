"""Prepare a source-pinned GLM MTP3 performance image without remote actions."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
CACHE_COMMIT = "b5aca7cd3d3f7e7a14636bf6e5fa1f50a9650168"
BASE_IMAGE = "ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:67dc0ae453baaae6831ccec1d259b4ef8b236a8b0dc9f747d901b95c66ec1987"
BASE_ID = "sha256:2e41b1e934a85ff7c21b780532db2f0a0e978df081e52f4ae2bf11f8992fb24f"
PLACEMENT = "2657cdd2e54a097c9544e4c79ae62c0646db6db123ff24e4f0c384238c3a1e8d"
TRANSPORT = "056243fad27d224b82e437925ffa2aed42037e6bd29f239f56076a832f6ca5cb"


def prepare(cache, placement, transport, output):
    if output.exists():
        raise ValueError("Image context destination must not exist")
    for path, expected in ((placement, PLACEMENT), (transport, TRANSPORT)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(
                f"Native artifact differs from qualified input: {path.name}"
            )
    revision = subprocess.check_output(
        ["git", "-C", str(cache), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != CACHE_COMMIT or subprocess.check_output(
        ["git", "-C", str(cache), "status", "--porcelain"]
    ):
        raise ValueError("SparkCache checkout must be clean at the pinned revision")
    archive = subprocess.check_output(
        ["git", "-C", str(cache), "archive", "HEAD", "sparkcache"]
    )
    output.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        for member in bundle.getmembers():
            name = Path(member.name)
            if (
                name.is_absolute()
                or ".." in name.parts
                or not name.parts
                or name.parts[0] != "sparkcache"
            ):
                raise ValueError("Source archive path escaped package")
            if member.isfile():
                path = output / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(bundle.extractfile(member).read())
    for name in ("checkpoints", "reasoning", "attribution", "continuation", "mhc-prefill"):
        shutil.copytree(
            HERE / name,
            output / name,
            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
        )
    shutil.copytree(
        HERE.parent / "experiments/cache-reuse",
        output / "reuse",
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
    )
    subprocess.run(
        [
            "python",
            str(HERE / "transport/package.py"),
            "bundle",
            "--native-library",
            str(transport),
            "--native-sha256",
            TRANSPORT,
            "--output",
            str(output / "bundle"),
        ],
        check=True,
    )
    shutil.copyfile(placement, output / "libspark_cache_placement.so")
    for name in ("install.py", "verify.py", "start.py", "Dockerfile"):
        shutil.copyfile(HERE / name, output / name)
    startup = output / "startup"
    startup.mkdir()
    for name in ("serve_with_warmup.py", "warmup_dflash.py", "startup_admission.py",
                 "scheduler_liveness.py"):
        shutil.copyfile(HERE.parent.parent / "glm53-flash-jj-r8-gb10" / name,
                        startup / name)
    files = {
        p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.rglob("*")
        if p.is_file()
    }
    record = {
        "schema": "sparkring-mtp3-performance-context/v1",
        "status": "implemented",
        "sparkcache_commit": CACHE_COMMIT,
        "base_reference": BASE_IMAGE,
        "base_image_id": BASE_ID,
        "files": files,
    }
    (output / "context.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparkcache", type=Path, required=True)
    parser.add_argument("--placement-library", type=Path, required=True)
    parser.add_argument("--transport-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(
        args.sparkcache, args.placement_library, args.transport_library, args.output
    )
    print(
        json.dumps(
            {"source": result["sparkcache_commit"], "files": len(result["files"])}
        )
    )
