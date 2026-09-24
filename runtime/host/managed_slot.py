"""Read-only ownership check for the legacy GLM managed-service namespace."""


def inspect_slot(name, image_id, rank, *, root="/", run=None):
    import json
    from pathlib import Path
    import re
    import subprocess
    run = run or subprocess.run
    names = ("opt/sparkring/managed-mesh", "etc/sparkring/managed-mesh",
             "etc/systemd/system/sparkring-mesh.service", "etc/systemd/system/sparkring-mesh-model.service")
    occupied = ["/" + value for value in names if (Path(root) / value).exists() or (Path(root) / value).is_symlink()]
    if not occupied:
        return {"available": True, "occupied": []}
    path = Path(root) / "etc/sparkring/managed-mesh/service.json"
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)) or path.stat().st_size > 65536:
        return {"available": False, "occupied": occupied}
    config = json.loads(path.read_text())
    ident = config.get("container_id", "")
    if not isinstance(ident, str) or not re.fullmatch(r"[0-9a-f]{64}", ident):
        return {"available": False, "occupied": occupied}
    result = run(["docker", "--context", "default", "inspect", ident], capture_output=True, text=True, timeout=30)
    observed = json.loads(result.stdout)[0] if result.returncode == 0 else {}
    same = (config.get("rank") == rank and config.get("container_image") == image_id
            and observed.get("Name") == "/" + name + "-r" + str(rank) and observed.get("Image") == image_id
            and observed.get("Config", {}).get("Labels", {}).get("io.sparkring.container-spec") == "glm-tp4/v1")
    return {"available": same, "occupied": occupied}
