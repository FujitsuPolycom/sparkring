"""Run a hash-pinned benchmark and compare identical measured grids."""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys

from .contracts import require, sha
from .io import command, write_json
from .serving_checks import benchmark_measurements


def compare(candidate, baseline, limits):
    require(
        set(candidate) == set(baseline) and candidate,
        "Control and candidate measurement grids differ",
    )
    rows = []
    for name in sorted(candidate):
        before, after = baseline[name], candidate[name]
        require(
            len(before) >= 3 and len(after) >= 3,
            "Every performance cell needs at least three samples",
        )
        control = statistics.median(before)
        measured = statistics.median(after)
        family = (
            "prefill"
            if name.startswith("prefill-")
            else "steps"
            if name.endswith("-steps")
            else "decode"
        )
        tolerance = limits[family]
        require(
            type(tolerance) in (int, float) and 0 <= tolerance < 1,
            "Invalid performance tolerance",
        )
        rows.append(
            {
                "metric": name,
                "baseline_median": control,
                "candidate_median": measured,
                "ratio": measured / control,
                "max_regression_fraction": tolerance,
                "passed": measured >= control * (1 - tolerance),
            }
        )
    return {"passed": all(row["passed"] for row in rows), "comparisons": rows}


def validate_config(config):
    require(isinstance(config, dict), "Benchmark configuration is required")
    for name in ("contexts", "prefill_contexts", "concurrency"):
        require(
            isinstance(config.get(name), list)
            and config[name]
            and all(type(n) is int and n > 0 for n in config[name]),
            "Benchmark dimensions must be positive integer lists",
        )
    require(
        max(config["concurrency"]) <= 16,
        "Qualification concurrency exceeds the bounded matrix",
    )
    require(
        type(config.get("max_tokens")) is int and 64 <= config["max_tokens"] <= 4096,
        "Output length is outside the qualification budget",
    )
    require(
        type(config.get("duration")) in (int, float) and 5 <= config["duration"] <= 60,
        "Invalid benchmark duration",
    )
    require(
        type(config.get("temperature")) in (int, float)
        and 0 <= config["temperature"] <= 2,
        "Explicit sampling temperature is required",
    )
    require(
        config.get("repeats", 3) == 3, "Qualification uses three matched repetitions"
    )
    require(
        set(config.get("limits", {})) == {"prefill", "decode", "steps"},
        "All throughput thresholds must be explicit",
    )


def matched_metadata(path, config, model):
    value = json.loads(Path(path).read_text())["metadata"]
    expected = {
        "model": model,
        "decode_mode": "duration",
        "duration_per_test": config["duration"],
        "max_tokens": config["max_tokens"],
        "temperature": config["temperature"],
        "concurrency_levels": config["concurrency"],
        "context_lengths": config["contexts"],
        "ignore_eos": True,
    }
    require(
        all(value.get(key) == item for key, item in expected.items()),
        "Benchmark workload differs from the protected comparison",
    )
    require(
        value.get("loop_detection", {}).get("enabled") is True,
        "Benchmark loop detection must remain enabled",
    )


def run(config, *, host, port, model, output, policy_root, timeout=1800):
    validate_config(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    root = Path(policy_root).resolve()
    script = (root / config["script"]).resolve()
    require(
        script.is_file()
        and not script.is_symlink()
        and sha(script.read_bytes()) == config["script_sha256"],
        "Benchmark script differs from its approved identity",
    )
    references = []
    for item in config["baseline"]:
        path = (root / item["path"]).resolve()
        require(
            path.is_file()
            and not path.is_symlink()
            and sha(path.read_bytes()) == item["sha256"],
            "Baseline benchmark bytes differ",
        )
        references.append(path)
        matched_metadata(path, config, model)
    baseline = benchmark_measurements(references)
    paths = []
    for index in range(3):
        destination = (output / f"candidate-{index + 1}.json").resolve()
        argv = [
            config.get("python", sys.executable),
            str(script),
            "--host",
            host,
            "--port",
            str(port),
            "--model",
            model,
            "--no-hw-monitor",
            "--no-resume",
            "--no-calibration-cache",
            "--display-mode",
            "plain",
            "--token-targeting",
            "exact",
            "--concurrency",
            ",".join(map(str, config["concurrency"])),
            "--contexts",
            ",".join(map(str, config["contexts"])),
            "--prefill-contexts",
            ",".join(map(str, config["prefill_contexts"])),
            "--max-tokens",
            str(config["max_tokens"]),
            "--duration",
            str(config["duration"]),
            "--temperature",
            str(config["temperature"]),
            "--output",
            str(destination),
        ]
        result = command(argv, cwd=output, seconds=timeout, limit=8 * 1024**2)
        (output / f"candidate-{index + 1}.stdout.log").write_bytes(result["stdout"])
        (output / f"candidate-{index + 1}.stderr.log").write_bytes(result["stderr"])
        require(
            not result["uncertain"] and result["returncode"] == 0,
            "Benchmark process failed or exceeded its budget",
        )
        require(
            sha(script.read_bytes()) == config["script_sha256"],
            "Benchmark script changed during measurement",
        )
        paths.append(destination)
        matched_metadata(destination, config, model)
    candidate = benchmark_measurements(paths)
    comparison = compare(
        candidate["measurements"], baseline["measurements"], config["limits"]
    )
    report = {
        "schema": "sparkring-throughput-comparison/v1",
        "candidate": candidate,
        "baseline": baseline,
        "benchmark_sha256": config["script_sha256"],
        "config": config,
        **comparison,
    }
    write_json(output / "comparison.json", report)
    return report
