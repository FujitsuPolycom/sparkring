#!/usr/bin/env python3
"""Install (optionally), check and measure one installer profile on one cluster, then draft its record.

Steps, in order: install, readiness, functional, stress, throughput, record;
--full-matrix adds the matrix step before record.
Each step saves its result in --out; a later invocation with the same --out
reuses every complete saved result and runs only what is missing, so an
interrupted run resumes where it stopped. --redo STEP sets a step's saved
outputs aside (renamed, never deleted) and runs it again. See README.md.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Callable

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from performance.harnesses.acceptance import checks, install, profile_info, record, runners, throughput  # noqa: E402

STEPS = ("readiness", "functional", "stress", "throughput", "record")
# Runs only when requested (--full-matrix, or named in --steps): it takes much
# longer than the standard measurement.
OPTIONAL = ("matrix",)
RESULTS = {"install": "install.json", "readiness": "readiness.json", "functional": "functional.json",
           "stress": "stress.json", "throughput": "throughput.json", "matrix": "matrix.json", "record": "record.json"}
OUTPUTS = {"install": ("install-launch.json", "install", "install.json"), "readiness": ("readiness.json",),
           "functional": ("functional.json", "functional.txt"), "stress": ("stress.json", "stress-responses.json"),
           "throughput": ("throughput", "throughput.json"), "matrix": ("matrix", "matrix.json"),
           "record": ("record.json",)}
DEFAULT_CLIENT = "a separate machine on Node A's network"


def harness_revision():
    """This checkout's commit, marked when the harness directory has uncommitted changes."""
    try:
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=12", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "performance/harnesses/acceptance"],
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return head + (" with uncommitted harness changes" if dirty else "")


def _log(line):
    print(line, file=sys.stderr, flush=True)


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc)


@dataclass
class Environment:
    """Every effect outside the output directory and the record directory; tests replace these."""
    ssh: Callable = runners.SshRunner
    http: object = field(default_factory=runners.HttpClient)
    run_process: Callable = runners.run_process
    clock: Callable = time.monotonic
    sleep: Callable = time.sleep
    now: Callable = _utc_now
    log: Callable = _log
    hostname: Callable = socket.gethostname
    revision: Callable = harness_revision


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_text(path, text):
    path = Path(path)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def write_json(path, value):
    write_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def split_target(target):
    """`user@host` -> (user, host); `alias` -> (None, alias)."""
    if not target:
        return None, None
    user, _, host = target.rpartition("@")
    return user or None, host


class Acceptance:
    def __init__(self, args, profile, env):
        self.args, self.profile, self.env = args, profile, env
        self.out = Path(args.out)
        self.base_url = f"http://{args.api_host}:{profile.port}/v1"
        self.source = install.parse_source(args.install) if args.install else None

    def stamp(self):
        return self.env.now().strftime("%Y%m%dT%H%M%SZ")

    def saved(self, step):
        path = self.out / RESULTS[step]
        if not path.is_file():
            return None
        result = read_json(path)
        return result if result.get("complete") else None

    def set_aside(self, step):
        suffix = self.stamp()
        for name in OUTPUTS[step]:
            path = self.out / name
            if path.exists():
                target = path.with_name(f"{path.name}.{suffix}")
                path.rename(target)
                self.env.log(f"{step}: kept earlier output as {target.name}")

    def run(self):
        self.out.mkdir(parents=True, exist_ok=True)
        for step in self.args.redo:
            self.set_aside(step)
        steps = (["install"] if self.source else []) + list(self.args.steps)
        if self.args.full_matrix and "matrix" not in steps:
            steps.insert(steps.index("record") if "record" in steps else len(steps), "matrix")
        ok = True
        for step in steps:
            result = self.saved(step)
            if result is not None:
                self.env.log(f"{step}: using the saved result in {self.out / RESULTS[step]}")
            else:
                try:
                    result = getattr(self, "step_" + step)()
                except install.InstallError as error:
                    self.env.log(f"{step}: {error}")
                    return 1
                if result.get("complete"):
                    write_json(self.out / RESULTS[step], result)
            if not result.get("ok"):
                ok = False
                reason = "; ".join(result.get("problems") or [result.get("error") or "check failed"])
                self.env.log(f"{step}: not passed: {reason}")
                if step in ("install", "readiness") or not result.get("complete"):
                    self.env.log("Stopping before the remaining steps.")
                    return 1
        return 0 if ok else 1

    def step_install(self):
        ssh = self.env.ssh(self.args.node_a)
        launch_path = self.out / "install-launch.json"
        if launch_path.is_file():
            launch = read_json(launch_path)
            self.env.log(f"install: following run {launch['run_id']} on Node A, started {launch['started_at']}")
        else:
            run_id = f"{self.profile.id}-{self.stamp()}"
            command = install.install_command(self.source, self.profile.id, self.args.install_arg)
            launch = {"run_id": run_id, "command": command, "source": asdict(self.source),
                      "started_at": self.env.now().isoformat(timespec="seconds")}
            # Saved before the launch, so a later invocation follows this run
            # instead of starting a second installer.
            write_json(launch_path, launch)
            self.env.log(f"install: starting on Node A: {command}")
            install.launch(ssh, run_id, command)
        run_id = launch["run_id"]
        exit_code = install.wait(ssh, run_id, timeout=self.args.install_timeout, interval=self.args.poll_interval,
                                 clock=self.env.clock, sleep=self.env.sleep, log=self.env.log)
        stdout, stderr = install.fetch(ssh, run_id, "stdout.json"), install.fetch(ssh, run_id, "stderr.log")
        directory = self.out / "install"
        directory.mkdir(exist_ok=True)
        write_text(directory / "stdout.json", stdout)
        write_text(directory / "stderr.log", stderr)
        write_text(directory / "exit_code", f"{exit_code}\n")
        try:
            summary = install.summarize(stdout, stderr, exit_code, profile=self.profile.id)
        except install.InstallError as error:
            summary = {"state": None, "exit_code": exit_code, "ok": False, "problems": [str(error)]}
        return {**summary, "run_id": run_id, "source": launch["source"], "complete": True}

    def step_readiness(self):
        result = checks.wait_ready(self.env.http, self.base_url, expected=self.profile.served_model_name,
                                   timeout=self.args.ready_timeout, interval=min(self.args.poll_interval, 10),
                                   clock=self.env.clock, sleep=self.env.sleep, log=self.env.log)
        if result["ok"]:
            self.env.log(f"readiness: {self.profile.served_model_name} listed after {result['seconds']} s")
        return {**result, "complete": result["ok"]}

    def chat(self):
        return checks.Chat(self.env.http, self.base_url, self.profile.served_model_name,
                           thinking_off=self.profile.thinking_off, thinking_on=self.profile.thinking_on)

    def step_functional(self):
        result = checks.run_functional(self.chat(), features=self.profile.features, skip=set(self.args.skip_check),
                                       log=lambda line: self.env.log("functional: " + line))
        write_text(self.out / "functional.txt", checks.functional_text(result))
        return {**result, "complete": True}

    def step_stress(self):
        self.env.log(f"stress: {self.args.stress_rounds} rounds of 32 requests")
        summary, responses = checks.run_stress(self.chat(), rounds=self.args.stress_rounds, clock=self.env.clock)
        write_json(self.out / "stress-responses.json", responses)
        self.env.log(f"stress: n={summary['n']} degenerate={summary['degenerate']} wrong={summary['wrong']} "
                     f"errors={summary['errors']} seconds={summary['seconds']}")
        return {**summary, "complete": True}

    def step_throughput(self):
        if not self.args.bench_dir or not (Path(self.args.bench_dir) / throughput.BENCH_PROGRAM).is_file():
            raise ValueError(f"--bench-dir must name the directory holding {throughput.BENCH_PROGRAM}")
        directory = self.out / "throughput"
        directory.mkdir(exist_ok=True)
        runs, files = [], []
        repeats = self.args.repeats
        for index in range(1, repeats + 1):
            stem = f"tp{self.profile.nodes}-matrix" + ("" if repeats == 1 else f"-run{index}")
            matrix_path = directory / f"{stem}.json"
            if not self._readable(matrix_path):
                if matrix_path.exists():
                    matrix_path.rename(matrix_path.with_name(f"{matrix_path.name}.{self.stamp()}"))
                command = throughput.bench_command(self.args.bench_python, self.args.bench_dir,
                                                   host=self.args.api_host, port=self.profile.port,
                                                   model=self.profile.served_model_name, output=matrix_path.resolve())
                self.env.log(f"throughput: run {index} of {repeats}; log {directory / (stem + '.log')}")
                code = self.env.run_process(command, log_path=directory / f"{stem}.log", cwd=self.args.bench_dir,
                                            timeout=self.args.bench_timeout)
                if code != 0 or not self._readable(matrix_path):
                    return {"ok": False, "complete": False,
                            "error": f"benchmark run {index} exited {code} without a matrix; see {stem}.log"}
            runs.append(throughput.extract(read_json(matrix_path)))
            files.append(matrix_path.name)
        summary = {**throughput.combine(runs), "files": files, "complete": True}
        print(record.readme_line(self.profile, summary), flush=True)
        return summary

    def step_matrix(self):
        if not self.args.bench_dir or not (Path(self.args.bench_dir) / throughput.BENCH_PROGRAM).is_file():
            raise ValueError(f"--bench-dir must name the directory holding {throughput.BENCH_PROGRAM}")
        directory = self.out / "matrix"
        directory.mkdir(exist_ok=True)
        stem = f"tp{self.profile.nodes}-full-matrix"
        matrix_path = directory / f"{stem}.json"
        if not self._readable(matrix_path):
            if matrix_path.exists():
                matrix_path.rename(matrix_path.with_name(f"{matrix_path.name}.{self.stamp()}"))
            command = throughput.bench_command(self.args.bench_python, self.args.bench_dir,
                                               host=self.args.api_host, port=self.profile.port,
                                               model=self.profile.served_model_name, output=matrix_path.resolve(),
                                               full=True)
            self.env.log(f"matrix: full matrix; log {directory / (stem + '.log')}")
            code = self.env.run_process(command, log_path=directory / f"{stem}.log", cwd=self.args.bench_dir,
                                        timeout=self.args.bench_timeout * 4)
            if code != 0 or not self._readable(matrix_path):
                return {"ok": False, "complete": False,
                        "error": f"the full matrix exited {code} without a matrix; see {stem}.log"}
        full = throughput.extract_full(read_json(matrix_path))
        failed = [f"{context} tokens, {level} streams: {cell['failure']}"
                  for context, row in full["decode"].items() for level, cell in row.items() if cell.get("failure")]
        return {**full, "file": matrix_path.name, "problems": failed, "ok": not failed, "complete": True}

    def _installed_release(self, installed):
        """The image release from the saved installation, reading its result document if the summary predates the field."""
        if not installed:
            return None
        if "image_release" in installed:
            return installed["image_release"]
        stdout = self.out / "install/stdout.json"
        if not stdout.is_file():
            return None
        try:
            document, _ = install.result_document(stdout.read_text(encoding="utf-8"))
        except install.InstallError:
            return None
        return install.image_release(document)

    @staticmethod
    def _readable(path):
        try:
            return isinstance(read_json(path).get("results"), list)
        except (OSError, ValueError, AttributeError):
            return False

    def step_record(self):
        args, profile = self.args, self.profile
        installed = self.saved("install")
        release = self._installed_release(installed)
        if release and release != profile.image:
            # Installed with --image-lock: the record names the image that served.
            profile = replace(profile, image=release, release=f"runtime/releases/{release}/release.json")
        date = args.date or self.env.now().strftime("%Y%m%d")
        name = record.record_name(profile.image, args.topic or profile.id, date)
        record_root = Path(args.record_root)
        markdown_path, directory = record_root / f"{name}.md", record_root / name
        if markdown_path.exists() or directory.exists():
            raise ValueError(f"{markdown_path} or {directory} already exists; choose another --topic or --date")
        user, host = split_target(args.node_a)
        names = [n for n in {self.env.hostname(), host, args.api_host, *args.private_name} if n]
        users = [user] if user else []
        replace_map = {args.api_host: "NODE_A", **({host: "NODE_A"} if host else {})}

        def text(value):
            return record.sanitize_text(value, replace=replace_map, names=names, users=users)
        launch = None
        if installed and not installed.get("ok"):
            raise ValueError("The saved installation did not complete; a record needs a complete installation "
                             "(--redo install) or a run without one in a separate --out directory")
        if installed and (self.out / "install-launch.json").is_file():
            launch = read_json(self.out / "install-launch.json")
        functional, stress, summary = self.saved("functional"), self.saved("stress"), self.saved("throughput")
        full = self.saved("matrix")
        contents, files = {}, {}
        if installed and (self.out / "install/stderr.log").is_file():
            files["install_phases"] = "install-phases.txt"
            contents["install-phases.txt"] = text(install.phases((self.out / "install/stderr.log").read_text(encoding="utf-8")))
        if functional:
            files["functional"] = "functional.txt"
            contents["functional.txt"] = text(checks.functional_text(functional))
        if stress:
            files["stress"] = "stress.json"
            public = {key: value for key, value in stress.items() if key != "complete"}
            contents["stress.json"] = text(json.dumps(public, indent=2, ensure_ascii=False) + "\n")
        if summary:
            files["matrices"] = summary["files"]
            for file_name in summary["files"]:
                matrix = throughput.sanitize_matrix(read_json(self.out / "throughput" / file_name),
                                                    names=names, accounts=users)
                contents[file_name] = json.dumps(matrix, indent=2, ensure_ascii=False) + "\n"
        if full:
            files["full_matrix"] = full["file"]
            matrix = throughput.sanitize_matrix(read_json(self.out / "matrix" / full["file"]), names=names, accounts=users)
            contents[full["file"]] = json.dumps(matrix, indent=2, ensure_ascii=False) + "\n"
        source = install.Source(**launch["source"]) if launch else None
        markdown = record.render(profile=profile, name=name, record_dir=record_root, repo_root=ROOT, files=files,
                                 install=installed, source=source, functional=functional, stress=stress, summary=summary, full=full, status=args.status,
                                 client=args.client, harness_revision=self.env.revision())
        markdown = text(markdown)
        # Every file passed its private-data check before anything is written.
        directory.mkdir(parents=True)
        for file_name, value in contents.items():
            write_text(directory / file_name, value)
        write_text(markdown_path, markdown)
        self.env.log(f"record: wrote {markdown_path} and {len(contents)} files in {directory}")
        result = {"record": str(markdown_path), "directory": str(directory), "complete": True, "ok": True}
        if summary:
            result["readme"] = record.readme_line(profile, summary)
            print(result["readme"], flush=True)
        return result


def step_list(value):
    items = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in items if item not in (*STEPS, *OPTIONAL, "install")]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown step {', '.join(unknown)}; steps are install, {', '.join((*STEPS, *OPTIONAL))}")
    return items


def json_object(value):
    try:
        parsed = json.loads(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"not JSON: {error}") from None
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("expected a JSON object of request fields")
    return parsed


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--profile", required=True, help="installer profile ID from profiles/catalog.json")
    p.add_argument("--api-host", required=True, help="Node A's address as the client reaches it; the port comes "
                                                     "from the profile")
    p.add_argument("--node-a", help="SSH target of Node A (user@host or an ssh alias); required with --install")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--install", metavar="SOURCE",
                      help="install first: published, published:REF, or bundle:PATH_ON_NODE_A:REF")
    mode.add_argument("--skip-install", action="store_true", help="measure the deployment already running")
    p.add_argument("--steps", type=step_list, default=list(STEPS),
                   help=f"comma-separated steps after installation (default {','.join(STEPS)})")
    p.add_argument("--redo", type=step_list, default=[], help="set these steps' saved outputs aside and run them again")
    p.add_argument("--out", required=True, help="directory for raw outputs and saved step results; keep it outside Git")
    p.add_argument("--full-matrix", action="store_true",
                   help="also measure the full matrix: 1, 2, 4, 8 and 16 streams at 8K, 32K, 64K and 128K context")
    p.add_argument("--stress-rounds", type=positive, default=8, help="correctness-screen rounds of 32 requests (default 8)")
    p.add_argument("--repeats", type=positive, default=1, help="benchmark runs; the record reports medians (default 1)")
    p.add_argument("--bench-dir", help="llm-inference-bench checkout holding llm_decode_bench.py")
    p.add_argument("--bench-python", default=sys.executable, help="Python that runs the benchmark (default: this one)")
    p.add_argument("--record-root", default=str(ROOT / "performance/records/images"), help="where the record is written")
    p.add_argument("--topic", help="record name topic (default: the profile ID)")
    p.add_argument("--date", help="record name date, YYYYMMDD (default: today, UTC)")
    p.add_argument("--status", choices=record.STATUSES, help="record status (default: the profile's status)")
    p.add_argument("--client", default=DEFAULT_CLIENT, help=f"client description in the record (default: {DEFAULT_CLIENT})")
    p.add_argument("--skip-check", action="append", default=[], type=lambda value: value.replace("-", " "),
                   choices=[c[0] for c in checks.FUNCTIONAL_CHECKS], metavar="CHECK",
                   help="skip one functional check (count, arithmetic, code, tool-call, forced-tool-call, image, "
                        "thinking-on); repeatable")
    p.add_argument("--thinking-off", type=json_object, help="request fields that disable thinking, replacing the "
                                                            "profile's smoke settings")
    p.add_argument("--thinking-on", type=json_object, help="request fields for the reasoning check (default: none)")
    p.add_argument("--private-name", action="append", default=[],
                   help="another host or name the record must not contain; repeatable")
    p.add_argument("--install-arg", action="append", default=[], metavar="ARG",
                   help="further argument for sparkring install, repeatable (e.g. --install-arg=--image-lock "
                        "--install-arg=/var/tmp/lock.json)")
    p.add_argument("--install-timeout", type=positive, default=14400, help="seconds to wait for the installer")
    p.add_argument("--ready-timeout", type=positive, default=3600, help="seconds to wait for /v1/models")
    p.add_argument("--bench-timeout", type=positive, default=3600, help="seconds per benchmark run")
    p.add_argument("--poll-interval", type=positive, default=30, help="seconds between installer polls")
    return p


def main(argv=None, env=None):
    env = env or Environment()
    args = parser().parse_args(argv)
    if args.install and not args.node_a:
        env.log("error: --install needs --node-a")
        return 2
    if "install" in args.steps:
        env.log("error: --steps lists the steps after installation; --install selects installation")
        return 2
    try:
        profile = profile_info.load(args.profile)
        if args.thinking_off is not None:
            profile = replace(profile, thinking_off=args.thinking_off)
        if args.thinking_on is not None:
            profile = replace(profile, thinking_on=args.thinking_on)
        return Acceptance(args, profile, env).run()
    except (ValueError, OSError) as error:
        env.log(f"error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
