"""``plan`` and ``run`` of the serving A/B runner (README.md in this directory)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from performance.harnesses.serving_ab import measure, remote, report, setlock, spec, verify  # noqa: E402
from runtime.common import qwen_flash_next  # noqa: E402

SIRCL = ROOT / "spark_transport" / "sircl"
CACHE_PLACEHOLDER = "/serving-ab-cache"


def log(message: str, out: Path | None = None) -> None:
    line = f"[{time.strftime('%H:%M:%S', time.gmtime())}] {message}"
    print(line, flush=True)
    if out is not None:
        with open(out / "campaign.log", "a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def positions_of(text: str) -> list[int]:
    if "-" in text and "," not in text:
        a, b = (int(x) for x in text.split("-"))
        return list(range(a, b + 1))
    return [int(x) for x in text.split(",")]


def order_of(text: str) -> list[tuple[str, str, bool]]:
    """``W:S+,W:N,S+,N`` -> [(label, arm, warm-up)]: warm-ups ``W-<arm>``, measured starts numbered per arm."""
    out, counts = [], {}
    for item in text.split(","):
        warm = item.startswith("W:")
        arm = item[2:] if warm else item
        if arm not in spec.ARMS:
            raise SystemExit(f"--order: unknown arm {arm!r}")
        if warm:
            out.append((f"W-{arm}", arm, True))
        else:
            counts[arm] = counts.get(arm, 0) + 1
            out.append((f"{arm}{counts[arm]}", arm, False))
    return out


def profile_config(profile_id: str) -> tuple[dict, Path]:
    deployment = json.loads((ROOT / "profiles" / profile_id / "profile.json").read_text(encoding="utf-8"))
    path = ROOT / deployment["configuration"]["path"]
    return qwen_flash_next.canonical(qwen_flash_next.read(path)), path


def arg_value(arguments: list[str], flag: str) -> str:
    return arguments[arguments.index(flag) + 1]


def bundle(site_path: str, positions: list[int], run_id: str, nccl: str, args, stage: bool = False,
           fused_norm: bool = False) -> dict:
    command = [sys.executable, "-m", "sparkring_sircl.vllm.serve", "bundle", "--site", site_path,
               "--positions", ",".join(map(str, positions)), "--run-id", run_id, "--nccl", nccl]
    if nccl == "never":
        command += ["--require-no-nccl"]
        for flag in ("capacity", "dispatch"):
            if getattr(args, flag):
                command += [f"--{flag}", str(getattr(args, flag))]
        if args.tuning_table:
            command += ["--tuning-table", args.tuning_table]
        if fused_norm:
            command += ["--fused-norm", "on"]
    if stage:
        command += ["--stage", "--image", args.image.removeprefix("sha256:")]
    result = subprocess.run(command, cwd=SIRCL, capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(f"bundle {run_id}: {result.stderr.strip()[-1500:]}")
    return json.loads(result.stdout)


def build_plan(args) -> dict:
    site = remote.load_site(args.site)
    positions = positions_of(args.positions)
    arms = list(dict.fromkeys(arm for _, arm, _ in order_of(args.order)))
    profile, config_path = profile_config(args.profile)
    world = qwen_flash_next.node_count(profile)
    if world != len(positions):
        raise SystemExit(f"{args.profile} runs {world} ranks; --positions names {len(positions)}")
    sparks = [site.sparks[p] for p in positions]
    explicit = dict(item.split("=", 1) for item in args.model_path)
    checkpoints = []
    for spark in sparks:
        if str(spark.position) in explicit:
            checkpoints.append({"spark": spark.name, "position": spark.position,
                                "chosen": {"path": explicit[str(spark.position)], "matches": None}, "candidates": []})
        else:
            checkpoints.append(remote.find_checkpoint(spark, profile["model"]))
    missing = [c["spark"] for c in checkpoints if not c["chosen"]]
    if missing:
        raise SystemExit(f"no verified copy of {profile['model']['repository']}@{profile['model']['revision']} on "
                         f"{missing}; candidates: {[c['candidates'] for c in checkpoints if not c['chosen']]}")
    master = sparks[0].lan_address
    bases = [qwen_flash_next.render(profile, rank=r, master=master, host_ip=spark.lan_address,
                                    interface=site.lan_interface, image=args.image, model=checkpoints[r]["chosen"]["path"],
                                    cache=CACHE_PLACEHOLDER, remote=True)
             for r, spark in enumerate(sparks)]
    run = args.run_id
    bundles = {arm: bundle(args.site, positions, f"{run}-{spec.slug(arm)}", "never", args, fused_norm=arm == "S+")
               for arm in arms if arm in spec.SIRCL_ARMS}
    auto = bundle(args.site, positions, f"{run}-auto", "auto", args) if "N" in arms else None
    # S and S+ differ only in SIRCL settings and share compile and JIT caches; every other arm has its own.
    caches = {arm: f"{site.remote_dir}/serving-ab/{run}/cache-{spec.slug('S' if arm in spec.SIRCL_ARMS else arm)}"
              for arm in arms}
    extra = dict(item.split("=", 1) for item in args.sircl_env)
    commands = {arm: spec.render_arm(arm, bases, docker=sparks[0].docker, run=run, models=[c["chosen"]["path"]
                                                                                            for c in checkpoints],
                                     caches=caches, host_ips=[s.lan_address for s in sparks],
                                     bundle=bundles.get(arm), bundle_auto=auto, ring_size=site.size,
                                     positions=positions, roce_slot=args.roce_slot, extra_env=extra)
                for arm in arms}
    arguments = list(profile["vllm_args"])
    return {"schema": "serving-ab-plan/v1", "profile": args.profile, "config": str(config_path.relative_to(ROOT)),
            "model": profile["model"], "served_model_name": profile["served_model_name"], "image": args.image,
            "positions": positions, "sparks": [s.name for s in sparks], "api": f"http://{master}:{arg_value(arguments, '--port')}",
            "port": int(arg_value(arguments, "--port")), "context_limit": int(arg_value(arguments, "--max-model-len")),
            "order": [list(item) for item in order_of(args.order)], "metrics": args.metrics, "arms": arms,
            "checkpoints": checkpoints, "caches": caches, "sircl_env": extra, "roce_slot": args.roce_slot,
            "bundles": {arm: {"run_id": b["run_id"], "nccl": b["nccl"], "tuning": b.get("tuning"),
                              "remote": b["remote"]} for arm, b in bundles.items()},
            "nccl_auto_policy": auto and auto["nccl"], "commands": commands}


def print_plan(plan: dict) -> None:
    print(f"profile {plan['profile']} ({plan['served_model_name']}) on positions {plan['positions']} "
          f"{plan['sparks']}, image {plan['image']}, API {plan['api']}")
    print(f"order {[label for label, _, _ in plan['order']]}, metrics {plan['metrics']}")
    print("checkpoints:")
    for c in plan["checkpoints"]:
        chosen = c["chosen"]
        print(f"  position {c['position']} {c['spark']}: {chosen['path']}"
              + (f" ({chosen.get('shards')} shards, {chosen.get('shard_bytes')} bytes, config and index match)"
                 if chosen.get("matches") else ""))
    for arm, b in plan["bundles"].items():
        tables = (b.get("tuning") or {}).get("tables") or []
        print(f"arm {arm}: bundle {b['run_id']} NCCL policy {b['nccl']}; tuning tables "
              + (", ".join(f"{t['hash']} ({t['key'].get('shape')}, {t['key'].get('world')} ranks)" for t in tables) or "none"))
    if plan["nccl_auto_policy"]:
        print(f"arm N: NCCL policy of this group under --nccl auto: {plan['nccl_auto_policy']}")
    for arm, ranks in plan["commands"].items():
        for r, tokens in enumerate(ranks):
            print(f"\n# arm {arm}, rank {r} ({plan['sparks'][r]})\n{spec.shell(tokens)}")
    sircl = next((a for a in plan["arms"] if a in spec.SIRCL_ARMS), None)
    if sircl and "N" in plan["arms"]:
        print(f"\n# {sircl} against N, rank 0")
        print(json.dumps(spec.diff(plan["commands"][sircl][0], plan["commands"]["N"][0]), indent=1))


def start(plan: dict, site: remote.Site, arm: str, out: Path) -> None:
    sparks = [site.sparks[p] for p in plan["positions"]]
    for spark, tokens in zip(sparks, plan["commands"][arm]):
        remote.run(spark, remote.sudo(f"mkdir -p {plan['caches'][arm]}"))
        remote.run(spark, "bash -s", stdin=("set -e\n" + spec.shell(tokens) + "\n").encode())
    deadline = time.monotonic() + 3600
    while time.monotonic() < deadline:
        try:
            body = urllib.request.urlopen(plan["api"] + "/v1/models", timeout=10).read().decode()
            if plan["served_model_name"] in body:
                return
        except OSError:
            pass
        states = [remote.run(s, remote.sudo(f"docker inspect --format '{{{{.State.Status}}}}' {name_of(t)}"),
                             check=False).strip() for s, t in zip(sparks, plan["commands"][arm])]
        if any(state != "running" for state in states):
            raise RuntimeError(f"arm {arm}: containers {states}")
        time.sleep(30)
    raise RuntimeError(f"arm {arm}: not ready within 3600 s")


def name_of(tokens: list[str]) -> str:
    return tokens[tokens.index("--name") + 1]


def receipts(plan: dict, site: remote.Site, arm: str) -> list[dict | None]:
    if arm not in spec.SIRCL_ARMS:
        return []
    directory = plan["bundles"][arm]["remote"]["receipts"]
    out = []
    for r, p in enumerate(plan["positions"]):
        text = remote.run(site.sparks[p], remote.sudo(f"cat {directory}/rank{r}-tp-0.json"), check=False)
        out.append(json.loads(text) if text.strip() else None)
    return out


def collect(plan: dict, site: remote.Site, arm: str, out: Path) -> list[str]:
    logs = []
    for r, (p, tokens) in enumerate(zip(plan["positions"], plan["commands"][arm])):
        text = remote.run(site.sparks[p], remote.sudo(f"docker logs {name_of(tokens)} 2>&1"), check=False, timeout=600)
        (out / f"log-r{r}.txt").write_text(text, encoding="utf-8")
        (out / f"inspect-r{r}.json").write_text(
            remote.run(site.sparks[p], remote.sudo(f"docker inspect {name_of(tokens)}"), check=False), encoding="utf-8")
        logs.append(text)
    return logs


def stop(plan: dict, site: remote.Site, arm: str) -> int:
    for p, tokens in zip(plan["positions"], plan["commands"][arm]):
        remote.run(site.sparks[p], remote.sudo(f"docker rm -f {name_of(tokens)}"), check=False)
    return sum(int(remote.run(site.sparks[p], remote.sudo(f"docker ps -q --filter label=serving-ab={plan['run_id']} | wc -l"),
                              check=False).strip() or 0) for p in plan["positions"])


def run_campaign(args) -> int:
    plan = build_plan(args)
    plan["run_id"] = args.run_id
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
    site = remote.load_site(args.site)
    lock_host = site.sparks[args.lock_host]
    holder = (setlock.SetLock(lock_host, args.lock_name, plan["positions"], log=lambda m: log(m, out))
              if args.lock == "set" else setlock.Held(log=lambda m: log(m, out)))
    with holder:
        for arm in plan["arms"]:
            if arm in spec.SIRCL_ARMS:
                log(f"stage arm {arm}", out)
                bundle(args.site, plan["positions"], f"{args.run_id}-{spec.slug(arm)}", "never", args, stage=True,
                       fused_norm=arm == "S+")
        measured = []
        for label, arm, warm in plan["order"]:
            d = out / label
            d.mkdir(parents=True, exist_ok=True)
            log(f"{label}: start arm {arm}", out)
            t0 = time.monotonic()
            try:
                start(plan, site, arm, d)
            except Exception as error:
                log(f"{label}: {error}", out)
                collect(plan, site, arm, d)
                stop(plan, site, arm)
                return 1
            (d / "ready.json").write_text(json.dumps({"ready_seconds": round(time.monotonic() - t0)}), encoding="utf-8")
            logs = collect(plan, site, arm, d)
            evidence = verify.check(arm, logs, receipts(plan, site, arm))
            (d / "evidence.json").write_text(json.dumps(evidence, indent=1), encoding="utf-8")
            log(f"{label}: ready after {round(time.monotonic() - t0)} s; arm checks "
                f"{'passed' if evidence['passed'] else 'FAILED: ' + '; '.join(evidence['problems'])}", out)
            if not evidence["passed"]:
                stop(plan, site, arm)
                return 1
            codes = measure.measure(d, host=site.sparks[plan["positions"][0]].lan_address, port=plan["port"],
                                    model=plan["served_model_name"], context_limit=plan["context_limit"],
                                    bench_dir=args.bench_dir, metrics="warmup" if warm else args.metrics)
            (d / "measure.json").write_text(json.dumps(codes, indent=1), encoding="utf-8")
            if arm == "S+":
                after = [verify.fused_calls(r) for r in receipts(plan, site, arm)]
                (d / "fused-after.json").write_text(json.dumps(after, indent=1), encoding="utf-8")
            collect(plan, site, arm, d)
            left = stop(plan, site, arm)
            log(f"{label}: measured {codes}; stopped ({left} campaign containers left)", out)
            if not warm:
                measured.append(label)
            if left:
                return 1
        doc = report.summary(out, measured)
        (out / "summary.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
        (out / "tables.txt").write_text(report.tables(doc) + "\n", encoding="utf-8")
        print(report.tables(doc))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m performance.harnesses.serving_ab",
                                     description="Serve one installer profile under several transports and compare them.")
    parser.add_argument("action", choices=("plan", "run"))
    parser.add_argument("--site", required=True, help="sircl-ring-site/v1 site file (hosts, SSH targets, LAN addresses)")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--positions", required=True, help="fabric positions, rank r at the r-th: 0-1, 2-3, 0-7")
    parser.add_argument("--image", required=True, help="image ID, sha256:<64 hex>, present on every Spark")
    parser.add_argument("--order", required=True, help="starts in order: W:<arm> is a warm-up, e.g. W:S+,W:N,S+,N,N,S+")
    parser.add_argument("--metrics", choices=tuple(measure.METRICS), default="phase1")
    parser.add_argument("--run-id", required=True, help="names containers, caches and SIRCL run directories")
    parser.add_argument("--output", required=True, help="result directory on this machine")
    parser.add_argument("--lock", choices=("set", "global-held"), default="set",
                        help="set: take the set lock of --positions (setlock.py); global-held: the caller already holds "
                             "the ring's global lock (bash hwlock.sh OWNER python -m ...), so take none")
    parser.add_argument("--lock-name", default=None, help="set-lock owner name (default serve-<positions>)")
    parser.add_argument("--lock-host", type=int, default=0, help="site position of the lock host (default 0)")
    parser.add_argument("--tuning-table", help="SIRCL tuning table for the S arms (bundle --tuning-table)")
    parser.add_argument("--capacity", type=int, help="SIRCL all-reduce capacity of the S arms (bundle --capacity)")
    parser.add_argument("--dispatch", type=int, help="SIRCL dispatch ceiling of the S arms (bundle --dispatch)")
    parser.add_argument("--sircl-env", action="append", default=[], metavar="KEY=VALUE",
                        help="SIRCL_* variable added to the S arms at the container level, e.g. SIRCL_REDUCE_LINK_BLOCKS=1")
    parser.add_argument("--roce-slot", action="store_true",
                        help="S arms keep VLLM_ENABLE_ROCE_ALLREDUCE=1 so SIRCL's roce_slot shim takes vLLM's RoCE slot")
    parser.add_argument("--model-path", action="append", default=[], metavar="POSITION=PATH",
                        help="checkpoint directory of one position instead of discovering it")
    parser.add_argument("--bench-dir", default=str(Path.home() / "llm-inference-bench"),
                        help="llm-inference-bench checkout holding llm_decode_bench.py")
    args = parser.parse_args(argv)
    if not args.image.startswith("sha256:"):
        args.image = "sha256:" + args.image
    args.lock_name = args.lock_name or "serve-" + args.positions.replace(",", "-")
    if args.action == "plan":
        plan = build_plan(args)
        print_plan(plan)
        Path(args.output).mkdir(parents=True, exist_ok=True)
        (Path(args.output) / "plan.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
        return 0
    return run_campaign(args)
