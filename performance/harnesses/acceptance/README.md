# Profile acceptance harness

Status: **implemented**. Offline tests replace SSH, HTTP and the benchmark
with fakes; they do not qualify any profile.

`accept_profile.py` qualifies one installer profile on one cluster: it can
install the profile with the one-line installer, checks that the API serves
the profile's model, runs functional checks and a correctness screen,
measures throughput with llm-inference-bench, and drafts an evidence record
in `performance/records/images/`. The drafted prose states only what the
saved results show; edit it before committing.

## Steps

| Step | Action | Saved in `--out` |
|---|---|---|
| `install` | Only with `--install`. Runs `install.sh --profile ID --yes --json` on Node A, detached from the SSH session, and polls until it exits | `install-launch.json`, `install/stdout.json`, `install/stderr.log`, `install/exit_code`, `install.json` |
| `readiness` | Polls `http://API_HOST:PORT/v1/models` until it lists the profile's served model name; another name fails the step | `readiness.json` |
| `functional` | Counting, arithmetic and code with thinking off; an automatic and a forced tool call; a description of a generated red/blue image; arithmetic with thinking on, which must return reasoning text | `functional.json`, `functional.txt` |
| `stress` | `--stress-rounds` rounds (default 8) of 32 greedy requests through 16 threads: 24 short questions with known answers and 8 questions about a code hidden in about 6K tokens | `stress.json` (n, degenerate, wrong, errors, seconds), `stress-responses.json` |
| `throughput` | `--repeats` runs (default 1) of `llm_decode_bench.py`: temperature 1.0, exact token targeting, 1, 8 and 16 streams, no added context, 20 s cells after a 5 s warm-up, up to 2,048 output tokens, cold 8K, 64K and 128K prefill prompts | `throughput/tpN-matrix[-runK].json` and `.log`, `throughput.json`; prints the README values |
| `record` | Sanitizes the evidence and writes `<image-short>-<topic>-<YYYYMMDD>.md` and a directory of the same name under `--record-root` | `record.json`; prints the README values |

The port, served model name, image and node count come from
`profiles/<id>/profile.json`, its `config.json` and the release the profile
selects. The functional checks and the screen send the same requests as the
`functional.py` and `stress.py` programs published with the
[DeepSeek-V4.1-Flash record](../../records/images/dev-20260927-h2dstaging-deepseek-v41-tp4-20260927.md),
except for the thinking switch below.

**Thinking switch.** Requests that expect a direct answer carry the profile's
`config.json` `smoke` settings, which the installer's own smoke request also
uses: `chat_template_kwargs.thinking` for DeepSeek, `enable_thinking` for
MiMo, and low `reasoning_effort` for GLM, whose template always reasons.
Profiles without `smoke` get `{"chat_template_kwargs": {"enable_thinking":
false}}`. The thinking-on check sends the chat template's default.
`--thinking-off JSON` and `--thinking-on JSON` replace either setting.

**Supported checks.** The tool-call checks run when the profile sets
`--enable-auto-tool-choice`, the image check when `--limit-mm-per-prompt`
allows images, and the thinking-on check when it sets `--reasoning-parser`.
Other checks are reported as skipped. `--skip-check CHECK` (`count`,
`arithmetic`, `code`, `tool-call`, `forced-tool-call`, `image` or
`thinking-on`) skips one more.

A step passes when: the installer's result has state `complete` and exit
status 0; the API lists the served name; no functional check fails; the
screen has no degenerate or failed response (wrong answers are reported, not
failed); every benchmark cell is valid with no request errors. A failed
installation or readiness step stops the run. The exit status is 0 when every
step passed, 1 when one did not, and 2 for a usage or data error, including a
refused record.

## Requirements

- On the client: Python 3.10 or later (standard library only), an `ssh`
  client that reaches Node A with a key (`BatchMode=yes`; no password
  prompts), and an [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench)
  checkout holding `llm_decode_bench.py` 0.6.2 with its dependencies installed
  for `--bench-python` (default: the Python running the harness).
- On Node A, for `--install`: `curl` for the published script, `git` for a
  bundle, and `sudo` without a password prompt for the SSH user, because the
  installer runs without a terminal.

## Usage

Run from the repository root. Keep `--out` outside the repository: it holds
unsanitized outputs with addresses.

Install the published `main` script on a pair whose Node A is
`user@192.0.2.10`, then check and measure it:

```bash
python performance/harnesses/acceptance/accept_profile.py \
  --profile mimo-v26-flash-mopd-tp2 --node-a user@192.0.2.10 --api-host 192.0.2.10 \
  --install published --bench-dir /path/to/llm-inference-bench \
  --out /path/outside/git/accept-mimo-tp2
```

`--install` takes three forms:

| Form | Command run on Node A |
|---|---|
| `published` | `curl -fsSL https://raw.githubusercontent.com/FujitsuPolycom/sparkring/main/install.sh \| bash -s -- --profile ID --yes --json` |
| `published:REF` | The same script and source at branch, tag or commit `REF` (`--ref REF`) |
| `bundle:PATH:REF` | `git clone -q --branch REF PATH src && bash src/install.sh --repository PATH --ref REF --profile ID --yes --json`; `PATH` is absolute or starts with `~/` on Node A, and `REF` is a branch or tag in the bundle |

Copy a bundle to Node A first, for example `git bundle create sparkring.bundle
BRANCH` and `scp sparkring.bundle spark-r0:`, then pass
`--install bundle:~/sparkring.bundle:BRANCH`. `--skip-install` measures the
deployment already running and never opens SSH. Exactly one of the two is
required.

Other options: `--steps` selects the steps after installation (default
`readiness,functional,stress,throughput,record`); `--topic` and `--date` name
the record (defaults: the profile ID and today in UTC); `--status` sets the
record status (`implemented`, `qualified` or `research-only`; default: the
profile's); `--client` describes the client; `--private-name` adds names the
record must not contain; `--install-timeout`, `--ready-timeout`,
`--bench-timeout` and `--poll-interval` bound the waits. `--help` lists all.

## Resuming

Each step saves its result in `--out`. Running the same command again reuses
every complete saved result and runs only what is missing: a failed
benchmark run is repeated, while completed benchmark runs are kept. An
installation is never started twice from one `--out`: `install-launch.json`
is written before the launch, and a later invocation follows the same run on
Node A until it exits, which also recovers from a lost SSH connection or an
installer that outlives `--install-timeout`. A failed readiness check is
repeated on the next invocation; a completed but failed installation is
reused and stops the run again.

`--redo STEP[,STEP]` renames the step's saved outputs with a UTC time suffix
and runs the step again; `--redo install` starts a separate installation.

## What it changes

On Node A the harness creates `~/sparkring-acceptance/<profile>-<UTC time>/`
holding `command.sh`, `stdout.json`, `stderr.log` and `exit_code`, and the
installer makes the changes `sparkring install --yes` makes. Every other step
sends inference requests only. Nothing on Node A or in `--out` is removed;
an existing run directory, record file or record directory is never reused
or overwritten.

## Record sanitization

The record directory receives `install-phases.txt` (the `sparkring install`
progress lines), `functional.txt`, `stress.json` and the benchmark matrices.
Each matrix keeps only `metadata`, `prefill`, `results`, `summary_table`,
`burst_results`, `burst_summary_table` and `methodology`, with
`metadata.server` set to `http://NODE_A`. In text files, the API host and the
SSH host become `NODE_A` and other private IPv4 addresses become
`ADDRESS_1`, `ADDRESS_2`, and so on.

The harness then refuses to write anything if a file still contains a private
IPv4 address (`10.x.x.x`, `172.16.x.x` to `172.31.x.x`, `192.168.x.x`, or
`100.64.x.x` to `100.127.x.x`), the client's hostname, the SSH or API host, a
`--private-name`, or the SSH user. Matrices hold no model text, so the SSH
user is refused anywhere as a whole word; in text files, which hold model
replies, it is refused in account forms (`user@`, `/home/user`). Hostnames of
the other Sparks in the installer's progress lines are kept unless named with
`--private-name`.

## Tests

```bash
python -m pytest performance/harnesses/acceptance -q
```

One test runs the detached-installer scripts in a local Bash with `HOME`
redirected to a temporary directory; it is skipped without Bash, `setsid`
and `nohup`.
