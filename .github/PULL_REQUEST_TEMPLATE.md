<!--
  Thanks for contributing to SparkRing.

  Every box below is a real gate, not paperwork. If a section does not apply,
  write "n/a" and why — do not delete it, and do not tick a box you have not
  actually satisfied. A PR that claims a test passed when it did not is worse
  than a PR with a failing test.

  New here? Read CONTRIBUTING.md first — especially "The two-lane claim
  policy", which governs what this change is allowed to claim.
-->

## What this changes

<!-- One paragraph. What and why, not how. Link the issue or proposal. -->

Closes #

---

## 1. Which lane does this target?

<!-- Exactly one. See CONTRIBUTING.md, "The two-lane claim policy". -->

- [ ] **Public-functional** — must work from this tree, on hardware anyone can
      obtain. No claim about the published performance numbers.
- [ ] **Reference-performance** — targets the pinned reference runtime and the
      numbers measured on it. Requires evidence from an acceptance-gated
      window (see section 4).
- [ ] **Docs only** — no behaviour change, no measurement.
- [ ] **Developer experience / CI / tooling** — no serving impact.

**Lane statement (one sentence, required):**
<!-- e.g. "Public-functional: adds a shape to the TP4 dispatch table; it does
     not touch the reference runtime and makes no performance claim." -->

---

## 2. Provenance

- [ ] **This change contains no code copied or adapted from another project.**

If that box is **not** ticked, complete the following — a PR with third-party
lineage and no provenance block will not be merged:

| Field | Answer |
|---|---|
| Source project | |
| Upstream URL / commit | |
| License | |
| Files or ideas taken | |
| `THIRD_PARTY_NOTICES.md` updated? | |

### If this adds or modifies a runtime patch under `runtime/patches/`

- [ ] The patch is pinned to the exact upstream commit recorded in
      `runtime/runtime-lock.json`.
- [ ] `runtime/patches/vllm/preimages.json` records the **SHA-256 of the exact
      upstream preimage file(s)** this patch applies to.
- [ ] `git apply --check` is clean with **zero fuzz and zero offsets** against
      that pinned upstream tree.
- [ ] The patch is described in `runtime/patches/vllm/README.md`.
- [ ] I understand the overlay is fail-closed: on a preimage mismatch it must
      refuse to start rather than guess.

Preimage hash(es) added or changed:

```text

```

---

## 3. Tests

**Exact commands run, and their exact output.** Paste the summary lines —
"tests pass" is not evidence.

### Offline (CPU-only, no GPU, no cluster)

```console
$ python -m pytest spark_transport sparkcache runtime -q

```

```console
$ ruff check --select E,F,W --ignore E501 .

```

- [ ] I added or extended a test that fails without this change.
      If not, explain why the change is untestable offline:
      <!-- e.g. "pure CUDA kernel; covered by the manual gate in section 4" -->

### Native (only if this touches C++/CUDA)

CPU-only portion — this one runs in CI:

```console
$ cmake -S sparkcache/native -B build/native -DSPARK_CACHE_PLACEMENT_ENABLE_CUDA=OFF
$ cmake --build build/native --parallel
$ ctest --test-dir build/native --output-on-failure

```

CUDA / hardware portion — **CI cannot run this.** If you touched
`spark_transport/` or any `.cu` file, paste the output of the full CTest suite
run on real hardware, or state plainly that you could not run it:

```console
$ ctest --test-dir <build> --output-on-failure

```

- [ ] I ran the native suite on hardware and pasted the result above.
- [ ] I could **not** run it (no hardware). I am asking a maintainer to gate it.

---

## 4. Evidence for any performance claim

If this PR, its title, its commit messages, or any doc it edits states a
number — throughput, latency, speedup, pool size, acceptance rate — fill this
in. If it states no number, tick the first box and skip the rest.

- [ ] This PR makes **no** performance claim.

| Field | Answer |
|---|---|
| Claim (with units) | |
| Configuration label (TP/DCP, concurrency, context length, KV bytes/rank) | |
| Window length and cell count | |
| Baseline compared against | |
| Same window as the baseline, or separate windows? | |
| Gates that passed (request errors, graph census, transport audit) | |
| Link to the evidence JSON | |

- [ ] The configuration label is complete enough that someone could tell what
      the number does **not** cover (for example: shared-prefix contexts are a
      concurrency baseline, not a unique-context capacity result).
- [ ] I am not presenting a public-lane measurement as a reference-lane
      measurement, and I am not claiming reproducibility that the acceptance
      gate has not actually produced.

---

## 5. No private identifiers

SparkRing is a **public** repository.

- [ ] **I attest that this change introduces no private identifiers**: no
      RFC 1918 or other site IP addresses, no SSH accounts, no hostnames, no
      absolute host filesystem paths, no private image or repository names,
      and no credentials of any kind (keys, tokens, passwords).
- [ ] Any address I did add uses an RFC 5737 documentation range
      (`192.0.2.0/24`, `198.51.100.0/24`, `203.0.113.0/24`), and any site
      value uses a `<PLACEHOLDER>` token.
- [ ] If I leaked a credential in an earlier push to this branch, I have said
      so below and rotated it. Force-pushing does not un-leak it.

> The `release-safety` CI job is a backstop for this attestation, not a
> replacement for it — it only knows the shapes it was taught.
>
> The separate `hardware-identifiers` job is **advisory and never blocks**.
> Names like the ConnectX-7 / RoCE interfaces are deterministic Linux
> predictable-interface names produced by the GB10 PCI topology: every DGX
> Spark generates the same ones, so they identify the hardware model this
> repository openly targets, not your cluster. Prefer a flag or environment
> variable in **new** code, but an advisory hit is not a merge blocker.

---

## 6. Housekeeping

- [ ] I matched the style of the surrounding code.
- [ ] My contribution is offered under the project license, Apache-2.0.
- [ ] I updated the docs that this change makes wrong.
