# Contributing to SparkRing

Bug reports, questions, ideas, documentation fixes, reproduction reports, and
code are welcome. You do not need DGX Spark hardware or maintainer approval to
participate. Partial reports are useful; share what you know and maintainers
can help identify what is missing.

The supported deployments and their maturity are listed in
[`README.md`](README.md). This guide intentionally does not repeat that matrix.

## Issues and discussions

Open an issue when something is broken or unclear, or when you want to request
an improvement. Blank issues are enabled. The lightweight issue forms are
prompts, not requirements.

A useful bug report usually includes:

- the commit, release, or deployment profile, if known;
- what you expected and what happened instead;
- the shortest reproduction you have; and
- relevant logs with credentials, addresses, hostnames, accounts, and private
  paths removed.

It is fine to omit information you cannot obtain. Negative results and failed
reproduction attempts are welcome.

Use [GitHub Discussions](https://github.com/FujitsuPolycom/sparkring/discussions)
for open-ended questions if you prefer a conversation, but an issue is also
acceptable. Report suspected vulnerabilities privately as described in
[`SECURITY.md`](SECURITY.md).

## Pull requests

- Keep the change focused and explain the resulting behavior and why it is
  useful. Opening an issue first can help with a large redesign, but it is not
  required.
- Run the checks relevant to the files you changed. The
  [CI workflow](.github/workflows/ci.yml) is the source of truth. If you cannot
  run a check, say so in the pull request; unavailable cluster hardware does
  not prevent submission.
- Add or update tests when behavior changes, and update documentation that the
  change makes inaccurate.
- Do not commit credentials, private keys, site addresses, SSH targets,
  accounts, private host paths, model files, or private image names.
- Identify copied or adapted work and its license. Update
  [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) when required. Contributions
  are offered under the repository's Apache-2.0 license.

The pull request template asks for a summary and relevant validation. It does
not require every contributor to reproduce hardware-only validation.

## Measurements and hardware work

When a contribution reports a performance, capacity, or correctness result,
state the deployment configuration, hardware, workload, units, conditions,
and limitations. Distinguish offline results from live cluster evidence. An
ordinary code or documentation contribution does not need a hardware evidence
bundle.

Starting, stopping, rebuilding, or changing a remote host requires the host
owner's explicit permission. Read-only and offline contributions do not need
that permission.
