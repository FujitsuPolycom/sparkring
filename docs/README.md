# SparkRing documentation

Find the page for what you want to do.

## Install

- [Quick start](../README.md#quick-start): the one-line install command and
  the profile table.
- [Install guide](operations/install.md): requirements, cabling, options and
  uninstalling.
- [Install Builder](https://fujitsupolycom.github.io/sparkring/): pick a model,
  checkpoint and layout and copy the commands
  ([how it works](operations/compose-builder.md)).
- [Manual setup](operations/setup.md) and
  [Docker Compose files](operations/compose-files.md), without the installer.

## Operate

- [All commands](operations/commands.md): status, logs, switching models,
  stopping and starting.
- [Status dashboard](operations/dashboard.md): what a running model reports.
- [Install reference](operations/install-reference.md): serving settings,
  checkpoints, [two models on one ring](operations/install-reference.md#two-models-on-one-ring),
  storage and security.
- [Enhancements by model](../performance/enhancements.md): every speed
  enhancement, how to enable it and its measured gain; check a deployment
  with `python scripts/check_enhancements.py PROFILE`.

## Troubleshoot

- [Install troubleshooting](operations/install.md#troubleshooting): common
  messages and their fixes.
- [When a model stops serving](operations/install-reference.md#when-a-model-stops-serving):
  automatic recovery and the admin tunnel.
- [Ring networking](operations/fabric-repair.md): inspect and repair fabric
  routing.

## Contribute

- [Contributing](../CONTRIBUTING.md): issues, pull requests and hardware work.
- [Local checks and CI](development/testing.md) and the
  [writing standard](development/writing.md).
- [Repository ownership](development/layout.md): where code and documents
  belong.
- [Contributing an installer profile](development/installer-profiles.md) and
  the [release procedure](development/releases.md).

## Understand the architecture

- [Architecture](architecture/overview.md): topology, serving container and
  collective path.
- [SIRCL](architecture/sircl.md): SparkRing's collective transport for 2 to 8
  Sparks, with ring sessions and the four-rank native sessions of retained
  images.
- [What is a SparkRing image?](operations/images.md)
- [Profile catalog](../profiles/README.md): every saved deployment
  configuration and its guide.

## Reproduce history

- [Benchmarks](../performance/benchmarks.md) and the
  [evidence records](../performance/README.md): measured configurations and
  results.
- [Release selections](../runtime/releases/README.md): pinned releases and
  preserved inputs.
- [Retired deployments](history/deployment-variants.md) and
  [retained runtime compositions](history/runtime-compositions.md).
- [Repository layout adoption](development/repository-layout-adoption.md): the
  record of the move to the present layout.
