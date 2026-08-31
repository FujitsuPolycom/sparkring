# Distribute an image archive through the direct fabric

Status: **implemented** with GPU-free command and state-transition coverage.
Direct-fabric throughput is **research-only**. Live four-rank image import is
**unsupported** by retained repository evidence.

`scripts/fanout_image_archive.py` downloads one immutable image archive on a
configured seed rank. It then forwards the verified file through three direct
links so the management network carries one download instead of four.

The tool reads ranks, cycle edges, direct addresses, and SSH targets from a
validated SparkRing site file. It does not contain deployment hostnames or
assume that rank numbers map to particular addresses.

## Interface and evidence

| Mode | Remote effect | Output schema |
|---|---|---|
| default plan | none | `sparkring-image-archive-fabric-plan/v1` |
| `--verify` | reads final-file type, SHA-256, and byte count on every rank | `sparkring-image-archive-fabric-verification/v1` |
| `--execute --create-only` | creates, resumes, and verifies archive files | `sparkring-image-archive-fabric-receipt/v1` |
| `--execute` | creates and verifies archive files, then imports and inspects the named image on every rank | `sparkring-image-archive-fabric-receipt/v1` |

`--execute` requires `--confirmation FANOUT_IMAGE_ARCHIVE`. Every remote
operation has an overall `--timeout`, and every SSH or rsync connection uses
`--connect-timeout`. The defaults are 7,200 and 45 seconds respectively.
`--output` writes the same JSON document printed to standard output.

Inspect the complete command interface without contacting a host:

```bash
python scripts/fanout_image_archive.py --help
```

## Requirements

- A four-rank SparkRing site file whose direct edges form one cycle.
- Passwordless SSH from the operator to each rank's management target.
- Passwordless SSH between adjacent ranks on their direct-link addresses.
- `curl`, `sha256sum`, `rsync`, and Docker on every applicable host.
- One HTTP or HTTPS archive URL without embedded credentials or query tokens.
- An explicit absolute target directory dedicated to image archives.

The target directory must contain at least three components below `/` so a
broad path such as `/var/tmp` is rejected.
The archive name must be one filename without separators. Final and partial
files are constructed beneath that directory.

## Plan

Planning is the default and performs no remote operation:

```bash
python scripts/fanout_image_archive.py \
  --site /secure/site.yaml \
  --source-url https://images.example/runtime-arm64.tar.zst \
  --archive-name runtime-arm64.tar.zst \
  --expected-sha256 <64-lowercase-hex> \
  --target-directory /var/lib/sparkring/images \
  --seed-rank 0 \
  --create-only
```

The JSON plan names every management SSH command and every fabric hop. It also
records the expected archive SHA-256, target directory, selected path through
the cycle, MTU, link speed, optional image reference, and expected image ID. The
default first hop is the seed's lowest-numbered direct neighbour. Use
`--first-hop-rank` to select the other direction around the cycle.

## Verify existing files

Verification is read-only. It requires the final archive to exist with the
expected digest on every rank:

```bash
python scripts/fanout_image_archive.py \
  --site /secure/site.yaml \
  --archive-name runtime-arm64.tar.zst \
  --expected-sha256 <64-lowercase-hex> \
  --target-directory /var/lib/sparkring/images \
  --verify \
  --output ./evidence/image-archive-verification.json
```

## Create files without importing the image

Execution requires an explicit confirmation token. `--create-only` downloads,
forwards, and verifies the archive without invoking Docker image import:

```bash
python scripts/fanout_image_archive.py \
  --site /secure/site.yaml \
  --source-url https://images.example/runtime-arm64.tar.zst \
  --archive-name runtime-arm64.tar.zst \
  --expected-sha256 <64-lowercase-hex> \
  --target-directory /var/lib/sparkring/images \
  --seed-rank 0 \
  --create-only \
  --execute \
  --confirmation FANOUT_IMAGE_ARCHIVE \
  --output ./evidence/image-archive-fanout.json
```

## Import one image on every rank

Omit `--create-only`, provide the image reference stored in the archive, and
require its local image ID:

```bash
python scripts/fanout_image_archive.py \
  --site /secure/site.yaml \
  --source-url https://images.example/runtime-arm64.tar.zst \
  --archive-name runtime-arm64.tar.zst \
  --expected-sha256 <64-lowercase-hex> \
  --target-directory /var/lib/sparkring/images \
  --image registry.example/runtime@sha256:<manifest-digest> \
  --expected-image-id sha256:<config-digest> \
  --execute \
  --confirmation FANOUT_IMAGE_ARCHIVE \
  --output ./evidence/image-import.json
```

An archive created with `docker image save sha256:<image-id>` may contain no
repository tag. After loading such an archive, the tool verifies that the
expected image ID exists locally before applying the requested tag. An
existing tag that points at another image remains a conflict.

## File and interruption behavior

- An exact final file is reused without downloading or transferring it.
- A final path with another digest, or a non-regular file at that path,
  rejects the operation. The tool never overwrites it.
- Downloads and fabric transfers write `.<archive-name>.partial` beneath the
  target directory.
- `curl --continue-at -` resumes the seed download when the server supports
  byte ranges.
- `rsync --partial --append-verify` preserves and verifies an interrupted hop.
- A verified partial file is hard-linked to the final name. Hard-link creation
  is atomic and refuses an existing final path.
- Every destination computes SHA-256 after its hop. The evidence receipt names
  whether a rank reused an existing file or received it from another rank.
- A command error or timeout stops subsequent actions. Exact final files and
  bounded partial files already created on earlier ranks remain in place; the
  utility performs no distributed rollback. A retry reuses exact finals and
  resumes eligible partials.
- Image imports run in rank order after all four archives verify. If an import
  stops partway through, already imported images remain. A retry accepts only
  the requested tag mapped to the expected image ID.

The transfer command binds SSH to the source rank's address on the selected
cycle edge and connects to the peer address from the same edge. Management SSH
is used only to start and inspect operations.

## Limitations

The utility supports exactly four ranks whose configured direct edges form one
cycle. It chooses one three-hop path around that cycle and executes hops
sequentially; it does not broadcast, stripe one archive across links, or use
both directions concurrently.

The utility verifies archive and optional image identity. It does not validate
model serving, collective transport, SparkCache behavior, available disk
capacity, source-server throughput, or achieved fabric throughput. Image
archives are large, and download, local storage, hashing, Docker import, or one
slow hop can dominate elapsed time even when every direct link is 200 Gb/s.

Host-key enrollment, SSH authorization, remote tool installation, archive
creation, and cleanup policy remain operator responsibilities. The source URL
must use HTTP or HTTPS and cannot contain credentials, query parameters, or a
fragment. The tool does not publish an image or upload evidence.
