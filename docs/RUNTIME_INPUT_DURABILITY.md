# Runtime input durability

A runtime build consumes inputs this repository does not contain: base
images, upstream sources, wheels, and model weights. `runtime/runtime-lock.json`
names each one. This document states what that naming guarantees, what it
does not, and what a maintainer does about the difference.

## What the lock guarantees

Every external input is named by an identity that cannot be moved by the
party serving it:

| Input | Identity |
|---|---|
| Builder and runtime base images | `nvcr.io/nvidia/cuda` by SHA-256 content digest |
| vLLM, SparkInfer, FlashInfer, NCCL, DeepGEMM | Repository URL plus a full 40-character commit |
| FlashInfer wheels | Version plus per-wheel SHA-256 |
| Model weights | Hugging Face repository plus an immutable revision hash, with the pinned revision's `config.json` SHA-256 |
| Overlay files, NCCL patches, public runtime inputs | Repository-relative path plus SHA-256 |

`scripts/check_runtime_inputs.py` asserts these properties. Its default
mode is OFFLINE: it reads the lock and the checkout, rejects any input
named by a branch, a tag, an abbreviated commit, or a movable reference,
and rehashes every in-tree artifact the lock records. It runs in the
offline suite, so an input that loses its immutable identity fails
before a build consumes it.

## What the lock does not guarantee

An identity is not a copy. A content digest states which bytes are
correct; it does not oblige anyone to keep serving them. Every external
input remains retrievable only while its host chooses to serve it, and a
repository can be made private, renamed, or deleted without notice.

`scripts/check_runtime_inputs.py --check-remote` observes that
separately. It contacts the public sources the lock names and reports
which pinned identities still resolve. It reaches the network, mutates
nothing, and contacts no configured Spark. Run it on a schedule rather
than only when a build fails: the useful time to discover that an
upstream has disappeared is before the bytes are needed.

`scripts/pull_pinned_images.py` retrieves the images the locks pin, by
digest, from wherever their publisher serves them, and confirms the local
store holds that digest. It republishes nothing, so it carries no
redistribution obligation for an image this project does not own. Its
`--plan` mode prints what it would pull and contacts nothing; pulling
writes to the local container store and is therefore MUTATES HOST.

Pointing at an image and republishing one are different acts. The first
needs no permission from its publisher; the second does, and the table
below bounds it.

## Licensing bounds on mirroring

Redistribution rights differ by input and decide what a mirror may
contain.

| Input | License | Redistribution |
|---|---|---|
| vLLM | Apache-2.0 | Permitted with notices |
| SparkInfer, B12X | Apache-2.0 | Permitted with notices |
| FlashInfer, DeepGEMM | Per their published licenses | Check before mirroring |
| NCCL | Apache-2.0 per its `LICENSE.txt` | Permitted, preserving NVIDIA's copyright and license text |
| Base images | NVIDIA container licensing | Governs any republished derivative; confirm before publishing an image |
| Model weights | Per the model repository's own terms | Frequently gated; a mirror may be prohibited |

`THIRD_PARTY_NOTICES.md` carries the authoritative statement for each
component this repository ships. A mirror of an Apache-2.0 source must
carry that project's `LICENSE` and `NOTICE` files alongside the bytes.

## Mirroring procedure

A mirror converts an identity into a preserved artifact. For each source
the lock pins, and only where the licence above permits it:

1. Fetch the exact commit rather than a branch:
   `git fetch --depth 1 <repository> <commit>`.
   Every source the lock names is retrievable this way, including
   commits that are not themselves refs.
2. Produce a stable archive of that commit and record its SHA-256
   beside the existing `commit` field. A commit hash identifies a tree;
   an archive hash identifies the bytes a build actually consumed.
3. Store the archive where its availability does not depend on the
   upstream host, and carry the project's licence and notice files with
   it.
4. Record the mirror location in the lock so a build can fall back to it
   without a maintainer editing the builder.

Steps 2 through 4 change `runtime/runtime-lock.json`, so
`scripts/check_runtime_inputs.py` must pass afterwards.

## Publishing a runtime image

Publishing a built image by immutable digest removes the largest barrier
for anyone reproducing a deployment: it replaces a multi-source build
with a single pull. It is bounded by the base-image licensing above, and
by whether every layer it carries may be redistributed.

An image published this way is pinned the same way its inputs are — by
digest, recorded in the lock — and inherits the same distinction. The
digest states which image is correct. Keeping it served is a separate,
ongoing act.
