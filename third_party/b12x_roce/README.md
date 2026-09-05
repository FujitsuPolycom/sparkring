# RoCEnante communication source

Status: **research-only**. This directory vendors only the `b12x.comm.roce`
package used by the four-rank hybrid transport. The remaining B12X package,
including model kernels, comes from the pinned operator image.

`provenance.json` binds the canonical source-tree digest and source revision.
The revision identifies the vendored source; consumers do not need an
unpublished Git ref because the complete selected source is included here.
The bundle builder verifies its digest before copying it. `LICENSE` contains
the upstream Apache-2.0 license; source copyright notices are retained.

The implementation derives from Local Inference Lab's
[B12X PR 295](https://github.com/local-inference-lab/b12x/pull/295), with
the six-QP two-path peer mapping and hardware-forwarded opposite-peer path
used by the SparkRing mesh profile. It is not an unmodified installation of
the PR or a replacement for the B12X kernel package.
