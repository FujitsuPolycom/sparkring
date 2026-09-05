# RoCEnante communication source

Status: **research-only**. This directory vendors only the `b12x.comm.roce`
package used by the four-rank hybrid transport. The remaining B12X package,
including model kernels, comes from the pinned operator image.

`provenance.json` binds the canonical source-tree digest and source revision.
The revision identifies the vendored source; consumers do not need an
unpublished Git ref because the complete selected source is included here.
The bundle builder verifies its digest before copying it. `LICENSE` contains
the upstream Apache-2.0 license; source copyright notices are retained.

## Attribution and design origins

RoCEnante originates with Local Inference Lab's contributors, not SparkRing.
Credit includes Luke (`lukealonso`), the Local Inference Lab community. The implementation derives from
[B12X PR 295](https://github.com/local-inference-lab/b12x/pull/295), with
the six-QP two-path peer mapping and hardware-forwarded opposite-peer path
used by the SparkRing mesh profile. It is not an unmodified installation of
the PR or a replacement for the B12X kernel package.

That communication work and its
[vLLM integration in PR 597](https://github.com/local-inference-lab/vllm/pull/597)
motivated SparkRing's exploration of hardware-forwarded opposite-peer paths
on a physical ring. SparkRing adapts the donor package's peer mapping and
combines it with SIRCL and its serving supervisor. The adapter's capability
agreement and post-step health checks draw on the vLLM integration ideas;
the complete vLLM PR is not vendored here.
