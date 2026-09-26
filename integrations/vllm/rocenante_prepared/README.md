# Prepared adaptive RoCEnante transport

Status: **implemented**. The installer image serves every installer profile
with this transport. The bounded two/four-rank probe qualification below
applies to the unpaced manifest `7d8beed57e54`.
The profile identity `tp2-rocenante-adaptive-prepared` bridges SparkRing's
adaptive peer-path transport to B12X's prepared execution API. It is separate
from the immutable `tp2-rocenante-adaptive` source bundle.

The installer image `dev-20260925-qwendecode-cuda1342-nccl2323-status031` pins
transport manifest `2eef276d5403`. Its twelve files are the files that
`package.py` stages from this directory, with the same SHA-256 values. The
manifest that `package.py` writes here, `8e6bc565ae7f`, differs only in the
B12X file paths it records for its base image and in its qualification fields.

The native proxy paces each stripe on a hardware-forwarded (two-hop) path: it
posts signaled 32 KiB chunks and keeps at most 128 KiB of unacknowledged bytes
per queue pair. `B12X_ROCE_FORWARD_WINDOW_BYTES` and
`B12X_ROCE_FORWARD_CHUNK_BYTES` override the window and chunk; a zero window
restores single-write posting. The window assumes the ConnectX hairpin queue at
8,192 entries (`hairpin_queue_size 8192`, which `sparkring install` applies on
four-Spark rings); with the driver default of 1,024 it must be 32 KiB or less.

The [manifest](manifest.json) records source origins and every installed file.
The native proxy, peer-path policy and collective kernel math retain the
SparkRing source from `60d8d68486540ce9ddb2702dd545fda6b347c087`, derived from
Luke and Local Inference Lab's RoCEnante work. The prepared API derives from
B12X `a83336581a3a907076e60797df69ab66df5a2ff1`. Their Apache-2.0
[license](LICENSE) and source notices are retained.

## Execution contract

- The caller establishes the communicator and declares `query_from_runtime`
  and `plan`. Preparation compiles retained reduce/gather callables and allocates
  shared staging buffers. `all_reduce` and `all_gather` require the matching plan;
  runtime execution performs no kernel lookup.
- Two paths per peer remain the default. Peer HCA indices `0/2` select the two
  PCI domains of physical cage p0 from the four-function inventory. Inherited
  four-path configurations remain research-only, not enabled by this profile.
- Message-size-dependent grids and separate arrival counters for each grid are
  retained. The prepared query binds peer mapping, path count and capacities;
  the kernel's `opposite_paths` argument is not the HCA inventory count.
- Shared staging writes wait for preceding stream work. Completion events follow
  output copies. Padded gathers return independent output storage even when a
  pack-sized, unaligned input would otherwise expose shared scratch.
  Capture-stream admission, health checks and fatal timeout
  behavior retain the transport's source contract.

## Package and select

```bash
python3 integrations/vllm/rocenante_prepared/package.py \
  --destination /tmp/tp2-rocenante-adaptive-prepared
```

The destination must not exist. The command verifies source hashes and copies
only the manifest-bound runtime files. Install that directory under
`/opt/sparkring/transports/tp2-rocenante-adaptive-prepared` and explicitly admit
the profile in the image's transport selector before selecting its exact
manifest hash. The [image integration procedure](INSTALLATION.md) binds the
replacement selector's preimage and B12X API dependencies. This source addition
does not install a hook or select a running model. Do not substitute this
manifest hash for the legacy bundle's identity.

## Checks

```bash
python -m pytest integrations/vllm/rocenante_prepared/test_prepared_transport.py \
  integrations/vllm/rocenante_prepared/test_probe.py -q
```

Twenty-eight CPU checks pass. They cover retained wire/kernel source, prepared API
and program identities, peer-path compile arguments, staging/output ownership,
selector compatibility, packaging and counter interpretation.
They do not establish network connectivity, GPU numerical results, graph replay,
performance or compatibility of the separate TP4 weighted-mesh bundle.

Before serving, run the [two/four-rank probe](INSTALLATION.md#bounded-real-hardware-probe)
with the selected HCAs:
BF16/FP16/FP32 reduction, dim-0/last-dim gather, small/large messages, interleaved
grid sizes and CUDA graph replay with frozen kernel resolution. Verify native
proxy health and the selected manifest on every rank. Qwen's model smoke follows
those transport checks; no image/profile promotion is implied by CPU success.

The [hardware record](../../../performance/records/transport/rocenante-prepared-a75bd02f-20260918.md)
qualifies the unpaced manifest `7d8beed57e54` on image `a75bd02ffc1d`: the two-
and four-rank runs each passed 15 cases per rank, including four frozen-kernel
graph replays with stable addresses and no replay allocation.
For the paced proxy, four GB10 ranks with `hairpin_queue_size 8192` ran 200
back-to-back operations in three trials in the serving image. A 2 MiB
all-reduce took 656 µs with 42,867 hairpin drops and 10 retransmission timeouts
without the window, and 348–356 µs with none with it. A 4 MiB-per-rank
all-gather took 1,491 µs with 125,468 drops and 219 timeouts without the window,
and 769–788 µs with none with it. Results matched NCCL in every case.
The preparation session repeats multi-rank selection races instead of trusting
unequal rank-local selection caches. The archived `e8577c447a69` record retains
its own evidence; neither record establishes model performance or compatibility
of other selections.
