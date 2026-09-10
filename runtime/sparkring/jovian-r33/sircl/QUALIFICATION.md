# R33 SIRCL ARM64 build and qualification

Status: implemented build procedure; native build and hardware qualification
are pending.

## Decision

The CUDA 13.0 binary is rollback-only for the R33 image. The public receipt
identifies an AArch64 SM121 artifact built with CUDA 13.0.88 from
`spark_transport` tree `2aac02232a9115037723aa1dd40483a5693a3e1e`, but it
does not record ELF dependencies, embedded CUDA images, CUDA-driver execution
under 13.3, or behavior in the R33 process. The R33 foundation requires CUDA
13.3, PyTorch `cf30153c4c131c8164ee7798e5022d810682e2cb`, and NCCL 2.31.2.
Compatibility is therefore unproved even though the compiled SIRCL source
files have not changed since the old receipt.

The source lock uses public commit
`f26895a158f586918b34136367fc63c060af6a60` and `spark_transport` tree
`3ea295fb7adc7e8c1823412f9b1cc549679eea77`. This is the public-main state
inspected on 2026-09-10 for the R33 integration. It also binds the current
native-MTP3 mesh profile and both recorded transport bundle manifests. The
mesh profile is research-only and specifies CUDA 13.3; its compute pins do not
replace R33's vLLM, B12X, PyTorch, or NCCL pins.

## Audit findings

The native CMake project accepts its CUDA toolkit and architecture from the
caller. It requires CUDA and ibverbs, builds `libspark_transport_capi.so`, and
does not itself enforce CUDA 13.3, AArch64, or SM121. The build script enforces
those inputs before configuring CMake, archives the locked public commit, and
refuses an existing output directory.

The old receipt's `alternating_stream_cuda_smoke` result is too narrow to
qualify SIRCL stream switching. The executable links only `CUDA::cudart`. It
alternates two marker kernels and synchronizes a host event before every switch.
It does not load `libspark_transport_capi.so`, construct the four fused RDMA
endpoints, submit `spark_tp4_fused_prefill_all_reduce_rows`, or overlap caller
streams. Keep the smoke as a CUDA sanity check, but do not use it as evidence
for fused-session behavior.

The four-rank `spark_tp4_fused_prefill_probe` is the focused endpoint gate. It
checks four distinct dual-rail endpoints, MTU 4096, peer direction and arena
metadata, cooperative launch support, exact BF16 results, noninteger inputs,
input/output guards, per-endpoint wire bytes, completion counts, and three
latency boundaries. It exercises the fused proxy directly rather than the C
API, so an additional C-API stream-switch probe is required.

SIRCL remains a TP4 transport. The R33 TP2 single-DAC profile must use its
source-bound NCCL path; merely packaging this library must not activate SIRCL
for TP2.

## Build

Run inside the assembled R33 ARM64 foundation, with CUDA 13.3 available at
`/usr/local/cuda-13.3` or set `CUDA_HOME` explicitly:

```bash
bash runtime/sparkring/jovian-r33/sircl/build-sircl-cu133-sm121.sh \
  /src/sparkring /artifacts/sircl-cu133-sm121
```

The build passes only when the public commit, transport tree, mesh-profile
blobs, AArch64 host, CUDA 13.3 compiler, SM121 image, and complete CTest run
match the lock. Preserve the produced source archive, library, probes, logs,
ELF/CUDA inspection, checksums, and `build-receipt.json` together.

## Qualification gates

1. Run the build in the exact R33 foundation. Confirm the receipt records zero
   CTest failures, `readelf` reports AArch64, `cuobjdump` reports `sm_121`, and
   `ldd` resolves every dependency from the intended image. Run a process-load
   smoke using the R33 Python process and `ctypes.CDLL` before model startup.
2. On the four-rank direct-cable mesh, run
   `spark_tp4_fused_prefill_probe` with the exact current primary/secondary
   peers, devices, GID indices, control-port namespace, proxy CPU placement,
   and MTU 4096. Require zero result mismatches, zero guard corruptions, exact
   per-endpoint byte/completion accounting, healthy status on every rank, and
   clean teardown. Test query rows 1, the profile's captured rows
   16/20/24/28/32, a representative prefill row count, and 8192.
3. Add and run a four-rank C-API probe that creates
   `spark_tp4_fused_prefill_handle`, owns two nonblocking CUDA streams, and
   submits at least 1,000 operations through
   `spark_tp4_fused_prefill_all_reduce_rows`. Alternate streams and operation
   slots, use separate guarded input/output buffers per in-flight operation,
   place verification kernels after each submission on its caller stream, and
   avoid host synchronization between submissions. Require exact BF16 output,
   untouched guards, monotonic proxy sequences, no poisoned health status,
   bounded teardown, and no deadlock under slot reuse. Repeat with a host
   synchronization between operations as a control.
4. Run the ordinary eager and graph TP4 probes from the same build so the fused
   path does not regress graph capture, deferred credits, dual-port striping,
   vocabulary all-gather, or health reporting. Record all rank logs and the
   exact topology inputs. A single-rank CUDA pass is not evidence for this gate.
5. Load the library in the R33 TP4 serving profile and verify backend activation
   from the rank-wide capability record. Run bounded exact-output prefill and
   decode cases, then the long-prefill regression. Require no
   `sample_tokens` timeout, healthy SIRCL status after output publication, and
   successful fallback for calls outside SIRCL admission rules.
6. Start the TP2 single-DAC profile from the same generic image. Verify NCCL is
   active and SIRCL is not constructed. This proves packaging isolation; it
   does not qualify SIRCL at TP2.

Do not publish or mark the artifact qualified until gates 1 through 6 have
receipts tied to the same image digest, source lock, transport SHA-256, and
resolved profile inputs.
