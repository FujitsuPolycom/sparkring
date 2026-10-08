"""vLLM adapter of SIRCL ring sessions: everything that depends on a vLLM version.

- :mod:`.platform`, :mod:`.cuda_platform`: the ``vllm.platform_plugins`` entry
  point and SIRCL's ``CudaPlatform`` subclass;
- :mod:`.communicator`: the device communicator that puts SIRCL in front of
  every collective of every vLLM group;
- :mod:`.adapter`: per-group placement, NCCL policy, session and dispatch,
  independent of vLLM;
- :mod:`.fabric`: layouts, group fabrics, NCCL policy, route maps and relay
  load;
- :mod:`.planner`, :mod:`.executor`: rank-invariant plans and their execution
  on a ring session;
- :mod:`.guard`: PyNccl suppression, the NCCL dispatch check and the
  ``torch.distributed`` tripwire;
- :mod:`.tp_slot`, :mod:`.dcp_collectives`: the tensor-parallel group's session
  behind vLLM's RoCE all-reduce slot, and the collectives of a
  decode-context-parallel group;
- :mod:`.tp4`: SIRCL's four-rank native sessions and when they may serve;
- :mod:`.mhc`, :mod:`.qwen_hc`: prefill row ownership carried by SIRCL on
  groups without PyNccl, GLM-5.3-Flash's mHC rows (the ``mhc_prefill_shard``
  shim) and Qwen3.8's hyper-connection rows (the ``qwen_hc_prefill_shard``
  shim);
- :mod:`.norm_fusion`: the fused all-reduce + residual add + RMSNorm at vLLM's
  post-all-reduce norm sites (``SIRCL_FUSED_NORM``, the
  ``fused_allreduce_rms_norm`` shim);
- :mod:`.plugin`, :mod:`.shims`: the ``vllm.general_plugins`` entry point and
  the version-pinned shims;
- :mod:`.pins`, :mod:`.hooks`: the pinned vLLM builds and the hook table;
- :mod:`.sessionapi`, :mod:`.groupops`, :mod:`.emulation`: the session
  interface the adapter calls, CPU-group votes, and a CPU reference session
  with emulated ranks for tests;
- :mod:`.p2p`, :mod:`.p2p_emulation`: which groups get SIRCL point-to-point
  channels and their forward windows (the instance plan), and a CPU reference
  of the channels with emulated ranks for tests;
- :mod:`.receipt`, :mod:`.settings`: per-rank receipts and the adapter's
  environment variables;
- :mod:`.serve`: the launcher that serves a SparkRing profile with the
  adapter on Sparks of a ring (``RUNBOOK.md``).

Importing this package does not import vLLM; :mod:`.communicator` and
:mod:`.cuda_platform` do.
"""
