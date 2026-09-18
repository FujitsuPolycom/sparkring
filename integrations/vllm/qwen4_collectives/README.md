# Qwen collective policy for the prepared RoCEnante adapter

Status: implemented; GPU transport and serving qualification pending.

This artifact retains the Qwen collective policy's rank-agreed byte cutoff and
all-gather selection while targeting vLLM's prepared RoCEnante interface.
It keeps the compatibility module name `qwen38_collective_policy` so the
`qwen-collectives` feature bootstrap can load its explicitly recorded replacement.
The R37 artifact in `../qwen38_collectives` is unchanged.

The methods wrapped by the policy remain `_exchange_vote`, `should_custom_ar`
and `should_all_gather`. Their underlying eligibility/voting behavior is
preserved. The prepared adapter adds declaration and plan ownership; the policy
must not replace its `custom_all_reduce`, `all_gather` or `_prepared_plan` methods.
Runtime capability rejection still wins over the size cutoff. Transport errors
remain fail-stop rather than triggering an after-launch fallback.

Admission checks the complete source preimages and the used method signatures.
Trace mode additionally binds the CUDA graph module and forwards its variadic
call unchanged. Trace disabled does not import or patch the graph module.

CPU tests execute methods extracted from the exact reconciled source, including
prepared-plan forwarding, threshold boundaries, runtime rejection, rank voting,
trace selection and source drift. Set `SPARKRING_VLLM_SOURCE_ROOT` to that tree
when running `test_prepared_policy.py`. Without it, source-dependent tests skip;
this is not acceptance evidence.

The selected adaptive RoCE package must separately provide the prepared API.
Changing this policy's source binding does not make an incompatible frozen
transport bundle compatible.
