# Compiler CPU placement

Status: implemented; CPU regression checks passed. Image and serving
qualification must identify a build containing this implementation.

SIRCL graph sessions pin the serving submission thread and transport-progress
thread to separate CPUs. Compiler processes spawned afterward inherit the
submission thread's affinity unless they explicitly select another mask.
Increasing the compiler-process count does not widen that inherited mask.

The B12X compiler initializer accepts `B12X_COMPILE_CPU_AFFINITY`, a
comma-separated list of CPU IDs and inclusive ranges. For example, `12-19`
allows compiler children to use eight CPUs while a deployment retains SIRCL
submission/progress CPUs 10/11. Those numbers are site-specific; every requested
CPU must be available in the container's effective cpuset.

The setting changes only the compiler child's current thread. Its descendants
inherit that mask; the serving parent retains its affinity. An unset setting
preserves inherited behavior. Empty/malformed values are invalid. If the kernel
rejects or intersects the requested mask, initialization restores the inherited
mask and raises an error. Compiler-process count and Torch's one-thread compiler
setting are unchanged.

[GPU-free regression checks](../../images/upgrades/checks/compiler_worker_affinity.py)
cover parsing, unchanged unset behavior, rejected masks, restoration and
initializer-only placement. They do not establish compilation throughput or
serving performance. No existing profile default is changed by this note.
