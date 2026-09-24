# Moby seccomp profile

Unmodified default profile and Apache-2.0 license from Moby's Docker 29.2.1
source, commit `6bc6209b88a7a834c91f77d848e025c79e0227a1`:
[upstream source](https://github.com/moby/moby/tree/6bc6209b88a7a834c91f77d848e025c79e0227a1/vendor/github.com/moby/profiles/seccomp).

SparkRing's [loader policy](../../runtime/common/loader-seccomp.json) is derived
from this file. Its only semantic change is allowing `io_uring_setup`,
`io_uring_enter` and `io_uring_register` for the B12X native checkpoint loader.
The policy applies only to explicitly selected external-image containers;
it does not change the Docker daemon or the host kernel policy. A regression
test compares every other rule against the upstream copy.
