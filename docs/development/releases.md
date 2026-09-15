# Release and recommendation changes

Merging implementation and recommending a deployment are separate decisions.
Useful code, documentation and research profiles can be reviewed without owning
Spark hardware. Maintainers own release qualification.

For a publication or default promotion, identify the exact source/dependency
pins, model revision and quantization, image digest, TP/DCP, topology and workload.
Verify source/build inputs and installed paths, then perform the relevant
hardware correctness, failure/recovery and serving checks on authorized hosts.
State cold/warm cache behavior, prompt lengths, speculation, concurrent load,
units, repetitions and limitations for performance claims. A component test is
not full-stack evidence.

Preserve public image names and immutable digests. Select a distinct release
when build inputs change; do not rewrite a publication receipt, retag evidence,
or update preserved hashes merely to pass CI. Record a rollback image and its
compatible site configuration before changing an operational default.

Prepare the candidate and local PR description before requesting adoption.
Pushes, GitHub posts, merges, image publication and cluster operations require
the user's applicable authorization. Normal reviewed Git history makes rollback
possible; repository adoption does not require rewriting main or forcing a push.

A partial lifecycle fix may be accepted as a mitigation. For example, increasing
peer-response silence tolerance does not handle disappearance of the local
management address. Describe the solved condition and retain a follow-up for
remaining behavior before declaring an issue resolved. Do not auto-close issues
solely because a reporter lacks hardware evidence.
