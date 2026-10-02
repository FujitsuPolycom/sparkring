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

A shared installer image lock (`sparkring-installer-image/v2`),
`runtime/releases/<release>/installer-image.json`, pins one image and lists the
installer profiles admitted to run on it. Admitting a profile to that image
changes only the lock's sorted `profiles` list: the lock and its `sha256` entry
in the same directory's `release.json` are rewritten in place, and the release
keeps its identifier because the image and its receipts are unchanged. Any
other change to the lock, such as its image, a receipt digest or a size, needs
a distinct release. Each installer deployment's identity includes the lock, so
after an admission the next `sparkring install` of any profile on the lock
creates a separate deployment. [Contributing an installer profile](installer-profiles.md)
gives the admission steps. A new installer image release also names the
builder of its final layer in the `releases` field of
[builders.json](../../runtime/images/builders.json); the repository layout check
rejects an installer image lock without one. [Installer image builders](../../runtime/images/installer-images.md)
describes the chain. A GitHub release that publishes an installer image adds
its tag and the image's release name to
[installer-releases.json](../../runtime/releases/installer-releases.json), so
`sudo sparkring install --image TAG` selects that image.

Prepare the candidate and local PR description before requesting adoption.
Pushes, GitHub posts, merges, image publication and cluster operations require
the user's applicable authorization. Normal reviewed Git history makes rollback
possible; repository adoption does not require rewriting main or forcing a push.

A partial lifecycle fix may be accepted as a mitigation. For example, increasing
peer-response silence tolerance does not handle disappearance of the local
management address. Describe the solved condition and retain a follow-up for
remaining behavior before declaring an issue resolved. Do not auto-close issues
solely because a reporter lacks hardware evidence.
