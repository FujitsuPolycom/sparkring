# Exact serving source inputs

The [manifest](manifest.json) pins upstream commits, complete carried patches and
accepted source-tree hashes for the release. Independent Linux reconstruction
applied each patch to its pinned upstream snapshot and reproduced the working
runtime's source identity. The hash is SparkRing's canonical file/mode inventory
digest, not a Git commit or Git tree ID.
Reconstruction excludes `.agents` and `.claude` metadata directories; Git
metadata and Python bytecode/cache directories are not source inputs.

The patch files retain upstream licensing and attribution. They contain the
runtime reconciliation, including B12X sparse selection and Qwen multimodal HC
dispatch/startup auditing. They do not contain model weights or benchmark results
from the deployment trials.

Use the [image-upgrade tooling](../../../images/upgrades/README.md) for controlled
source acceptance and native-wheel reuse. Do not apply these files directly to a
running container or relabel an installed source contract after changing code.
