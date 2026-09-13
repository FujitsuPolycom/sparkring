# LIL R37 GLM Spark source composition

This composition installs pinned LIL R37 vLLM and B12X sources with SparkRing
mHC, checkpoint coalescing and SparkCache integration over the published R35
ARM64 foundation. It retains standalone NCCL, SIRCL and SparkCache native
libraries. Status: **Experimental**. Source reproduction does not establish
model correctness or long-duration collective stability.

The exact tested image is published as
`ghcr.io/fujitsupolycom/sparkring:r37-arm64-beeb32253aa7`.
The [publication record](publication.json) pins its immutable registry digest
and config ID. Use the [TP4 quickstart](../../../../profiles/glm53-flash-spark-tp4-dcp1-sparkcache/README.md)
or [experimental TP2 instructions](../../../../profiles/glm53-flash-spark-tp2-dcp1-sparkcache/README.md)
to deploy it; rebuilding is optional.

[source-lock.json](source-lock.json) records exact source trees, patch/archive
hashes and native comparison paths. [descriptor.json](descriptor.json) is the
tested image installation descriptor, including the parent authored-file
inventory. [baseline-native.json](baseline-native.json) identifies inherited
standalone libraries. The [lease contract](vllm-connector-jobs-lil-r37.json)
binds the composed vLLM file bytes. Published R35 inputs remain unchanged.
The [frozen image entrypoint](candidate-image.py) preserves the published
executable bytes recorded in [runtime-artifacts.json](runtime-artifacts.json);
the shared implementation remains in `runtime/images/candidate_image.py`.

## Reconstruct and package sources

Run from the SparkRing repository root with Git supporting `archive --mtime`
and Python 3.12. Use a new disposable build directory outside the repository.
No model files or running services are needed. Source archives are generated
locally; the complete parent installed receipt is extracted from its immutable
image instead of duplicated in Git.

```bash
REPO=$PWD
BUILD=$(mktemp -d "$HOME/sparkring-r37-build.XXXXXX")
export REPO BUILD
python3 - <<'PYTHON'
import hashlib, json, os, pathlib, shutil, subprocess, sys
repo = pathlib.Path(os.environ['REPO'])
build = pathlib.Path(os.environ['BUILD'])
recipe = repo / 'runtime/images/compositions/lil-r37-glm-spark'
r35 = repo / 'runtime/images/sparkring-r35'
lock = json.loads((recipe / 'source-lock.json').read_text())
foundation = json.loads((r35 / 'source-lock.json').read_text())
sys.path.insert(0, str(repo))
from runtime.images.candidate_sources import package
context = build / 'context'
context.mkdir()
for name, spec in lock['components'].items():
    source = build / name
    subprocess.run(['git', 'clone', '--no-checkout', lock['repositories'][name], str(source)], check=True)
    def git(*args):
        return subprocess.check_output(['git', '-C', str(source), *args])
    old = foundation['components'][name]
    git('fetch', 'origin', old['base_commit'], spec['base_commit'])
    git('checkout', '--detach', old['base_commit'])
    old_patch = r35 / 'patches' / old['patch']
    if hashlib.sha256(old_patch.read_bytes()).hexdigest() != old['patch_sha256']:
        raise ValueError('R35 patch hash mismatch')
    git('apply', '--index', str(old_patch))
    if git('write-tree').decode().strip() != spec['native_comparison']['reference']:
        raise ValueError('Integrated foundation tree mismatch')
    # Retain the integrated tree in the disposable object store without
    # publishing a commit or altering any release reference.
    git('-c', 'user.name=Source builder', '-c', 'user.email=builder@example.invalid',
        'stash', 'push', '-m', 'Integrated foundation comparison')
    git('checkout', '--detach', spec['base_commit'])
    patch = recipe / spec['patch']
    if hashlib.sha256(patch.read_bytes()).hexdigest() != spec['patch_sha256']:
        raise ValueError('R37 patch hash mismatch')
    git('apply', '--index', str(patch))
    result = package(name, source, context, spec)
    if result['archive_sha256'] != spec['archive_sha256']:
        raise ValueError('Source archive differs from tested artifact')
for name in ('descriptor.json', 'vllm-connector-jobs-lil-r37.json'):
    shutil.copyfile(recipe / name, context / name)
artifacts = json.loads((recipe / 'runtime-artifacts.json').read_text())
entrypoint = recipe / 'candidate-image.py'
if hashlib.sha256(entrypoint.read_bytes()).hexdigest() != artifacts['files']['/opt/sparkring/bin/candidate-image.py']:
    raise ValueError('Published entrypoint identity mismatch')
shutil.copyfile(entrypoint, context / 'candidate_image.py')
shutil.copyfile(repo / 'runtime/images/Dockerfile.candidate', context / 'Dockerfile.candidate')
PYTHON
```

The comparison uses reconstructed **integrated R35 trees**, not bare upstream
R35 commits. B12X kernels authored in Python still require fresh GPU validation,
even when its packaging metadata passes the native comparison.

## Extract the parent receipt and build

On an ARM64 Docker builder, retain the same `REPO` and `BUILD` values:

```bash
PARENT=ghcr.io/fujitsupolycom/sparkring@sha256:3eb8138453e5cc5ce1f436caf232e03b84e23e094a49e376428d1ebfe26c4742
docker pull --platform linux/arm64 "$PARENT"
PARENT_ID=$(docker image inspect --format '{{.Id}}' "$PARENT")
test "$PARENT_ID" = sha256:7b698d4299aaaebb359e287d75c7f18275311b6a6d56322d9767b0b4f35cd60b
docker run --rm --network none --pull never --entrypoint cat "$PARENT_ID" \
  /opt/sparkring/receipts/r35-installed.json > "$BUILD/context/parent-installed.json"
python3 - <<'PYTHON'
import hashlib, json, os, pathlib
recipe = pathlib.Path(os.environ['REPO']) / 'runtime/images/compositions/lil-r37-glm-spark'
lock = json.loads((recipe / 'source-lock.json').read_text())
receipt = pathlib.Path(os.environ['BUILD']) / 'context/parent-installed.json'
if hashlib.sha256(receipt.read_bytes()).hexdigest() != lock['parent_installed_sha256']:
    raise ValueError('Parent installed receipt identity mismatch')
PYTHON
docker build --network none --pull=false --build-arg PARENT="$PARENT" \
  -f "$BUILD/context/Dockerfile.candidate" -t local/sparkring:lil-r37-glm-spark "$BUILD/context"
docker run --rm --network none local/sparkring:lil-r37-glm-spark verify
```

A rebuild may have a different OCI image ID because image construction records
metadata separately from source content. Record its own image ID and verification
receipt; do not represent it as the already tested image. The installer verifies
parent bytes, installs exact authored source files, removes obsolete authored
files and adds the source-bound contract. It does not rebuild inherited native
extensions or download dependencies. No registry publication occurs in these
commands.

The registered GLM deployment adapter accepts the published image identity.
A rebuild with a different image ID needs a separately reviewed composition
registration and runtime-artifact record before deployment; its evidence is not
inherited from the published image. Retain model/topology-specific admission and cache namespaces; this build
recipe does not turn GLM-specific SparkCache evidence into Qwen or DeepSeek
qualification. Compare exact-answer, cache-restart, prefill/decode and prolonged
mixed workload results before adopting a rebuilt image.
