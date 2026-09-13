# Launch a local R35 image

Use the maintained TP2 and managed TP4 renderers with an explicit local R35
receipt. This selects the pinned R35 source composition without changing published
R33 profiles or images. Status: **Experimental**. A receipt proves the declared
image composition; it does not qualify a model, topology or workload.

## Record the image

On an ARM64 Docker host with the locally built image, set `IMAGE` to its complete
`sha256:` image ID and `RECORD` to an empty private directory. From the repository
root, collect the installed receipt and verification result:

```bash
docker run --rm --network none --pull never --entrypoint cat "$IMAGE" \
  /opt/sparkring/receipts/r35-installed.json > "$RECORD/installed.json"
docker run --rm --network none --pull never "$IMAGE" verify > "$RECORD/verification.json"
python runtime/common/r35.py --image-id "$IMAGE" \
  --installed-receipt "$RECORD/installed.json" \
  --verification "$RECORD/verification.json" --output "$RECORD/image.json"
```

The receipt binds source trees, parent source lock, installed-file inventory,
entrypoint, connector lease contract, native libraries and TP2 capability
evidence. Only exact local ARM64 image IDs are accepted. Before lifecycle
mutation, launch admission compares Docker's image identity and freshly executed
image verification with that receipt. It does not load a model for verification.

## TP2

Use the [pair prerequisites and private rank inputs](../../profiles/glm53-flash-spark-tp2-dcp1-sparkcache/README.md), model directory, cache directory and private
rank environment. The active memory guard and stopped-workload checks remain
required. Run the following on each host with its rank and local paths:

```bash
python runtime/common/tp2.py plan --rank "$RANK" --master "$MASTER" \
  --model-dir "$MODEL_DIR" --cache-dir "$CACHE_DIR" --env-file "$ENV_FILE" \
  --image "$IMAGE" --runtime-receipt "$RECORD/image.json" --sparkcache
```

After reviewing both plans, use the same arguments with `create`, then `start`
on rank 1 followed by rank 0. Omit `--sparkcache` for cache off.
`--r33-sparkcache` remains a compatibility spelling; the receipt selects the
runtime release. The R35 adapter preserves MTP3, B12X KDA prefill, mHC and
coalescing, and omits GLM's incompatible GDN decode selector. Rank zero gets
an API readiness check; the headless rank does not.

The catalog launcher intentionally keeps its published receipt and cache
selection. Use this explicit release interface for local R35 testing rather
than overriding a published catalog profile's identity.

## TP4

Use the managed mesh site's private topology and an extracted, verified SIRCL
bundle. Select `tp4-dcp1`, `tp4-dcp1-sparkcache`, `tp4-dcp4`, or
`tp4-dcp4-sparkcache` in `runtime_profile`. DCP1 remains the default deployment;
DCP4 is an alternative. Do not supply `r33_profile_contract_roots`: R35 verifies
its installed contract directly.

For a controlled R35 runtime comparison, the private site can include:

```json
"runtime_tuning": {
  "omp_threads": 1,
  "graph_submit_cpu": 10,
  "graph_progress_cpu": 11,
  "direct_doorbell": true
}
```

All four fields are required when this object is present. Thread count must be
an integer from 1 to 20, CPU indices integers from 0 to 19, and `direct_doorbell`
a JSON boolean. CPU availability and affinity still require host validation.
The settings apply to all ranks and remain in the installed private site;
canonical regeneration rejects edits made only to generated rank environments.
Omitting the object preserves the profile defaults. R33 receipts reject it.
Setting `direct_doorbell` to `false` selects the experimental command-ring
submission path for testing; it does not establish that path's stability.

```bash
python runtime/glm53-spark-mtp3-mesh/profile.py render \
  --site "$SITE" --bundle "$BUNDLE" --output "$LAUNCH" \
  --image-receipt "$RECORD/image.json"
```

The generated rank environment selects R35's entrypoint and source-bound cache
contract. The launcher supplies the fixed rank-zero API check. Follow the
[managed mesh plan/create/install procedure](../../profiles/glm53-spark-mtp3-managed-mesh-tp4/README.md)
with this same image receipt.
Its controller validates the image before accepting the container. Rank-zero
scheduler observation runs as a managed host unit, separately from API readiness.
Keep model mounts read-only and use distinct candidate container/cache identities.

The NIC steering marker is a separate host executable, not part of the R35
image receipt. Managed installation verifies its maintained C source, the
[reviewed external artifact record](../../runtime/glm53-spark-mtp3-mesh/host-marker-artifact.json)
and that record's source/binary provenance, then hashes the configured host
binary. Missing or changed evidence is rejected. This preserves the prepared
host fabric without claiming the inference image contains the marker.

### Isolated managed installation

Use `--deployment-name r35-managed-20260913` on the managed installer to keep
an existing default installation intact. The identifier accepts lowercase letters,
digits and internal hyphens, with a maximum of 63 characters. It derives:

- Code: `/opt/sparkring/deployments/r35-managed-20260913`
- Private configuration: `/etc/sparkring/deployments/r35-managed-20260913`
- Controller state: `/run/sparkring-r35-managed-20260913`
- Units: `sparkring-r35-managed-20260913-mesh.service`,
  `sparkring-r35-managed-20260913-model.service`, and rank zero's
  `sparkring-r35-managed-20260913-scheduler-liveness.service`.

Keep the private site's `state_root` consistent with that derived controller
state path. Supply the same deployment name to `managed_cluster.py` for `up`,
`start-model`, `stop-model`, `down`, `recover` and `status`. The unit renderer
also accepts the option. Named deployments do not accept arbitrary code/config
root overrides. Omitting the option preserves the default installation paths
and unit names.

The install plan records the name. Apply regenerates its paths and complete
unit text, verifies the source snapshot, and rechecks the pinned stopped
container before writing. Existing target directories or named unit files are
rejected; no deployment is overwritten or automatically removed.

Rendering or image verification does not establish native collective stability.
Retain semantic, cache-restart, concurrency and stability evidence for the exact
image and settings before adoption. These commands do not publish an image or
alter a published profile.

## Bounded GLM conversations

GLM checkpoint revision `df116c4fb16b1d37ae43d2cfd624de26ffbc832e` defaults
to Max reasoning effort when effort is omitted. Its template retains earlier
reasoning by default (`clear_thinking=false`) and ignores `enable_thinking`.
An experimental client configuration for bounded conversations is explicit
`reasoning_effort="low"` with `chat_template_kwargs={"clear_thinking": true}`.
This removes prior reasoning while retaining final answers, changing rendered
context length and cache boundaries. It is a client setting under qualification,
not an image/profile default or a resolution of native transport stalls.
