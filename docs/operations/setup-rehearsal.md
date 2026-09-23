# Rehearse the setup instructions

Status: **hardware rehearsal pending**. Offline checks establish document and
planner consistency, not working cables, RDMA or inference. Test the pair first;
schedule the four-node rehearsal separately after review and explicit approval.

## Record the starting state

Record privately: exact checkout commit, profile/variant, rank labels and SSH
targets, OS/driver versions, existing workloads, fabric addresses, model paths,
image identities and free space on each destination filesystem. Distinguish
prepared hosts from factory-reset hosts in every result.

For an unpublished branch, use the Git bundle procedure below. It preserves
revision identity and tracked-file discovery, which a plain source archive lacks.
Do not install `main` and assume it contains the branch's changes. Keep private
addresses, credentials and host output outside Git.

## Hand off an unpublished branch

On the controller in **Bash**, from the committed setup branch, create a private
bundle. These commands are local; they do not publish or contact a Spark:

```bash
git diff --quiet
git diff --cached --quiet
mkdir -p .sparkring
HANDOFF=$(mktemp -d "$PWD/.sparkring/setup-handoff.XXXXXX")
git symbolic-ref --short HEAD > "$HANDOFF/branch.txt"
git rev-parse HEAD > "$HANDOFF/revision.txt"
git bundle create "$HANDOFF/source.bundle" "$(cat "$HANDOFF/branch.txt")"
git bundle verify "$HANDOFF/source.bundle"
(cd "$HANDOFF" && sha256sum source.bundle > SHA256SUMS)
printf 'Transfer this directory after test approval: %s\n' "$HANDOFF"
```

After the test window is approved, transfer that directory to each intended
rank through its verified management connection. On **each rank**, set `HANDOFF`
to the received directory's absolute path. Use a new checkout path; cloning
refuses an existing nonempty destination:

```bash
HANDOFF=REPLACE_WITH_ABSOLUTE_RECEIVED_HANDOFF_DIRECTORY
CHECKOUT="$HOME/sparkring-setup-test"
(cd "$HANDOFF" && sha256sum --check SHA256SUMS)
git clone --branch "$(cat "$HANDOFF/branch.txt")" "$HANDOFF/source.bundle" "$CHECKOUT"
test "$(git -C "$CHECKOUT" rev-parse HEAD)" = "$(cat "$HANDOFF/revision.txt")"
cd "$CHECKOUT"
```

Use this checkout for every subsequent command. The bundled branch has no
upstream network dependency. Record the destination path in the private inventory.

## Pair rehearsal

1. Read [setup](setup.md) literally from a fresh Bash session. Record each missing
   input, unexplained decision or command that requires another document to repair.
2. Run `setup show` and local storage checks on both hosts. Compare the selection
   to the image/checkpoint already present before choosing reuse flags.
3. Inspect the existing pair with [pair-network](pair-network.md), sections 2 and 4.
   On a prepared pair, skip fresh-network changes. Record both Socket Direct
   devices, GIDs, MTUs and neighbor pings. A prepared-pair pass does not qualify
   section 3's fresh-network procedure.
4. Review the profile's container plans. Stop only the explicitly selected
   workloads in the agreed test window. Verify image and checkpoint identities,
   create the test deployment using its documented lifecycle, then start workers
   before rank 0. Retain the prior deployment for rollback.
5. Wait through cold preparation. Check both rank logs, health, model ID and a
   short generation request. Record expected versus actual output and durations.
6. Stop and restart through the guide's documented sequence from a fresh shell.
   Confirm required variables can be restored from the saved selection/site.
   Test persistent-cache restore separately when selected; health alone is not
   proof of restored cache contents.
7. Stop the test deployment and return to the recorded prior state. Report every
   manual intervention, skipped step and remaining unknown.

## Four-node approval gate

Review the pair report and resulting fixes before proceeding. The four-node
test needs a separate approved window and recorded fabric/service ownership.
Do not infer permission to change four-node networking from a successful pair test.
Repeat the same installation, serving and restart checks with all four ranks and
the managed mesh's native checks. Never reload NIC drivers beneath live RDMA users.

## Acceptance record

| Result | Required evidence |
|---|---|
| Host preparation | Tool/access results, exact source revision, destination space |
| Network configuration | Per-rank device/IP/GID/MTU results; native transport checks separately |
| Artifact verification | Selected digest, local image IDs and checkpoint verification on every rank |
| Basic serving | All-rank readiness, expected model ID and successful generation |
| Restart | Correct order, restored variables, retained assets and new readiness observations |
| Cache persistence, if tested | Matching fixtures and physical restore evidence on every rank |
| Documentation usability | Every guess, failed command, manual correction and missing recovery step |

Report each row as pass, fail or not tested. Preserve bounded scope; a smoke test
does not establish performance, maximum context or sustained-load stability.
