# GLM TP4 Compose creation

Status: **Development**. R35/R37 containers are generated from the same GLM
specification for Docker and Compose. CPU tests compare all four ranks, DCP1/DCP4,
and cache on/off against the retained launcher. [Docker and Compose creation](../../../performance/records/glm53-flash/glm-container-creation.json)
passed on one GB10 host with the containers stopped. GLM serving and four-rank
managed-lifecycle acceptance of this path remain pending.

## Host preparation

Follow the [primary quickstart](../README.md) through image/model/fabric staging.
Use its explicit R37 receipt and the deployment suite's staged preparation file.
Install the Docker Compose plugin and Python dependencies on each host.

## Configure and render

Select Compose only for the stopped-container creation plan:

```bash
python3 scripts/sparkring.py deploy runtime-plan create \
  --preparation /path/to/staged/preparation.json --container-backend compose \
  --output /path/to/create-plan.json
```

Review and apply this plan with the [deployment suite](../../../docs/operations/deployment-suite.md#create-containers-install-services-and-test-the-mesh),
then continue its install, mesh, native-check, managed-start and readiness phases.
The creation plan generates private `rankN/plan.json` and `compose.yaml` under
`launch-containers/`, beside each host's authenticated `launch/` directory.
It checks their canonical inputs before use.
Do not edit generated YAML or start it independently of the managed gates.

## Scope of translation

The shared specification preserves image/lease verification, mHC, KDA coalescing,
SIRCL, NCCL, graph settings, mounts and health behavior. The managed coordinator
continues to own memory preparation, authenticated fabric readiness, scheduler
observation and recovery. The general deployment suite accepts DCP1; the
quickstart's separate R33 DCP4 reproduction remains unchanged.

Generate YAML from the staged preparation so container settings remain bound to
the source and lifecycle contracts. Published images and frozen inputs retain
their recorded identities.
