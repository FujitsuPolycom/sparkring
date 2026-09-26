# Qwen3.8-Flash-Next QAD on four Sparks: Compose files

[`compose.rank0.yaml`](compose.rank0.yaml) to [`compose.rank3.yaml`](compose.rank3.yaml)
run the [`qwen38-flash-next-qad-tp4`](../README.md) profile on four DGX Sparks,
one Compose project per Spark. Rank 0 serves `Qwen3.8-Flash-Next-NVFP4-QAD-TP4`
on port 8015, with no API key. Each rank is the container
[`sparkring install`](../../../docs/operations/install.md) runs, on image
`dev-20260925-qwendecode-cuda1342-nccl2323-status031`, without the per-rank
runtime-status binding file.

The files are generated from the [configuration](../config.json) and the
[example site](site.example.yaml); regenerate them instead of editing the YAML.

## Use

`sparkring install` is the supported way to run this profile. The ranks need
the four-Spark native mesh that it creates; these files do not create it.

To render files for your own ring, copy the site example and fill in every
host, address, interface, directory and fabric identity:

```bash
mkdir -p .sparkring
cp profiles/qwen38-flash-next-qad-tp4/compose/site.example.yaml .sparkring/qwen-qad.site.yaml
# Edit .sparkring/qwen-qad.site.yaml.
python3 scripts/sparkring.py compose render qwen38-flash-next-qad-tp4 \
  --site .sparkring/qwen-qad.site.yaml --output .sparkring/deployments/qwen-qad
python3 scripts/sparkring.py compose check --deployment .sparkring/deployments/qwen-qad
```

On each Spark, pull the image by its digest
(`ghcr.io/fujitsupolycom/sparkring@sha256:451c5e23a90e0df2fc904e8851aab12c3ec9ffdcd1258b6f14cf502222e46b5f`)
and run `docker compose -f compose.yaml up -d` with that rank's
`rankN/compose.yaml`: ranks 1, 2 and 3 first, then rank 0. Compose reads the
loader seccomp policy from `runtime/common/loader-seccomp.json` under the
site's `repository` directory on each Spark.

Not supported: `sparkring compose start` for this profile. It stops with
`Shared toolchain images are admitted by the installer image lock`.
