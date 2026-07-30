# NF3 NVFP4+FP8-RoPE public-bootstrap validation

## Result

On 2026-07-30, the public clean-checkout path built and attested one ARM64
SparkRing image, distributed that exact image to four directly cabled DGX
Sparks, generated the `nvfp4-rope8` profile, launched all four ranks, completed
every configured CUDA-graph capture, and served a correct OpenAI-compatible
chat completion.

This is the acceptance receipt for the commands in
[QUICKSTART.md](QUICKSTART.md). It is deliberately a configuration and
correctness gate, not a performance claim.

## Immutable inputs

| Item | Identity |
|---|---|
| Public launcher checkout | `6d936c41439348797fd2d3a0401988f12aa854cd` |
| Image-source checkout | `267289e259a9ecd86c8cdfd8e0ee4a607d37701c` |
| Built image tag | `sparkring/glm52-nf3-nvfp4-rope8:267289e259a9` |
| Image ID on every rank | `sha256:ab6bddba38aac663e2427f139aec20e022eced3716d7e204e1e881742f9da11d` |
| Target checkpoint | `madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid@66f3623dd8fefb5ca8046706912d5d31c8d196af` |
| MTP draft | `aidendle94/GLM-5.2-MXFP4-Experts-GPTQ@46537e0e16fcd156627800139b41b9c497fc7ee2`, subdirectory `mtp-draft` |
| Generated launch SHA-256 | `8c321717dfafbc95c4f8bf7022c082b71a2f450a505ab38938ea7822ec3c7fa7` |
| B12X commit | `33b0655e289927092f99f0df380907cc4eb38c1d` |
| DGX-Spark port commit | `3d3e3af92b58988a73b6266980806d3de4005ddd` |

The image was built before two launcher-only corrections at `3845d2a` and
`6d936c4`. Those commits change generated launch arguments and graph-session
controls, not files installed in the image, so the already-attested image was
intentionally reused. The final launch plan and all four running containers
were created from the later public checkout.

## Bootstrap and distribution

The bootstrap reused complete checkpoint files already present on the Sparks;
it did not redownload them. It validated the immutable model and draft
identities, built the thin derived image on rank 0, and used only direct
200 Gb/s ring edges for the 20+ GB image archive:

```text
rank 0 -> rank 1
rank 0 -> rank 3
rank 1 -> rank 2
```

The management network carried SSH commands and status only. All four ranks
reported the same image ID before launch.

## Effective serving configuration

| Setting | Observed value |
|---|---|
| served model | `glm-5.2-nf3-hybrid` |
| target path | `/models/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid` |
| TP / DCP | `4 / 4` |
| DCP backend | `ag_rs` |
| adaptive MTP | depths `2,4`, window `32`, true adaptive drafting |
| maximum sequences | `8` |
| maximum batch tokens | `4096` |
| maximum query rows / graph width | `40 / 40` |
| maximum model length | `262,144` |
| KV profile | NVFP4 compressed latent + FP8 RoPE, per-token scale |
| KV allocation | `7,000,000,000` bytes/rank |
| reported KV capacity | **875,520 tokens** |
| maximum 262,144-token concurrency | `3.34x` |
| NF3 workspace reserve | `805,306,368` bytes/rank |
| load format | `fastsafetensors` |
| prefix caching | enabled |
| SparkCache | disabled |
| reasoning / tool parser | `glm45 / glm47` |
| NCCL transport policy | `NCCL_NET=IB`, ring-only patched NCCL |

The effective Docker command, not unrelated variables inherited from the
community base image, is authoritative. The public launcher emits and attests
the command above.

## Startup evidence

The accepted run observed:

- target weights: `46/46` shards in `133.12 s`;
- target plus draft model memory: `86.87 GiB` per rank;
- 75 hybrid NF3 plans, each retaining 64 NVFP4 and 192 NF3 experts;
- target piecewise graphs: `15/15`;
- target full graphs: `16/16`;
- MTP prefill piecewise graphs: `15/15`;
- MTP prefill full graphs: `16/16`;
- decode full-graph buckets: all 8 configured buckets;
- custom graph vocabulary session on ports `10110/10111`;
- `Spark TP4 vocabulary graph captured_nodes ... count=1`;
- final `Graph capturing finished in 109 secs`;
- `/health`: HTTP `200`;
- `/v1/models`: `glm-5.2-nf3-hybrid`, maximum length `262144`.

No rank logged `NET/Socket`, no rank restarted, and all four containers
remained `running` after generation.

## Deterministic correctness request

The final gate used temperature 0, seed `20260730`, and 768 completion tokens:

```json
{
  "model": "glm-5.2-nf3-hybrid",
  "temperature": 0,
  "seed": 20260730,
  "max_tokens": 768,
  "messages": [
    {
      "role": "user",
      "content": "Write a complete Python function named triangular(n: int) -> int using the arithmetic formula, then explain it in exactly two sentences. Do not call tools."
    }
  ]
}
```

The request finished with `finish_reason: stop`. The reasoning parser returned
a separate reasoning channel, and the final content was:

````text
```python
def triangular(n: int) -> int:
    return n * (n + 1) // 2
```

This function calculates the nth triangular number by applying the arithmetic
formula `n * (n + 1) / 2`, which efficiently sums all integers from 1 to n.
Using integer division (`//`) ensures that the returned result is a precise
integer type without any floating-point decimals.
````

The response used 43 prompt tokens and 550 completion tokens and returned the
expected vLLM system fingerprint. A smaller 192-token gate correctly generated
but exhausted its budget in the reasoning channel; new smoke tests should
therefore use at least 768 tokens when the reasoning parser is enabled.

## Reproduce the gate

Follow [QUICKSTART.md](QUICKSTART.md), selecting:

```bash
python scripts/bootstrap_nf3.py execute \
  --site scripts/config/site.yaml \
  --profile nvfp4-rope8 \
  --confirmation BOOTSTRAP-NF3-ALL-FOUR
```

Then check:

```bash
python scripts/sparkring_launcher.py \
  --site .sparkring/bootstrap/site.yaml \
  --launch-config .sparkring/bootstrap/launch.nvfp4-rope8.json \
  --execute status

curl -fsS http://RANK0-MANAGEMENT-IP:8000/health
curl -fsS http://RANK0-MANAGEMENT-IP:8000/v1/models
```

The normal pre-launch full preflight requires API and master ports to be free.
If rerun after the model is serving, exactly those rank-0 port checks will
properly report occupied; the accepted post-launch evidence instead uses
launcher status, container restart counts, image IDs, API health, and the
correctness request.
