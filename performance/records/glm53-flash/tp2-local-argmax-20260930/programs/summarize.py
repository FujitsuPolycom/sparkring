"""Build results/summary.json from the raw pass and benchmark files in the working directory.

Inputs: decode_passes.py outputs named {stock,stock2,flag}-{greedy,sample}-*.json and
llm-inference-bench outputs named bench-{stock,stock2,flag}-N.json. "stock2" is the
stock deployment after reinstallation; it counts as the stock variant.
"""
import glob
import itertools
import json
import statistics


def variant_of(name):
    return "flag" if name.startswith("flag") else "stock"


def spec(document):
    delta = document["spec_delta"]
    drafts = delta["vllm:spec_decode_num_drafts_total"]
    draft_tokens = delta["vllm:spec_decode_num_draft_tokens_total"]
    accepted = delta["vllm:spec_decode_num_accepted_tokens_total"]
    return {
        "drafts": int(drafts),
        "draft_tokens": int(draft_tokens),
        "accepted_tokens": int(accepted),
        "acceptance_rate": round(accepted / draft_tokens, 4),
        "mean_acceptance_length": round(1 + accepted / drafts, 4),
        "per_position_acceptance": [
            round(delta[f"vllm:spec_decode_num_accepted_tokens_per_pos_total[{position}]"] / drafts, 4)
            for position in range(3)
        ],
    }


def first_divergence(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def main():
    out = {"greedy_passes": {}, "sampled_passes": {}, "greedy_pairs": [], "bench_runs": [],
           "temperature_1_pooled": {}}
    greedy = {}
    for path in sorted(glob.glob("*-greedy-*.json")):
        document = json.load(open(path, encoding="utf-8"))
        name = path[:-5]
        greedy[name] = document
        tokens = sum(len(result["token_ids"] or []) for result in document["results"])
        out["greedy_passes"][name] = spec(document) | {"output_tokens": tokens}
    for path in sorted(glob.glob("*-sample-*.json")):
        out["sampled_passes"][path[:-5]] = spec(json.load(open(path, encoding="utf-8")))

    for a, b in itertools.combinations(sorted(greedy), 2):
        pairs = list(zip(greedy[a]["results"], greedy[b]["results"]))
        divergences = [first_divergence(x["token_ids"], y["token_ids"]) for x, y in pairs]
        out["greedy_pairs"].append({
            "a": a, "b": b,
            "kind": "same-variant" if variant_of(a) == variant_of(b) else "cross-variant",
            "identical_prompts": sum(x["token_ids"] == y["token_ids"] for x, y in pairs),
            "prompts": len(pairs),
            "first_divergence_median": statistics.median(divergences),
            "first_divergence_mean": round(statistics.mean(divergences), 1),
            "first_divergence_min": min(divergences),
            "first_divergence_max": max(divergences),
        })

    pool = {}

    def add(variant, drafts, draft_tokens, accepted):
        totals = pool.setdefault(variant, [0, 0, 0])
        totals[0] += drafts
        totals[1] += draft_tokens
        totals[2] += accepted

    for path in sorted(glob.glob("bench-*.json")):
        if "metrics" in path:
            continue
        label = path[len("bench-"):-5]
        for result in json.load(open(path, encoding="utf-8"))["results"]:
            row = {
                "run": label,
                "variant": label.rsplit("-", 1)[0],
                "concurrency": result["concurrency"],
                "tokens_per_s": round(result["aggregate_tps"], 2),
                "steps_per_s": round(result["server_steps_per_s"], 2),
                "acceptance_length": round(result["server_accept_len_effective"], 3),
                "ttft_p50_ms": round(result["ttft_p50"] * 1000, 1),
                "itl_p50_ms": round(result["inter_token_latency_p50"] * 1000, 2),
                "drafts": result["server_spec_drafts"],
                "draft_tokens": result["server_spec_draft_tokens"],
                "accepted_tokens": result["server_spec_accepted_tokens"],
            }
            out["bench_runs"].append(row)
            add(variant_of(label), row["drafts"], row["draft_tokens"], row["accepted_tokens"])
    for name, passes in out["sampled_passes"].items():
        add(variant_of(name), passes["drafts"], passes["draft_tokens"], passes["accepted_tokens"])
    for variant, (drafts, draft_tokens, accepted) in pool.items():
        out["temperature_1_pooled"][variant] = {
            "drafts": drafts, "draft_tokens": draft_tokens, "accepted_tokens": accepted,
            "acceptance_rate": round(accepted / draft_tokens, 4),
            "mean_acceptance_length": round(1 + accepted / drafts, 4),
        }
    with open("summary.json", "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1)


if __name__ == "__main__":
    main()
