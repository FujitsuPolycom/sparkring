"""Score a fixed text with prompt_logprobs and compare two servers' token probabilities.

Usage: logprob_probe.py BASE_URL OUTPUT_JSON
       logprob_probe.py --compare A_JSON B_JSON
Each saved position maps token IDs to {logprob, rank}: the prompt token, plus the
rank-1 token when the prompt token is not rank 1. Comparing two runs reports the
mean, 99th percentile and maximum absolute difference of the prompt tokens'
log-probabilities, and how often the rank-1 token agrees.
"""
import json
import statistics
import sys
import urllib.request

PARAGRAPH = (
    "The history of computing is a story of abstraction layered on abstraction. Early machines were programmed by "
    "rewiring panels; later, stored programs let the same hardware compute anything that could be described. "
    "Assemblers replaced raw opcodes with mnemonics, compilers replaced assembly with expressions, and operating "
    "systems turned scarce hardware into shared, protected resources. Each layer hid the one beneath it while "
    "leaking just enough of its behavior to matter.\n\n"
    "def merge_sorted(left, right):\n    result = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n"
    "        if left[i] <= right[j]:\n            result.append(left[i])\n            i += 1\n        else:\n"
    "            result.append(right[j])\n            j += 1\n    result.extend(left[i:])\n    result.extend(right[j:])\n"
    "    return result\n\n"
    "Consensus protocols such as Paxos and Raft make a group of unreliable machines agree on an ordered log, and the "
    "price of that agreement is latency: every committed entry needs a quorum of acknowledgements.\n\n"
    '{"orders": [{"id": 1041, "customer": "Ada", "items": 3, "total": 58.20}, {"id": 1042, "customer": "Grace", '
    '"items": 1, "total": 12.99}]}\n\n'
)
TEXT = PARAGRAPH * 4


def run(base, output):
    base = base.rstrip("/")
    model = json.loads(urllib.request.urlopen(base + "/models", timeout=30).read())["data"][0]["id"]
    body = {"model": model, "prompt": TEXT, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1}
    request = urllib.request.Request(base + "/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    positions = json.loads(urllib.request.urlopen(request, timeout=600).read())["choices"][0]["prompt_logprobs"][1:]
    json.dump({"model": model, "positions": positions}, open(output, "w"))
    print(model, "scored tokens:", len(positions))


def view(position):
    ranked = {token: value["rank"] for token, value in position.items()}
    top = min(ranked, key=ranked.get)
    prompt = next((token for token, rank in ranked.items() if rank != 1), top)
    return position[prompt]["logprob"], top


def compare(first, second):
    a = json.load(open(first))["positions"]
    b = json.load(open(second))["positions"]
    n = min(len(a), len(b))
    diffs, agree = [], 0
    for x, y in zip(a[:n], b[:n]):
        lx, tx = view(x)
        ly, ty = view(y)
        diffs.append(abs(lx - ly))
        agree += tx == ty
    diffs.sort()
    print(f"tokens {n}: mean |dlogprob| {statistics.mean(diffs):.4f}, p99 {diffs[int(0.99 * (n - 1))]:.4f}, "
          f"max {diffs[-1]:.4f}, rank-1 agreement {agree / n:.4f}")


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        run(sys.argv[1], sys.argv[2])
