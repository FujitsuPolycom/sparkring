"""Viewer-owned counter windows for the terminal; the service never imports this."""
import math


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


class ViewerRates:
    def __init__(self):
        self.members = {}

    def apply(self, document):
        now = document["generated_at"]
        active = set()
        for group in document.get("groups", []):
            for member in group["members"]:
                active.add(member["id"])
                self.member(member, now, document.get("collector", {}).get("quiet", False))
        self.members = {key: value for key, value in self.members.items() if key in active}
        return document

    def member(self, member, now, quiet=False):
        fallback = member.get("counters")
        if not fallback:  # An older service can still supply derived values.
            return
        plugin = member.get("plugin_counters")
        use = (isinstance(plugin, dict) and plugin.get("schema") == "sparkring-stats/v1"
               and number(plugin.get("at")) and -5 <= now - plugin["at"] <= 180
               and number(plugin.get("monotonic")) and bool(plugin.get("run_id"))
               and all(number(plugin.get(k)) and plugin[k] >= 0 for k in ("prompt", "gen", "running", "waiting", "kv"))
               and (not number(fallback.get("run_id")) or plugin["at"] >= fallback["run_id"]))
        sample = ({"source": "plugin", "at": plugin["at"], "run_id": plugin["run_id"],
                   "clock": plugin["monotonic"], "values": plugin} if use else
                  {**fallback, "clock": fallback.get("at")})
        source, values, stamp = sample["source"], sample["values"], sample["clock"]
        state = self.members.setdefault(member["id"], {"samples": [], "history": [], "key": None})
        samples, history = state["samples"], state["history"]
        key = (source, sample["run_id"])
        keys = ("gen", "prompt", "accepted", "drafts", "draft_tokens", "steps", "ttft_sum", "ttft_count",
                "pc_hits", "pc_queries", "step_seconds")
        if state["key"] != key or (samples and (stamp is None or stamp < samples[-1]["clock"] or any(
                number(values.get(k)) and number(samples[-1]["values"].get(k))
                and values[k] < samples[-1]["values"][k] for k in keys))):
            samples.clear()
            history.clear()
            state["key"] = key
        fresh = (not quiet and member.get("state") == "serving" and number(stamp)
                 and number(sample["at"]) and -5 <= now - sample["at"] <= (180 if use else 125))
        added = fresh and (not samples or stamp > samples[-1]["clock"])
        if added:
            samples.append(sample)
            state["samples"] = samples = [s for s in samples if stamp - s["clock"] <= 360][-100:]

        def delta(k, seconds):
            if not fresh or len(samples) < 2:
                return None
            newest = samples[-1]
            base = next((s for s in samples if s["clock"] >= stamp - seconds - .5), samples[0])
            if base is newest:
                base = samples[-2]
            a, b = newest["values"].get(k), base["values"].get(k)
            dt = newest["clock"] - base["clock"]
            return (a - b, dt) if number(a) and number(b) and a >= b and dt > 0 else None

        def rate(k, seconds=15):
            d = delta(k, seconds)
            return d[0] / d[1] if d else None

        def ratio(a, b):
            x, y = delta(a, 300), delta(b, 300)
            return x[0] / y[0] if x and y and y[0] > 0 else None

        prefill = None
        if fresh and number(values.get("running")) and values["running"] > 0 and len(samples) >= 2:
            a, b = samples[-1], samples[-2]
            x, y = a["values"].get("prompt"), b["values"].get("prompt")
            if number(x) and number(y) and x > y:
                prefill = (x - y) / (a["clock"] - b["clock"])
        acceptance = ratio("accepted", "drafts")
        member["live"] = {**(member.get("live") or {}),
                          **{k: values.get(k) for k in ("running", "waiting", "kv")},
                          "source": source, "read_at": sample["at"], "prefill_tps": prefill,
                          "prompt_tps": prefill, "output_tps": rate("gen"), "output_tps_5m": rate("gen", 300),
                          "steps_ps": rate("steps") if use and values.get("step_timing") else None,
                          "acceptance": ratio("accepted", "draft_tokens"),
                          "accept_length_5m": 1 + acceptance if acceptance is not None else None,
                          "ttft_5m": ratio("ttft_sum", "ttft_count"), "prefix_hit_5m": ratio("pc_hits", "pc_queries")}
        if use:
            fallback_view = state.setdefault("fallback", ViewerRates())
            other = {**member, "plugin_counters": None}
            fallback_view.member(other, now, quiet)
            member["live"]["prefix_hit_5m"] = other["live"].get("prefix_hit_5m")
        if added:
            history.append([sample["at"], rate("gen"), values.get("running"), values.get("waiting"),
                            values.get("kv"), prefill, source])
        state["history"] = member["history"] = [h for h in history if now - h[0] <= 21600][-4321:]
