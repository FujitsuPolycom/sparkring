// Viewer-local history. No network, timers, storage or server-side rate work.
class ViewerRates {
  constructor() { this.members = new Map(); }
  seed(doc) {
    this.serviceStarted = doc.service_started;
    for (const [id, rows] of Object.entries(doc.counter_history || {}))
      for (const row of rows) this.ingest(id, row);
  }
  ingest(id, row) {
    if (row.seq <= (this.members.get(id)?.seq ?? -1)) return;
    this.member({ id, state: "serving", ...row }, row.observed_at, false);
    this.members.get(id).seq = row.seq;
  }
  apply(doc) {
    if (this.serviceStarted !== undefined && doc.service_started !== this.serviceStarted) {
      for (const state of this.members.values()) {
        state.samples = []; state.key = null; delete state.seq; delete state.fallback;
      }
    }
    this.serviceStarted = doc.service_started;
    const ids = new Set();
    for (const g of doc.groups || []) for (const m of g.members) {
      ids.add(m.id); this.member(m, doc.generated_at, !!doc.collector?.quiet);
    }
    for (const id of this.members.keys()) if (!ids.has(id)) this.members.delete(id);
    return doc;
  }
  member(m, now, quiet) {
    for (const row of m.counter_samples || []) this.ingest(m.id, row);
    const fallback = m.counters;
    if (!fallback) {
      const state = this.members.get(m.id);
      if (state) {
        state.samples = [];
        if (state.history.length && state.history.at(-1).slice(1).some(v => v !== null))
          state.history.push([now, null, null, null, null, null, null]);
        state.history = state.history.filter(h => now - h[0] <= 21600).slice(-4321);
        m.history = state.history;
      }
      return;
    }
    const p = m.plugin_counters, num = Number.isFinite;
    const use = p && p.schema === "sparkring-stats/v1" && num(p.at) && now - p.at >= -5
      && now - p.at <= 180 && num(p.monotonic) && typeof p.run_id === "string" && p.run_id.length > 0 && p.run_id.length <= 128
      && ["prompt", "gen", "running", "waiting", "kv"].every(k => num(p[k]) && p[k] >= 0)
      && (!num(fallback.run_id) || p.at >= fallback.run_id);
    const sample = use ? { source: "plugin", at: p.at, run_id: p.run_id, clock: p.monotonic, values: p }
      : { ...fallback, clock: fallback.at };
    const { source, values: v, clock: stamp } = sample;
    if (!this.members.has(m.id)) this.members.set(m.id, { samples: [], history: [], key: null });
    const state = this.members.get(m.id), key = JSON.stringify([source, sample.run_id]);
    const keys = ["gen", "prompt", "accepted", "drafts", "draft_tokens", "steps", "ttft_sum", "ttft_count",
      "pc_hits", "pc_queries", "step_seconds"];
    const last = state.samples.at(-1);
    if (quiet || m.boundary || key !== state.key || (last && (sample.at - last.at > 185 || !num(stamp) || stamp < last.clock
      || keys.some(k => num(v[k]) && num(last.values[k]) && v[k] < last.values[k])))) {
      state.samples = []; state.key = key;
      if (state.history.length && state.history.at(-1).slice(1).some(v => v !== null))
        state.history.push([sample.at, null, null, null, null, null, null]);
    }
    const fresh = !quiet && m.state === "serving" && num(stamp) && num(sample.at)
      && now - sample.at >= -5 && now - sample.at <= (use ? 180 : 125);
    const added = fresh && (!state.samples.length || stamp > state.samples.at(-1).clock);
    if (added) {
      state.samples.push(sample);
      state.samples = state.samples.filter(s => stamp - s.clock <= 360).slice(-100);
    }
    const samples = state.samples;
    const delta = (k, seconds) => {
      if (!fresh || samples.length < 2) return null;
      const newest = samples.at(-1);
      let base = samples.find(s => s.clock >= stamp - seconds - .5) || samples[0];
      if (base === newest) base = samples.at(-2);
      const a = newest.values[k], b = base.values[k], dt = newest.clock - base.clock;
      return num(a) && num(b) && a >= b && dt > 0 ? [a - b, dt] : null;
    };
    const rate = (k, seconds = 15) => { const d = delta(k, seconds); return d ? d[0] / d[1] : null; };
    const ratio = (a, b) => {
      const x = delta(a, 300), y = delta(b, 300); return x && y && y[0] > 0 ? x[0] / y[0] : null;
    };
    let prefill = null;
    if (fresh && v.running > 0 && samples.length >= 2) {
      const a = samples.at(-1), b = samples.at(-2), x = a.values.prompt, y = b.values.prompt;
      if (num(x) && num(y) && x > y) prefill = (x - y) / (a.clock - b.clock);
    }
    const acceptance = ratio("accepted", "drafts");
    m.live = { ...m.live, running: v.running, waiting: v.waiting, kv: v.kv, source, read_at: sample.at,
      prefill_tps: prefill, prompt_tps: prefill, output_tps: rate("gen"), output_tps_5m: rate("gen", 300),
      steps_ps: use && v.step_timing ? rate("steps") : null, acceptance: ratio("accepted", "draft_tokens"),
      accept_length_5m: acceptance === null ? null : 1 + acceptance,
      ttft_5m: ratio("ttft_sum", "ttft_count"), prefix_hit_5m: ratio("pc_hits", "pc_queries") };
    if (use) {
      state.fallback ||= new ViewerRates();
      const other = { ...m, plugin_counters: null, counter_samples: [] };
      state.fallback.member(other, now, quiet);
      m.live.prefix_hit_5m = other.live.prefix_hit_5m;
    }
    if (added) state.history.push([sample.at, rate("gen"), v.running, v.waiting, v.kv, prefill, source]);
    state.history = state.history.filter(h => now - h[0] <= 21600).slice(-4321);
    m.history = state.history;
  }
}
