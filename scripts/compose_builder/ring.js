/* The same group ordering drives nodes, cables, model labels and allocation squares. */
const SparkRingFigure = (() => {
  const escape = text => String(text).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
  function draw(size, groups, profiles, measurements = []) {
    const sorted = groups.map((arc, i) => ({ arc, profile: profiles[i], index: i })).filter(g => g.arc)
      .sort((a, b) => Math.min(...a.arc) - Math.min(...b.arc));
    const groupOf = n => sorted.findIndex(g => g.arc.includes(n));
    const color = i => i < 0 ? 'var(--line)' : `var(--g${i % 4 + 1})`;
    const points = size === 2 ? [[95, 130], [305, 130]] : Array.from({ length: size }, (_, i) => {
      const angle = -Math.PI / 2 + 2 * Math.PI * i / size;
      return [200 + 143 * Math.cos(angle), 175 + 143 * Math.sin(angle)];
    });
    let svg = '<svg viewBox="0 0 400 355" role="img" aria-label="Spark allocation and direct cabling">';
    for (let i = 0; i < size; i++) {
      const j = (i + 1) % size, [x, y] = points[i], [xx, yy] = points[j];
      const g = groupOf(i), c = color(g === groupOf(j) ? g : -1);
      if (size === 2) svg += `<line x1="${x}" y1="${y + (i ? 7 : -7)}" x2="${xx}" y2="${yy + (i ? 7 : -7)}" stroke="${c}" stroke-width="3"/>`;
      else svg += `<line x1="${x}" y1="${y}" x2="${xx}" y2="${yy}" stroke="${c}" stroke-width="3"/>`;
      if (size > 3 && g >= 0 && g === groupOf(j) && sorted[g].arc.length > 2) {
        const mx = (x + xx) / 2, my = (y + yy) / 2;
        svg += `<path d="M${mx} ${my - 4}l4 4 -4 4 -4 -4Z" fill="var(--relay)"/>`;
      }
    }
    points.forEach(([x, y], i) => {
      svg += `<rect x="${x - 26}" y="${y - 16}" width="52" height="32" rx="6" fill="var(--surface)" stroke="${color(groupOf(i))}" stroke-width="1.5"/>`;
      svg += `<circle cx="${x - 14}" cy="${y}" r="3" fill="${color(groupOf(i))}"/><text x="${x + 7}" y="${y + 5}" fill="var(--ink)" font-size="13" text-anchor="middle">${i}</text>`;
      svg += `<text x="${x}" y="${y - 24}" fill="var(--muted)" font-size="10" text-anchor="middle">${i === 0 ? 'Node A' : 'Spark ' + i}</text>`;
    });
    if (sorted.length === 1) {
      const g = sorted[0], y = size === 2 ? 210 : 170;
      svg += `<text x="200" y="${y}" text-anchor="middle" fill="var(--g1)" font-size="13">${escape(g.profile.model_name)}</text>`;
      svg += `<text x="200" y="${y + 23}" text-anchor="middle" fill="var(--muted)" font-size="11">TP${g.arc.length} · Sparks ${g.arc.join(',')}</text>`;
      const layout = size === 2 ? 'pair' : size === 4 ? 'ring of four' : size === 8 ? 'ring of eight' : '';
      const speed = measurements.find(r => r.model === g.profile.model_name && r.nodes === size && r.layout === layout);
      if (speed) svg += `<text x="200" y="${y + 57}" text-anchor="middle" fill="var(--ink)" font-size="25">${speed.decode[0]}<tspan font-size="11"> tok/s</tspan></text>`;
    } else svg += `<text x="200" y="175" text-anchor="middle" fill="var(--muted)" font-size="13">${sorted.length} model groups</text>`;
    svg += '</svg><p class="eyebrow" style="text-align:center">Spark allocation</p><div class="allocation">';
    for (let i = 0; i < size; i++) svg += `<span style="border-color:${color(groupOf(i))};background:${color(groupOf(i))};color:${groupOf(i) < 0 ? 'var(--muted)' : 'var(--bg)'}">${i}</span>`;
    svg += '</div><ul class="ring-legend">' + sorted.map((g, i) => `<li style="color:${color(i)}">${escape(g.profile.model_name)} · TP${g.arc.length} · Sparks ${g.arc.join(', ')}</li>`).join('') + '</ul>';
    return svg + '<p class="hint">Positions follow cable order from Node A. Amber diamonds mark relay-capable links.</p>'
      + (measurements.length ? '<p class="hint">Shown speeds: one stream at 32K, release defaults. See the speed table for conditions.</p>' : '');
  }
  return { draw };
})();
