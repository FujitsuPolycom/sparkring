(() => {
  // Preserve links created before the landing page occupied the site root.
  const incoming = new URLSearchParams(location.search);
  if (incoming.has('profile') || incoming.has('p') || incoming.get('mode') === 'compose') {
    location.replace('builder.html' + location.search + location.hash);
    return;
  }
  const data = JSON.parse(document.getElementById('site-data').textContent);
  // On its own site the quick start fetches the install.sh copy published there, whose default --ref is
  // the page's source (site.py hosted_installer); elsewhere it fetches install.sh at that ref.
  const command = document.getElementById('install-command');
  if (command && data.meta.site_url && location.href.startsWith(data.meta.site_url)) command.textContent = command.dataset.hosted;
  const profiles = data.profiles, byId = Object.fromEntries(profiles.map(p => [p.id, p]));
  const esc = text => String(text).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
  const color = i => i < 0 ? 'var(--muted)' : `var(--g${i % 4 + 1})`;
  const arc = n => Array.from({ length: n }, (_, i) => i);
  let size = 2, example = 0;
  function examples(n) {
    const pair = profiles.find(p => p.id === 'glm53-flash-nvfp4-spark-tp2');
    const four = profiles.find(p => p.id === 'deepseek-v41-flash-tp4');
    const group = (p, positions) => ({ id: p.id, arc: positions });
    if (n === 3) return [{ name: 'Pair + one free Spark', groups: [group(pair, arc(2))] }];
    if (n === 5) return [{ name: 'TP4 + one free Spark', groups: [group(four, arc(4))] }];
    const pairs = profiles.filter(p => p.nodes === 2 && p.status !== 'research-only');
    const split = { name: `${n / 2} pairs`, groups: Array.from({ length: n / 2 }, (_, i) => group(pairs[i % pairs.length], [2 * i, 2 * i + 1])) };
    // Research profile provenance: codex/glm53-tp6 at 438b6189,
    // profiles/glm53-nvfp4-csf-tp6/profile.json. No shipped image admits it.
    if (n === 6) return [split, {
      name: 'GLM-5.3 TP6 · research',
      groups: [{ id: 'glm53-nvfp4-csf-tp6', arc: arc(6), model_name: 'GLM-5.3', status: 'research-only' }],
      note: 'Research only: this profile targets Sparks 0–5 of an eight-Spark ring. It is not available in this builder and has no shipped image support.'
    }];
    const whole = n === 2 ? pair : n === 4 ? four : profiles.find(p => p.id === 'glm53-nvfp4-tp8');
    const research = whole.status === 'research-only' ? ' · research' : '';
    const result = [{ name: `One model · TP${n}${research}`, groups: [group(whole, arc(n))] }];
    if (n > 2) result.push(split);
    if (n === 8) result.push({ name: 'Two halves', groups: profiles.filter(p => p.nodes === 4).slice(0, 2).map((p, i) => group(p, arc(4).map(x => x + i * 4))) });
    return result;
  }
  function drawRing(n, groups) {
    const cx = 200, cy = 200, r = 140;
    const point = (i, radius) => {
      const a = -Math.PI / 2 + i * 2 * Math.PI / n;
      return [cx + radius * Math.cos(a), cy + radius * Math.sin(a)];
    };
    const points = n === 2 ? [[110, 200], [290, 200]] : arc(n).map(i => point(i, r));
    let svg = `<svg viewBox="0 0 400 400" role="img" aria-label="${n} directly cabled Sparks, numbered from 0">`;
    if (n === 2) {
      svg += '<path d="M140 188 Q200 160 260 188 M140 212 Q200 240 260 212" fill="none" stroke="var(--g1)" stroke-width="3"/>';
    } else {
      points.forEach(([x, y], i) => {
        const [xx, yy] = points[(i + 1) % n];
        svg += `<line x1="${x}" y1="${y}" x2="${xx}" y2="${yy}" stroke="var(--g1)" stroke-width="3" stroke-linecap="round"/>`;
      });
      // Odd rings have two equally distant positions; follow the clockwise route.
      const far = Math.floor(n / 2), relays = far - 1;
      if (relays > 0) {
        const route = arc(far + 1).map(i => point(i, r - 42).map(v => v.toFixed(1)).join(',')).join(' ');
        svg += '<defs><marker id="relay-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10" fill="none" stroke="var(--relay)" stroke-width="1.5"/></marker></defs>';
        svg += `<polyline points="${route}" fill="none" stroke="var(--relay)" stroke-width="2" stroke-dasharray="6 6" stroke-linejoin="round" marker-end="url(#relay-arrow)"/>`;
        svg += `<text x="200" y="226" text-anchor="middle" font-size="12" fill="var(--relay)">0 → ${far}: ${relays} relay${relays === 1 ? '' : 's'}</text>`;
      }
    }
    points.forEach(([x, y], i) => {
      const g = groups.findIndex(group => group.arc.includes(i));
      svg += `<g><rect x="${x - 30}" y="${y - 19}" width="60" height="38" rx="7" fill="var(--surface)" stroke="var(--ink)" stroke-width="1.5"/>`;
      if (g >= 0) svg += `<rect x="${x - 30}" y="${y - 19}" width="60" height="38" rx="7" fill="${color(g)}" opacity=".22"/>`;
      svg += `<circle cx="${x - 18}" cy="${y}" r="3.5" fill="${color(g)}"/><text x="${x + 6}" y="${y + 5}" text-anchor="middle" font-size="14" fill="var(--ink)">${i}</text></g>`;
    });
    if (n > 2) svg += `<text x="200" y="200" text-anchor="middle" font-size="17" fill="var(--muted)">${n} Sparks</text>`;
    return svg + '</svg>';
  }
  function render() {
    const choices = examples(size), selected = choices[example];
    const names = { 2: 'Pair', 3: 'Ring of three', 4: 'Ring of four', 5: 'Ring of five', 6: 'Ring of six', 8: 'Ring of eight' };
    document.getElementById('ring-choices').innerHTML = '<span class="seg-label">Sparks</span><span class="seg">'
      + [2, 3, 4, 5, 6, 8].map(n => `<button data-size="${n}" aria-label="${n} Sparks" aria-pressed="${n === size}">${n}</button>`).join('')
      + `</span><span class="seg-name">${names[size]}</span>`;
    document.getElementById('ring-examples').innerHTML = choices.map((e, i) => `<button data-example="${i}" aria-pressed="${i === example}">${e.name}</button>`).join('');
    document.getElementById('ring-figure').innerHTML = drawRing(size, selected.groups);
    document.getElementById('ring-caption').textContent = size === 2 ? 'Two Sparks, two direct ConnectX-7 cables.'
      : size === 3 ? 'Three Sparks, each directly connected to the other two.'
      : `${size} Sparks, two neighbours each; a Spark relays to reach across the ring.`;
    const free = arc(size).filter(i => !selected.groups.some(g => g.arc.includes(i)));
    document.getElementById('example-list').innerHTML = selected.groups.map((g, i) => {
      const p = byId[g.id] || g;
      return `<li><i style="background:${color(i)}"></i><span><b>${esc(p.model_name)}</b> · Sparks ${g.arc.join(', ')}${p.status === 'research-only' ? ' · research' : ''}<br><code>--profile ${esc(g.id)}</code></span></li>`;
    }).join('') + (free.length ? `<li class="muted">Spark ${free.join(', ')} stays free.</li>` : '')
      + (selected.note ? `<li class="muted">${esc(selected.note)}</li>` : '');
    const query = new URLSearchParams({ sparks: size, models: selected.groups.map(g => g.id).join(';'), where: selected.groups.map(g => g.arc.join(',')).join(';') });
    document.getElementById('load-example').href = 'builder.html?' + query;
    document.getElementById('load-example').hidden = Boolean(selected.note);
  }
  document.getElementById('ring-choices').addEventListener('click', e => { if (e.target.dataset.size) { size = Number(e.target.dataset.size); example = 0; render(); } });
  document.getElementById('ring-examples').addEventListener('click', e => { if (e.target.dataset.example !== undefined) { example = Number(e.target.dataset.example); render(); } });
  render();
})();
