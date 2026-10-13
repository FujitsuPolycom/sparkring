(() => {
  // Preserve links created before the landing page occupied the site root.
  const incoming = new URLSearchParams(location.search);
  if (incoming.has('profile') || incoming.get('mode') === 'compose') {
    location.replace('builder.html' + location.search + location.hash);
    return;
  }
  const data = JSON.parse(document.getElementById('site-data').textContent);
  const profiles = data.profiles, byId = Object.fromEntries(profiles.map(p => [p.id, p]));
  let size = 2, example = 0;
  function examples(n) {
    const whole = profiles.find(p => p.nodes === n && p.id.startsWith(n === 8 ? 'glm53-nvfp4-tp8' : n === 4 ? 'deepseek' : 'glm53-flash-nvfp4-spark-tp2')) || profiles.find(p => p.nodes === n);
    const result = [{ name: `One model · TP${n}`, groups: [{ id: whole.id, arc: Array.from({ length: n }, (_, i) => i) }] }];
    if (n > 2) {
      const pairs = profiles.filter(p => p.nodes === 2 && p.status !== 'research-only');
      result.push({ name: `${n / 2} pairs`, groups: Array.from({ length: n / 2 }, (_, i) => ({ id: pairs[i % pairs.length].id, arc: [2 * i, 2 * i + 1] })) });
    }
    if (n === 8) result.push({ name: 'Two halves', groups: profiles.filter(p => p.nodes === 4).slice(0, 2).map((p, i) => ({ id: p.id, arc: [0, 1, 2, 3].map(x => x + i * 4) })) });
    return result;
  }
  function render() {
    const choices = examples(size), selected = choices[example];
    document.getElementById('ring-choices').innerHTML = [2, 4, 8].map(n => `<button data-size="${n}" aria-pressed="${n === size}">${n === 2 ? 'Pair' : 'Ring of ' + (n === 4 ? 'four' : 'eight')}</button>`).join('');
    document.getElementById('ring-examples').innerHTML = choices.map((e, i) => `<button data-example="${i}" aria-pressed="${i === example}">${e.name}</button>`).join('');
    document.getElementById('ring-figure').innerHTML = SparkRingFigure.draw(size, selected.groups.map(g => g.arc), selected.groups.map(g => byId[g.id]), data.speed);
    const query = new URLSearchParams({ sparks: size, models: selected.groups.map(g => g.id).join(';'), where: selected.groups.map(g => g.arc.join(',')).join(';') });
    document.getElementById('load-example').href = 'builder.html?' + query;
  }
  document.getElementById('ring-choices').addEventListener('click', e => { if (e.target.dataset.size) { size = Number(e.target.dataset.size); example = 0; render(); } });
  document.getElementById('ring-examples').addEventListener('click', e => { if (e.target.dataset.example !== undefined) { example = Number(e.target.dataset.example); render(); } });
  render();
})();
