(() => {
  'use strict';
  const E = SparkRingEngine, data = JSON.parse(document.getElementById('site-data').textContent);
  const $ = id => document.getElementById(id), byId = Object.fromEntries(data.profiles.map(p => [p.id, p]));
  const esc = text => String(text ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
  const option = (value, text, selected) => `<option value="${esc(value)}"${selected ? ' selected' : ''}>${esc(text)}</option>`;
  const query = new URLSearchParams(location.search), aliases = data.aliases || {};
  data.meta.installer = E.hostedInstaller(data.meta, location.href);
  let size = Number(query.get('sparks')) || 2;
  if (size < 2 || size > 8 || !Number.isInteger(size)) size = 2;
  // The planner keeps no saved state: it starts on the install default image unless a link names another.
  let image = data.images.find(r => r.default) || data.images[0];
  const freshSlot = id => E.planSlot(id);
  const first = data.profiles.find(p => p.nodes <= size && image.profiles[p.id]) || data.profiles[0];
  let slots = [freshSlot(first.id)], setupOnly = query.get('setup') === '1', names = Array.from({ length: 8 }, () => ({ host: '', host_ip: '' }));
  // A builder link (`profile` or `p`, by ID or short name) selects its model, Sparks and image (E.planLink).
  const linked = query.has('plan') ? null : E.planLink(location.search, byId, data.images, aliases);
  // Shareable plans contain only public choices, never machine names or addresses.
  if (query.has('plan')) {
    try {
      const saved = JSON.parse(query.get('plan'));
      if (Number.isInteger(saved.size) && saved.size >= 2 && saved.size <= 8) size = saved.size;
      if (Array.isArray(saved.slots) && saved.slots.length && saved.slots.length <= 4 && saved.slots.every(s => byId[s.id])) {
        slots = saved.slots.map(s => ({ ...freshSlot(s.id), where: String(s.where || 'auto'), checkpoint: s.checkpoint ?? null,
          entries: s.entries && typeof s.entries === 'object' ? s.entries : {}, endpoint: '' }));
      }
      image = data.images.find(r => r.name === saved.image) || image;
      setupOnly = saved.setupOnly === true;
    } catch (_) { /* Invalid links leave the usable default. */ }
  } else if (linked) {
    ({ size, slots } = linked);
    image = data.images.find(r => r.name === linked.image);
  } else {
    const ids = (query.get('models') || query.get('model') || '').split(';'), where = (query.get('where') || '').split(';');
    if (ids.length <= 4 && ids.every(id => byId[id])) slots = ids.map((id, i) => ({ ...freshSlot(id), where: where[i] || 'auto' }));
  }
  function selectedProfile(slot) {
    const base = byId[slot.id], admission = image.profiles[slot.id];
    const checkpoints = base.checkpoints.filter(c => !admission || admission.checkpoints.includes(c.name))
      .map(c => ({ ...c, install_default: admission?.preferred ? c.name === admission.preferred : c.default }));
    return { ...base, checkpoints, installable: Boolean(admission), image_option: image.default ? null : image.option,
      image_capabilities: image.capabilities, image_release: image.name, sircl: image.sircl };
  }
  function checkpoint(slot, p) {
    return (slot.checkpoint !== null ? p.checkpoints.find(c => c.name === slot.checkpoint) : null) || p.checkpoints.find(c => c.install_default) || p.checkpoints[0];
  }
  function slotData(slot) {
    const p = selectedProfile(slot), cp = checkpoint(slot, p);
    if (!cp) return { profile: p, where: slot.where, problems: [{ message: 'No checkpoint is admitted by this image.' }] };
    const read = E.readSelection(p, cp, slot.entries, { mode: 'set', address: slot.endpoint });
    return { profile: p, checkpoint: cp.name, settings: read.settings, where: slot.where, problems: read.problems,
      endpoint: { address: read.address } };
  }
  function renderSlots() {
    $('setup-only').checked = setupOnly;
    $('model-slots').hidden = setupOnly; $('add-model').hidden = setupOnly;
    $('model-slots').innerHTML = slots.map((slot, i) => {
      const p = selectedProfile(slot), cp = checkpoint(slot, p), fit = data.profiles.filter(p => E.arcs(size, p.nodes).length);
      if (!fit.some(p => p.id === slot.id)) fit.unshift(byId[slot.id]);
      const where = E.arcs(size, p.nodes);
      const fields = (cp?.settings || []).filter(row => E.offered(p, row)).map(row => {
        const id = `model-${i}-${row.name}`, value = slot.entries[row.name];
        let field;
        if (row.switch) field = `<input type="checkbox" id="${id}" data-setting="${row.name}"${value ? ' checked' : ''}>`;
        // A SparkCache payload check (export._setting_rows `choice`): a list whose first, empty entry keeps the profile's value.
        else if (row.choice) field = `<select id="${id}" data-setting="${row.name}">${option('', row.profile + ' (default)', !value)}${row.values.filter(v => v !== row.profile).map(v => option(v, v, value === v)).join('')}</select>`;
        // A SparkCache size (`size`) takes a unit letter, such as 48G; the whole-number settings get the numeric keyboard.
        else field = `<input id="${id}" data-setting="${row.name}" value="${esc(value ?? '')}" placeholder="${esc(row.profile ?? '')}"${row.address || row.size ? '' : ' inputmode="numeric"'}>`;
        return `<label for="${id}">${esc(row.option)}${row.profile !== undefined ? ' · default ' + esc(row.profile) : ''}</label>` + field
          + `<p class="hint">${esc(row.help)}</p>`;
      }).join('');
      return `<div class="planner-card model-row" data-slot="${i}" style="border-left-color:var(--g${i % 4 + 1})">
        <div class="model-head"><h3>Model ${i + 1}</h3>${slots.length > 1 ? `<button type="button" data-remove="${i}">Remove model ${i + 1}</button>` : ''}</div>
        <label for="model-${i}">Model / profile</label><select id="model-${i}" data-model>${fit.map(q => option(q.id, `${q.model_name} · TP${q.nodes} · ${q.id}${q.status === 'research-only' ? ' (research)' : ''}`, q.id === slot.id)).join('')}</select>
        <label for="where-${i}">Where</label><select id="where-${i}" data-where>${option('auto', 'Automatic — first free group from Node A', slot.where === 'auto')}${where.map(a => option(a.join(','), 'Sparks ' + a.join(', '), slot.where === a.join(','))).join('')}</select>
        <p class="hint">Thinking ${esc(cp?.thinking || p.thinking)} · ${esc(cp?.variant || p.variant)}. <a href="https://github.com/${esc(data.meta.repository)}/blob/${esc(data.meta.ref)}/${esc(p.guide)}">Profile guide</a></p>
        <details><summary>Checkpoint, serving settings and API</summary>
        <label for="checkpoint-${i}">Checkpoint</label><select id="checkpoint-${i}" data-checkpoint>${p.checkpoints.map(c => option(c.name || '', (c.name || 'Profile default') + (c.install_default ? ' (installer default)' : ''), c === cp)).join('')}</select>
        ${fields}<label for="endpoint-${i}">Address people use (optional)</label><input id="endpoint-${i}" data-endpoint value="${esc(slot.endpoint)}" placeholder="llm.example.net">
        <p class="hint">Leave blank to use the first Spark in this group. Its API may only be reachable from Node A if that Spark has no LAN cable.</p></details></div>`;
    }).join('');
  }
  function renderOutput() {
    const form = $('plan-form').value, approval = $('plan-approval');
    approval.querySelector('[value="plan"]').disabled = form === 'script';
    if (form === 'script' && approval.value === 'plan') approval.value = 'ask';
    if ($('plan-order').value === 'ask' && approval.value === 'yes') approval.value = 'ask';
    const pack = E.slicePack(size, setupOnly ? [] : slots.map(slotData), data.meta,
      { setupOnly, form, pin: $('plan-source').value === 'checkout', approval: approval.value,
        order: $('plan-order').value, sparks: names.slice(0, size), downloadLimit: $('plan-limit').value });
    $('plan-problems').textContent = pack.problems.join(' ');
    $('plan-commands').innerHTML = pack.ready ? pack.commands.map((c, i) => `<div class="command-box"><h3>${i + 1} · ${esc(c.what)}</h3><pre id="plan-command-${i}">${esc(c.command)}</pre><button data-copy="plan-command-${i}">Copy command</button>${c.endpoint ? `<p class="hint">API <code>${esc(c.endpoint)}</code></p>` : ''}</div>`).join('') : '<p>Resolve the choices above to build commands.</p>';
    $('plan-share').disabled = !pack.ready;
    $('plan-ring').innerHTML = SparkRingFigure.draw(size, pack.groups, setupOnly ? [] : slots.map(s => byId[s.id]));
    const sorted = pack.groups.map((arc, i) => ({ arc, i })).filter(g => g.arc).sort((a, b) => Math.min(...a.arc) - Math.min(...b.arc));
    sorted.forEach((g, color) => { const row = document.querySelector(`[data-slot="${g.i}"]`); if (row) row.style.borderLeftColor = `var(--g${color + 1})`; });
    $('add-model').disabled = slots.length >= 4 || !data.profiles.some(p => E.arcs(size, p.nodes).some(a => a.every(n => !pack.used.includes(n))));
    $('plan-source-note').textContent = $('plan-source').value === 'checkout' ? E.sourceName(data.meta) + '. Commands pin this source.' : 'Commands fetch main when you run them.';
  }
  function renderNames() {
    $('plan-sparks').hidden = $('plan-order').value !== 'fill';
    $('plan-sparks').innerHTML = names.slice(0, size).map((n, i) => `<label for="plan-host-${i}">${i === 0 ? 'Node A' : 'Spark ' + i} · SSH name</label><input id="plan-host-${i}" data-rank="${i}" data-key="host" value="${esc(n.host)}"><label for="plan-ip-${i}">IP address (optional)</label><input id="plan-ip-${i}" data-rank="${i}" data-key="host_ip" value="${esc(n.host_ip)}">`).join('');
  }
  $('spark-count').innerHTML = Array.from({ length: 7 }, (_, i) => option(i + 2, `${i + 2} Sparks`, i + 2 === size)).join('');
  $('plan-image').innerHTML = data.images.map(row => option(row.name, row.option + (row.default ? ' (default)' : ''), row === image)).join('');
  $('spark-count').addEventListener('change', () => { size = Number($('spark-count').value); renderSlots(); renderNames(); renderOutput(); });
  $('setup-only').addEventListener('change', () => { setupOnly = $('setup-only').checked; renderSlots(); renderOutput(); });
  $('plan-image').addEventListener('change', () => { image = data.images.find(r => r.name === $('plan-image').value); slots.forEach(s => { s.checkpoint = null; s.entries = {}; }); renderSlots(); renderOutput(); });
  $('add-model').addEventListener('click', () => { const p = data.profiles.find(p => p.nodes === 2 && image.profiles[p.id]); if (p) slots.push(freshSlot(p.id)); renderSlots(); renderOutput(); $('model-' + (slots.length - 1)).focus(); });
  $('model-slots').addEventListener('click', e => {
    if (e.target.dataset.remove !== undefined) { const i = Number(e.target.dataset.remove); slots.splice(i, 1); renderSlots(); renderOutput(); $('model-' + Math.min(i, slots.length - 1)).focus(); }
  });
  $('model-slots').addEventListener('change', e => {
    const container = e.target.closest('[data-slot]'); if (!container) return;
    const i = Number(container.dataset.slot), slot = slots[i], field = e.target, focus = field.id;
    if (field.hasAttribute('data-model')) { slots[i] = freshSlot(field.value); renderSlots(); $(focus).focus(); }
    else if (field.hasAttribute('data-where')) slot.where = field.value;
    else if (field.hasAttribute('data-checkpoint')) { slot.checkpoint = field.value || null; slot.entries = {}; renderSlots(); $(focus).closest('details').open = true; $(focus).focus(); }
    renderOutput();
  });
  $('model-slots').addEventListener('input', e => {
    const container = e.target.closest('[data-slot]'); if (!container) return;
    const slot = slots[Number(container.dataset.slot)], field = e.target;
    if (field.dataset.setting) slot.entries[field.dataset.setting] = field.type === 'checkbox' ? field.checked : field.value;
    if (field.hasAttribute('data-endpoint')) slot.endpoint = field.value;
    renderOutput();
  });
  $('plan-sparks').addEventListener('input', e => { if (e.target.dataset.rank) { names[Number(e.target.dataset.rank)][e.target.dataset.key] = e.target.value; renderOutput(); } });
  ['plan-form', 'plan-source', 'plan-approval', 'plan-order'].forEach(id => $(id).addEventListener('change', () => { renderNames(); renderOutput(); }));
  $('plan-limit').addEventListener('input', renderOutput);
  // One model with the choices the planner starts with shares the short link `p=NAME` (E.planQuery).
  $('plan-share').addEventListener('click', async () => {
    const query = E.planQuery({ size, slots, image: image.name, setupOnly }, byId, data.images, aliases);
    const url = location.origin + location.pathname + '?' + query;
    try { await navigator.clipboard.writeText(url); $('plan-share').textContent = 'Copied builder link'; }
    catch (_) { $('plan-share').textContent = 'Copy link from address bar'; history.replaceState(null, '', '?' + query); }
  });
  if (query.has('profile') || query.get('mode') === 'compose') $('advanced').open = true;
  renderSlots(); renderNames(); renderOutput();
})();
