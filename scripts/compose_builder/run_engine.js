// Render verify.py's cases with the Compose builder engine.
// Usage: node run_engine.js DATA_JSON CASES_JSON > OUTPUTS_JSON
// A case with `archive` set also returns the deployment archive, base64-encoded.
const fs = require('fs');
const Engine = require('./engine.js');

(async () => {
  const data = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
  const cases = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
  const byId = Object.fromEntries(data.profiles.map(p => [p.id, p]));
  const meta = { repository: data.repository, tag: data.tag, commit: data.commit, ref: data.ref };
  const outputs = [];
  for (const c of cases) {
    const profile = byId[c.profile];
    const result = await Engine.render(profile, c.site, c.settings, c.checkpoint);
    if (result.ok && c.archive) {
      const checkpoint = Engine.checkpointOf(profile, c.checkpoint);
      const { filename, bytes } = Engine.archive(profile, checkpoint, c.site, c.settings, result, meta, new Date(2026, 0, 1));
      result.archive = { filename, base64: Buffer.from(bytes).toString('base64') };
    }
    outputs.push(result);
  }
  process.stdout.write(JSON.stringify(outputs));
})().catch(error => { console.error(error); process.exit(1); });
