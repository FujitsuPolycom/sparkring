// Compose builder engine: per-rank files from templates that runtime/common/compose.py
// rendered with a sentinel site (scripts/compose_builder/export.py). The engine validates a
// site as compose.site_settings does, substitutes the site's values for the sentinel tokens
// line by line with PyYAML's quoting rules, applies serving settings as
// runtime/common/serving.py does, and computes the deployment identity and manifest as
// compose.build does; scripts/compose_builder/verify.py checks that the results are equal.
// Inputs are limited to characters whose YAML and JSON encodings it reproduces exactly;
// anything else is refused rather than encoded differently.
const SparkRingEngine = (() => {
  'use strict';

  // ---- PyYAML 6.0 implicit resolvers (yaml/resolver.py) -------------------------------
  const RESOLVERS = [
    [/^(?:yes|Yes|YES|no|No|NO|true|True|TRUE|false|False|FALSE|on|On|ON|off|Off|OFF)$/, 'yYnNtTfFoO'],
    [/^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?|\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?|[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$/, '-+0123456789.'],
    [/^(?:[-+]?0b[0-1_]+|[-+]?0[0-7_]+|[-+]?(?:0|[1-9][0-9_]*)|[-+]?0x[0-9a-fA-F_]+|[-+]?[1-9][0-9_]*(?::[0-5]?[0-9])+)$/, '-+0123456789'],
    [/^(?:<<)$/, '<'],
    [/^(?:~|null|Null|NULL|)$/, '~nN'],
    [/^(?:[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]|[0-9][0-9][0-9][0-9]-[0-9][0-9]?-[0-9][0-9]?(?:[Tt]|[ \t]+)[0-9][0-9]?:[0-9][0-9]:[0-9][0-9](?:\.[0-9]*)?(?:[ \t]*(?:Z|[-+][0-9][0-9]?(?::[0-9][0-9])?))?)$/, '0123456789'],
    [/^(?:=)$/, '='],
  ];
  // Characters every substituted value is limited to: printable ASCII without space, quotes,
  // backslash, '#', '$' or flow/anchor indicators that would change PyYAML's scalar analysis.
  const SAFE = /^[A-Za-z0-9._\/+=@~:,{}"\-]*$/;

  function resolves(text) {
    if (text === '') return true;
    return RESOLVERS.some(([pattern, first]) => first.includes(text[0]) && pattern.test(text));
  }
  function plainAllowed(text) {
    if (!text || text.startsWith('---') || text.startsWith('...')) return false;
    if ('#,[]{}&*!|>\'"%@`'.includes(text[0])) return false;
    if ('-?:'.includes(text[0]) && text.length === 1) return false;
    if (/:$/.test(text) || /[\s]/.test(text)) return false;
    return true;
  }
  function yamlScalar(text) {
    if (!SAFE.test(text)) throw new Error('unsupported character in ' + JSON.stringify(text));
    return plainAllowed(text) && !resolves(text) ? text : "'" + text.replace(/'/g, "''") + "'";
  }
  function yamlUnquote(scalar) {
    if (scalar.startsWith("'")) {
      if (!scalar.endsWith("'")) throw new Error('unterminated scalar in template');
      return scalar.slice(1, -1).replace(/''/g, "'");
    }
    if (scalar.startsWith('"')) throw new Error('double-quoted scalar in template');
    return scalar;
  }

  // ---- Python json.dumps(sort_keys=True, indent=2) + "\n" (compose.encoded) -----------
  function asciiJson(text) {
    // Python's ensure_ascii escapes everything outside ' '..'~', DEL included.
    return text.replace(/[\u007f-￿]/g, ch => '\\u' + ch.charCodeAt(0).toString(16).padStart(4, '0'));
  }
  function sortKeys(value) {
    if (Array.isArray(value)) return value.map(sortKeys);
    if (value && typeof value === 'object') {
      const out = {};
      for (const key of Object.keys(value).sort()) out[key] = sortKeys(value[key]);
      return out;
    }
    return value;
  }
  function pyFloat(value) {
    if (!Number.isFinite(value)) throw new Error('non-finite number');
    return value;
  }
  function encoded(document) {
    const text = JSON.stringify(sortKeys(document), (k, v) => typeof v === 'number' && !Number.isInteger(v) ? pyFloat(v) : v, 2);
    return asciiJson(text) + '\n';
  }
  function jsonString(text) { return asciiJson(JSON.stringify(text)); }

  async function sha256(text) {
    const bytes = new TextEncoder().encode(text);
    const digest = await crypto.subtle.digest('SHA-256', bytes);
    return Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, '0')).join('');
  }

  // ---- Site validation (compose.site_settings, qwen_flash_next.site_inputs and container_spec,
  // qwen_mesh.validate_site_reference), in the generator's order ---------------------------
  const isInt = v => typeof v === 'number' && Number.isInteger(v);
  const IPV4 = /^(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])(?:\.(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])){3}$/;
  const CONTROL = /[\u0000-\u001f\u007f]/;

  function posixNormal(value) {
    // str(PurePosixPath(value)) == value for an absolute path without '//', '.' parts or a trailing '/'.
    return value.startsWith('/') && !value.startsWith('//') && !/\/\/|\/\.(\/|$)|.\/$/.test(value);
  }
  function parts(value) { return value.split('/').filter(Boolean); }
  function relative(a, b) {
    const pa = parts(a), pb = parts(b);
    return pb.length <= pa.length && pb.every((p, i) => pa[i] === p);
  }
  function linuxPath(value) {
    if (typeof value !== 'string') throw new Error('Host paths must be strings');
    if (!value.startsWith('/') || !posixNormal(value) || parts(value).includes('..') || value === '/'
        || /[\\,]/.test(value) || CONTROL.test(value)) {
      throw new Error('Host paths must be normalized absolute Linux paths');
    }
    return value;
  }
  function limited(value, what) {
    if (!/^[A-Za-z0-9._\/+=@~-]+$/.test(value)) {
      throw new Error(what + ' may use only letters, digits and . _ / + = @ ~ - in this builder: ' + value);
    }
  }

  function validateSite(site, nodes) {
    const keys = ['schema', 'name', 'master', 'ranks'];
    if (!site || typeof site !== 'object' || Array.isArray(site) || Object.keys(site).length !== 4 || !keys.every(k => k in site)) {
      throw new Error('Site requires only schema, name, master and ranks; secrets and overrides are not accepted');
    }
    if (site.schema !== 'sparkring-compose-site/v1') throw new Error('Unsupported Compose site schema');
    if (typeof site.name !== 'string' || !/^[a-z][a-z0-9-]{0,39}$/.test(site.name)) {
      throw new Error('Site name must be a lowercase deployment name, at most 40 characters');
    }
    if (!Array.isArray(site.ranks) || site.ranks.length !== nodes) throw new Error(`This TP${nodes} profile requires exactly ${nodes} hosts`);
    const rankKeys = ['rank', 'host', 'host_ip', 'interface', 'hcas', 'gid', 'model', 'cache', 'repository', 'deployment_root'];
    if (nodes === 4) rankKeys.push('fabric');
    const hosts = new Set(), addresses = new Set();
    site.ranks.forEach((rank, number) => {
      if (!rank || typeof rank !== 'object' || Object.keys(rank).length !== rankKeys.length || !rankKeys.every(k => k in rank)
          || !isInt(rank.rank) || rank.rank !== number) {
        throw new Error('Site ranks must be ordered from zero with the documented host fields');
      }
      const host = rank.host;
      if (typeof host !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9_.@-]*$/.test(host) || host === 'controller'
          || hosts.has(host) || addresses.has(rank.host_ip)) {
        throw new Error('Ranks require distinct SSH targets and IP addresses');
      }
      hosts.add(host);
      addresses.add(rank.host_ip);
      const paths = ['model', 'cache', 'repository', 'deployment_root'].map(k => linuxPath(rank[k]));
      for (let i = 0; i < paths.length; i++) {
        for (const other of paths.slice(i + 1)) {
          if (relative(paths[i], other) || relative(other, paths[i])) throw new Error('Model, cache, repository and deployment roots must be disjoint');
        }
      }
      if (typeof rank.host_ip !== 'string' || !IPV4.test(rank.host_ip)) {
        if (typeof rank.host_ip === 'string' && rank.host_ip.includes(':')) throw new Error('This builder accepts IPv4 addresses only: ' + rank.host_ip);
        throw new Error('host-ip must be a concrete IP address');
      }
      if (typeof site.master !== 'string' || (!IPV4.test(site.master) && !/^[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?$/.test(site.master))) {
        throw new Error('master must be a concrete IP address or hostname');
      }
      if (typeof rank.interface !== 'string' || !/^[A-Za-z0-9_][A-Za-z0-9_.-]{0,14}$/.test(rank.interface)) {
        throw new Error('interface must be one explicit network interface');
      }
      if (relative(rank.model, rank.cache) || relative(rank.cache, rank.model)) throw new Error('Model/cache paths must be disjoint');
      if (nodes === 4) {
        const f = rank.fabric;
        if (!f || typeof f !== 'object' || Object.keys(f).length !== 3 || !['site_path', 'site_sha256', 'plan_sha256'].every(k => k in f)) {
          throw new Error('Fabric reference requires only site_path, site_sha256 and plan_sha256');
        }
        const v = f.site_path;
        if (typeof v !== 'string' || !v.startsWith('/') || v === '/' || !posixNormal(v) || parts(v).includes('..') || v.includes('\\') || CONTROL.test(v)) {
          throw new Error('Fabric site_path must be a normalized absolute Linux file path');
        }
        if (['site_sha256', 'plan_sha256'].some(k => typeof f[k] !== 'string' || !/^[0-9a-f]{64}$/.test(f[k]))) {
          throw new Error('Fabric site and canonical plan require exact lowercase SHA-256 hashes');
        }
      }
    });
    if (site.master !== site.ranks[0].host_ip) throw new Error("Master must be the API rank's host_ip");
    site.ranks.forEach(rank => {
      const h = rank.hcas;
      if (!Array.isArray(h) || h.length !== nodes || new Set(h).size !== nodes || h.some(x => typeof x !== 'string' || !/^[A-Za-z0-9_]{1,64}$/.test(x))) {
        throw new Error("Select the profile's distinct HCA functions in cable order");
      }
      if (!isInt(rank.gid) || rank.gid < 0 || rank.gid > 255) throw new Error('GID index must be an integer from 0 to 255');
    });
    // Values the builder writes into YAML must keep PyYAML's encoding; see SAFE.
    site.ranks.forEach(rank => {
      for (const key of ['model', 'cache', 'repository', 'deployment_root']) limited(rank[key], key.replace('_', ' '));
      if (nodes === 4) limited(rank.fabric.site_path, 'Mesh site path');
    });
  }

  // Every site field validateSite would refuse, as {rank, key, message}: `rank` is null for the
  // deployment name, `key` a rank field or fabric.<field>. The page shows each message beside its
  // field; validateSite stays the authority and reports the generator's first refusal.
  const PATH_KEYS = ['model', 'cache', 'repository', 'deployment_root'];
  const PATH_LABELS = { model: 'model directory', cache: 'cache directory', repository: 'SparkRing checkout', deployment_root: 'deployment root' };
  function fieldProblems(site, nodes) {
    const problems = [];
    const add = (rank, key, message) => problems.push({ rank, key, message });
    if (typeof site.name !== 'string' || !/^[a-z][a-z0-9-]{0,39}$/.test(site.name)) {
      add(null, 'name', 'Start with a lowercase letter; use lowercase letters, digits and hyphens, at most 40 characters.');
    }
    const usable = value => {
      try { linuxPath(value); limited(value, ''); return true; } catch (_) { return false; }
    };
    (site.ranks || []).forEach((rank, i) => {
      const others = site.ranks.slice(0, i);
      if (typeof rank.host !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9_.@-]*$/.test(rank.host) || rank.host === 'controller') {
        add(i, 'host', 'An SSH host or user@host: letters, digits and . _ @ -.');
      } else if (others.some(r => r.host === rank.host)) {
        add(i, 'host', 'Spark ' + others.findIndex(r => r.host === rank.host) + ' has this host too.');
      }
      if (typeof rank.host_ip !== 'string' || !IPV4.test(rank.host_ip)) {
        add(i, 'host_ip', 'An IPv4 address, such as 192.0.2.10.');
      } else if (others.some(r => r.host_ip === rank.host_ip)) {
        add(i, 'host_ip', 'Spark ' + others.findIndex(r => r.host_ip === rank.host_ip) + ' has this address too.');
      }
      if (typeof rank.interface !== 'string' || !/^[A-Za-z0-9_][A-Za-z0-9_.-]{0,14}$/.test(rank.interface)) {
        add(i, 'interface', 'One network interface name, at most 15 characters.');
      }
      const h = rank.hcas;
      if (!Array.isArray(h) || h.length !== nodes || new Set(h).size !== nodes || h.some(x => typeof x !== 'string' || !/^[A-Za-z0-9_]{1,64}$/.test(x))) {
        add(i, 'hcas', `${nodes} distinct HCA names in cable order, separated by commas.`);
      }
      if (!isInt(rank.gid) || rank.gid < 0 || rank.gid > 255) add(i, 'gid', 'A whole number from 0 to 255.');
      PATH_KEYS.forEach((key, k) => {
        const value = rank[key];
        if (!usable(value)) {
          add(i, key, 'An absolute Linux path of letters, digits and . _ / + = @ ~ -, without a trailing /.');
          return;
        }
        const overlap = PATH_KEYS.slice(0, k).find(other => usable(rank[other]) && (relative(value, rank[other]) || relative(rank[other], value)));
        if (overlap) add(i, key, 'Must not be inside, or contain, the ' + PATH_LABELS[overlap] + '.');
      });
      if (nodes === 4) {
        const f = rank.fabric || {};
        if (!usable(f.site_path)) add(i, 'fabric.site_path', 'An absolute Linux file path of letters, digits and . _ / + = @ ~ -.');
        for (const key of ['site_sha256', 'plan_sha256']) {
          if (typeof f[key] !== 'string' || !/^[0-9a-f]{64}$/.test(f[key])) add(i, 'fabric.' + key, '64 lowercase hexadecimal characters.');
        }
      }
    });
    return problems;
  }

  // ---- Serving settings (runtime/common/serving.py) ----------------------------------
  function option(name) { return '--' + name.replace(/_/g, '-'); }
  function normalized(profile, values) {
    const rows = Object.fromEntries(profile.settings.map(r => [r.name, r]));
    const result = {};
    for (const name of Object.keys(values || {}).sort()) {
      const value = values[name];
      if (value === null || value === undefined) continue;
      const row = rows[name];
      if (!row) throw new Error('Unknown serving setting: ' + name);
      if (row.switch) {
        if (value !== true) throw new Error(option(name) + ' is a switch without a value');
        result[name] = true;
        continue;
      }
      if (!isInt(value) || value < row.minimum) throw new Error(`${option(name)} takes a whole number of at least ${row.minimum}`);
      result[name] = value;
    }
    return result;
  }
  // serving.apply refuses a value above its ceiling after the site has been validated.
  function checkCeilings(profile, settings) {
    for (const row of profile.settings) {
      const value = settings[row.name];
      if (value !== undefined && row.maximum !== null && row.maximum !== undefined && value > row.maximum) {
        throw new Error(`${option(row.name)} ${value} is more than ${row.maximum}, a tenth above the profile's ${row.profile}. A larger value can exhaust a Spark's memory while the model starts.`);
      }
    }
  }
  const FLAGS = {
    max_images: ['--limit-mm-per-prompt', 'image', 1], max_videos: ['--limit-mm-per-prompt', 'video', 1],
    context_length: ['--max-model-len', null, 1], max_concurrency: ['--max-num-seqs', null, 1],
    kv_cache_gib: ['--kv-cache-memory-bytes', null, 2 ** 30],
  };
  function describe(profile, settings) {
    const rows = Object.fromEntries(profile.settings.map(r => [r.name, r]));
    return Object.keys(settings).sort().map(name => rows[name].switch
      ? option(name) + ' (profile: off)'
      : `${option(name)} ${settings[name]} (profile: ${rows[name].profile})`);
  }
  function warnings(profile, settings) {
    const row = profile.settings.find(r => r.name === 'kv_cache_gib');
    if (!row || settings.kv_cache_gib === undefined || settings.kv_cache_gib <= row.profile) return [];
    return [`--kv-cache-gib ${settings.kv_cache_gib} is above the profile's ${row.profile} GiB, leaving each Spark ${settings.kv_cache_gib - row.profile} GiB less for images and long requests. This value has not been validated as stable.`];
  }
  // New command value for a vLLM flag, from its current template value.
  function servingValue(name, current, value) {
    const [, key, scale] = FLAGS[name];
    if (key === null) return String(value * scale);
    const limits = JSON.parse(current);
    if (!limits || typeof limits !== 'object' || !(key in limits)) throw new Error(`${option(name)} does not apply to this profile`);
    limits[key] = value;
    return JSON.stringify(limits);
  }

  // ---- Template substitution ---------------------------------------------------------
  function replacements(site, n, identity, sentinelIdentity) {
    const row = site.ranks[n];
    const pairs = [['zqsite', site.name], ['198.18.0.77', site.master], [`198.18.${n}.77`, row.host_ip],
      [`zqif${n}`, row.interface], [`/zqmodel${n}/m`, row.model], [`/zqcache${n}/c`, row.cache],
      [`/zqrepo${n}/r`, row.repository], [sentinelIdentity, identity]];
    row.hcas.forEach((hca, i) => pairs.push([`zqhca${n}${'abcd'[i]}`, hca]));
    return pairs;
  }
  function substitute(text, pairs) {
    let out = text, hit = false;
    for (const [token, value] of pairs) {
      if (out.includes(token)) { out = out.split(token).join(value); hit = true; }
    }
    return [out, hit];
  }
  const GID_KEYS = ['NCCL_IB_GID_INDEX', 'B12X_ROCE_GID_INDEX'];

  function composeText(template, pairs, gidSentinel, gid, edits) {
    const lines = template.split('\n');
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const m = line.match(/^(\s*(?:- |[A-Za-z0-9_.-]+: ))(.+)$/);
      if (!m) continue;
      const key = m[1].trim().replace(/:$/, '');
      if (GID_KEYS.includes(key)) {
        if (yamlUnquote(m[2]) !== gidSentinel) throw new Error('template GID line changed');
        lines[i] = m[1] + yamlScalar(String(gid));
        continue;
      }
      const [value, hit] = substitute(yamlUnquote(m[2]), pairs);
      if (hit) lines[i] = m[1] + yamlScalar(value);
    }
    for (const [flag, name, value] of edits) {
      const at = lines.findIndex(l => /^\s*- /.test(l) && l.trim() === '- ' + flag);
      if (at < 0) throw new Error(`${option(name)} does not apply to this profile: it sets no ${flag}`);
      const m = lines[at + 1].match(/^(\s*- )(.+)$/);
      lines[at + 1] = m[1] + yamlScalar(servingValue(name, yamlUnquote(m[2]), value));
    }
    return lines.join('\n');
  }

  function containerText(template, pairs, gidSentinel, gid, edits) {
    const lines = template.split('\n');
    const literal = /"((?:[^"\\]|\\.)*)"/g;
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const key = (line.match(/^\s*"([^"]+)":/) || [])[1];
      if (GID_KEYS.includes(key)) {
        const m = line.match(/^(\s*"[^"]+": )"([^"]*)"(,?)$/);
        if (!m || m[2] !== gidSentinel) throw new Error('template GID line changed');
        lines[i] = m[1] + jsonString(String(gid)) + m[3];
        continue;
      }
      let hit = false;
      lines[i] = line.replace(literal, (whole, body, offset) => {
        if (key && offset === line.indexOf('"')) return whole;
        const [value, changed] = substitute(JSON.parse(whole), pairs);
        if (!changed) return whole;
        hit = true;
        return jsonString(value);
      });
      if (!hit) lines[i] = line;
    }
    for (const [flag, name, value] of edits) {
      const at = lines.findIndex(l => l.trim() === JSON.stringify(flag) + ',');
      if (at < 0) throw new Error(`${option(name)} does not apply to this profile: it sets no ${flag}`);
      const m = lines[at + 1].match(/^(\s*)("(?:[^"\\]|\\.)*")(,?)$/);
      lines[at + 1] = m[1] + jsonString(servingValue(name, JSON.parse(m[2]), value)) + m[3];
    }
    return lines.join('\n');
  }

  // ---- site.yaml: yaml.safe_dump(site, sort_keys=False) ----------------------------
  function yamlValue(value) { return isInt(value) ? String(value) : yamlScalar(String(value)); }
  function siteYaml(site) {
    const out = [];
    for (const [key, value] of Object.entries(site)) {
      if (key !== 'ranks') { out.push(`${key}: ${yamlValue(value)}`); continue; }
      out.push('ranks:');
      for (const rank of value) {
        let first = true;
        for (const [k, v] of Object.entries(rank)) {
          const lead = first ? '- ' : '  ';
          first = false;
          if (Array.isArray(v)) { out.push(`${lead}${k}:`); for (const item of v) out.push(`  - ${yamlValue(item)}`); }
          else if (v && typeof v === 'object') { out.push(`${lead}${k}:`); for (const [fk, fv] of Object.entries(v)) out.push(`    ${fk}: ${yamlValue(fv)}`); }
          else out.push(`${lead}${k}: ${yamlValue(v)}`);
        }
      }
    }
    return out.join('\n') + '\n';
  }

  // ---- Build -------------------------------------------------------------------------
  // The profile's checkpoint entry that `name` selects: the default for none, a listed name or an alias.
  function checkpointOf(profile, name) {
    const list = profile.checkpoints;
    if (name === null || name === undefined) return list[0];
    const found = list.find(c => c.name === name || c.aliases.includes(name));
    if (found) return found;
    if (list.length < 2) throw new Error('This profile offers no checkpoint choice');
    throw new Error('Select a checkpoint the profile lists: ' + list.map(c => c.name).join(', '));
  }

  // Settings and checkpoint without a site, as `sparkring install` checks them before it plans.
  function servingCheck(profile, requested, checkpointName) {
    try {
      const checkpoint = checkpointOf(profile, checkpointName);
      const settings = normalized(checkpoint, requested);
      checkCeilings(checkpoint, settings);
      return { ok: true, checkpoint, settings, lines: describe(checkpoint, settings), warnings: warnings(checkpoint, settings) };
    } catch (error) {
      return { ok: false, error: error.message || String(error) };
    }
  }

  // `checkpointName` selects the checkpoint as `sparkring install --checkpoint` does; the
  // serving settings' profile values and ceilings are that checkpoint's.
  async function render(profile, site, requested, checkpointName) {
    try {
      const checkpoint = checkpointOf(profile, checkpointName);
      const settings = normalized(checkpoint, requested);
      validateSite(site, profile.nodes);
      checkCeilings(checkpoint, settings);
      const variant = checkpoint.variants[settings.save_cpu ? 'on' : 'off'];
      const numeric = Object.keys(settings).filter(k => k !== 'save_cpu').sort();
      const identityInputs = { profile: profile.id, site, inputs: profile.identity_inventory, ...profile.options };
      if (Object.keys(settings).length) identityInputs.serving = settings;
      if (!checkpoint.default) identityInputs.checkpoint = checkpoint.name;
      const identity = await sha256(encoded(identityInputs));
      const files = {};
      for (let n = 0; n < site.ranks.length; n++) {
        const pairs = replacements(site, n, identity, variant.identity);
        const gidSentinel = String(201 + n);
        const edits = numeric.map(name => [FLAGS[name][0], name, settings[name]]);
        files[`rank${n}/compose.yaml`] = composeText(variant.ranks[n].compose, pairs, gidSentinel, site.ranks[n].gid, edits);
        files[`rank${n}/container.json`] = containerText(variant.ranks[n].container, pairs, gidSentinel, site.ranks[n].gid, edits);
      }
      const digests = {};
      for (const [name, text] of Object.entries(files)) digests[name] = await sha256(text);
      const manifest = {
        schema: 'sparkring-compose-deployment/v1', id: identity, profile: profile.id, site, inputs: profile.inputs,
        image: profile.image, image_id: profile.image_id, files: digests,
        qualification: 'Generated configuration only; Compose serving is not qualified.',
      };
      const lines = describe(checkpoint, settings), warn = warnings(checkpoint, settings);
      if (Object.keys(settings).length) manifest.serving = settings;
      if (!checkpoint.default) manifest.checkpoint = checkpoint.name;
      Object.assign(manifest, profile.options);
      return { ok: true, id: identity, files, deployment: encoded(manifest), site_yaml: siteYaml(site), serving_lines: lines,
        warnings: warn, checkpoint: checkpoint.name };
    } catch (error) {
      return { ok: false, error: error.message || String(error) };
    }
  }

  // ---- Deployment archive ------------------------------------------------------------
  // A stored (uncompressed) ZIP whose top directory is the deployment as
  // `sparkring compose render` writes it, with the site file and a README beside it.
  // Unix modes match render's: 0700 directories, 0600 files.
  const CRC = (() => {
    const table = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      table[n] = c >>> 0;
    }
    return table;
  })();
  function crc32(bytes) {
    let c = 0xffffffff;
    for (let i = 0; i < bytes.length; i++) c = CRC[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
  }
  function zip(entries, when) {
    const enc = new TextEncoder();
    const dosTime = (when.getHours() << 11) | (when.getMinutes() << 5) | (when.getSeconds() >> 1);
    const dosDate = ((when.getFullYear() - 1980) << 9) | ((when.getMonth() + 1) << 5) | when.getDate();
    const locals = [], centrals = [];
    let offset = 0;
    for (const entry of entries) {
      const name = enc.encode(entry.name);
      const data = entry.dir ? new Uint8Array(0) : enc.encode(entry.text);
      const crc = entry.dir ? 0 : crc32(data);
      const mode = entry.dir ? 0o40700 : 0o100600;
      const local = new DataView(new ArrayBuffer(30));
      local.setUint32(0, 0x04034b50, true); local.setUint16(4, 20, true); local.setUint16(6, 0, true);
      local.setUint16(8, 0, true); local.setUint16(10, dosTime, true); local.setUint16(12, dosDate, true);
      local.setUint32(14, crc, true); local.setUint32(18, data.length, true); local.setUint32(22, data.length, true);
      local.setUint16(26, name.length, true); local.setUint16(28, 0, true);
      const central = new DataView(new ArrayBuffer(46));
      central.setUint32(0, 0x02014b50, true); central.setUint16(4, (3 << 8) | 20, true); central.setUint16(6, 20, true);
      central.setUint16(8, 0, true); central.setUint16(10, 0, true); central.setUint16(12, dosTime, true);
      central.setUint16(14, dosDate, true); central.setUint32(16, crc, true); central.setUint32(20, data.length, true);
      central.setUint32(24, data.length, true); central.setUint16(28, name.length, true); central.setUint16(30, 0, true);
      central.setUint16(32, 0, true); central.setUint16(34, 0, true); central.setUint16(36, 0, true);
      central.setUint32(38, ((mode << 16) | (entry.dir ? 0x10 : 0)) >>> 0, true); central.setUint32(42, offset, true);
      locals.push(new Uint8Array(local.buffer), name, data);
      centrals.push(new Uint8Array(central.buffer), name);
      offset += 30 + name.length + data.length;
    }
    const size = centrals.reduce((n, part) => n + part.length, 0);
    const end = new DataView(new ArrayBuffer(22));
    end.setUint32(0, 0x06054b50, true); end.setUint16(8, entries.length, true); end.setUint16(10, entries.length, true);
    end.setUint32(12, size, true); end.setUint32(16, offset, true);
    const parts = [...locals, ...centrals, new Uint8Array(end.buffer)];
    const out = new Uint8Array(parts.reduce((n, part) => n + part.length, 0));
    let at = 0;
    for (const part of parts) { out.set(part, at); at += part.length; }
    return out;
  }

  // ---- Command lines -------------------------------------------------------------------
  // `meta` is the data's source: {repository, tag, commit, ref, since_tag, commits_since};
  // `ref` is the release tag, or the commit when the page was built from an untagged
  // checkout, which is named by the release before it and the commits since.
  function sourceName(meta) {
    if (meta.tag) return 'SparkRing ' + meta.tag;
    if (meta.since_tag) return `SparkRing ${meta.since_tag} + ${meta.commits_since} commit${meta.commits_since === 1 ? '' : 's'}`;
    return 'SparkRing at ' + meta.commit.slice(0, 12);
  }
  // A profile rendered on a non-default installer image carries the `--image` value that selects it.
  function selectionWords(profile, checkpoint, settings) {
    const words = [];
    if (profile.image_option) words.push('--image', profile.image_option);
    if (!checkpoint.default) words.push('--checkpoint', checkpoint.name);
    for (const name of Object.keys(settings).sort()) {
      words.push(...(settings[name] === true ? [option(name)] : [option(name), String(settings[name])]));
    }
    return words;
  }
  // `form` "script" is the one-command installer for a Spark without SparkRing (install.sh,
  // which passes every other option to `sparkring install`); "installed" runs the installed
  // package. `pin` fetches install.sh from meta.ref and builds that source instead of main.
  // `approval` is "ask" (no flag), "plan" or "yes". `downloadLimit` uses the syntax of
  // runtime/host/settings.py download_limit.
  function installCommand(profile, checkpoint, settings, meta, opts) {
    const words = ['--profile', profile.id, ...selectionWords(profile, checkpoint, settings)];
    if (opts.downloadLimit) words.push('--download-limit', opts.downloadLimit);
    if (opts.approval === 'plan') words.push('--plan');
    if (opts.approval === 'yes') words.push('--yes');
    if (opts.form === 'installed') return 'sudo sparkring install ' + words.join(' ');
    const ref = opts.pin ? meta.ref : 'main';
    return `curl -fsSL https://raw.githubusercontent.com/${meta.repository}/${ref}/install.sh | bash -s -- `
      + (opts.pin ? `--ref ${ref} ` : '') + words.join(' ');
  }
  // The `sparkring compose render` command that writes the same deployment.
  function renderCommand(profile, checkpoint, settings, site) {
    return ['sparkring compose render', profile.id, '--site', site.name + '.site.yaml', '--output', site.name,
      ...selectionWords(profile, checkpoint, settings)].join(' ');
  }
  function validDownloadLimit(text) {
    if (!text) return true;
    if (text.trim().toLowerCase() === 'none') return true;
    const m = text.trim().match(/^(\d+(?:\.\d+)?)([mg])bit$/i);
    return Boolean(m) && Number(m[1]) * (m[2].toLowerCase() === 'm' ? 1e6 : 1e9) >= 1e6;
  }
  // Where a derived checkpoint's directory is: `sudo sparkring install --checkpoint` writes it.
  function derivedDirectory(checkpoint) {
    return `/srv/sparkring/<cluster>/checkpoints/${checkpoint.model_repository.replace('/', '--')}/${checkpoint.model_revision}`;
  }

  function readme(profile, checkpoint, site, settings, output, meta) {
    const ranks = site.ranks.length;
    const changed = output.serving_lines.length > 0;
    const model = checkpoint.derived
      ? ['     the complete checkpoint in its model directory. No repository publishes this derived checkpoint:',
        `       sudo sparkring install --profile ${profile.id} --checkpoint ${checkpoint.name}`,
        `     writes it on every Spark into ${derivedDirectory(checkpoint)}.`]
      : ['     the complete checkpoint in its model directory, every shard checked, for example:',
        `       hf download ${checkpoint.model_repository} --revision ${checkpoint.model_revision} --local-dir <model directory>`];
    const lines = [
      `SparkRing Compose deployment "${site.name}": ${profile.id}`,
      `${profile.title}. ${sourceName(meta)} (${meta.commit.slice(0, 12)}).`,
      'PRIVATE: contains this site\'s addresses and storage paths.',
      '',
      'This directory is what this command writes from SparkRing\'s source above and the site file beside it:',
      '  ' + renderCommand(profile, checkpoint, settings, site),
      `Deployment ID ${output.id}`,
      '',
      'Ranks',
      ...site.ranks.map((r, i) => `  rank${i}  ${r.host}  ${r.host_ip}` + (i === 0 ? `  API http://${site.master}:${checkpoint.port}/v1, model ${checkpoint.served_model_name}` : '')),
      '',
      'Checkpoint',
      `  ${checkpoint.name || 'the profile\'s model'}${checkpoint.default ? ' (default)' : ''}: ${checkpoint.model_repository} at revision ${checkpoint.model_revision}`,
      ...checkpoint.changes.map(c => '  ' + c),
      '',
      'Serving settings',
      ...(changed ? output.serving_lines.map(l => '  ' + l) : ["  the profile's values"]),
      ...output.warnings.map(w => '  Warning: ' + w),
      '',
      'Image',
      `  ${profile.image}`,
      `  image ID ${profile.image_id} (${profile.image_release})`,
      '',
      'Start',
      '  1. Copy rankN/compose.yaml to the Spark of rank N.',
      `  2. On every Spark: docker pull ${profile.image}`,
      `     docker image inspect --format '{{.Id}}' ${profile.image} must print ${profile.image_id}.`,
      '     The files use pull_policy: never.',
      `  3. Every Spark needs the same SparkRing source at its repository path (Compose reads`,
      '     runtime/common/loader-seccomp.json from it),',
      ...model,
      `     Each fabric address's RoCE v2 GID must be at index ${site.ranks[0].gid}` + (ranks === 4 ? ', and the prepared mesh fabric in place.' : '.'),
      `  4. Start rank ${ranks === 2 ? '1' : '1 to ' + (ranks - 1)} first, then rank 0: docker compose -f rankN/compose.yaml up -d`,
      `  5. curl http://${site.master}:${checkpoint.port}/v1/models lists ${checkpoint.served_model_name} when the model serves.`,
      '     The first start on this image compiles and tunes kernels, about 10 minutes.',
      '',
      '`sparkring compose check --deployment <this directory>` checks these files against the source.',
      'The containers do not restart on their own (restart: no). Generated configuration only; Compose serving is not qualified.',
    ];
    if (profile.installable) {
      lines.push('', 'Without Compose, `sparkring install` sets up the Sparks, downloads the image and checkpoint and',
        'starts the same model. On Node A:',
        '  ' + installCommand(profile, checkpoint, settings, meta, { form: 'script', pin: true, approval: 'ask' }));
    }
    lines.push('');
    return lines.join('\n');
  }

  function archive(profile, checkpoint, site, settings, output, meta, when = new Date()) {
    const root = site.name + '/';
    const entries = [{ name: root, dir: true }, { name: root + 'deployment.json', text: output.deployment }];
    for (let n = 0; n < site.ranks.length; n++) {
      entries.push({ name: `${root}rank${n}/`, dir: true });
      entries.push({ name: `${root}rank${n}/compose.yaml`, text: output.files[`rank${n}/compose.yaml`] });
      entries.push({ name: `${root}rank${n}/container.json`, text: output.files[`rank${n}/container.json`] });
    }
    entries.push({ name: `${root}${site.name}.site.yaml`, text: output.site_yaml });
    entries.push({ name: root + 'README.txt', text: readme(profile, checkpoint, site, settings, output, meta) });
    return { filename: site.name + '.zip', bytes: zip(entries, when), names: entries.filter(e => !e.dir).map(e => e.name) };
  }

  return { render, archive, installCommand, renderCommand, derivedDirectory, sourceName, validDownloadLimit, servingCheck,
    checkpointOf, option, yamlScalar, resolves, encoded, siteYaml, validateSite, fieldProblems };
})();
if (typeof module !== 'undefined') module.exports = SparkRingEngine;
