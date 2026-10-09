// Install Builder engine: per-rank files from templates that runtime/common/compose.py
// rendered with a sentinel site (scripts/compose_builder/export.py). The engine validates a
// site as compose.site_settings does, substitutes the site's values for the sentinel tokens
// line by line with PyYAML's quoting rules, applies serving settings as
// runtime/common/serving.py does, and computes the deployment identity and manifest as
// compose.build does; scripts/compose_builder/verify.py checks that the results are equal.
// Inputs are limited to characters whose YAML and JSON encodings it reproduces exactly;
// anything else is refused rather than encoded differently. It also writes the page's
// `sparkring install` commands, the command pack of a layout and the page's shareable links.
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

  // ---- Site validation (compose.site_settings, toolchain_profiles.site_inputs and container_spec,
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

  // The problems with the Spark names and addresses that name the Sparks in a command pack
  // (sparkNames), as {rank, key, message} with `key` "host" or "host_ip": each Spark needs a
  // name or an address, a name is a host or user@host as a site's, an address is IPv4, and no
  // two Sparks share either.
  function sparkProblems(ranks) {
    const problems = [];
    const add = (rank, key, message) => problems.push({ rank, key, message });
    const text = value => String(value ?? '').trim();
    const role = i => i === 0 ? 'Node A' : 'Spark ' + i;
    ranks.forEach((rank, i) => {
      const others = ranks.slice(0, i), host = text(rank.host), address = text(rank.host_ip);
      if (!host && !address) {
        add(i, 'host', 'Enter a name or an address.');
        return;
      }
      const sameHost = others.findIndex(r => text(r.host) === host);
      if (host && (!/^[A-Za-z0-9][A-Za-z0-9_.@-]*$/.test(host) || host === 'controller')) add(i, 'host', 'A name or user@name: letters, digits and . _ @ -.');
      else if (host && sameHost >= 0) add(i, 'host', role(sameHost) + ' has this name too.');
      const sameAddress = others.findIndex(r => text(r.host_ip) === address);
      if (address && !IPV4.test(address)) add(i, 'host_ip', 'An IPv4 address, such as 192.0.2.10.');
      else if (address && sameAddress >= 0) add(i, 'host_ip', role(sameAddress) + ' has this address too.');
    });
    return problems;
  }

  // ---- Serving settings (runtime/common/serving.py) ----------------------------------
  function option(name) { return '--' + name.replace(/_/g, '-'); }
  // serving.listenable: a dotted IPv4 address outside 0.0.0.0/8 and below 224.0.0.0.
  function listenable(value) {
    if (typeof value !== 'string' || !IPV4.test(value)) return false;
    const first = Number(value.split('.')[0]);
    return first > 0 && first < 224;
  }
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
      if (row.address) {
        if (!listenable(value)) {
          throw new Error(`${option(name)} takes one IPv4 address of the Spark that serves the API, such as 192.0.2.10; without it the API listens on every address`);
        }
        result[name] = value;
        continue;
      }
      const largest = row.largest ?? null;
      if (!isInt(value) || value < row.minimum || (largest !== null && value > largest)) {
        throw new Error(`${option(name)} takes a whole number ` + (largest !== null ? `from ${row.minimum} to ${largest}` : `of at least ${row.minimum}`));
      }
      result[name] = value;
    }
    return result;
  }
  // serving.apply, after the site has been validated and in setting-name order, refuses a port
  // the API may not take (serving.reserved_ports) and a value above its ceiling.
  function checkCeilings(profile, settings) {
    const rows = Object.fromEntries(profile.settings.map(r => [r.name, r]));
    for (const name of Object.keys(settings).sort()) {
      const row = rows[name], value = settings[name];
      if (!row || row.switch) continue;
      for (const [port, reason] of row.reserved || []) {
        if (value === port) throw new Error(`${option(name)} ${port} is ${reason}. Choose another port.`);
      }
      if (row.maximum !== null && row.maximum !== undefined && value > row.maximum) {
        throw new Error(`${option(name)} ${value} is more than ${row.maximum}, a tenth above the profile's ${row.profile}. A larger value can exhaust a Spark's memory while the model starts.`);
      }
    }
  }
  // serving.check_image: a switch that needs an image capability (row.needs) the profile's image
  // lacks (profile.image_capabilities) is refused, naming the image's release.
  const offered = (profile, row) => !row.needs || (profile.image_capabilities || []).includes(row.needs);
  function checkImage(profile, checkpoint, settings) {
    const rows = Object.fromEntries(checkpoint.settings.map(r => [r.name, r]));
    for (const name of Object.keys(settings).sort()) {
      const row = rows[name];
      if (row && row.switch && !offered(profile, row)) {
        throw new Error(`${option(name)} needs an image whose vLLM reads ${row.flag}, and ${profile.image_release} does not. `
          + `Leave out ${option(name)} or choose another image.`);
      }
    }
  }
  const FLAGS = {
    max_images: ['--limit-mm-per-prompt', 'image', 1], max_videos: ['--limit-mm-per-prompt', 'video', 1],
    context_length: ['--max-model-len', null, 1], max_concurrency: ['--max-num-seqs', null, 1],
    kv_cache_gib: ['--kv-cache-memory-bytes', null, 2 ** 30], api_port: ['--port', null, 1], api_bind: ['--host', null, 1],
  };
  // Settings that only the API rank's command takes (serving.API_RANK).
  const API_RANK = ['api_bind'];
  // The API rank's health check asks the API at 127.0.0.1 on the profile's port; with the
  // endpoint's port or listen address it asks there instead (serving.container). Returns the
  // [before, after] text of its URL, or null without either setting.
  function healthEdit(checkpoint, settings) {
    if (settings.api_port === undefined && settings.api_bind === undefined) return null;
    const port = (checkpoint.settings.find(r => r.name === 'api_port') || {}).profile;
    return [`//127.0.0.1:${port}/`, `//${settings.api_bind ?? '127.0.0.1'}:${settings.api_port ?? port}/`];
  }
  const HEALTH = '--api-port and --api-bind do not apply to this profile: its health check does not ask the API at 127.0.0.1 on its --port';
  function editHealth(lines, health) {
    if (!health) return;
    const at = lines.findIndex(line => line.includes(health[0]));
    if (at < 0) throw new Error(HEALTH);
    lines[at] = lines[at].split(health[0]).join(health[1]);
  }

  // ---- API endpoint --------------------------------------------------------------------
  // runtime/host/api_endpoint.shown_address: the address `--api-address` names, a host name or an
  // IP address without a scheme, port or path. An IPv4 address or a host name is returned as
  // given and an IPv6 address without brackets; the installer writes an IPv6 address in its
  // shortest form.
  function apiAddress(value) {
    const text = String(value ?? '').trim(), bare = text.replace(/^\[|\]$/g, '');
    if (IPV4.test(bare)) return bare;
    if (bare.includes(':') && /^[0-9A-Fa-f:.]+$/.test(bare) && (bare.match(/::/g) || []).length <= 1) return bare.toLowerCase();
    const labels = text.replace(/\.+$/, '').split('.');
    if (!text || text.length > 253 || !labels.every(label => /^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$/.test(label))) {
      throw new Error(`--api-address takes a host name or an IP address, such as llm.example.net or 192.0.2.10, without http:// or a port: ${value}`);
    }
    return text;
  }
  const urlHost = host => host.includes(':') ? `[${host}]` : host;
  // The URL of a model's API: at `address` (the shown address), else the listen address, else
  // `host`, the address the API answers on without either; at the endpoint's port, else the
  // checkpoint's.
  function apiUrl(checkpoint, settings, host, address) {
    return `http://${urlHost(address || settings.api_bind || host)}:${settings.api_port ?? checkpoint.port}/v1`;
  }
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
    if (typeof value === 'string') return value;
    if (key === null) return String(value * scale);
    const limits = JSON.parse(current);
    if (!limits || typeof limits !== 'object' || !(key in limits)) throw new Error(`${option(name)} does not apply to this profile`);
    limits[key] = value;
    return JSON.stringify(limits);
  }

  // ---- Field entries ---------------------------------------------------------------------
  // The page keeps what each field holds, its entry, apart from the settings the commands and
  // files take. An entry is the text typed, or a value from a link or saved choices; the readers
  // here turn it into a value, or into the problem `sparkring install` would refuse it with. A
  // field with a problem gives no value, and the page offers no command or file until it is
  // fixed, so a refused value never falls back to the profile's silently.
  const wholeNumber = (row, n) => row.largest !== undefined && row.largest !== null ? String(n) : n.toLocaleString('en-US');
  // A serving setting's entry, against `row`, one of a checkpoint's settings (export._setting_rows):
  // {value, problem}, value undefined for an empty field, which keeps the profile's value.
  function readSetting(row, entry) {
    const text = String(entry ?? '').trim();
    if (text === '') return { value: undefined, problem: '' };
    if (row.address) return listenable(text) ? { value: text, problem: '' } : { problem: 'An IPv4 address of the Spark, such as 192.0.2.10.' };
    if (!/^\d+$/.test(text)) return { problem: 'Enter a whole number.' };
    const n = Number(text), largest = row.largest ?? null, maximum = row.maximum ?? null;
    if (!Number.isSafeInteger(n)) return { problem: 'Enter a smaller number.' };
    if (largest !== null && (n < row.minimum || n > largest)) return { problem: `From ${wholeNumber(row, row.minimum)} to ${wholeNumber(row, largest)}.` };
    if (n < row.minimum) return { problem: 'At least ' + wholeNumber(row, row.minimum) + '.' };
    if (maximum !== null && n > maximum) return { problem: 'At most ' + wholeNumber(row, maximum) + '.' };
    const taken = (row.reserved || []).find(([port]) => port === n);
    if (taken) return { problem: `This is ${taken[1]}.` };
    return { value: n, problem: '' };
  }
  // The entry of the address shown for a model (--api-address): {value, problem}, value null when empty.
  function readAddress(entry) {
    const text = String(entry ?? '').trim();
    if (!text) return { value: null, problem: '' };
    try { return { value: apiAddress(text), problem: '' }; } catch (_) { return { problem: 'A name or an address, without http:// or a port.' }; }
  }
  // The download limit's entry: {value, problem}, value '' when empty.
  function readDownloadLimit(entry) {
    const text = String(entry ?? '').trim();
    return validDownloadLimit(text) ? { value: text, problem: '' } : { problem: 'Use none, or a rate of at least 1Mbit, such as 850Mbit or 2Gbit.' };
  }
  // A selection's entries: `entries` holds each serving field's entry by setting name, `endpoint`
  // is the API endpoint choice {mode, address}. The endpoint's fields count only when its mode is
  // "set"; an entry for a setting the checkpoint does not have, or for a switch the profile's
  // image cannot apply, which the page does not offer, is left out. Returns {settings,
  // address, problems}: the settings and shown address the commands take, and [{field, message}]
  // for every field with a problem, `field` being a setting name or "api_address".
  function readSelection(profile, checkpoint, entries, endpoint) {
    const set = (endpoint || {}).mode === 'set', settings = {}, problems = [];
    for (const row of checkpoint.settings) {
      const entry = (entries || {})[row.name];
      if (entry === undefined || entry === null || (row.endpoint && !set)) continue;
      if (row.switch) {
        if (entry === true && offered(profile, row)) settings[row.name] = true;
        continue;
      }
      const read = readSetting(row, entry);
      if (read.problem) problems.push({ field: row.name, message: read.problem });
      else if (read.value !== undefined) settings[row.name] = read.value;
    }
    const shown = set ? readAddress((endpoint || {}).address) : { value: null, problem: '' };
    if (shown.problem) problems.push({ field: 'api_address', message: shown.problem });
    return { settings, address: shown.value ?? null, problems };
  }

  // ---- KV cache token estimate ---------------------------------------------------------
  // How many tokens a checkpoint's KV cache holds at the chosen size, scaled from the
  // engine-reported pool its data carries (checkpoint.capacity, export.kv_measurement):
  // tokens = measured tokens x chosen KV bytes per Spark / measured KV bytes per Spark, rounded to
  // two significant figures, and full-context requests = tokens / the context window, to one
  // decimal: how many requests of the whole window the cache holds by size, not a number of
  // requests that was served at once. `settings` are servingCheck's; a setting left out takes the
  // checkpoint's value. Returns null for a checkpoint without a KV cache setting; `measured` is
  // false without a measurement, which the page reports instead of guessing. A measurement names
  // its checkpoint and KV size, and its record (`source`, a repository path), `conditions` and
  // `kv_evidence` when its data does.
  const GIB = 2 ** 30;
  function twoFigures(value) {
    if (!(value > 0)) return 0;
    const step = 10 ** (Math.floor(Math.log10(value)) - 1);
    return Math.round(value / step) * step;
  }
  function kvEstimate(checkpoint, settings) {
    const rows = Object.fromEntries(checkpoint.settings.map(r => [r.name, r]));
    if (!rows.kv_cache_gib) return null;
    const gib = settings.kv_cache_gib ?? rows.kv_cache_gib.profile, measured = checkpoint.capacity;
    if (!measured) return { gib, measured: false };
    const tokens = measured.tokens * gib * GIB / measured.kv_bytes_per_rank;
    const context = settings.context_length ?? (rows.context_length || {}).profile;
    return { gib, measured: true, tokens: twoFigures(tokens), requests: context ? Math.round(tokens / context * 10) / 10 : null,
      measured_gib: measured.kv_bytes_per_rank / GIB, measured_checkpoint: measured.checkpoint,
      source: measured.source ?? null, conditions: measured.conditions ?? null, kv_evidence: measured.kv_evidence ?? null };
  }
  // The estimate in words, or null without a KV cache setting: {gib, line, basis, source,
  // conditions}. `basis` names the measurement it is scaled from ('' without one), and `source`
  // and `conditions` are that measurement's.
  function kvText(checkpoint, settings) {
    const kv = kvEstimate(checkpoint, settings);
    if (!kv) return null;
    if (!kv.measured) return { gib: kv.gib, line: 'Not measured for this model', basis: '', source: null, conditions: null };
    const n = kv.tokens;
    const tokens = 'About ' + (n >= 1e6 ? (n / 1e6).toLocaleString('en-US', { maximumFractionDigits: 1 }) + ' million'
      : n.toLocaleString('en-US')) + ' tokens';
    const requests = kv.requests === null ? ''
      : ` · room for about ${kv.requests.toFixed(1)} full-length requests by size; not a tested concurrency`;
    const gib = kv.measured_gib.toLocaleString('en-US', { maximumFractionDigits: 2 });
    return { gib: kv.gib, line: tokens + requests, basis: `Estimated from ${kv.measured_checkpoint || 'a measurement'} at ${gib} GiB`,
      source: kv.source, conditions: kv.conditions };
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

  function composeText(template, pairs, gidSentinel, gid, edits, health) {
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
    editHealth(lines, health);
    return lines.join('\n');
  }

  function containerText(template, pairs, gidSentinel, gid, edits, health) {
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
    editHealth(lines, health);
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
      checkImage(profile, checkpoint, settings);
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
      checkImage(profile, checkpoint, settings);
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
        // The API rank alone takes the listen address, and its health check asks the API there.
        const edits = numeric.filter(name => n === 0 || !API_RANK.includes(name)).map(name => [FLAGS[name][0], name, settings[name]]);
        const health = n === 0 ? healthEdit(checkpoint, settings) : null;
        files[`rank${n}/compose.yaml`] = composeText(variant.ranks[n].compose, pairs, gidSentinel, site.ranks[n].gid, edits, health);
        files[`rank${n}/container.json`] = containerText(variant.ranks[n].container, pairs, gidSentinel, site.ranks[n].gid, edits, health);
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
  // runtime/host/settings.py download_limit. `on` ("0,1" or "2,3") places a two-Spark
  // profile on that half of a four-Spark ring. `apiAddress` is the address shown for the
  // model (apiAddress()).
  function installCommand(profile, checkpoint, settings, meta, opts) {
    const words = ['--profile', profile.id, ...(opts.on ? ['--on', opts.on] : []), ...selectionWords(profile, checkpoint, settings)];
    if (opts.apiAddress) words.push('--api-address', opts.apiAddress);
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

  // ---- Command pack --------------------------------------------------------------------
  // A layout is how the Sparks serve: "pair", two Sparks with one model; "ring", four Sparks
  // with one four-Spark model; "halves", four Sparks with one two-Spark model on each half,
  // Sparks 0 and 1 (`--on 0,1`) and Sparks 2 and 3 (`--on 2,3`), as runtime/host/placement.py
  // defines them. A source offers "halves" only when its `sparkring install` defines --on
  // (features.ring_halves, export.features).
  const LAYOUT_SPARKS = { pair: 2, ring: 4, halves: 4 };
  const HALVES = ['0,1', '2,3'];
  function layouts(features) {
    return ['pair', 'ring', ...(features && features.ring_halves ? ['halves'] : [])];
  }

  // How the pack names each Spark, in rank order. Node A is rank 0: the Spark where setup, and
  // so the first install, runs, and where every command runs. The cabling ranks the others: every
  // cable joins port 0 of one Spark to port 1 of the next, and rank r+1 is the Spark on rank r's
  // port 0. `sparks`, the user's [{host, host_ip}] in rank order, names a Spark by its name or
  // address and its API by its address, or its name without one; a Spark without either, or
  // every Spark without `sparks`, is "Node A" or "Spark N", with NODE_A or SPARK_N standing for
  // its address.
  function sparkNames(count, sparks) {
    return Array.from({ length: count }, (_, rank) => {
      const role = rank === 0 ? 'Node A' : 'Spark ' + rank;
      const row = (sparks && sparks[rank]) || {};
      const name = String(row.host || '').trim(), address = String(row.host_ip || '').trim();
      if (!name && !address) return { role, short: role, label: role, address: rank === 0 ? 'NODE_A' : 'SPARK_' + rank, given: false };
      return { role, short: name || address, label: `${name || address} (${role})`, address: address || name, given: true };
    });
  }

  // The ordered commands of a layout, in groups:
  //   install: the layout's installation. For a new installation (opts.form "script") the first
  //            command is the one-command installer, which passes --on to `sparkring install`;
  //            every later command runs the `sudo sparkring install` it installed. A second half
  //            runs after the first: SparkRing runs one installation at a time.
  //   check:   `sudo sparkring status`, and `sudo sparkring cabling --bandwidth` when
  //            features.cable_check.
  //   switch:  on a ring, the commands that serve the other layout, last and `optional`, with
  //            `note` saying so: they replace the models the install group starts. Installing a
  //            half stops the four-Spark model, and installing the four-Spark model stops both
  //            halves' models.
  // Each command is {where, what, command, endpoint, error}: `endpoint` is the API it serves,
  // and `error` why it has no command. `plan` is {layout, features, main, halves}, where a
  // selection is {profile, checkpoint: name or null, settings: requested values, endpoint,
  // problems}: `main` is the pair's or the ring's model, and the four-Spark model the halves
  // layout switches to; `halves` are the two halves' models, which the ring layout switches to. A
  // selection's optional `endpoint` is {ask, address}: `ask` leaves out --yes so that the
  // installation asks where the model's API listens, and `address` is the address shown for it
  // (--api-address); its port and listen address are the settings api_port and api_bind. Its
  // optional `problems` are its fields' (readSelection). `opts` carries installCommand's form,
  // pin, approval and downloadLimit, and `order`: "auto", "ask", which leaves out --yes so that
  // every installation asks before it changes a Spark, or "fill", which names the Sparks after
  // `opts.sparks`. A new installation is never plan-only: `install.sh --plan` stops on a Spark
  // without the SparkRing package, so "plan" asks before each change instead.
  //
  // `problems` lists every field the pack cannot use, as {field, message, pair}: a selection's
  // fields (`pair` 0 or 1 for a half, else null), the download limit ("download_limit") and, for
  // "fill", each Spark's name or address ("rank<N>-host", "rank<N>-host_ip"); `field` is null for
  // a selection's settings that `sparkring install` refuses together. While any remains, no
  // command is given and the pack is not ready.
  function commandPack(plan, meta, opts) {
    const layout = plan.layout, features = plan.features || {};
    const names = sparkNames(LAYOUT_SPARKS[layout], opts.order === 'fill' ? opts.sparks : null);
    const approval = (opts.order === 'ask' && opts.approval === 'yes') || (opts.form === 'script' && opts.approval === 'plan')
      ? 'ask' : opts.approval;
    const pairOf = half => `${names[2 * half].short} and ${names[2 * half + 1].short}`;
    const run = 'On ' + names[0].label + ', as a user with sudo';
    const problems = [];
    const limit = readDownloadLimit(opts.downloadLimit);
    if (limit.problem) problems.push({ field: 'download_limit', message: limit.problem, pair: null });
    if (opts.order === 'fill') {
      for (const x of sparkProblems(opts.sparks || [])) problems.push({ field: `rank${x.rank}-${x.key}`, message: x.message, pair: null });
    }
    let first = true, asks = false;
    // One install command; `apiRank` is the rank that serves its API and `pair` the half whose
    // fields the selection holds.
    const install = (selection, on, apiRank, what, pair = null) => {
      const form = first ? opts.form : 'installed';
      first = false;
      const check = servingCheck(selection.profile, selection.settings, selection.checkpoint);
      const endpoint = selection.endpoint || {};
      let address = null, problem = check.ok ? null : check.error;
      if (!problem && endpoint.address) {
        try { address = apiAddress(endpoint.address); } catch (error) { problem = error.message; }
      }
      for (const x of selection.problems || []) problems.push({ ...x, pair });
      if (problem) problems.push({ field: null, message: problem, pair });
      else if ((selection.problems || []).length) problem = 'Fix the marked fields.';
      if (endpoint.ask) asks = true;
      const model = check.ok ? selection.profile.model_name + (check.checkpoint.default ? '' : ' (' + check.checkpoint.name + ')')
        : selection.profile.model_name;
      return {
        where: form === 'script' ? 'On ' + names[0].label + ', the Spark connected to your network, as a user with sudo' : run,
        what: (form === 'script' ? (layout === 'halves' ? 'Installs SparkRing on all four Sparks and starts ' : 'Installs SparkRing and starts ')
          : 'Starts ') + model + what,
        command: problem ? '' : installCommand(selection.profile, check.checkpoint, check.settings, meta,
          { form, pin: opts.pin, approval: endpoint.ask && approval === 'yes' ? 'ask' : approval,
            downloadLimit: limit.value, on, apiAddress: address }),
        endpoint: problem ? null : apiUrl(check.checkpoint, check.settings, names[apiRank].address, address),
        error: problem,
      };
    };
    const next = ' Run it after the command above finishes.';
    const groups = [];
    let switching = null;
    const optional = { optional: true, note: 'Optional: switch layouts later. This stops the models above.' };
    if (layout === 'halves') {
      groups.push({ key: 'install', title: 'Install', commands: [
        install(plan.halves[0], HALVES[0], 0, ` on the first pair: ${pairOf(0)}.`, 0),
        install(plan.halves[1], HALVES[1], 2, ` on the second pair: ${pairOf(1)}.` + next, 1),
      ] });
      switching = { key: 'switch', title: 'Switch to one model', ...optional, commands: [
        install(plan.main, null, 0, " on all four Sparks, and stops both pairs' models."),
      ] };
    } else {
      groups.push({ key: 'install', title: 'Install', commands: [
        install(plan.main, null, 0, layout === 'pair' ? ' on both Sparks.' : ' on all four Sparks.'),
      ] });
      if (layout === 'ring' && features.ring_halves) {
        switching = { key: 'switch', title: 'Switch to two models', ...optional, commands: [
          install(plan.halves[0], HALVES[0], 0, ` on ${pairOf(0)}, and stops the model on all four.`),
          install(plan.halves[1], HALVES[1], 2, ` on ${pairOf(1)}.` + next),
        ] };
      }
    }
    const check = [{ where: run, what: layout === 'halves' ? "Shows each pair's model separately." : 'Shows each Spark and the model.',
      command: 'sudo sparkring status', endpoint: null, error: null }];
    if (features.cable_check) {
      check.push({ where: run, what: "Measures each cable's speed. It skips cables a running model uses.",
        command: 'sudo sparkring cabling --bandwidth', endpoint: null, error: null });
    }
    groups.push({ key: 'check', title: 'Check', commands: check });
    if (switching) groups.push(switching);

    // Where to run the commands and where each model answers. In the ring order, Node A's port 0
    // leads to Spark 1 and its port 1 to Spark 3; Spark 2 is the only Spark that no cable joins to
    // Node A. Sparks the user named are called by their names and addresses.
    const roles = [];
    const far = names[2] && names[2].short;
    const at = (spark, unnamed) => spark.given ? spark.address : unnamed;
    if (layout === 'halves') {
      const [a, b, c, d] = names;
      roles.push({ spark: 'First pair', text: `${a.short} and ${b.short}. Answers at ${at(a, "Node A's address")}.` });
      roles.push({ spark: 'Second pair', text: `${c.short} and ${d.short}. Answers at ${at(c, "Spark 2's address")}.` });
    } else {
      const others = names.slice(1), shorts = others.map(spark => spark.short);
      const named = shorts.length < 3 ? shorts.join(' and ') : shorts.slice(0, -1).join(', ') + ' and ' + shorts[shorts.length - 1];
      roles.push({ spark: names[0].label, text: `Run the commands here. The model answers at ${at(names[0], 'its address')}.` });
      roles.push({ spark: others.some(spark => spark.given) ? named : others.length > 1 ? 'The other three' : 'The other Spark',
        text: others.length > 1 ? 'Nothing to run on them.' : 'Nothing to run on it.' });
    }
    const commands = groups.flatMap(group => group.commands);
    if (problems.length) {
      for (const c of commands) Object.assign(c, { command: '', error: c.error || 'Fix the marked fields.' });
    }
    const notes = ['If an install stops early, run it again; it picks up where it left off.'];
    const serves = spark => commands.some(c => c.endpoint && c.endpoint.startsWith(`http://${spark.address}:`));
    // NODE_A and SPARK_2 stand for the addresses of Sparks the user did not name.
    const placeholders = names.filter(spark => !spark.given && serves(spark));
    if (placeholders.length) {
      notes.push('Replace ' + placeholders.map(spark => `${spark.address} with ${spark.role}'s address`).join(' and ') + '.');
    }
    // A half's API on Spark 2 is its own LAN address, or its administration address without one.
    if (names.length > 2 && serves(names[2])) {
      notes.push(`If ${far} has no network cable of its own, only ${names[0].given ? names[0].short : 'Node A'} can reach the second pair's model.`);
    }
    if (opts.order === 'ask') notes.push('Each install shows which Spark is which and asks before it changes anything.');
    if (asks) notes.push("An install set to ask lists the Spark's addresses and asks which address and port the model uses.");
    return {
      layout, roles, groups, notes, problems,
      run: layout === 'halves' ? `Run every command on ${names[0].given ? names[0].short : 'Node A'}.` : null,
      order: layout === 'halves' ? "Node A is the Spark you run setup on. The cables number the others: Spark 1 is on Node A's "
        + "port 0, Spark 3 is on its port 1, and Spark 2 is the one with no cable to Node A." : null,
      ready: !problems.length && commands.every(c => c.command),
    };
  }

  // ---- Shareable links -----------------------------------------------------------------
  // A link carries the page's choices as query parameters, never a site value: `profile` and
  // its `checkpoint` and changed serving settings (one parameter per setting) for the pair's or
  // the ring's model; on a ring, `first` and `second` with `first.checkpoint`, `first.<setting>`
  // and so on for the halves' models; `layout`, `mode`, `image` and the install options. A
  // selection's API endpoint choice other than automatic adds `endpoint` (`ask` or `set`) and,
  // when set, `api_address`; its port and listen address are settings.
  // `choices` is {layout, mode, image, main, halves, install}, a selection being
  // {profile: id, checkpoint: name or null, settings} with an optional endpoint {mode, address}.
  const LINK_HALVES = ['first', 'second'];
  const INSTALL_CHOICES = { form: ['script', 'installed'], source: ['release', 'main'], approval: ['ask', 'plan', 'yes'],
    order: ['auto', 'ask', 'fill'] };
  const ENDPOINT_MODES = ['ask', 'set'];
  function linkQuery(choices) {
    const query = new URLSearchParams();
    const add = (key, selection) => {
      const prefix = key === 'profile' ? '' : key + '.';
      query.set(key, selection.profile);
      if (selection.checkpoint) query.set(prefix + 'checkpoint', selection.checkpoint);
      for (const name of Object.keys(selection.settings || {}).sort()) {
        query.set(prefix + name, selection.settings[name] === true ? '1' : String(selection.settings[name]));
      }
      const endpoint = selection.endpoint || {};
      if (ENDPOINT_MODES.includes(endpoint.mode)) query.set(prefix + 'endpoint', endpoint.mode);
      if (endpoint.mode === 'set' && endpoint.address) query.set(prefix + 'api_address', endpoint.address);
    };
    add('profile', choices.main);
    query.set('layout', choices.layout);
    query.set('mode', choices.mode);
    if (choices.layout !== 'pair') choices.halves.forEach((selection, i) => add(LINK_HALVES[i], selection));
    if (choices.image) query.set('image', choices.image);
    for (const [key, value] of Object.entries(choices.install)) if (value) query.set(key, value);
    return query.toString();
  }
  // The choices a link's query names, against `byId` (profile ID -> profile) and the layouts that
  // `features` offers; null without a known `profile`. A value the page does not offer is left
  // out (null for a half, mode and image); a layout that does not fit `profile` follows its size.
  function linkChoices(search, byId, features) {
    const query = new URLSearchParams(search);
    const selection = (key, nodes) => {
      const p = byId[query.get(key)];
      if (!p || (nodes && p.nodes !== nodes)) return null;
      const prefix = key === 'profile' ? '' : key + '.', name = query.get(prefix + 'checkpoint');
      const checkpoint = p.checkpoints.some(c => c.name === name && !c.default) ? name : null;
      const settings = {};
      for (const row of checkpointOf(p, checkpoint).settings) {
        const text = query.get(prefix + row.name);
        if (text === null) continue;
        if (row.switch) { if (text === '1') settings[row.name] = true; }
        else if (row.address) { if (IPV4.test(text)) settings[row.name] = text; }
        else if (/^\d+$/.test(text)) settings[row.name] = Number(text);
      }
      const mode = query.get(prefix + 'endpoint');
      if (!ENDPOINT_MODES.includes(mode)) return { profile: p.id, checkpoint, settings };
      return { profile: p.id, checkpoint, settings,
        endpoint: { mode, address: mode === 'set' ? query.get(prefix + 'api_address') || '' : '' } };
    };
    const main = selection('profile');
    if (!main) return null;
    const nodes = byId[main.profile].nodes, wanted = query.get('layout');
    const layout = layouts(features).includes(wanted) && LAYOUT_SPARKS[wanted] === nodes ? wanted : nodes === 4 ? 'ring' : 'pair';
    const install = {};
    for (const [key, values] of Object.entries(INSTALL_CHOICES)) if (values.includes(query.get(key))) install[key] = query.get(key);
    install.limit = query.get('limit') || '';
    return { layout, main, halves: LINK_HALVES.map(key => selection(key, 2)),
      mode: ['install', 'compose'].includes(query.get('mode')) ? query.get('mode') : null, image: query.get('image'), install };
  }

  // `address` is the address shown for the model (apiAddress()), or null.
  function readme(profile, checkpoint, site, settings, output, meta, address = null) {
    const ranks = site.ranks.length, api = apiUrl(checkpoint, settings, site.master, address);
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
      ...site.ranks.map((r, i) => `  rank${i}  ${r.host}  ${r.host_ip}` + (i === 0 ? `  API ${api}, model ${checkpoint.served_model_name}` : '')),
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
      `  5. curl ${api}/models lists ${checkpoint.served_model_name} when the model serves.`,
      '     The first start on this image compiles and tunes kernels, about 10 minutes.',
      '',
      '`sparkring compose check --deployment <this directory>` checks these files against the source.',
      'The containers do not restart on their own (restart: no). Generated configuration only; Compose serving is not qualified.',
    ];
    if (profile.installable) {
      lines.push('', 'Without Compose, `sparkring install` sets up the Sparks, downloads the image and checkpoint and',
        'starts the same model. On Node A:',
        '  ' + installCommand(profile, checkpoint, settings, meta, { form: 'script', pin: true, approval: 'ask', apiAddress: address }));
    }
    lines.push('');
    return lines.join('\n');
  }

  // `address` is the address shown for the model in its README, or null.
  function archive(profile, checkpoint, site, settings, output, meta, when = new Date(), address = null) {
    const root = site.name + '/';
    const entries = [{ name: root, dir: true }, { name: root + 'deployment.json', text: output.deployment }];
    for (let n = 0; n < site.ranks.length; n++) {
      entries.push({ name: `${root}rank${n}/`, dir: true });
      entries.push({ name: `${root}rank${n}/compose.yaml`, text: output.files[`rank${n}/compose.yaml`] });
      entries.push({ name: `${root}rank${n}/container.json`, text: output.files[`rank${n}/container.json`] });
    }
    entries.push({ name: `${root}${site.name}.site.yaml`, text: output.site_yaml });
    entries.push({ name: root + 'README.txt', text: readme(profile, checkpoint, site, settings, output, meta, address) });
    return { filename: site.name + '.zip', bytes: zip(entries, when), names: entries.filter(e => !e.dir).map(e => e.name) };
  }

  return { render, archive, installCommand, renderCommand, derivedDirectory, sourceName, validDownloadLimit, servingCheck,
    checkpointOf, option, yamlScalar, resolves, encoded, siteYaml, validateSite, fieldProblems, apiAddress, apiUrl, listenable,
    layouts, LAYOUT_SPARKS, sparkNames, sparkProblems, commandPack, linkQuery, linkChoices, kvEstimate, kvText,
    readSetting, readAddress, readDownloadLimit, readSelection, offered };
})();
if (typeof module !== 'undefined') module.exports = SparkRingEngine;
