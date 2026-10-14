"""Repository-backed public pages; no model inventory or measurements live in templates."""
from decimal import Decimal, ROUND_HALF_UP
from html import escape, unescape
import json
import posixpath
import re
from urllib.parse import urlsplit

import markdown

from runtime.common import image_lock, installer_image, profiles, setup, thinking, toolchain_profiles
from runtime.host import models
from scripts.compose_builder import export

HERE, ROOT = export.HERE, export.ROOT
# The default --ref of install.sh, with the comment that explains it (hosted_installer).
INSTALLER_REF = ('# The branch that publishes this script. A copy fetched from another branch,\n'
                 '# tag or commit must be run with a matching --ref.\n'
                 'REF="main"\n')


def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


def section(text, title):
    return text.split(f"## {title}\n", 1)[1].split("\n## ", 1)[0].strip()


def repository_url(path, meta):
    return f"https://github.com/{meta['repository']}/blob/{meta['ref']}/{path}"


def prose(text, source, meta):
    """Render repository Markdown, preserving local anchors and resolving document-relative links."""
    rendered = markdown.markdown(text, extensions=["tables", "fenced_code", "toc"])

    def link(match):
        target = match[1]
        if urlsplit(target).scheme or target.startswith("#"):
            return match[0]
        path = posixpath.normpath(posixpath.join(posixpath.dirname(source), target))
        if path.startswith("docs/operations/commands.md"):
            target = "commands.html" + path.removeprefix("docs/operations/commands.md")
        elif path == "README.md#profiles":
            target = "index.html#profiles"
        else:
            target = repository_url(path, meta)
        return f'href="{escape(target, quote=True)}"'

    rendered = re.sub(r'href="([^"]+)"', link, rendered)

    def responsive(match):
        table = match[0]
        headers = [unescape(re.sub('<[^>]+>', '', h)) for h in re.findall(r'<th>(.*?)</th>', table, re.S)]
        column = 0

        def cell(match):
            nonlocal column
            label = headers[column % len(headers)]
            column += 1
            return f'<td data-label="{escape(label, quote=True)}">{match[1]}</td>'

        if headers:
            table = re.sub(r'<td>(.*?)</td>', cell, table, flags=re.S)
        return table.replace('<table>', '<table class="responsive-table">', 1)

    return re.sub(r'<table>.*?</table>', responsive, rendered, flags=re.S)


def release_record(meta):
    """Use the named release, or the newest accepted numeric release recorded in this tree."""
    if meta.get("tag") and (ROOT / "runtime/releases" / meta["tag"] / "status.json").is_file():
        return json.loads(read(f"runtime/releases/{meta['tag']}/status.json"))
    records = [json.loads(p.read_text(encoding="utf-8")) for p in (ROOT / "runtime/releases").glob("*/status.json")
               if re.fullmatch(r"\d+\.\d+\.\d+", p.parent.name)]
    return max((r for r in records if r.get("maturity") == "accepted"),
               key=lambda r: tuple(map(int, r["release"].split("."))))


def installer_catalog():
    names = json.loads(read("profiles/model-names.json"))
    result = []
    for row in models.catalog():
        if not row["automated"]:
            continue
        definition, _ = profiles.load(row["profile"])
        config = toolchain_profiles.read(ROOT / definition["configuration"]["path"])
        default, choices = toolchain_profiles.checkpoint_names(config)
        checkpoints = []
        for name in ([default, *sorted(n for n in choices if n != default)] if default else [None]):
            selected = toolchain_profiles.checkpoint_settings(config, name)
            args = selected["vllm_args"]
            repository = selected["model"]["repository"]
            checkpoints.append({"name": name, "default": name == default, "install_default": name == default,
                                "aliases": [], "settings": export._setting_rows(args),
                                "port": export._argument(args, "--port"),
                                "served_model_name": selected.get("served_model_name", row["profile"]),
                                "model_repository": repository,
                                "variant": names["quant_labels"].get(repository, repository.split("/")[-1])})
        repository = row["model"]["repository"]
        result.append({"id": row["profile"], "model_name": names["models"].get(repository, row["title"]),
                       "variant": names["quant_labels"].get(repository, repository.split("/")[-1]),
                       "nodes": row["nodes"], "status": definition["status"],
                       "thinking": thinking.summary(thinking.of(row["profile"])), "guide": row["guide"],
                       "checkpoints": checkpoints})
    return result


def planner_images(catalog):
    """The images the planner offers for ``catalog``'s profiles, the install default first (export.offered_images).

    ``default`` marks the image `sparkring install` uses without --image, and
    ``profiles`` maps each profile the image runs to its install-default
    checkpoint (``preferred``) and the checkpoints the image reads.
    """
    result = []
    names = [row["name"] for row in installer_image.catalog()]
    for row in export.offered_images([profile["id"] for profile in catalog]):
        offered = {}
        for profile in catalog:
            ident = profile["id"]
            if ident not in installer_image.profiles_of(row["lock"]):
                continue
            offered[ident] = {
                "preferred": image_lock.preferred_checkpoint(row["lock"], ident),
                "checkpoints": [c["name"] for c in profile["checkpoints"]
                                if not image_lock.checkpoint_problem(row["lock"], setup.selection(ident, c["name"]))],
            }
        result.append({"name": row["name"], "option": export._image_option(row, names),
                       "default": row["default"], "profiles": offered,
                       "sircl": row["lock"].get("schema", "").endswith("/v3"),
                       "capabilities": sorted(installer_image.capabilities(row["name"]))})
    return result


def speed_rows():
    """Read the release's consolidated speed record in README, including links to its underlying runs.

    The individual run records do not all contain the 32K cells. The README
    owns the consolidated nine-row selection; never infer cells from ranges.
    """
    def whole(value):
        return int(Decimal(value.strip().replace(",", "")).quantize(Decimal("1"), rounding=ROUND_HALF_UP))

    rows = []
    for line in section(read("README.md"), "Speed").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 8 or not cells[2].isdigit():
            continue
        rows.append({"model": cells[0], "checkpoint": cells[1], "nodes": int(cells[2]),
                     "profile": cells[3].strip("`"), "decode": [whole(n) for n in cells[4].split("/")],
                     "prefill": whole(cells[5]), "coding": whole(cells[6]), "layout": cells[7].split(":", 1)[0]})
    if not rows:
        raise ValueError("README Speed must contain the release's consolidated measurements")
    return rows


def data(meta):
    catalog = installer_catalog()
    release = release_record(meta)
    images = planner_images(catalog)
    # The planner starts on the install default; each profile's port and variant are its install default's there.
    default_image = next(image for image in images if image["default"])
    for profile in catalog:
        preferred = default_image["profiles"].get(profile["id"], {}).get("preferred")
        checkpoint = next((c for c in profile["checkpoints"] if c["name"] == preferred), profile["checkpoints"][0])
        profile["variant"] = checkpoint["variant"]
        profile["port"] = checkpoint["port"]
    return {"profiles": catalog, "images": images, "speed": speed_rows(),
            "release": release["release"],
            "release_record": release["record"], "meta": meta,
            "aliases": export.aliases({profile["id"] for profile in catalog})}


def short_link(profile_id, aliases):
    """``builder.html?p=NAME``: the planner's short link to a profile, by its short name when it has one."""
    name = min((alias for alias, target in aliases.items() if target == profile_id), default=profile_id)
    return f"builder.html?p={name}"


def hosted_installer(ref, site_url):
    """install.sh as the site publishes it: the checkout's script with ``ref`` as its default --ref.

    `curl -fsSL SITE/install.sh | bash -s -- --profile PROFILE` then builds the
    source the site's pages describe without --ref. ``ref`` is a tag or a commit.
    """
    if not re.fullmatch(r"[0-9A-Za-z][0-9A-Za-z._-]*", ref or ""):
        raise ValueError(f"install.sh cannot take {ref!r} as its default --ref")
    text = read("install.sh")
    if text.count(INSTALLER_REF) != 1:
        raise ValueError("install.sh must set its default REF once, after the comment hosted_installer replaces")
    return text.replace(INSTALLER_REF, f"# The source that the Install Builder at {site_url} describes; this copy of\n"
                                       "# install.sh is published there (scripts/compose_builder/site.py).\n"
                                       f'REF="{ref}"\n')


def json_script(ident, value):
    text = json.dumps(value, separators=(",", ":")).replace("<", "\\u003c")
    return f'<script type="application/json" id="{ident}">{text}</script>'


def nav():
    return ('<nav class="site-nav" aria-label="Main"><div class="site-wrap">'
            '<a class="brand" href="index.html">◌ SparkRing</a><a href="index.html#profiles">Profiles</a>'
            '<a href="index.html#speed">Speed</a><a href="builder.html">Install Builder</a>'
            '<a href="commands.html">Command guide</a><button type="button" id="theme">Light theme</button></div></nav>')


def document(title, body, scripts=""):
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{escape(title)}</title><link rel="stylesheet" href="tokens.css">'
            '<link rel="stylesheet" href="site.css"><script src="theme.js"></script></head><body>'
            + nav() + body + '<footer class="site-footer site-wrap">SparkRing · Directly cabled DGX Sparks. '
            '<a href="builder.html">Build an installation</a> · <a href="commands.html">Commands</a></footer>'
            '<script src="site.js"></script>' + scripts + '</body></html>')


def table(headers, rows, ident):
    return (f'<div class="table-wrap"><table class="responsive-table" id="{ident}"><thead><tr>'
            + ''.join(f'<th scope="col">{escape(h)}</th>' for h in headers) + '</tr></thead><tbody>'
            + ''.join('<tr>' + ''.join(f'<td data-label="{escape(h)}">{v}</td>' for h, v in zip(headers, row)) + '</tr>'
                      for row in rows) + '</tbody></table></div>')


def landing(value):
    meta = value["meta"]
    intro = read("README.md").split("\n## Quick start", 1)[0].split("\n\n")
    license_notice, description = intro[1], intro[2]
    description = description.replace("two to eight NVIDIA DGX Sparks\ncabled directly", "two or more directly cabled DGX Sparks\nconnected")
    # The site's own install.sh copy (hosted_installer) on the site; landing.js shows it there.
    cmd = (f"curl -fsSL https://raw.githubusercontent.com/{meta['repository']}/{meta['ref']}/install.sh "
           f"| bash -s -- --ref {meta['ref']} --profile PROFILE")
    hosted = f"curl -fsSL {meta['site_url']}install.sh | bash -s -- --profile PROFILE"
    profile_rows = []
    for p in value["profiles"]:
        status = "research" if p["status"] in ("research-only", "unsupported") else "tested"
        profile_rows.append([escape(p["model_name"]), escape(p["variant"]), str(p["nodes"]),
                             f'<a href="{escape(short_link(p["id"], value["aliases"]))}"><code>{p["id"]}</code></a>',
                             escape(str(p["port"])), escape(p["thinking"]), f'<span class="pill {status}">{status}</span>'])
    profile_rows.append(['No model · setup only', 'Set up the network and dashboard', '2–8',
                         '<a href="builder.html?setup=1">No model: set up the Sparks only</a>', '—', '—', '—'])
    speed = [[escape(r["model"]), str(r["nodes"]), escape(r["layout"]), *map(str, r["decode"]),
              f'{r["prefill"]:,}', str(r["coding"])] for r in value["speed"]]
    speed_section = section(read("README.md"), "Speed")
    conditions = speed_section.split('\n| Model', 1)[0]
    body = f'''<main class="site-wrap">
      <div class="hero"><div><p class="eyebrow">DGX Spark · ConnectX-7 · no switch</p>
      <h1>Large language models on two or more directly cabled DGX Sparks</h1>
      <div class="command-box"><pre id="install-command" data-hosted="{escape(hosted)}">{escape(cmd)}</pre><button data-copy="install-command">Copy command</button></div>
      <p class="hint">Choose <code>PROFILE</code> below or use the <a href="builder.html">Install Builder</a>.</p>
      <p>The installer asks before making changes. Use <code>--plan</code> to see what a deployment would do on your stack. <a href="commands.html">Command guide</a></p>
      <p class="mono muted">Release {escape(value['release'])} · {len(value['profiles'])} installer profiles</p></div>
      <div class="ring-panel landing-ring"><div class="ring-controls" id="ring-choices" role="group" aria-label="Number of Sparks"></div><div id="ring-figure"></div>
      <div class="cable-legend"><span>Direct cable</span><span>Relayed through a Spark</span></div>
      <p class="hint" id="ring-caption"></p><p class="eyebrow">Example setups</p>
      <div class="ring-controls" id="ring-examples" role="group" aria-label="Example setups"></div>
      <ul id="example-list" aria-live="polite"></ul><p><a class="button" id="load-example" href="builder.html">Load into builder →</a></p></div></div>
      <div class="notice">{prose(license_notice, 'README.md', meta)}</div>
      <section class="site-section" id="profiles"><h2>Profiles</h2>
      {table(['Model', 'Variant', 'Sparks', '--profile', 'Port', 'Thinking', 'Status'], profile_rows, 'profile-table')}</section>
      <section class="site-section" id="speed"><h2>Speed</h2>{prose(conditions, 'README.md', meta)}
      {table(['Model', 'Sparks', 'Layout', '1 stream', '4 streams', '8 streams', 'Prefill 32K', 'Coding peak'], speed, 'speed-table')}</section>
      <section class="site-section" id="sircl"><h2>SIRCL and net-sircl</h2>{prose(section(read('README.md'), 'SIRCL'), 'README.md', meta)}</section></main>'''
    return document("SparkRing Builder", body,
                    json_script("site-data", value) + '<script src="ring.js"></script><script src="landing.js"></script>')


def commands(meta):
    return document("Commands · SparkRing Builder", '<main class="site-wrap guide">' + prose(read("docs/operations/commands.md"),
                                                                               "docs/operations/commands.md", meta) + '</main>')


def builder(html, value):
    # Keep the complete verified Compose UI and its hints; the planner adds general slices.
    html = html.replace('</head>', '<link rel="stylesheet" href="tokens.css"><link rel="stylesheet" href="site.css">'
                        '<script src="theme.js"></script></head>')
    html = re.sub(r'<script src="https://cdnjs[^>]+></script>', '<script src="js-yaml.min.js"></script>', html)
    planner = (HERE / "planner.html").read_text(encoding="utf-8")
    html = html.replace('<body>', '<body>' + nav() + planner + '<details class="advanced site-wrap" id="advanced">'
                        '<summary>Docker Compose packages and detailed deployment controls</summary>'
                        '<p class="hint">This independent editor preserves per-Spark files, YAML import, checkpoints, serving settings, '
                        'API addresses, release selection and layout-switch commands for Compose-supported pairs and rings of four.</p>')
    html = html.replace('<script type="application/json" id="data">', '</details><script type="application/json" id="data">')
    return html.replace('</body>', json_script("site-data", value) + '<script src="site.js"></script><script src="ring.js"></script>'
                        '<script src="planner.js"></script></body>')


def write(output, builder_html, meta):
    value = data(meta)
    for name, content in {"index.html": landing(value), "builder.html": builder(builder_html, value),
                          "commands.html": commands(meta)}.items():
        (output / name).write_text(content, encoding="utf-8", newline="\n")
    (output / "install.sh").write_text(hosted_installer(meta["ref"], meta["site_url"]), encoding="utf-8", newline="\n")
    for name in ("tokens.css", "site.css", "theme.js", "site.js", "ring.js", "landing.js", "planner.js", "js-yaml.min.js"):
        (output / name).write_bytes((HERE / name).read_bytes())
