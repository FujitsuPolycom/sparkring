"""Public page provenance, navigation and general installer allocation contracts."""
from html.parser import HTMLParser
import re
from urllib.parse import unquote, urlsplit

import pytest

from runtime.common import installer, profiles, thinking
from runtime.host import placement
from scripts.compose_builder import export, site
from scripts.compose_builder.test_compose_builder import run_engine, NODE

META = {"repository": export.REPOSITORY, "ref": "a" * 40, "commit": "a" * 40,
        "tag": None, "since_tag": None, "commits_since": None}


@pytest.fixture(scope="module")
def public():
    return site.data(META)


def engine(script, value):
    if NODE is None:
        pytest.skip("Node.js unavailable")
    return run_engine(NODE, script, value)


def test_landing_profiles_match_installer_catalog_and_thinking(public):
    assert {p["id"] for p in public["profiles"]} == set(installer.INSTALLABLE)
    html = site.landing(public)
    for p in public["profiles"]:
        table = html.split('id="profile-table"', 1)[1].split('</table>', 1)[0]
        assert table.count(f'<code>{p["id"]}</code>') == 1
        assert p["thinking"] == thinking.summary(thinking.of(p["id"]))
        assert p["status"] == profiles.load(p["id"])[0]["status"]
    assert "No model · setup only" in html
    assert "Local Inference Lab License, Version 1.0" in html
    assert f"--ref {public['release']} --profile PROFILE" in html


def test_speed_table_matches_consolidated_release_record(public):
    source = site.section(site.read("README.md"), "Speed")
    raw = [line for line in source.splitlines() if line.startswith('| ') and re.search(r'\| [248] \|', line)]
    assert len(public["speed"]) == len(raw)
    for row, line in zip(public["speed"], raw):
        cells = [s.strip() for s in line.strip('|').split('|')]
        assert row["profile"] == cells[3].strip('`')
        assert row["layout"] == cells[7].split(':')[0]
        assert all(abs(n - float(s.replace(',', ''))) <= .5 for n, s in
                   zip([*row["decode"], row["prefill"], row["coding"]], [*cells[4].split('/'), cells[5], cells[6]]))
    # The record exposes the .5 rounding edge; the website uses familiar half-up rounding.
    assert next(r for r in public["speed"] if r["coding"] == 112)["model"] == "Qwen3.8-Flash-Next"
    assert 'path of four: four Sparks' not in site.landing(public).split('id="speed-table"')[1].split('</table>')[0]


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.links, self.resources = set(), [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            assert attrs["id"] not in self.ids, f'Duplicate id: {attrs["id"]}'
            self.ids.add(attrs["id"])
        if tag == "a" and "href" in attrs:
            self.links.append(attrs["href"])
        if tag in ("script", "img", "link"):
            target = attrs.get("src", attrs.get("href"))
            if target:
                self.resources.append(target)


def test_pages_links_local_assets_themes_and_command_anchors(public):
    # Full builder markup is included, but expensive template data is tested by its existing suite.
    builder = site.builder(export.page({**META, "profiles": [], "features": {}, "images": []}), public)
    pages = {"index.html": site.landing(public), "commands.html": site.commands(META), "builder.html": builder}
    parsed = {}
    for name, html in pages.items():
        parser = Links()
        parser.feed(html)
        parsed[name] = parser
        assert 'href="tokens.css"' in html
        assert 'id="theme"' in html
        assert 'name="viewport"' in html
        for resource in parser.resources:
            assert not urlsplit(resource).scheme, f"External request: {resource}"
            assert (site.HERE / resource).is_file()
    for name, parser in parsed.items():
        for target in parser.links:
            split = urlsplit(target)
            if split.scheme:
                prefix = site.repository_url('', META)
                if target.startswith(prefix):
                    local = unquote(target[len(prefix):].split('#')[0])
                    assert (site.ROOT / local).exists(), local
                continue
            path = split.path or name
            assert path in parsed, target
            if split.fragment:
                assert unquote(split.fragment) in parsed[path].ids, target


def test_automatic_and_explicit_slices_match_installer_aligned_candidates():
    cases = []
    for size in range(2, 9):
        for count in (2, 4, 6, 8):
            if count > size:
                continue
            layout = {"size": size, "shape": "pair" if size == 2 else "cycle"}
            expected = [list(a) for a in placement.aligned(layout, count) if placement.unsupported(layout, a if count != size else None) is None]
            slots = [{"profile": {"nodes": count}, "where": "auto"} for _ in expected]
            cases.append([size, slots, expected])
    actual = engine("console.log(JSON.stringify(value.map(([n,s]) => E.resolveSlices(n,s))));", cases)
    for found, (_, _, expected) in zip(actual, cases):
        assert found["groups"] == expected and not found["problems"]


def test_slices_wrap_reserve_explicit_before_automatic_and_block_overlap():
    def slot(count, where="auto"):
        return {"profile": {"nodes": count}, "where": where}
    cases = [[8, [slot(4), slot(2, "0,1"), slot(2)]],
             [8, [slot(4, "6,7,0,1"), slot(4, "2,3,4,5")]],
             [4, [slot(2, "0,1"), slot(2, "1,2")]],
             [8, [slot(4, "0,2,3,4")]], [8, [slot(6)]], [4, [slot(4), slot(2)]]]
    found = engine("console.log(JSON.stringify(value.map(([n,s]) => E.resolveSlices(n,s))));", cases)
    assert found[0]["groups"] == [[4, 5, 6, 7], [0, 1], [2, 3]]
    assert found[1]["groups"] == [[6, 7, 0, 1], [2, 3, 4, 5]]
    assert not found[0]["problems"] and not found[1]["problems"]
    assert all(row["problems"] for row in found[2:])


def test_slice_commands_and_setup_only_keep_existing_install_contracts(public):
    pair = next(p for p in public["profiles"] if p["id"] == "qwen38-flash-next-tp2")
    result = engine("""
const p = value.p, meta = value.meta, opts = {form:'script',pin:true,approval:'ask'};
const slots = [{profile:p,where:'auto'}, {profile:p,where:'2,3',settings:{max_concurrency:4}}];
console.log(JSON.stringify({
  slices:E.slicePack(4,slots,meta,opts),
  overlap:E.slicePack(4,[{profile:p,where:'0,1'},{profile:p,where:'1,2'}],meta,opts),
  fresh:E.slicePack(8,[],meta,{...opts,setupOnly:true}),
  existing:E.slicePack(8,[],meta,{...opts,form:'installed',setupOnly:true}),
  plan:E.slicePack(4,slots,meta,{...opts,form:'installed',approval:'plan'}),
  invalid:E.slicePack(4,slots,meta,{...opts,downloadLimit:'oops'}),
  unavailable:E.slicePack(2,[{profile:{...p,installable:false}}],meta,opts)
}));""", {"p": pair, "meta": META})
    assert result["slices"]["ready"]
    commands = result["slices"]["commands"]
    assert "--on 0,1" in commands[0]["command"] and f"--ref {META['ref']}" in commands[0]["command"]
    assert commands[1]["command"].startswith("sudo sparkring install ")
    assert "--on 2,3" in commands[1]["command"] and "--max-concurrency 4" in commands[1]["command"]
    assert result["fresh"]["commands"][0]["command"].endswith('--package-only')
    assert result["fresh"]["commands"][1]["command"] == 'sudo sparkring setup'
    assert result["existing"]["commands"][0]["command"] == 'sudo sparkring setup'
    assert all(c["command"].endswith('--plan') for c in result["plan"]["commands"][:-1])
    for key in ("overlap", "invalid", "unavailable"):
        assert not result[key]["ready"]
        assert all(not c["command"] for c in result[key]["commands"])


def test_vendored_yaml_matches_existing_integrity():
    import base64
    import hashlib
    digest = base64.b64encode(hashlib.sha512((site.HERE / 'js-yaml.min.js').read_bytes()).digest()).decode()
    assert digest == 'CSBhVREyzHAjAFfBlIBakjoRUKp5h7VSweP0InR/pAJyptH7peuhCsqAI/snV+TwZmXZqoUklpXp6R6wMnYf5Q=='
