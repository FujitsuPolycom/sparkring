"""Public page provenance, navigation and general installer allocation contracts."""
from html.parser import HTMLParser
import re
from urllib.parse import unquote, urlsplit

import pytest

from runtime.common import installer, installer_image, profiles, serving, thinking
from runtime.host import placement
from scripts.compose_builder import export, site
from scripts.compose_builder.test_compose_builder import run_engine, NODE

META = {"repository": export.REPOSITORY, "ref": "a" * 40, "commit": "a" * 40,
        "tag": None, "since_tag": None, "commits_since": None, "site_url": export.site_url()}
DCP2, DCP2_SHORT = "glm53-flash-nvfp4-spark-tp2-dcp2", "glm53-flash-tp2-dcp2"


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
    table = html.split('id="profile-table"', 1)[1].split('</table>', 1)[0]
    offered = {profile for image in public["images"] for profile in image["profiles"]}
    for p in public["profiles"]:
        # The table names a profile by the short name --profile takes, and lists only profiles an offered image runs.
        name = site.short_name(p["id"], public["aliases"])
        assert table.count(f'<code>{name}</code>') == (1 if p["id"] in offered else 0)
        assert p["offered"] == (p["id"] in offered)
        assert p["status"] == profiles.load(p["id"])[0]["status"]
    for hidden in installer_image.UNLISTED:
        assert not next(p for p in public["profiles"] if p["id"] == hidden)["offered"]
        assert hidden not in table
    assert f'{len([p for p in public["profiles"] if p["offered"]])} installer profiles' in html
    for alias in ("glm53-flash-tp2", "glm53-flash-tp4", "glm53-flash-tp2-dcp2", "qwen38-flash-next-tp4"):
        assert f'<code>{alias}</code>' in table
    assert "No model · setup only" in html
    assert "Local Inference Lab License, Version 1.0" in html
    # The quick start pins the page's source; on the site, landing.js shows the site's own install.sh.
    assert f"install.sh | bash -s -- --ref {META['ref']} --profile PROFILE" in html
    assert 'data-hosted="curl -fsSL https://fujitsupolycom.github.io/sparkring/install.sh | bash -s -- --profile PROFILE"' in html
    # Each profile links to the builder by its short link.
    assert f'href="builder.html?p={DCP2_SHORT}"' in html and 'href="builder.html?p=qwen38-flash-next-tp2"' in html


def test_thinking_shows_the_installed_default_with_the_profile_s_serving_defaults(public):
    for p in public["profiles"]:
        for checkpoint in p["checkpoints"]:
            record = thinking.of(p["id"], checkpoint["name"])
            effort = serving.defaults(p["id"], checkpoint["name"]).get("reasoning_effort")
            expected = thinking.summary(record) if effort is None else thinking.summary({**record, "level": effort})
            assert checkpoint["thinking"] == expected
    # GLM-5.3's chat template thinks at max without an effort; its profiles install with high.
    glm = next(p for p in public["profiles"] if p["id"] == "glm53-flash-nvfp4-spark-tp2")
    assert glm["thinking"] == "always · high" and thinking.summary(thinking.of(glm["id"])) == "always · max"
    table = site.landing(public).split('id="profile-table"', 1)[1].split('</table>', 1)[0]
    assert "always · high" in table and "always · max" not in table


def test_variant_labels_are_short(public):
    names = site.json.loads(site.read("profiles/model-names.json"))
    for p in public["profiles"]:
        for label in [p["variant"], *(c["variant"] for c in p["checkpoints"])]:
            assert len(label) <= 12, f"{p['id']}: add a short_labels entry for {label!r}"
        assert all(c["variant"] == site.variant_label(c["model_repository"], names) for c in p["checkpoints"])
    pair = next(p for p in public["profiles"] if p["id"] == "glm53-flash-nvfp4-spark-tp2")
    assert next(c for c in pair["checkpoints"] if c["name"] == "csf")["variant"] == "CSF"


def test_release_label_names_the_release_the_page_is_built_for(public):
    records = [site.json.loads(p.read_text(encoding="utf-8")) for p in (site.ROOT / "runtime/releases").glob("*/status.json")]
    for record in records:
        if site.RELEASE.fullmatch(str(record.get("release"))):
            # A record may live in its installer image's directory; the release field identifies it.
            assert site.release_record({**META, "tag": record["release"]}) == record
            assert site.release_record({**META, "since_tag": record["release"], "commits_since": 2}) == record
    assert site.release_record({**META, "tag": "2026.10.4"})["image_release"] == "dev-20261013-dcp2prefill-status038-sircl-tp6"
    assert site.release_record({**META, "tag": "2099.1.0"}) == {"release": "2099.1.0", "record": None}
    assert site.release_label({**META, "tag": "2026.10.4"}, "2026.10.4") == "Release 2026.10.4"
    assert site.release_label({**META, "since_tag": "2026.10.4", "commits_since": 1}, "2026.10.4") == "Release 2026.10.4 + 1 commit"
    # Without a release tag, the highest-numbered record, whatever its maturity.
    newest = max((r["release"] for r in records if site.RELEASE.fullmatch(str(r.get("release")))),
                 key=lambda name: tuple(map(int, name.split("."))))
    assert public["release"] == newest and f'>Release {newest} · ' in site.landing(public)


def test_landing_pair_is_one_cable():
    script = (site.HERE / "landing.js").read_text(encoding="utf-8")
    assert "'Two Sparks, one direct ConnectX-7 cable.'" in script and "two direct" not in script
    pair = script.split("if (n === 2) {", 1)[1].split("} else {", 1)[0]
    assert pair.count("<line ") == 1 and "<path" not in pair


def test_planner_offers_the_install_default_then_its_rollback_image(public):
    from runtime.common import image_lock
    rows = image_lock.catalog()
    assert [(row["name"], row["default"]) for row in public["images"]] == [
        (image_lock.default_row()["name"], True), (export.rollback_row(rows)["name"], False)]


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


def test_the_site_s_install_sh_defaults_to_the_page_s_source(tmp_path):
    hosted = site.hosted_installer("2026.10.4", META["site_url"])
    assert '\nREF="2026.10.4"\n' in hosted and hosted.count("REF=\"") == 1
    import shutil
    import subprocess
    bash = shutil.which("bash")
    if bash:
        path = tmp_path / "install.sh"
        path.write_text(hosted, encoding="utf-8", newline="\n")
        subprocess.run([bash, "-n", str(path)], check=True)
    for ref in ("", "main; rm -rf /", "$(id)", "-x"):
        with pytest.raises(ValueError):
            site.hosted_installer(ref, META["site_url"])


def test_planner_links_select_their_model_sparks_and_image(public):
    by_id = {p["id"]: p for p in public["profiles"]}
    rollback = next(row for row in public["images"] if not row["default"])
    searches = [f"p={DCP2_SHORT}", f"profile={DCP2}&sparks=8", f"p={DCP2_SHORT}&image={rollback['option']}",
                f"profile={DCP2}&checkpoint=nonexistent&max_concurrency=4", "p=glm53-flash-tp4&layout=halves&first="
                + DCP2 + "&second=qwen38-flash-next-tp2&second.max_concurrency=2", "p=unknown", "models=qwen38-flash-next-tp2"]
    found = engine("""
const { byId, images, aliases, searches } = value;
console.log(JSON.stringify(searches.map(search => E.planLink(search, byId, images, aliases))));""",
                   {"byId": by_id, "images": public["images"], "aliases": public["aliases"], "searches": searches})
    default = public["images"][0]["name"]
    assert [(plan["size"], [(s["id"], s["where"]) for s in plan["slots"]], plan["image"]) for plan in found[:5]] == [
        (2, [(DCP2, "auto")], default), (8, [(DCP2, "auto")], default), (2, [(DCP2, "auto")], rollback["name"]),
        (2, [(DCP2, "auto")], default), (4, [(DCP2, "0,1"), ("qwen38-flash-next-tp2", "2,3")], default)]
    assert found[3]["slots"][0]["entries"] == {"max_concurrency": "4"} and found[3]["slots"][0]["checkpoint"] is None
    assert found[4]["slots"][1]["entries"] == {"max_concurrency": "2"}
    assert found[5:] == [None, None]


def test_planner_shares_the_short_link_for_the_plan_it_starts_with(public):
    by_id = {p["id"]: p for p in public["profiles"]}
    default, rollback = public["images"][0]["name"], public["images"][1]["name"]

    def slot(**changes):
        return {"id": DCP2, "where": "auto", "checkpoint": None, "entries": {}, "endpoint": "", **changes}
    plans = [{"size": 2, "slots": [slot()], "image": default, "setupOnly": False},
             {"size": 2, "slots": [slot(entries={"max_concurrency": ""})], "image": default, "setupOnly": False},
             {"size": 4, "slots": [slot()], "image": default, "setupOnly": False},
             {"size": 2, "slots": [slot()], "image": rollback, "setupOnly": False},
             {"size": 2, "slots": [slot(entries={"max_concurrency": "4", "api_bind": "10.0.0.5"})], "image": default, "setupOnly": False},
             {"size": 2, "slots": [slot(where="0,1")], "image": default, "setupOnly": False},
             {"size": 2, "slots": [], "image": default, "setupOnly": True}]
    found = engine("""
const { byId, images, aliases, plans } = value;
console.log(JSON.stringify(plans.map(plan => E.planQuery(plan, byId, images, aliases))));""",
                   {"byId": by_id, "images": public["images"], "aliases": public["aliases"], "plans": plans})
    assert found[:2] == [f"p={DCP2_SHORT}"] * 2
    assert all(query.startswith("plan=") for query in found[2:])
    # A plan link never carries the listen address.
    assert "api_bind" not in found[4] and "max_concurrency" in found[4]


class Captured(BaseException):
    """Raised in place of parsing, carrying the parser a command builds and the words it would parse."""

    def __init__(self, parser, words):
        super().__init__()
        self.parser, self.words = parser, list(words)


def command_parser(words, monkeypatch):
    """The argparse parser `sparkring WORDS` parses with, and the words it receives, without running the command."""
    import argparse
    import sys
    from runtime.host import controller
    from scripts import sparkring

    def capture(self, args=None, namespace=None):
        raise Captured(self, args)
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    monkeypatch.setattr(argparse.ArgumentParser, "parse_known_args", capture)
    try:
        # On Node A, the installed package routes up, down and status to the controller.
        if words[0] in ("up", "down", "status", "restart"):
            controller._lifecycle(words, output=sys.stdout, locks=None)
        else:
            sparkring._main(words)
    except Captured as captured:
        return captured.parser, captured.words
    raise AssertionError(f"sparkring {' '.join(words)} parsed no arguments")


def assert_real_command(command, monkeypatch):
    """Every subcommand, positional and --flag of ``command`` (``sudo sparkring ...``) is one its parser accepts."""
    import argparse
    import shlex
    words = shlex.split(command.split("#", 1)[0])
    words = words[words.index("sparkring") + 1:]
    parser, words = command_parser(words, monkeypatch)
    positionals = [a for a in parser._actions if not a.option_strings and not isinstance(a, argparse._SubParsersAction)]
    skip = False
    for word in words:
        if skip:
            skip = False
            continue
        if word.startswith("-"):
            action = parser._option_string_actions.get(word.split("=", 1)[0])
            assert action is not None, f"{command}: {word} is not an option of {parser.prog}"
            skip = action.nargs != 0 and "=" not in word
            continue
        sub = next((a for a in parser._actions if isinstance(a, argparse._SubParsersAction)), None)
        if sub is not None and word in sub.choices:
            parser = sub.choices[word]
            positionals = [a for a in parser._actions if not a.option_strings and not isinstance(a, argparse._SubParsersAction)]
            continue
        assert positionals, f"{command}: {parser.prog} takes no argument {word}"
        action = positionals.pop(0)
        assert action.choices is None or word in action.choices, f"{command}: {word} is not a choice of {parser.prog}"


def guide_commands():
    """Each `sparkring` command the guide shows in code, and each install flag of its flag table as an install command."""
    text = site.command_guide()
    commands = re.findall(r"^((?:sudo )?sparkring [^#\n]*)", text, re.M)
    commands += re.findall(r"`((?:sudo )?sparkring [^`]*)`", text)
    flags = site.section(text, "Common install flags")
    for cell in re.findall(r"^\| (`--.*?) \|", flags, re.M):
        commands += ["sparkring install " + flag for flag in re.findall(r"`([^`]+)`", cell)]
    return [c.strip() for c in commands]


def test_command_guide_shows_only_real_commands_and_flags(monkeypatch):
    commands = guide_commands()
    assert len(commands) >= 30
    for command in commands:
        assert_real_command(command, monkeypatch)
    with pytest.raises(AssertionError):
        assert_real_command("sudo sparkring install --no-such-flag", monkeypatch)
    with pytest.raises(AssertionError):
        assert_real_command("sudo sparkring dashboard quiet maybe", monkeypatch)


def test_command_guide_is_short_and_takes_its_table_from_the_reference():
    html = site.commands(META)
    main = html.split("<main", 1)[1].split("</main>", 1)[0]
    headings = re.findall(r"<h2[^>]*>(.*?)</h2>", main)
    assert headings == ["Everyday commands", "All commands", "Common install flags", "Tools", "Dashboard"]
    assert main.count('>Run these commands on Node A, the Spark connected to your network.<') == 1
    # Every row of the reference's All commands table, and each command it names, is on the page.
    reference = site.read(site.COMMANDS)
    rows = [line for line in site.section(reference, "All commands").splitlines() if line.startswith("| [")]
    assert len(rows) >= 20 and main.count("<tr>") >= len(rows) + 1
    for row in rows:
        for name in re.findall(r"\[`([^`]+)`\]", row.split("|")[1]):
            assert f"<code>{name}</code>" in main
    # Links into the reference open it on GitHub's main branch at a heading it has.
    url = site.reference_url(META)
    assert url == "https://github.com/FujitsuPolycom/sparkring/blob/main/docs/operations/commands.md"
    assert main.count(f'href="{url}">Full reference →</a>') == 2
    anchors = set(re.findall(r'id="([^"]+)"', site.prose(reference, site.COMMANDS, META)))
    linked = re.findall(rf'href="{re.escape(url)}#([^"]+)"', main)
    assert linked and all(anchor in anchors for anchor in linked)
    assert 'href="#' not in main
    # The reference's details stay there.
    for detail in ("Command highlighting", "display-name.json", "install.sh", "--control-cidr", "Fabric bandwidth, both directions",
                   "exit with 0", "viewer lease"):
        assert detail not in main, detail
    assert len(re.sub(r"<[^>]+>", "", main)) < 6000


def test_landing_context_and_kv_match_the_readme_installer_table(public):
    from runtime.common.profile_aliases import canonical
    from scripts.generate_profiles import INSTALLER_TABLE
    html = site.landing(public)
    head = html.split('id="profile-table"', 1)[1].split('</thead>', 1)[0]
    assert re.findall(r'<th scope="col">(.*?)</th>', head) == [
        "Model", "Variant", "Sparks", "--profile", "Context", "KV tokens", "Thinking", "Status"]
    table = html.split('id="profile-table"', 1)[1].split('</table>', 1)[0]
    cells = {re.search(r"<code>(.*?)</code>", row)[1]: re.findall(r'<td data-label="([^"]+)">(.*?)</td>', row)
             for row in table.split("<tr>")[2:] if "<code>" in row}
    lines = site.read("README.md").splitlines()
    header = next(n for n, line in enumerate(lines) if line.startswith(INSTALLER_TABLE))
    readme = [[c.strip() for c in line.strip().strip("|").split("|")] for line in lines[header + 2:]
              if line.startswith("| ")]
    readme = readme[:next((i for i, row in enumerate(readme) if len(row) != 7), len(readme))]
    assert len(readme) >= 10
    for row in readme:
        profile = next(p for p in public["profiles"] if p["id"] == canonical(row[3].strip("`")))
        landing = dict(cells[site.short_name(profile["id"], public["aliases"])])
        assert landing["Context"] == row[4]
        assert landing["KV tokens"] == (re.sub(r"\]\(.*\)$", "", row[5]).lstrip("[") if row[5] != "—" else "—")
        assert landing["Thinking"] == row[6] == profile["thinking"]
    # A profile without a recorded pool for its installed checkpoint shows a dash, never an estimate.
    unmeasured = next(p for p in public["profiles"] if p["id"] == "glm53-flash-csf-tp8")
    assert unmeasured["kv"] is None and dict(cells["glm53-flash-csf-tp8"])["KV tokens"] == "—"
