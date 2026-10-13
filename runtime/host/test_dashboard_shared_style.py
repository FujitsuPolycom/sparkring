"""Shared visual tokens, polygon geometry and offline screenshot fixtures."""
from copy import deepcopy
import os
from pathlib import Path

import pytest

from runtime.host import dashboard_tui, test_dashboard_charts
from runtime.host.test_dashboard import PAIRS, collector
from runtime.host.test_dashboard_layout import GEOMETRY

smooth_page = test_dashboard_charts.smooth_page
update_page = test_dashboard_charts.update_page


def style_fixture(value, halves):
    members = value["groups"][0]["members"]
    ring = value["rings"][0]
    if halves:
        members[:] = members[:2]
    names = ["GLM-5.3", "Qwen3.8-Flash-Next", "MiMo-V2.6", "DeepSeek-V4.1"]
    for i, member in enumerate(members):
        positions = list(range(i * (4 if halves else 2), (i + 1) * (4 if halves else 2)))
        member.update(name=f"tp{positions[0]}-{positions[-1]}", model_short=names[i],
                      model=f"{names[i]}-TP{len(positions)}", state="serving", state_label="Serving",
                      state_since=value["generated_at"] - 3240)
        member["arc"].update(positions=positions, first=positions[0], last=positions[-1],
                             size=len(positions), color=i)
        member["layout"].update(tp=len(positions), count=len(positions), label=f"TP{len(positions)}")
        member["settings"]["tensor_parallel_size"] = len(positions)
        address = "192.0.2.50" if positions[0] == 0 else f"192.0.2.{10 + positions[0]}"
        member["head"].update(position=positions[0], address=address, name=f"spark-{positions[0]}")
        member["endpoint"] = f"http://{address}:{member['head']['port']}/v1"
        member["sparks"] = [dict(name=f"spark-{p}", position=p, up=True) for p in positions]
        member["live"].update(output_tps=42.4 + i * 9, prefill_tps=1250.4, kv=.256, running=2)
        for j, sample in enumerate(member["history"]):
            sample[1] = 42.4 + i * 9 + (j % 5 - 2) * 2
            sample[5] = 1250.4 + (j % 4 - 2) * 40
    for p, short in zip(ring["positions"], ["edfd", "ebb8", "ebee", "931e", "aa42", "3286", "0a0f", "4a87"]):
        p.update(short=short, up=True)
    ring["discovery"]["at"] = value["generated_at"]
    return members, ring


COLORS = """() => [...document.querySelectorAll('article.card')].map(card => {
  const id = card.dataset.member;
  const find = selector => [...document.querySelectorAll(selector)].find(e => e.dataset.member === id);
  const css = (el, property) => getComputedStyle(el).getPropertyValue(property);
  return [css(card, 'border-left-color'), css(find('.node').querySelector('rect'), 'stroke'),
    css(find('.allocation .slot'), 'background-color'), css(find('.cable.in'), 'stroke'),
    ...[...card.querySelectorAll('.chart path')].map(p => css(p, 'stroke'))];
})"""


@pytest.mark.parametrize("width", [1400, 375])
@pytest.mark.parametrize("theme", ["dark", "light"])
@pytest.mark.parametrize("halves", [True, False], ids=["two-halves", "four-pairs"])
def test_shared_colors_geometry_and_screenshots(smooth_page, width, theme, halves):
    page, value, requests = smooth_page
    page.set_viewport_size({"width": width, "height": 1000})
    page.emulate_media(color_scheme=theme)
    members, ring = style_fixture(value, halves)
    update_page(page)
    page.wait_for_timeout(200)  # Let border-colour transitions finish before comparing pixels.
    expected = (["rgb(63, 195, 207)", "rgb(116, 212, 148)", "rgb(172, 142, 246)", "rgb(122, 166, 246)"]
                if theme == "dark" else
                ["rgb(11, 127, 138)", "rgb(47, 138, 74)", "rgb(107, 91, 181)", "rgb(58, 100, 181)"])
    assert page.evaluate(COLORS) == [[color] * 7 for color in expected[:len(members)]]
    assert page.evaluate(GEOMETRY) == []
    assert page.locator(".node rect").count() == 8
    assert page.locator(".node .node-dot").count() == 8
    assert page.locator(".allocation .slot").all_inner_texts() == list("01234567")
    assert page.locator(".relay").count() == (6 if halves else 0)
    assert all("A" not in d for d in page.locator(".cable").evaluate_all("els => els.map(e => e.getAttribute('d'))"))
    assert "42 tok/s" in page.locator(".lrate").first.text_content()
    assert ("TP4" if halves else "TP2") in page.locator(".l2").first.text_content()
    output = os.environ.get("SPARKRING_STYLE_SCREENSHOTS")
    if output:
        directory = Path(output)
        directory.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(directory / f"{'two-halves' if halves else 'four-pairs'}-{theme}-{width}.png"), full_page=True)
    before = list(requests)
    page.evaluate("window.styleNodes = [...document.querySelectorAll('.node, .band, .allocation .slot, .card, .chart svg')]; render()")
    assert page.evaluate("styleNodes.every(e => e.isConnected)")
    for i, card in enumerate(page.locator("article.card").all()):
        card.locator("[data-graph='4']").click()
        assert page.locator("#graph-body path").evaluate("e => getComputedStyle(e).stroke") == expected[i]
        page.keyboard.press("Escape")
    assert requests == before


def test_polygon_sizes_pair_cables_and_separate_rings(smooth_page):
    page, value, _ = smooth_page
    members, ring = style_fixture(value, True)
    original = deepcopy(ring["positions"])
    members[:] = members[:1]
    for size in range(2, 9):
        ring.update(size=size, positions=original[:size], closed=size != 2)
        members[0]["arc"].update(positions=list(range(size)), size=size)
        update_page(page)
        assert page.locator(".node rect").count() == size
        assert page.locator(".cable").count() == size
        if size == 2:
            assert page.locator(".cable").nth(0).get_attribute("d") != page.locator(".cable").nth(1).get_attribute("d")
        assert page.evaluate(GEOMETRY) == []
    second = deepcopy(ring)
    second["id"] = "second-ring"
    for r in (ring, second):
        r.update(size=4, positions=original[:4])
    members[0]["arc"].update(positions=list(range(4)), size=4)
    clone = deepcopy(members[0])
    clone.update(id="second-member", ring=second["id"])
    clone["arc"]["color"] = 1
    members.append(clone)
    value["rings"].append(second)
    update_page(page)
    assert page.locator(".ringfig").count() == 2
    assert page.evaluate(GEOMETRY) == []


def test_muted_unmanaged_and_dim_stopped(smooth_page):
    page, value, _ = smooth_page
    members, _ = style_fixture(value, False)
    members[0]["unmanaged"] = True
    members[1].update(unmanaged=True, gpu_job=True)
    members[2].update(state="stopped", state_label="Stopped")
    update_page(page)
    assert page.locator(".node.outline rect").count() == 4
    assert page.locator(".allocation .outline").count() == 4
    assert page.locator("article.outline").count() == 2
    assert page.locator(".node.dim").count() == 2
    assert page.locator("article.st-stopped").evaluate("e => Number(getComputedStyle(e).opacity) < 1")
    assert page.locator(".allocation .outline").first.evaluate("e => getComputedStyle(e).backgroundColor") == "rgba(0, 0, 0, 0)"
    assert page.evaluate(GEOMETRY) == []


def test_collector_colors_follow_lowest_position_and_terminal_palette(tmp_path):
    found, _, _ = collector(tmp_path, rows=tuple(reversed(PAIRS)))
    found.cycle()
    members = found.snapshot()["groups"][0]["members"]
    assert [(min(m["arc"]["positions"]), m["arc"]["color"]) for m in members] == [(0, 0), (2, 1), (4, 2), (6, 3)]
    assert [dashboard_tui.group_color(m) for m in members] == [dashboard_tui.INFO, dashboard_tui.GOOD,
                                                             dashboard_tui.ACCENT, dashboard_tui.BLUE]
    assert dashboard_tui.group_color({"unmanaged": True, "arc": {"color": 0}}) == dashboard_tui.MUTED
