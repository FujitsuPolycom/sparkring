"""Offline browser regressions for arc labels, activity and cable visibility."""
import os
from pathlib import Path

import pytest

from runtime.host.test_dashboard_charts import smooth_page, update_page
from runtime.host.test_dashboard_layout import GEOMETRY
from runtime.host.test_dashboard_shared_style import style_fixture


def ring_fixture(value, layout):
    members, ring = style_fixture(value, layout == "two-halves")
    if layout == "stopped-tp4-tp2":
        members[:] = members[:3]
        for i, (member, positions, name) in enumerate(zip(members, [list(range(4)), [4, 5], [6, 7]],
                ["DeepSeek-V4.1", "Qwen3.8-Flash-Next", "GLM-5.3-Flash"])):
            member.update(name=f"tp{positions[0]}-{positions[-1]}", model=name, model_short=name,
                          state="serving" if i == 2 else "stopped", state_label="Serving" if i == 2 else "Stopped")
            member["arc"].update(positions=positions, first=positions[0], last=positions[-1], size=len(positions))
            member["layout"].update(tp=len(positions), count=len(positions), label=f"TP{len(positions)}")
            member["settings"]["tensor_parallel_size"] = len(positions)
            member["head"]["position"] = positions[0]
            member["sparks"] = [dict(name=f"spark-{p}", position=p, up=True if i == 2 or p == 0 else None)
                                for p in positions]
            if i == 2:
                member["settings"]["decode_context_parallel_size"] = 2
        for p in ring["positions"]:
            # Local controller is up even when its recorded model is stopped.
            p["up"] = True if p["position"] in (0, 6, 7) else None
    for p in ring["positions"]:
        member = next(m for m in members if p["position"] in m["arc"]["positions"])
        p.update(member=member["id"], members=[member["id"]])
    return members, ring


SECTORS = """() => {
  const failures = [];
  for (const ring of data.rings) {
    const svg = [...document.querySelectorAll('.ringfig')].find(e => e.dataset.ring === ring.id).querySelector('svg');
    for (const member of data.groups.flatMap(g => g.members).filter(m => m.ring === ring.id)) {
      const label = [...svg.querySelectorAll('.band')].find(e => e.dataset.member === member.id).querySelector('.lbl');
      const b = label.getBBox(), dx = b.x + b.width / 2 - 200, dy = b.y + b.height / 2 - 190;
      const angle = (Math.atan2(dy, dx) + Math.PI / 2 + 2 * Math.PI) % (2 * Math.PI);
      const step = 2 * Math.PI / ring.size;
      const start = (member.arc.first - .5) * step;
      const offset = (angle - start + 2 * Math.PI) % (2 * Math.PI);
      if (offset > member.arc.size * step || Math.hypot(dx, dy) < 25 || Math.hypot(dx, dy) >= 120)
        failures.push({name: member.model, angle, start, offset, dx, dy});
    }
  }
  return failures;
}"""


@pytest.mark.parametrize("layout", ["stopped-tp4-tp2", "four-pairs", "two-halves"])
@pytest.mark.parametrize("theme", ["dark", "light"])
@pytest.mark.parametrize("width", [1400, 375])
def test_labels_follow_arc_sectors(smooth_page, layout, theme, width):
    page, value, requests = smooth_page
    page.set_viewport_size({"width": width, "height": 1000})
    page.emulate_media(color_scheme=theme, reduced_motion="reduce")
    ring_fixture(value, layout)
    update_page(page)
    output = os.environ.get("SPARKRING_RING_SCREENSHOTS")
    if output:
        directory = Path(output)
        directory.mkdir(parents=True, exist_ok=True)
        page.locator("#ringpanel").screenshot(path=str(directory / f"{layout}-{theme}-{width}.png"))
    assert page.evaluate(SECTORS) == []
    assert page.evaluate(GEOMETRY) == []
    before = list(requests)
    page.evaluate("window.ringNodes = [...document.querySelectorAll('.node, .band, .cable, .traffic, .allocation .slot')]; render()")
    assert page.evaluate("ringNodes.every(e => e.isConnected)")
    assert requests == before
    for member in value["groups"][0]["members"]:
        member["live"]["output_tps"] += 1
    update_page(page)
    assert page.evaluate("ringNodes.every(e => e.isConnected)")
    assert requests[:-1] == before and "/api/status" in requests[-1]


def test_stopped_activity_and_visibility(smooth_page):
    page, value, _ = smooth_page
    ring_fixture(value, "stopped-tp4-tp2")
    update_page(page)
    tile = page.locator(".tile").filter(has=page.get_by_text("Sparks up", exact=True))
    assert "2 busy" in tile.inner_text()
    assert "1 idle" in tile.inner_text()
    assert "3/ 8" in tile.inner_text()
    assert "5 unobserved" in tile.inner_text()
    assert "2 of 8 Sparks busy" in page.locator("#summary").inner_text()
    assert page.locator(".allocation .dim").count() == 6
    assert page.locator(".allocation .dim").evaluate_all("els => els.every(e => Number(getComputedStyle(e).opacity) < 1)")
    assert page.locator(".node.dim rect").evaluate_all("els => els.every(e => getComputedStyle(e).strokeDasharray === 'none')")
    assert page.locator(".cable").evaluate_all("els => els.every(e => parseFloat(getComputedStyle(e).strokeWidth) >= 7.5)")
    assert page.locator(".cable-band").count() == 5
    assert page.locator(".node.serving").count() == 2
    assert page.locator(".node.serving rect").first.evaluate("e => getComputedStyle(e).filter !== 'none'")
    assert page.locator(".traffic").count() == 2          # one dot each way on the serving ring cable
    assert page.locator(".traffic.rev").count() == 1
    dot = page.locator(".traffic").first
    assert dot.evaluate("e => getComputedStyle(e).animationName") != "none"
    page.emulate_media(reduced_motion="reduce")
    assert dot.evaluate("e => getComputedStyle(e).animationName") == "none"
    point = dot.bounding_box()
    page.wait_for_timeout(100)
    assert dot.bounding_box() == point
    page.evaluate("window.savedSlots = [...document.querySelectorAll('.allocation .slot')]")
    members = value["groups"][0]["members"]
    members[0].update(state="serving", state_label="Serving")
    update_page(page)
    assert page.evaluate("savedSlots.every(e => e.isConnected)")
    assert page.locator(".allocation .dim").count() == 2
    assert "6 busy" in tile.inner_text()
    members[0].update(state="stopped", state_label="Stopped")
    update_page(page)
    assert page.locator(".allocation .dim").count() == 6
    assert "2 busy" in tile.inner_text()


def test_full_ring_label_center_and_wrapped_arc(smooth_page):
    page, value, _ = smooth_page
    members, ring = ring_fixture(value, "two-halves")
    members[:] = members[:1]
    members[0]["arc"].update(first=0, size=8, positions=list(range(8)))
    update_page(page)
    assert page.locator(".lbl").evaluate("e => {const b=e.getBBox(); return Math.abs(b.x+b.width/2-200)<2 && Math.abs(b.y+b.height/2-190)<3;}")
    members[0]["arc"].update(first=7, size=2, positions=[7, 0])
    update_page(page)
    assert page.evaluate(SECTORS) == []
    assert page.evaluate(GEOMETRY) == []


def test_active_allocation_wins_over_stopped_record(smooth_page):
    page, value, _ = smooth_page
    members, _ = ring_fixture(value, "stopped-tp4-tp2")
    members[2]["arc"].update(first=0, positions=[0, 1])
    update_page(page)
    assert page.locator(".allocation .slot").nth(0).get_attribute("data-member") == members[2]["id"]
    assert "dim" not in page.locator(".allocation .slot").nth(0).get_attribute("class")
    assert page.evaluate("ringCounts(data.rings, data.groups[0].members).busy") == 2
