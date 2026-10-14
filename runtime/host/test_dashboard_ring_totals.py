"""Offline ring totals, snapshot consistency and collision-free placement."""
from copy import deepcopy
import os
from pathlib import Path

import pytest

from runtime.host.test_dashboard_charts import smooth_page, update_page  # noqa: F401
from runtime.host.test_dashboard_layout import GEOMETRY
from runtime.host.test_dashboard_shared_style import style_fixture


def totals_fixture(value, size=8, count=3):
    members, ring = style_fixture(value, False)
    members[:] = members[:count]
    ring.update(size=size, positions=ring["positions"][:size], closed=size != 2, runs=[])
    for i, member in enumerate(members):
        # A pair can retain a stopped deployment alongside its two live groups.
        positions = list(range(i * size // count, (i + 1) * size // count)) or [0]
        member["arc"].update(positions=positions, first=positions[0], last=positions[-1], size=len(positions))
        member['name'] = f"tp{positions[0]}-{positions[-1]}"
        member["layout"].update(tp=len(positions), count=len(positions), label=f"TP{len(positions)}")
        member["settings"]["tensor_parallel_size"] = len(positions)
        member["sparks"] = [dict(name=f"spark-{p}", position=p, up=True) for p in positions]
        member["live"].update(output_tps=[20.2, 31.4, 99.9][i], running=[1, 2, 9][i], waiting=[0, 4, 8][i])
        if size == 2 and count == 3 and i == 0:
            member.update(state="stopped", state_label="Stopped")
    return members, ring


TOTALS_GEOMETRY = """() => {
  const failures = [];
  const overlap = (a, b) => Math.min(a.right, b.right) > Math.max(a.left, b.left) + 1 &&
    Math.min(a.bottom, b.bottom) > Math.max(a.top, b.top) + 1;
  for (const fig of document.querySelectorAll('.ringfig')) {
    const total = fig.querySelector('.ring-totals'), b = total.getBoundingClientRect();
    const f = fig.getBoundingClientRect();
    if (b.left < f.left - 1 || b.right > f.right + 1) failures.push('totals overflow');
    for (const other of fig.querySelectorAll('.lbl, .node, .allocation, .configuration'))
      if (overlap(b, other.getBoundingClientRect())) failures.push('totals overlap: ' + other.textContent);
    if (!total.closest('svg') && total.previousElementSibling.className !== 'allocation')
      failures.push('fallback must follow allocation');
  }
  return failures;
}"""


@pytest.mark.parametrize("width", [1400, 375])
@pytest.mark.parametrize("size", [2, 4, 8])
@pytest.mark.parametrize("count", [1, 2, 3])
def test_totals_geometry(smooth_page, width, size, count):  # noqa: F811
    page, value, _ = smooth_page
    page.set_viewport_size({"width": width, "height": 1000})
    page.emulate_media(color_scheme="dark", reduced_motion="reduce")
    totals_fixture(value, size, count)
    update_page(page)
    assert page.locator('.ring-totals').count() == 1
    assert page.evaluate(TOTALS_GEOMETRY) == []
    assert page.evaluate(GEOMETRY) == []
    output = os.environ.get("SPARKRING_TOTALS_SCREENSHOTS")
    if output:
        directory = Path(output)
        directory.mkdir(parents=True, exist_ok=True)
        page.locator("#ringpanel").screenshot(path=str(directory / f"ring-{size}-groups-{count}-{width}.png"))


def assert_totals(page, output, running, queued):
    block = page.locator('.ring-totals').first
    assert f"{output} tok/s" in block.text_content()
    assert f"{running} running · {queued} queued" in block.text_content()
    assert page.locator('#strip .tile').nth(1).locator('.v').inner_text() == f"{output}tok/s"
    requests = page.locator('#strip .tile').nth(2)
    assert requests.locator('.v').inner_text() == f"{running}running"
    assert requests.locator('.s').inner_text() == f"{queued} waiting"


def test_totals_share_card_snapshot_and_exclude_nonserving(smooth_page):  # noqa: F811
    page, value, requests = smooth_page
    members, _ = totals_fixture(value)
    members[2].update(state="starting", state_label="Starting")
    update_page(page)
    assert_totals(page, 52, 3, 4)  # Sum raw card rates, then round once.
    assert page.evaluate("""() => {
      const members = data.groups.flatMap(g => g.members);
      const t = servingTotals([...members, ...members]);
      return [Math.round(t.tps), t.running, t.waiting];
    }""") == [52, 3, 4]
    assert page.locator('article.card').nth(0).locator('.tiles .v').first.inner_text() == "20"
    assert page.locator('article.card').nth(1).locator('.tiles .v').first.inner_text() == "31"
    assert "tok/s" not in page.locator('#summary').inner_text()
    before = list(requests)
    page.evaluate("render()")
    assert requests == before
    members[0]['live'].update(output_tps=44.4, running=5, waiting=1)
    members[1]['unmanaged'] = True  # Discovered serving APIs count too.
    update_page(page)
    assert_totals(page, 76, 7, 5)
    assert page.locator('article.card').nth(0).locator('.tiles .v').first.inner_text() == "44"
    for m in members:
        m.update(state="stopped", state_label="Stopped")
    update_page(page)
    assert_totals(page, 0, 0, 0)
    members[0].update(state="serving", live=None)
    update_page(page)
    assert_totals(page, 0, 0, 0)
    members.clear()
    update_page(page)
    assert_totals(page, 0, 0, 0)
    assert page.locator('svg .ring-totals').count() == 1


@pytest.mark.parametrize("width", [1400, 375])
@pytest.mark.parametrize("theme", ["dark", "light"])
def test_center_fallback_and_per_ring_scope(smooth_page, width, theme):  # noqa: F811
    page, value, _ = smooth_page
    page.set_viewport_size({"width": width, "height": 1000})
    page.emulate_media(color_scheme=theme, reduced_motion="reduce")
    members, ring = totals_fixture(value, count=2)
    for member, positions in zip(members, [[0, 1], [4, 5]]):
        member['arc'].update(positions=positions, first=positions[0], last=positions[-1], size=2)
        member.update(name=f"tp{positions[0]}-{positions[-1]}")
        member['layout'].update(tp=2, count=2, label="TP2")
        member['settings']['tensor_parallel_size'] = 2
        member['sparks'] = [dict(name=f"spark-{p}", position=p, up=True) for p in positions]
    update_page(page)
    assert page.locator('svg .ring-totals').count() == 1
    assert page.evaluate(TOTALS_GEOMETRY) == []
    output = os.environ.get("SPARKRING_TOTALS_SCREENSHOTS")
    if output:
        directory = Path(output)
        directory.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(directory / f"center-{theme}-{width}.png"), full_page=True)
    # Large counters must switch to the wrapping block below the strip.
    members[0]['live'].update(output_tps=123456789, running=123456789, waiting=123456789)
    update_page(page)
    assert page.locator('svg .ring-totals').count() == 0
    assert page.evaluate(TOTALS_GEOMETRY) == []
    page.set_viewport_size({"width": 375, "height": 1000})
    assert page.evaluate(TOTALS_GEOMETRY) == []
    members[0]['live'].update(output_tps=20.2, running=1, waiting=0)
    other = deepcopy(ring)
    other.update(id="second-ring", name="Second ring")
    value['rings'].append(other)
    members[1]['ring'] = other['id']
    update_page(page)
    blocks = page.locator('.ring-totals').all_text_contents()
    assert "20 tok/s" in blocks[0] and "1 running · 0 queued" in blocks[0]
    assert "31 tok/s" in blocks[1] and "2 running · 4 queued" in blocks[1]
    assert page.locator('#strip .tile').nth(1).locator('.v').inner_text() == "52tok/s"
    assert page.evaluate(TOTALS_GEOMETRY) == []
