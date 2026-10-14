"""Spark details from local cache/startup fixtures; browsers intercept every request."""
import json
import os
from pathlib import Path
import time

import pytest

from runtime.host import dashboard, dashboard_admin, startup
from runtime.host.test_dashboard import collector, run_for
from runtime.host.test_dashboard_charts import smooth_page  # noqa: F401
from runtime.host.test_dashboard_startup import observation


def details_fixture(tmp_path):
    found, _, clock = collector(tmp_path)
    run_for(found, clock, 10, viewer=True)
    model = next(iter(found.models.values()))
    model.row["deployment_id"] = "details"
    model.runtime["settings"]["decode_context_parallel_size"] = 2
    model.state = "loading"
    found.live_startup = {"details": [
        observation("loading", clock.now, rank=0, done=12, total=43,
                    container_state="running", health="starting"),
        observation("failed", clock.now, rank=1, error="RuntimeError: CUDA out of memory",
                    container_state="exited", health="unhealthy")]}
    position = found.sources[0].fabric["positions"][0]
    links = [{"netdev": p["functions"]["primary"]["netdev"], "carrier": i == 0,
              "operstate": "UP" if i == 0 else "DOWN"} for i, p in enumerate(position["ports"].values())]
    found.node_reports = [{"node_id": position["node_id"], "observed_at": clock.now - 60,
                           "gpu": {"memory_used_bytes": 88.4e9, "memory_total_bytes": 128e9,
                                   "temperature_c": 62.4, "private": "secret"}, "links": links,
                           "containers": [{"name": "sr-0-r0", "state": "running", "health": "healthy"}]}]
    value = found.snapshot()
    value.update(generated_at=time.time(), poll_seconds=3600)
    return found, value


def test_details_projection_identity_age_health_and_allowlist(tmp_path):
    found, value = details_fixture(tmp_path)
    pos = value["rings"][0]["positions"]
    assert pos[0]["gpu"] == dict(memory_used_bytes=88.4e9, memory_total_bytes=128e9, temperature_c=62.4)
    assert pos[0]["observed_at"] == found.clock() - 60
    assert pos[1]["gpu"] == {} and pos[1]["observed_at"] is None
    assert len(pos[0]["ports"]) == 2 and len(pos[0]["links"]) == 2
    ranks = value["groups"][0]["members"][0]["spark_startup"]["ranks"]
    assert ranks[1]["health"] == "unhealthy" and ranks[1]["position"] == 1
    projected = startup.project(found.live_startup, ["details"], found.clock())
    assert projected["details"][0]["container_state"] == "running"
    # Serving hides the group startup banner but keeps individual container facts.
    next(iter(found.models.values())).state = "serving"
    member = found.snapshot()["groups"][0]["members"][0]
    assert member["startup"] is None and member["spark_startup"]["ranks"] == ranks
    state = tmp_path / "controller"
    state.mkdir()
    state.joinpath("fabric.json").write_bytes(found.sources[0].fabric_path.read_bytes())
    doc = found.node_reports[0]
    doc["control"] = {"peers": doc.pop("links"), "secret": "private"}
    rows = dashboard_admin.cached_node_rows(state, tmp_path / "cache", {"host": doc}, found.clock())
    assert rows[0]["links"] == pos[0]["links"] and rows[0]["gpu"] == pos[0]["gpu"]
    assert "secret" not in json.dumps(rows) and "private" not in json.dumps(rows)
    # A different identity at the same array position cannot lend its measurements.
    doc["node_id"] = "different-node"
    assert found.snapshot()["rings"][0]["positions"][0]["gpu"] == {}


def test_details_health_uses_existing_container_inspect_without_extra_reads():
    requests = []
    state = {"Status": "running", "StartedAt": "2026-10-14T10:00:00Z", "Health": {"Status": "healthy"}}

    def get(path):
        requests.append(path)
        if path.startswith("/containers/json?"):
            return json.dumps([{"Id": "a" * 64, "Labels": {"io.sparkring.deployment": "details", "io.sparkring.rank": "0"}}]).encode()
        assert path.endswith("/json")
        return json.dumps({"State": state}).encode()

    reader = startup.Reader(get)
    now = startup.stamp("2026-10-14T10:01:00Z")
    row = reader.read(now)["details"][0]
    assert row["container_state"] == "running" and row["health"] == "healthy"
    assert len(requests) == 2
    state["Health"] = {}
    # A missing health check on a later sample must not retain 'healthy'.
    reader.get = lambda path: b"" if "/logs?" in path else get(path)
    row = startup.project(reader.read(now + 1), ["details"], now + 1)["details"][0]
    assert "health" not in row


@pytest.mark.parametrize("width", [1400, 375])
@pytest.mark.parametrize("theme", ["dark", "light"])
@pytest.mark.parametrize("touch", [False, True])
def test_details_content_interaction_and_geometry(smooth_page, tmp_path, width, theme, touch):  # noqa: F811
    original, _, _ = smooth_page
    _, value = details_fixture(tmp_path)
    page_bytes, csp = dashboard.page_bytes()
    context = original.context.browser.new_context(viewport={"width": width, "height": 900},
                                                   color_scheme=theme, has_touch=touch)
    page = context.new_page()
    requests, errors = [], []
    page.on("pageerror", lambda error: errors.append(str(error)))

    def route(request):
        url = request.request.url
        requests.append(url)
        if "/api/status" in url:
            request.fulfill(json=value)
        elif url.endswith("/api/history"):
            request.fulfill(json={})
        elif url == "http://127.0.0.1:8090/":
            request.fulfill(body=page_bytes, content_type="text/html", headers={"Content-Security-Policy": csp})
        else:
            request.abort()

    page.route("**/*", route)
    page.goto("http://127.0.0.1:8090/")
    nodes = page.locator("#ringviz .node")
    nodes.first.wait_for()
    before = list(requests)
    card = page.get_by_role("tooltip")
    if touch:
        nodes.first.tap()
    else:
        nodes.first.hover()
    assert card.is_visible()
    content = card.inner_text()
    for text in ("spark0", "192.0.2.10", "Position", "Group", "Rank", "TP2 / DCP2",
                 "Loading weights 12/43", "running · starting", "88 / 128 GB", "62 °C",
                 "Port 0", "Port 1", "carrier up", "carrier down", "Updated", "Node cache"):
        assert text in content
    assert "unknown" not in content.lower()
    assert page.evaluate("""() => {
      const el = document.querySelector('#spark-details'), b = el.getBoundingClientRect();
      return b.left >= 8 && b.right <= innerWidth - 8 && b.top >= 8 && b.bottom <= innerHeight - 8
        && el.scrollWidth <= el.clientWidth && document.documentElement.scrollWidth <= innerWidth;
    }""")
    output = os.environ.get("SPARKRING_DETAILS_SCREENSHOTS")
    if output and touch == (width == 375):
        Path(output).mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(Path(output) / f"details-{theme}-{width}.png"))
    if touch:
        nodes.first.tap()
        assert not card.is_visible()
        nodes.first.tap()
        assert card.is_visible()
        page.locator("h1").tap()
        assert not card.is_visible()
    else:
        # The pointer can cross the gap and remain over the card to read it.
        card.hover()
        page.wait_for_timeout(200)
        assert card.is_visible()
        page.locator("h1").hover()
        page.wait_for_timeout(200)
        assert not card.is_visible()
    nodes.first.focus()
    assert card.is_visible()
    page.keyboard.press("Escape")
    assert not card.is_visible()
    assert nodes.first.evaluate("el => el === document.activeElement")
    page.keyboard.press("Tab")
    assert card.is_visible() and "CUDA out of memory" in card.inner_text()
    assert "GPU memory" not in card.inner_text() and "GPU temp" not in card.inner_text()
    page.keyboard.press("Escape")
    page.keyboard.press("Enter")
    assert card.is_visible()
    page.keyboard.press("Enter")
    assert not card.is_visible()
    nodes.first.focus()
    page.evaluate("render()")
    assert card.is_visible() and nodes.first.evaluate("el => el === document.activeElement")
    # All positions, including the right and bottom edges, clamp on a short viewport.
    page.set_viewport_size({"width": width, "height": 420})
    page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
    for i in range(nodes.count()):
        nodes.nth(i).focus()
        box = card.bounding_box()
        assert 7 <= box["x"] and box["x"] + box["width"] <= width - 7
        assert 7 <= box["y"] and box["y"] + box["height"] <= 413
        trigger = nodes.nth(i).bounding_box()
        assert box["y"] + box["height"] <= trigger["y"] or trigger["y"] + trigger["height"] <= box["y"]
    if touch:
        nodes.first.tap()
        assert card.is_visible()
        nodes.first.tap()
        assert not card.is_visible()
    assert requests == before and errors == []
    context.close()


@pytest.mark.parametrize("width", [1400, 375])
def test_details_escape_every_field_unknowns_refresh_and_removal(smooth_page, width):  # noqa: F811
    page, _, requests = smooth_page
    page.set_viewport_size({"width": width, "height": 900})
    payload = '<img src=x onerror="window.injected=1"> & \'test\''
    result = page.evaluate("""payload => {
      const ring = {positions: [{position: 0, name: payload, address: payload, ports: {
        0: {cable: payload, peer: {position: payload, port: payload}, functions: {primary: {netdev: payload}}}
      }, links: [{netdev: payload, operstate: payload}]}]};
      return sparkDetails(ring, 0, {name: payload, model: payload,
        startup: {ranks: [{position: 0, phase: 'failed', label: payload, error: payload,
          container_state: payload, health: payload}]}});
    }""", payload)
    page.locator("#spark-details").evaluate("(el, html) => { el.innerHTML = html; el.hidden = false; }", result)
    assert page.locator("#spark-details img").count() == 0 and page.evaluate("window.injected") is None
    assert payload in page.locator("#spark-details").inner_text()
    unknown = page.evaluate("sparkDetails({positions: [{position: 0}]}, 0, {})")
    assert "unknown" not in unknown.lower() and "Model" not in unknown and "Rank" not in unknown
    node = page.locator("#ringviz .node").first
    node.focus()
    before = list(requests)
    page.evaluate("data.groups[0].members[0].model = '<script>' + 'x'.repeat(2000); render()")
    assert page.locator("#spark-details").evaluate("el => el.scrollWidth <= el.clientWidth")
    box = page.locator("#spark-details").bounding_box()
    assert box["x"] >= 8 and box["x"] + box["width"] <= width - 8
    assert box["y"] >= 8 and box["y"] + box["height"] <= 892
    page.evaluate("data.groups[0].members[0].model = 'Replacement'; render()")
    assert "Replacement" in page.locator("#spark-details").inner_text()
    page.evaluate("data.rings = []; data.ring = null; render()")
    assert not page.locator("#spark-details").is_visible()
    assert requests == before


def test_details_ring_identity_and_unobserved_rank(smooth_page):  # noqa: F811
    page, _, requests = smooth_page
    before = list(requests)
    page.evaluate("""() => {
      const ring = structuredClone(data.rings[0]);
      ring.id = 'second-ring'; ring.name = 'Second ring'; ring.positions[0].name = 'second-host';
      data.rings.push(ring);
      const member = structuredClone(data.groups[0].members[0]);
      member.id = 'second-member'; member.ring = ring.id; member.model = 'Second model';
      member.state = 'loading'; member.nodes = [];
      member.startup = {phase: 'loading', label: 'Loading weights', done: 3, total: 43,
        ranks: [{position: 1, rank: 1, phase: 'loading', label: 'Loading weights', done: 3, total: 43}]};
      data.groups.push({kind: 'ring', members: [member]}); render();
    }""")
    page.locator('[data-ring="second-ring"] .node').first.focus()
    card = page.locator("#spark-details")
    assert "second-host" in card.inner_text() and "Second model" in card.inner_text()
    assert "Loading weights" not in card.inner_text() and "Rank" not in card.inner_text()
    page.locator('[data-ring="second-ring"] .node').nth(1).focus()
    assert "Loading weights 3/43" in card.inner_text()
    assert requests == before
