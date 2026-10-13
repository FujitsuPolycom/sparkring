"""Offline browser interactions; all HTTP responses are intercepted fixtures."""
import json
import os
from pathlib import Path
import time

import pytest

from runtime.host import dashboard
from runtime.host.test_dashboard import collector, run_for


@pytest.mark.parametrize("theme,touch", [("dark", False), ("light", True)])
def test_graph_hover_keyboard_touch_and_expansion_make_no_request(theme, touch, tmp_path):
    browser_api = pytest.importorskip("playwright.sync_api")
    found, _, clock = collector(tmp_path)
    run_for(found, clock, 10, viewer=True)
    value = json.loads(json.dumps(found.snapshot()))
    now = time.time()
    value.update(generated_at=now, poll_seconds=60)
    for group in value["groups"]:
        for member in group["members"]:
            member.pop("counters", None)
            member.pop("counter_samples", None)
            member["state"] = "serving"
            member["live"].update(source="metrics", prefill_tps=123.456, read_at=now)
            member["history"] = [[now - (72 - i) * 180, i + .123456, 1, 0, i / 100, 123.456, "metrics"]
                                 for i in range(73)]
    page_bytes, csp = dashboard.page_bytes()
    with browser_api.sync_playwright() as api:
        try:
            browser = api.chromium.launch(headless=True)
        except browser_api.Error:
            try:
                browser = api.chromium.launch(headless=True, channel="chrome")
            except browser_api.Error:
                pytest.skip("Chromium/Chrome unavailable for offline browser test")
        context = browser.new_context(color_scheme=theme, has_touch=touch,
                                      viewport={"width": 430 if touch else 1400, "height": 1000})
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
        expand = page.get_by_role("button", name="Expand Output tok/s", exact=False).first
        expand.wait_for()
        before = list(requests)
        page.locator("[data-graph='1'] svg").first.evaluate("el => window.originalGraph = el")
        page.evaluate("receivedAt -= 3; render()")
        assert page.locator("[data-graph='1'] svg").first.evaluate("el => el === window.originalGraph")
        if touch:
            expand.tap()
        else:
            svg = page.locator("[data-graph='1'] svg").first
            svg.hover(position={"x": 20, "y": 20})
            assert "Output tok/s:" in page.locator("[data-graph='1'] output").first.inner_text()
            expand.focus()
            page.keyboard.press("Enter")
        dialog = page.get_by_role("dialog")
        assert dialog.is_visible()
        slider = page.get_by_role("slider")
        slider.focus()
        page.keyboard.press("Home")
        assert "Output tok/s: 0 " in page.locator("#graph-value").inner_text()
        page.keyboard.press("End")
        assert "Output tok/s: 72 " in page.locator("#graph-value").inner_text()
        assert page.locator("#graph-body svg").bounding_box()["height"] >= 250
        # Updating cards must preserve the dialog's selected snapshot and focus.
        page.evaluate("render()")
        assert slider.evaluate("el => el === document.activeElement")
        output = os.environ.get("SPARKRING_CHART_SCREENSHOTS")
        if output:
            Path(output).mkdir(parents=True, exist_ok=True)
            page.wait_for_timeout(500)  # Let Chromium's touch highlight fade.
            page.screenshot(path=str(Path(output) / f"charts-{theme}.png"))
        page.keyboard.press("Escape")
        assert not dialog.is_visible()
        browser_api.expect(expand).to_be_focused()
        gaps = page.evaluate("""() => chart([[0, 2], [5, null], [10, 3]], 1, 10, "c1").svg""")
        assert "L" not in gaps.split('d="')[1].split('"')[0]
        assert requests == before and errors == []
        context.close()
        browser.close()


@pytest.fixture
def smooth_page(tmp_path):
    """A real Chromium document with a mutable, offline status response."""
    browser_api = pytest.importorskip("playwright.sync_api")
    found, _, clock = collector(tmp_path)
    run_for(found, clock, 10, viewer=True)
    value = json.loads(json.dumps(found.snapshot()))
    now = time.time()
    value.update(generated_at=now, poll_seconds=60)
    for group in value["groups"]:
        for member in group["members"]:
            member.pop("counters", None)
            member.pop("counter_samples", None)
            member.update(state="serving", logs=dict(enabled=True, host="fixture", commands=["example logs"]))
            member["live"].update(source="metrics", output_tps=None, prefill_tps=None, read_at=now)
            member["history"] = [[now - 120 + i * 5, 20.1, 1, 0, .257, 50.2, "metrics"] for i in range(25)]
    page_bytes, csp = dashboard.page_bytes()
    with browser_api.sync_playwright() as api:
        try:
            browser = api.chromium.launch(headless=True)
        except browser_api.Error:
            try:
                browser = api.chromium.launch(headless=True, channel="chrome")
            except browser_api.Error:
                pytest.skip("Chromium/Chrome unavailable for offline browser test")
        page = browser.new_page(viewport={"width": 1400, "height": 1000})
        requests, errors = [], []
        page.on("pageerror", lambda error: errors.append(str(error)))

        def route(request):
            url = request.request.url
            requests.append(url)
            if "/api/status" in url:
                request.fulfill(json=value)
            elif url.endswith("/api/history"):
                request.fulfill(json={})
            else:
                request.fulfill(body=page_bytes, content_type="text/html", headers={"Content-Security-Policy": csp})

        page.route("**/*", route)
        page.goto("http://127.0.0.1:8090/")
        page.locator("[data-graph='1'] svg").first.wait_for()
        yield page, value, requests
        assert errors == []
        browser.close()


def update_page(page):
    page.evaluate("async () => { await refresh(); await new Promise(requestAnimationFrame); }")


@pytest.mark.parametrize("width", [1400, 375])
def test_thinking_line_renders_and_updates_in_place(smooth_page, width):
    page, value, requests = smooth_page
    page.set_viewport_size({"width": width, "height": 1000})
    member = value["groups"][0]["members"][0]
    row = {"profile": "glm53-flash-nvfp4-spark-tp2"}
    member["thinking"] = dashboard.thinking_line(row, {})
    member["thinking_provenance"] = dashboard.thinking.provenance_text(dashboard.thinking_state(row, {}))
    update_page(page)
    card = page.locator("article.card").first
    line = card.locator(".thinking")
    assert line.is_visible() and line.inner_text() == "Thinking: always · high (levels: low, high or max)"
    details = card.locator(".thinking-details")
    assert not details.locator("p").is_visible()
    card.locator(".card-details > summary").click()
    details.locator("summary").click()
    assert details.locator("p").inner_text() == "Deployment default; Model default: always · max"
    line.evaluate("el => window.heldThinking = el")
    card.evaluate("el => window.heldCard = el")
    card.locator("details").first.evaluate("el => { el.open = true; window.heldDetail = el; }")
    before = len(requests)
    member["settings"]["default_chat_template_kwargs"] = {"reasoning_effort": "low"}
    member["thinking"] = dashboard.thinking_line(row, member["settings"])
    update_page(page)
    assert line.inner_text() == "Thinking: always · low (levels: low, high or max)"
    assert line.evaluate("el => el === heldThinking && !el.closest('details')")
    assert card.evaluate("el => el === heldCard")
    assert card.locator("details").first.evaluate("el => el === heldDetail && el.open")
    assert 'Chat defaults' in card.text_content() and '{"reasoning_effort":"low"}' in card.text_content()
    assert all("/api/status" in url for url in requests[before:])
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    output = os.environ.get("SPARKRING_CHART_SCREENSHOTS")
    if output:
        Path(output).mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(Path(output) / f"thinking-{width}.png"), full_page=True)
    member["thinking"] = None
    update_page(page)
    assert card.locator(".thinking").count() == 0
    assert card.evaluate("el => el === heldCard")


def test_unmanaged_four_pairs_with_reused_rendezvous(smooth_page, tmp_path):
    from runtime.host.test_dashboard_ring_observation import ring_observation
    page, value, _ = smooth_page
    found, _, _, _, _ = ring_observation(tmp_path)
    found.cycle()
    value.update(found.snapshot())
    value["poll_seconds"] = 60
    update_page(page)
    cards = page.locator("article.card")
    assert cards.count() == 4
    for pair in range(4):
        card = cards.filter(has_text=f"Research-pair-{pair}")
        assert card.count() == 1
        assert f"Sparks {pair * 2}, {pair * 2 + 1}" in card.inner_text()
        assert "TP2 / DCP2" in card.inner_text()
    assert page.locator("#ringviz .node.busy").count() == 8


def test_unmanaged_noncontiguous_ring_label_and_in_place_details(smooth_page, tmp_path):
    from runtime.host.test_dashboard_unmanaged import reports, install_reports
    page, value, requests = smooth_page
    found, _, clock = collector(tmp_path, rows=())
    install_reports(found, reports(found, clock, (0, 2, 3, 4, 6, 7)), tmp_path)
    found.cycle()
    value.update(found.snapshot())
    value["poll_seconds"] = 60
    update_page(page)
    card = page.locator("article.card")
    assert card.count() == 1
    card.locator(".card-details").evaluate("el => el.open = true")
    assert "Not installed by SparkRing: no automatic recovery, no install record" in card.inner_text()
    assert "positions 0, 2, 3, 4, 6, 7" in card.inner_text()
    assert page.locator("#ringviz .node.busy").count() == 6
    assert page.locator("#ringviz .node.idle").count() == 2
    assert page.locator("#ringviz .node.busy.outline rect").count() == 6
    assert page.locator("#ringviz .node.busy[data-position='1']").count() == 0
    card.locator(".launch-facts").evaluate("el => { el.open = true; window.heldDetails = el; }")
    before = len(requests)
    for i in range(3):
        value["generated_at"] += 1
        update_page(page)
        assert card.locator(".launch-facts").evaluate("el => el === heldDetails && el.open")
    assert all("/api/status" in url for url in requests[before:])
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    page.set_viewport_size({"width": 375, "height": 1000})
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    page.set_viewport_size({"width": 1400, "height": 1000})
    output = os.environ.get("SPARKRING_CHART_SCREENSHOTS")
    if output:
        Path(output).mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(Path(output) / "unmanaged-noncontiguous.png"), full_page=True)


@pytest.mark.parametrize("width,theme", [(1400, "dark"), (375, "light")])
def test_updates_keep_nodes_pointer_focus_selection_and_layout(smooth_page, width, theme):
    page, value, _ = smooth_page
    page.set_viewport_size({"width": width, "height": 1000})
    page.emulate_media(color_scheme=theme)
    card = page.locator("article.card").first
    card.locator(".card-details").evaluate("el => el.open = true")
    graph = card.locator("[data-graph='1']")
    graph.scroll_into_view_if_needed()
    graph.locator("svg").hover(position={"x": 50, "y": 20})
    graph.focus()
    page.evaluate("""() => {
      window.heldNodes = [...document.querySelectorAll('article.card, .tile, [data-graph] *, #ringviz svg, #ringviz g, #legend li')];
      const model = document.querySelector('article .model');
      const range = document.createRange(); range.selectNodeContents(model);
      getSelection().removeAllRanges(); getSelection().addRange(range);
      window.selectedText = getSelection().toString();
      window.shifts = [];
      window.shiftObserver = new PerformanceObserver(list => shifts.push(...list.getEntries().map(e => e.value)));
      shiftObserver.observe({type: 'layout-shift'});
    }""")
    page.wait_for_timeout(600)  # Let input settling finish before the measured interval.
    page.evaluate("shifts.length = 0")
    card_height = card.bounding_box()["height"]
    for i in range(10):
        value["generated_at"] += 1
        member = value["groups"][0]["members"][0]
        member["live"].update(output_tps=12345 + i, prefill_tps=12345 + i, running=12345 + i,
                             ttft_5m=12.345 + i, prefix_hit_5m=.357)
        member["history"].append([value["generated_at"], 40 + i, 1, 0, .267, 12345, "metrics"])
        update_page(page)
        page.wait_for_timeout(1000)
        assert page.evaluate("heldNodes.every(n => n.isConnected)")
        assert graph.evaluate("el => el === document.activeElement")
        assert card.locator(".card-details").evaluate("el => el.open")
        assert card.bounding_box()["height"] == card_height
        assert page.evaluate("getSelection().toString() === selectedText")
        assert graph.evaluate("el => el.querySelector('output').textContent === exactGraphValue(pointerSample(el.pointerX, el), 1)")
    cls = page.evaluate("shifts.reduce((sum, value) => sum + value, 0)")
    assert cls < .01, cls
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    output = os.environ.get("SPARKRING_CHART_SCREENSHOTS")
    if output:
        Path(output).mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(Path(output) / f"smooth-{width}-{theme}.png"))
        (Path(output) / f"layout-{width}.json").write_text(json.dumps(dict(seconds=10, cls=cls, width=width)))


def test_graph_windows_pause_rescale_and_frozen_dialog(smooth_page):
    page, value, requests = smooth_page
    graph = page.locator("[data-graph='1']").first
    # Two minutes fill the plot; the rightmost sample is effectively at now.
    assert graph.evaluate("el => { const p = el.querySelector('path'); return p.getBBox().width >= 285; }")
    member = value["groups"][0]["members"][0]
    now = value["generated_at"]
    member["history"] = [[now - 600 + i * 5, 1000 if i == 1 else 20, 1, 0, .25, 50, "metrics"] for i in range(121)]
    update_page(page)
    selector = page.locator("[data-controls] select").first
    assert graph.evaluate("el => el.plot.top") > 1000
    for option, seconds in [("300", 300), ("900", 600), ("3600", 600), ("10800", 600), ("auto", 600)]:
        selector.select_option(option)
        page.wait_for_function("wanted => document.querySelector('[data-graph]').plotSignature.includes(wanted)", arg=f'"{option}"')
        assert abs(graph.evaluate("el => el.plot.now - el.plot.start") - seconds) < 5
    selector.select_option("300")
    page.wait_for_function("() => document.querySelector('[data-graph]').plot.top < 100")
    assert page.evaluate("localStorage.getItem('sparkring-window')") == "300"
    # With a fixed window, an aging spike must leave the scale as time advances.
    member["history"] = [[now - 295 + i * 5, 1000 if i == 0 else 20, 1, 0, .25, 50, "metrics"] for i in range(60)]
    update_page(page)
    assert graph.evaluate("el => el.plot.top") > 1000
    value["generated_at"] += 10
    member["history"].append([value["generated_at"], 20, 1, 0, .25, 50, "metrics"])
    update_page(page)
    assert graph.evaluate("el => el.plot.top") < 100
    pause = page.locator("[data-pause]").first
    pause.click()
    page.wait_for_function("() => document.querySelector('[data-pause]').textContent === 'Resume'")
    frozen = graph.evaluate("el => [el.plot.start, el.plot.now, el.querySelector('path').getAttribute('d')]")
    for _ in range(3):
        value["generated_at"] += 5
        member["live"]["output_tps"] = 99999
        member["history"].append([value["generated_at"], 9000, 1, 0, .5, 50, "metrics"])
        update_page(page)
        assert graph.evaluate("el => [el.plot.start, el.plot.now, el.querySelector('path').getAttribute('d')]") == frozen
    assert "99,999" in page.locator(".tiles").first.inner_text()
    assert "Paused · 15 s ago" in page.locator(".pause-age").first.inner_text()
    graph.click()
    before = page.locator("#graph-value").inner_text()
    page.locator("#graph-window").select_option("21600")
    assert page.evaluate("graphSelection.rows.length") == 61
    # Network updates do not change the frozen dialog or its sample selection.
    before = page.locator("#graph-value").inner_text()
    value["generated_at"] += 5
    update_page(page)
    assert page.locator("#graph-value").inner_text() == before
    page.keyboard.press("Escape")
    pause.click()
    page.wait_for_function("() => document.querySelector('[data-graph]').plot.top > 9000")
    before_requests = len(requests)
    graph.click()
    page.locator("#graph-window").select_option("900")
    page.keyboard.press("Escape")
    assert len(requests) == before_requests


def test_hidden_document_has_no_mutations_and_age_tick_does_not_redraw(smooth_page):
    page, value, _ = smooth_page
    page.evaluate("""() => {
      window.mutations = [];
      window.observer = new MutationObserver(rows => mutations.push(...rows));
      observer.observe(document.body, {subtree:true, childList:true, characterData:true, attributes:true});
    }""")
    page.wait_for_timeout(1100)
    assert page.evaluate("mutations.every(m => m.target.parentElement?.closest('#fresh, [data-age], .pause-age'))")
    # Drive the browser visibility boundary deterministically in headless Chromium.
    page.evaluate("""() => {
      Object.defineProperty(document, 'hidden', {configurable:true, get:() => true});
      document.dispatchEvent(new Event('visibilitychange')); mutations.length = 0;
    }""")
    for _ in range(3):
        value["generated_at"] += 5
        value["groups"][0]["members"][0]["live"]["output_tps"] = 12345
        update_page(page)
    page.wait_for_timeout(1200)
    assert page.evaluate("mutations.length") == 0
    page.evaluate("delete document.hidden; document.dispatchEvent(new Event('visibilitychange'))")
    page.wait_for_function("() => document.querySelector('.tiles').textContent.includes('12,345')")


def test_member_identity_order_and_display_numbers(smooth_page):
    page, value, _ = smooth_page
    ids = page.locator("article.card").evaluate_all("els => els.map(e => e.dataset.member)")
    value["groups"][0]["members"].reverse()
    update_page(page)
    assert page.locator("article.card").evaluate_all("els => els.map(e => e.dataset.member)") == ids
    member = value["groups"][0]["members"].pop()
    update_page(page)
    assert page.locator("article.card").count() == len(ids) - 1
    value["groups"][0]["members"].append(member)
    update_page(page)
    assert page.locator("article.card").last.get_attribute("data-member") == member["id"]
    assert page.evaluate("[one(12.34), small(4.567), small(12.3), small(.45, 1), gb(24.5e9), gb(1.5e12), gb(12.5e12), pct(.257)]") == ["12", "4.57", "12", "0.5", "25 GB", "1.5 TB", "13 TB", "26%"]
    assert "26%" in page.evaluate("exactGraphValue([100, 1.5, 0, 0, .257], 4)")
    # A browser without storage still has working controls and data updates.
    page.evaluate("Object.defineProperty(window, 'localStorage', {get() {throw new Error('disabled');}}); views.clear(); scheduleRender()")
    update_page(page)
    page.locator("[data-controls] select").first.select_option("900")


def test_stored_history_draws_on_first_render_and_polls_only_deltas(tmp_path):
    browser_api = pytest.importorskip("playwright.sync_api")
    found, _, clock = collector(tmp_path)
    clock.now = int(time.time() // 5) * 5 - 10800
    run_for(found, clock, 10800, viewer=True, step=5)
    history = found.history_snapshot()
    cursor = history["history_cursor"]
    value = found.snapshot(since=cursor)
    value["poll_seconds"] = 60
    page_bytes, csp = dashboard.page_bytes()
    with browser_api.sync_playwright() as api:
        try:
            browser = api.chromium.launch(headless=True)
        except browser_api.Error:
            try:
                browser = api.chromium.launch(headless=True, channel="chrome")
            except browser_api.Error:
                pytest.skip("Chromium/Chrome unavailable for offline browser test")
        page = browser.new_page()
        requests, errors = [], []
        page.on("pageerror", lambda error: errors.append(str(error)))

        def route(request):
            url = request.request.url
            requests.append(url.removeprefix("http://127.0.0.1:8090"))
            if "/api/status" in url:
                request.fulfill(json=value)
            elif url.endswith("/api/history"):
                request.fulfill(json=history)
            else:
                request.fulfill(body=page_bytes, content_type="text/html", headers={"Content-Security-Policy": csp})

        page.route("**/*", route)
        page.goto("http://127.0.0.1:8090/")
        page.locator("[data-graph='1'] svg").first.wait_for()
        assert page.evaluate("data.groups[0].members[0].history.length") == 2160
        page.get_by_role("button", name="Expand Output tok/s", exact=False).first.click()
        page.get_by_role("slider").focus()
        page.keyboard.press("Home")
        assert "Output tok/s: 30 " in page.locator("#graph-value").inner_text()
        assert page.evaluate("graphSelection.rows.at(-1)[0] - graphSelection.rows[0][0]") > 10700
        clock.now += 5
        found.mark_viewed()
        found.cycle()
        value = found.snapshot(since=cursor)
        value["poll_seconds"] = 60
        page.evaluate("refresh()")
        assert page.evaluate("data.groups[0].members[0].history.length") == 2161
        assert requests == ["/", "/api/history", f"/api/status?since={cursor}", f"/api/status?since={cursor}"]
        assert errors == []
        browser.close()
