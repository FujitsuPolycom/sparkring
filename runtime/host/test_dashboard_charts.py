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
            page.screenshot(path=str(Path(output) / f"charts-{theme}.png"))
        page.keyboard.press("Escape")
        assert not dialog.is_visible()
        browser_api.expect(expand).to_be_focused()
        gaps = page.evaluate("""() => chart([[0, 2], [5, null], [10, 3]], 1, 10, "c1").svg""")
        assert "L" not in gaps.split('d="')[1].split('"')[0]
        assert requests == before and errors == []
        context.close()
        browser.close()


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
