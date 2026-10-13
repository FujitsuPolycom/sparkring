"""Offline reproduction of a manual GLM pair, four GPU jobs and a stopped Qwen pair."""
import json
import os
from pathlib import Path
import time

import pytest

from runtime.host import dashboard, dashboard_tui
from runtime.host.test_dashboard import collector
from runtime.host.test_dashboard_unmanaged import install_reports, reports


def layout_fixture(tmp_path):
    found, api, clock = collector(tmp_path, rows=(([4, 5], "192.0.2.14", 8020,
                                                 "Qwen3.8-Flash-Next-NVFP4-QAD-TP2"),))
    site = found.sources[0].site_path
    value = json.loads(site.read_text())
    value["deployments"][0]["stopped"] = True
    site.write_text(json.dumps(value))
    documents = reports(found, clock, (6, 7))
    for doc in documents:
        doc["model_containers"][0]["master_addr"] = "192.0.2.16"
        doc["model_containers"][0]["served_model"] = "GLM-5.3-Flash-NVFP4-Spark-TP2"
    api.models[("192.0.2.16", 9123)] = "GLM-5.3-Flash-NVFP4-Spark-TP2"
    api.behaviour[("192.0.2.14", 8020)] = "refused"
    for p in range(4):
        documents.append({"node_id": found.sources[0].fabric["positions"][p]["node_id"],
                          "observed_at": clock.now,
                          "model_containers": [{"managed": False, "engine": "unknown",
                                                "name": f"nsx4-net-sircl-nccl-test-r{p}",
                                                "image": "net-sircl/nccl-tests:20261013",
                                                "image_id": "abcdef123456"}]})
    install_reports(found, documents, tmp_path)
    found.cycle()
    value = json.loads(json.dumps(found.snapshot()))
    value.update(generated_at=time.time(), poll_seconds=3600)
    # Four-character host labels match the owner's dense eight-node ring.
    for ring in value["rings"]:
        for p, short in zip(ring["positions"], ["c0de", "abcd", "ebee", "fa11", "beef", "ca11", "deed", "f00d"]):
            p["short"] = short
    return value


GEOMETRY = """() => {
  const failures = [];
  const box = el => el.getBoundingClientRect();
  const inside = (a, b) => a.left >= b.left - 1 && a.right <= b.right + 1 &&
    a.top >= b.top - 1 && a.bottom <= b.bottom + 1;
  const overlap = (a, b) => Math.min(a.right, b.right) - Math.max(a.left, b.left) > 1 &&
    Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top) > 1;
  for (const el of document.querySelectorAll('.legend .pill, .card .pill, .card .tag, .card .chip')) {
    if (!el.checkVisibility()) continue;
    const parent = el.closest('li, article');
    if (!inside(box(el), box(parent))) failures.push('overflow: ' + el.textContent);
  }
  for (const svg of document.querySelectorAll('.ringfig svg')) {
    const labels = [...svg.querySelectorAll('.lbl')];
    const obstacles = [...svg.querySelectorAll('.node rect, .host, .downtag')];
    for (const [i, label] of labels.entries()) {
      if (!inside(box(label), box(svg))) failures.push('label outside ring: ' + label.textContent);
      for (const other of [...labels.slice(i + 1), ...obstacles]) {
        if (overlap(box(label), box(other))) failures.push('overlap: ' + label.textContent + ' / ' + other.textContent);
      }
    }
  }
  for (const el of document.querySelectorAll('.legend li, .card, .card .head, .card .sub, .card .model')) {
    if (!el.checkVisibility()) continue;
    if (el.scrollWidth > el.clientWidth + 1) failures.push('content overflow: ' + el.className);
  }
  if (document.documentElement.scrollWidth > innerWidth) failures.push('page overflow');
  return failures;
}"""


@pytest.mark.parametrize("width", [375, 1400])
def test_fixture_layout_and_in_place_updates(tmp_path, width):
    browser_api = pytest.importorskip("playwright.sync_api")
    value = layout_fixture(tmp_path)
    page_bytes, csp = dashboard.page_bytes()
    with browser_api.sync_playwright() as api:
        try:
            browser = api.chromium.launch(headless=True)
        except browser_api.Error:
            browser = api.chromium.launch(headless=True, channel="chrome")
        page = browser.new_page(viewport={"width": width, "height": 1000})
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
        page.locator(".legend .pill").first.wait_for()
        output = os.environ.get("SPARKRING_LAYOUT_SCREENSHOTS")
        if output:
            Path(output).mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(Path(output) / f"layout-{width}.png"))
            page.screenshot(path=str(Path(output) / f"layout-{width}-full.png"), full_page=True)
            page.locator("#legend").screenshot(path=str(Path(output) / f"legend-{width}.png"))
            page.locator(".card").first.screenshot(path=str(Path(output) / f"card-{width}.png"))
        failures = page.evaluate(GEOMETRY)
        if output:
            (Path(output) / f"checks-{width}.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
        assert failures == [], failures
        before = list(requests)
        page.evaluate("""() => { window.heldLayout = [...document.querySelectorAll('.card, .legend li, .band, .lbl')]; }""")
        page.evaluate("receivedAt -= 3; render()")
        assert page.evaluate("heldLayout.every(el => el.isConnected)")
        assert requests == before and errors == []
        text = page.locator("#legend").inner_text()
        assert "position 0" in text and "positions 4–5" in text
        assert "GPU in use" in text and "TP?" not in text
        assert page.locator("#legend .mid .t1").all_inner_texts().count("GPU in use") == 4
        assert page.locator('#legend .mid span[title="GLM-5.3-Flash-NVFP4-Spark-TP2"]').count() == 1
        assert "1 Spark" in page.locator("#stacks").text_content()
        assert "2 Sparks" in page.locator("#stacks").text_content()
        assert "1 Sparks" not in page.locator("#stacks").inner_text()
        for label in page.locator(".band").all():
            assert label.locator("title").text_content()
        # Exercise the longest server state too, and expanded container facts.
        page.evaluate("""() => {
          const m = data.groups.flatMap(g => g.members).find(m => m.state === 'serving');
          m.state = 'busy'; m.state_label = 'Busy: API unavailable'; render();
          document.querySelectorAll('.card-details, .launch-facts').forEach(el => el.open = true);
        }""")
        assert page.evaluate(GEOMETRY) == []
        for job in page.locator(".card").all():
            if job.locator(".model").inner_text().startswith("GPU in use"):
                assert "GPU in use · container nsx4-" in job.locator(".model").inner_text()
                assert "unknown" not in job.locator(".launch-facts").inner_text()
        assert page.evaluate("heldLayout.every(el => el.isConnected)")
        assert requests == before and errors == []
        # Eight independent groups leave less interior space than the owner's six.
        page.evaluate("""() => {
          const group = data.groups.find(g => g.kind === 'ring'), template = group.members[0];
          group.members = Array.from({length: 8}, (_, p) => ({...template, id: 'dense-' + p,
            name: 'Spark ' + p, arc: {...template.arc, positions: [p], first: p, last: p, size: 1}}));
          render();
        }""")
        assert page.evaluate(GEOMETRY) == []
        assert requests == before and errors == []
        browser.close()


def test_non_model_text_and_singular_plural(tmp_path):
    value = layout_fixture(tmp_path)
    members = value["groups"][0]["members"]
    jobs = [m for m in members if m.get("launch", {}).get("engine") == "unknown"]
    assert len(jobs) == 4  # Same image/name prefix alone is not proof of a shared run.
    for job in jobs:
        assert job["state_label"] == "GPU in use"
        assert job["name"] == f"Spark {job['arc']['first']}"
        assert "?" not in job["layout"]["label"]
        details = "\n".join("".join(text for text, _ in line)
                            for line in dashboard_tui.model_details(job, value["generated_at"]))
        assert "GPU in use" in details and "net-sircl/nccl-tests:20261013" in details
        assert f"position {job['arc']['first']}" in details and "positions " not in details
        assert "API unavailable" not in details and "TP?" not in details
        _, segments = dashboard_tui.model_line(job, value["generated_at"])
        assert "GPU in use" in "".join(text for text, _ in segments)
    pair = next(m for m in members if m["arc"]["positions"] == [6, 7])
    assert pair["name"] == "Sparks 6, 7"


@pytest.mark.parametrize("rendezvous,images,expected", [
    ([29500, 29500, 29500, 29500], ["job:a"] * 4, [[0, 1, 2, 3]]),
    ([29500, 29500, 29501, 29501], ["job:a"] * 4, [[0, 1], [2, 3]]),
    ([29500] * 4, ["job:a", "job:a", "job:b", "job:b"], [[0, 1], [2, 3]]),
    ([None] * 4, ["job:a"] * 4, [[0], [1], [2], [3]]),
])
def test_gpu_job_grouping_requires_shared_facts_without_api_reads(tmp_path, rendezvous, images, expected):
    found, api, clock = collector(tmp_path, rows=())
    documents = reports(found, clock, (0, 1, 2, 3), size=4)
    for doc, port, image in zip(documents, rendezvous, images):
        c = doc["model_containers"][0]
        c.clear()
        c.update(managed=False, name="same-name", image=image, engine="unknown")
        if port:
            c.update(master_addr="192.0.2.10", master_port=port)
        c["api_port"] = 9123  # Unknown engines never become API probe targets.
    install_reports(found, documents, tmp_path)
    found.cycle()
    members = found.snapshot()["groups"][0]["members"]
    assert [m["arc"]["positions"] for m in members] == expected
    assert all(m["state_label"] == "GPU in use" and m["endpoint"] is None for m in members)
    assert api.requests == []


def test_only_known_parallel_sizes_are_displayed():
    assert dashboard.parallelism({}) == ""
    assert dashboard.parallelism({"tp": 2, "pp": 1}) == "TP2 / PP1"
