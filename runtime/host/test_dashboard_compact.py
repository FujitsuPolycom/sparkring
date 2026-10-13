"""Compact cards and page-wide graph controls, using intercepted browser fixtures."""
import json
import os
from pathlib import Path
import shlex

import pytest

from runtime.host import dashboard, test_dashboard_charts
from runtime.host.test_dashboard_layout import GEOMETRY

smooth_page = test_dashboard_charts.smooth_page
update_page = test_dashboard_charts.update_page


@pytest.mark.parametrize("width", [375, 1400])
@pytest.mark.parametrize("source", ["metrics", "plugin"])
def test_compact_cards_density_details_and_accessibility(smooth_page, tmp_path, width, source):
    page, value, requests = smooth_page
    page.set_viewport_size({"width": width, "height": 1000})
    page.emulate_media(color_scheme="dark" if width == 1400 else "light")
    members = value["groups"][0]["members"]
    now = value["generated_at"]
    for i, member in enumerate(members):
        member.update(name=["glm", "qwen", "mimo", "Research-pair"][i], state_since=now - 3240)
        member["live"].update(source="metrics", output_tps=42, prefill_tps=1250, steps_ps=12,
                              acceptance=.72, running=2, read_at=now)
        member["thinking"] = "always · high (levels: low, high or max)"
        member["recovery"] = dict(enabled=True, state="serving", checked_at=now-19260,
                                   summary="Serving; no recovery action needed")
        if i in (1, 2):
            member.update(state="stopped", state_label="Stopped", live={}, history=[],
                          recovery=dict(enabled=False, summary="Recovery disabled"))
        if i == 3:
            member.update(unmanaged=True, history=[], recovery=None,
                          ownership_label="Not installed by SparkRing: no automatic recovery, no install record",
                          launch={"engine": "vllm", "image_id": "fixture-image"})
    members[0]["settings"]["decode_context_parallel_size"] = 2
    thought = dashboard.thinking_state({"profile": "glm53-flash-nvfp4-spark-tp2"}, {})
    members[0].update(thinking=dashboard.thinking.deployment_text(thought),
                      thinking_provenance=dashboard.thinking.provenance_text(thought))
    if source == "plugin":
        from runtime.host import dashboard_admin
        from runtime.host.test_dashboard import PAIRS, collector
        from runtime.host.test_dashboard_rates import document, member as rate_member

        found, _, clock = collector(tmp_path, rows=PAIRS[3:])
        found.reload_sources()
        model = next(iter(found.models.values()))
        model.row["deployment_id"] = "remote-deployment"
        for step in range(3):
            at = now - 10 + step * 5
            plugin = rate_member(document(at, 100 + step * 6252, 100 + step * 212,
                                          plugin=True))["plugin_counters"]
            plugin.update(running=2, kv=.256, steps=100 + step * 62,
                          accepted=100 + step * 72, draft_tokens=100 + step * 100)
            remote = {"node_id": "remote-head", "observed_at": at,
                      "telemetry": {"remote-deployment": plugin}}
            found.node_reports = dashboard_admin.cached_node_rows(
                tmp_path, tmp_path / "nodes.json", {"remote": remote}, at)
            members[0]["plugin_counters"] = found.plugin_snapshot(model, at)
            assert members[0]["plugin_counters"] == plugin
            members[0]["counters"] = dict(source="metrics", at=at, run_id=1,
                                           values=dict(prompt=10, gen=10, running=0, waiting=0, kv=0))
            update_page(page)
    update_page(page)
    cards = page.locator("article.card")
    serving, stopped, unmanaged = cards.nth(0), cards.nth(1), cards.nth(3)
    assert page.locator("#page-graphs select").count() == 1
    assert page.locator(".card [data-controls], .expand-graph").count() == 0
    assert page.get_by_text("Hover or expand to inspect samples.").count() == 0
    assert serving.locator(".tiles .tile").count() == 4
    assert "TP2 / DCP2" in serving.locator(".head").inner_text()
    assert serving.locator(".source").inner_text().startswith(
        ("plugin" if source == "plugin" else "/metrics") + " · sample")
    assert "12 steps/s · 72% draft acceptance" in serving.inner_text()
    assert serving.inner_text().count("/metrics") == (0 if source == "plugin" else 1)
    if source == "plugin":
        assert serving.locator(".tile .v").all_inner_texts() == ["42", "1,250", "2 / 16", "26%"]
    assert serving.inner_text().count(members[0]["model"]) == 1
    assert "recovery ok · 5 h ago" in serving.inner_text()
    assert "Recovery check:" not in serving.inner_text()
    assert stopped.locator(".tiles, .charts").count() == 0
    assert not stopped.locator(".sparks").is_visible()
    assert stopped.get_by_role("button", name="Copy start command for qwen").get_attribute("data-copy") == members[1]["start_command"]
    assert "Last served:" in stopped.inner_text() and "for 54 min" in stopped.inner_text()
    assert unmanaged.locator(".empty-plot:visible").count() == 3
    assert unmanaged.locator("[data-graph='1'] path").get_attribute("d") == "M0,78L300,78"
    assert page.evaluate(GEOMETRY) == []
    assert serving.locator(".tile .k").evaluate_all("els => els.every(e => e.scrollWidth <= e.clientWidth + 1)")
    for graph in serving.locator("[data-graph]").all():
        assert 48 <= graph.locator("svg").bounding_box()["height"] <= 64
    if width == 1400:
        assert serving.bounding_box()["height"] <= 400
        assert all(cards.nth(i).bounding_box()["height"] < 100 for i in (1, 2))
    output = os.environ.get("SPARKRING_COMPACT_SCREENSHOTS")
    if output:
        directory = Path(output)
        directory.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(directory / f"compact-{source}-{width}.png"), full_page=True)
        (directory / f"heights-{source}-{width}.json").write_text(json.dumps(cards.evaluate_all(
            "els => els.map(e => ({name: e.dataset.member, height: e.getBoundingClientRect().height}))"), indent=2))
    before = list(requests)
    graph = serving.locator("[data-graph='1']")
    graph.click()
    assert page.get_by_role("dialog").is_visible()
    page.keyboard.press("Escape")
    assert graph.evaluate("el => el === document.activeElement")
    for key in ("Enter", "Space"):
        page.keyboard.press(key)
        assert page.get_by_role("dialog").is_visible()
        page.keyboard.press("Escape")
        assert graph.evaluate("el => el === document.activeElement")
    unmanaged.locator("[data-graph='1']").click()
    assert page.locator("#graph-body .empty-plot").is_visible()
    assert not page.locator("#graph-body .y-axis").is_visible()
    assert page.get_by_role("slider").is_disabled()
    page.keyboard.press("Escape")
    stopped.locator(".card-details > summary").click()
    assert stopped.locator(".sparks").is_visible()
    assert "Recovery check: Recovery disabled" in stopped.inner_text()
    assert "Max concurrent" in stopped.text_content()
    serving.locator(".card-details > summary").click()
    assert "Recovery check: Serving; no recovery action needed" in serving.inner_text()
    assert "Output 5 min avg" in serving.inner_text()
    assert "Draft" in serving.inner_text() and "requests done" in serving.inner_text()
    assert serving.get_by_role("button", name="Re-read settings").is_visible()
    assert page.evaluate(GEOMETRY) == []
    assert requests == before


def test_global_window_pause_storage_and_state_transitions(smooth_page):
    page, value, requests = smooth_page
    selector = page.locator("#page-graphs select")
    selector.select_option("900")
    page.wait_for_function("() => [...document.querySelectorAll('[data-graph]')].every(e => e.plotSignature.includes('900'))")
    page.reload()
    page.locator("[data-graph]").first.wait_for()
    assert selector.input_value() == "900"
    pause = page.locator("[data-pause]")
    pause.click()
    page.wait_for_function("() => graphView.paused !== null && [...document.querySelectorAll('[data-graph]')].every(e => e.plot.now === graphView.paused)")
    frozen = page.locator("[data-graph]").evaluate_all("els => els.map(e => [e.plot.now, e.querySelector('path').getAttribute('d')])")
    for member in value["groups"][0]["members"]:
        member["history"].append([value["generated_at"] + 5, 123, 1, 0, .4, 789, "metrics"])
    value["generated_at"] += 5
    update_page(page)
    assert page.locator("[data-graph]").evaluate_all("els => els.map(e => [e.plot.now, e.querySelector('path').getAttribute('d')])") == frozen
    member = value["groups"][0]["members"][0]
    card = page.locator("article.card").first
    card.locator(".card-details > summary").click()
    card.evaluate("el => { window.heldCompactCard = el; window.heldCompactDetails = el.querySelector('.card-details'); }")
    for state in ("stopped", "not_ready", "idle", "offline", "not_installed"):
        member.update(state=state, state_label=state)
        update_page(page)
        assert card.locator(".tiles, .charts").count() == 0
        assert card.evaluate("el => el === heldCompactCard && el.querySelector('.card-details') === heldCompactDetails && heldCompactDetails.open")
    member.update(state="serving", state_label="Serving")
    update_page(page)
    assert card.locator("[data-graph]").count() == 3
    assert card.locator("[data-graph]").first.evaluate("el => el.plot.now") == frozen[0][0]
    before = list(requests)
    pause.click()
    page.wait_for_function("old => graphView.paused === null && document.querySelector('[data-graph]').plot.now > old", arg=frozen[0][0])
    assert card.locator("[data-graph]").first.evaluate("el => el.plot.now") > frozen[0][0]
    assert requests == before


def test_start_commands_select_the_saved_instance_and_preserve_quoting():
    row = dict(profile="glm53-flash-nvfp4-spark-tp2", id="glm53-flash-nvfp4-spark-tp2-i123", positions=[0, 1])
    assert shlex.split(dashboard.start_command(row)) == ["sudo", "sparkring", "up", row["profile"], "--instance", "i123"]
    assert dashboard.start_command({**row, "unmanaged": True}) is None
    assert shlex.split(dashboard.start_command({**row, "id": row["profile"]})) == ["sudo", "sparkring", "up", row["profile"]]
    assert shlex.split(dashboard.start_command(dict(profile=None, id="custom", positions=[2, 3]))) == ["sudo", "sparkring", "up", "--on", "2,3"]
    assert dashboard.start_command({**row, "profile": {}}) == "sudo sparkring up --on 0,1"
    strange = {**row, "profile": "example;echo", "id": "example;echo-slot $(x)"}
    assert shlex.split(dashboard.start_command(strange))[-1] == "slot $(x)"

def test_decode_parallel_size_comes_from_existing_runtime_response():
    from runtime.host.test_dashboard import runtime_document
    document = runtime_document()
    document["effective"]["decode_context_parallel_size"] = {"state": "known", "value": 2}
    assert dashboard.read_runtime_status(document)["settings"]["decode_context_parallel_size"] == 2
