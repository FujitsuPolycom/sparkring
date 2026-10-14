"""Display names are local presentation data; these tests never contact a Spark."""
import json

import pytest

from runtime.common import fabric_layout
from runtime.host import controller, dashboard_admin, dashboard_tui, node, operator_view, privilege, ring_name
from runtime.host.test_dashboard import collector
from runtime.host.test_dashboard_charts import smooth_page, update_page  # noqa: F401
from runtime.host.test_fabric import document_of
from scripts import sparkring

REAL_REFRESH = dashboard_admin.refresh_site


@pytest.fixture
def named_state(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, "STATE", tmp_path)
    monkeypatch.setattr(privilege, "running_as_root", lambda: True)
    monkeypatch.setattr(dashboard_admin, "SITE_PATH", tmp_path / "site.json")
    monkeypatch.setattr(dashboard_admin, "refresh_site", REAL_REFRESH)
    monkeypatch.setattr(dashboard_admin, "reload_service", lambda **kw: pytest.fail("name reloaded a service"))
    node.save(tmp_path, "cluster.json", {"name": "ring8c"})
    node.save(tmp_path, "fabric.json", {"cluster": "ring8c"})
    return tmp_path


def test_cli_set_get_clear_and_identity(named_state, capsys):
    identities = {p: p.read_bytes() for p in named_state.glob("*.json")}
    assert sparkring.main(["name"]) == 0
    assert capsys.readouterr().out.strip() == "ring8c"
    assert sparkring.main(["name", "  Draupnir  "]) == 0
    assert capsys.readouterr().out.strip() == "Draupnir"
    assert ring_name.read(named_state) == "Draupnir"
    assert json.loads((named_state / "site.json").read_text())["display_name"] == "Draupnir"
    assert sparkring.main(["name"]) == 0
    assert capsys.readouterr().out.strip() == "Draupnir"
    assert sparkring.main(["name", "--clear"]) == 0
    assert capsys.readouterr().out.strip() == "ring8c"
    assert not (named_state / ring_name.FILE).exists()
    assert json.loads((named_state / "site.json").read_text())["display_name"] is None
    assert all(p.read_bytes() == before for p, before in identities.items())


@pytest.mark.parametrize("value", ["", "  ", "a" * 41, "a\nb", "a\tb", "a\x1bb", "a\x7fb", "a\u202eb", "a\u200bb"])
def test_validation_rejects_invalid_names_without_writes(named_state, value):
    with pytest.raises(ValueError, match="printable"):
        ring_name.validate(value)
    assert sparkring.main(["name", value]) == 2
    assert not (named_state / ring_name.FILE).exists()


@pytest.mark.parametrize("value", ["Draupnir", "雪" * 40, "Cody's <Lab> & \"ring\"", "🪐 café"])
def test_printable_unicode(value):
    assert ring_name.validate("  " + value + "  ") == value


def test_cli_refuses_nonroot_changes_and_conflicting_options(named_state, monkeypatch):
    monkeypatch.setattr(privilege, "running_as_root", lambda: False)
    assert sparkring.main(["name", "Draupnir"]) == 2
    assert sparkring.main(["name", "--clear"]) == 2
    assert sparkring.main(["name", "Draupnir", "--clear"]) == 2
    assert not (named_state / ring_name.FILE).exists()


def test_nonroot_get_uses_published_operator_view(named_state, monkeypatch, capsys):
    from pathlib import Path
    read = Path.read_text
    def private(path, *args, **kwargs):
        if path == named_state / "cluster.json":
            raise PermissionError()
        return read(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", private)
    monkeypatch.setattr(privilege, "member", lambda: True)
    monkeypatch.setattr(operator_view, "load", lambda: {"cluster": {"name": "Draupnir"}})
    assert sparkring.main(["name"]) == 0
    assert capsys.readouterr().out.strip() == "Draupnir"
    assert any("Ring: Draupnir" == line for line in operator_view.lines({"cluster": {"name": "Draupnir"}}))


def test_collector_rename_and_clear_reuse_host_files_models_and_requests(tmp_path):
    found, api, clock = collector(tmp_path)
    source = found.sources[0]
    found.cycle()
    models = dict(found.models)
    fallback = source.fabric["cluster"]
    assert found.snapshot()["title"] == fallback
    path = source.site_path
    site = json.loads(path.read_text())
    requests = list(api.requests)
    for name in ("Draupnir", "Cody's <Lab> & ring", None):
        site["display_name"] = name
        node.save(path.parent, path.name, site)
        found.reload_sources()
        snapshot = found.snapshot()
        expected = name or fallback
        assert snapshot["title"] == snapshot["ring"]["name"] == snapshot["groups"][0]["name"] == expected
        assert all(found.models[key] is model for key, model in models.items())
        assert api.requests == requests


@pytest.mark.parametrize("width", [375, 1400])
def test_dashboard_escapes_name_and_updates_in_place(smooth_page, width):  # noqa: F811
    page, value, requests = smooth_page
    page.set_viewport_size({"width": width, "height": 1000})
    name = '<img src=x onerror=alert(1)> & "Lab"'
    before = len(requests)
    page.locator(".ringfig svg").first.evaluate("el => window.keptRing = el")
    for title in (name, "雪" * 40, "Draupnir", "ring8c"):
        value["title"] = value["rings"][0]["name"] = title
        update_page(page)
        assert page.locator("h1").inner_text() == title
        assert page.title() == title
        assert page.locator("#ringtitle").inner_text() == title + " · ring of 8"
        assert page.locator(".ringfig figcaption b").first.inner_text() == title
        assert page.locator("h1 img, #ringtitle img, figcaption img").count() == 0
        assert page.locator(".ringfig svg").first.evaluate("el => el === window.keptRing")
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    # Only the explicit test refreshes requested status; rendering itself sends nothing.
    assert len(requests) == before + 4


def test_tui_renders_literal_markup_and_escapes_terminal_controls():
    title = "Draupnir <lab> & 'ring'"
    segments = dashboard_tui.header_segments({"title": title}, age=0, interval=5)
    assert title in "".join(t for t, _ in segments)
    segments = dashboard_tui.header_segments({"title": "Lab\x1b[31m\n"}, age=0, interval=5)
    line = "".join(t for t, _ in segments)
    assert "\x1b" not in line and "\n" not in line
    assert r"Lab\x1b[31m\n" in line


def test_offline_setup_plan_shows_name_without_persisting(tmp_path, monkeypatch, capsys):
    plan, _, _ = document_of(fabric_layout.layout("cycle", 4))
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(plan["nodes"]))
    monkeypatch.setattr(controller, "STATE", tmp_path)
    args = ["--inventory", str(inventory), "--head-id", plan["nodes"][0]["node_id"],
            "--plan", "--name", "ring8c", "--output", str(tmp_path / "plan")]
    assert controller.setup(args + ["--display-name", " Draupnir "]) == 0
    assert "Ring name: Draupnir" in capsys.readouterr().out
    assert not (tmp_path / ring_name.FILE).exists()
    args[-1] = str(tmp_path / "fallback-plan")
    assert controller.setup(args) == 0
    assert "Ring name: ring8c" in capsys.readouterr().out

@pytest.mark.parametrize("name", [None, "Draupnir", "Lab 192.0.2.1"])
def test_status_and_operator_view_use_effective_name(tmp_path, monkeypatch, capsys, name):
    from runtime.host.test_fabric_bandwidth import recorded
    monkeypatch.setattr(controller, "STATE", tmp_path)
    value = recorded(tmp_path, 4)
    monkeypatch.setattr(node, "status", lambda: {"state": "network-configured"})
    monkeypatch.setattr(controller.discovery, "ssh", lambda *a, **kw: json.dumps({"state": "network-configured", "hostname": "spark"}))
    ring_name.save(tmp_path, name)
    expected = name or value["name"]
    assert controller.lifecycle(["status"]) == 0
    assert "Ring: " + expected in capsys.readouterr().out
    assert controller.lifecycle(["status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["ring_name"] == expected
    assert operator_view.view(tmp_path)["cluster"]["name"] == expected
