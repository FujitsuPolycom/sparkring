"""``sparkring dashboard tui``: the terminal view, rendered offline from status documents that the
collector builds from the recorded vLLM and runtime-status payloads and from the sanitized lab
sample, at full and narrow widths.

The guard tests fail if the view would open a socket to anything but the dashboard service on this
Spark, or start a process.
"""
import argparse
import ast
import json
from pathlib import Path
import socket
import threading

import pytest

from runtime.host import dashboard, dashboard_tui as tui
from runtime.host.test_dashboard import FIXTURES, NODE_A_LAN, PAIRS, collector, run_for, runtime_document

LOW_DISK = 0.01 * 4e12          # spark5's root filesystem: 1 % of 4 TB free


def snapshot_of(tmp_path):
    """Four pairs on a ring of eight after two viewed minutes. The pair on positions 2-3 answers
    with the recorded payloads; the others with synthetic ones whose counters advance."""
    found, api, clock = collector(tmp_path)
    for positions, host, port, _ in PAIRS:
        document = runtime_document(nodes=tuple(f"spark{p}" for p in positions))
        document["workers"]["ranks"][0]["identity"]["image_id"] = {"state": "known", "value": "sha256:" + "0123456789ab" * 5}
        if 5 in positions:
            document["workers"]["ranks"][1]["resources"]["filesystems"][0]["available_bytes"] = LOW_DISK
        api.bodies[(host, port, "/v1/sparkring/status")] = json.dumps(document).encode()
    api.bodies[("192.0.2.12", 8015, "/metrics")] = (FIXTURES / "vllm-metrics-tp2.prom").read_bytes()
    api.bodies[("192.0.2.12", 8015, "/v1/sparkring/status")] = (FIXTURES / "runtime-status-tp2.json").read_bytes()
    from runtime.host.dashboard_rates import ViewerRates
    rates = ViewerRates()
    run_for(found, clock, 115, viewer=True)
    rates.apply(found.snapshot())
    run_for(found, clock, 5, viewer=True)
    return rates.apply(json.loads(json.dumps(found.snapshot()))), found


def render(snapshot, width, *, details="all", height=None, events=False):
    view = {"selected": 0, "top": 0, "details": details, "events": events}
    rows = tui.frame(snapshot, view, width=width, height=height, age=1.0, interval=5.0)
    return rows, tui.text_lines(rows, width)


def segment_after(rows, label):
    """(text, style) of the segment that follows the first segment equal to `label` in each row."""
    found = []
    for segments, _ in rows:
        for index, (text, st) in enumerate(segments[:-1]):
            if text == label:
                found.append(segments[index + 1])
    return found


# -- rendering -----------------------------------------------------------------------------------
def test_the_view_shows_the_header_ring_cards_details_and_sparks(tmp_path):
    snapshot, _ = snapshot_of(tmp_path)
    rows, lines = render(snapshot, 140)
    text = "\n".join(lines)
    header = lines[0]
    for part in ("test", "4 serving", "8/8 Sparks busy", "reqs", "tok/s", "live 1s ago", "poll 2s"):
        assert part in header
    assert lines[2].startswith(" ●0 spark0 ━ ●1 spark1 ─ ●2 spark2 ━ ●3 spark3") and lines[2].endswith("─ ↺")
    card = next(line for line in lines if line.startswith(" ▾ tp2-0-1"))
    for part in ("serving", "TP2", "@0–1", "SIRCL", "up ", "t/s", "run 1", "kv 25% (1M tok)", "acc 2.50"):
        assert part in card
    for label in ("model", "endpoint", "layout", "arc", "transport", "head", "throughput", "tok/s", "kv",
                  "requests", "kv cache", "latency", "totals", "settings", "runtime", "logs"):
        assert f"│ {label:<10} " in text or f"└ {label:<10} " in text, label
    assert "GLM-5.3-Flash-NVFP4-Spark-TP2" in text and f"http://{NODE_A_LAN}:8000/v1" in text
    assert "TP2 pair · profile · spark0, spark1 · workers 2/2" in text
    assert "positions 0–1 · spark0 → spark1" in text and "deployment record" not in text
    assert [line.rstrip() for line in lines if " transport " in line][0].endswith("transport  SIRCL 0.3.2")
    assert "tp2-prepared" not in text and "RoCEnante" not in text and "bundle" not in text
    assert "spark0:8000 · container sr-0-r0" in text
    assert "of 1M tok (24G) · 4.50× concurrency @ 131k ctx" in text
    assert "sudo docker logs --tail 200 sr-0-r0  on spark0" in text and "image 0123456789ab" in text
    sparks = [line for line in lines if "rank " in line and " ram " in line]
    assert len(sparks) == 6        # the recorded runtime status of positions 2-3 reports no host resources
    assert "▲ 0 spark0" in sparks[0] and f" {NODE_A_LAN}  rank 0 · ram 101G/119G · disk free 931G" in sparks[0]
    assert "▲ 1 spark1" in sparks[1] and " 192.0.2.11  rank 1" in sparks[1]


def test_the_transport_line_names_roce_by_its_display_name_without_the_bundle(tmp_path):
    snapshot, _ = snapshot_of(tmp_path)
    member = snapshot["groups"][0]["members"][0]
    member["transport"] = dashboard.read_transport(runtime_document(collective="rocenante"))
    _, lines = render(snapshot, 140)
    line = next(line for line in lines if " transport " in line)
    assert line.rstrip().endswith("transport  RoCEnante (b12x)")
    assert not any("tp2-prepared" in text for text in lines)


def test_a_slow_first_token_is_coloured_on_the_card(tmp_path):
    snapshot, _ = snapshot_of(tmp_path)
    member = snapshot["groups"][0]["members"][0]
    for ttft, colour in ((0.4, 0), (1.5, tui.WARN), (4.0, tui.CRIT)):
        member["live"]["ttft_5m"] = ttft
        _, bits = tui.model_line(member, snapshot["generated_at"])
        assert (f"ttft {ttft:.1f}s", colour) in bits


def test_display_precision_exceptions_and_whole_units():
    assert tui.smallfmt(4.567) == "4.57"
    assert tui.smallfmt(12.34) == "12"
    assert tui.smallfmt(.45, 1) == "0.5"
    assert tui.smallfmt(12.34, 1) == "12"
    assert tui.bytesfmt(24 * (1 << 30)) == "24G"
    assert tui.bytesfmt(1.5 * (1 << 40)) == "1.5T"
    assert tui.bytesfmt(12.5 * (1 << 40)) == "12T"


def test_the_recorded_payloads_render_as_a_serving_pair(tmp_path):
    snapshot, _ = snapshot_of(tmp_path)
    _, lines = render(snapshot, 140)
    card = next(line for line in lines if line.startswith(" ▾ tp2-2-3"))
    assert "serving" in card and "TP2" in card and "@2–3" in card
    text = "\n".join(lines)
    assert "Qwen3.8-Flash-Next-NVFP4-QAD-TP2" in text


def test_the_sanitized_lab_sample_renders(tmp_path):
    lab = json.loads((FIXTURES / "lab-status-sample.json").read_text(encoding="utf-8"))
    _, lines = render(lab, 120)
    assert "0 serving" in lines[0] and "0/8 Sparks busy" in lines[0]
    assert lines[2].startswith(" ○0 0 ─ ○1 1 ─") and lines[2].endswith("─ ↺")
    assert any("no model deployment is recorded" in line for line in lines)
    _, lines = render(lab, 120, events=True)
    assert any("tp8-0-7" in line and "No model running" in line for line in lines)


@pytest.mark.parametrize("width", [40, 60, 80])
def test_a_narrow_terminal_gets_every_line_cut_to_fit(tmp_path, width):
    snapshot, _ = snapshot_of(tmp_path)
    rows, lines = render(snapshot, width)
    assert all(len(line) <= width - 1 for line in lines)
    assert lines[0].startswith(" test · 4 serving")
    ring = lines[2]
    assert ring.startswith(" ●0 ━ ●1 ─ ●2") and "spark1" not in ring     # host names give way first
    assert sum(1 for line in lines if line.startswith(" ▾ tp2-")) == 4


def test_the_interactive_frame_fills_the_screen_and_keeps_the_selection_in_view(tmp_path):
    snapshot, _ = snapshot_of(tmp_path)
    view = {"selected": 3, "top": 0, "details": True, "events": False}
    rows = tui.frame(snapshot, view, width=70, height=16, age=1.0, interval=5.0)
    assert len(rows) == 16 and "q quit" in tui.text_lines(rows[-1:], 70)[0]
    selected = [tui.text_lines([row], 70)[0] for row in rows if row[1]]
    assert len(selected) == 1 and selected[0].startswith(" ▾ tp2-6-7")
    assert view["top"] > 0


# -- disk free -----------------------------------------------------------------------------------
@pytest.mark.parametrize(("free", "expected"), [
    (0.01 * 4e12, tui.style(tui.CRIT, bold=True)),
    (0.019 * 4e12, tui.style(tui.CRIT, bold=True)),
    (0.03 * 4e12, tui.style(tui.WARN)),
    (0.25 * 4e12, 0),
    (None, tui.style(dim=True)),
])
def test_disk_free_below_the_threshold_is_red(free, expected):
    assert tui.disk_style(free, 4e12) == expected


def test_the_low_disk_spark_renders_its_disk_free_cell_red(tmp_path):
    snapshot, _ = snapshot_of(tmp_path)
    rows, _ = render(snapshot, 140)
    cells = segment_after(rows, " · disk free ")
    assert len(cells) == 6
    red = [text for text, st in cells if st == tui.style(tui.CRIT, bold=True)]
    assert red == ["37G"]                         # spark5 alone: 40 GB of 4 TB
    assert all(st == 0 for text, st in cells if text != "37G")


# -- what the view may contact -------------------------------------------------------------------
@pytest.fixture
def service(tmp_path):
    snapshot, found = snapshot_of(tmp_path)
    page, csp = dashboard.page_bytes()
    httpd = dashboard.ThreadingHTTPServer(("127.0.0.1", 0), dashboard.make_handler(found, page, csp))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1], found
    httpd.shutdown()
    httpd.server_close()


def arguments(**given):
    values = {"once": True, "details": True, "events": False, "width": 100, "url": None, "password_file": None,
              "passive": False}
    values.update(given)
    return argparse.Namespace(**values)


def test_the_view_connects_to_the_local_dashboard_service_only(service, monkeypatch, capsys):
    port, found = service
    connected = []
    real_connect = socket.socket.connect

    def connect(sock, address):
        connected.append(address[:2])
        return real_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "sendto", lambda *a: pytest.fail("the view sent a datagram"))
    found.last_view = None
    assert tui.main(arguments(url=f"http://127.0.0.1:{port}")) == 0
    out = capsys.readouterr().out
    assert out.startswith(" test · 4 serving") and "tp2-6-7" in out
    assert connected and set(connected) == {("127.0.0.1", port)}
    assert found.last_view is not None              # an open view counts as a viewer, as the page does


def test_a_passive_view_leaves_the_service_reads_unchanged(service, capsys):
    port, found = service
    found.last_view = None
    assert tui.main(arguments(url=f"http://127.0.0.1:{port}", passive=True)) == 0
    assert found.last_view is None and "4 serving" in capsys.readouterr().out


@pytest.mark.parametrize("url", ["http://192.0.2.99:8090", "http://spark3.example:8090", "https://127.0.0.1:8090",
                                 "http://127.0.0.1:8000/v1/chat/completions", "http://user@127.0.0.1:8090",
                                 "http://127.0.0.1:8090/?model=x", "ftp://127.0.0.1/"])
def test_an_address_other_than_this_spark_is_refused_before_any_socket_opens(url, monkeypatch):
    for name in ("connect", "connect_ex", "sendto"):
        monkeypatch.setattr(socket.socket, name, lambda *a: pytest.fail(f"a socket was opened for {url}"))
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: pytest.fail(f"a name was resolved for {url}"))
    with pytest.raises(tui.NotLocal):
        tui.main(arguments(url=url))
    assert dashboard.main(["tui", "--once", "--url", url]) == 2


def test_this_sparks_fabric_addresses_count_as_local(tmp_path, monkeypatch):
    from runtime.host.test_dashboard import fabric_file
    monkeypatch.setattr(socket, "gethostname", lambda: "spark0")
    addresses = tui.local_addresses(fabric_file(tmp_path))
    assert addresses == {"127.0.0.1", "::1", "192.0.2.10", NODE_A_LAN}
    assert tui.require_local(f"http://{NODE_A_LAN}:8090/", addresses) == f"http://{NODE_A_LAN}:8090"
    with pytest.raises(tui.NotLocal):
        tui.require_local("http://192.0.2.11:8090", addresses)          # spark1's address
    monkeypatch.setattr(socket, "gethostname", lambda: "spark1")
    assert tui.local_addresses(fabric_file(tmp_path)) == {"127.0.0.1", "::1", "192.0.2.11"}


def test_the_default_address_is_the_services_own_bind(tmp_path, monkeypatch):
    from runtime.host.test_dashboard import fabric_file
    path = fabric_file(tmp_path)
    monkeypatch.setattr(socket, "gethostname", lambda: "spark0")
    assert tui.local_url(path) == f"http://{NODE_A_LAN}:{dashboard.DEFAULT_PORT}"
    monkeypatch.setattr(socket, "gethostname", lambda: "spark3")
    assert tui.local_url(path) == f"http://127.0.0.1:{dashboard.DEFAULT_PORT}"



def test_without_an_address_the_view_reads_loopback_then_the_services_own_address(tmp_path, monkeypatch):
    from runtime.host.test_dashboard import fabric_file
    path = fabric_file(tmp_path)
    monkeypatch.setattr(socket, "gethostname", lambda: "spark0")
    assert tui.default_urls(path) == ["http://127.0.0.1:8090", f"http://{NODE_A_LAN}:8090"]
    monkeypatch.setattr(socket, "gethostname", lambda: "spark3")         # another host: loopback only
    assert tui.default_urls(path) == ["http://127.0.0.1:8090"]


class RefusingLoopback:
    """Refuses connections to 127.0.0.1 and answers the snapshot elsewhere; records every URL."""

    def __init__(self, snapshot, status=None):
        self.answer, self.status, self.urls = StaticOpener(snapshot), status, []

    def open(self, request, timeout=None):
        import urllib.error
        self.urls.append(request.full_url)
        if "127.0.0.1" in request.full_url:
            if self.status:
                raise urllib.error.HTTPError(request.full_url, self.status, "Unauthorized", {}, None)
            raise urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))
        return self.answer.open(request, timeout)


def test_a_refused_loopback_read_passes_to_the_lan_address(tmp_path):
    snapshot, _ = snapshot_of(tmp_path)
    opener = RefusingLoopback(snapshot)
    reader = tui.Reader(["http://127.0.0.1:8090", f"http://{NODE_A_LAN}:8090"], opener=opener)
    reader.once()
    assert reader.state()[0]["schema"] == dashboard.SCHEMA and reader.base == f"http://{NODE_A_LAN}:8090"
    reader.once()                                   # the address that answered is read first from then on
    assert opener.urls == ["http://127.0.0.1:8090/api/status", f"http://{NODE_A_LAN}:8090/api/status",
                           f"http://{NODE_A_LAN}:8090/api/status"]
    refusing = tui.Reader(["http://127.0.0.1:8090", f"http://{NODE_A_LAN}:8090"],
                          opener=RefusingLoopback(snapshot, status=401))
    refusing.once()                                 # an HTTP error is an answer: no other address is tried
    snapshot_read, error, _ = refusing.state()
    assert snapshot_read is None and "401" in error and refusing.opener.urls == ["http://127.0.0.1:8090/api/status"]


def test_the_default_view_reads_the_service_on_loopback(service, monkeypatch, capsys):
    port, _ = service
    monkeypatch.setattr(tui, "LOOPBACK_URL", f"http://127.0.0.1:{port}")
    monkeypatch.setattr(tui, "local_url", lambda fabric_path=None: f"http://127.0.0.1:{port}")
    connected = []
    real_connect = socket.socket.connect
    monkeypatch.setattr(socket.socket, "connect", lambda sock, address: connected.append(address[:2]) or
                        real_connect(sock, address))
    assert tui.main(arguments(url=None)) == 0
    assert "4 serving" in capsys.readouterr().out and set(connected) == {("127.0.0.1", port)}

def test_the_module_reads_one_path_and_can_start_no_process():
    tree = ast.parse(Path(tui.__file__).read_text(encoding="utf-8"))
    imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                for alias in node.names}
    imported |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert not imported & {"subprocess", "pty", "paramiko", "asyncssh", "fabric", "pexpect", "multiprocessing"}
    calls = {f"{node.func.value.id}.{node.func.attr}" for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)}
    assert not {c for c in calls if c.startswith("os.") and c.split(".")[1].startswith(
        ("system", "popen", "exec", "spawn", "fork", "posix_spawn"))}
    assert not {c for c in calls if c.startswith("socket.") and c != "socket.gethostname"}
    paths = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)
             and node.value.startswith("/") and len(node.value) > 1}
    assert paths == {"/api/status", "/api/status?viewer=passive"}


def test_the_command_is_registered(capsys):
    with pytest.raises(SystemExit) as stopped:
        dashboard.main(["tui", "--help"])
    assert stopped.value.code == 0 and "--passive" in capsys.readouterr().out


# -- the curses loop -----------------------------------------------------------------------------
class FakeScreen:
    def __init__(self, keys, height=24, width=90):
        self.keys, self.size = list(keys), (height, width)
        self.frames, self.current = [], {}

    def getmaxyx(self):
        return self.size

    def erase(self):
        self.current = {}

    def addnstr(self, y, x, text, n, attrs):
        assert x + n <= self.size[1] - 1 and y < self.size[0], "wrote into the last column or past the screen"
        line = self.current.setdefault(y, {})
        line[x] = text[:n]

    def refresh(self):
        self.frames.append(["".join(part for _, part in sorted(self.current.get(y, {}).items()))
                            for y in range(self.size[0])])

    def timeout(self, ms):
        pass

    def getch(self):
        return self.keys.pop(0) if self.keys else ord("q")


def fake_curses(screen):
    import types
    module = types.ModuleType("curses")
    module.error = type("error", (Exception,), {})
    module.KEY_UP, module.KEY_DOWN = 259, 258
    module.A_NORMAL, module.A_BOLD, module.A_DIM, module.A_REVERSE = 0, 1 << 16, 1 << 17, 1 << 18
    for number, name in enumerate(("BLACK", "RED", "GREEN", "YELLOW", "BLUE", "MAGENTA", "CYAN", "WHITE")):
        setattr(module, "COLOR_" + name, number)
    module.curs_set = module.start_color = module.use_default_colors = lambda *a: None
    module.has_colors = lambda: True
    module.init_pair = lambda *a: None
    module.color_pair = lambda n: n << 8
    module.wrapper = lambda run: run(screen)
    return module


class StaticOpener:
    def __init__(self, snapshot):
        self.body, self.urls = json.dumps(snapshot).encode(), []

    def open(self, request, timeout=None):
        import io
        self.urls.append(request.full_url)

        class Body(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.close()
        return Body(self.body)


def test_the_curses_loop_draws_selects_expands_and_quits(tmp_path, monkeypatch):
    snapshot, _ = snapshot_of(tmp_path)
    keys = [-1, ord("j"), ord("j"), ord("d"), ord("e"), ord("-"), ord("="), ord("="), ord("r"), -1, ord("q")]
    screen = FakeScreen(keys)
    monkeypatch.setitem(__import__("sys").modules, "curses", fake_curses(screen))
    opener = StaticOpener(snapshot)
    reader = tui.Reader("http://127.0.0.1:8090", opener=opener)
    reader.once()
    assert tui.interactive(reader) == 0
    first, last = screen.frames[0], screen.frames[-1]
    assert first[0].startswith(" test · 4 serving") and "q quit" in first[-1]
    assert any(line.startswith(" ▾ tp2-4-5") for line in last)          # two down, then expanded
    assert any(line.startswith(" events (e to hide)") for line in last)
    assert reader.interval == dashboard.METRICS_VIEWED_SECONDS - 1
    assert set(opener.urls) == {"http://127.0.0.1:8090/api/status"}
    assert reader.stop.is_set()
