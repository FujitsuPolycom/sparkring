"""Optional Chromium check against a loopback-only app with synthetic snapshots."""

import copy
import os
from pathlib import Path
import socket
import threading
import time

from fastapi import FastAPI
import pytest

from test_status_views import observed_fixture
from sparkring_runtime_status.plugin import StatusPlugin


def demo_snapshot():
    doc = observed_fixture()
    values = {"tensor_parallel_size": 4, "decode_context_parallel_size": 1,
              "pipeline_parallel_size": 1, "model_type": "mimo_v2", "max_model_len": 262144,
              "quantization": "fp8", "model_dtype": "torch.bfloat16", "max_num_seqs": 16,
              "attention_backend": "B12X", "decoder_attention_modules": ["B12xPagedAttentionImpl"],
              "sampled_qk_head_dims": [192], "sampled_value_head_dims": [128], "fuse_act_quant": True,
              "speculative_method": "dflash", "num_speculative_tokens": 5,
              "draft_model_type": "eagle", "draft_tensor_parallel_size": 4,
              "draft_load_format": "safetensors", "draft_kv_cache_dtype": "bfloat16",
              "draft_sample_method": "greedy", "rejection_sample_method": "standard",
              "adaptive_verification": False, "cudagraph_mode": "FULL_AND_PIECEWISE",
              "max_cudagraph_capture_size": 96, "kv_transfer_enabled": False,
              "kv_connector": None, "cache_dtype": "bfloat16", "block_size": 128,
              "kv_cache_memory_bytes": 20 * 1024**3, "load_format": "b12x",
              "loader_read_mode": "bounce", "loader_io_threads": 8}
    values.update(tp_rocenante_enabled=True, tp_roce_allreduce_max_bytes=2097152,
                  tp_nccl_version=23203, tp_roce_hcas='rdma0,rdma1')
    doc["effective"].update({key: {"state": "known", "value": value} for key, value in values.items()})
    for key in ("tensor_parallel_size", "block_size", "load_format"):
        doc["configured"]["arguments"][key] = copy.deepcopy(doc["effective"][key])
    env = {"VLLM_ENABLE_ROCE_ALLREDUCE": True, "VLLM_MXFP8_LM_HEAD": False,
           "VLLM_LM_HEAD_A16": True, "NCCL_IB_PRESERVE_PCI_DOMAIN": True,
           "NCCL_IB_SUBNET_AWARE_ROUTING": True, "NCCL_ALGO": "Ring",
           "NCCL_PROTO": "LL,LL128,Simple", "NCCL_MIN_NCHANNELS": 4, "NCCL_MAX_NCHANNELS": 4,
           "VLLM_ROCE_ALLREDUCE_MAX_SIZE": 2097152, "VLLM_ROCE_ALLGATHER_MAX_SIZE": 16777216}
    doc["configured"]["environment"] = {key: {"state": "known", "value": value} for key, value in env.items()}
    prototype = doc["workers"]["ranks"][0]
    doc["workers"]["ranks"] = []
    for rank in range(4):
        row = copy.deepcopy(prototype)
        row["identity"]["rank"]["value"] = rank
        row["effective"] = copy.deepcopy(doc["effective"])
        row["effective"]["kernel_preparation"] = {"state": "known", "session_state": "ready"}
        row["configured"] = copy.deepcopy(doc["configured"])
        doc["workers"]["ranks"].append(row)
    doc["workers"].update(expected_count=4, received_count=4, missing_count=0,
                          cache_age_seconds=0.04, stale=False)
    return doc


def test_browser_refresh_pause_failure_recovery_and_mobile_layout():
    playwright = pytest.importorskip("playwright.sync_api")
    uvicorn = pytest.importorskip("uvicorn")
    document = demo_snapshot()
    with playwright.sync_playwright() as browser_runtime:
        executable = os.environ.get('SPARKRING_CHROMIUM_EXECUTABLE', browser_runtime.chromium.executable_path)
        if not Path(executable).is_file():
            pytest.skip("Chromium is not installed")
        calls = []
        class Snapshot:
            async def snapshot(self):
                calls.append(1)
                return copy.deepcopy(document)
        app = FastAPI()
        StatusPlugin().attach_router(app)
        app.state.sparkring_status_service = Snapshot()
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", ws="none"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        browser = None
        try:
            for _ in range(100):
                if server.started:
                    break
                time.sleep(0.05)
            assert server.started
            browser = browser_runtime.chromium.launch(executable_path=executable)
            page = browser.new_page(viewport={"width": 1440, "height": 1050})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.clock.install()
            url = f"http://127.0.0.1:{port}/v1/sparkring/status/view"
            page.goto(url)
            assert "SparkRing shared-2026.09.4-rc.4" in page.title()
            assert page.locator(".card").first.inner_text().startswith("Worker reports\n4/4")
            assert page.locator("#report").evaluate("el => el.scrollWidth <= el.clientWidth")
            assert '32.0 GiB' in page.locator('#nodes').inner_text()
            assert '200.0 Gb/s' in page.locator('#links').inner_text()
            assert '13.4.2' in page.locator('#versions').inner_text()
            initial = len(calls)
            with page.expect_response(url):
                page.clock.run_for(5100)
            playwright.expect(page.locator("#updated")).to_contain_text("Updated")
            assert len(calls) == initial + 1
            page.locator("#auto").uncheck()
            page.clock.run_for(15000)
            assert len(calls) == initial + 1
            page.locator("#group-1 > summary").click()
            assert not page.locator("#group-1").evaluate("el => el.open")
            document["workers"]["stale"] = True
            page.locator("#refresh").click()
            playwright.expect(page.locator(".card").first).to_contain_text("stale")
            assert not page.locator("#group-1").evaluate("el => el.open")
            page.route(url, lambda route: route.abort())
            page.locator("#refresh").click()
            playwright.expect(page.locator("#connection")).to_contain_text("previous snapshot")
            assert "outdated" in page.locator("#report").get_attribute("class")
            page.unroute(url)
            document["workers"]["stale"] = False
            page.locator("#refresh").click()
            playwright.expect(page.locator("#connection")).to_have_text("")
            assert "outdated" not in (page.locator("#report").get_attribute("class") or "")
            page.locator("#group-1 > summary").click()
            output = os.environ.get("SPARKRING_BROWSER_ARTIFACTS")
            if output:
                Path(output).mkdir(parents=True, exist_ok=True)
                page.locator('#links').screenshot(path=str(Path(output) / 'status-transport.png'))
                page.locator('#nodes').screenshot(path=str(Path(output) / 'status-nodes.png'))
                page.evaluate('window.scrollTo(0, 0)')
                page.screenshot(path=str(Path(output) / "status-desktop.png"))
            page.set_viewport_size({"width": 390, "height": 844})
            page.evaluate('window.scrollTo(0, 0)')
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            if output:
                page.screenshot(path=str(Path(output) / "status-mobile.png"))
            assert not errors
        finally:
            if browser:
                browser.close()
            server.should_exit = True
            thread.join(timeout=5)
            sock.close()
            assert not thread.is_alive()
