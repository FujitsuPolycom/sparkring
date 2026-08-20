"""GPU-free tests for the operator preflight."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import socket
import sys
import types
import unittest
from unittest.mock import patch

PLUGIN_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_ROOT / "src"))

from sparkring import preflight  # noqa: E402


def _by_name(checks):
    return {check.name: check for check in checks}


class PreflightTest(unittest.TestCase):
    def setUp(self) -> None:
        from sparkring.plugin import _VENDOR

        if _VENDOR not in sys.path:
            sys.path.insert(0, _VENDOR)
        import spark_tp4_query_row_provider

        spark_tp4_query_row_provider._ambient_cache.clear()

    def test_default_environment_widths_and_mode_pass(self) -> None:
        checks = _by_name(preflight.run_preflight({}))
        self.assertTrue(checks["env-widths"].passed)
        self.assertIn("6144", checks["env-widths"].detail)
        self.assertTrue(checks["env-mode"].passed)
        self.assertIn("disabled", checks["env-mode"].detail)

    def test_provider_unset_passes_with_generic_policy(self) -> None:
        checks = _by_name(preflight.run_preflight({}))
        check = checks["query-row-provider"]
        self.assertTrue(check.passed)
        self.assertIn("not configured", check.detail)
        self.assertIn("contiguous", check.detail)

    def test_configured_synthetic_provider_reports_rows(self) -> None:
        module = types.ModuleType("_preflight_rows_ok")
        module.provider_query_rows = lambda environ: [2, 5, 9]
        sys.modules["_preflight_rows_ok"] = module
        try:
            checks = _by_name(preflight.run_preflight(
                {"VLLM_SPARK_TP4_QUERY_ROW_PROVIDER": "_preflight_rows_ok"}
            ))
            check = checks["query-row-provider"]
            self.assertTrue(check.passed)
            self.assertIn("_preflight_rows_ok", check.detail)
            self.assertIn("3 rows", check.detail)
        finally:
            del sys.modules["_preflight_rows_ok"]

    def test_missing_provider_fails_with_named_diagnostic(self) -> None:
        checks = _by_name(preflight.run_preflight(
            {"VLLM_SPARK_TP4_QUERY_ROW_PROVIDER": "_preflight_rows_absent"}
        ))
        check = checks["query-row-provider"]
        self.assertFalse(check.passed)
        self.assertIn("VLLM_SPARK_TP4_QUERY_ROW_PROVIDER", check.detail)
        self.assertIn("_preflight_rows_absent", check.detail)
        self.assertEqual(check.severity, "required")

    def test_malformed_provider_output_fails_closed(self) -> None:
        module = types.ModuleType("_preflight_rows_bad")
        module.provider_query_rows = lambda environ: []
        sys.modules["_preflight_rows_bad"] = module
        try:
            checks = _by_name(preflight.run_preflight(
                {"VLLM_SPARK_TP4_QUERY_ROW_PROVIDER": "_preflight_rows_bad"}
            ))
            self.assertFalse(checks["query-row-provider"].passed)
        finally:
            del sys.modules["_preflight_rows_bad"]

    def test_bad_widths_fail(self) -> None:
        checks = _by_name(
            preflight.run_preflight(
                {"VLLM_SPARK_TP4_EAGER_WIDTHS": "abc"}
            )
        )
        self.assertFalse(checks["env-widths"].passed)
        self.assertIn("VLLM_SPARK_TP4_EAGER_WIDTHS",
                      checks["env-widths"].detail)

    def test_bad_mode_fails(self) -> None:
        checks = _by_name(
            preflight.run_preflight({"VLLM_SPARK_TP4_MODE": "yolo"})
        )
        self.assertFalse(checks["env-mode"].passed)

    def test_port_namespace_pass_and_collision_fail(self) -> None:
        good = _by_name(
            preflight.run_preflight({"VLLM_SPARK_TP4_MODE": "custom"})
        )
        self.assertTrue(good["port-namespace"].passed)
        bad = _by_name(
            preflight.run_preflight(
                {
                    "VLLM_SPARK_TP4_MODE": "custom",
                    "SPARK_TP4_CONTROL_PORT0": "9470",
                    "SPARK_TP4_CONTROL_PORT1": "9470",
                }
            )
        )
        self.assertFalse(bad["port-namespace"].passed)

    def test_hca_devices_with_fake_sysfs(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as sysfs:
            gid_dir = os.path.join(
                sysfs, "rocep1s0f0", "ports", "1", "gids"
            )
            os.makedirs(gid_dir)
            open(os.path.join(gid_dir, "3"), "w").close()
            environ = {"SPARK_TP4_DEVICE0": "rocep1s0f0"}
            with patch.object(preflight, "_SYSFS_INFINIBAND", sysfs):
                good = _by_name(preflight.run_preflight(environ))
                self.assertTrue(good["hca-devices"].passed)
                bad = _by_name(
                    preflight.run_preflight(
                        {"SPARK_TP4_DEVICE0": "rocep1s0f9"}
                    )
                )
                self.assertFalse(bad["hca-devices"].passed)

    def test_hca_devices_without_sysfs_is_info_pass(self) -> None:
        with patch.object(
            preflight, "_SYSFS_INFINIBAND", "/definitely/not/here"
        ):
            checks = _by_name(
                preflight.run_preflight(
                    {"SPARK_TP4_DEVICE0": "rocep1s0f0"}
                )
            )
        self.assertTrue(checks["hca-devices"].passed)
        self.assertEqual(checks["hca-devices"].severity, "info")

    def test_native_library_env_and_missing(self) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False) as handle:
            path = handle.name
        try:
            good = _by_name(
                preflight.run_preflight(
                    {"SPARK_TP4_LIBRARY": path}
                )
            )
            self.assertTrue(good["native-library"].passed)
        finally:
            os.unlink(path)
        with patch.object(
            preflight, "_NATIVE_LIBRARY", "/definitely/not/here.so"
        ):
            bad = _by_name(preflight.run_preflight({}))
        self.assertFalse(bad["native-library"].passed)

    def test_vllm_compat_stub(self) -> None:
        stub = types.ModuleType("sparkring._compat")
        stub.compat_report = lambda: {
            "ok": True,
            "vllm_version": "0.27.1",
            "communicator_path": "x:CudaCommunicator",
            "detail": "resolved",
        }
        with patch.dict(sys.modules, {"sparkring._compat": stub}):
            good = _by_name(preflight.run_preflight({}))
        self.assertTrue(good["vllm-compat"].passed)
        stub.compat_report = lambda: {
            "ok": False,
            "vllm_version": None,
            "communicator_path": None,
            "detail": "no vllm",
        }
        with patch.dict(sys.modules, {"sparkring._compat": stub}):
            bad = _by_name(preflight.run_preflight({}))
        self.assertFalse(bad["vllm-compat"].passed)

    def test_provider_raising_non_valueerror_stays_in_the_report(self) -> None:
        def explode(environ):
            raise RuntimeError("provider backend unavailable")

        module = types.ModuleType("_preflight_rows_raise")
        module.provider_query_rows = explode
        sys.modules["_preflight_rows_raise"] = module
        environ = {
            "VLLM_SPARK_TP4_QUERY_ROW_PROVIDER": "_preflight_rows_raise",
            "VLLM_SPARK_TP4_MODE": "custom",
        }
        try:
            checks = _by_name(preflight.run_preflight(environ))
        finally:
            del sys.modules["_preflight_rows_raise"]
        check = checks["query-row-provider"]
        self.assertFalse(check.passed)
        self.assertEqual(check.severity, "required")
        self.assertIn("VLLM_SPARK_TP4_QUERY_ROW_PROVIDER", check.detail)
        self.assertIn("_preflight_rows_raise", check.detail)
        self.assertIn("RuntimeError", check.detail)
        # The reservation plan resolves the same row policy, so it sees
        # the same exception and must also report instead of aborting.
        self.assertFalse(checks["port-namespace"].passed)
        self.assertIn("RuntimeError", checks["port-namespace"].detail)
        # Every later check still produced a result.
        self.assertIn("peer-reachability", checks)

    def test_main_json_survives_a_raising_provider(self) -> None:
        import contextlib
        import io

        def explode(environ):
            raise RuntimeError("provider backend unavailable")

        module = types.ModuleType("_preflight_rows_raise_cli")
        module.provider_query_rows = explode
        sys.modules["_preflight_rows_raise_cli"] = module
        environ = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("VLLM_SPARK", "SPARK_TP4"))
        }
        environ["VLLM_SPARK_TP4_QUERY_ROW_PROVIDER"] = (
            "_preflight_rows_raise_cli"
        )
        environ["VLLM_SPARK_TP4_MODE"] = "custom"
        stream = io.StringIO()
        try:
            with patch.dict(os.environ, environ, clear=True):
                with contextlib.redirect_stdout(stream):
                    code = preflight.main(["--json"])
        finally:
            del sys.modules["_preflight_rows_raise_cli"]
        self.assertEqual(code, 1)  # query-row-provider fails
        payload = json.loads(stream.getvalue())
        names = {entry["name"] for entry in payload}
        self.assertIn("query-row-provider", names)
        self.assertIn("peer-reachability", names)

    def test_peer_check_skipped_by_default(self) -> None:
        checks = _by_name(
            preflight.run_preflight(
                {"SPARK_TP4_PEER0": "192.0.2.1"}
            )
        )
        self.assertTrue(checks["peer-reachability"].passed)
        self.assertEqual(checks["peer-reachability"].severity, "info")

    def test_peer_reachable_over_loopback(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            checks = _by_name(
                preflight.run_preflight(
                    {
                        "SPARK_TP4_PEER0": "127.0.0.1",
                        "SPARK_TP4_CONTROL_PORT0": str(port),
                    },
                    connect=True,
                )
            )
        finally:
            listener.close()
        check = checks["peer-reachability"]
        self.assertTrue(check.passed)
        self.assertEqual(check.severity, "required")
        self.assertIn(f"127.0.0.1:{port}", check.detail)

    def test_refused_peer_counts_as_routable(self) -> None:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()  # nothing listens on this port now
        # How fast a host turns a closed port into a refusal is a host
        # property: loopback RST is immediate on Linux and takes about
        # two seconds on Windows. Raise the bound so this exercises the
        # refusal branch rather than the timeout branch anywhere.
        with patch.object(preflight, "_CONNECT_TIMEOUT_SECONDS", 10.0):
            checks = _by_name(
                preflight.run_preflight(
                    {
                        "SPARK_TP4_PEER0": "127.0.0.1",
                        "SPARK_TP4_CONTROL_PORT0": str(port),
                    },
                    connect=True,
                )
            )
        check = checks["peer-reachability"]
        self.assertTrue(check.passed)
        self.assertEqual(check.severity, "required")
        self.assertIn(f"127.0.0.1:{port}", check.detail)

    def test_unroutable_peer_is_a_required_failure(self) -> None:
        class _UnroutableSocket:
            """Every connect fails the way an absent route does.

            Substituted for socket.socket because reaching a genuinely
            unroutable address requires a host outside this checkout.
            """

            def __init__(self, family, kind):
                pass

            def settimeout(self, seconds):
                pass

            def connect(self, address):
                raise OSError("network is unreachable")

            def close(self):
                pass

        with patch.object(preflight.socket, "socket", _UnroutableSocket):
            checks = _by_name(
                preflight.run_preflight(
                    {"SPARK_TP4_PEER0": "127.0.0.1"}, connect=True
                )
            )
        check = checks["peer-reachability"]
        self.assertFalse(check.passed)
        self.assertEqual(check.severity, "required")
        self.assertIn("127.0.0.1:11000", check.detail)
        self.assertIn("network is unreachable", check.detail)

    def test_non_integer_control_port_is_reported_not_raised(self) -> None:
        checks = _by_name(
            preflight.run_preflight(
                {
                    "SPARK_TP4_PEER0": "127.0.0.1",
                    "SPARK_TP4_CONTROL_PORT0": "eleven-thousand",
                },
                connect=True,
            )
        )
        check = checks["peer-reachability"]
        self.assertFalse(check.passed)
        self.assertEqual(check.severity, "required")
        self.assertIn("SPARK_TP4_CONTROL_PORT0/1", check.detail)

    def _write_vendor(self, root, contents):
        """Write module files and return their true digests by name."""

        modules = {}
        for name, data in contents.items():
            with open(os.path.join(root, name), "wb") as handle:
                handle.write(data)
            modules[name] = hashlib.sha256(data).hexdigest()
        return modules

    def _write_manifest(self, root, modules):
        path = os.path.join(root, "MANIFEST.json")
        with open(path, "wb") as handle:
            handle.write(json.dumps({"modules": modules}).encode("utf-8"))
        return path

    def test_vendor_integrity_without_manifest_is_info_pass(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as root:
            absent = os.path.join(root, "MANIFEST.json")
            with patch.object(preflight, "_VENDOR_MANIFEST", absent):
                checks = _by_name(preflight.run_preflight({}))
        check = checks["vendor-integrity"]
        self.assertTrue(check.passed)
        self.assertEqual(check.severity, "info")

    def test_vendor_integrity_matches_recorded_digests(self) -> None:
        import tempfile

        # CRLF bytes: the digest covers the file as written, so a
        # newline-translating read would compute a different hash.
        contents = {
            "spark_a.py": b"x = 1\r\ny = 2\r\n",
            "spark_b.py": b"z = 3\n",
        }
        with tempfile.TemporaryDirectory() as root:
            modules = self._write_vendor(root, contents)
            manifest = self._write_manifest(root, modules)
            with patch.object(preflight, "_VENDOR_MANIFEST", manifest):
                checks = _by_name(preflight.run_preflight({}))
        check = checks["vendor-integrity"]
        self.assertTrue(check.passed)
        self.assertEqual(check.severity, "required")
        self.assertIn("2 vendored modules", check.detail)

    def test_vendor_integrity_fails_on_altered_and_missing(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as root:
            modules = self._write_vendor(root, {"spark_a.py": b"x = 1\n"})
            modules["spark_gone.py"] = "0" * 64
            manifest = self._write_manifest(root, modules)
            with open(os.path.join(root, "spark_a.py"), "wb") as handle:
                handle.write(b"x = 2\n")  # altered after the manifest
            with patch.object(preflight, "_VENDOR_MANIFEST", manifest):
                checks = _by_name(preflight.run_preflight({}))
        check = checks["vendor-integrity"]
        self.assertFalse(check.passed)
        self.assertEqual(check.severity, "required")
        self.assertIn("spark_a.py: sha256", check.detail)
        self.assertIn("spark_gone.py: missing", check.detail)

    def test_main_json_and_exit_codes(self) -> None:
        import contextlib
        import io

        stream = io.StringIO()
        environ = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("VLLM_SPARK", "SPARK_TP4"))
        }
        with patch.dict(os.environ, environ, clear=True):
            with patch.object(
                preflight, "_NATIVE_LIBRARY", "/definitely/not/here.so"
            ):
                with contextlib.redirect_stdout(stream):
                    code = preflight.main(["--json"])
        self.assertEqual(code, 1)  # native-library fails
        payload = json.loads(stream.getvalue())
        self.assertIn("native-library", {c["name"] for c in payload})


if __name__ == "__main__":
    unittest.main()
