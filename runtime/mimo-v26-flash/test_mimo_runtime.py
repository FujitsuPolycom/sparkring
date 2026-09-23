"""GPU-free checks for the MiMo image contract and launch commands."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("mimo_check_image", HERE / "check_image.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
CONTRACT = json.loads((HERE / "b12x-image.json").read_text())


def metadata():
    return {"Architecture": "arm64", "Config": {"Labels": {
        f"org.sparkring.candidate.{name}-commit": source["revision"]
        for name, source in CONTRACT["sources"].items()
    }}}


class ImageContractTests(unittest.TestCase):
    def test_accepts_pinned_arm64_image(self):
        MODULE.validate(metadata(), CONTRACT)

    def test_rejects_base_wrong_revision_and_architecture(self):
        for bad in (
            {"Architecture": "arm64", "Config": {"Labels": None}},
            {**metadata(), "Architecture": "amd64"},
            {"Architecture": "arm64", "Config": {"Labels": {
                **metadata()["Config"]["Labels"],
                "org.sparkring.candidate.b12x-commit": "wrong",
            }}},
        ):
            with self.subTest(metadata=bad), self.assertRaises(ValueError):
                MODULE.validate(bad, CONTRACT)

    def test_dockerfile_matches_contract(self):
        dockerfile = (HERE / "Dockerfile").read_text()
        self.assertIn("FROM " + CONTRACT["base_image"]["reference"], dockerfile)
        for source in CONTRACT["sources"].values():
            self.assertIn(source["revision"], dockerfile)


@unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "requires POSIX bash")
class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        fake = self.bin / "docker"
        fake.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "with open(os.environ['DOCKER_CALLS'], 'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
            "if sys.argv[1:3] == ['image', 'inspect']:\n"
            "    m=json.loads(os.environ['IMAGE_METADATA'])\n"
            "    print(json.dumps([m]))\n"
            "else: print('test-container')\n"
        )
        fake.chmod(0o755)
        self.model = self.root / "model"
        (self.model / "dflash").mkdir(parents=True)
        for file in ("config.json", "model.safetensors.index.json", "dflash/config.json",
                     "dflash/dflash_draft_model.safetensors"):
            (self.model / file).write_text("{}")
        self.calls = self.root / "calls.jsonl"
        self.env = {**os.environ, "PATH": f"{self.bin}:{os.environ['PATH']}",
                    "DOCKER_CALLS": str(self.calls), "IMAGE_METADATA": json.dumps(metadata())}

    def invoke(self, kind, action="--run"):
        rank_env = self.root / "rank.env"
        rank_env.write_text(
            f"RANK=0\nHOST_IP=192.0.2.1\nMASTER_ADDR=192.0.2.1\nMGMT_IFNAME=test0\n"
            f"ROCE_HCA_PAIR=test0,test1\nNCCL_IB_HCA_LIST=test0:1,test1:1\n"
            f"MODEL_DIR='{self.model}'\nCACHE_DIR='{self.root / 'cache'}'\n"
            "IMAGE=test-image\nSIRCL=0\n"
        )
        return subprocess.run(["bash", str(HERE / f"launch-{kind}.sh"), action, str(rank_env)],
                              env=self.env, text=True, capture_output=True)

    def recorded(self):
        return [json.loads(line) for line in self.calls.read_text().splitlines()]

    def test_pair_and_ring_construct_native_b12x_commands(self):
        for kind in ("pair", "ring"):
            with self.subTest(kind=kind):
                result = self.invoke(kind)
                self.assertEqual(result.returncode, 0, result.stderr)
                command = [call for call in self.recorded() if call[0] == "run"][-1]
                self.assertEqual(command[command.index("--attention-backend") + 1], "B12X")
                self.assertEqual(command[command.index("--load-format") + 1], "safetensors")
                self.assertEqual(command[command.index("--kv-cache-dtype") + 1], "bfloat16")
                spec = json.loads(command[command.index("--speculative-config") + 1])
                self.assertEqual(spec["attention_backend"], "B12X")
                self.assertEqual(spec["kv_cache_dtype"], "auto")
                graphs = json.loads(command[command.index("--compilation-config") + 1])
                self.assertEqual(graphs["max_cudagraph_capture_size"], 64)
                self.assertIn("VLLM_USE_V2_MODEL_RUNNER=1", command)
                self.assertFalse(any("site-packages" in part for part in command))

    def test_wrong_image_fails_before_container_removal(self):
        self.env["IMAGE_METADATA"] = json.dumps({"Architecture": "arm64", "Config": {}})
        for kind in ("pair", "ring"):
            with self.subTest(kind=kind):
                result = self.invoke(kind)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(any(call[0] in ("rm", "run") for call in self.recorded()))

    def test_check_does_not_start_or_remove_containers(self):
        for kind in ("pair", "ring"):
            with self.subTest(kind=kind):
                result = self.invoke(kind, "--check")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertFalse(any(call[0] in ("rm", "run") for call in self.recorded()))


if __name__ == "__main__":
    unittest.main()
