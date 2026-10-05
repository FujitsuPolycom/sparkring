# Any Hugging Face model on two Sparks

Status: **research-only**. No generic deployment has been run on hardware.

This profile is the template that `sudo sparkring install --model OWNER/NAME`
uses on two Sparks (a pair, or one half of a four-Spark ring). It is not installed by
its own name.

```bash
sudo sparkring install --model OWNER/NAME[@REVISION] [--name SERVED_NAME] -- [vLLM arguments]
```

- **Model.** A public Hugging Face repository with safetensors weights and a
  `model.safetensors.index.json`. The plan pins every file of the revision
  (default: the default branch) by size and SHA-256, and the installation
  verifies each file as it does for SparkRing's own profiles.
- **Command.** SparkRing sets the parallel layout, the API on port 8000, the
  fabric settings that SparkRing's two Sparks profiles share, a context of at most
  32,768 tokens, 16 concurrent requests and 16 GiB of KV cache per Spark.
  Arguments after `--` go to vLLM and replace or add to these; SparkRing's own
  options (host, port, parallel sizes, node rank, master address, model path)
  are refused.
- **Settings.** `--context-length`, `--max-concurrency`, `--kv-cache-gib`,
  `--api-port` and `--api-bind` apply as they do to other profiles.

[Any Hugging Face model](../../docs/operations/install-reference.md#any-hugging-face-model)
describes the behavior and its limits.
