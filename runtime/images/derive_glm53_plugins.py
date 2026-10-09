"""Add the GLM-5.3 model-side speedup plugins to a kraken-line image as one layer.

The layer stacks on the kraken serving image whose vLLM and B12X Python
sources read the GLM-5.3-Flash CSF checkpoint
(``sparkring-dev/kraken:csf-sircl-libsircl-20261008``). It adds two vLLM
general plugins to the serving interpreter's site-packages, each with a
dist-info directory whose entry point registers it in ``vllm.general_plugins``;
vLLM loads a plugin only when ``VLLM_PLUGINS`` names it, and each plugin is
off until its own environment flag selects it:

- ``glm_dsa_indexer_split`` 1.1.0
  (``integrations/vllm/glm_dsa_indexer_split``): each decode context parallel
  group scores and merges its block of a step's DSA indexer prefill rows, and
  one all-gather over the tensor-parallel group fills the top-k buffer for
  every row on every rank (``GLM_DSA_INDEXER_SPLIT=1``);
- ``glm53full_speedups`` 1.1.0 (``integrations/vllm/glm53full_speedups``):
  the fused ``q_a``/``kv_a`` latent projection is column-parallel over TP8
  with one all-gather (``GLM53FULL_LATENT_SHARD=1``) and the MTP ``eh_proj``
  is row-parallel (``GLM53FULL_EH_PROJ_TP=1``).

Every plugin wraps image functions at load time and refuses to run on any
other file version: it pins the SHA-256 of every ``vllm`` or ``b12x`` file it
edits or relies on to the value it records for this image, so no ``vllm`` or
``b12x`` file changes and the image's ``verify`` keeps passing. The layer
records every added file in the image's external-base receipt, so ``verify``
checks the plugins' bytes too, and writes the provenance receipt
``/opt/sparkring/receipts/derived-glm53-plugins.json``.

Actions (none pushes or publishes an image):

- ``prepare --parent-lock LOCK --output CONTEXT``: write the Docker build
  context from the parent's v3 lock and receipts (or the local parent image).
- ``record --context CONTEXT --image ID --name NAME --output LOCK``: probe the
  built image, check the layer as installation does, admit it for every
  profile of the parent lock and write the derived lock.
- ``build --context CONTEXT --tag TAG --name NAME --output LOCK``: tag the
  parent, build, then record.

Status: research-only; the plugins' CPU tests run against the image's own
sources, and no image built from this layer has been measured.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from runtime.images.derived_layer import Layer, main, sha  # noqa: E402

SITE = "/usr/local/lib/python3.12/dist-packages/"
SPLIT = "integrations/vllm/glm_dsa_indexer_split/"
SPEEDUPS = "integrations/vllm/glm53full_speedups/"

# Every added file: the site-packages path, its repository source and the
# SHA-256 of the repository bytes (the layer pins each result; additions have
# no inherited SHA-256).
ADDED = {
    SITE + "glm_dsa_indexer_split/__init__.py": SPLIT + "glm_dsa_indexer_split/__init__.py",
    SITE + "glm_dsa_indexer_split/layout.py": SPLIT + "glm_dsa_indexer_split/layout.py",
    SITE + "glm_dsa_indexer_split/runtime.py": SPLIT + "glm_dsa_indexer_split/runtime.py",
    SITE + "glm_dsa_indexer_split-1.1.0.dist-info/METADATA": SPLIT
    + "dist-info/glm_dsa_indexer_split-1.1.0.dist-info/METADATA",
    SITE + "glm_dsa_indexer_split-1.1.0.dist-info/entry_points.txt": SPLIT
    + "dist-info/glm_dsa_indexer_split-1.1.0.dist-info/entry_points.txt",
    SITE + "glm_dsa_indexer_split-1.1.0.dist-info/top_level.txt": SPLIT
    + "dist-info/glm_dsa_indexer_split-1.1.0.dist-info/top_level.txt",
    SITE + "glm53full_speedups/__init__.py": SPEEDUPS + "glm53full_speedups/__init__.py",
    SITE + "glm53full_speedups-1.1.0.dist-info/METADATA": SPEEDUPS
    + "dist-info/glm53full_speedups-1.1.0.dist-info/METADATA",
    SITE + "glm53full_speedups-1.1.0.dist-info/entry_points.txt": SPEEDUPS
    + "dist-info/glm53full_speedups-1.1.0.dist-info/entry_points.txt",
    SITE + "glm53full_speedups-1.1.0.dist-info/top_level.txt": SPEEDUPS
    + "dist-info/glm53full_speedups-1.1.0.dist-info/top_level.txt",
}


def replace(read, receipt):
    return {target: (ROOT / source).read_bytes() for target, source in ADDED.items()}


LAYER = Layer(
    name="glm53-plugins",
    purpose="the glm_dsa_indexer_split and glm53full_speedups vLLM general plugins for GLM-5.3 at TP8",
    replace=replace,
    provenance="/opt/sparkring/receipts/derived-glm53-plugins.json",
    pins={target: (None, sha((ROOT / source).read_bytes())) for target, source in ADDED.items()},
)

if __name__ == "__main__":
    main(LAYER)
