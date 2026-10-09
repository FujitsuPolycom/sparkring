"""Add the GLM-5.3 model-side speedup plugins to a kraken-line image as one layer.

The layer stacks on the kraken serving image whose vLLM and B12X Python
sources read the GLM-5.3-Flash CSF checkpoint
(``sparkring-dev/kraken:csf-sircl-libsircl-20261008``). It adds three vLLM
general plugins to the serving interpreter's site-packages, each with a
dist-info directory whose entry point registers it in ``vllm.general_plugins``;
vLLM loads a plugin only when ``VLLM_PLUGINS`` names it, and each plugin is
off until its own environment flags select it:

- ``glm_dsa_indexer_split`` 1.1.0
  (``integrations/vllm/glm_dsa_indexer_split``): each decode context parallel
  group scores and merges its block of a step's DSA indexer prefill rows, and
  one all-gather over the tensor-parallel group fills the top-k buffer for
  every row on every rank (``GLM_DSA_INDEXER_SPLIT=1``);
- ``glm53full_speedups`` 1.1.0 (``integrations/vllm/glm53full_speedups``):
  the fused ``q_a``/``kv_a`` latent projection is column-parallel over TP8
  with one all-gather (``GLM53FULL_LATENT_SHARD=1``) and the MTP ``eh_proj``
  is row-parallel (``GLM53FULL_EH_PROJ_TP=1``);
- ``glm_dcp_decode_comm`` 2.0.1 (``integrations/vllm/glm_dcp_decode_comm``):
  exact changes to the DSA attention's decode context parallel collectives on
  a SIRCL DCP session, five items behind the ``GLM_DCP_DECODE_*`` flags (the
  query pack, the communication-stream overlap, the indexer ``wk`` overlap,
  the selection reuse and the fused all-to-all combine), with an audit mode
  that counts differing words against the image's computation
  (``GLM_DCP_DECODE_AUDIT=1``). Besides the image's ``vllm`` and ``b12x``
  files it pins the SIRCL files it relies on to SIRCL 0.3.2: on a parent
  whose SIRCL layer is another build it serves with its items off and
  refuses at startup when an item flag is on.

Every plugin wraps image functions at load time and refuses to run on any
other file version: it pins the SHA-256 of every ``vllm``, ``b12x`` or SIRCL
file it edits or relies on to the value it records for this image, so no image
file changes and the image's ``verify`` keeps passing. The layer
records every added file in the image's external-base receipt, so ``verify``
checks the plugins' bytes too, and writes the provenance receipt
``/opt/sparkring/receipts/derived-glm53-plugins.json``.

The layer declares the three plugins with their versions (``PLUGINS``, read
from the dist-info files it adds). From the parent's v3 lock (the lock the
libsircl layer wrote, ``runtime/images/libsircl_layer.py``) it derives a v3
lock that keeps the parent's SIRCL and libsircl layers unchanged and lists the
three plugins in ``vllm_plugins``; ``sparkring install`` refuses a profile
whose ``VLLM_PLUGINS`` names them on an image whose lock does not
(``runtime/common/image_lock.py``, ``plugin_problem``). From a v1 or v2 parent
lock it derives a lock of that schema, which cannot list the plugins.

Actions (none pushes or publishes an image):

- ``prepare --parent-lock LOCK --output CONTEXT``: write the Docker build
  context from the parent's lock and receipts (or the local parent image).
- ``record --context CONTEXT --image ID --name NAME --output LOCK``: probe the
  built image (each plugin's entry point and version; for a v3 lock, also the
  SIRCL and libsircl layers it keeps), check the layer as installation does,
  admit it for every profile of the parent lock and write the derived lock.
- ``build --context CONTEXT --tag TAG --name NAME --output LOCK``: tag the
  parent, build, then record.

Status: research-only. The plugins' CPU tests run against the image's own
sources. One image built from this layer with the first two plugins only
(``af06e272``) served the settings of profile ``glm53-nvfp4-tp8`` on one
eight-Spark ring (``profiles/glm53-nvfp4-tp8/README.md``); no image with all
three plugins has been built, and no installation from a lock that ``record``
wrote has run.
"""
from __future__ import annotations

import configparser
from email.parser import BytesParser
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from runtime.images.derived_layer import Layer, main, sha  # noqa: E402

SITE = "/usr/local/lib/python3.12/dist-packages/"
SPLIT = "integrations/vllm/glm_dsa_indexer_split/"
SPEEDUPS = "integrations/vllm/glm53full_speedups/"
DCP = "integrations/vllm/glm_dcp_decode_comm/"
DCP_MODULES = ("__init__.py", "_scatter_pack_cute.py", "kernels.py", "layout.py", "reference.py", "runtime.py")
DCP_DIST_INFO = "glm_dcp_decode_comm-2.0.1.dist-info/"

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
    **{SITE + "glm_dcp_decode_comm/" + name: DCP + "glm_dcp_decode_comm/" + name for name in DCP_MODULES},
    **{SITE + DCP_DIST_INFO + name: DCP + "dist-info/" + DCP_DIST_INFO + name
       for name in ("METADATA", "entry_points.txt", "top_level.txt")},
}


def replace(read, receipt):
    return {target: (ROOT / source).read_bytes() for target, source in ADDED.items()}


def plugins():
    """Each added plugin's ``vllm.general_plugins`` entry-point name -> the version of its distribution,
    read from the dist-info files the layer adds."""
    found = {}
    for target, source in ADDED.items():
        if not target.endswith(".dist-info/entry_points.txt"):
            continue
        points = configparser.ConfigParser(interpolation=None)
        points.optionxform = str
        points.read_string((ROOT / source).read_text(encoding="utf-8"))
        metadata = BytesParser().parsebytes((ROOT / source).with_name("METADATA").read_bytes())
        for name in points["vllm.general_plugins"]:
            found[name] = metadata["Version"]
    return dict(sorted(found.items()))


PLUGINS = plugins()


LAYER = Layer(
    name="glm53-plugins",
    purpose="the glm_dsa_indexer_split, glm53full_speedups and glm_dcp_decode_comm vLLM general plugins for "
            "GLM-5.3 at TP8",
    replace=replace,
    provenance="/opt/sparkring/receipts/derived-glm53-plugins.json",
    pins={target: (None, sha((ROOT / source).read_bytes())) for target, source in ADDED.items()},
    plugins=PLUGINS,
)

if __name__ == "__main__":
    main(LAYER)
