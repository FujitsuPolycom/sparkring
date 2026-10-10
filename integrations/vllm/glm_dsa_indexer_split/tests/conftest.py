"""Session setup: the CPU import environment, then the plugin, before the indexer module loads.

``register`` runs here with every prefill step eligible, so the image's
``b12x_indexer`` module is patched on its first import through the plugin's
import hook, as in a worker. The refusals run in fresh interpreters
(``test_register.py``).
"""

import atexit

import split_image_env  # noqa: F401  (must precede every vllm import; loads the image's modules)
import glm_dsa_indexer_split
import pytest
from glm_dsa_indexer_split import runtime

glm_dsa_indexer_split.register()
# pytest closes its capture streams before interpreter exit; the counters'
# exit report is a worker log line, not part of what the tests check.
atexit.unregister(runtime._log_at_exit)


@pytest.fixture(scope="session")
def image_sources() -> "object":
    """The directory of the target image's Python sources (``split_image_env.ROOT``)."""
    return split_image_env.ROOT
