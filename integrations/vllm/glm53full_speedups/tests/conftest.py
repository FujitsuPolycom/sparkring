"""Session setup: the CPU import environment for the speedups tests."""

import _env  # registers the package stand-ins
import pytest

_env.load()  # loads the image's parameter and linear modules


@pytest.fixture(scope="session")
def image_sources() -> "object":
    """The directory of the target image's Python sources (``_env.ROOT``)."""
    return _env.ROOT
