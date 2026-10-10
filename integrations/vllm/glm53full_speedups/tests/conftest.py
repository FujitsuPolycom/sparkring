"""Session setup: the CPU import environment for the speedups tests."""

import speedups_image_env  # registers the package stand-ins
import pytest

speedups_image_env.load()  # loads the image's parameter and linear modules


@pytest.fixture(scope="session")
def image_sources() -> "object":
    """The directory of the target image's Python sources (``speedups_image_env.ROOT``)."""
    return speedups_image_env.ROOT
