"""``python -m sparkring_sircl.vllm.serve``: the serve launcher's commands (see :mod:`.cli`)."""

import sys

from .cli import main

sys.exit(main())
