"""Plan or explicitly execute a profile-selected deployment command."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.common.launch import main  # noqa: E402

if __name__ == '__main__':
    main()
