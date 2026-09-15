"""Select an image build command from the maintained builder catalog."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.images.build import main  # noqa: E402

if __name__ == '__main__':
    main()
