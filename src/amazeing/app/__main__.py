"""``python -m amazeing.app`` and the PyInstaller entry script."""

import sys

from amazeing.app import main

if __name__ == "__main__":
    sys.exit(main())
