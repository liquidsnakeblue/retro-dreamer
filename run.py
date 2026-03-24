#!/usr/bin/env python3
"""Entry point for retro-dreamer."""

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.server import main

if __name__ == "__main__":
    main()
