#!/usr/bin/env python3
"""Entry point. `python3 r1_mission_cli.py --help`, or symlink it as r1_mission."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r1_mission.node import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
