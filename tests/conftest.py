"""Make the flat modules under app/ importable from the repo-root test run."""
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP))
