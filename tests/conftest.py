import sys
from pathlib import Path

# Let tests import the shared fixtures in `golden.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))
