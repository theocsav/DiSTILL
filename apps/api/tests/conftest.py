from pathlib import Path
import sys

API_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(API_ROOT))
sys.path.insert(0, str(REPO_ROOT))
