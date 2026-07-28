import os
from pathlib import Path
import sys

import pytest

API_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(API_ROOT))
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def isolate_environment():
    """Restore os.environ after every test.

    Each module's create_client() helper configures the app by mutating the process
    environment and reloading settings. Without this, a variable one module sets
    (say DATA_UPLOADS_DIR) leaks into a module that does not set it, and that module
    then points at the previous test's tmp_path.
    """
    original = os.environ.copy()
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(original)
