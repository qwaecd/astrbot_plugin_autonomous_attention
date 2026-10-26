"""Test fixtures kept inside the workspace.

The DeepSeek sandbox on Windows does not grant the test process access to the
system temp directory, so tests use a local directory instead of pytest's
default ``tmp_path`` implementation.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

_TEST_ROOT = Path(__file__).resolve().parent / ".astrbot-root"
os.environ.setdefault("ASTRBOT_ROOT", str(_TEST_ROOT))
_TEST_ROOT.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def tmp_path() -> Path:
    root = Path(__file__).resolve().parent / ".tmp"
    root.mkdir(parents=True, exist_ok=True)
    path = root / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path
