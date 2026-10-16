from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def artifacts(request: pytest.FixtureRequest) -> Path:
    """A per-test directory for evidence, kept after the run."""
    base = Path(os.environ.get("SSGRASPING_ARTIFACTS") or ROOT / "artifacts")
    path = base / str(request.node.name)
    path.mkdir(parents=True, exist_ok=True)
    return path
