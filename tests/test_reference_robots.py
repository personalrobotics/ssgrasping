"""Both reference robots open in MuJoCo from their installed packages (#2).

The evidence is ``reference_robots.json``: for Geodude and ADA, the package location,
the manipulators, end effectors, and grippers the skills will select explicitly, what
the runtime confirmed, and the time after ten steps. It shows the test environment can
host the two-robot proof; ssgrasping itself imports neither package.

Reproduce with ``uv run pytest tests/test_reference_robots.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ssrobot import RobotContext, load_installed_package
from ssrobot.mujoco import MujocoRuntime

STEPS = 10


def _open(module: str) -> dict[str, Any]:
    package = load_installed_package(module)
    description = package.description
    with RobotContext(description, MujocoRuntime(package)) as ctx:
        for _ in range(STEPS):
            ctx.step()
        return {
            "installed": "site-packages" in str(package.root),
            "robot": description.name,
            "manipulators": [m.name for m in description.manipulators],
            "end_effectors": [e.name for e in description.end_effectors],
            "grippers": [g.name for g in description.grippers],
            "confirmed_commands": len(ctx.info.commands),
            "confirmed_channels": len(ctx.info.channels),
            "time_ns": ctx.now.time_ns,
            "trajectory_period_ns": ctx.trajectory_contract.period_ns,
        }


def test_both_reference_robots_open_from_installed_packages(artifacts: Path) -> None:
    report = {module: _open(module) for module in ("geodude_assets", "ada_assets")}
    (artifacts / "reference_robots.json").write_text(json.dumps(report, indent=2) + "\n")
    for module, robot in report.items():
        assert robot["installed"], module
        assert robot["time_ns"] == STEPS * robot["trajectory_period_ns"], module
    assert len(report["geodude_assets"]["manipulators"]) == 4
    assert report["ada_assets"]["manipulators"] == ["arm"]
