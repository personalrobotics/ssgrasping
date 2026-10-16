"""Geodude picks a can off its work surface and places it upright elsewhere on it.

This is the first fixed-gripper baseline: robot-neutral ``grasp`` and ``place`` skills,
with every robot-specific choice in explicit data. Geodude's right arm and Robotiq 2F-140
are selected, and the test world is a primitive 66 mm can on Geodude's own work surface.
The data comes from the libraries that own it:
- grasp regions: sstsr's ``Robotiq2F140.grasp_cylinder_side`` templates at the can;
- placement regions: sstsr's ``StablePlacer.place_cylinder`` on a target spot;
- planning: sscbirrt, with ssik kinematics for the arm, at a validation resolution fine
  enough for the straight approach, lift, and descent; and a second, iteration-bounded
  sscbirrt for the skills' checks that a straight move also plans from a candidate goal.

A second run asks for a grasp where there is no can, and must fail ``acquired`` with
nothing attached. The evidence is ``grasp_place.json``: every stage, with its time,
scene revision, held and attached state, and the kind of evidence it rests on, and the
seeds each check tried. Two runs write the same bytes.

Reproduce with ``uv run pytest tests/test_grasp_place.py``.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sscbirrt import CBiRRTConfig
from ssrobot import (
    Pose,
    RobotContext,
    Tsr,
    WrenchGuard,
    WrenchPart,
    load_installed_package,
)
from ssrobot.mujoco import MujocoRuntime
from ssrobot.planning import CBiRRT
from tsr.hands.parallel_jaw import Robotiq2F140
from tsr.placement.stable_placer import StablePlacer

from ssgrasping import GraspSpec, PlaceSpec, SkillResult, grasp, place

RADIUS, HEIGHT = 0.033, 0.123  # prl_assets' can geometric_properties
SURFACE_Z = 0.753  # the top of Geodude's work surface (vention_base)
CAN_XY = (0.35, -0.40)
TARGET_XY = (0.20, -0.50)
STANDOFF = 0.08
CHECK_ITERATIONS = 300  # feasible checks plan well within this; infeasible ones stop

SCENE = f"""<mujoco>
  <worldbody>
    <body name="can" pos="{CAN_XY[0]} {CAN_XY[1]} {SURFACE_Z + HEIGHT / 2}">
      <freejoint/>
      <geom type="cylinder" size="{RADIUS} {HEIGHT / 2}" mass="0.05"
            friction="0.6 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>
"""


def _matrix(pose: Pose) -> np.ndarray:
    w, x, y, z = pose.quat_wxyz
    T = np.eye(4)
    T[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
    T[:3, 3] = pose.position
    return T


def _pose(T: np.ndarray) -> Pose:
    R = T[:3, :3]
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return Pose(
        position=tuple(float(v) for v in T[:3, 3]),  # type: ignore[arg-type]
        quat_wxyz=(w / n, x / n, y / n, z / n),
    )


def _tsr(t: Any) -> Tsr:
    """An instantiated sstsr region as ssrobot's ``Tsr``."""
    bounds = [tuple(float(v) for v in row) for row in t.Bw]
    return Tsr(
        origin=_pose(t.T0_w),
        offset=_pose(t.Tw_e),
        position_bounds=tuple(bounds[:3]),  # type: ignore[arg-type]
        rotation_bounds=tuple(bounds[3:]),  # type: ignore[arg-type]
    )


def _grasp_spec(can: Pose) -> GraspSpec:
    """Every sstsr side grasp of the can, backed off along the approach axis; the skill
    chooses among them."""
    templates = Robotiq2F140().grasp_cylinder_side(RADIUS, HEIGHT)
    bottom = _matrix(can)
    bottom[:3, 3] -= bottom[:3, 2] * HEIGHT / 2  # sstsr frames a cylinder at its base
    back = np.eye(4)
    back[2, 3] = -STANDOFF
    return GraspSpec(
        manipulator="right_arm",
        end_effector="right_hand",
        gripper="right_gripper",
        opening_channel="right_gripper_opening",
        object="can",
        support="vention_base",
        pregrasps=[
            _tsr(dataclasses.replace(t, Tw_e=t.Tw_e @ back).instantiate(bottom)) for t in templates
        ],
        approach_m=STANDOFF,
        held_opening=(0.2, 0.95),
        lift_m=0.06,
    )


def _place_spec() -> PlaceSpec:
    """Upright on the work surface at the target, by sstsr's placer, which frames the
    can at its centre."""
    placer = StablePlacer(table_x=0.06, table_y=0.06)
    upright = next(
        t for t in placer.place_cylinder(RADIUS, HEIGHT, subject="can") if t.variant == "-z"
    )
    surface = np.eye(4)
    surface[:3, 3] = (*TARGET_XY, SURFACE_Z)
    region = upright.instantiate(surface)
    return PlaceSpec(
        manipulator="right_arm",
        end_effector="right_hand",
        gripper="right_gripper",
        object="can",
        target="vention_base",
        regions=[_tsr(region)],
        preplace_m=0.04,
        overshoot_m=0.01,
        guard=WrenchGuard(
            channel="right_wrist_wrench", part=WrenchPart.FORCE, threshold=3.0, tare=True
        ),
        retreat_m=0.08,
    )


def _stages(result: SkillResult) -> list[dict[str, Any]]:
    return [
        {
            "name": s.name,
            "succeeded": s.succeeded,
            "detail": s.detail,
            "time_ns": s.time_ns,
            "revision": s.revision,
            "held": s.held,
            "attached": s.attached,
            "evidence": s.evidence.value,
        }
        for s in result.stages
    ]


def _scenario(scene: Path, *, move_can: bool) -> dict[str, Any]:
    package = load_installed_package("geodude_assets")
    planner = CBiRRT.from_package(package, "right_arm", config=CBiRRTConfig(edge_resolution=0.01))
    checker = CBiRRT.from_package(
        package,
        "right_arm",
        config=CBiRRTConfig(edge_resolution=0.01, max_iterations=CHECK_ITERATIONS),
    )
    with RobotContext(package.description, MujocoRuntime(package, scene=scene)) as ctx:
        can = next(o.pose for o in ctx.snapshot().objects if o.name == "can")
        if move_can:  # grasp where the can is not: a miss
            can = Pose(
                position=(can.position[0], can.position[1] + 0.20, can.position[2]),
                quat_wxyz=can.quat_wxyz,
            )
        picked = grasp(ctx, planner, _grasp_spec(can), check_planner=checker)
        report: dict[str, Any] = {"grasp": _stages(picked), "grasp_values": picked.values}
        if picked.succeeded:
            placed = place(ctx, planner, _place_spec(), check_planner=checker)
            report["place"] = _stages(placed)
            report["place_values"] = placed.values
        report["attached_at_end"] = [a.object for a in ctx.scene.attachments]
    return report


def test_geodude_picks_and_places_a_can(artifacts: Path, tmp_path: Path) -> None:
    scene = tmp_path / "can.xml"
    scene.write_text(SCENE)
    success = _scenario(scene, move_can=False)
    missed = _scenario(scene, move_can=True)
    report = {"success": success, "missed": missed}
    (artifacts / "grasp_place.json").write_text(json.dumps(report, indent=2) + "\n")

    names = [s["name"] for s in success["grasp"]]
    assert names == [
        "open",
        "pregrasp",
        "approach",
        "close",
        "acquired",
        "attach",
        "lift",
        "restrict",
        "clear",
        "retained",
    ], success["grasp"]
    assert all(s["succeeded"] for s in success["grasp"]), success["grasp"]
    assert all(s["succeeded"] for s in success["place"]), success["place"]
    assert [s["name"] for s in success["place"]] == [
        "transfer",
        "allow_target",
        "descend",
        "release",
        "retreat",
        "placed",
    ]
    assert success["attached_at_end"] == []

    last = missed["grasp"][-1]
    assert (
        last["name"] == "acquired" and not last["succeeded"] and last["detail"].startswith("missed")
    )
    assert missed["attached_at_end"] == [] and "place" not in missed
