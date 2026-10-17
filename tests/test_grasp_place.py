"""Geodude picks a can off its work surface and places it upright elsewhere on it.

This is the first fixed-gripper baseline: robot-neutral ``grasp`` and ``place`` skills,
with every robot-specific choice in explicit data. Geodude's right arm and Robotiq 2F-140
are selected, and the test world is a primitive 66 mm can on Geodude's own work surface.
The data comes from the libraries that own it:
- grasp regions: sstsr's ``Robotiq2F140.grasp_cylinder_side`` templates at the can;
- placement regions: sstsr's ``StablePlacer.place_cylinder`` on a target spot, with its
  yaw bounded so the final check tests all six bounds;
- planning: sscbirrt, with ssik kinematics for the arm, at a validation resolution fine
  enough for the straight approach, lift, and descent; and a second, iteration-bounded
  sscbirrt for the skills' checks that a straight move also plans from a candidate goal.

Each scenario is one contract case, recorded in ``grasp_place.json`` with every stage's
time, scene revision, held and attached state, and evidence kind, and the attachment
the skill left declared:
- ``success``: grasp, then place, on a manual clock; run twice, byte-identical.
- ``missed``: a grasp where there is no can fails ``acquired`` with nothing attached.
- ``preflight``: an unknown channel fails ``preflight`` before anything is commanded.
- ``external``: the same grasp on an externally clocked runtime, driven by ``run_skill``
  through ``update()`` alone.
- ``cancelled``: an external-clock grasp cancelled while its approach runs.
- ``drifted``: the can drifts about 0.3 mm in the fingers during the lift, past a
  0.1 mm attachment monitor. ``restrict`` refuses to re-declare an attachment the
  monitor reports has left its transform, so the report is not discarded, and the
  attachment is kept. (In ``success``, the transform declared at ``attach`` is the one
  the attachment still carries after ``restrict``.)
- ``unconfirmed_release``: the gripper opens only halfway, so ``release`` is not
  confirmed and the object stays attached.
- ``trajectory_only``: a planner that returns trajectories without paths still grasps.
The environment the artifact was made in is recorded with it; byte-identity is claimed
within one environment.

Reproduce with ``uv run pytest tests/test_grasp_place.py``.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import math
import platform
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from sscbirrt import CBiRRTConfig
from ssrobot import (
    ClockMode,
    Planner,
    PlanResult,
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

from ssgrasping import (
    GraspSpec,
    PlaceSpec,
    SkillExecution,
    SkillResult,
    SkillState,
    grasp,
    place,
    run_skill,
)

RADIUS, HEIGHT = 0.033, 0.123  # prl_assets' can geometric_properties
SURFACE_Z = 0.753  # the top of Geodude's work surface (vention_base)
UP = (0.0, 0.0, 1.0)  # the work surface's outward normal
CAN_XY = (0.35, -0.40)
TARGET_XY = (0.20, -0.50)
TARGET_YAW = (-0.5, 0.5)
STANDOFF = 0.08
CHECK_ITERATIONS = 300  # feasible checks plan well within this; infeasible ones stop
MAX_TICKS = 200_000
# The attachment monitor's rotation tolerance: a cylinder pinched between two pads swings
# about the finger axis as the arm accelerates (0.101 rad at the peak of the transfer,
# settling back to under 0.01 rad), and that swing is still a held can.
SWING_TOLERANCE_RAD = 0.2
# The guarded descent's force threshold: the tared wrist force peaks near 3 N as the arm
# speeds up and slows down in free motion, and passes 12 N within 20 ms of the can
# touching the surface, rising to about 30 N.
CONTACT_N = 8.0
DRIFT_TOLERANCE_M = 1e-4  # below the ~0.3 mm the can drifts in the fingers on the lift


def _scene() -> str:
    """A 50 g can on the work surface."""
    return f"""<mujoco>
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


def _grasp_spec(can: Pose, **changes: Any) -> GraspSpec:
    """Every sstsr side grasp of the can, backed off along the approach axis; the skill
    chooses among them."""
    templates = Robotiq2F140().grasp_cylinder_side(RADIUS, HEIGHT)
    bottom = _matrix(can)
    bottom[:3, 3] -= bottom[:3, 2] * HEIGHT / 2  # sstsr frames a grasped cylinder at its base
    back = np.eye(4)
    back[2, 3] = -STANDOFF
    spec = GraspSpec(
        manipulator="right_arm",
        end_effector="right_hand",
        gripper="right_gripper",
        opening_channel="right_gripper_opening",
        object="can",
        support="vention_base",
        normal=UP,
        pregrasps=[
            _tsr(dataclasses.replace(t, Tw_e=t.Tw_e @ back).instantiate(bottom)) for t in templates
        ],
        approach_m=STANDOFF,
        held_opening=(0.2, 0.95),
        lift_m=0.06,
    )
    return dataclasses.replace(spec, **changes)


def _place_spec(**changes: Any) -> PlaceSpec:
    """Upright on the work surface at the target, by sstsr's placer, which frames the
    can at its centre; its yaw bounded to ``TARGET_YAW``."""
    placer = StablePlacer(table_x=0.06, table_y=0.06)
    upright = next(
        t for t in placer.place_cylinder(RADIUS, HEIGHT, subject="can") if t.variant == "-z"
    )
    surface = np.eye(4)
    surface[:3, 3] = (*TARGET_XY, SURFACE_Z)
    region = _tsr(upright.instantiate(surface))
    roll, pitch, _ = region.rotation_bounds
    region = dataclasses.replace(region, rotation_bounds=(roll, pitch, TARGET_YAW))
    spec = PlaceSpec(
        manipulator="right_arm",
        end_effector="right_hand",
        gripper="right_gripper",
        opening_channel="right_gripper_opening",
        object="can",
        target="vention_base",
        normal=UP,
        regions=[region],
        preplace_m=0.04,
        overshoot_m=0.01,
        terminal_m=0.01,
        guard=WrenchGuard(
            channel="right_wrist_wrench", part=WrenchPart.FORCE, threshold=CONTACT_N, tare=True
        ),
        retreat_m=0.08,
    )
    return dataclasses.replace(spec, **changes)


class ExternalClock:
    """A test runtime: Geodude in MuJoCo, declared externally clocked, that advances one
    control tick each time it is polled. The context cannot step it; only updates move
    time, which keeps the run deterministic."""

    def __init__(self, inner: MujocoRuntime) -> None:
        self._inner = inner

    def open(self, description: Any) -> Any:
        return dataclasses.replace(self._inner.open(description), clock_mode=ClockMode.EXTERNAL)

    def poll(self) -> Any:
        self._inner.step()
        return self._inner.poll()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class TrajectoryOnly:
    """A planner adapter that returns its successes as trajectories without paths, as
    the neutral planning contract allows, each validated on its own by the planner's
    validator."""

    def __init__(self, inner: Planner) -> None:
        self._inner = inner

    def plan(
        self, request: Any, scene: Any, *, abort: Callable[[], bool] | None = None
    ) -> PlanResult:
        result = self._inner.plan(request, scene, abort=abort)
        if not result.succeeded or result.path is None or result.trajectory is None:
            return result
        report = self._inner.validator(request, scene).validate_trajectory(result.trajectory)
        return dataclasses.replace(result, path=None, validation=(report,))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _planners(package: Any) -> tuple[Planner, Planner]:
    def make(**config: Any) -> Planner:
        return CBiRRT.from_package(
            package, "right_arm", config=CBiRRTConfig(edge_resolution=0.01, **config)
        )

    return make(), make(max_iterations=CHECK_ITERATIONS)


def _report(result: SkillResult) -> dict[str, Any]:
    a = result.attachment
    return {
        "state": result.state.value,
        "stages": [
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
        ],
        "attachment": None
        if a is None
        else {
            "end_effector": a.end_effector,
            "transform": [list(a.transform.position), list(a.transform.quat_wxyz)],
            "allow": list(a.allow),
            "held": a.held,
        },
        "values": result.values,
    }


Session = Callable[[RobotContext, Planner, Planner, Pose], dict[str, Any]]


def _session(
    scene: Path,
    run: Session,
    *,
    external: bool = False,
    strip: bool = False,
    runtime_options: dict[str, Any] | None = None,
) -> Any:
    package = load_installed_package("geodude_assets")
    planner, checker = _planners(package)
    if strip:
        planner, checker = TrajectoryOnly(planner), TrajectoryOnly(checker)
    options = {"attach_tolerance_rad": SWING_TOLERANCE_RAD, **(runtime_options or {})}
    runtime: Any = MujocoRuntime(package, scene=scene, **options)
    if external:
        runtime = ExternalClock(runtime)
    with RobotContext(package.description, runtime) as ctx:
        can = next(o.pose for o in ctx.snapshot().objects if o.name == "can")
        return run(ctx, planner, checker, can)


def _drive(ctx: RobotContext, execution: SkillExecution) -> SkillResult:
    if ctx.info.clock_mode is ClockMode.MANUAL:
        return run_skill(ctx, execution, max_ticks=MAX_TICKS)
    return run_skill(ctx, execution, timeout_s=600.0, poll_interval_s=0.0)


def _pick(changes: dict[str, Any] | None = None) -> Session:
    def run(ctx: RobotContext, planner: Planner, checker: Planner, can: Pose) -> dict[str, Any]:
        spec = _grasp_spec(can, **(changes or {}))
        return {"grasp": _report(_drive(ctx, grasp(ctx, planner, spec, check_planner=checker)))}

    return run


def _pick_and_place(changes: dict[str, Any] | None = None) -> Session:
    def run(ctx: RobotContext, planner: Planner, checker: Planner, can: Pose) -> dict[str, Any]:
        picked = _drive(ctx, grasp(ctx, planner, _grasp_spec(can), check_planner=checker))
        report = {"grasp": _report(picked)}
        if picked.succeeded:
            spec = _place_spec(**(changes or {}))
            report["place"] = _report(_drive(ctx, place(ctx, planner, spec, check_planner=checker)))
        return report

    return run


def _missed(ctx: RobotContext, planner: Planner, checker: Planner, can: Pose) -> dict[str, Any]:
    elsewhere = Pose(
        position=(can.position[0], can.position[1] + 0.20, can.position[2]),
        quat_wxyz=can.quat_wxyz,
    )
    spec = _grasp_spec(elsewhere)
    return {"grasp": _report(_drive(ctx, grasp(ctx, planner, spec, check_planner=checker)))}


def _cancelled(ctx: RobotContext, planner: Planner, checker: Planner, can: Pose) -> dict[str, Any]:
    """Drive the grasp by hand on the external clock, and cancel it mid-approach."""
    execution = grasp(ctx, planner, _grasp_spec(can), check_planner=checker)
    cancelled_command = None
    for _ in range(MAX_TICKS):
        ctx.update()
        execution.advance()
        active = execution.active
        if execution.stage == "approach" and active is not None:
            execution.cancel()
            cancelled_command = active.status.state.value
            break
    assert execution.result is not None
    return {"grasp": _report(execution.result), "cancelled_command": cancelled_command}


def _environment() -> dict[str, str]:
    versions = {
        name: importlib.metadata.version(name)
        for name in ("ssgrasping", "ssrobot", "mujoco", "sscbirrt", "sstsr", "numpy")
    }
    return {"python": platform.python_version(), **versions}


def _states(report: dict[str, Any], part: str) -> list[tuple[str, bool]]:
    return [(s["name"], s["succeeded"]) for s in report[part]["stages"]]


def test_geodude_picks_and_places_a_can(artifacts: Path, tmp_path: Path) -> None:
    plain = tmp_path / "can.xml"
    plain.write_text(_scene())

    first = json.dumps(_session(plain, _pick_and_place()), indent=2)
    again = json.dumps(_session(plain, _pick_and_place()), indent=2)
    scenarios = {
        "success": json.loads(first),
        "missed": _session(plain, _missed),
        "preflight": _session(plain, _pick({"opening_channel": "right_gripper_width"})),
        "external": _session(plain, _pick(), external=True),
        "cancelled": _session(plain, _cancelled, external=True),
        "drifted": _session(
            plain, _pick(), runtime_options={"attach_tolerance_m": DRIFT_TOLERANCE_M}
        ),
        "unconfirmed_release": _session(plain, _pick_and_place({"open_opening": 0.5})),
        "trajectory_only": _session(plain, _pick(), strip=True),
    }
    report = {
        "environment": _environment(),
        "repeat_identical": first == again,
        "scenarios": scenarios,
    }
    (artifacts / "grasp_place.json").write_text(json.dumps(report, indent=2) + "\n")

    assert first == again, "two runs in one environment differ"

    success = scenarios["success"]
    assert success["grasp"]["state"] == SkillState.SUCCEEDED
    assert [n for n, _ in _states(success, "grasp")] == [
        "preflight", "open", "pregrasp", "approach", "close", "acquired",
        "attach", "lift", "restrict", "clear", "retained",
    ]  # fmt: skip
    declared = success["grasp"]["values"]["attach_transform"]
    assert success["grasp"]["attachment"]["transform"] == declared  # kept through restrict
    assert success["place"]["state"] == SkillState.SUCCEEDED, success["place"]["stages"]
    assert [n for n, _ in _states(success, "place")] == [
        "preflight", "transfer", "allow_target", "descend", "contact", "release", "retreat",
        "placed",
    ]  # fmt: skip
    assert success["place"]["attachment"] is None

    missed = scenarios["missed"]["grasp"]
    assert missed["state"] == SkillState.FAILED and missed["attachment"] is None
    assert missed["stages"][-1]["name"] == "acquired"
    assert missed["stages"][-1]["detail"].startswith("missed")

    pre = scenarios["preflight"]["grasp"]
    assert pre["state"] == SkillState.FAILED and [s["name"] for s in pre["stages"]] == ["preflight"]
    assert pre["stages"][0]["time_ns"] == 0  # nothing ran

    assert scenarios["external"]["grasp"]["state"] == SkillState.SUCCEEDED
    assert scenarios["external"]["grasp"]["stages"][-1]["name"] == "retained"

    cancelled = scenarios["cancelled"]
    assert cancelled["grasp"]["state"] == SkillState.CANCELLED
    assert cancelled["grasp"]["stages"][-1]["name"] == "approach"
    assert cancelled["cancelled_command"] == "canceled"
    assert cancelled["grasp"]["attachment"] is None

    drifted = scenarios["drifted"]["grasp"]
    assert drifted["state"] == SkillState.FAILED, drifted["stages"]
    assert drifted["stages"][-1]["name"] == "restrict"
    assert drifted["stages"][-1]["evidence"] == "privileged"
    assert drifted["stages"][-1]["held"] is True  # conservative: it may still be in hand
    assert drifted["attachment"] is not None and drifted["attachment"]["held"] is False

    unconfirmed = scenarios["unconfirmed_release"]["place"]
    assert unconfirmed["state"] == SkillState.FAILED, unconfirmed["stages"]
    assert unconfirmed["stages"][-1]["name"] == "release"
    assert unconfirmed["stages"][-1]["held"] is None
    assert unconfirmed["attachment"] is not None

    assert scenarios["trajectory_only"]["grasp"]["state"] == SkillState.SUCCEEDED
