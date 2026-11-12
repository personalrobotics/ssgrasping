"""ADA's hand declares that its opening does not establish a grasp, and so grasps and
lifts a cylinder (#47).

ADA's bare-hand package (``ada_assets.models``) closes on the 4 cm reference cylinder,
standing on the over-bed table's top with nothing else in the scene. The JACO2's fingers
pinch it, then roll it into the palm, and the close ends ``reached_target``: the tendon
reaches its closed target around the object it holds. Two grasps of the same scene,
recorded in ``ada_grasp.json``, differ only in the profile's declaration:
- ``default``: with the opening taken to establish a grasp, the close reaching its target
  reads as a miss: the grasp reopens the hand on the cylinder it held and fails
  ``missed``.
- ``declared``: with ``opening_establishes_grasp=False``, the opening decides nothing. The
  grasp succeeds, with ``acquired`` and ``retained`` unobserved (``no_portable_signal``),
  and the lift ran with no opening guard. Scored by privileged state, for evaluation only,
  the cylinder rose with the hand by the lift, and the attachment monitor's report of the
  lift is kept beside the estimate.

The cylinder's pose is registered from its privileged pose channel, standing in for
perception. That the planned held geometry is not conservative once the cylinder is
seated in the palm is personalrobotics/ssrobot#242's, deferred: nothing here plans near
an obstacle.

Reproduce with ``uv run pytest tests/test_ada_grasp.py``.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from sscbirrt import CBiRRTConfig
from ssrobot import ObjectPoses, ObservationRequest, Pose, RobotContext, load_installed_package
from ssrobot.mujoco import MujocoRuntime
from ssrobot.planning import CBiRRT
from tsr import KinovaJaco2TwoFinger

from ssgrasping import (
    GraspObject,
    GraspRegion,
    Primitive,
    SkillResult,
    grasp,
    grasp_regions,
    run_skill,
    sstsr,
)
from ssgrasping.outcomes import EstimateReason, Reason, Transition, TransitionOutcome

RADIUS, HALF_HEIGHT = 0.02, 0.06
# ssrobot's references/scenes/ada-cylinder.xml: the over-bed table's collision top, and
# the cylinder where that scene stands it.
CYLINDER_POSE = ((0.3082, -0.6416, 0.7950), (0.270668, 0.0, 0.0, 0.962673))
SCENE = f"""<mujoco model="ada-cylinder-open">
  <worldbody>
    <body name="ada_table" pos="0.06 -0.48 0">
      <geom name="ada_table_top" type="box" pos="0 0 0.714" size="0.926 0.371 0.021"
            friction="0.8 0.005 0.0001" mass="10"/>
    </body>
    <body name="cylinder" pos="{" ".join(map(str, CYLINDER_POSE[0]))}"
          quat="{" ".join(map(str, CYLINDER_POSE[1]))}">
      <freejoint/>
      <geom type="cylinder" size="{RADIUS} {HALF_HEIGHT}" mass="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""
UP = (0.0, 0.0, 1.0)
LIFT_M = 0.06
LIFT_TOLERANCE_M = 0.01
"""How far short of the lift the cylinder's rise may fall: it settles in the hand."""
GRIPPER_TIMEOUT_NS = 30_000_000_000
"""The JACO2's close takes about 12 s to seat the cylinder: a failure bound, not a wait."""
MAX_TICKS = 100_000


def _grasp(scene: Path, **declaration: bool) -> tuple[SkillResult, dict[str, Any]]:
    package = load_installed_package("ada_assets.models")
    description = package.description
    (end_effector,) = description.end_effectors
    (manipulator,) = description.manipulators
    assert end_effector.gripper is not None
    planner = CBiRRT.from_package(
        package, manipulator.name, config=CBiRRTConfig(edge_resolution=0.01)
    )
    profile = sstsr.profile(
        KinovaJaco2TwoFinger(),
        manipulator=manipulator.name,
        end_effector=end_effector.name,
        gripper=end_effector.gripper,
        opening_channel="gripper_opening",
        **declaration,
    )
    runtime = MujocoRuntime(package, scene=scene)
    with RobotContext(description, runtime, object_poses=ObjectPoses.REGISTERED) as ctx:
        pose = ObservationRequest(channels=("cylinder.pose",))

        def height() -> float:
            return float(ctx.observe(pose).readings[0].value[2])  # type: ignore[index]

        observation = ctx.observe(pose)
        x, y, z, w, i, j, k = observation.readings[0].value  # type: ignore[misc]
        registration = Pose(position=(x, y, z), quat_wxyz=(w, i, j, k))
        ctx.register("cylinder", registration, stamp=observation.stamp, source="pose_channel")
        regions = grasp_regions(
            GraspObject(
                "cylinder",
                registration,
                (GraspRegion("body", Primitive("cylinder", (RADIUS, 2 * HALF_HEIGHT))),),
            ),
            profile,
            sstsr.SstsrTemplates(KinovaJaco2TwoFinger()),
            approach_m=0.05,
        )
        spec = regions.spec(
            support="ada_table",
            normal=UP,
            lift_m=LIFT_M,
            clearance_m=0.005,
            gripper_timeout_ns=GRIPPER_TIMEOUT_NS,
        )
        before = height()
        result = run_skill(ctx, grasp(ctx, planner, spec), max_ticks=MAX_TICKS)
        return result, {"rise_m": round(height() - before, 4)}


def _report(result: SkillResult, measured: dict[str, Any]) -> dict[str, Any]:
    return {
        "state": result.state.value,
        "stages": [
            {
                "name": s.name,
                "succeeded": s.succeeded,
                "reason": None if s.reason is None else s.reason.value,
                "detail": s.detail,
                "held": s.held,
            }
            for s in result.stages
        ],
        "transitions": [json.loads(json.dumps(dataclasses.asdict(t))) for t in result.transitions],
        "evaluations": [
            {
                "transition": e.estimate.transition.value,
                "estimate": e.estimate.outcome.value,
                "reference": e.reference.value,
                "source": e.source,
            }
            for e in result.evaluations
        ],
        "privileged": measured,
    }


def test_ada_grasps_and_lifts_under_its_declaration(artifacts: Path, tmp_path: Path) -> None:
    scene = tmp_path / "scene.xml"
    scene.write_text(SCENE)

    default, _ = _grasp(scene)  # the profile's default: the opening establishes a grasp
    assert not default.succeeded
    assert default.stage.name == "obstructed" and default.stage.reason is Reason.MISSED

    declared, measured = _grasp(scene, opening_establishes_grasp=False)
    assert declared.succeeded, declared.stage
    # The premise: in both, the JACO2's close around the cylinder reached its target.
    for result in (default, declared):
        close = result.values["gripper"]["close"]  # type: ignore[index]
        assert close["termination"] == "reached_target", close
    by_transition = {t.transition: t for t in declared.transitions}
    for transition in (Transition.ACQUIRED, Transition.RETAINED):
        estimate = by_transition[transition]
        assert estimate.outcome is TransitionOutcome.UNOBSERVED, estimate
        assert estimate.reason is EstimateReason.NO_PORTABLE_SIGNAL, estimate
    assert by_transition[Transition.RETAINED].evidence == ()  # no guard read the opening
    # Privileged, for evaluation only: the cylinder rose with the hand.
    assert measured["rise_m"] >= LIFT_M - LIFT_TOLERANCE_M, measured

    report = {
        "package": "ada_assets.models",
        "scene": "the over-bed table's top and the 4 cm cylinder, nothing else",
        "close_termination": close["termination"],
        "default": _report(default, {}),
        "declared": _report(declared, measured),
    }
    (artifacts / "ada_grasp.json").write_text(json.dumps(report, indent=2) + "\n")
