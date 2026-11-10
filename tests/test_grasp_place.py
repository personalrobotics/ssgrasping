"""Geodude picks a can off its work surface and places it upright elsewhere on it.

This is the first fixed-gripper baseline: robot-neutral ``grasp`` and ``place`` skills,
with every robot-specific choice in explicit data. Geodude's right arm and Robotiq 2F-140
are selected, and the test world is a primitive 66 mm can on Geodude's own work surface.
The data comes from the libraries that own it:
- grasp regions: the can as a ``GraspObject`` cylinder at its observed pose, with a
  ``GraspingProfile`` for the 2F-140, and its side grasps from sstsr's templates
  (``ssgrasping.sstsr``);
- placement regions: sstsr's ``StablePlacer.place_cylinder`` on a target spot, with its
  yaw bounded so the final check tests all six bounds;
- planning: sscbirrt, with ssik kinematics for the arm, at a validation resolution fine
  enough for the straight approach, lift, and descent; and a second, iteration-bounded
  sscbirrt for the skills' checks that a straight move also plans from a candidate goal.

Each scenario is one contract case, recorded in ``grasp_place.json`` with every stage's
time, scene revision, held and attached state, evidence kind, and, for a failed stage,
its reason and cause, and the attachment the skill left declared:
- ``success``: grasp, then place, on a manual clock; run twice, byte-identical.
- ``missed``: a grasp where there is no can fails ``obstructed`` with nothing attached.
- ``obstructed``: a grasp where there is no can, but a fixed post stands between the
  fingers. Closed, they would touch the post, which is neither the object nor its
  support, so the close is refused before the fingers move: the grasp fails
  ``gripper_obstructed`` at ``close``, with nothing attached and the fingers open.
- ``stuck``: a full can stuck to the surface (glued, or in a spill). The close stops
  short on it, but the lift cannot pull free: the pads' friction on it, about 55 N,
  exceeds what the arm lifts, so the lift stalls short of its goal and fails
  ``command_failed``, with the can still reported held and attached, the conservative
  state. With real masses, a rigid can either rises with this hand or stops the lift;
  none slips out, so no scenario here reaches ``acquired``'s refutation.
- ``preflight``: an unknown channel fails ``preflight`` before anything is commanded.
- ``undeclared``: Geodude's package with its grippers' ``open_positions`` and
  ``closed_positions`` removed. No gripper command could be checked against the scene,
  so the grasp fails ``preflight`` before anything is commanded.
- ``external``: the same grasp on an externally clocked runtime, driven by ``run_skill``
  through ``update()`` alone.
- ``cancelled``: an external-clock grasp cancelled while its approach runs.
- ``drifted``: the can drifts about 0.3 mm in the fingers during the lift, past a
  0.1 mm attachment monitor. The monitor decides nothing: the grasp succeeds on the
  opening, and the lift's ``retained`` evaluation records the monitor's refutation.
  (In ``success``, the transform declared at ``attach`` is the one the attachment still
  carries after ``restrict``.)
- ``unconfirmed_release``: the gripper opens only halfway, so ``release`` is not
  confirmed and the object stays attached.
- ``beside_marker``: a marker stands upright on the work surface where ``success``'s
  release opens the hand's outer finger link (``MARKER_XY``), clear of the hand holding
  the can. The transfer that would release there is rejected, since the fingers would
  open into the marker anywhere in the descent's terminal band, and the can is placed
  from another, the release clear of it.
- ``trajectory_only``: a planner that returns trajectories without paths still grasps.
- ``gripper_timeout``: a 20 ms gripper deadline, shorter than the close, fails ``close``
  with ``timed_out``; the open, on a hand already open, ends within it.
- ``tare_settled``: whether joint rest settles the wrist enough to tare. ``success`` is
  run again, identically, up to the tick its arm is observed at rest before the descent
  (``success``'s ``tare_rest``), and the place is cancelled there, before the descent is
  submitted. The arm then holds for ``TARE_HORIZON_NS``, and the wrist's force is read
  every tick: it must stay within ``TARE_SETTLED_FRACTION`` of the guard's threshold of
  its value at rest, so taring then shifts the contact the guard detects by at most
  that fraction.
The lift and the place's transfer each estimate ``retained`` from an opening guard, with
the attachment monitor kept beside it for evaluation only (``success``). The place
records ``placed`` as ``unobserved``, scored against the can's privileged pose for
evaluation only. Every failed stage, in every scenario, says why as a ``Reason``, with a
cause in the vocabulary its reason names (the README's table); no successful stage does
(``_reasons``). The environment the artifact was made in is recorded with it;
byte-identity is claimed within one environment.

Reproduce with ``uv run pytest tests/test_grasp_place.py``.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import math
import platform
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from sscbirrt import CBiRRTConfig
from ssrobot import (
    ClockMode,
    GuardCause,
    ObjectPoses,
    ObservationRequest,
    Planner,
    PlanResult,
    PlanStatus,
    Pose,
    RestOutcome,
    RobotContext,
    Tsr,
    WrenchGuard,
    WrenchPart,
    load_installed_package,
    load_package,
)
from ssrobot.mujoco import MujocoRuntime
from ssrobot.planning import CBiRRT
from tsr.hands.parallel_jaw import Robotiq2F140
from tsr.placement.stable_placer import StablePlacer

from ssgrasping import (
    GraspObject,
    GraspRegion,
    GraspSpec,
    PlaceSpec,
    Primitive,
    Reason,
    SkillExecution,
    SkillResult,
    SkillState,
    Transition,
    grasp,
    grasp_regions,
    place,
    run_skill,
    sstsr,
)

RADIUS, HEIGHT = 0.033, 0.123  # prl_assets' can geometric_properties
# A full 12 fl oz can of these dimensions: 355 ml of drink (about 0.37 kg) and a 14 g
# aluminium can. Real loads only: a failure is made by a plausible physical cause.
CAN_MASS = 0.38
CAN_FRICTION = 0.6
SURFACE_Z = 0.753  # the top of Geodude's work surface (vention_base)
UP = (0.0, 0.0, 1.0)  # the work surface's outward normal
CAN_XY = (0.35, -0.40)
ELSEWHERE = 0.20  # how far in y from the can ``_missed`` aims
TARGET_XY = (0.20, -0.50)
TARGET_YAW = (-0.5, 0.5)
STANDOFF = 0.08
CHECK_ITERATIONS = 300  # feasible checks plan well within this; infeasible ones stop
MAX_TICKS = 200_000
GRIPPER_DEADLINE_NS = 20_000_000  # shorter than any close of the 2F-140 in this scene
# The attachment monitor's rotation tolerance: a cylinder pinched between two pads can swing
# about the finger axis as the arm accelerates, and that swing is still a held can. Earlier
# grasps swung it up to 0.101 rad at the peak of the transfer. Re-measured with the full
# 0.38 kg can at sstsr 4.2 (#3), over four placements: at most 0.0021 rad (0.0018 rad with
# the 50 g can these tests used before).
SWING_TOLERANCE_RAD = 0.2
# The guarded descent's force threshold. Measured at ssrobot 917795a, which reads the wrist
# at the state each tick reaches (#162), over four placements on Geodude's table: the
# tared force is 1.8-2.2 N as the descent starts, peaks at 4.4-5.1 N in free motion as
# the arm decelerates toward the line's end 3-6 mm above the surface, is 6-18 N on the
# first tick of contact, crosses 8 N within one tick of touching, and presses at 31-53 N.
# 8 N is 2.9 N (1.58x) above the free-motion peak; that peak scales with descent speed.
# Re-measured at ssrobot 640b9a4 with sscbirrt 3.11, which retimes the descent: the
# free-motion peak is 4.4-5.1 N over the three of those placements whose transfer keeps the
# can, so the margin stands. Again at geodude_assets 7b740c3 (torsional friction, the
# recalibrated gripper): 4.3-4.9 N. Those were with a 50 g can. Re-measured with the full
# 0.38 kg can at sstsr 4.2 (#3), over four placements: 5.4-5.6 N (4.7 N with 50 g), so 8 N
# is 2.4 N (1.44x) above the free-motion peak.
CONTACT_N = 8.0
TARE_HORIZON_NS = 1_000_000_000
"""How long ``tare_settled`` holds after joint rest: the fixed pre-tare wait it replaces."""
TARE_SETTLED_FRACTION = 0.1
"""How much of the guard's threshold the wrist's force may change by after joint rest."""
DRIFT_TOLERANCE_M = 1e-4  # below the ~0.3 mm the can drifts in the fingers on the lift
# Held transfers are planned at this fraction of the joints' velocity and acceleration
# limits. At sscbirrt 3.11's full-limit timing, the drop proof's transfer swung the wrist
# at 3.5 rad/s and threw the 50 g can it then used from the fingers. At 0.7 the can held in the drop
# proof, but moving the can's start by up to 1 mm either way turned it more than the
# monitor's 0.2 rad in the hand in 6 of 40 runs, on any machine; at 0.5, in none of 40,
# turning at most 0.056 rad.
CARRY_SCALING = 0.5
# Where ``success``'s first transfer releases the can with the right outer finger link
# opened (geodude_assets e9538a2): 38 mm outboard of it closed on the can, at the fingers'
# height. A marker (8 mm radius, 107 mm tall) standing there touches only the opened hand.
MARKER_XY = (0.0647, -0.4354)
MARKER_RADIUS, MARKER_HALF_HEIGHT = 0.008, 0.0535


def _scene(post: bool = False, stuck: bool = False, marker: bool = False) -> str:
    """A full can on the work surface; with ``post``, a fixed post of the can's size where
    ``_missed`` aims; with ``marker``, a marker standing at ``MARKER_XY``; with ``stuck``,
    the can held to the surface (glued, or stuck in a spill) by a weld, though it is still
    a free object to the runtime. No can this size
    slips from the 2F-140's grip by its weight: at about 46 N a pad, slipping at friction
    0.6 would take over 5.6 kg, denser than lead."""
    fixed = (
        f"""
    <body name="post" pos="{CAN_XY[0]} {CAN_XY[1] + ELSEWHERE} {SURFACE_Z + HEIGHT / 2}">
      <geom type="cylinder" size="{RADIUS} {HEIGHT / 2}"/>
    </body>"""
        if post
        else ""
    )
    if marker:
        fixed += f"""
    <body name="marker" pos="{MARKER_XY[0]} {MARKER_XY[1]} {SURFACE_Z + MARKER_HALF_HEIGHT}">
      <geom type="cylinder" size="{MARKER_RADIUS} {MARKER_HALF_HEIGHT}"/>
    </body>"""
    weld = '\n  <equality>\n    <weld body1="can"/>\n  </equality>' if stuck else ""
    return f"""<mujoco>
  <worldbody>
    <body name="can" pos="{CAN_XY[0]} {CAN_XY[1]} {SURFACE_Z + HEIGHT / 2}">
      <freejoint/>
      <geom type="cylinder" size="{RADIUS} {HEIGHT / 2}" mass="{CAN_MASS}"
            friction="{CAN_FRICTION} 0.005 0.0001"/>
    </body>{fixed}
  </worldbody>{weld}
</mujoco>
"""


def _tsr(t: Any) -> Tsr:
    """An instantiated sstsr region as ssrobot's ``Tsr``."""
    bounds = [tuple(float(v) for v in row) for row in t.Bw]
    return Tsr(
        origin=sstsr.pose(t.T0_w),
        offset=sstsr.pose(t.Tw_e),
        position_bounds=tuple(bounds[:3]),  # type: ignore[arg-type]
        rotation_bounds=tuple(bounds[3:]),  # type: ignore[arg-type]
    )


HAND = Robotiq2F140()
PROFILE = sstsr.profile(
    HAND,
    manipulator="right_arm",
    end_effector="right_hand",
    gripper="right_gripper",
    opening_channel="right_gripper_opening",
)


def _grasp_spec(can: Pose, **changes: Any) -> GraspSpec:
    """Every sstsr side grasp of the can at ``can``, backed off along the approach axis by
    the approach; the skill chooses among them."""
    regions = grasp_regions(
        GraspObject(
            "can",
            can,
            (GraspRegion("body", Primitive("cylinder", (RADIUS, HEIGHT)), strategies=("side",)),),
        ),
        PROFILE,
        sstsr.SstsrTemplates(HAND),
        approach_m=STANDOFF,
    )
    spec = regions.spec(support="vention_base", normal=UP, lift_m=0.06)
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
        carry_scaling=CARRY_SCALING,
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
                "reason": None if s.reason is None else s.reason.value,
                "cause": s.cause,
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
        "transitions": [json.loads(json.dumps(dataclasses.asdict(t))) for t in result.transitions],
        "evaluations": [
            {
                "transition": e.estimate.transition.value,
                "stage": e.estimate.stage,
                "estimate": e.estimate.outcome.value,
                "reference": e.reference.value,
                "source": e.source,
                "agrees": e.agrees,
            }
            for e in result.evaluations
        ],
    }


Session = Callable[[RobotContext, Planner, Planner, Pose], dict[str, Any]]


def _session(
    scene: Path,
    run: Session,
    *,
    external: bool = False,
    strip: bool = False,
    runtime_options: dict[str, Any] | None = None,
    wrap: Callable[[Any], Any] | None = None,
    registered: bool = False,
    package: Any = None,
) -> Any:
    package = package or load_installed_package("geodude_assets")
    planner, checker = _planners(package)
    if strip:
        planner, checker = TrajectoryOnly(planner), TrajectoryOnly(checker)
    options = {"attach_tolerance_rad": SWING_TOLERANCE_RAD, **(runtime_options or {})}
    runtime: Any = MujocoRuntime(package, scene=scene, **options)
    if external:
        runtime = ExternalClock(runtime)
    if wrap is not None:
        runtime = wrap(runtime)
    poses = ObjectPoses.REGISTERED if registered else ObjectPoses.LIVE
    with RobotContext(package.description, runtime, object_poses=poses) as ctx:
        if registered:  # each object where its privileged pose has it: the registration
            for name in ctx.info.objects:
                observation = ctx.observe(ObservationRequest(channels=(f"{name}.pose",)))
                x, y, z, w, i, j, k = observation.readings[0].value  # type: ignore[misc]
                at = Pose(position=(x, y, z), quat_wxyz=(w, i, j, k))
                ctx.register(name, at, stamp=observation.stamp, source="pose_channel")
        can = next(o.known_pose() for o in ctx.snapshot().objects if o.name == "can")
        return run(ctx, planner, checker, can)


def _undeclared(work: Path) -> Any:
    """Geodude's package, copied to ``work`` with its grippers' declared positions
    removed."""
    root = work / "geodude_assets"
    shutil.copytree(load_installed_package("geodude_assets").root, root)
    manifest = root / "ssrobot.toml"
    kept = [
        line
        for line in manifest.read_text().splitlines(keepends=True)
        if not line.startswith(("open_positions =", "closed_positions ="))
    ]
    manifest.write_text("".join(kept))
    package = load_package(root)
    assert all(not g.open_positions for g in package.description.grippers)
    return package


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


def _tare_settled(rest_ns: int) -> Session:
    def run(ctx: RobotContext, planner: Planner, checker: Planner, can: Pose) -> dict[str, Any]:
        picked = _drive(ctx, grasp(ctx, planner, _grasp_spec(can), check_planner=checker))
        placing = place(ctx, planner, _place_spec(), check_planner=checker)
        while ctx.now.time_ns < rest_ns:  # as success ran, to the tick the arm rested
            placing.advance()
            ctx.step()
        placing.cancel("measuring the wrist at rest, before the descent")
        assert placing.result is not None
        request = ObservationRequest(channels=("right_wrist_wrench",))

        def wrist() -> tuple[float, ...]:
            value = ctx.observe(request).readings[0].value
            assert isinstance(value, tuple)
            return value[:3]

        force = [wrist()]
        while ctx.now.time_ns < rest_ns + TARE_HORIZON_NS:
            ctx.step()
            force.append(wrist())
        drift = [math.dist(f, force[0]) for f in force]
        return {
            "grasp": _report(picked),
            "place": _report(placing.result),
            "rest_ns": rest_ns,
            "force_at_rest": list(force[0]),
            "max_change_n": max(drift),
            "change_at_horizon_n": drift[-1],
        }

    return run


def _missed(ctx: RobotContext, planner: Planner, checker: Planner, can: Pose) -> dict[str, Any]:
    elsewhere = Pose(
        position=(can.position[0], can.position[1] + ELSEWHERE, can.position[2]),
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


CAUSES: dict[str, set[str] | None] = {
    Reason.NOT_PLANNED: {s.value for s in PlanStatus},
    Reason.NOT_AT_REST: {o.value for o in RestOutcome},
    Reason.NO_CONTACT: {c.value for c in GuardCause},
    Reason.REFUSED: None,  # an ssrobot error or refusal code
    Reason.COMMAND_FAILED: None,  # an execution diagnostic code, or the execution state
    Reason.COMMAND_CANCELED: None,
    Reason.TRANSITION_REFUTED: {t.value for t in Transition},
    Reason.GRIPPER_OBSTRUCTED: {"open", "closed"},
}
"""The vocabulary of each reason's cause: an ssrobot enum's values, or any ssrobot code
(None). A reason not listed has no cause."""


def _reasons(report: Any) -> list[tuple[str, str, str | None]]:
    """Check every stage recorded anywhere in ``report``: a success says no reason, a
    failure a ``Reason``, with a cause only where its reason names one, in that
    vocabulary, and never prose. The failures' (stage, reason, cause)."""
    failures = []

    def walk(o: Any) -> None:
        if isinstance(o, dict):
            if {"name", "succeeded", "reason", "cause"} <= o.keys():
                reason, cause = o["reason"], o["cause"]
                if o["succeeded"]:
                    assert reason is None and cause is None, o
                else:
                    assert reason in set(Reason), o
                    if reason not in CAUSES:
                        assert cause is None, o
                    elif cause is not None:
                        vocabulary = CAUSES[reason]
                        assert re.fullmatch(r"[a-z][a-z0-9_]*", cause), o
                        assert vocabulary is None or cause in vocabulary, o
                    failures.append((o["name"], reason, cause))
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(report)
    return failures


def _failed(report: dict[str, Any]) -> tuple[str, str, str | None, bool | None, bool]:
    """A failed skill's deciding stage: its name, reason, cause, held, and attached."""
    s = report["stages"][-1]
    assert not s["succeeded"], s
    return s["name"], s["reason"], s["cause"], s["held"], s["attached"]


def _states(report: dict[str, Any], part: str) -> list[tuple[str, bool]]:
    return [(s["name"], s["succeeded"]) for s in report[part]["stages"]]


def test_geodude_picks_and_places_a_can(artifacts: Path, tmp_path: Path) -> None:
    plain = tmp_path / "can.xml"
    plain.write_text(_scene())
    posted = tmp_path / "post.xml"
    posted.write_text(_scene(post=True))
    stuck = tmp_path / "stuck.xml"
    stuck.write_text(_scene(stuck=True))
    marked = tmp_path / "marker.xml"
    marked.write_text(_scene(marker=True))

    first = json.dumps(_session(plain, _pick_and_place()), indent=2)
    again = json.dumps(_session(plain, _pick_and_place()), indent=2)
    scenarios = {
        "success": json.loads(first),
        "missed": _session(plain, _missed),
        "obstructed": _session(posted, _missed),
        "stuck": _session(stuck, _pick()),
        "preflight": _session(plain, _pick({"opening_channel": "right_gripper_width"})),
        "undeclared": _session(plain, _pick(), package=_undeclared(tmp_path)),
        "external": _session(plain, _pick(), external=True),
        "cancelled": _session(plain, _cancelled, external=True),
        "drifted": _session(
            plain, _pick(), runtime_options={"attach_tolerance_m": DRIFT_TOLERANCE_M}
        ),
        "unconfirmed_release": _session(plain, _pick_and_place({"open_opening": 0.5})),
        "beside_marker": _session(marked, _pick_and_place()),
        "trajectory_only": _session(plain, _pick(), strip=True),
        "gripper_timeout": _session(plain, _pick({"gripper_timeout_ns": GRIPPER_DEADLINE_NS})),
    }
    rested_ns = scenarios["success"]["place"]["values"]["tare_rest"]["end_ns"]
    scenarios["tare_settled"] = _session(plain, _tare_settled(rested_ns))
    report = {
        "environment": _environment(),
        "repeat_identical": first == again,
        "scenarios": scenarios,
    }
    (artifacts / "grasp_place.json").write_text(json.dumps(report, indent=2) + "\n")

    assert first == again, "two runs in one environment differ"
    _reasons(scenarios)

    success = scenarios["success"]
    assert success["grasp"]["state"] == SkillState.SUCCEEDED
    assert [n for n, _ in _states(success, "grasp")] == [
        "preflight", "open", "pregrasp", "approach", "close", "obstructed",
        "attach", "lift", "acquired", "restrict", "clear", "retained",
    ]  # fmt: skip
    # The lift and the place's transfer carry the can under an opening guard: each
    # confirms retained, from the opening alone, and the attachment monitor agrees. The
    # place's release leaves placed unobserved: no portable signal tells where the can
    # rests, and its privileged pose, which finds it placed, decides nothing.
    for part, stage in (("grasp", "lift"), ("place", "transfer")):
        estimate = success[part]["transitions"][0]
        assert (estimate["transition"], estimate["stage"], estimate["outcome"]) == (
            "retained", stage, "confirmed",
        ), estimate  # fmt: skip
        assert [e["evidence"] for e in estimate["evidence"]] == ["portable"]
        below = next(c for c in estimate["criteria"] if c["name"] == "below")
        assert estimate["evidence"][0]["value"][0] >= below["value"]
        assert success[part]["evaluations"][0] == {
            "transition": "retained",
            "stage": stage,
            "estimate": "confirmed",
            "reference": "confirmed",
            "source": "attachment_monitor",
            "agrees": True,
        }
    placed = success["place"]["transitions"][1]
    assert (placed["transition"], placed["stage"], placed["outcome"], placed["reason"]) == (
        "placed", "release", "unobserved", "no_portable_signal",
    )  # fmt: skip
    assert success["place"]["evaluations"][1] == {
        "transition": "placed",
        "stage": "release",
        "estimate": "unobserved",
        "reference": "confirmed",
        "source": "can.pose",
        "agrees": None,
    }
    declared = success["grasp"]["values"]["attach_transform"]
    assert success["grasp"]["attachment"]["transform"] == declared  # kept through restrict
    assert success["place"]["state"] == SkillState.SUCCEEDED, success["place"]["stages"]
    assert [n for n, _ in _states(success, "place")] == [
        "preflight", "transfer", "allow_target", "descend", "contact", "release", "retreat",
    ]  # fmt: skip
    assert success["place"]["attachment"] is None
    # The descent, which tares the wrist, starts only once the arm is observed at rest
    # after the transfer, by the runtime's declared criteria.
    tare = success["place"]["values"]["tare_rest"]
    transferred = next(s for s in success["place"]["stages"] if s["name"] == "allow_target")
    assert tare["outcome"] == "at_rest" and not tare["privileged"], tare
    assert tare["start_ns"] == transferred["time_ns"] < tare["end_ns"]
    assert tare["window"][1] == tare["end_ns"]

    # At rest, the wrist has settled enough to tare: its force changes by less than
    # TARE_SETTLED_FRACTION of the guard's threshold over the hold that follows.
    settled = scenarios["tare_settled"]
    assert settled["place"]["state"] == SkillState.CANCELLED
    assert _failed(settled["place"])[:2] == ("descend", Reason.CANCELLED)
    assert settled["max_change_n"] < TARE_SETTLED_FRACTION * CONTACT_N, settled

    missed = scenarios["missed"]["grasp"]
    assert missed["state"] == SkillState.FAILED and missed["attachment"] is None
    assert _failed(missed) == ("obstructed", Reason.MISSED, None, False, False)

    # Fingers that would close on something other than the object, or its support, are
    # never closed: the close is refused, named by what they would touch, and nothing is
    # attached. The fingers are still open, as the open stage left them.
    obstructed = scenarios["obstructed"]["grasp"]
    assert obstructed["state"] == SkillState.FAILED, obstructed["stages"]
    assert _failed(obstructed) == ("close", Reason.GRIPPER_OBSTRUCTED, "closed", None, False)
    assert obstructed["attachment"] is None
    assert "close" not in obstructed["values"]["gripper"]  # never commanded
    refused = obstructed["stages"][-1]
    assert refused["evidence"] == "planned"
    assert "post/" in refused["detail"], refused["detail"]
    stuck_can = scenarios["stuck"]["grasp"]
    assert stuck_can["state"] == SkillState.FAILED, stuck_can["stages"]
    assert "acquired" not in [s["name"] for s in stuck_can["stages"]]
    assert _failed(stuck_can) == ("lift", Reason.COMMAND_FAILED, "goal_not_reached", True, True)

    pre = scenarios["preflight"]["grasp"]
    assert pre["state"] == SkillState.FAILED and [s["name"] for s in pre["stages"]] == ["preflight"]
    assert pre["stages"][0]["time_ns"] == 0  # nothing ran
    assert _failed(pre)[:2] == ("preflight", Reason.UNAVAILABLE)
    undeclared = scenarios["undeclared"]["grasp"]
    assert [s["name"] for s in undeclared["stages"]] == ["preflight"], undeclared["stages"]
    assert undeclared["stages"][0]["time_ns"] == 0  # nothing ran
    assert _failed(undeclared)[:2] == ("preflight", Reason.UNAVAILABLE)
    assert "declares no open_positions" in undeclared["stages"][0]["detail"]

    assert scenarios["external"]["grasp"]["state"] == SkillState.SUCCEEDED
    assert scenarios["external"]["grasp"]["stages"][-1]["name"] == "retained"

    cancelled = scenarios["cancelled"]
    assert cancelled["grasp"]["state"] == SkillState.CANCELLED
    assert _failed(cancelled["grasp"])[:3] == ("approach", Reason.CANCELLED, None)
    assert cancelled["cancelled_command"] == "canceled"
    assert cancelled["grasp"]["attachment"] is None

    drifted = scenarios["drifted"]["grasp"]
    assert drifted["state"] == SkillState.SUCCEEDED, drifted["stages"]
    assert drifted["evaluations"][0] == {
        "transition": "retained",
        "stage": "lift",
        "estimate": "confirmed",
        "reference": "refuted",
        "source": "attachment_monitor",
        "agrees": False,
    }

    # A release is chosen where the opened hand fits anywhere the guard may stop the
    # descent: the transfer that would open the fingers into the marker is rejected like
    # an infeasible plan, and the can is placed from another.
    beside = scenarios["beside_marker"]["place"]
    assert beside["state"] == SkillState.SUCCEEDED, beside["stages"]
    tried = beside["values"]["transfer_attempts"]
    assert tried[-1]["opened"] == [], tried
    rejected = [a for a in tried[:-1] if a["opened"]]
    assert rejected, tried  # the transfer releasing beside the marker was tried, and rejected
    for attempt in rejected:
        assert all(c.startswith("marker/right_ur5e/gripper/") for c in attempt["opened"]), tried
    # In ``success``, that first transfer was kept: only the marker rejects it.
    assert scenarios["success"]["place"]["values"]["transfer_attempts"][0]["opened"] == []

    unconfirmed = scenarios["unconfirmed_release"]["place"]
    assert unconfirmed["state"] == SkillState.FAILED, unconfirmed["stages"]
    assert _failed(unconfirmed) == ("release", Reason.RELEASE_UNCONFIRMED, None, None, True)
    assert unconfirmed["attachment"] is not None

    assert scenarios["trajectory_only"]["grasp"]["state"] == SkillState.SUCCEEDED

    # Each gripper stage waits on its command's own result, kept in the evidence: the
    # hand starts open, so its open asks for no motion; the close stops short on the can.
    gripped = success["grasp"]["values"]["gripper"]
    assert (gripped["open"]["termination"], gripped["open"]["direction"]) == (
        "reached_target",
        "stationary",
    )
    assert (gripped["close"]["termination"], gripped["close"]["direction"]) == (
        "stopped_short",
        "closing",
    )
    assert success["place"]["values"]["gripper"]["release"]["termination"] == "reached_target"

    # A gripper deadline is a failure bound: a close that has not ended by it fails.
    timed = scenarios["gripper_timeout"]["grasp"]
    assert timed["state"] == SkillState.FAILED, timed["stages"]
    assert _failed(timed)[:3] == ("close", Reason.COMMAND_FAILED, "deadline_exceeded")
