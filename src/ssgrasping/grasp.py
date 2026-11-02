"""Grasp an object resting on a support, and lift it clear.

``grasp`` returns a passive ``SkillExecution`` (see ``ssgrasping.execution``) that runs
these stages, recording each:
1. ``preflight``: every reference resolves and every command and channel the grasp
   uses is available, with a guard on the opening channel for the lift. Nothing is
   commanded before it passes.
2. ``open``: open the gripper, so the pregrasp is planned with the fingers as they will
   approach. The command must succeed: it ends when the fingers' motion ends, and its
   ``GripperResult`` is kept in ``values["gripper"]``, as for every gripper stage.
3. ``pregrasp``: plan into the pregrasp regions, and keep the plan only if the straight
   approach also plans from where it ends; otherwise plan again with the next seed, up
   to ``pregrasp_attempts``. The pregrasp alone is checked only for collision, so a
   region can be reachable while its approach is not. Then execute it.
4. ``approach``: move straight along the hand's approach axis, by ``approach_m``,
   planned again from where the arm actually is. With lookahead, the approach the
   pregrasp was checked with, made on its projection before it executed, continues on
   arrival; if the pregrasp did not run as that plan, the approach is planned while the
   pregrasp executes, on its prediction, and continued on arrival.
5. ``close``: close the gripper. The command must succeed, ending when the fingers stop.
6. ``obstructed``: the close stopped short of its target (ssrobot's
   ``GripperResult``): something is between the fingers. This is necessary and
   provisional: the object stops them, but so can anything else, such as the support. If
   the close reached its target, the grasp missed: it reopens the gripper, the one
   cleanup whose meaning is unambiguous, and fails. Whether the opening matches the
   object's width is personalrobotics/ssgrasping#28's.
7. ``attach``: declare the object attached, allowed to touch the hand and its support:
   the hypothesis the lift is planned under, not yet a claim that it is held.
8. ``lift``: move straight along ``normal``, away from the support, by ``lift_m``,
   under an opening guard midway between the opening the close stopped short at and
   ``close_opening``. If the guard stops the lift, the fingers closed past the object's
   width: ``retained`` is refuted, and the lift fails with the object reported not
   held.
9. ``acquired``: close the gripper again. If it still stops short, what stopped the
   fingers rose with them; if it reaches its target, it stayed behind, and the grasp
   fails. This is portable evidence, and the grasp's only claim that it holds the
   object.
10. ``restrict``: replace the allowance with the hand alone, keeping the attachment's
    transform, so nothing else may be touched from now on.
11. ``clear``: the held object is clear of everything, by the planner's validator.
12. ``retained``: the lift's estimate of ``retained`` (``SkillResult.transitions``),
    confirmed, or unobserved when the opening had nowhere to close to. It never stands
    in for ``acquired``. Where the runtime has the object's privileged pose, the
    attachment monitor's report is kept beside it, with the object's pose before and
    after the lift, for evaluation only (``SkillResult.evaluations``).

After ``attach``, a failure or cancellation leaves the attachment declared and reports
the object as held: the physically conservative state, since it may be in the hand.
Retrying or recovering is the application's.

Lookahead stops at ``close``: whether the fingers are obstructed decides what follows. The
stages after it plan from the actual state: ``attach`` and ``restrict`` change the scene.
A ``pregrasp`` made beforehand, such as by ``Candidates`` while another motion ran, is
executed in place of choosing one; if it cannot continue, one is chosen as usual.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ssrobot import (
    Execution,
    GripperTermination,
    ManipulatorView,
    OffsetGoal,
    Planner,
    PlanProjection,
    PlanResult,
    Pose,
    PredictedSnapshot,
    RobotContext,
    SceneSnapshot,
    Tsr,
)

from ssgrasping._poses import axis
from ssgrasping._skill import (
    GRIPPER_TIMEOUT_NS,
    Ahead,
    Body,
    Kept,
    Lookahead,
    Run,
    check_spec,
    preflight,
    retention,
    unit,
)
from ssgrasping.execution import SkillExecution
from ssgrasping.outcomes import Criterion, Evidence, Reason


@dataclass(frozen=True, slots=True)
class GraspSpec:
    """Everything a grasp needs, explicitly: nothing is inferred from names."""

    manipulator: str
    """The manipulator that moves, by name."""
    end_effector: str
    """The end effector the object is held by; its frame is the one the regions bind."""
    gripper: str
    opening_channel: str
    """The gripper-opening channel read to verify the grasp: one value, 0 closed to 1
    open."""
    object: str
    support: str
    """The robot frame or scene fixture the object rests on; allowed only for the lift."""
    normal: tuple[float, float, float]
    """The support's outward unit normal, in the world: the lift moves along it."""
    pregrasps: Sequence[Tsr]
    """Where the end effector frame may be before the approach, in the world."""
    approach_m: float
    """How far the approach moves along the end effector frame's z axis."""
    lift_m: float
    open_opening: float = 1.0
    close_opening: float = 0.0
    gripper_timeout_ns: int = GRIPPER_TIMEOUT_NS
    """The deadline on each gripper command, on the runtime clock: a failure bound, not a
    settle. A command ends when the fingers' motion ends; one that has not by then fails
    its stage (``timed_out``)."""
    line_tolerance_m: float = 1e-3
    """Position tolerance of the straight approach and lift."""
    pregrasp_attempts: int = 8
    """How many seeds to try for a pregrasp whose approach also plans."""
    clearance_m: float = 0.0
    """The clearance the pregrasp, a move through free space, keeps from the
    environment (``PlanningRequest.clearance_m``). The approach and the lift, which must
    come close to the object and its support, keep none."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "normal", unit(self.normal, "normal"))
        check_spec(len(self.pregrasps) > 0, "at least one pregrasp region", "pregrasps")
        for name in ("approach_m", "lift_m", "line_tolerance_m"):
            check_spec(getattr(self, name) > 0, "must be positive", name)
        for name in ("open_opening", "close_opening"):
            check_spec(0.0 <= getattr(self, name) <= 1.0, "between 0 and 1", name)
        check_spec(self.gripper_timeout_ns > 0, "must be positive", "gripper_timeout_ns")
        check_spec(self.pregrasp_attempts >= 1, "at least 1", "pregrasp_attempts")
        check_spec(
            math.isfinite(self.clearance_m) and self.clearance_m >= 0,
            "must be finite and non-negative",
            "clearance_m",
        )


def grasp(
    ctx: RobotContext,
    planner: Planner,
    spec: GraspSpec,
    *,
    check_planner: Planner | None = None,
    lookahead: Lookahead | None = None,
) -> SkillExecution:
    """A pending grasp of ``spec.object`` that lifts it clear of its support; see the
    module. Advance it, or drive it with ``run_skill``.

    ``check_planner`` plans the approach checks from candidate pregrasps (``planner``
    when None). A feasible short line plans in few iterations, so a planner with a small
    iteration budget rejects an infeasible one quickly, and deterministically, where
    ``planner`` would spend its whole budget first.

    With ``lookahead``, the approach the pregrasp was checked with, made on its projection
    before the pregrasp executed, continues once the pregrasp has run, instead of being
    planned again: the pregrasp ran as that plan, or continued it, rebased onto the
    actual state, to the same end. It removes the wait for the approach without
    overlapping the pregrasp's motion. If the pregrasp did not run as that plan (it was
    replanned, or rebased to another end), the approach is planned while the pregrasp
    executes, on its prediction, instead. None of this changes any stage's meaning or
    evidence kind; a stage that ran from a plan made ahead says so in its detail.

    A grasp raced by ``Candidates`` runs from its selection: ``Selection.grasp``."""
    run = Run(ctx, "grasp", spec.object, lookahead=lookahead)
    checker = planner if check_planner is None else check_planner
    return SkillExecution(run, _grasp(run, planner, checker, spec, None, None))


def _selected_grasp(
    ctx: RobotContext,
    planner: Planner,
    checker: Planner,
    spec: GraspSpec,
    lookahead: Lookahead | None,
    pregrasp: PlanResult | Ahead,
    approach: Kept,
) -> SkillExecution:
    """``grasp``, executing ``pregrasp``, planned beforehand, in place of choosing one,
    with ``approach`` the approach it was checked with: a ``Selection``'s, only."""
    run = Run(ctx, "grasp", spec.object, lookahead=lookahead)
    return SkillExecution(run, _grasp(run, planner, checker, spec, pregrasp, approach))


def _grasp(
    run: Run,
    planner: Planner,
    checker: Planner,
    spec: GraspSpec,
    planned: PlanResult | Ahead | None,
    kept: Kept | None,
) -> Body:
    ctx = run.ctx
    preflight(
        run,
        manipulator=spec.manipulator,
        end_effector=spec.end_effector,
        gripper=spec.gripper,
        support=spec.support,
        channels={spec.opening_channel: (1,)},
        carries=spec.opening_channel,
    )
    arm = ctx.manipulator(spec.manipulator)

    yield from run.grip("open", spec.gripper, spec.open_opening, spec.gripper_timeout_ns)
    run.record(
        "open", True, f"opening {run.observe(spec.opening_channel)[0]:.3f}", Evidence.PORTABLE
    )

    checked = [kept]  # the approach the pregrasp to run was checked with

    def choose() -> PlanResult:
        result, check, projection = run.choose(
            "pregrasp",
            "approach",
            lambda snapshot, seed: arm.plan_to_tsr(
                list(spec.pregrasps),
                snapshot=snapshot,
                planner=planner,
                seed=seed,
                clearance_m=spec.clearance_m,
            ),
            lambda start: _approach(arm, ctx, start, checker, spec),
            spec.pregrasp_attempts,
            planner=planner,
        )
        assert check is not None and projection is not None  # an approach follows
        checked[0] = Kept(projection, check, checker)
        return result

    def ahead(execution: Execution) -> Ahead | None:
        """The checked approach, continued from this execution of the pregrasp it was
        checked with; failing that, the approach planned ahead on the pregrasp's
        prediction."""
        approach = checked[0]
        if approach is not None:
            continued = run.kept(
                "approach", execution, approach.projection, approach.plan, approach.planner
            )
            if continued is not None:
                return continued
        return run.ahead("approach", execution, ahead_request, planner)

    def replan_approach() -> PlanResult:
        return _approach(arm, ctx, ctx.snapshot(), planner, spec)

    def ahead_request(prediction: PredictedSnapshot) -> Any:
        return arm.request(_approach_goal(arm, ctx, prediction, spec), snapshot=prediction)

    if isinstance(planned, Ahead):
        run.adopt(planned)
    upcoming = yield from run.move(
        "pregrasp",
        choose() if planned is None else planned,
        planner,
        replan=choose,
        then=ahead,
    )

    run.begin("approach")
    approach = replan_approach() if upcoming is None else upcoming
    yield from run.move("approach", approach, planner, replan=replan_approach)
    run.boundary("close", "grasp verification: the closed opening decides what follows")

    yield from run.grip("close", spec.gripper, spec.close_opening, spec.gripper_timeout_ns)
    opening = float(run.observe(spec.opening_channel)[0])
    run.record("close", True, f"opening {opening:.3f}", Evidence.PORTABLE)

    closed = run.values["gripper"]["close"]  # type: ignore[index]
    if closed["termination"] == GripperTermination.REACHED_TARGET:
        run.held = False
        yield from run.grip("obstructed", spec.gripper, spec.open_opening, spec.gripper_timeout_ns)
        run.record(
            "obstructed",
            False,
            f"missed: the fingers reached their target, at opening {opening:.3f}",
            Evidence.PORTABLE,
            Reason.MISSED,
        )
    run.held = True  # conservatively, from here on: it may be in the hand
    run.record(
        "obstructed",
        True,
        f"the fingers stopped short, at opening {opening:.3f}",
        Evidence.PORTABLE,
    )

    # ssrobot's own default allowance (the hand), read back, plus the support.
    run.begin("attach")
    ctx.attach(spec.object, spec.end_effector)
    attachment = ctx.scene.attachment(spec.object)
    assert attachment is not None
    hand = attachment.allow
    run.values["attach_transform"] = [
        list(attachment.transform.position),
        list(attachment.transform.quat_wxyz),
    ]
    run.replace_allowance("attach", (*hand, spec.support))
    run.record("attach", True, f"allowed {', '.join((*hand, spec.support))}", Evidence.PLANNED)

    run.begin("lift")
    lift = arm.plan_to_end_effector_offset(
        spec.normal, spec.lift_m, planner=planner, position_tolerance=spec.line_tolerance_m
    )

    carry = retention(
        spec.opening_channel,
        Criterion("held_opening", opening, "observed: the opening the close stopped short at"),
        Criterion("close_opening", spec.close_opening, "GraspSpec.close_opening"),
    )
    yield from run.move("lift", lift, planner, retain=carry)

    yield from run.grip("acquired", spec.gripper, spec.close_opening, spec.gripper_timeout_ns)
    lifted = float(run.observe(spec.opening_channel)[0])
    reclosed = run.values["gripper"]["acquired"]  # type: ignore[index]
    if reclosed["termination"] == GripperTermination.REACHED_TARGET:
        run.record(
            "acquired",
            False,
            f"lost: after the lift, the fingers reached their target, at opening {lifted:.3f}",
            Evidence.PORTABLE,
            Reason.LOST,
        )
    run.record(
        "acquired",
        True,
        f"held: after the lift, the fingers still stop short, at opening {lifted:.3f}",
        Evidence.PORTABLE,
    )

    run.boundary("restrict", "re-declares the attachment's allowance: a scene change")
    run.replace_allowance("restrict", hand)
    run.record("restrict", True, f"allowed {', '.join(hand)}", Evidence.PLANNED)

    snapshot = ctx.snapshot()
    q = tuple(snapshot.position(j) for j in ctx.description.group(arm.group).joints)
    scene = ctx.planning_scene(snapshot, arm.group)
    contacts = sorted({f"{c.first}/{c.second}" for c in scene.contacts(q)})
    clear = scene.is_valid(q)
    run.record(
        "clear",
        clear,
        "clear of everything" if not contacts else f"in contact: {', '.join(contacts)}",
        Evidence.PLANNED,
        None if clear else Reason.IN_CONTACT,
    )

    lifted_estimate = run.transitions[-1]
    run.record(
        "retained",
        True,
        f"{lifted_estimate.outcome.value} through the lift: {lifted_estimate.reason.value}",
        Evidence.PORTABLE,
    )


def _approach(
    arm: ManipulatorView,
    ctx: RobotContext,
    snapshot: SceneSnapshot | PlanProjection,
    planner: Planner,
    spec: GraspSpec,
) -> PlanResult:
    """The straight approach by ``approach_m`` along the end effector's z axis, from
    ``snapshot``."""
    return arm.plan(_approach_goal(arm, ctx, snapshot, spec), snapshot=snapshot, planner=planner)


def _pose(
    arm: ManipulatorView,
    ctx: RobotContext,
    snapshot: SceneSnapshot | PredictedSnapshot | PlanProjection,
) -> Pose:
    """Where the end effector is in ``snapshot``."""
    frame = arm.frame
    assert frame is not None  # the end effector's frame
    state = snapshot.state if isinstance(snapshot, PredictedSnapshot | PlanProjection) else snapshot
    q = tuple(state.position(j) for j in ctx.description.group(arm.group).joints)
    return ctx.planning_scene(snapshot, arm.group).forward_kinematics(q, frame)


def _approach_goal(
    arm: ManipulatorView,
    ctx: RobotContext,
    snapshot: SceneSnapshot | PredictedSnapshot | PlanProjection,
    spec: GraspSpec,
) -> OffsetGoal:
    """The approach: ``approach_m`` along the end effector's z axis where it is in
    ``snapshot``, keeping its orientation."""
    x, y, z = axis(_pose(arm, ctx, snapshot), 2)
    return OffsetGoal(
        direction=(x, y, z),
        distance=spec.approach_m,
        position_tolerance=spec.line_tolerance_m,
        rotation_tolerance=1e-2,
    )
