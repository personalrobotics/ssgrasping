"""Grasp an object resting on a support, and lift it clear.

``grasp`` returns a passive ``SkillExecution`` (see ``ssgrasping.execution``) that runs
these stages, recording each:
1. ``preflight``: every reference resolves and every command and channel the grasp
   uses is available. Nothing is commanded before it passes.
2. ``open``: open the gripper, so the pregrasp is planned with the fingers as they will
   approach. The command must succeed.
3. ``pregrasp``: plan into the pregrasp regions, and keep the plan only if the straight
   approach also plans from where it ends; otherwise plan again with the next seed, up
   to ``pregrasp_attempts``. The pregrasp alone is checked only for collision, so a
   region can be reachable while its approach is not. Then execute it.
4. ``approach``: move straight along the hand's approach axis, by ``approach_m``,
   planned again from where the arm actually is; with lookahead, planned while the
   pregrasp executes, on its prediction, and continued on arrival.
5. ``close``: close the gripper. The command must succeed.
6. ``acquired``: classify the opening as held, missed, or indeterminate. This is
   portable evidence, read before anything is attached. Only ``held`` continues; a miss
   reopens the gripper, the one cleanup whose meaning is unambiguous.
7. ``attach``: declare the object held, allowed to touch the hand and its support.
8. ``lift``: move straight along ``normal``, away from the support, by ``lift_m``.
9. ``restrict``: replace the allowance with the hand alone, keeping the attachment's
   transform, so nothing else may be touched from now on.
10. ``clear``: the held object is clear of everything, by the planner's validator.
11. ``retained``: the attachment monitor still reports the object held at the
    transform declared at ``attach``. This is privileged evidence about the motion; it
    never stands in for ``acquired``.

After ``attach``, a failure or cancellation leaves the attachment declared and reports
the object as held: the physically conservative state. Retrying or recovering is the
application's.

Lookahead stops at ``close``: whether the object is held decides what follows. The
stages after it plan from the actual state: ``attach`` and ``restrict`` change the scene.
A ``pregrasp`` made beforehand, such as by ``Candidates`` while another motion ran, is
executed in place of choosing one; if it cannot continue, one is chosen as usual.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ssrobot import (
    ManipulatorView,
    OffsetGoal,
    Planner,
    PlanResult,
    PredictedSnapshot,
    RobotContext,
    SceneSnapshot,
    Tsr,
)

from ssgrasping._poses import axis
from ssgrasping._skill import Ahead, Body, Run, check_spec, preflight, unit
from ssgrasping.execution import SkillExecution
from ssgrasping.outcomes import Evidence


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
    held_opening: tuple[float, float]
    """The opening range, after closing, that means the object is between the fingers.
    Below it is a miss; above it, indeterminate."""
    lift_m: float
    open_opening: float = 1.0
    close_opening: float = 0.0
    settle_ns: int = 1_000_000_000
    """How long the fingers settle after a gripper command, on the runtime clock."""
    line_tolerance_m: float = 1e-3
    """Position tolerance of the straight approach and lift."""
    pregrasp_attempts: int = 8
    """How many seeds to try for a pregrasp whose approach also plans."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "normal", unit(self.normal, "normal"))
        check_spec(len(self.pregrasps) > 0, "at least one pregrasp region", "pregrasps")
        for name in ("approach_m", "lift_m", "line_tolerance_m"):
            check_spec(getattr(self, name) > 0, "must be positive", name)
        low, high = self.held_opening
        check_spec(0.0 <= low <= high <= 1.0, "0 <= low <= high <= 1", "held_opening")
        for name in ("open_opening", "close_opening"):
            check_spec(0.0 <= getattr(self, name) <= 1.0, "between 0 and 1", name)
        check_spec(self.settle_ns >= 0, "must be non-negative", "settle_ns")
        check_spec(self.pregrasp_attempts >= 1, "at least 1", "pregrasp_attempts")


def grasp(
    ctx: RobotContext,
    planner: Planner,
    spec: GraspSpec,
    *,
    check_planner: Planner | None = None,
    lookahead: bool = False,
    pregrasp: PlanResult | Ahead | None = None,
) -> SkillExecution:
    """A pending grasp of ``spec.object`` that lifts it clear of its support; see the
    module. Advance it, or drive it with ``run_skill``.

    ``check_planner`` plans the approach checks from candidate pregrasps (``planner``
    when None). A feasible short line plans in few iterations, so a planner with a small
    iteration budget rejects an infeasible one quickly, and deterministically, where
    ``planner`` would spend its whole budget first."""
    run = Run(ctx, "grasp", spec.object, lookahead=lookahead)
    checker = planner if check_planner is None else check_planner
    return SkillExecution(run, _grasp(run, planner, checker, spec, pregrasp))


def _grasp(
    run: Run,
    planner: Planner,
    checker: Planner,
    spec: GraspSpec,
    planned: PlanResult | Ahead | None,
) -> Body:
    ctx = run.ctx
    preflight(
        run,
        manipulator=spec.manipulator,
        end_effector=spec.end_effector,
        gripper=spec.gripper,
        support=spec.support,
        channels={spec.opening_channel: (1,)},
    )
    arm = ctx.manipulator(spec.manipulator)

    yield from run.grip("open", spec.gripper, spec.open_opening, spec.settle_ns)
    run.record(
        "open", True, f"opening {run.observe(spec.opening_channel)[0]:.3f}", Evidence.PORTABLE
    )

    def choose() -> PlanResult:
        return run.choose(
            "pregrasp",
            "approach",
            lambda snapshot, seed: arm.plan_to_tsr(
                list(spec.pregrasps), snapshot=snapshot, planner=planner, seed=seed
            ),
            lambda start: _approach(arm, ctx, start, checker, spec),
            spec.pregrasp_attempts,
        )

    def replan_approach() -> PlanResult:
        return _approach(arm, ctx, ctx.snapshot(), planner, spec)

    if isinstance(planned, Ahead):
        run.adopt(planned)
    upcoming = yield from run.move(
        "pregrasp",
        choose() if planned is None else planned,
        planner,
        replan=choose,
        then=lambda execution: run.ahead(
            "approach",
            execution,
            lambda prediction: arm.request(
                _approach_goal(arm, ctx, prediction, spec), snapshot=prediction
            ),
            planner,
        ),
    )

    run.begin("approach")
    approach = replan_approach() if upcoming is None else upcoming
    yield from run.move("approach", approach, planner, replan=replan_approach)
    run.boundary("close", "grasp verification: the closed opening decides what follows")

    yield from run.grip("close", spec.gripper, spec.close_opening, spec.settle_ns)
    opening = float(run.observe(spec.opening_channel)[0])
    run.values["closed_opening"] = opening
    run.record("close", True, f"opening {opening:.3f}", Evidence.PORTABLE)

    low, high = spec.held_opening
    if opening < low:
        run.held = False
        yield from run.grip("acquired", spec.gripper, spec.open_opening, spec.settle_ns)
        run.record("acquired", False, f"missed: opening {opening:.3f} < {low}", Evidence.PORTABLE)
    if opening > high:
        run.record(
            "acquired", False, f"indeterminate: opening {opening:.3f} > {high}", Evidence.PORTABLE
        )
    run.held = True
    run.record(
        "acquired", True, f"held: opening {opening:.3f} in [{low}, {high}]", Evidence.PORTABLE
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
    run.replace_allowance("attach", spec.end_effector, (*hand, spec.support))
    run.values["hand_allowance"] = list(hand)
    run.record("attach", True, f"allowed {', '.join((*hand, spec.support))}", Evidence.PLANNED)

    before = _object_position(ctx, spec.object)
    run.begin("lift")
    lift = arm.plan_to_end_effector_offset(
        spec.normal, spec.lift_m, planner=planner, position_tolerance=spec.line_tolerance_m
    )
    yield from run.move("lift", lift, planner)

    run.boundary("restrict", "re-declares the attachment's allowance: a scene change")
    run.replace_allowance("restrict", spec.end_effector, hand)
    run.record("restrict", True, f"allowed {', '.join(hand)}", Evidence.PLANNED)

    snapshot = ctx.snapshot()
    q = tuple(snapshot.position(j) for j in ctx.description.group(arm.group).joints)
    scene = ctx.planning_scene(snapshot, arm.group)
    contacts = sorted({f"{c.first}/{c.second}" for c in scene.contacts(q)})
    run.record(
        "clear",
        scene.is_valid(q),
        "clear of everything" if not contacts else f"in contact: {', '.join(contacts)}",
        Evidence.PLANNED,
    )

    after = _object_position(ctx, spec.object)
    risen = sum((a - b) * n for a, b, n in zip(after, before, spec.normal, strict=True))
    run.values["object_risen_m"] = round(risen, 6)
    current = ctx.scene.attachment(spec.object)
    retained = current is not None and current.held
    run.record(
        "retained",
        retained,
        f"attachment monitor: held={retained}, object rose {risen:.4f} m",
        Evidence.PRIVILEGED,
    )


def _approach(
    arm: ManipulatorView,
    ctx: RobotContext,
    snapshot: SceneSnapshot,
    planner: Planner,
    spec: GraspSpec,
) -> PlanResult:
    """The straight approach along the end effector's z axis, from ``snapshot``."""
    return arm.plan(_approach_goal(arm, ctx, snapshot, spec), snapshot=snapshot, planner=planner)


def _approach_goal(
    arm: ManipulatorView,
    ctx: RobotContext,
    snapshot: SceneSnapshot | PredictedSnapshot,
    spec: GraspSpec,
) -> OffsetGoal:
    """The approach: ``approach_m`` along the end effector's z axis where it is in
    ``snapshot``, keeping its orientation."""
    frame = arm.frame
    assert frame is not None  # the end effector's frame
    state = snapshot.state if isinstance(snapshot, PredictedSnapshot) else snapshot
    q = tuple(state.position(j) for j in ctx.description.group(arm.group).joints)
    pose = ctx.planning_scene(snapshot, arm.group).forward_kinematics(q, frame)
    x, y, z = axis(pose, 2)
    return OffsetGoal(
        direction=(x, y, z),
        distance=spec.approach_m,
        position_tolerance=spec.line_tolerance_m,
        rotation_tolerance=1e-2,
    )


def _object_position(ctx: RobotContext, name: str) -> tuple[float, float, float]:
    return next(o.pose.position for o in ctx.snapshot().objects if o.name == name)
