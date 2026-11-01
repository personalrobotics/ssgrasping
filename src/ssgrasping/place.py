"""Place a held object in a region on a target support, and let go.

``place`` returns a passive ``SkillExecution`` (see ``ssgrasping.execution``) that runs
these stages, recording each. Every straight move is along the support's explicit
outward ``normal``: away from it to pre-place and retreat, toward it to descend.
1. ``preflight``: every reference resolves, and every command, channel, and the guard
   the placement uses is available. Nothing is commanded before it passes.
2. ``transfer``: plan the end effector into the placement regions moved ``preplace_m``
   along the normal, with the hand alone allowed, at ``carry_scaling`` of the joints'
   velocity and acceleration limits, and execute. A plan is kept only if
   the descent, with the target allowed, also plans from where it ends; up to
   ``transfer_attempts`` seeds are tried. The regions are given for the object; the
   held object's attachment transform turns them into regions for the end effector.
3. ``allow_target``: replace the allowance with the hand and the target support, for
   the terminal descent only, keeping the attachment's transform.
4. ``descend``: wait until the arm's joints are observed at rest, by the criteria the
   runtime declares for them, within ``rest_timeout_ns`` (ssrobot's ``watch_rest``), so
   the guard tares a settled wrist and not the transfer's transient; any other end of
   the watch fails the stage before anything is submitted. Then plan straight toward the
   support by
   ``preplace_m + overshoot_m``, compile it for the runtime, and execute it under
   ``guard``. ``guard_triggered`` is the expected stop; ``trajectory_completed`` without
   it is a failed placement.
5. ``contact``: the stop was the terminal contact, not an early or unrelated one. From
   the measured joints, the end effector's kinematics, and the declared attachment
   transform, the held object must be in a region once moved back by some distance in
   ``[-overshoot_m, terminal_m]`` along the normal.
6. ``release``: open the gripper, require the command to succeed and the opening to
   reach ``released_opening``, then detach. An unconfirmed release keeps the attachment
   and reports the object's held state as unknown.
7. ``retreat``: move straight along the normal by ``retreat_m``.
8. ``placed``: the object rests in one of the regions, within ``region_tolerance`` of
   all six bounds, by its privileged pose.

A failure before ``release`` leaves the object attached and held.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

from ssrobot import (
    GuardCause,
    GuardedJointTrajectory,
    Planner,
    PlanResult,
    Pose,
    RestOutcome,
    RobotContext,
    SceneSnapshot,
    Tsr,
    WrenchGuard,
    prepare,
)

from ssgrasping._poses import compose, inside, invert, offset_interval
from ssgrasping._skill import (
    GRIPPER_TIMEOUT_NS,
    REST_TIMEOUT_NS,
    Body,
    Lookahead,
    Run,
    Wait,
    check_spec,
    preflight,
    rest_evidence,
    unit,
)
from ssgrasping.execution import SkillExecution
from ssgrasping.outcomes import Evidence, Reason


@dataclass(frozen=True, slots=True)
class PlaceSpec:
    """Everything a placement needs, explicitly."""

    manipulator: str
    end_effector: str
    gripper: str
    opening_channel: str
    """The gripper-opening channel read to confirm release: one value, 0 closed to 1
    open."""
    object: str
    target: str
    """The robot frame or scene fixture the object is placed on; allowed only for the
    terminal descent."""
    normal: tuple[float, float, float]
    """The target support's outward unit normal, in the world."""
    regions: Sequence[Tsr]
    """Where the object's frame may rest when placed, in the world."""
    preplace_m: float
    """How far from the regions, along the normal, the transfer ends and the descent
    starts."""
    overshoot_m: float
    """How far past the regions the descent may go before contact stops it."""
    terminal_m: float
    """How far short of the regions a stop still counts as the terminal contact."""
    guard: WrenchGuard
    """The force-torque condition that stops the descent on contact."""
    retreat_m: float
    region_tolerance: tuple[float, float] = (2e-3, 0.05)
    """How far, in metres and radians, a placed object may lie outside a region's
    bounds and still count as in it: it settles after release."""
    open_opening: float = 1.0
    released_opening: float = 0.9
    """The opening that confirms the fingers have let go."""
    gripper_timeout_ns: int = GRIPPER_TIMEOUT_NS
    """The deadline on each gripper command, on the runtime clock: a failure bound, not a
    settle. A command ends when the fingers' motion ends; one that has not by then fails
    its stage (``timed_out``)."""
    rest_timeout_ns: int = REST_TIMEOUT_NS
    """How long after the transfer the arm may take to be observed at rest before the
    guarded descent, which tares on the settled wrist, fails: a failure bound, not a
    wait."""
    line_tolerance_m: float = 1e-3
    carry_scaling: float = 1.0
    """The fraction of the joints' velocity and acceleration limits the transfer, made
    while holding the object, is planned at: a held object can slip from a grasp that
    holds it by friction alone."""
    transfer_attempts: int = 8
    """How many seeds to try for a transfer whose descent also plans."""

    clearance_m: float = 0.0
    """The clearance the transfer, a move through free space carrying the object, keeps
    from the environment (``PlanningRequest.clearance_m``). The guarded descent and the
    retreat, which must come close to the target, keep none."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "normal", unit(self.normal, "normal"))
        check_spec(len(self.regions) > 0, "at least one region", "regions")
        for name in ("preplace_m", "retreat_m", "line_tolerance_m"):
            check_spec(getattr(self, name) > 0, "must be positive", name)
        for name in ("overshoot_m", "terminal_m"):
            check_spec(getattr(self, name) >= 0, "must be non-negative", name)
        check_spec(self.gripper_timeout_ns > 0, "must be positive", "gripper_timeout_ns")
        check_spec(self.rest_timeout_ns > 0, "must be positive", "rest_timeout_ns")
        check_spec(self.terminal_m <= self.preplace_m, "at most preplace_m", "terminal_m")
        metres, radians = self.region_tolerance
        check_spec(metres >= 0 and radians >= 0, "non-negative", "region_tolerance")
        for name in ("open_opening", "released_opening"):
            check_spec(0.0 <= getattr(self, name) <= 1.0, "between 0 and 1", name)
        check_spec(self.transfer_attempts >= 1, "at least 1", "transfer_attempts")
        check_spec(0.0 < self.carry_scaling <= 1.0, "in (0, 1]", "carry_scaling")
        check_spec(
            math.isfinite(self.clearance_m) and self.clearance_m >= 0,
            "must be finite and non-negative",
            "clearance_m",
        )


def place(
    ctx: RobotContext,
    planner: Planner,
    spec: PlaceSpec,
    *,
    check_planner: Planner | None = None,
    lookahead: Lookahead | None = None,
) -> SkillExecution:
    """A pending placement of the held ``spec.object`` on ``spec.target``; see the
    module. Advance it, or drive it with ``run_skill``.

    ``check_planner`` plans the descent checks from candidate transfers, as in
    ``grasp``.

    With ``lookahead``, the boundaries where the placement cannot look ahead are logged in
    ``values["lookahead"]``: its stages after the transfer depend on outcomes not yet
    known."""
    run = Run(ctx, "place", spec.object, lookahead=lookahead)
    checker = planner if check_planner is None else check_planner
    return SkillExecution(run, _place(run, planner, checker, spec))


def _place(run: Run, planner: Planner, checker: Planner, spec: PlaceSpec) -> Body:
    ctx = run.ctx
    attachment = ctx.scene.attachment(spec.object)
    run.held = None if attachment is None else True
    preflight(
        run,
        manipulator=spec.manipulator,
        end_effector=spec.end_effector,
        gripper=spec.gripper,
        support=spec.target,
        channels={spec.opening_channel: (1,), spec.guard.channel: (6,)},
        guard=spec.guard,
    )
    if attachment is None:
        run.record(
            "transfer",
            False,
            f"{spec.object!r} is not attached",
            Evidence.PLANNED,
            Reason.NOT_ATTACHED,
        )
    assert attachment is not None
    arm = ctx.manipulator(spec.manipulator)
    hand = attachment.allow
    object_to_hand = invert(attachment.transform)
    raised = [
        _moved(_for_hand(r, object_to_hand), spec.normal, spec.preplace_m) for r in spec.regions
    ]
    transfer, _, _ = run.choose(
        "transfer",
        "descend",
        lambda snapshot, seed: arm.plan_to_tsr(
            raised,
            snapshot=snapshot,
            planner=planner,
            seed=seed,
            velocity_scaling=spec.carry_scaling,
            acceleration_scaling=spec.carry_scaling,
            clearance_m=spec.clearance_m,
        ),
        lambda start: _toward(ctx, _allowing(start.state, spec), checker, spec),
        spec.transfer_attempts,
        planner=planner,
    )
    yield from run.move("transfer", transfer, planner)

    run.boundary("allow_target", "allows the target: a scene change; the descent is guarded")
    run.begin("allow_target")
    run.replace_allowance("allow_target", spec.end_effector, (*hand, spec.target))
    run.record("allow_target", True, f"allowed {', '.join((*hand, spec.target))}", Evidence.PLANNED)

    yield from _descend(run, planner, spec)

    run.begin("contact")
    stop = _held_pose(run, spec)
    reached = [offset_interval(stop, r, spec.normal, spec.region_tolerance) for r in spec.regions]
    usable = [
        (i, (lo, hi))
        for i, span in enumerate(reached)
        if span is not None
        for lo, hi in [span]
        if lo <= spec.terminal_m and hi >= -spec.overshoot_m
    ]
    run.values["contact_pose"] = {
        "position": list(stop.position),
        "quat_wxyz": list(stop.quat_wxyz),
    }
    run.record(
        "contact",
        bool(usable),
        f"held object {usable[0][1][0]:.4f}..{usable[0][1][1]:.4f} m from region {usable[0][0]}"
        if usable
        else f"stopped outside the terminal band of every region, at {stop.position}",
        Evidence.PORTABLE,
        None if usable else Reason.OUTSIDE_REGIONS,
    )

    yield from run.grip("release", spec.gripper, spec.open_opening, spec.gripper_timeout_ns)
    opening = float(run.observe(spec.opening_channel)[0])
    run.values["released_opening"] = opening
    if opening < spec.released_opening:
        run.held = None
        run.record(
            "release",
            False,
            f"opening {opening:.3f} < {spec.released_opening}: release not confirmed",
            Evidence.PORTABLE,
            Reason.RELEASE_UNCONFIRMED,
        )
    ctx.detach(spec.object)
    run.held = False
    run.record("release", True, f"opening {opening:.3f}; detached", Evidence.PORTABLE)

    run.begin("retreat")
    retreat = arm.plan_to_end_effector_offset(
        spec.normal, spec.retreat_m, planner=planner, position_tolerance=spec.line_tolerance_m
    )
    yield from run.move("retreat", retreat, planner)

    pose = next(o.pose for o in ctx.snapshot().objects if o.name == spec.object)
    run.values["placed_pose"] = {"position": list(pose.position), "quat_wxyz": list(pose.quat_wxyz)}
    within = [i for i, r in enumerate(spec.regions) if inside(pose, r, spec.region_tolerance)]
    run.record(
        "placed",
        bool(within),
        f"in region {within[0]}" if within else f"outside every region at {pose.position}",
        Evidence.PRIVILEGED,
        None if within else Reason.OUTSIDE_REGIONS,
    )


def _descend(run: Run, planner: Planner, spec: PlaceSpec) -> Body:
    """The guarded terminal descent: observed at rest, planned, compiled for the runtime,
    run."""
    ctx = run.ctx
    run.begin("descend")
    arm = ctx.manipulator(spec.manipulator)
    rest = yield from run.rest(arm.group, spec.rest_timeout_ns)
    run.values["tare_rest"] = rest_evidence(rest)
    if rest.outcome is not RestOutcome.AT_REST:
        evidence = Evidence.PRIVILEGED if rest.channel.privileged else Evidence.PORTABLE
        run.record(
            "descend",
            False,
            f"rest {rest.outcome.value}: {rest.detail}",
            evidence,
            Reason.NOT_AT_REST,
            rest.outcome.value,
        )
    snapshot = ctx.snapshot()
    plan = _toward(ctx, snapshot, planner, spec)
    if not plan.succeeded:
        run.record(
            "descend",
            False,
            f"planning {plan.status.value}",
            Evidence.PLANNED,
            Reason.NOT_PLANNED,
            plan.status.value,
        )
    scene = ctx.planning_scene(snapshot, plan.request.group)
    ready = prepare(
        plan,
        scene,
        validator=planner.validator(plan.request, scene),
        description=ctx.description,
        contract=ctx.trajectory_contract,
    )
    command = GuardedJointTrajectory(trajectory=ready.command, guard=spec.guard)
    execution = ctx.submit(command, snapshot=snapshot, source="planner")
    yield Wait(execution=execution)
    run.finished("descend", execution)
    outcome = execution.status.guard
    assert outcome is not None  # a succeeded guarded trajectory reports its outcome
    run.values["descent"] = {
        "cause": outcome.cause.value,
        "measure": outcome.measure,
        "elapsed_ns": outcome.elapsed_ns,
    }
    contact = outcome.cause is GuardCause.GUARD_TRIGGERED
    run.record(
        "descend",
        contact,
        f"{outcome.cause.value}: measure {outcome.measure:.3f}"
        + ("" if contact else ", no contact: a failed placement"),
        Evidence.PORTABLE,
        None if contact else Reason.NO_CONTACT,
        None if contact else outcome.cause.value,
    )


def _held_pose(run: Run, spec: PlaceSpec) -> Pose:
    """Where the held object is, by the measured joints, the end effector's kinematics,
    and the declared attachment transform: no simulator object state."""
    ctx = run.ctx
    arm = ctx.manipulator(spec.manipulator)
    frame = arm.frame
    assert frame is not None  # the end effector's frame
    attachment = ctx.scene.attachment(spec.object)
    assert attachment is not None
    snapshot = ctx.snapshot()
    q = tuple(snapshot.position(j) for j in ctx.description.group(arm.group).joints)
    hand = ctx.planning_scene(snapshot, arm.group).forward_kinematics(q, frame)
    return compose(hand, attachment.transform)


def _toward(
    ctx: RobotContext, snapshot: SceneSnapshot, planner: Planner, spec: PlaceSpec
) -> PlanResult:
    """The straight descent toward the support from ``snapshot``, to ``overshoot_m``
    past the regions."""
    x, y, z = spec.normal
    return ctx.manipulator(spec.manipulator).plan_to_end_effector_offset(
        (-x, -y, -z),
        spec.preplace_m + spec.overshoot_m,
        snapshot=snapshot,
        planner=planner,
        position_tolerance=spec.line_tolerance_m,
    )


def _allowing(snapshot: SceneSnapshot, spec: PlaceSpec) -> SceneSnapshot:
    """``snapshot`` with the target also allowed to the held object, as the descent will
    have it: a hypothetical, for planning only."""
    attachments = tuple(
        replace(a, allow=tuple(sorted({*a.allow, spec.target}))) if a.object == spec.object else a
        for a in snapshot.attachments
    )
    return replace(snapshot, attachments=attachments)


def _for_hand(region: Tsr, object_to_hand: Pose) -> Tsr:
    """``region`` for the object frame, as a region for the end effector frame holding it."""
    return replace(region, offset=compose(region.offset, object_to_hand))


def _moved(region: Tsr, direction: tuple[float, float, float], distance: float) -> Tsr:
    """``region`` moved ``distance`` along the world ``direction``."""
    position = tuple(
        p + distance * d for p, d in zip(region.origin.position, direction, strict=True)
    )
    return replace(region, origin=Pose(position=position, quat_wxyz=region.origin.quat_wxyz))  # type: ignore[arg-type]
