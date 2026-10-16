"""Place a held object in a region on a target support, and let go.

``place`` runs these stages, recording each:
1. ``transfer``: plan the end effector into the placement regions raised by
   ``preplace_m``, with the hand alone allowed, and execute. A plan is kept only if the
   descent, with the target allowed, also plans from where it ends; up to
   ``transfer_attempts`` seeds are tried. The regions are given for the object; the
   held object's attachment transform turns them into regions for the end effector.
2. ``allow_target``: replace the allowance with the hand and the target support, for
   the terminal descent only.
3. ``descend``: hold still for ``settle_ticks``, so the guard tares a settled wrist and
   not the transfer's transient; plan straight down by ``preplace_m + overshoot_m``,
   compile it for the runtime, and execute it under ``guard``. ``guard_triggered`` is the expected
   contact; ``trajectory_completed`` without contact is a failed placement.
4. ``release``: open the gripper, then detach.
5. ``retreat``: move straight up by ``retreat_m``.
6. ``placed``: the object rests in one of the regions, by its privileged pose.

A failure before ``release`` leaves the object attached and held.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

from ssrobot import (
    ExecutionState,
    GuardCause,
    GuardedJointTrajectory,
    Planner,
    PlanResult,
    Pose,
    RobotContext,
    SceneSnapshot,
    SsrobotError,
    Tsr,
    WrenchGuard,
    prepare,
)

from ssgrasping._poses import compose, invert, roll_pitch_yaw
from ssgrasping._skill import Failed, Run
from ssgrasping.outcomes import Evidence, SkillResult


@dataclass(frozen=True, slots=True)
class PlaceSpec:
    """Everything a placement needs, explicitly."""

    manipulator: str
    end_effector: str
    gripper: str
    object: str
    target: str
    """The robot frame or scene fixture the object is placed on; allowed only for the
    terminal descent."""
    regions: Sequence[Tsr]
    """Where the object's frame may rest when placed, in the world."""
    preplace_m: float
    """How far above the regions the transfer ends, and the descent starts."""
    overshoot_m: float
    """How far past the regions the descent may go before contact stops it."""
    guard: WrenchGuard
    """The force-torque condition that stops the descent on contact."""
    retreat_m: float
    upright_rad: float = 0.1
    """How far from upright, in roll or pitch, the placed object may rest."""
    open_opening: float = 1.0
    settle_ticks: int = 500
    line_tolerance_m: float = 1e-3
    max_ticks: int = 20_000
    transfer_attempts: int = 8
    """How many seeds to try for a transfer whose descent also plans."""


def place(
    ctx: RobotContext, planner: Planner, spec: PlaceSpec, *, check_planner: Planner | None = None
) -> SkillResult:
    """Place the held ``spec.object`` on ``spec.target``; see the module.

    ``check_planner`` plans the descent checks from candidate transfers, as in
    ``grasp``."""
    run = Run(ctx, "place", spec.object)
    checker = planner if check_planner is None else check_planner
    run.held = True
    arm = ctx.manipulator(spec.manipulator)
    try:
        attachment = ctx.scene.attachment(spec.object)
        if attachment is None:
            run.held = None
            run.record("transfer", False, f"{spec.object!r} is not attached", Evidence.PLANNED)
        assert attachment is not None
        hand = attachment.allow
        object_to_hand = invert(attachment.transform)
        raised = [_raised(_for_hand(r, object_to_hand), spec.preplace_m) for r in spec.regions]
        transfer = run.choose(
            "transfer",
            "descend",
            lambda snapshot, seed: arm.plan_to_tsr(
                raised, snapshot=snapshot, planner=planner, seed=seed
            ),
            lambda start: _down(ctx, _allowing(start, spec), checker, spec),
            spec.transfer_attempts,
        )
        run.move("transfer", transfer, planner, spec.max_ticks)

        run.replace_allowance(spec.end_effector, (*hand, spec.target))
        run.record(
            "allow_target", True, f"allowed {', '.join((*hand, spec.target))}", Evidence.PLANNED
        )

        _descend(run, ctx, planner, spec)

        run.grip(spec.gripper, spec.open_opening, spec.settle_ticks)
        ctx.detach(spec.object)
        run.held = False
        run.record("release", True, "opened and detached", Evidence.COMMANDED)

        retreat = arm.plan_to_end_effector_offset(
            (0.0, 0.0, 1.0),
            spec.retreat_m,
            planner=planner,
            position_tolerance=spec.line_tolerance_m,
        )
        run.move("retreat", retreat, planner, spec.max_ticks)

        pose = next(o.pose for o in ctx.snapshot().objects if o.name == spec.object)
        run.values["placed_pose"] = {
            "position": list(pose.position),
            "quat_wxyz": list(pose.quat_wxyz),
        }
        inside = [i for i, r in enumerate(spec.regions) if _inside(pose, r, spec.upright_rad)]
        run.record(
            "placed",
            bool(inside),
            f"in region {inside[0]}" if inside else f"outside every region at {pose.position}",
            Evidence.PRIVILEGED,
        )
    except Failed:
        pass
    return run.result()


def _descend(run: Run, ctx: RobotContext, planner: Planner, spec: PlaceSpec) -> None:
    """The guarded terminal descent: planned, compiled for the runtime, executed."""
    for _ in range(spec.settle_ticks):
        ctx.step()
    view = ctx.manipulator(spec.manipulator)
    snapshot = ctx.snapshot()
    plan = _down(ctx, snapshot, planner, spec)
    if not plan.succeeded:
        run.record("descend", False, f"planning {plan.status.value}", Evidence.PLANNED)
    scene = ctx.planning_scene(snapshot, view.group)
    try:
        ready = prepare(
            plan,
            scene,
            validator=planner.validator(plan.request, scene),
            description=ctx.description,
            contract=ctx.trajectory_contract,
        )
    except SsrobotError as e:
        run.record("descend", False, f"{e.code}: {e.message}", Evidence.PLANNED)
    command = GuardedJointTrajectory(trajectory=ready.command, guard=spec.guard)
    status = ctx.run_until(
        ctx.submit(command, snapshot=snapshot, source="planner"), max_ticks=spec.max_ticks
    )
    outcome = status.guard
    if status.state is not ExecutionState.SUCCEEDED or outcome is None:
        code = "" if status.diagnostic is None else f": {status.diagnostic.code}"
        run.record("descend", False, f"{status.state.value}{code}", Evidence.COMMANDED)
    assert outcome is not None
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
    )


def _down(
    ctx: RobotContext,
    snapshot: SceneSnapshot,
    planner: Planner,
    spec: PlaceSpec,
) -> PlanResult:
    """The straight descent from ``snapshot``, to ``overshoot_m`` past the regions."""
    return ctx.manipulator(spec.manipulator).plan_to_end_effector_offset(
        (0.0, 0.0, -1.0),
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


def _raised(region: Tsr, height: float) -> Tsr:
    x, y, z = region.origin.position
    return replace(
        region, origin=Pose(position=(x, y, z + height), quat_wxyz=region.origin.quat_wxyz)
    )


def _inside(pose: Pose, region: Tsr, upright_rad: float) -> bool:
    """Whether ``pose`` is in ``region``: its displacement from the region's origin,
    less the offset, within the position bounds, and its roll and pitch within
    ``upright_rad``. Yaw is free."""
    local = compose(compose(invert(region.origin), pose), invert(region.offset))
    within = all(
        low - 1e-3 <= v <= high + 1e-3
        for v, (low, high) in zip(local.position, region.position_bounds, strict=True)
    )
    roll, pitch, _ = roll_pitch_yaw(local.quat_wxyz)
    return (
        within and abs(roll) <= upright_rad and abs(pitch) <= upright_rad and not math.isnan(roll)
    )
