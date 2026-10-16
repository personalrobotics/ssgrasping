"""Grasp an object resting on a support, and lift it clear.

``grasp`` runs these stages, recording each:
1. ``open``: open the gripper, so the pregrasp is planned with the fingers as they
   will approach.
2. ``pregrasp``: plan into the pregrasp regions, and keep the plan only if the straight
   approach also plans from where it ends; otherwise plan again with the next seed, up to
   ``pregrasp_attempts``. The pregrasp alone is checked only for collision, so a region
   can be reachable while its approach is not. Then execute it.
3. ``approach``: move straight along the hand's approach axis, by ``approach_m``,
   planned again from where the arm actually is.
4. ``close``: close the gripper.
5. ``acquired``: classify the opening as held, missed, or indeterminate. This is
   portable evidence, read before anything is attached. Only ``held`` continues; a miss
   reopens the gripper, the one cleanup whose meaning is unambiguous.
6. ``attach``: declare the object held, allowed to touch the hand and its support.
7. ``lift``: move straight up by ``lift_m``.
8. ``restrict``: replace the allowance with the hand alone, so nothing else may be
   touched from now on.
9. ``clear``: the held object is clear of everything, by the planner's validator.
10. ``retained``: the attachment monitor still reports the object held. This is
    privileged evidence about the motion; it never stands in for ``acquired``.

After ``attach``, a failure leaves the attachment declared and reports the object as
held: the physically conservative state. Retrying or recovering is the application's.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ssrobot import ManipulatorView, Planner, PlanResult, RobotContext, SceneSnapshot, Tsr

from ssgrasping._poses import axis
from ssgrasping._skill import Failed, Run
from ssgrasping.outcomes import Evidence, SkillResult


@dataclass(frozen=True, slots=True)
class GraspSpec:
    """Everything a grasp needs, explicitly: nothing is inferred from names."""

    manipulator: str
    """The manipulator that moves, by name."""
    end_effector: str
    """The end effector the object is held by; its frame is the one the regions bind."""
    gripper: str
    opening_channel: str
    """The gripper-opening channel read to verify the grasp."""
    object: str
    support: str
    """The robot frame or scene fixture the object rests on; allowed only for the lift."""
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
    settle_ticks: int = 500
    line_tolerance_m: float = 1e-3
    """Position tolerance of the straight approach and lift."""
    max_ticks: int = 20_000
    pregrasp_attempts: int = 8
    """How many seeds to try for a pregrasp whose approach also plans."""


def grasp(
    ctx: RobotContext, planner: Planner, spec: GraspSpec, *, check_planner: Planner | None = None
) -> SkillResult:
    """Grasp ``spec.object`` and lift it clear of its support; see the module.

    ``check_planner`` plans the approach checks from candidate pregrasps (``planner``
    when None). A feasible short line plans in few iterations, so a planner with a small
    iteration budget rejects an infeasible one quickly, and deterministically, where
    ``planner`` would spend its whole budget first."""
    run = Run(ctx, "grasp", spec.object)
    checker = planner if check_planner is None else check_planner
    arm = ctx.manipulator(spec.manipulator)
    try:
        run.grip(spec.gripper, spec.open_opening, spec.settle_ticks)
        run.record(
            "open", True, f"opening {run.observe(spec.opening_channel)[0]:.3f}", Evidence.PORTABLE
        )
        pregrasp = run.choose(
            "pregrasp",
            "approach",
            lambda snapshot, seed: arm.plan_to_tsr(
                list(spec.pregrasps), snapshot=snapshot, planner=planner, seed=seed
            ),
            lambda start: _approach(arm, ctx, start, checker, spec),
            spec.pregrasp_attempts,
        )
        run.move("pregrasp", pregrasp, planner, spec.max_ticks)

        approach = _approach(arm, ctx, ctx.snapshot(), planner, spec)
        run.move("approach", approach, planner, spec.max_ticks)

        run.grip(spec.gripper, spec.close_opening, spec.settle_ticks)
        opening = float(run.observe(spec.opening_channel)[0])
        run.values["closed_opening"] = opening
        run.record("close", True, f"opening {opening:.3f}", Evidence.PORTABLE)
        low, high = spec.held_opening
        if opening < low:
            run.held = False
            run.grip(spec.gripper, spec.open_opening, spec.settle_ticks)  # unambiguous
            run.record(
                "acquired", False, f"missed: opening {opening:.3f} < {low}", Evidence.PORTABLE
            )
        if opening > high:
            run.record(
                "acquired",
                False,
                f"indeterminate: opening {opening:.3f} > {high}",
                Evidence.PORTABLE,
            )
        run.held = True
        run.record(
            "acquired", True, f"held: opening {opening:.3f} in [{low}, {high}]", Evidence.PORTABLE
        )

        # ssrobot's own default allowance (the hand), read back, plus the support.
        ctx.attach(spec.object, spec.end_effector)
        hand = ctx.scene.attachment(spec.object).allow  # type: ignore[union-attr]
        run.replace_allowance(spec.end_effector, (*hand, spec.support))
        run.values["hand_allowance"] = list(hand)
        run.record("attach", True, f"allowed {', '.join((*hand, spec.support))}", Evidence.PLANNED)

        before = _object_z(ctx, spec.object)
        lift = arm.plan_to_end_effector_offset(
            (0.0, 0.0, 1.0), spec.lift_m, planner=planner, position_tolerance=spec.line_tolerance_m
        )
        run.move("lift", lift, planner, spec.max_ticks)

        run.replace_allowance(spec.end_effector, hand)
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

        attachment = ctx.scene.attachment(spec.object)
        risen = _object_z(ctx, spec.object) - before
        run.values["object_risen_m"] = round(risen, 6)
        retained = attachment is not None and attachment.held
        run.record(
            "retained",
            retained,
            f"attachment monitor: held={retained}, object rose {risen:.4f} m",
            Evidence.PRIVILEGED,
        )
    except Failed:
        pass
    return run.result()


def _approach(
    arm: ManipulatorView,
    ctx: RobotContext,
    snapshot: SceneSnapshot,
    planner: Planner,
    spec: GraspSpec,
) -> PlanResult:
    """The straight approach along the end effector's z axis, from ``snapshot``."""
    frame = arm.frame
    assert frame is not None  # the end effector's frame
    q = tuple(snapshot.position(j) for j in ctx.description.group(arm.group).joints)
    pose = ctx.planning_scene(snapshot, arm.group).forward_kinematics(q, frame)
    return arm.plan_to_end_effector_offset(
        axis(pose, 2),
        spec.approach_m,
        snapshot=snapshot,
        planner=planner,
        position_tolerance=spec.line_tolerance_m,
    )


def _object_z(ctx: RobotContext, name: str) -> float:
    return next(o.pose.position[2] for o in ctx.snapshot().objects if o.name == name)
