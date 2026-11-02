"""Drop a held object into a container from above.

``drop`` is the container counterpart of ``place``: it returns a passive
``SkillExecution`` (see ``ssgrasping.execution``) that runs these stages, recording
each. Nothing is lowered under a guard and nothing touches the container; the object
falls.
1. ``preflight``: every reference resolves, and every command and channel the drop uses
   is available, with a guard on the opening channel for the transfer. Nothing is
   commanded before it passes.
2. ``transfer``: plan the end effector into the drop regions, with the hand alone
   allowed, at ``carry_scaling`` of the joints' velocity and acceleration limits, and
   execute. With a retreat, a plan is kept only if the retreat also plans from where it
   ends; up to ``transfer_attempts`` seeds are tried. The regions are given for the end
   effector, as a container's drop zone usually is: the hand over the opening, pointing
   in. The transfer carries the object under an opening guard midway between the
   opening it is held at when the drop begins and ``close_opening``: if the guard stops
   it, ``retained`` is refuted, and the transfer fails with the object reported not
   held.
3. ``release``: open the gripper, require the command to succeed and the opening to
   reach ``released_opening``, and detach. An unconfirmed release keeps the attachment
   and reports the object's held state as unknown.
   A confirmed release then hands the object to the caller's ``released`` policy.
4. ``retreat``: move straight along the container's outward ``normal`` by
   ``retreat_m``; none when ``retreat_m`` is 0, so the next motion can start, and be
   planned ahead, from where the transfer ends.

The drop ends there: the robot is free to move on while the object falls. ``landed`` is
estimated at the confirmed release, and is always ``unobserved``
(``no_portable_signal``): no portable signal yet tells where the object came to rest,
and whether to assume it is the application's. Where the runtime has the object's
privileged pose, ``landing`` (``ssgrasping.landing``) scores that estimate, for
evaluation only, alongside whatever the robot does next.

A failure before ``release`` leaves the object attached, and held unless the transfer
refuted ``retained``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ssrobot import (
    Planner,
    PlanProjection,
    PlanResult,
    RobotContext,
    SceneSnapshot,
    Tsr,
)

from ssgrasping._skill import (
    GRIPPER_TIMEOUT_NS,
    Body,
    Lookahead,
    Run,
    check_spec,
    preflight,
    retention,
    unit,
)
from ssgrasping.execution import SkillExecution
from ssgrasping.outcomes import Criterion, Evidence, Reason, Transition, TransitionEstimate


@dataclass(frozen=True, slots=True)
class DropSpec:
    """Everything a drop needs, explicitly."""

    manipulator: str
    end_effector: str
    gripper: str
    opening_channel: str
    """The gripper-opening channel read to confirm release: one value, 0 closed to 1
    open."""
    object: str
    container: str
    """The robot frame or scene fixture the object is dropped into; never allowed to
    touch the hand or the held object."""
    normal: tuple[float, float, float]
    """The container opening's outward unit normal, in the world: the retreat's
    direction."""
    regions: Sequence[Tsr]
    """Where the end effector may be when it lets go, in the world."""
    retreat_m: float
    open_opening: float = 1.0
    close_opening: float = 0.0
    """The closing target the object is held with: the grasp's ``close_opening``. The
    fingers run on toward it if the object leaves the hand."""
    released_opening: float = 0.9
    """The opening that confirms the fingers have let go."""
    gripper_timeout_ns: int = GRIPPER_TIMEOUT_NS
    """The deadline on each gripper command, on the runtime clock: a failure bound, not a
    settle. A command ends when the fingers' motion ends; one that has not by then fails
    its stage (``timed_out``)."""
    line_tolerance_m: float = 1e-3
    carry_scaling: float = 1.0
    """The fraction of the joints' velocity and acceleration limits the transfer, made
    while holding the object, is planned at: a held object can slip from a grasp that
    holds it by friction alone."""
    transfer_attempts: int = 8
    """How many seeds to try for a transfer whose retreat also plans."""
    clearance_m: float = 0.0
    """The clearance the transfer, a move through free space carrying the object, keeps
    from the environment (``PlanningRequest.clearance_m``). The retreat, which starts
    over the container, keeps none."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "normal", unit(self.normal, "normal"))
        check_spec(len(self.regions) > 0, "at least one region", "regions")
        check_spec(self.retreat_m >= 0, "must be non-negative", "retreat_m")
        check_spec(self.line_tolerance_m > 0, "must be positive", "line_tolerance_m")
        check_spec(self.gripper_timeout_ns > 0, "must be positive", "gripper_timeout_ns")
        for name in ("open_opening", "close_opening", "released_opening"):
            check_spec(0.0 <= getattr(self, name) <= 1.0, "between 0 and 1", name)
        check_spec(self.transfer_attempts >= 1, "at least 1", "transfer_attempts")
        check_spec(0.0 < self.carry_scaling <= 1.0, "in (0, 1]", "carry_scaling")
        check_spec(
            math.isfinite(self.clearance_m) and self.clearance_m >= 0,
            "must be finite and non-negative",
            "clearance_m",
        )


def drop(
    ctx: RobotContext,
    planner: Planner,
    spec: DropSpec,
    *,
    check_planner: Planner | None = None,
    lookahead: Lookahead | None = None,
    released: Callable[[TransitionEstimate], None] | None = None,
) -> SkillExecution:
    """A pending drop of the held ``spec.object`` into ``spec.container``; see the
    module. Advance it, or drive it with ``run_skill``.

    ``check_planner`` plans the retreat checks from candidate transfers, as in
    ``grasp``.

    With ``lookahead``, the boundaries where the drop cannot look ahead are logged in
    ``values["lookahead"]``: its stages after the transfer depend on outcomes not yet
    known.

    ``released`` is the application's policy for the object once the release is
    confirmed and it is detached: it is called with the estimate the release leaves
    ``unobserved``, before anything else is planned. With registered object poses
    (ssrobot's ``ObjectPoses.REGISTERED``) the object's pose is then unknown, and a
    retreat cannot plan around it until the policy registers or removes it."""
    run = Run(ctx, "drop", spec.object, lookahead=lookahead)
    checker = planner if check_planner is None else check_planner
    return SkillExecution(run, _drop(run, planner, checker, spec, released))


def _drop(
    run: Run,
    planner: Planner,
    checker: Planner,
    spec: DropSpec,
    released: Callable[[TransitionEstimate], None] | None,
) -> Body:
    ctx = run.ctx
    attachment = ctx.scene.attachment(spec.object)
    run.held = None if attachment is None else True
    preflight(
        run,
        manipulator=spec.manipulator,
        end_effector=spec.end_effector,
        gripper=spec.gripper,
        support=spec.container,
        channels={spec.opening_channel: (1,)},
        carries=spec.opening_channel,
    )
    if attachment is None:
        run.record(
            "transfer",
            False,
            f"{spec.object!r} is not attached",
            Evidence.PLANNED,
            Reason.NOT_ATTACHED,
        )
    arm = ctx.manipulator(spec.manipulator)
    retreats = spec.retreat_m > 0

    def choose() -> PlanResult:
        transfer, _, _ = run.choose(
            "transfer",
            "retreat" if retreats else None,
            lambda snapshot, seed: arm.plan_to_tsr(
                spec.regions,
                snapshot=snapshot,
                planner=planner,
                seed=seed,
                velocity_scaling=spec.carry_scaling,
                acceleration_scaling=spec.carry_scaling,
                clearance_m=spec.clearance_m,
            ),
            (lambda start: _away(ctx, start, checker, spec)) if retreats else None,
            spec.transfer_attempts,
            planner=planner,
        )
        return transfer

    carry = retention(
        spec.opening_channel,
        Criterion(
            "held_opening",
            float(run.observe(spec.opening_channel)[0]),
            "observed: the opening when the drop began",
        ),
        Criterion("close_opening", spec.close_opening, "DropSpec.close_opening"),
    )
    yield from run.move("transfer", choose(), planner, retain=carry)

    run.boundary(
        "release", "whether the fingers let go decides what follows; detaching changes the scene"
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
    landed = run.unobserved(Transition.LANDED, "release")
    if released is not None:
        released(landed)

    if retreats:
        run.begin("retreat")
        yield from run.move("retreat", _away(ctx, ctx.snapshot(), planner, spec), planner)


def _away(
    ctx: RobotContext,
    snapshot: SceneSnapshot | PlanProjection,
    planner: Planner,
    spec: DropSpec,
) -> PlanResult:
    """The straight retreat along the container's normal from ``snapshot``."""
    return ctx.manipulator(spec.manipulator).plan_to_end_effector_offset(
        spec.normal,
        spec.retreat_m,
        snapshot=snapshot,
        planner=planner,
        position_tolerance=spec.line_tolerance_m,
    )
