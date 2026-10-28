"""Drop a held object into a container from above.

``drop`` is the container counterpart of ``place``: it returns a passive
``SkillExecution`` (see ``ssgrasping.execution``) that runs these stages, recording
each. Nothing is lowered under a guard and nothing touches the container; the object
falls.
1. ``preflight``: every reference resolves, and every command and channel the drop uses
   is available. Nothing is commanded before it passes.
2. ``transfer``: plan the end effector into the drop regions, with the hand alone
   allowed, at ``carry_scaling`` of the joints' velocity and acceleration limits, and
   execute. With a retreat, a plan is kept only if the retreat also plans from where it
   ends; up to ``transfer_attempts`` seeds are tried. The regions are given for the end
   effector, as a container's drop zone usually is: the hand over the opening, pointing
   in.
3. ``release``: first, the attachment monitor must still report the object held: one
   that left the grasp during the transfer fails here, with the attachment kept and
   the object conservatively reported held. Then open the gripper, require the command
   to succeed and the opening to reach ``released_opening``, and detach. An unconfirmed
   release keeps the attachment and reports the object's held state as unknown.
4. ``retreat``: move straight along the container's outward ``normal`` by
   ``retreat_m``; none when ``retreat_m`` is 0, so the next motion can start, and be
   planned ahead, from where the transfer ends.
5. ``dropped``, only if ``inside`` is given: after ``land_ns`` (a temporary fixed wait,
   until ssrobot can report the object at rest: personalrobotics/ssrobot#221), the
   object rests in one of the ``inside`` regions, within ``region_tolerance`` of all six
   bounds, by its privileged pose.

A failure before ``release`` leaves the object attached and held.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ssrobot import Planner, PlanProjection, PlanResult, RobotContext, SceneSnapshot, Tsr

from ssgrasping._poses import inside
from ssgrasping._skill import (
    GRIPPER_TIMEOUT_NS,
    Body,
    Lookahead,
    Run,
    Wait,
    check_spec,
    preflight,
    unit,
)
from ssgrasping.execution import SkillExecution
from ssgrasping.outcomes import Evidence


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
    inside: Sequence[Tsr] = ()
    """Where the object's frame may come to rest for the drop to count, in the world;
    empty to skip the ``dropped`` check, which reads privileged object poses."""
    region_tolerance: tuple[float, float] = (2e-3, 0.05)
    """How far, in metres and radians, a dropped object may lie outside an ``inside``
    region's bounds and still count as in it."""
    open_opening: float = 1.0
    released_opening: float = 0.9
    """The opening that confirms the fingers have let go."""
    gripper_timeout_ns: int = GRIPPER_TIMEOUT_NS
    """The deadline on each gripper command, on the runtime clock: a failure bound, not a
    settle. A command ends when the fingers' motion ends; one that has not by then fails
    its stage (``timed_out``)."""
    land_ns: int = 1_000_000_000
    """How long after the retreat the object is given to come to rest before ``dropped``.
    A temporary fixed wait: it is to wait on the object observed at rest instead, once
    ssrobot can report that (personalrobotics/ssrobot#221)."""
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
        check_spec(self.land_ns >= 0, "must be non-negative", "land_ns")
        check_spec(self.gripper_timeout_ns > 0, "must be positive", "gripper_timeout_ns")
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


def drop(
    ctx: RobotContext,
    planner: Planner,
    spec: DropSpec,
    *,
    check_planner: Planner | None = None,
    lookahead: Lookahead | None = None,
) -> SkillExecution:
    """A pending drop of the held ``spec.object`` into ``spec.container``; see the
    module. Advance it, or drive it with ``run_skill``.

    ``check_planner`` plans the retreat checks from candidate transfers, as in
    ``grasp``.

    With ``lookahead``, the boundaries where the drop cannot look ahead are logged in
    ``values["lookahead"]``: its stages after the transfer depend on outcomes not yet
    known."""
    run = Run(ctx, "drop", spec.object, lookahead=lookahead)
    checker = planner if check_planner is None else check_planner
    return SkillExecution(run, _drop(run, planner, checker, spec))


def _drop(run: Run, planner: Planner, checker: Planner, spec: DropSpec) -> Body:
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
    )
    if attachment is None:
        run.record("transfer", False, f"{spec.object!r} is not attached", Evidence.PLANNED)
    arm = ctx.manipulator(spec.manipulator)
    retreats = spec.retreat_m > 0
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
    yield from run.move("transfer", transfer, planner)

    run.boundary(
        "release", "whether the fingers let go decides what follows; detaching changes the scene"
    )
    run.begin("release")
    carried = ctx.scene.attachment(spec.object)
    if carried is None or not carried.held:
        run.record(
            "release",
            False,
            "attachment monitor: the object left its declared transform during the transfer",
            Evidence.PRIVILEGED,
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
        )
    ctx.detach(spec.object)
    run.held = False
    run.record("release", True, f"opening {opening:.3f}; detached", Evidence.PORTABLE)

    if retreats:
        run.begin("retreat")
        yield from run.move("retreat", _away(ctx, ctx.snapshot(), planner, spec), planner)

    if not spec.inside:
        return
    run.begin("dropped")
    yield Wait(until_ns=ctx.now.time_ns + spec.land_ns)
    pose = next(o.pose for o in ctx.snapshot().objects if o.name == spec.object)
    run.values["dropped_pose"] = {
        "position": list(pose.position),
        "quat_wxyz": list(pose.quat_wxyz),
    }
    within = [i for i, r in enumerate(spec.inside) if inside(pose, r, spec.region_tolerance)]
    run.record(
        "dropped",
        bool(within),
        f"in region {within[0]}" if within else f"outside every region at {pose.position}",
        Evidence.PRIVILEGED,
    )


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
