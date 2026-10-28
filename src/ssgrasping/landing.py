"""Observe where a dropped object comes to rest, alongside whatever the robot does next.

A ``drop`` ends at the confirmed release, so the robot is free to move on at once.
Whether the object then landed where it should is a separate question, and ``landing``
answers it: a passive ``SkillExecution`` (see ``ssgrasping.execution``) that commands
nothing and plans nothing, so it runs alongside any other skill. It has one stage:
1. ``landed``: after ``land_ns`` (a temporary fixed wait, until ssrobot can report the
   object at rest: personalrobotics/ssrobot#221), the object rests in one of the
   ``inside`` regions, within ``region_tolerance`` of all six bounds, by its privileged
   pose. An object still attached, or no longer in the scene, has not landed.

Create the landing as soon as the drop returns: ``land_ns`` counts from then. Advance it
with ``run_together``, beside the next skill, or from the driver's own step loop.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ssrobot import RobotContext, Tsr

from ssgrasping._poses import inside
from ssgrasping._skill import Body, Run, Wait, check_spec
from ssgrasping.execution import SkillExecution
from ssgrasping.outcomes import Evidence


@dataclass(frozen=True, slots=True)
class LandingSpec:
    """Where a released object must come to rest, and when to look."""

    object: str
    inside: Sequence[Tsr]
    """Where the object's frame may come to rest, in the world. The check reads
    privileged object poses."""
    region_tolerance: tuple[float, float] = (2e-3, 0.05)
    """How far, in metres and radians, a landed object may lie outside an ``inside``
    region's bounds and still count as in it."""
    land_ns: int = 1_000_000_000
    """How long after the landing is created the object is given to come to rest. A
    temporary fixed wait: it is to wait on the object observed at rest instead, once
    ssrobot can report that (personalrobotics/ssrobot#221). It delays only this check,
    never the robot."""

    def __post_init__(self) -> None:
        check_spec(len(self.inside) > 0, "at least one region", "inside")
        check_spec(self.land_ns >= 0, "must be non-negative", "land_ns")
        metres, radians = self.region_tolerance
        check_spec(metres >= 0 and radians >= 0, "non-negative", "region_tolerance")


def landing(ctx: RobotContext, spec: LandingSpec) -> SkillExecution:
    """A pending observation of where ``spec.object`` comes to rest; see the module.
    Advance it, or drive it with ``run_skill`` or ``run_together``."""
    run = Run(ctx, "landing", spec.object)
    return SkillExecution(run, _landing(run, spec))


def _landing(run: Run, spec: LandingSpec) -> Body:
    ctx = run.ctx
    run.begin("landed")
    yield Wait(until_ns=ctx.now.time_ns + spec.land_ns)
    if run.attached():
        run.record("landed", False, f"{spec.object!r} is still attached", Evidence.PLANNED)
    pose = next((o.pose for o in ctx.snapshot().objects if o.name == spec.object), None)
    if pose is None:
        run.record("landed", False, f"{spec.object!r} is not in the scene", Evidence.PRIVILEGED)
    assert pose is not None
    run.values["landed_pose"] = {
        "position": list(pose.position),
        "quat_wxyz": list(pose.quat_wxyz),
    }
    within = [i for i, r in enumerate(spec.inside) if inside(pose, r, spec.region_tolerance)]
    run.record(
        "landed",
        bool(within),
        f"in region {within[0]}" if within else f"outside every region at {pose.position}",
        Evidence.PRIVILEGED,
    )
