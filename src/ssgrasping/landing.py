"""Score a dropped object's ``landed`` estimate against where it came to rest, alongside
whatever the robot does next: evaluation only.

A ``drop`` ends at the confirmed release, so the robot is free to move on at once, and
records ``landed`` as ``unobserved``: no portable signal yet tells where the object came
to rest. ``landing`` scores that estimate where the runtime has the object's privileged
pose: a passive ``SkillExecution`` (see ``ssgrasping.execution``) that commands nothing
and plans nothing, so it runs alongside any other skill. Its reference decides nothing:
it is a ``TransitionEvaluation`` in its result, never a stage's outcome. It has one
stage:
1. ``observed``: after ``land_ns`` (a temporary fixed wait, until a landing can be judged
   on observed object rest: personalrobotics/ssgrasping#36), the object's privileged
   pose is read, and the estimate is scored: confirmed if the object rests in one of
   the ``inside`` regions, within ``region_tolerance`` of all six bounds. The stage
   fails only if nothing can be read: the object is still attached, or the runtime
   declares no pose channel for it.

Create the landing as soon as the drop returns, with the drop's ``landed`` estimate:
``land_ns`` counts from then. Advance it with ``run_together``, beside the next skill,
or from the driver's own step loop.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ssrobot import RobotContext, Tsr

from ssgrasping._poses import inside
from ssgrasping._skill import Body, Run, Wait, check_spec
from ssgrasping.execution import SkillExecution
from ssgrasping.outcomes import Evidence, Reason, Transition, TransitionEstimate


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


def landing(ctx: RobotContext, spec: LandingSpec, estimate: TransitionEstimate) -> SkillExecution:
    """A pending evaluation of ``estimate``, the drop's ``landed``, against where
    ``spec.object`` comes to rest; see the module.

    Advance it, or drive it with ``run_skill`` or ``run_together``."""
    if estimate.transition is not Transition.LANDED:
        raise ValueError(f"landing scores a landed estimate, not {estimate.transition.value}")
    run = Run(ctx, "landing", spec.object)
    return SkillExecution(run, _landing(run, spec, estimate))


def _landing(run: Run, spec: LandingSpec, estimate: TransitionEstimate) -> Body:
    ctx = run.ctx
    run.begin("observed")
    yield Wait(until_ns=ctx.now.time_ns + spec.land_ns)
    if run.attached():
        run.record(
            "observed",
            False,
            f"{spec.object!r} is still attached",
            Evidence.PLANNED,
            Reason.STILL_ATTACHED,
        )
    run.evaluate(
        estimate, lambda pose: any(inside(pose, r, spec.region_tolerance) for r in spec.inside)
    )
    if not run.evaluations:
        run.record(
            "observed",
            False,
            f"the runtime declares no privileged pose of {spec.object!r}",
            Evidence.PLANNED,
            Reason.UNAVAILABLE,
        )
    run.record("observed", True, "scored against the privileged pose", Evidence.PRIVILEGED)
