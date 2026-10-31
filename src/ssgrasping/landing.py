"""Observe where a dropped object comes to rest, alongside whatever the robot does next.

A ``drop`` ends at the confirmed release, so the robot is free to move on at once.
Whether the object then landed where it should is a separate question, and ``landing``
answers it: a passive ``SkillExecution`` (see ``ssgrasping.execution``) that commands
nothing and plans nothing, so it runs alongside any other skill. It has one stage:
1. ``landed``: the object is observed at rest, by the caller's ``rest`` criteria, within
   ``rest_timeout_ns`` (ssrobot's ``watch_rest``), and the reading that established rest
   lies in one of the ``inside`` regions, within ``region_tolerance`` of all six bounds.
   An object still attached when the landing starts has not landed, and is not watched.
   A watch that ends any other way (``deadline``, ``unavailable``, ``malformed``,
   ``canceled``) fails the stage with that outcome, and no region is judged. The stage's
   evidence is the pose channel's: privileged simulator truth, or portable.

``rest`` has no default: when an object counts as still depends on its pose source and
the task, so the application declares it, with the evidence it was measured from.
Create the landing as soon as the drop returns: the timeout counts from then. Advance it
with ``run_together``, beside the next skill, or from the driver's own step loop.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ssrobot import ObjectRest, Pose, RestCriteria, RestOutcome, RobotContext, Tsr

from ssgrasping._poses import inside
from ssgrasping._skill import Body, Run, Wait, check_spec, rest_evidence
from ssgrasping.execution import SkillExecution
from ssgrasping.outcomes import Evidence


@dataclass(frozen=True, slots=True)
class LandingSpec:
    """Where a released object must come to rest, and how to tell it has."""

    object: str
    inside: Sequence[Tsr]
    """Where the object's frame may come to rest, in the world, judged on the reading of
    its pose channel that established rest."""
    rest: RestCriteria
    """When the object counts as at rest: the application's declared criteria for its
    pose channel, with their ``source``."""
    rest_timeout_ns: int
    """How long after the landing is created the object may take to come to rest before
    the landing fails: a failure bound, not a wait. It delays only this check, never the
    robot."""
    region_tolerance: tuple[float, float] = (2e-3, 0.05)
    """How far, in metres and radians, a landed object may lie outside an ``inside``
    region's bounds and still count as in it."""

    def __post_init__(self) -> None:
        check_spec(len(self.inside) > 0, "at least one region", "inside")
        check_spec(isinstance(self.rest, RestCriteria), "a RestCriteria", "rest")
        check_spec(
            isinstance(self.rest_timeout_ns, int) and self.rest_timeout_ns > 0,
            "a positive number of nanoseconds",
            "rest_timeout_ns",
        )
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
    if run.attached():
        run.record("landed", False, f"{spec.object!r} is still attached", Evidence.PLANNED)
    watch = ctx.watch_rest(
        ObjectRest(object=spec.object, criteria=spec.rest),
        deadline_ns=ctx.now.time_ns + spec.rest_timeout_ns,
    )
    yield Wait(watch=watch)
    rest = watch.result
    assert rest is not None  # a watch is done once it has a result
    run.values["landed_rest"] = rest_evidence(rest)
    evidence = Evidence.PRIVILEGED if rest.channel.privileged else Evidence.PORTABLE
    if rest.outcome is not RestOutcome.AT_REST:
        run.record("landed", False, f"rest {rest.outcome.value}: {rest.detail}", evidence)
    x, y, z, qw, qx, qy, qz = rest.window[-1].value  # the reading that established rest
    pose = Pose(position=(x, y, z), quat_wxyz=(qw, qx, qy, qz))
    run.values["landed_pose"] = {
        "position": list(pose.position),
        "quat_wxyz": list(pose.quat_wxyz),
    }
    within = [i for i, r in enumerate(spec.inside) if inside(pose, r, spec.region_tolerance)]
    run.record(
        "landed",
        bool(within),
        f"in region {within[0]}" if within else f"outside every region at {pose.position}",
        evidence,
    )
