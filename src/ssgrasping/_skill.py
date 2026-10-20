"""What the staged skills share: recording stages, and waiting on ssrobot calls.

A skill body is a generator that yields a ``Wait`` wherever it waits on the runtime; the
helpers here that wait are generators too, used with ``yield from``. Every motion goes
through ssrobot's public planning and execution calls. Nothing here plans, solves IK,
retimes, advances a clock, or touches MuJoCo.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass, replace
from typing import Any

from ssrobot import (
    CommandKind,
    Execution,
    ExecutionState,
    GripperCommand,
    ObservationRequest,
    Planner,
    PlanResult,
    RobotContext,
    SceneSnapshot,
    SsrobotError,
    ValidationError,
    WrenchGuard,
)

from ssgrasping.outcomes import Evidence, SkillResult, SkillState, Stage


@dataclass(frozen=True, slots=True)
class Wait:
    """What a skill waits for: an execution to finish, or the clock to reach a time."""

    execution: Execution | None = None
    until_ns: int | None = None

    def satisfied(self, ctx: RobotContext) -> bool:
        if self.execution is not None:
            return self.execution.done
        assert self.until_ns is not None
        return ctx.now.time_ns >= self.until_ns


Body = Generator[Wait, None, None]


class Failed(Exception):
    """A stage failed; the skill ends with the stages so far."""


class Run:
    """One skill's progress: its stages, the stage in progress, and measured values."""

    def __init__(self, ctx: RobotContext, skill: str, obj: str) -> None:
        self.ctx = ctx
        self.skill = skill
        self.obj = obj
        self.stage: str | None = None
        self.stages: list[Stage] = []
        self.values: dict[str, object] = {}
        self.held: bool | None = None

    def attached(self) -> bool:
        return self.ctx.scene.attachment(self.obj) is not None

    def begin(self, name: str) -> None:
        """Name the stage now in progress, for a failure or cancellation inside it."""
        self.stage = name

    def record(self, name: str, succeeded: bool, detail: str, evidence: Evidence) -> None:
        self.stage = name
        self._append(name, succeeded, detail, evidence)
        if not succeeded:
            raise Failed(f"{name}: {detail}")

    def fail(self, detail: str) -> None:
        """Record the stage in progress as failed by an expected ssrobot refusal."""
        self._append(self.stage or "start", False, detail, Evidence.COMMANDED)

    def cancelled(self, reason: str) -> None:
        """Record the stage in progress as cancelled."""
        self._append(self.stage or "start", False, reason, Evidence.COMMANDED)

    def ok(self) -> SkillState:
        good = bool(self.stages) and all(s.succeeded for s in self.stages)
        return SkillState.SUCCEEDED if good else SkillState.FAILED

    def result(self, state: SkillState) -> SkillResult:
        return SkillResult(
            skill=self.skill,
            state=state,
            stages=tuple(self.stages),
            attachment=self.ctx.scene.attachment(self.obj),
            values=dict(self.values),
        )

    def _append(self, name: str, succeeded: bool, detail: str, evidence: Evidence) -> None:
        self.stages.append(
            Stage(
                name=name,
                succeeded=succeeded,
                detail=detail,
                time_ns=self.ctx.now.time_ns,
                revision=self.ctx.scene.revision,
                held=self.held,
                attached=self.attached(),
                evidence=evidence,
            )
        )

    def move(self, name: str, plan: PlanResult, planner: Planner) -> Body:
        """Execute ``plan`` with ``ctx.execute`` and wait for it to finish."""
        self.begin(name)
        if not plan.succeeded:
            reason = "; ".join(d.message for d in plan.diagnostics) or plan.status.value
            self.record(name, False, f"planning {plan.status.value}: {reason}", Evidence.PLANNED)
        try:
            execution = self.ctx.execute(plan, planner=planner)
        except SsrobotError as e:
            self.record(name, False, f"{e.code}: {e.message}", Evidence.PLANNED)
        yield Wait(execution=execution)
        self.finished(name, execution)
        self.record(name, True, "succeeded", Evidence.COMMANDED)

    def finished(self, name: str, execution: Execution) -> None:
        """Fail ``name`` unless ``execution`` succeeded."""
        status = execution.status
        if status.state is not ExecutionState.SUCCEEDED:
            detail = status.state.value
            if status.diagnostic is not None:
                detail += f": {status.diagnostic.code}"
            self.record(name, False, detail, Evidence.COMMANDED)

    def grip(self, name: str, gripper: str, opening: float, settle_ns: int) -> Body:
        """Command ``gripper`` to ``opening``, require the command to succeed, then wait
        ``settle_ns`` for the fingers to settle."""
        self.begin(name)
        execution = self.ctx.submit(GripperCommand(gripper=gripper, opening=opening))
        yield Wait(execution=execution)
        self.finished(name, execution)
        yield Wait(until_ns=self.ctx.now.time_ns + settle_ns)

    def observe(self, channel: str) -> Any:
        return self.ctx.observe(ObservationRequest(channels=(channel,))).readings[0].value

    def replace_allowance(self, name: str, end_effector: str, allow: Sequence[str]) -> None:
        """Re-declare the held object's attachment with ``allow``, as a scene change and
        not a release: the same transform, detached and attached again with no step
        between them.

        Everything is checked before anything changes, and a failure records ``name``
        with the attachment untouched. An attachment the monitor already reports left its
        transform is not re-declared: that would discard the report, and the runtime
        would refuse the transform anyway."""
        attachment = self.ctx.scene.attachment(self.obj)
        if attachment is None:
            self.record(name, False, f"{self.obj!r} is not attached", Evidence.PLANNED)
        assert attachment is not None
        if not attachment.held:
            self.record(
                name,
                False,
                "attachment monitor: the object left its declared transform",
                Evidence.PRIVILEGED,
            )
        known = {f.name for f in self.ctx.description.frames} | set(self.ctx.info.fixtures)
        unknown = sorted(set(allow) - known)
        if unknown:
            self.record(
                name,
                False,
                f"neither robot frames nor fixtures: {', '.join(unknown)}",
                Evidence.PLANNED,
            )
        self.ctx.detach(self.obj)
        self.ctx.attach(self.obj, end_effector, transform=attachment.transform, allow=tuple(allow))

    def choose(
        self,
        name: str,
        then: str,
        plan: Callable[[SceneSnapshot, int], PlanResult],
        follow: Callable[[SceneSnapshot], PlanResult],
        attempts: int,
    ) -> PlanResult:
        """A plan for ``name`` from which the straight motion ``then`` also plans.

        A goal region is checked only for collision, so a plan can reach it at a
        configuration from which the next straight move is infeasible. Each seed's plan
        is kept only if ``follow`` plans from where it ends; this checks, and does not
        replace, the plan made again from where the arm actually ends up. The seeds
        tried are recorded in ``values[name + "_attempts"]``."""
        self.begin(name)
        snapshot = self.ctx.snapshot()
        tried: list[dict[str, object]] = []
        self.values[f"{name}_attempts"] = tried
        for seed in range(attempts):
            result = plan(snapshot, seed)
            if not result.succeeded:
                tried.append({"seed": seed, name: result.status.value})
                continue
            joints, q = _end(result)
            check = follow(_at(snapshot, joints, q))
            tried.append({"seed": seed, name: "succeeded", then: check.status.value})
            if check.succeeded:
                return result
        self.record(
            name, False, f"no plan in {attempts} seeds from which {then} plans", Evidence.PLANNED
        )
        raise AssertionError  # unreachable: record raised


def _end(result: PlanResult) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Where a successful plan ends: its path's last waypoint, or, for a planner that
    returns only a trajectory, the trajectory's goal."""
    if result.path is not None:
        return result.path.joints, result.path.waypoints[-1]
    assert result.trajectory is not None  # a successful plan has one or the other
    return result.trajectory.joints, result.trajectory.goal


def _at(snapshot: SceneSnapshot, joints: Sequence[str], q: Sequence[float]) -> SceneSnapshot:
    """``snapshot`` with ``joints`` at ``q``: a hypothetical start, for planning only."""
    at = dict(zip(joints, q, strict=True))
    positions = tuple(
        at.get(j, v) for j, v in zip(snapshot.joints, snapshot.positions, strict=True)
    )
    return replace(snapshot, positions=positions)


def preflight(
    run: Run,
    *,
    manipulator: str,
    end_effector: str,
    gripper: str,
    support: str,
    channels: dict[str, tuple[int, ...]],
    guard: WrenchGuard | None = None,
) -> None:
    """Record ``preflight``: every reference resolves, and every command, channel, and
    guard the skill will use is declared available, before anything is commanded."""
    ctx = run.ctx
    d, info = ctx.description, ctx.info
    problems = []
    arms = {m.name: m for m in d.manipulators}
    arm = arms.get(manipulator)
    if arm is None:
        problems.append(f"no manipulator {manipulator!r}")
    if end_effector not in {e.name for e in d.end_effectors}:
        problems.append(f"no end effector {end_effector!r}")
    if gripper not in {g.name for g in d.grippers}:
        problems.append(f"no gripper {gripper!r}")
    if run.obj not in info.objects:
        problems.append(f"no scene object {run.obj!r}")
    if support not in {f.name for f in d.frames} | set(info.fixtures):
        problems.append(f"{support!r} is neither a robot frame nor a fixture")
    commands = {(c.component, c.kind) for c in info.commands}
    if (gripper, CommandKind.GRIPPER) not in commands:
        problems.append(f"gripper commands for {gripper!r} are unavailable")
    if arm is not None and (arm.group, CommandKind.JOINT_TRAJECTORY) not in commands:
        problems.append(f"joint trajectories for {arm.group!r} are unavailable")
    specs = {c.name: c for c in d.channels}
    for name, shape in channels.items():
        if name not in info.channels:
            problems.append(f"channel {name!r} is unavailable")
        elif specs[name].shape != shape:
            problems.append(f"channel {name!r} has shape {specs[name].shape}, not {shape}")
    if guard is not None and arm is not None:
        guards = [g for g in info.guards if g.component == arm.group]
        if not any(guard.channel in g.channels and (g.tare or not guard.tare) for g in guards):
            problems.append(f"no guard on {arm.group!r} reads {guard.channel!r} as asked")
    run.record(
        "preflight",
        not problems,
        "; ".join(problems)
        or (
            "references, commands, and channels"
            if guard is None
            else "references, commands, channels, and guard"
        )
        + " available",
        Evidence.PLANNED,
    )


def check_spec(ok: bool, message: str, path: str) -> None:
    """Refuse a malformed skill spec at construction."""
    if not ok:
        raise ValidationError("invalid_spec", message, path=path)


def unit(v: Sequence[float], path: str) -> tuple[float, float, float]:
    """``v`` as a unit 3-vector, refusing anything else."""
    check_spec(len(v) == 3 and all(math.isfinite(c) for c in v), "a 3-vector", path)
    x, y, z = (float(c) for c in v)
    check_spec(abs(math.sqrt(x * x + y * y + z * z) - 1.0) < 1e-6, "a unit vector", path)
    return (x, y, z)
