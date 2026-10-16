"""What the staged skills share: recording stages, and running ssrobot calls to an end.

Every motion goes through ssrobot's public planning and execution calls. Nothing here
plans, solves IK, retimes, steps a clock on its own schedule, or touches MuJoCo.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

from ssrobot import (
    ExecutionState,
    GripperCommand,
    ObservationRequest,
    Planner,
    PlanResult,
    RobotContext,
    SceneSnapshot,
    SsrobotError,
)

from ssgrasping.outcomes import Evidence, SkillResult, SkillStatus, Stage


class Failed(Exception):
    """A stage failed; the skill ends with the stages so far."""


class Run:
    """One skill's progress: its stages, and the values it measured."""

    def __init__(self, ctx: RobotContext, skill: str, obj: str) -> None:
        self.ctx = ctx
        self.skill = skill
        self.obj = obj
        self.stages: list[Stage] = []
        self.values: dict[str, object] = {}
        self.held: bool | None = None

    def attached(self) -> bool:
        return self.ctx.scene.attachment(self.obj) is not None

    def record(self, name: str, succeeded: bool, detail: str, evidence: Evidence) -> None:
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
        if not succeeded:
            raise Failed(f"{name}: {detail}")

    def result(self) -> SkillResult:
        ok = bool(self.stages) and all(s.succeeded for s in self.stages)
        return SkillResult(
            skill=self.skill,
            status=SkillStatus.SUCCEEDED if ok else SkillStatus.FAILED,
            stages=tuple(self.stages),
            values=dict(self.values),
        )

    def move(self, name: str, plan: PlanResult, planner: Planner, max_ticks: int) -> None:
        """Execute ``plan`` with ``ctx.execute`` and wait for it, bounded."""
        if not plan.succeeded:
            reason = "; ".join(d.message for d in plan.diagnostics) or plan.status.value
            self.record(name, False, f"planning {plan.status.value}: {reason}", Evidence.PLANNED)
        try:
            execution = self.ctx.execute(plan, planner=planner)
        except SsrobotError as e:
            self.record(name, False, f"{e.code}: {e.message}", Evidence.PLANNED)
        status = self.ctx.run_until(execution, max_ticks=max_ticks)
        succeeded = status.state is ExecutionState.SUCCEEDED
        detail = status.state.value
        if status.diagnostic is not None:
            detail += f": {status.diagnostic.code}"
        self.record(name, succeeded, detail, Evidence.COMMANDED)

    def grip(self, gripper: str, opening: float, settle_ticks: int) -> None:
        """Command ``gripper`` to ``opening`` and let it settle for ``settle_ticks``."""
        self.ctx.run_until(
            self.ctx.submit(GripperCommand(gripper=gripper, opening=opening)), max_ticks=1
        )
        for _ in range(settle_ticks):
            self.ctx.step()

    def observe(self, channel: str) -> Any:
        return self.ctx.observe(ObservationRequest(channels=(channel,))).readings[0].value

    def replace_allowance(self, end_effector: str, allow: Sequence[str]) -> None:
        """Re-declare the held object's attachment with ``allow``, as a scene change and
        not a release: detach and attach again, with no step between them."""
        self.ctx.detach(self.obj)
        self.ctx.attach(self.obj, end_effector, allow=tuple(allow))

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
        snapshot = self.ctx.snapshot()
        tried: list[dict[str, object]] = []
        self.values[f"{name}_attempts"] = tried
        for seed in range(attempts):
            result = plan(snapshot, seed)
            if not result.succeeded or result.path is None:
                tried.append({"seed": seed, name: result.status.value})
                continue
            end = _at(snapshot, result.path.joints, result.path.waypoints[-1])
            check = follow(end)
            tried.append({"seed": seed, name: "succeeded", then: check.status.value})
            if check.succeeded:
                return result
        self.record(
            name, False, f"no plan in {attempts} seeds from which {then} plans", Evidence.PLANNED
        )
        raise AssertionError  # unreachable: record raised


def _at(snapshot: SceneSnapshot, joints: Sequence[str], q: Sequence[float]) -> SceneSnapshot:
    """``snapshot`` with ``joints`` at ``q``: a hypothetical start, for planning only."""
    at = dict(zip(joints, q, strict=True))
    positions = tuple(
        at.get(j, v) for j, v in zip(snapshot.joints, snapshot.positions, strict=True)
    )
    return replace(snapshot, positions=positions)
