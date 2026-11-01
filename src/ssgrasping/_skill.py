"""What the staged skills share: recording stages, and waiting on ssrobot calls.

A skill body is a generator that yields a ``Wait`` wherever it waits on the runtime, or
on planning running on a worker; the helpers here that wait are generators too, used
with ``yield from``. Every motion goes through ssrobot's public planning and execution
calls. Nothing here plans, solves IK, retimes, advances a clock, or touches MuJoCo.

A carry runs under an opening guard (``Retention``), and its ``retained`` transition is
estimated from how the guard ended: a guard that stops the carry refutes it and fails
the stage at once, with the object reported not held.

Given a ``Lookahead``, a stage whose successor's request is fully determined plans that
successor while its own motion executes, on ``ctx.predict`` of it (``Run.ahead``), and
the successor runs as a continuation (``Run.move`` given an ``Ahead``). Every lookahead,
and every boundary where a skill stops looking ahead because the next request depends
on an outcome not yet known, is logged in ``values["lookahead"]``.
"""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Callable, Generator, Mapping, Sequence
from concurrent.futures import Future, wait
from dataclasses import dataclass, field, replace
from typing import Any

from ssrobot import (
    CommandKind,
    Continuation,
    Execution,
    ExecutionState,
    GripperCommand,
    GuardCause,
    JointRest,
    ObservationRequest,
    OpeningGuard,
    Planner,
    PlanningRequest,
    PlanProjection,
    PlanResult,
    Pose,
    PredictedSnapshot,
    Quantity,
    RestResult,
    RestWatch,
    RobotContext,
    SceneSnapshot,
    SsrobotError,
    ValidationError,
    WrenchGuard,
    dumps,
)

from ssgrasping.outcomes import (
    Criterion,
    EstimateReason,
    Evidence,
    Observed,
    Reason,
    SkillResult,
    SkillState,
    Stage,
    Transition,
    TransitionEstimate,
    TransitionEvaluation,
    TransitionOutcome,
)


@dataclass(frozen=True, slots=True)
class Wait:
    """What a skill waits for, exactly one of: an execution to finish, the clock to reach
    a time, a plan being made on a worker, or a rest watch to end. Waiting on a plan alone
    needs no time to pass: a driver may block on it instead of stepping (``planning``)."""

    execution: Execution | None = None
    until_ns: int | None = None
    future: Future[Any] | None = None
    watch: RestWatch | None = None

    def __post_init__(self) -> None:
        given = (self.execution, self.until_ns, self.future, self.watch)
        if sum(w is not None for w in given) != 1:
            raise ValueError("a Wait waits on exactly one thing")

    @property
    def planning(self) -> bool:
        return self.future is not None

    def satisfied(self, ctx: RobotContext) -> bool:
        if self.future is not None:
            return self.future.done()
        if self.execution is not None:
            return self.execution.done
        if self.watch is not None:
            return self.watch.done
        assert self.until_ns is not None
        return ctx.now.time_ns >= self.until_ns


Body = Generator[Wait, None, None]

GRIPPER_TIMEOUT_NS = 5_000_000_000
"""The default deadline on a gripper command (``gripper_timeout_ns``): a failure bound
for a detector or controller that never reports the end of the fingers' motion, not a
wait. The slowest measured motion is ADA's unobstructed open, 1.49 s; Geodude's is 1.0 s
(personalrobotics/ssrobot#220)."""

REST_TIMEOUT_NS = 2_000_000_000
"""The default timeout on a joint-rest watch (``rest_timeout_ns``): a failure bound for
joints never observed at rest, not a wait; a watch ends as soon as rest is observed. The
latest measured rest, after a trajectory ends, is ADA's at 614 ms, Geodude's at 336 ms
(personalrobotics/ssrobot#225's evidence), so the bound leaves over 1.3 s of headroom."""
RECOVER = ("source_not_succeeded", "unpredicted_scene_change", "not_at_rest", "rest_unobservable")
"""Continuation refusals that fail the stage, for the application's recovery: the motion
before it failed; the scene changed in a way the prediction did not record; or rest,
ssrobot's gate before anything moves again, was not established (not observed at rest
within ``rest_timeout_ns``, or unobservable). Nothing is submitted around them. Every other refusal
(a small arrival mismatch, a rebase the planner cannot make or that is invalid in the
actual scene) is replanned from the actual state."""
MONITOR = "attachment_monitor"
"""The privileged reference a ``retained`` estimate is scored against, where there is one."""


@dataclass(frozen=True, slots=True)
class Lookahead:
    """How a skill looks ahead: pass one to plan each stage whose request is fully
    determined while the motion before it runs."""

    unsettled: tuple[str, ...] = ()
    """Objects every prediction makes no claim about, such as ones still settling."""
    arrival_tolerance: float | None = None
    """How far, per joint, the arrival may be from a continuation's start before the stage
    is replanned from where the arm is; None for the runtime's declared goal tolerance."""
    rest_timeout_ns: int = REST_TIMEOUT_NS
    """How long, after its source succeeds, a continuation watches for the joints to come
    to rest before the stage fails for recovery: a failure bound, not a wait."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "unsettled", tuple(self.unsettled))
        check_spec(
            isinstance(self.rest_timeout_ns, int) and self.rest_timeout_ns > 0,
            "a positive number of nanoseconds",
            "rest_timeout_ns",
        )
        if self.arrival_tolerance is not None:
            check_spec(
                math.isfinite(self.arrival_tolerance) and self.arrival_tolerance >= 0,
                "must be finite and non-negative",
                "arrival_tolerance",
            )


@dataclass(frozen=True, slots=True)
class Retention:
    """How a carry's ``retained`` is estimated: the opening guard it runs under, or None
    when the opening cannot tell (``no_opening_margin``), and the criteria behind it."""

    guard: OpeningGuard | None
    criteria: tuple[Criterion, ...]


def retention(channel: str, held: Criterion, target: Criterion) -> Retention:
    """The opening guard for a carry of an object held at ``held``, by fingers closing on
    ``target``: midway between them. An object that leaves lets the fingers run on toward
    the target; one still in hand keeps them at its width."""
    if held.value <= target.value:
        return Retention(None, (held, target))
    below = Criterion(
        "below",
        (held.value + target.value) / 2,
        f"midway between {held.name} and {target.name}",
    )
    return Retention(OpeningGuard(channel=channel, below=below.value), (held, target, below))


@dataclass(frozen=True, slots=True)
class Kept:
    """A follow-on plan kept from a feasibility check: ``plan``, made by ``planner`` on
    ``projection``, the projection of the plan before it. It continues only from the
    execution of exactly that plan (``Run.kept``)."""

    projection: PlanProjection
    plan: PlanResult
    planner: Planner


@dataclass(frozen=True, slots=True)
class Follow:
    """The next skill's first motion, to plan while this skill's last one runs: its
    ``stage`` name, the ``request`` for it on the prediction of that motion, and the
    ``planner`` that plans it, which nothing else may use until the plan is ready. A grasp
    plans it during its lift (``grasp(follow=...)``), and the next skill continues the
    plan (``drop(planned=...)``)."""

    stage: str
    request: Callable[[PredictedSnapshot], PlanningRequest]
    planner: Planner


@dataclass(slots=True)
class Ahead:
    """A stage's motion being planned on a prediction while the motion before it runs."""

    stage: str
    prediction: PredictedSnapshot
    planner: Planner
    future: Future[PlanResult]
    log: dict[str, object]
    """This lookahead's entry in ``values["lookahead"]``."""
    stop: threading.Event = field(default_factory=threading.Event)
    continuation: Continuation | None = None
    """A continuation already bound, for a plan kept from a feasibility check made on
    the projection of the plan its source runs; ``prediction`` is its source's."""

    def cancel(self) -> None:
        """Ask the planner to stop: the prediction will not be used."""
        self.stop.set()


def lookahead_entry(stage: str, prediction: PredictedSnapshot, **more: object) -> dict[str, object]:
    """A lookahead's log entry: what was planned ahead, on which prediction."""
    return {
        "stage": stage,
        "during": prediction.source,
        "prediction": {
            "base": prediction.base,
            "lineage": list(prediction.lineage),
            "moved": list(prediction.moved),
            "held": [o.name for o in prediction.held],
            "unsettled": list(prediction.unsettled),
            "releases": list(prediction.releases),
            "allowances": list(prediction.allowances),
        },
        **more,
    }


def rest_evidence(rest: RestResult) -> dict[str, object]:
    """What a rest result establishes, as structured evidence: how the watch ended, on
    which channel and by whose criteria, the window of readings that justified rest, how
    many readings it judged, and the longest still run it saw."""
    window = rest.window
    return {
        "outcome": rest.outcome.value,
        "detail": rest.detail,
        "channel": rest.channel.channel,
        "privileged": rest.channel.privileged,
        "criteria_source": rest.criteria.source,
        "start_ns": rest.start.time_ns,
        "end_ns": rest.end.time_ns,
        "window": [window[0].time_ns, window[-1].time_ns] if window else None,
        "terminal": list(window[-1].value) if window else None,
        "samples": rest.samples,
        "longest": [rest.longest[0].time_ns, rest.longest[-1].time_ns] if rest.longest else None,
    }


class Failed(Exception):
    """A stage failed; the skill ends with the stages so far."""


class Run:
    """One skill's progress: its stages, the stage in progress, and measured values."""

    def __init__(
        self,
        ctx: RobotContext,
        skill: str,
        obj: str,
        *,
        lookahead: Lookahead | None = None,
    ) -> None:
        self.ctx = ctx
        self.skill = skill
        self.obj = obj
        self.stage: str | None = None
        self.stages: list[Stage] = []
        self.values: dict[str, object] = {}
        self.held: bool | None = None
        self.transitions: list[TransitionEstimate] = []
        self.evaluations: list[TransitionEvaluation] = []
        self.lookahead = lookahead
        self._aheads: list[Ahead] = []
        self.upcoming: Ahead | None = None
        """The next skill's first motion, planned ahead and handed on if this one
        succeeds (``Follow``)."""

    def _log(self) -> list[dict[str, object]]:
        log = self.values.setdefault("lookahead", [])
        assert isinstance(log, list)
        return log

    def boundary(self, stage: str, reason: str) -> None:
        """Log that ``stage`` is not planned ahead, and why: its request depends on an
        outcome not yet known. Only when looking ahead."""
        if self.lookahead is not None:
            self._log().append({"stage": stage, "boundary": reason})

    def ahead(
        self,
        stage: str,
        source: Execution,
        request: Callable[[PredictedSnapshot], PlanningRequest],
        planner: Planner,
        *,
        allowances: Mapping[str, Sequence[str]] | None = None,
    ) -> Ahead | None:
        """Plan ``stage`` on ``ctx.predict(source)``, on a worker, while ``source`` runs.
        None if the skill does not look ahead, or, logged, if ``source`` is unpredictable.
        ``allowances`` are the attachments the skill will re-declare before ``stage``
        runs, and their allow sets (``ctx.predict``'s). ``planner`` must not be used for
        anything else until the plan is ready."""
        if self.lookahead is None:
            return None
        try:
            prediction = self.ctx.predict(
                source, unsettled=self.lookahead.unsettled, allowances=allowances
            )
            built = request(prediction)
        except SsrobotError as e:
            self._log().append({"stage": stage, "boundary": f"unpredictable: {e.code}"})
            return None
        log = lookahead_entry(stage, prediction)
        self._log().append(log)
        stop = threading.Event()
        future = self.ctx.plan_async(built, snapshot=prediction, planner=planner, abort=stop.is_set)
        ahead = Ahead(stage, prediction, planner, future, log, stop)
        self._aheads.append(ahead)
        return ahead

    def kept(
        self,
        stage: str,
        source: Execution,
        projection: PlanProjection,
        plan: PlanResult,
        planner: Planner,
    ) -> Ahead | None:
        """``plan``, made on ``projection`` before ``source`` was submitted, as the
        continuation of exactly that execution: ``ctx.continuation(plan, projection=,
        source=)``, with the lookahead's ``unsettled``. ``source`` may have run the
        projected plan itself, or continued it, rebased onto the actual state, to the same
        end. None if the skill does not look ahead, or, logged, if the binding is refused
        (``source`` does not run the projected plan, say because it was replanned, or
        rebased to another end); the stage is then planned again."""
        if self.lookahead is None:
            return None
        try:
            continuation = self.ctx.continuation(
                plan,
                planner=planner,
                projection=projection,
                source=source,
                unsettled=self.lookahead.unsettled,
            )
        except SsrobotError as e:
            self._log().append({"stage": stage, "boundary": f"kept plan refused: {e.code}"})
            return None
        prediction = continuation.prediction
        log = lookahead_entry(stage, prediction, kept="the feasibility check, planned before")
        self._log().append(log)
        future: Future[PlanResult] = Future()
        future.set_result(plan)
        ahead = Ahead(stage, prediction, planner, future, log, continuation=continuation)
        self._aheads.append(ahead)
        return ahead

    def adopt(self, ahead: Ahead) -> None:
        """Log a lookahead made before the skill began, such as by ``Candidates``."""
        if self.lookahead is None:
            self.lookahead = Lookahead()
        self._log().append(ahead.log)
        self._aheads.append(ahead)

    def close(self, succeeded: bool = False) -> None:
        """Stop every lookahead not used, and wait for it to stop; the skill has ended.
        If it ``succeeded``, the next skill's first motion (``upcoming``) is handed on
        instead, with a log entry of its own for the next skill."""
        if succeeded and self.upcoming is not None and "decision" not in self.upcoming.log:
            handed = self.upcoming
            handed.log["decision"] = "handed on: continued by the next skill"
            handed.log = {k: v for k, v in handed.log.items() if k != "decision"}
            self._aheads.remove(handed)
        else:
            self.upcoming = None
        for ahead in self._aheads:
            if "decision" not in ahead.log:
                ahead.cancel()
                wait([ahead.future])  # its planner is free again once this returns
                ahead.log["decision"] = "abandoned: the skill ended first"

    def attached(self) -> bool:
        return self.ctx.scene.attachment(self.obj) is not None

    def begin(self, name: str) -> None:
        """Name the stage now in progress, for a failure or cancellation inside it."""
        self.stage = name

    def record(
        self,
        name: str,
        succeeded: bool,
        detail: str,
        evidence: Evidence,
        reason: Reason | None = None,
        cause: str | None = None,
    ) -> None:
        """Record ``name`` as ended; a failure, with its ``reason`` (and ``cause``), ends
        the skill."""
        self.stage = name
        self._append(name, succeeded, detail, evidence, reason, cause)
        if not succeeded:
            raise Failed(f"{name}: {detail}")

    def fail(self, detail: str, cause: str) -> None:
        """Record the stage in progress as failed by an expected ssrobot refusal, ``cause``
        its code."""
        self._append(
            self.stage or "start", False, detail, Evidence.COMMANDED, Reason.REFUSED, cause
        )

    def cancelled(self, detail: str) -> None:
        """Record the stage in progress as cancelled."""
        self._append(
            self.stage or "start", False, detail, Evidence.COMMANDED, Reason.CANCELLED, None
        )

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
            transitions=tuple(self.transitions),
            evaluations=tuple(self.evaluations),
        )

    def _append(
        self,
        name: str,
        succeeded: bool,
        detail: str,
        evidence: Evidence,
        reason: Reason | None,
        cause: str | None,
    ) -> None:
        # A failure always says why, a success never does, and a cause qualifies a reason.
        assert (reason is None) == succeeded, (name, succeeded, reason)
        assert cause is None or reason is not None, (name, cause)
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
                reason=reason,
                cause=cause,
            )
        )

    def move(
        self,
        name: str,
        plan: PlanResult | Ahead,
        planner: Planner,
        *,
        replan: Callable[[], PlanResult] | None = None,
        then: Callable[[Execution], Ahead | None] | None = None,
        retain: Retention | None = None,
    ) -> Generator[Wait, None, Ahead | None]:
        """Execute ``plan`` and wait for it to finish.

        A ``PlanResult`` is executed with ``ctx.execute``; given ``replan``, one the robot
        no longer starts from (``start_mismatch``) is planned again. An ``Ahead`` (a plan made on a
        prediction) runs as a continuation once its source has finished: refusals in
        ``RECOVER`` fail the stage; any other refusal, or a failed plan on the prediction,
        is replanned with ``replan`` from the actual state, and recorded so.

        ``then``, called once this motion is submitted, may start planning the next
        stage ahead; its ``Ahead`` is returned.

        Given ``retain``, the motion carries the object: it runs under the opening guard,
        and its ``retained`` is estimated (``_retained``)."""
        self.begin(name)
        detail = "succeeded"
        guard = None if retain is None else retain.guard
        start = None if retain is None else self.privileged_pose()
        if isinstance(plan, Ahead):
            execution, detail = yield from self._continue(name, plan, planner, replan, guard)
        elif replan is not None and plan.succeeded:
            # A plan made beforehand: if the robot is no longer where it starts, plan again.
            try:
                execution = self.ctx.execute(plan, planner=planner, guard=guard)
            except SsrobotError as e:
                if e.code != "start_mismatch":
                    self.record(
                        name,
                        False,
                        f"{e.code}: {e.message}",
                        Evidence.PLANNED,
                        Reason.REFUSED,
                        e.code,
                    )
                execution, detail = self._replanned(
                    name,
                    replan,
                    planner,
                    "replanned: start_mismatch",
                    Reason.REFUSED,
                    e.code,
                    guard,
                )
        else:
            execution = self._execute(name, plan, planner, guard)
        upcoming = None if then is None else then(execution)
        yield Wait(execution=execution)
        self.finished(name, execution)
        if retain is not None:
            self._retained(name, execution, retain, start)
        self.record(name, True, detail, Evidence.COMMANDED)
        return upcoming

    def _retained(
        self, name: str, execution: Execution, retain: Retention, start: Observed | None
    ) -> None:
        """Estimate ``retained`` over the carry ``execution`` ran: confirmed if its opening
        guard did not trigger, refuted if it did, which fails ``name`` with the object
        reported not held, and unobserved without a guard. Where the runtime has the
        object's privileged pose channel, the attachment monitor's report is kept beside
        the estimate as a privileged reference, for evaluation only, with the object's
        pose at the carry's ``start`` and end."""
        status = execution.status
        evidence: tuple[Observed, ...] = ()
        if retain.guard is None:
            outcome, why = TransitionOutcome.UNOBSERVED, EstimateReason.NO_OPENING_MARGIN
        else:
            assert status.guard is not None  # a guarded trajectory's success carries one
            fell = status.guard.cause is GuardCause.GUARD_TRIGGERED
            outcome = TransitionOutcome.REFUTED if fell else TransitionOutcome.CONFIRMED
            why = EstimateReason.OPENING_FELL if fell else EstimateReason.OPENING_HELD
            last = status.guard.window[-1]
            evidence = (
                Observed(retain.guard.channel, status.stamp.time_ns, last, Evidence.PORTABLE),
            )
        estimate = TransitionEstimate(
            Transition.RETAINED, name, outcome, why, evidence, retain.criteria
        )
        self.transitions.append(estimate)
        end = self.privileged_pose()
        if end is not None and self.ctx.scene.attachment(self.obj) is not None:
            left = self.ctx.violation(self.obj) is not None
            reference = TransitionOutcome.REFUTED if left else TransitionOutcome.CONFIRMED
            monitor = Observed(
                MONITOR, self.ctx.now.time_ns, (float(not left),), Evidence.PRIVILEGED
            )
            poses = tuple(p for p in (start, end) if p is not None)
            self.evaluations.append(
                TransitionEvaluation(estimate, reference, MONITOR, (monitor, *poses))
            )
        if outcome is TransitionOutcome.REFUTED:
            assert retain.guard is not None
            self.held = False
            self.record(
                name,
                False,
                f"the opening fell below {retain.guard.below:.3f}: the object left the hand",
                Evidence.PORTABLE,
                Reason.TRANSITION_REFUTED,
                Transition.RETAINED.value,
            )

    def _execute(
        self, name: str, plan: PlanResult, planner: Planner, guard: OpeningGuard | None = None
    ) -> Execution:
        if not plan.succeeded:
            reason = "; ".join(d.message for d in plan.diagnostics) or plan.status.value
            self.record(
                name,
                False,
                f"planning {plan.status.value}: {reason}",
                Evidence.PLANNED,
                Reason.NOT_PLANNED,
                plan.status.value,
            )
        try:
            return self.ctx.execute(plan, planner=planner, guard=guard)
        except SsrobotError as e:
            self.record(
                name, False, f"{e.code}: {e.message}", Evidence.PLANNED, Reason.REFUSED, e.code
            )
        raise AssertionError  # unreachable: record raised

    def _continue(
        self,
        name: str,
        ahead: Ahead,
        planner: Planner,
        replan: Callable[[], PlanResult] | None,
        guard: OpeningGuard | None,
    ) -> Generator[Wait, None, tuple[Execution, str]]:
        ctx, log = self.ctx, ahead.log
        if not ahead.future.done():
            yield Wait(future=ahead.future)
        plan = ahead.future.result()
        if not plan.succeeded:
            log["decision"] = f"replanned: planning on the prediction {plan.status.value}"
            return self._replanned(
                name,
                replan,
                planner,
                log["decision"],
                Reason.NOT_PLANNED,
                plan.status.value,
                guard,
            )
        continuation = ahead.continuation or ctx.continuation(
            plan, planner=ahead.planner, prediction=ahead.prediction
        )
        look = self.lookahead or Lookahead()
        since = ctx.now.time_ns
        rest = yield from self.rest(plan.request.group, look.rest_timeout_ns)
        result = ctx.execute_continuation(
            continuation, rest=rest, arrival_tolerance=look.arrival_tolerance, guard=guard
        )
        report = result.report
        log.update(
            refusal=report.refusal,
            detail=report.detail,
            max_start_gap=report.max_start_gap,
            rest=rest_evidence(rest),
            rebase=report.rebase,
            waited_for_rest_ns=ctx.now.time_ns - since,
        )
        if result.execution is not None:
            log["decision"] = "continued"
            return result.execution, "succeeded (planned ahead, continued)"
        if report.refusal in RECOVER:
            log["decision"] = "failed: for recovery"
            evidence = (
                Evidence.PRIVILEGED
                if any(d.what.startswith("object ") for d in report.discrepancies)
                else Evidence.PLANNED
            )
            refused = f"continuation refused, {report.refusal}: {report.detail}"
            self.record(name, False, refused, evidence, Reason.REFUSED, report.refusal)
        log["decision"] = f"replanned: {report.refusal}"
        return self._replanned(
            name, replan, planner, log["decision"], Reason.REFUSED, report.refusal, guard
        )

    def rest(self, component: str, timeout_ns: int) -> Generator[Wait, None, RestResult]:
        """Watch ``component``'s joints until they are observed at rest, by the criteria
        the runtime declares for their channel, or ``timeout_ns`` passes; the result."""
        watch = self.ctx.watch_rest(
            JointRest(component=component), deadline_ns=self.ctx.now.time_ns + timeout_ns
        )
        yield Wait(watch=watch)
        assert watch.result is not None  # a watch is done once it has a result
        return watch.result

    def _replanned(
        self,
        name: str,
        replan: Callable[[], PlanResult] | None,
        planner: Planner,
        why: object,
        reason: Reason,
        cause: str | None,
        guard: OpeningGuard | None = None,
    ) -> tuple[Execution, str]:
        """Plan ``name`` again from the actual state, with ``replan``, and execute it.
        Without ``replan``, ``name`` fails with the ``reason`` and ``cause`` that made it
        replan; ``why`` is their prose."""
        if replan is None:
            self.record(
                name, False, f"{why}, and nothing to replan with", Evidence.PLANNED, reason, cause
            )
        assert replan is not None
        plan = replan()
        self.begin(name)
        return self._execute(name, plan, planner, guard), f"succeeded ({why})"

    def finished(self, name: str, execution: Execution) -> None:
        """Fail ``name`` unless ``execution`` succeeded."""
        status = execution.status
        if status.state is not ExecutionState.SUCCEEDED:
            detail = status.state.value
            if status.diagnostic is not None:
                detail += f": {status.diagnostic.code}"
            canceled = status.state is ExecutionState.CANCELED
            self.record(
                name,
                False,
                detail,
                Evidence.COMMANDED,
                Reason.COMMAND_CANCELED if canceled else Reason.COMMAND_FAILED,
                status.state.value if status.diagnostic is None else status.diagnostic.code,
            )

    def grip(self, name: str, gripper: str, opening: float, timeout_ns: int) -> Body:
        """Command ``gripper`` to ``opening`` and wait for the command to end, which it
        does when the fingers' motion ends (``GripperResult``); require it to succeed by
        ``timeout_ns``. The result is kept in ``values["gripper"][name]``."""
        self.begin(name)
        command = GripperCommand(gripper=gripper, opening=opening)
        execution = self.ctx.submit(command, timeout_ns=timeout_ns)
        yield Wait(execution=execution)
        self.finished(name, execution)
        result = execution.status.gripper
        assert result is not None  # a gripper command's success always carries one
        results: dict[str, Any] = self.values.setdefault("gripper", {})  # type: ignore[assignment]
        results[name] = json.loads(dumps(result))

    def observe(self, channel: str) -> Any:
        return self.ctx.observe(ObservationRequest(channels=(channel,))).readings[0].value

    def privileged_pose(self) -> Observed | None:
        """The object's pose from its privileged pose channel, for evaluation only; None
        where the runtime declares none."""
        channel = next(
            (
                c.name
                for c in self.ctx.info.scene_channels
                if c.source == self.obj and c.quantity is Quantity.POSE and c.privileged
            ),
            None,
        )
        if channel is None:
            return None
        observation = self.ctx.observe(ObservationRequest(channels=(channel,)))
        reading = observation.readings[0].value
        assert isinstance(reading, tuple)  # a pose is seven floats
        value = tuple(float(v) for v in reading)
        return Observed(channel, observation.stamp.time_ns, value, Evidence.PRIVILEGED)

    def unobserved(self, transition: Transition, stage: str) -> TransitionEstimate:
        """Record ``transition`` as estimated in ``stage`` with no portable signal that can
        tell: ``unobserved``, for the application's policy to assume or not."""
        estimate = TransitionEstimate(
            transition,
            stage,
            TransitionOutcome.UNOBSERVED,
            EstimateReason.NO_PORTABLE_SIGNAL,
            (),
            (),
        )
        self.transitions.append(estimate)
        return estimate

    def evaluate(self, estimate: TransitionEstimate, judge: Callable[[Pose], bool]) -> None:
        """Score ``estimate`` against the object's privileged pose, as ``judge`` reads it,
        where the runtime declares the pose channel: a ``TransitionEvaluation`` for
        evaluation only. Nothing is returned, so no stage can branch on it."""
        observed = self.privileged_pose()
        if observed is None:
            return
        x, y, z, w, i, j, k = observed.value
        pose = Pose(position=(x, y, z), quat_wxyz=(w, i, j, k))
        reference = TransitionOutcome.CONFIRMED if judge(pose) else TransitionOutcome.REFUTED
        self.evaluations.append(
            TransitionEvaluation(estimate, reference, observed.channel, (observed,))
        )

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
            self.record(
                name, False, f"{self.obj!r} is not attached", Evidence.PLANNED, Reason.NOT_ATTACHED
            )
        assert attachment is not None
        if not attachment.held:
            self.record(
                name,
                False,
                "attachment monitor: the object left its declared transform",
                Evidence.PRIVILEGED,
                Reason.ATTACHMENT_LOST,
            )
        known = {f.name for f in self.ctx.description.frames} | set(self.ctx.info.fixtures)
        unknown = sorted(set(allow) - known)
        if unknown:
            self.record(
                name,
                False,
                f"neither robot frames nor fixtures: {', '.join(unknown)}",
                Evidence.PLANNED,
                Reason.UNAVAILABLE,
            )
        self.ctx.detach(self.obj)
        self.ctx.attach(self.obj, end_effector, transform=attachment.transform, allow=tuple(allow))

    def choose(
        self,
        name: str,
        then: str | None,
        plan: Callable[[SceneSnapshot, int], PlanResult],
        follow: Callable[[PlanProjection], PlanResult] | None,
        attempts: int,
        *,
        planner: Planner,
    ) -> tuple[PlanResult, PlanResult | None, PlanProjection | None]:
        """A plan for ``name`` from which the straight motion ``then`` also plans; with
        the follow-on plan and the projection it was planned on.

        Each seed's plan is prepared for the runtime first (``ctx.prepare``, with
        ``planner``, the one that executes it): a plan whose command is not ready
        (``execution_unready``) is rejected, before anything executes, and the next seed
        tried. The follow-on plan is not prepared here: it is rebased into another motion,
        which its continuation prepares. A goal region is checked only for collision, so
        a plan can reach it at a configuration from which the next straight move is
        infeasible. Each seed's plan is kept only if ``follow`` plans on its projection
        (``ctx.project``): where it would leave the robot. That follow-on plan is
        returned, so a skill may continue it (``Run.kept``) once the plan runs, instead of
        planning it again. The seeds tried are recorded in
        ``values[name + "_attempts"]``."""
        self.begin(name)
        snapshot = self.ctx.snapshot()
        tried: list[dict[str, object]] = []
        self.values[f"{name}_attempts"] = tried
        for seed in range(attempts):
            result = plan(snapshot, seed)
            if not result.succeeded:
                tried.append({"seed": seed, name: result.status.value})
                continue
            try:
                self.ctx.prepare(result, planner=planner)
            except ValidationError as e:
                if e.code != "execution_unready":
                    raise
                tried.append({"seed": seed, name: "unready", "detail": e.message})
                continue
            if follow is None or then is None:  # nothing follows: the first plan will do
                tried.append({"seed": seed, name: "succeeded"})
                return result, None, None
            projection = self.ctx.project(result, snapshot=snapshot)
            check = follow(projection)
            tried.append({"seed": seed, name: "succeeded", then: check.status.value})
            if check.succeeded:
                return result, check, projection
        detail = f"no plan in {attempts} seeds" + (
            "" if then is None else f" from which {then} plans"
        )
        self.record(name, False, detail, Evidence.PLANNED, Reason.NOT_PLANNED)
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
    carries: str | None = None,
) -> None:
    """Record ``preflight``: every reference resolves, and every command, channel, and
    guard the skill will use is declared available, before anything is commanded. A skill
    that carries the object names the opening channel its carries are guarded on
    (``carries``)."""
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
    if carries is not None and arm is not None:
        if not any(g.component == arm.group and carries in g.channels for g in info.guards):
            problems.append(f"no guard on {arm.group!r} reads {carries!r}")
    run.record(
        "preflight",
        not problems,
        "; ".join(problems)
        or (
            "references, commands, and channels"
            if guard is None and carries is None
            else "references, commands, channels, and guards"
        )
        + " available",
        Evidence.PLANNED,
        Reason.UNAVAILABLE if problems else None,
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
