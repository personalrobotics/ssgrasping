"""Race a grasp across candidates, such as one per arm, before anything moves.

``Candidates`` is passive, like a ``SkillExecution``: advance it, or drive it with
``run_together``. Each candidate is a grasp spec, the planners for it, and optionally the
drop it leads to. Every candidate is evaluated on its complete planned-only feasibility
prefix: every motion that can be planned before the first outcome not yet known (whether
the fingers hold the object), with the follow-on checks the skills make:

1. ``pregrasp``: into the pregrasp regions, kept only if the straight ``approach`` from
   where it ends, and the ``lift`` from where that ends, holding the object as the grasp
   would (below) with the support allowed, also plan; up to the spec's
   ``pregrasp_attempts`` seeds.
2. With a drop, ``transfer``: from the lifted configuration, holding the object as the
   grasp would (at its pose relative to the hand at the end of the approach, allowed to
   touch the hand), into the drop regions at the drop's ``carry_scaling``, kept only if
   the ``retreat`` also plans from where it ends; up to the drop's ``transfer_attempts``
   seeds. A candidate whose container is out of reach fails here, however short its
   approach.

All candidates plan at once, each on its own worker, with its own planners. A
candidate's cost is the total duration of its prefix's plans. The selection is by
priority, then the lowest cost, then declaration order. Once a candidate succeeds, every
candidate of strictly lower priority is stopped (``canceled_by_priority``); equal
priorities finish. Nothing is commanded.

Given ``after``, a motion executing now, the prefix is planned on ``ctx.predict(after)``
(``unsettled`` names objects it makes no claim about, such as one just dropped): the
selected pregrasp is then an ``Ahead``, which ``grasp`` continues once ``after`` ends.
Otherwise it is planned on a snapshot captured now.

The report records the basis (and the prediction's lineage), every candidate's outcome
with every plan it made, and the selection; it is deterministic for deterministic
planners. When each candidate planned, in wall time, is in ``intervals_s``, outside it.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Generator, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from typing import Any

from ssrobot import (
    Attachment,
    Execution,
    OffsetGoal,
    Planner,
    PlanningRequest,
    PlanResult,
    PredictedSnapshot,
    RobotContext,
    SceneSnapshot,
    SsrobotError,
    TsrGoal,
)

from ssgrasping._poses import compose, invert
from ssgrasping._skill import Ahead, Wait, _at, _end, lookahead_entry
from ssgrasping.drop import DropSpec
from ssgrasping.grasp import GraspSpec, _approach_goal

RULE = "priority, then the lowest total duration of the prefix's plans, then declaration order"


@dataclass(frozen=True, slots=True)
class Candidate:
    """One way to grasp, and the drop it leads to."""

    label: str
    grasp: GraspSpec
    planner: Planner
    checker: Planner
    """Plans the follow-on checks (approach, lift, retreat), as in ``grasp``."""
    drop: DropSpec | None = None
    priority: int = 0


@dataclass(frozen=True, slots=True)
class Selection:
    """The selected candidate, and its pregrasp: a plan on a captured snapshot, or an
    ``Ahead`` on the prediction, for ``grasp(..., pregrasp=...)``."""

    candidate: Candidate
    pregrasp: PlanResult | Ahead


@dataclass(slots=True)
class _Race:
    candidate: Candidate
    body: Generator[Wait, None, None] | None = None
    wait: Wait | None = None
    done: bool = False
    stop: threading.Event = field(default_factory=threading.Event)
    record: dict[str, Any] = field(default_factory=dict)
    plans: list[dict[str, Any]] = field(default_factory=list)
    pregrasp: PlanResult | None = None
    cost: float = 0.0
    started: float = 0.0
    ended: float = 0.0


class Candidates:
    """A race of grasp candidates on their feasibility prefixes; see the module."""

    def __init__(
        self,
        ctx: RobotContext,
        candidates: Sequence[Candidate],
        *,
        after: Execution | None = None,
        unsettled: Sequence[str] = (),
    ) -> None:
        if not candidates:
            raise ValueError("at least one candidate")
        planners = [p for c in candidates for p in (c.planner, c.checker)]
        if len({id(p) for p in planners}) != len(planners):
            raise ValueError("each candidate plans on its own worker: give each its own planners")
        self._ctx = ctx
        self._prediction: PredictedSnapshot | None = None
        if after is not None and not after.done:
            self._prediction = ctx.predict(after, unsettled=unsettled)
            self._basis: SceneSnapshot | PredictedSnapshot = self._prediction
        else:
            self._basis = ctx.snapshot()
        self._t0 = time.monotonic()
        self._races = [_Race(c) for c in candidates]
        for race in self._races:
            race.body = self._prefix(race)
            race.record = {"label": race.candidate.label, "priority": race.candidate.priority}
            race.started = time.monotonic()
        self._selection: Selection | None = None
        self._report: dict[str, Any] | None = None

    @property
    def done(self) -> bool:
        return self._report is not None

    @property
    def planning(self) -> tuple[Future[Any], ...]:
        """The plans being made now; a race never needs the runtime's time to pass."""
        return tuple(
            r.wait.future
            for r in self._races
            if not r.done and r.wait is not None and r.wait.future is not None
        )

    @property
    def selection(self) -> Selection | None:
        return self._selection

    @property
    def report(self) -> dict[str, Any] | None:
        return self._report

    @property
    def intervals_s(self) -> list[tuple[str, float, float]]:
        """When each candidate planned, in seconds of wall time since the race began."""
        return [(r.candidate.label, r.started - self._t0, r.ended - self._t0) for r in self._races]

    def advance(self) -> bool:
        """Resume every candidate whose plan is ready; select once all have ended."""
        if self.done:
            return True
        for race in self._races:
            while not race.done and (race.wait is None or race.wait.satisfied(self._ctx)):
                assert race.body is not None
                try:
                    race.wait = next(race.body)
                except StopIteration:
                    race.done, race.ended = True, time.monotonic()
                    if race.record.get("outcome") == "succeeded":
                        self._outrank(race.candidate.priority)
        if all(r.done for r in self._races):
            self._select()
        return self.done

    def _outrank(self, priority: int) -> None:
        for race in self._races:
            if race.candidate.priority < priority and not race.done:
                race.stop.set()

    def _plan(
        self,
        race: _Race,
        stage: str,
        seed: int,
        request: PlanningRequest,
        planner: Planner,
        on: Any,
    ) -> Generator[Wait, None, PlanResult | None]:
        """Plan ``request`` on a worker; None once the race stopped this candidate."""
        if race.stop.is_set():
            return None
        future = self._ctx.plan_async(request, snapshot=on, planner=planner, abort=race.stop.is_set)
        yield Wait(future=future)
        result = future.result()
        if race.stop.is_set():
            return None
        duration = None if result.trajectory is None else result.trajectory.duration
        race.plans.append(
            {"stage": stage, "seed": seed, "status": result.status.value, "duration_s": duration}
        )
        if result.succeeded:
            race.cost += duration or 0.0
        return result

    def _prefix(self, race: _Race) -> Generator[Wait, None, None]:
        try:
            yield from self._evaluate(race)
        except SsrobotError as e:  # a refused request: this candidate is infeasible
            race.record.update(outcome="failed", detail=f"{e.code}: {e.message}")
        if race.stop.is_set() and race.record.get("outcome") != "succeeded":
            race.record.update(outcome="canceled_by_priority", detail="a higher priority succeeded")
        race.record["plans"] = race.plans
        if race.record.get("outcome") == "succeeded":
            race.record["cost_s"] = race.cost

    def _evaluate(self, race: _Race) -> Generator[Wait, None, None]:
        ctx, c, g = self._ctx, race.candidate, race.candidate.grasp
        arm = ctx.manipulator(g.manipulator)
        basis = self._basis
        state = basis.state if isinstance(basis, PredictedSnapshot) else basis
        for seed in range(g.pregrasp_attempts):
            race.plans.clear()
            race.cost = 0.0
            pregrasp = yield from self._plan(
                race,
                "pregrasp",
                seed,
                arm.request(TsrGoal(tsrs=tuple(g.pregrasps)), snapshot=basis, seed=seed),
                c.planner,
                basis,
            )
            if pregrasp is None:
                return
            if not pregrasp.succeeded:
                continue
            at = _at(state, *_end(pregrasp))
            approach = yield from self._plan(
                race,
                "approach",
                seed,
                arm.request(_approach_goal(arm, ctx, at, g), snapshot=at),
                c.checker,
                at,
            )
            if approach is None:
                return
            if not approach.succeeded:
                continue
            grasped = _at(state, *_end(approach))
            gripping = self._holding(arm, g, grasped, grasped, (g.support,))
            lift = yield from self._plan(
                race,
                "lift",
                seed,
                arm.request(
                    OffsetGoal(
                        direction=g.normal,
                        distance=g.lift_m,
                        position_tolerance=g.line_tolerance_m,
                        rotation_tolerance=1e-2,
                    ),
                    snapshot=gripping,
                ),
                c.checker,
                gripping,
            )
            if lift is None:
                return
            if lift.succeeded:
                race.pregrasp = pregrasp
                break
        else:
            race.record.update(
                outcome="failed",
                detail=f"no pregrasp in {g.pregrasp_attempts} seeds whose approach and lift plan",
            )
            return
        if c.drop is not None:
            ok = yield from self._transfer(race, arm, c.drop, grasped, _at(state, *_end(lift)))
            if not ok:
                return
        race.record.update(outcome="succeeded", detail="every motion of the prefix plans")

    def _transfer(
        self,
        race: _Race,
        arm: Any,
        drop: DropSpec,
        grasped: SceneSnapshot,
        lifted: SceneSnapshot,
    ) -> Generator[Wait, None, bool]:
        """Plan the drop's transfer, holding the object as the grasp would, and its
        retreat check."""
        holding = self._holding(arm, race.candidate.grasp, grasped, lifted)
        c = race.candidate
        for seed in range(drop.transfer_attempts):
            transfer = yield from self._plan(
                race,
                "transfer",
                seed,
                arm.request(
                    TsrGoal(tsrs=tuple(drop.regions)),
                    snapshot=holding,
                    seed=seed,
                    velocity_scaling=drop.carry_scaling,
                    acceleration_scaling=drop.carry_scaling,
                ),
                c.planner,
                holding,
            )
            if transfer is None:
                return False
            if not transfer.succeeded:
                continue
            over = _at(holding, *_end(transfer))
            retreat = yield from self._plan(
                race,
                "retreat",
                seed,
                arm.request(
                    OffsetGoal(
                        direction=drop.normal,
                        distance=drop.retreat_m,
                        position_tolerance=drop.line_tolerance_m,
                        rotation_tolerance=1e-2,
                    ),
                    snapshot=over,
                ),
                c.checker,
                over,
            )
            if retreat is None:
                return False
            if retreat.succeeded:
                return True
        race.record.update(
            outcome="failed",
            detail=f"no transfer to {drop.container!r} in {drop.transfer_attempts} seeds "
            "from which the retreat plans",
        )
        return False

    def _holding(
        self,
        arm: Any,
        g: GraspSpec,
        grasped: SceneSnapshot,
        lifted: SceneSnapshot,
        also: Sequence[str] = (),
    ) -> SceneSnapshot:
        """``lifted`` with the object held as the grasp would hold it: at its pose relative
        to the hand where the approach ends, allowed to touch the hand's frames (ssrobot's
        default allowance) and ``also``, as the grasp allows its support for the lift."""
        ctx = self._ctx
        frame = ctx.description.end_effector(g.end_effector).frame
        joints = ctx.description.group(arm.group).joints

        def hand(snapshot: SceneSnapshot) -> Any:
            q = tuple(snapshot.position(j) for j in joints)
            return ctx.planning_scene(snapshot, arm.group).forward_kinematics(q, frame)

        pose = next(o.pose for o in grasped.objects if o.name == g.object)
        transform = compose(invert(hand(grasped)), pose)
        held = Attachment(
            object=g.object,
            end_effector=g.end_effector,
            transform=transform,
            allow=tuple(sorted({*_hand_frames(ctx, g.end_effector), *also})),
        )
        carried = compose(hand(lifted), transform)
        objects = tuple(
            replace(o, pose=carried) if o.name == g.object else o for o in lifted.objects
        )
        attachments = tuple(sorted((*lifted.attachments, held), key=lambda a: a.object))
        return replace(lifted, objects=objects, attachments=attachments)

    def _select(self) -> None:
        succeeded = [
            (i, r) for i, r in enumerate(self._races) if r.record.get("outcome") == "succeeded"
        ]
        chosen = min(
            succeeded, key=lambda ir: (-ir[1].candidate.priority, ir[1].cost, ir[0]), default=None
        )
        basis: dict[str, Any]
        if self._prediction is not None:
            basis = {"on": "prediction", **lookahead_entry("pregrasp", self._prediction)}
        else:
            assert isinstance(self._basis, SceneSnapshot)
            basis = {"on": "snapshot", "snapshot": self._basis.fingerprint()}
        self._report = {
            "basis": basis,
            "rule": RULE,
            "candidates": [r.record for r in self._races],
            "selected": None if chosen is None else chosen[1].candidate.label,
        }
        if chosen is None:
            return
        race = chosen[1]
        assert race.pregrasp is not None
        pregrasp: PlanResult | Ahead = race.pregrasp
        if self._prediction is not None:
            ready: Future[PlanResult] = Future()
            ready.set_result(race.pregrasp)
            pregrasp = Ahead(
                stage="pregrasp",
                prediction=self._prediction,
                planner=race.candidate.planner,
                future=ready,
                log=lookahead_entry(
                    "pregrasp", self._prediction, raced=[r.candidate.label for r in self._races]
                ),
            )
        self._selection = Selection(candidate=race.candidate, pregrasp=pregrasp)


def _hand_frames(ctx: RobotContext, end_effector: str) -> tuple[str, ...]:
    """The frames ssrobot allows a held object to touch by default: every frame at or
    below the end effector's gripper frame (or its own frame, without a gripper)."""
    d = ctx.description
    effector = d.end_effector(end_effector)
    root = effector.frame if effector.gripper is None else d.gripper(effector.gripper).frame
    parents = {f.name: f.parent for f in d.frames}

    def below(frame: str | None) -> bool:
        while frame is not None and frame != root:
            frame = parents[frame]
        return frame == root

    return tuple(sorted(f for f in parents if below(f)))
