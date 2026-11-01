"""Race a grasp across candidates, such as one per arm, before anything moves.

``Candidates`` is passive, like a ``SkillExecution``: advance it, or drive it with
``run_together``. Each candidate is a grasp spec, the planners for it, and optionally the
drop it leads to. Every candidate is evaluated on its complete planned-only feasibility
prefix: every motion that can be planned before the first outcome not yet known (whether
the fingers hold the object), as one ssrobot ``Chain`` of stages, each planned on where
the stage before it would leave the robot:

1. ``pregrasp``: into the pregrasp regions, one attempt per seed, up to the spec's
   ``pregrasp_attempts``;
2. ``approach``: the straight approach from the pregrasp's projection by ``approach_m``,
   the depth ``grasp`` guarantees, as its own pregrasp check plans it, with the check
   planner. With ``deepest``, the selected grasp then approaches as deep as plans, down
   to that;
3. ``lift``: from the approach's projection, holding the object as the grasp would (at
   its pose relative to the hand there, allowed to touch the hand and the support), with
   the check planner;
4. with a drop, ``transfer``: from the lift's projection, holding the object allowed to
   touch only the hand, into the drop regions at the drop's ``carry_scaling``, up to the
   drop's ``transfer_attempts`` seeds from the same lift; and, when the drop retreats,
   ``retreat`` from where it ends, with the check planner. A candidate whose container is
   out of reach fails here, however short its approach.

The pregrasp's seeds are attempts at the whole prefix, so a pregrasp whose approach or
lift does not plan is replaced by the next. Once every transfer seed has failed, though,
the candidate fails: no other pregrasp is tried (``Stage.retry_prefix=False``), which
bounds the work to one prefix and ``transfer_attempts`` transfers. That is a search
policy, not a proof: another pregrasp, or another IK branch for it, could in principle
make the transfer feasible.

ssrobot plans the stages (``RobotContext.start_chains``), validates each on a fresh scene
before the next is built, and selects: each candidate's first seed whose stages are all
valid, costing the sum of their durations; then priority, then cost, then declaration
order. A candidate's success stops every candidate of strictly lower priority. Nothing is
commanded.

Given ``after``, a motion executing now, the prefix is planned on ``ctx.predict(after)``
(``unsettled`` names objects it makes no claim about, such as one just dropped;
``releases``, held objects to be released when ``after`` ends, such as one dropped from
where a transfer ends, each at the pose it is expected to rest at; ``joints``, other
joints as they are expected then, such as the gripper opened to release it): the
selected pregrasp then continues once ``after`` ends. Otherwise it is planned on a
snapshot captured now.

The selection runs its candidate's grasp (``Selection.grasp``): its raced pregrasp, and
the approach it was checked with, kept for the grasp to continue once the pregrasp has
run; it was planned before the pregrasp ran, not while it runs. The report is ssrobot's
``ChainsReport``, with the basis and a summary per candidate (``canceled`` when the race
was stopped, ``canceled_by_priority`` when a higher priority succeeded); it is
deterministic for deterministic planners.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from typing import Any

from ssrobot import (
    Attachment,
    Chain,
    ChainsRun,
    Execution,
    OffsetGoal,
    Planner,
    PlanResult,
    Pose,
    PredictedSnapshot,
    RobotContext,
    SceneSnapshot,
    Stage,
    StageRequest,
    StageResult,
    TsrGoal,
    dumps,
)

from ssgrasping._poses import compose, invert
from ssgrasping._skill import Ahead, Follow, Kept, Lookahead, lookahead_entry
from ssgrasping.drop import DropSpec
from ssgrasping.execution import SkillExecution
from ssgrasping.grasp import GraspSpec, _approach_goal, _selected_grasp


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
    """The selected candidate, and the prefix it was selected on, inseparable from it:
    ``grasp`` runs that candidate's grasp, with its planners, from its raced pregrasp and
    the approach it was checked with. ``pregrasp`` is the raced pregrasp plan, for
    inspection; it runs only through ``grasp``."""

    candidate: Candidate
    pregrasp: PlanResult
    _start: PlanResult | Ahead = field(repr=False)
    """The pregrasp as it runs: the plan, or, raced on a prediction, its ``Ahead``."""
    _approach: Kept = field(repr=False)

    def grasp(
        self,
        ctx: RobotContext,
        *,
        lookahead: Lookahead | None = None,
        follow: Follow | None = None,
    ) -> SkillExecution:
        """The selected candidate's grasp: ``ssgrasping.grasp`` of its spec, with its
        planner and check planner, executing the raced pregrasp in place of choosing
        one, and, with ``lookahead``, continuing the approach it was checked with, and
        planning ``follow`` during the lift (``ssgrasping.grasp``)."""
        c = self.candidate
        return _selected_grasp(
            ctx, c.planner, c.checker, c.grasp, lookahead, self._start, self._approach, follow
        )


class Candidates:
    """A race of grasp candidates on their feasibility prefixes; see the module."""

    def __init__(
        self,
        ctx: RobotContext,
        candidates: Sequence[Candidate],
        *,
        after: Execution | None = None,
        unsettled: Sequence[str] = (),
        releases: Mapping[str, Pose] | None = None,
        joints: Mapping[str, float] | None = None,
    ) -> None:
        if not candidates:
            raise ValueError("at least one candidate")
        self._ctx = ctx
        self._candidates = tuple(candidates)
        self._prediction: PredictedSnapshot | None = None
        if after is not None and not after.done:
            self._prediction = ctx.predict(
                after, unsettled=unsettled, releases=releases, joints=joints
            )
            self._basis: SceneSnapshot | PredictedSnapshot = self._prediction
        else:
            self._basis = ctx.snapshot()
        chains = [self._chain(c) for c in self._candidates]
        # One stage of one seed of each chain plans at a time: a worker each.
        self._run: ChainsRun = ctx.start_chains(chains, snapshot=self._basis, workers=len(chains))
        self._selection: Selection | None = None
        self._report: dict[str, Any] | None = None

    @property
    def done(self) -> bool:
        return self._report is not None

    @property
    def planning(self) -> tuple[Future[Any], ...]:
        """The plans being made now; a race never needs the runtime's time to pass."""
        return self._run.planning

    @property
    def selection(self) -> Selection | None:
        return self._selection

    @property
    def report(self) -> dict[str, Any] | None:
        return self._report

    def advance(self) -> bool:
        """Advance ssrobot's race; select once it is done."""
        if self.done:
            return True
        if self._run.advance():
            self._select()
        return self.done

    def close(self) -> None:
        """Stop the race if it has not finished, and wait for its planners; may be called
        again."""
        self._run.close()
        if not self.done:
            self._select()

    def _chain(self, c: Candidate) -> Chain:
        """The candidate's prefix as ssrobot stages; see the module."""
        ctx, g = self._ctx, c.grasp
        arm = ctx.manipulator(g.manipulator)
        basis = self._basis

        def pregrasp(previous: StageResult | None) -> StageRequest:
            request = arm.request(
                TsrGoal(tsrs=tuple(g.pregrasps)), snapshot=basis, clearance_m=g.clearance_m
            )
            return StageRequest(snapshot=basis, request=request, planner=c.planner)

        def approach(previous: StageResult | None) -> StageRequest:
            assert previous is not None
            at = previous.projection
            request = arm.request(
                _approach_goal(arm, ctx, at, g),
                snapshot=at,
                seed=previous.plan.request.seed,
            )
            return StageRequest(snapshot=at, request=request, planner=c.checker)

        def lift(previous: StageResult | None) -> StageRequest:
            assert previous is not None
            at = previous.projection
            held = self._held(arm, g, at.state, also=(g.support,))
            attachments = tuple(sorted((*at.state.attachments, held), key=lambda a: a.object))
            state = replace(at.state, attachments=attachments)
            request = arm.request(
                OffsetGoal(
                    direction=g.normal,
                    distance=g.lift_m,
                    position_tolerance=g.line_tolerance_m,
                    rotation_tolerance=1e-2,
                ),
                snapshot=state,
                seed=previous.plan.request.seed,
            )
            return StageRequest(
                snapshot=at, request=request, planner=c.checker, attachments=attachments
            )

        stages = [Stage("pregrasp", pregrasp), Stage("approach", approach), Stage("lift", lift)]
        drop = c.drop
        if drop is not None:

            def transfer(previous: StageResult | None) -> StageRequest:
                assert previous is not None and drop is not None
                at = previous.projection
                hand = _hand_frames(ctx, g.end_effector)
                attachments = tuple(
                    replace(a, allow=hand) if a.object == g.object else a
                    for a in at.state.attachments
                )
                state = replace(at.state, attachments=attachments)
                request = arm.request(
                    TsrGoal(tsrs=tuple(drop.regions)),
                    snapshot=state,
                    seed=previous.plan.request.seed,
                    velocity_scaling=drop.carry_scaling,
                    acceleration_scaling=drop.carry_scaling,
                    clearance_m=drop.clearance_m,
                )
                return StageRequest(
                    snapshot=at, request=request, planner=c.planner, attachments=attachments
                )

            # Retried from the same lift, as ``drop`` retries it; and, once every seed
            # fails, the candidate fails rather than planning another prefix: see the
            # module.
            stages.append(
                Stage(
                    "transfer",
                    transfer,
                    seeds=tuple(range(drop.transfer_attempts)),
                    retry_prefix=False,
                )
            )
            if drop.retreat_m > 0:

                def retreat(previous: StageResult | None) -> StageRequest:
                    assert previous is not None and drop is not None
                    at = previous.projection
                    request = arm.request(
                        OffsetGoal(
                            direction=drop.normal,
                            distance=drop.retreat_m,
                            position_tolerance=drop.line_tolerance_m,
                            rotation_tolerance=1e-2,
                        ),
                        snapshot=at,
                        seed=previous.plan.request.seed,
                    )
                    return StageRequest(snapshot=at, request=request, planner=c.checker)

                stages.append(Stage("retreat", retreat))
        return Chain(
            label=c.label,
            stages=tuple(stages),
            seeds=tuple(range(g.pregrasp_attempts)),
            priority=c.priority,
        )

    def _held(
        self, arm: Any, g: GraspSpec, at: SceneSnapshot, also: Sequence[str] = ()
    ) -> Attachment:
        """The object held as the grasp would hold it from ``at``: at its pose relative to
        the hand there, allowed to touch the hand's frames (ssrobot's default allowance)
        and ``also``, as the grasp allows its support for the lift."""
        ctx = self._ctx
        frame = ctx.description.end_effector(g.end_effector).frame
        q = tuple(at.position(j) for j in ctx.description.group(arm.group).joints)
        hand = ctx.planning_scene(at, arm.group).forward_kinematics(q, frame)
        pose = next(o.pose for o in at.objects if o.name == g.object)
        return Attachment(
            object=g.object,
            end_effector=g.end_effector,
            transform=compose(invert(hand), pose),
            allow=tuple(sorted({*_hand_frames(ctx, g.end_effector), *also})),
        )

    def _select(self) -> None:
        result = self._run.result
        report = json.loads(dumps(result.report))
        basis: dict[str, Any]
        if self._prediction is not None:
            basis = {"on": "prediction", **lookahead_entry("pregrasp", self._prediction)}
        else:
            assert isinstance(self._basis, SceneSnapshot)
            basis = {"on": "snapshot", "snapshot": self._basis.fingerprint()}
        selected = result.selected
        self._report = {
            "basis": basis,
            "rule": report["rule"],
            "candidates": [_summary(c, report) for c in report["chains"]],
            "selected": None if selected is None else selected[0].label,
            "chains": report,
        }
        if selected is None:
            return
        chain, _, stages = selected
        candidate = self._candidates[[c.label for c in self._candidates].index(chain.label)]
        pregrasp: PlanResult | Ahead = stages[0].plan
        if self._prediction is not None:
            ready: Future[PlanResult] = Future()
            ready.set_result(stages[0].plan)
            pregrasp = Ahead(
                stage="pregrasp",
                prediction=self._prediction,
                planner=candidate.planner,
                future=ready,
                log=lookahead_entry(
                    "pregrasp", self._prediction, raced=[c.label for c in self._candidates]
                ),
            )
        self._selection = Selection(
            candidate=candidate,
            pregrasp=stages[0].plan,
            _start=pregrasp,
            _approach=Kept(stages[0].projection, stages[1].plan, candidate.checker),
        )


def _summary(chain: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    """One candidate, from its chain's record and the race's report: its outcome, why,
    its cost, and every stage of every seed tried: how it ended (``status``) and, if the
    planner returned, the planner's own status (``plan_status``). A candidate whose last stage never
    finished was stopped: by the caller, when the race was (``canceled``); by a strictly
    higher priority's success, when one was chosen and stops lower ones
    (``canceled_by_priority``)."""
    attempts = chain["attempts"]
    if chain["chosen"] is not None:
        seed = attempts[chain["chosen"]]["seed"]
        outcome, detail = "succeeded", f"seed {seed}: every motion of the prefix plans"
    else:
        last = attempts[-1]["stages"][-1]
        unfinished = last["outcome"] in ("not_started", "canceled")
        outranked = report["cancel_lower"] and any(
            c["chosen"] is not None and c["priority"] > chain["priority"] for c in report["chains"]
        )
        if unfinished and report["stopped"] is not None:
            outcome = "canceled"
            detail = f"the race was stopped ({report['stopped']}) at {last['name']}"
        elif unfinished and outranked:
            outcome = "canceled_by_priority"
            detail = f"a higher priority succeeded; stopped at {last['name']}"
        else:
            outcome = "failed"
            detail = (
                f"no seed whose prefix plans; the last stopped at {last['name']}: {last['detail']}"
            )
    return {
        "label": chain["label"],
        "priority": chain["priority"],
        "outcome": outcome,
        "detail": detail,
        "cost_s": chain["cost_s"],
        "plans": [
            {
                "seed": a["seed"],
                "stage": s["name"],
                "stage_seed": s["seed"],
                "status": s["outcome"],
                "plan_status": s["status"],
                "duration_s": s["duration_s"],
            }
            for a in attempts
            for s in a["stages"]
        ],
    }


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
