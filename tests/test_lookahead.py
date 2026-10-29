"""Geodude looks ahead within its grasp, races its arms for a can, and plans the next can
while it retreats from a drop.

The world is ``test_drop``'s can and right-side bin, plus a second can (``can_b``) on the
right of the work surface and a second bin on the left arm's side. Grasps, drops, and
planners are ``test_grasp_place``'s and ``test_drop``'s, for whichever arm is named.

Each scenario is one contract case, recorded in ``lookahead.json``:
- ``plain``: grasp, then drop, without lookahead: the reference.
- ``lookahead``: the same with a ``Lookahead``: every stage has the same outcome and
  evidence kind; the approach is the pregrasp's feasibility check, made on its
  projection before the pregrasp executed, kept and continued on arrival rather than
  planned again; and the log names where each skill stopped looking ahead
  (``close``, ``restrict``, ``release``). Run twice, byte-identical: a driver blocks on a
  plan instead of stepping, so the runtime's timeline does not depend on how long
  planning takes.
- ``transfer_ahead``: the same grasp, given the drop's transfer to plan (``Follow``,
  ``transfer_request``), plans it on a worker while its lift runs, on the lift's
  prediction, which expects ``restrict`` to narrow the attachment's allowance to the
  hand; the grasp hands it on, and the drop, without a retreat, continues it from where
  the lift ended. The transfer is planned once, never on the context's thread, and every
  stage has the same outcome and evidence kind as ``lookahead``'s. Before that, a drop
  that retreats refuses the handed-on transfer, as ``transfer_request`` refuses it: the
  retreat's feasibility is not planned with it.
- ``transfer_disturbed``: the same, with ``can_b`` declared attached between the grasp
  and the drop, a scene change the lift's prediction did not expect: the transfer fails
  for recovery instead of continuing.
- ``mismatch``: an arrival tolerance no arrival meets forces the approach to be
  replanned from the actual state; the stage still succeeds, and says it was replanned.
- ``scene_change``: a scene change the prediction did not record (``can_b`` declared
  attached while the pregrasp runs) fails the approach for recovery instead.
- ``clearance``: both arms race for ``can`` with a 5 mm clearance on the grasp's and
  the drop's free-space stages, and the selected arm grasps, looking ahead, and drops it.
  Every command compiled keeps exactly its stage's clearance: the raced pregrasp and the
  transfer 5 mm, the approach (planned ahead and continued) and the lift none.
- ``not_at_rest``, ``rest_unobservable``: the arm's joint velocities are reported moving,
  or valid only before the arrival. The approach planned ahead is refused, waiting
  within ``rest_wait_ns`` for rest or, unobservable, at once; the grasp fails for
  recovery, and nothing is submitted after the first refusal.
- ``race``: both arms race for ``can``; the left arm is preferred (priority 1) but drops
  into the right bin, which it cannot reach, so its prefix fails at ``transfer``, every
  transfer seed tried from the same lift, after its pregrasp, approach, and lift
  planned, and the right arm is selected. Nothing moves
  before the selection. The right arm then grasps with the raced pregrasp and drops
  ``can`` without a retreat. While its transfer carries ``can`` to the bin, the race for
  ``can_b``, each arm into its own bin with the right arm, on its side, preferred, is
  planned on the transfer's prediction, which expects ``can`` to be released where the
  transfer ends; the selection follows the rule, and its pregrasp continues from there
  as soon as ``can`` has been released. ``can``'s ``landing`` is observed alongside
  ``can_b``'s grasp, and finds it in the bin. Each raced grasp continues the
  approach its race planned, ``can_b``'s from its pregrasp's continuation, and plans
  none again. Whether the left arm was stopped by
  priority or finished depends on timing, so that one field of ``race_b`` may differ
  between runs.

- ``deepest``: the right arm races alone with a grasp whose deepest approach is blocked
  (10 cm past ``approach_m``) and whose shorter ones plan. The race checks the approach
  the grasp guarantees, ``approach_m``, so the candidate is selected, and its grasp
  approaches as deep as plans, between the two.
- ``closed``: both arms race, and the race is closed before it finishes: each candidate
  is ``canceled``, the race ``aborted``; none claims a higher priority stopped it.

``timing.json``, outside the claim, records how long each scenario planned on the
context's thread and on workers, and how many plans of each kind of stage it made, in
all and by arm. Reproduce with ``uv run pytest tests/test_lookahead.py``.
"""

from __future__ import annotations

import dataclasses
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sscbirrt import CBiRRTConfig
from ssrobot import (
    OffsetGoal,
    Planner,
    PlanResult,
    Pose,
    RobotContext,
    TraceKind,
    TsrGoal,
    ValidationError,
    load_installed_package,
)
from ssrobot.mujoco import MujocoRuntime
from ssrobot.planning import CBiRRT

from ssgrasping import (
    Candidate,
    Candidates,
    Follow,
    Lookahead,
    SkillState,
    drop,
    grasp,
    landing,
    run_together,
)
from ssgrasping.drop import transfer_request
from tests.test_drop import BIN, BIN_FLOOR, _drop_spec, _landing_spec
from tests.test_drop import _scene as _can_and_bin
from tests.test_grasp_place import (
    CHECK_ITERATIONS,
    HEIGHT,
    MAX_TICKS,
    RADIUS,
    SURFACE_Z,
    SWING_TOLERANCE_RAD,
    _drive,
    _environment,
    _grasp_spec,
    _report,
)

CAN_B = (0.10, -0.55)
LEFT_BIN = (-BIN[0], BIN[1], BIN[2])
calls: list[dict[str, Any]] = []


def _stage(request: Any) -> str:
    """Which kind of stage a request plans, from its goal: into regions at the carry's
    scaling, a transfer, otherwise a pregrasp; straight along the hand's (horizontal)
    approach axis, an approach; straight up, a lift or a retreat."""
    goal = request.goal
    if isinstance(goal, TsrGoal):
        return "transfer" if request.limits.velocity_scaling < 1.0 else "pregrasp"
    if isinstance(goal, OffsetGoal):
        return "approach" if abs(goal.direction[2]) < 0.9 else "lift_or_retreat"
    return type(goal).__name__


class _Timed:
    """A planner that records how long each plan took, on which thread, and the kind of
    stage it planned."""

    def __init__(self, inner: Planner, scenario: list[str], arm: str) -> None:
        self._inner, self._scenario, self._arm = inner, scenario, arm

    def plan(self, request: Any, scene: Any, *, abort: Any = None) -> Any:
        start = time.perf_counter()
        result = self._inner.plan(request, scene, abort=abort)
        calls.append(
            {
                "scenario": self._scenario[0],
                "arm": self._arm,
                "stage": _stage(request),
                "context_thread": threading.current_thread() is threading.main_thread(),
                "s": time.perf_counter() - start,
            }
        )
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _scene() -> str:
    """``test_drop``'s can and bin, a second can, and a bin on the left."""
    base = _can_and_bin()
    left = base[base.index('    <body name="recycle_bin"') : base.index("  </worldbody>")]
    left = left.replace('name="recycle_bin"', 'name="recycle_bin_left"').replace(
        f'pos="{BIN[0]} {BIN[1]} {BIN[2]}"', f'pos="{LEFT_BIN[0]} {LEFT_BIN[1]} {LEFT_BIN[2]}"'
    )
    left = left.replace('name="bin_', 'name="left_bin_')
    can_b = f"""    <body name="can_b" pos="{CAN_B[0]} {CAN_B[1]} {SURFACE_Z + HEIGHT / 2}">
      <freejoint/>
      <geom type="cylinder" size="{RADIUS} {HEIGHT / 2}" mass="0.05"
            friction="0.6 0.005 0.0001"/>
    </body>
"""
    return base.replace("  </worldbody>", left + can_b + "  </worldbody>")


def _arm(side: str) -> dict[str, str]:
    return {
        "manipulator": f"{side}_arm",
        "end_effector": f"{side}_hand",
        "gripper": f"{side}_gripper",
        "opening_channel": f"{side}_gripper_opening",
    }


def _grasp(can: Pose, obj: str, side: str) -> Any:
    return _grasp_spec(can, object=obj, **_arm(side))


def _drop(obj: str, side: str, bin_side: str) -> Any:
    """``test_drop``'s drop into the bin on ``bin_side``, by the ``side`` arm."""
    spec = _drop_spec(object=obj, **_arm(side))
    if bin_side == "right":
        return spec
    dx = LEFT_BIN[0] - BIN[0]

    def moved(t: Any) -> Any:
        x, y, z = t.origin.position
        return dataclasses.replace(t, origin=dataclasses.replace(t.origin, position=(x + dx, y, z)))

    return dataclasses.replace(
        spec,
        container="recycle_bin_left",
        regions=[moved(t) for t in spec.regions],
    )


def _planners(package: Any, side: str, scenario: list[str]) -> tuple[Planner, Planner]:
    def make(**config: Any) -> Planner:
        planner = CBiRRT.from_package(
            package, f"{side}_arm", config=CBiRRTConfig(edge_resolution=0.01, **config)
        )
        return _Timed(planner, scenario, side)

    return make(), make(max_iterations=CHECK_ITERATIONS)


def _session(
    scene: Path, name: str, run: Any, sinks: Any = (), wrap: Callable[[Any], Any] | None = None
) -> Any:
    package = load_installed_package("geodude_assets")
    scenario = [name]
    arms = {side: _planners(package, side, scenario) for side in ("right", "left")}
    runtime: Any = MujocoRuntime(package, scene=scene, attach_tolerance_rad=SWING_TOLERANCE_RAD)
    if wrap is not None:
        runtime = wrap(runtime)
    with RobotContext(package.description, runtime, sinks=sinks) as ctx:
        poses = {o.name: o.pose for o in ctx.snapshot().objects}
        return run(ctx, arms, poses)


def _pick_and_drop(lookahead: Lookahead | None) -> Any:
    def run(ctx: RobotContext, arms: Any, poses: Any) -> dict[str, Any]:
        planner, checker = arms["right"]
        spec = _grasp(poses["can"], "can", "right")
        picked = _drive(ctx, grasp(ctx, planner, spec, check_planner=checker, lookahead=lookahead))
        report = {"grasp": _report(picked)}
        if picked.succeeded:
            dropping = drop(
                ctx,
                planner,
                _drop("can", "right", "right"),
                check_planner=checker,
                lookahead=lookahead,
            )
            report["drop"] = _report(_drive(ctx, dropping))
        return report

    return run


def _transfer_ahead(disturb: bool) -> Any:
    def run(ctx: RobotContext, arms: Any, poses: Any) -> dict[str, Any]:
        planner, checker = arms["right"]
        spec = _grasp(poses["can"], "can", "right")
        into = dataclasses.replace(_drop("can", "right", "right"), retreat_m=0.0)
        follow = Follow("transfer", transfer_request(ctx, into), planner)
        grasping = grasp(
            ctx, planner, spec, check_planner=checker, lookahead=Lookahead(), follow=follow
        )
        picked = _drive(ctx, grasping)
        report = {"grasp": _report(picked), "handed_on": grasping.upcoming is not None}
        if picked.succeeded:
            # A drop that retreats refuses a transfer planned ahead, and leaves it as it is.
            retreating = _drop("can", "right", "right")
            refusals = []
            for attempt in (
                lambda: transfer_request(ctx, retreating),
                lambda: drop(ctx, planner, retreating, planned=grasping.upcoming),
            ):
                try:
                    attempt()
                except ValidationError as e:
                    refusals.append([e.code, e.path])
            report["retreat_refused"] = refusals
            if disturb:
                ctx.attach("can_b", "left_hand")
            dropping = drop(
                ctx,
                planner,
                into,
                check_planner=checker,
                lookahead=Lookahead(),
                planned=grasping.upcoming,
            )
            report["drop"] = _report(_drive(ctx, dropping))
        return report

    return run


class _Unrested:
    """MujocoRuntime whose joint velocity readings never show the arm at rest: with
    ``moving``, every joint moving at 1 rad/s; otherwise valid before the observation,
    too old to show the arrival."""

    def __init__(self, inner: Any, *, moving: bool) -> None:
        self.inner, self.moving = inner, moving

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def observe(self, request: Any) -> Any:
        observation = self.inner.observe(request)

        def reported(r: Any) -> Any:
            if not r.channel.endswith("_qd"):
                return r
            if self.moving:
                return dataclasses.replace(r, value=tuple(1.0 for _ in r.value))
            before = dataclasses.replace(r.stamp, time_ns=observation.stamp.time_ns - 2_000_000)
            return dataclasses.replace(r, stamp=before)

        return dataclasses.replace(
            observation, readings=tuple(reported(r) for r in observation.readings)
        )


UNRESTED_WAIT_NS = 100_000_000


def _unrested(order: list[list[Any]]) -> Any:
    """The right arm grasps ``can``, looking ahead, while the arm is never observed at
    rest: the approach planned ahead is refused, and the grasp fails for recovery.
    ``order`` collects every submission and continuation decision, in order."""

    def run(ctx: RobotContext, arms: Any, poses: Any) -> dict[str, Any]:
        planner, checker = arms["right"]
        spec = _grasp(poses["can"], "can", "right")
        look = Lookahead(rest_wait_ns=UNRESTED_WAIT_NS)
        picked = _drive(ctx, grasp(ctx, planner, spec, check_planner=checker, lookahead=look))
        return {"grasp": _report(picked), "order": order}

    return run


def _ordered(order: list[list[Any]]) -> Callable[[Any], None]:
    def sink(r: Any) -> None:
        if r.kind is TraceKind.SUBMITTED:
            order.append(["submitted", None])
        elif r.kind is TraceKind.CONTINUED:
            order.append(["continued", r.payload.refusal])

    return sink


def _scene_changed(ctx: RobotContext, arms: Any, poses: Any) -> dict[str, Any]:
    """Declare ``can_b`` held by the idle left hand while the pregrasp runs: a scene
    change no prediction of the pregrasp records."""
    planner, checker = arms["right"]
    spec = _grasp(poses["can"], "can", "right")
    execution = grasp(ctx, planner, spec, check_planner=checker, lookahead=Lookahead())
    injected = False

    def inject() -> None:
        nonlocal injected
        if not injected and execution.stage == "pregrasp" and execution.active is not None:
            ctx.attach("can_b", "left_hand")
            injected = True

    run_together(ctx, [execution], max_ticks=MAX_TICKS, tick=inject)
    assert execution.result is not None
    return {"grasp": _report(execution.result), "injected": injected}


def _raced(ctx: RobotContext, arms: Any, poses: Any) -> dict[str, Any]:
    # can: the left arm preferred, but into the right bin, out of its reach.
    before = ctx.snapshot()
    gripper = ctx.description.gripper("right_gripper").joints
    opened = {j: before.position(j) for j in gripper}  # the gripper starts open
    race = Candidates(
        ctx,
        [
            Candidate(
                label="right",
                grasp=_grasp(poses["can"], "can", "right"),
                planner=arms["right"][0],
                checker=arms["right"][1],
                drop=_drop("can", "right", "right"),
            ),
            Candidate(
                label="left",
                grasp=_grasp(poses["can"], "can", "left"),
                planner=arms["left"][0],
                checker=arms["left"][1],
                drop=_drop("can", "left", "right"),
                priority=1,
            ),
        ],
    )
    started_ns = ctx.now.time_ns
    run_together(ctx, [race], max_ticks=MAX_TICKS)
    after = ctx.snapshot()
    moved = max(
        (abs(a - b), j)
        for j, a, b in zip(after.joints, after.positions, before.positions, strict=True)
    )
    out: dict[str, Any] = {
        "race": race.report,
        "nothing_moved_before_selection": not ctx.executions and moved[0] == 0.0,
        # What the race let happen, to diagnose an intermittent CI failure of the above.
        "during_race": {
            "runtime_s": (ctx.now.time_ns - started_ns) / 1e9,
            "executions": len(ctx.executions),
            "most_moved": [moved[1], moved[0]],
        },
    }
    selection = race.selection
    assert selection is not None and selection.candidate.drop is not None
    planner, checker = arms[selection.candidate.label]
    look = Lookahead()
    before_grasp = len(calls)
    picked = _drive(
        ctx,
        selection.grasp(ctx, lookahead=look),
    )
    out["grasp"] = _report(picked)
    out["grasp_planned"] = _counted(calls[before_grasp:])
    assert picked.succeeded, out["grasp"]["stages"]
    no_retreat = dataclasses.replace(selection.candidate.drop, retreat_m=0.0)
    dropping = drop(ctx, planner, no_retreat, check_planner=checker, lookahead=look)

    # can_b: raced, each arm into its own bin, while the transfer carries can to the bin,
    # where it is expected to land upright on the floor.
    landed = Pose(
        position=(BIN[0], BIN[1], BIN[2] + BIN_FLOOR + HEIGHT / 2), quat_wxyz=(1, 0, 0, 0)
    )
    following: Candidates | None = None

    def start_next() -> None:
        nonlocal following
        if following is None and dropping.stage == "transfer" and dropping.active is not None:
            out["transfer"] = dropping.active.id
            following = Candidates(
                ctx,
                [
                    Candidate(
                        label=side,
                        grasp=_grasp(poses["can_b"], "can_b", side),
                        planner=arms[side][0],
                        checker=arms[side][1],
                        drop=_drop("can_b", side, side),
                        priority=1 if side == "right" else 0,
                    )
                    for side in ("right", "left")
                ],
                after=dropping.active,
                releases={"can": landed},
                joints=opened,
            )

    run_together(ctx, [dropping], max_ticks=MAX_TICKS, tick=start_next)
    assert dropping.result is not None and following is not None
    out["drop"] = _report(dropping.result)
    run_together(ctx, [following], max_ticks=MAX_TICKS)
    out["race_b"] = following.report
    chosen = following.selection
    assert chosen is not None and chosen.candidate.drop is not None
    planner, checker = arms[chosen.candidate.label]
    later = Lookahead(unsettled=("can",))
    before_grasp_b = len(calls)
    # Where can comes to rest is observed alongside can_b's grasp, not before it.
    landing_a = landing(ctx, _landing_spec())
    grasping_b = chosen.grasp(ctx, lookahead=later)
    assert run_together(ctx, [grasping_b, landing_a], max_ticks=MAX_TICKS)
    assert grasping_b.result is not None and landing_a.result is not None
    picked_b = grasping_b.result
    out["landing"] = _report(landing_a.result)
    out["grasp_b"] = _report(picked_b)
    out["grasp_b_planned"] = _counted(calls[before_grasp_b:])
    if picked_b.succeeded:
        dropped_b = drop(
            ctx, planner, chosen.candidate.drop, check_planner=checker, lookahead=later
        )
        out["drop_b"] = _report(_drive(ctx, dropped_b))
    return out


def _counted(made: list[dict[str, Any]]) -> dict[str, int]:
    """How many plans of each kind of stage ``made`` holds."""
    counts: dict[str, int] = {}
    for c in made:
        counts[c["stage"]] = counts.get(c["stage"], 0) + 1
    return dict(sorted(counts.items()))


CLEARANCE_M = 0.005


def _cleared(compiled: list[float]) -> Any:
    """Both arms race for ``can`` with a clearance on their grasps' and drops' free-space
    stages; the selected arm grasps, looking ahead, and drops it. ``compiled`` collects
    the clearance of every command compiled, in order."""

    def run(ctx: RobotContext, arms: Any, poses: Any) -> dict[str, Any]:
        def spec(side: str) -> Candidate:
            return Candidate(
                label=side,
                grasp=dataclasses.replace(
                    _grasp(poses["can"], "can", side), clearance_m=CLEARANCE_M
                ),
                planner=arms[side][0],
                checker=arms[side][1],
                drop=dataclasses.replace(
                    _drop("can", side, "right"), clearance_m=CLEARANCE_M, retreat_m=0.0
                ),
            )

        race = Candidates(ctx, [spec("right"), spec("left")])
        run_together(ctx, [race], max_ticks=MAX_TICKS)
        selection = race.selection
        assert selection is not None and selection.candidate.drop is not None
        assert isinstance(selection.pregrasp, PlanResult)
        planner, checker = arms[selection.candidate.label]
        look = Lookahead()
        picked = _drive(
            ctx,
            selection.grasp(ctx, lookahead=look),
        )
        out: dict[str, Any] = {
            "selected": selection.candidate.label,
            "raced_pregrasp_clearance_m": selection.pregrasp.request.clearance_m,
            "grasp": _report(picked),
        }
        if picked.succeeded:
            dropping = drop(
                ctx, planner, selection.candidate.drop, check_planner=checker, lookahead=look
            )
            out["drop"] = _report(_drive(ctx, dropping))
        out["compiled_clearance_m"] = list(compiled)
        return out

    return run


DEEPEST_PAST_M = 0.10
"""How far past ``approach_m`` the ``deepest`` scenario's grasp asks to approach: into the
can, so the longest line is blocked and a shorter one plans."""


def _deepest(ctx: RobotContext, arms: Any, poses: Any) -> dict[str, Any]:
    """The right arm races alone, with a grasp whose deepest approach is blocked
    (``DEEPEST_PAST_M`` past ``approach_m``) and whose shorter ones plan. The race checks
    the approach the grasp guarantees, ``approach_m``, so the candidate is selected; the
    selected grasp then approaches as deep as plans, short of the blocked line."""
    base = _grasp(poses["can"], "can", "right")
    spec = dataclasses.replace(
        base, deepest=lambda pose: base.approach_m + DEEPEST_PAST_M, approach_tries=4
    )
    race = Candidates(
        ctx,
        [Candidate(label="right", grasp=spec, planner=arms["right"][0], checker=arms["right"][1])],
    )
    run_together(ctx, [race], max_ticks=MAX_TICKS)
    out: dict[str, Any] = {"race": race.report, "approach_m": base.approach_m}
    selection = race.selection
    if selection is not None:
        out["grasp"] = _report(_drive(ctx, selection.grasp(ctx, lookahead=Lookahead())))
    return out


def _closed(ctx: RobotContext, arms: Any, poses: Any) -> dict[str, Any]:
    """Both arms race for ``can``, and the race is closed before it can finish: no
    candidate was outranked, so none is reported as stopped by priority."""
    race = Candidates(
        ctx,
        [
            Candidate(
                label=side,
                grasp=_grasp(poses["can"], "can", side),
                planner=arms[side][0],
                checker=arms[side][1],
                drop=_drop("can", side, side),
                priority=1 if side == "right" else 0,
            )
            for side in ("right", "left")
        ],
    )
    race.close()
    return {"race": race.report, "selected": race.selection is not None}


def _kinds(report: dict[str, Any]) -> list[tuple[str, bool, str]]:
    return [(s["name"], s["succeeded"], s["evidence"]) for s in report["stages"]]


def _entry(report: dict[str, Any], stage: str) -> dict[str, Any]:
    return next(
        e for e in report["values"]["lookahead"] if e["stage"] == stage and "boundary" not in e
    )


def test_geodude_looks_ahead_and_races_its_arms(artifacts: Path, tmp_path: Path) -> None:
    scene = tmp_path / "two_cans_two_bins.xml"
    scene.write_text(_scene())
    calls.clear()
    first = json.dumps(_session(scene, "lookahead", _pick_and_drop(Lookahead())), indent=2)
    again = json.dumps(_session(scene, "lookahead_again", _pick_and_drop(Lookahead())), indent=2)
    moving: list[list[Any]] = []
    stale: list[list[Any]] = []
    scenarios = {
        "plain": _session(scene, "plain", _pick_and_drop(None)),
        "lookahead": json.loads(first),
        "transfer_ahead": _session(scene, "transfer_ahead", _transfer_ahead(disturb=False)),
        "transfer_disturbed": _session(scene, "transfer_disturbed", _transfer_ahead(disturb=True)),
        "mismatch": _session(scene, "mismatch", _pick_and_drop(Lookahead(arrival_tolerance=1e-9))),
        "scene_change": _session(scene, "scene_change", _scene_changed),
        "race": _session(scene, "race", _raced),
        "not_at_rest": _session(
            scene,
            "not_at_rest",
            _unrested(moving),
            sinks=[_ordered(moving)],
            wrap=lambda rt: _Unrested(rt, moving=True),
        ),
        "rest_unobservable": _session(
            scene,
            "rest_unobservable",
            _unrested(stale),
            sinks=[_ordered(stale)],
            wrap=lambda rt: _Unrested(rt, moving=False),
        ),
        "deepest": _session(scene, "deepest", _deepest),
        "closed": _session(scene, "closed", _closed),
        "clearance": _session(
            scene,
            "clearance",
            _cleared(compiled := []),
            sinks=[
                lambda r: (
                    compiled.append(r.payload.request.clearance_m)
                    if r.kind is TraceKind.COMPILED
                    else None
                )
            ],
        ),
    }
    report = {
        "environment": _environment(),
        "repeat_identical": first == again,
        "scenarios": scenarios,
    }
    (artifacts / "lookahead.json").write_text(json.dumps(report, indent=2) + "\n")
    timing: dict[str, dict[str, Any]] = {}
    for c in calls:
        row = timing.setdefault(
            c["scenario"], {"context_thread_s": 0.0, "workers_s": 0.0, "planned": {}}
        )
        row["context_thread_s" if c["context_thread"] else "workers_s"] += c["s"]
    for name, row in timing.items():
        made = [c for c in calls if c["scenario"] == name]
        row["planned"] = _counted(made)
        row["by_arm"] = {
            arm: {
                "planned": _counted([c for c in made if c["arm"] == arm]),
                "s": sum(c["s"] for c in made if c["arm"] == arm),
            }
            for arm in sorted({c["arm"] for c in made})
        }
    (artifacts / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")

    assert first == again, "two runs in one environment differ"

    plain, ahead = scenarios["plain"], scenarios["lookahead"]
    for skill in ("grasp", "drop"):
        assert plain[skill]["state"] == ahead[skill]["state"] == SkillState.SUCCEEDED
        assert _kinds(plain[skill]) == _kinds(ahead[skill]), skill
    approach = _entry(ahead["grasp"], "approach")
    assert approach["decision"] == "continued", approach
    assert approach["prediction"]["lineage"] == [approach["prediction"]["base"]]
    stage = next(s for s in ahead["grasp"]["stages"] if s["name"] == "approach")
    assert stage["detail"] == "succeeded (planned ahead, continued)"
    boundaries = [
        e["stage"]
        for skill in ("grasp", "drop")
        for e in ahead[skill]["values"]["lookahead"]
        if "boundary" in e
    ]
    assert boundaries == ["close", "restrict", "release"]
    assert "lookahead" not in plain["grasp"]["values"]

    handed = scenarios["transfer_ahead"]
    assert handed["handed_on"]
    assert handed["retreat_refused"] == [["invalid_spec", "retreat_m"]] * 2
    for skill in ("grasp", "drop"):
        assert handed[skill]["state"] == SkillState.SUCCEEDED, handed[skill]["stages"]
        # lookahead's drop retreats; this one, whose transfer is planned ahead, does not
        expected = [k for k in _kinds(ahead[skill]) if k[0] != "retreat"]
        assert _kinds(handed[skill]) == expected, skill
    planned_ahead = _entry(handed["grasp"], "transfer")
    assert planned_ahead["prediction"]["allowances"] == ["can"], planned_ahead
    assert planned_ahead["decision"] == "handed on: continued by the next skill"
    assert _entry(handed["drop"], "transfer")["decision"] == "continued"
    stage = next(s for s in handed["drop"]["stages"] if s["name"] == "transfer")
    assert stage["detail"] == "succeeded (planned ahead, continued)"
    transfers = [c for c in calls if c["scenario"] == "transfer_ahead" and c["stage"] == "transfer"]
    assert [c["context_thread"] for c in transfers] == [False], transfers  # once, on a worker
    disturbed = scenarios["transfer_disturbed"]
    assert disturbed["grasp"]["state"] == SkillState.SUCCEEDED
    assert disturbed["drop"]["state"] == SkillState.FAILED
    last = disturbed["drop"]["stages"][-1]
    assert last["name"] == "transfer" and not last["succeeded"]
    assert last["detail"].startswith("continuation refused, unpredicted_scene_change"), last
    assert _entry(disturbed["drop"], "transfer")["decision"] == "failed: for recovery"

    mismatch = scenarios["mismatch"]["grasp"]
    assert mismatch["state"] == SkillState.SUCCEEDED
    assert _entry(mismatch, "approach")["decision"] == "replanned: arrival_out_of_tolerance"
    stage = next(s for s in mismatch["stages"] if s["name"] == "approach")
    assert stage["succeeded"] and "replanned: arrival_out_of_tolerance" in stage["detail"]

    changed = scenarios["scene_change"]
    assert changed["injected"]
    assert changed["grasp"]["state"] == SkillState.FAILED
    last = changed["grasp"]["stages"][-1]
    assert last["name"] == "approach" and not last["succeeded"]
    assert last["detail"].startswith("continuation refused, unpredicted_scene_change")
    assert _entry(changed["grasp"], "approach")["decision"] == "failed: for recovery"

    raced = scenarios["race"]
    assert raced["nothing_moved_before_selection"], raced["during_race"]
    outcomes = {c["label"]: c for c in raced["race"]["candidates"]}
    assert raced["race"]["selected"] == "right"
    left = outcomes["left"]
    assert left["outcome"] == "failed"
    # Some seed planned its pregrasp, approach, and lift, validated, and failed at the
    # transfer into a bin out of its reach: every transfer seed, tried from that same
    # lift, without planning the prefix again.
    chains = {c["label"]: c for c in raced["race"]["chains"]["chains"]}
    transfer_seeds = chains["left"]["stage_seeds"][3]
    reached = [
        a
        for a in chains["left"]["attempts"]
        if [(s["name"], s["outcome"]) for s in a["stages"]][:3]
        == [("pregrasp", "valid"), ("approach", "valid"), ("lift", "valid")]
    ]
    assert reached
    for a in reached:
        tries = a["stages"][3:]
        assert [(s["name"], s["seed"]) for s in tries] == [("transfer", k) for k in transfer_seeds]
        assert all(s["outcome"] != "valid" for s in tries)
    assert chains["left"]["chosen"] is None
    assert raced["grasp"]["state"] == raced["drop"]["state"] == SkillState.SUCCEEDED

    race_b = raced["race_b"]
    assert race_b["basis"]["on"] == "prediction"
    assert race_b["basis"]["prediction"]["held"] == []  # released, not carried on
    assert race_b["basis"]["prediction"]["releases"] == ["can"]
    assert [s["name"] for s in raced["drop"]["stages"]] == [
        "preflight", "transfer", "release",
    ]  # fmt: skip
    # can_b's grasp starts at the release; can's landing is judged a land_ns later, by
    # then far into can_b's pregrasp, and finds can in the bin.
    released_ns = raced["drop"]["stages"][-1]["time_ns"]
    assert raced["grasp_b"]["stages"][0]["time_ns"] == released_ns
    assert raced["landing"]["state"] == SkillState.SUCCEEDED, raced["landing"]["stages"]
    landed = raced["landing"]["stages"][0]
    assert (landed["name"], landed["evidence"]) == ("landed", "privileged")
    assert landed["time_ns"] == released_ns + _landing_spec().land_ns
    feasible = [c for c in race_b["candidates"] if c["outcome"] == "succeeded"]
    rule = min(feasible, key=lambda c: (-c["priority"], c["cost_s"]))
    assert race_b["selected"] == rule["label"] == "right"
    pregrasp = _entry(raced["grasp_b"], "pregrasp")
    assert pregrasp["during"] == raced["transfer"]  # planned while the can was carried
    assert pregrasp["decision"] == "continued", pregrasp
    # Each raced grasp continues the approach its race planned, and plans none again,
    # though can_b's pregrasp ran as a continuation, rebased onto the actual arrival.
    for picked, planned in (("grasp", "grasp_planned"), ("grasp_b", "grasp_b_planned")):
        kept = _entry(raced[picked], "approach")
        assert kept.get("kept") == "the feasibility check, planned before", kept
        assert "approach" not in raced[planned], raced[planned]
    assert raced["grasp_b"]["state"] == SkillState.SUCCEEDED
    assert raced["drop_b"]["state"] == SkillState.SUCCEEDED

    # Neither rest refusal submits a command: not at rest, it waits within the declared
    # rest_wait_ns, unobservable, it fails at once; either way the grasp fails for
    # recovery, and nothing is submitted after the first refusal.
    for refusal in ("not_at_rest", "rest_unobservable"):
        unrested = scenarios[refusal]
        assert unrested["grasp"]["state"] == SkillState.FAILED
        last = unrested["grasp"]["stages"][-1]
        assert last["name"] == "approach" and not last["succeeded"]
        assert last["detail"].startswith(f"continuation refused, {refusal}"), last
        entry = _entry(unrested["grasp"], "approach")
        assert entry["decision"] == "failed: for recovery", entry
        order = unrested["order"]
        first = order.index(["continued", refusal])
        assert ["submitted", None] in order[:first]  # the pregrasp
        assert all(o == ["continued", refusal] for o in order[first:]), order
    assert _entry(scenarios["not_at_rest"]["grasp"], "approach")["waited_for_rest_ns"] >= (
        UNRESTED_WAIT_NS
    )
    assert _entry(scenarios["rest_unobservable"]["grasp"], "approach")["waited_for_rest_ns"] == 0
    # Unobservable, it is refused once; not at rest, again each tick until the wait ends.
    decided = {
        refusal: [o for o in scenarios[refusal]["order"] if o[0] == "continued"]
        for refusal in ("not_at_rest", "rest_unobservable")
    }
    assert decided["rest_unobservable"] == [["continued", "rest_unobservable"]]
    assert len(decided["not_at_rest"]) > 1

    # A grasp whose deepest approach is blocked races on the approach it guarantees: it is
    # selected, and approaches as deep as plans, deeper than approach_m and short of the
    # blocked line.
    deepest = scenarios["deepest"]
    assert deepest["race"]["selected"] == "right", deepest["race"]["candidates"]
    assert deepest["grasp"]["state"] == SkillState.SUCCEEDED, deepest["grasp"]["stages"]
    reached = deepest["grasp"]["values"]["approach_m"]
    assert deepest["approach_m"] < reached < deepest["approach_m"] + DEEPEST_PAST_M

    # A race closed before it finished reports its candidates canceled by the caller, not
    # by a higher priority's success.
    closed = scenarios["closed"]
    assert not closed["selected"] and closed["race"]["chains"]["stopped"] == "aborted"
    assert {c["outcome"] for c in closed["race"]["candidates"]} == {"canceled"}

    # Only the free-space stages keep the clearance, whichever path planned them: the
    # race's pregrasp, the approach planned ahead and continued, the lift, the transfer.
    cleared = scenarios["clearance"]
    assert cleared["raced_pregrasp_clearance_m"] == CLEARANCE_M
    assert cleared["grasp"]["state"] == cleared["drop"]["state"] == SkillState.SUCCEEDED
    assert cleared["compiled_clearance_m"] == [CLEARANCE_M, 0.0, 0.0, CLEARANCE_M]
