"""Geodude looks ahead within its grasp, races its arms for a can, and plans the next can
while it retreats from a drop.

The world is ``test_drop``'s can and right-side bin, plus a second can (``can_b``) on the
right of the work surface and a second bin on the left arm's side. Grasps, drops, and
planners are ``test_grasp_place``'s and ``test_drop``'s, for whichever arm is named.

Each scenario is one contract case, recorded in ``lookahead.json``:
- ``plain``: grasp, then drop, without lookahead: the reference.
- ``lookahead``: the same with a ``Lookahead``: every stage has the same outcome and
  evidence kind; the approach was planned while the pregrasp executed, on its
  prediction, and continued; and the log names where each skill stopped looking ahead
  (``close``, ``restrict``, ``release``). Run twice, byte-identical: a driver blocks on a
  plan instead of stepping, so the runtime's timeline does not depend on how long
  planning takes.
- ``mismatch``: an arrival tolerance no arrival meets forces the approach to be
  replanned from the actual state; the stage still succeeds, and says it was replanned.
- ``scene_change``: a scene change the prediction did not record (``can_b`` declared
  attached while the pregrasp runs) fails the approach for recovery instead.
- ``clearance``: both arms race for ``can`` with a 5 mm clearance on the grasp's and
  the drop's free-space stages, and the selected arm grasps, looking ahead, and drops it.
  Every command compiled keeps exactly its stage's clearance: the raced pregrasp and the
  transfer 5 mm, the approach (planned ahead and continued) and the lift none.
- ``race``: both arms race for ``can``; the left arm is preferred (priority 1) but drops
  into the right bin, which it cannot reach, so its prefix fails at ``transfer`` after
  its pregrasp, approach, and lift planned, and the right arm is selected. Nothing moves
  before the selection. The right arm then grasps with the raced pregrasp and drops
  ``can`` without a retreat. While its transfer carries ``can`` to the bin, the race for
  ``can_b``, each arm into its own bin with the right arm, on its side, preferred, is
  planned on the transfer's prediction, which expects ``can`` to be released where the
  transfer ends; the selection follows the rule, and its pregrasp continues from there
  once ``can`` has been released and has landed. Whether the left arm was stopped by
  priority or finished depends on timing, so that one field of ``race_b`` may differ
  between runs.

``timing.json``, outside the claim, records how long each scenario planned on the
context's thread and on workers. Reproduce with ``uv run pytest tests/test_lookahead.py``.
"""

from __future__ import annotations

import dataclasses
import json
import threading
import time
from pathlib import Path
from typing import Any

from sscbirrt import CBiRRTConfig
from ssrobot import Planner, PlanResult, Pose, RobotContext, TraceKind, load_installed_package
from ssrobot.mujoco import MujocoRuntime
from ssrobot.planning import CBiRRT

from ssgrasping import (
    Candidate,
    Candidates,
    Lookahead,
    SkillState,
    drop,
    grasp,
    run_together,
)
from tests.test_drop import BIN, BIN_FLOOR, _drop_spec
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


class _Timed:
    """A planner that records how long each plan took, and on which thread."""

    def __init__(self, inner: Planner, scenario: list[str]) -> None:
        self._inner, self._scenario = inner, scenario

    def plan(self, request: Any, scene: Any, *, abort: Any = None) -> Any:
        start = time.perf_counter()
        result = self._inner.plan(request, scene, abort=abort)
        calls.append(
            {
                "scenario": self._scenario[0],
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
        inside=[moved(t) for t in spec.inside],
    )


def _planners(package: Any, side: str, scenario: list[str]) -> tuple[Planner, Planner]:
    def make(**config: Any) -> Planner:
        planner = CBiRRT.from_package(
            package, f"{side}_arm", config=CBiRRTConfig(edge_resolution=0.01, **config)
        )
        return _Timed(planner, scenario)

    return make(), make(max_iterations=CHECK_ITERATIONS)


def _session(scene: Path, name: str, run: Any, sinks: Any = ()) -> Any:
    package = load_installed_package("geodude_assets")
    scenario = [name]
    arms = {side: _planners(package, side, scenario) for side in ("right", "left")}
    runtime = MujocoRuntime(package, scene=scene, attach_tolerance_rad=SWING_TOLERANCE_RAD)
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


def _moved(ctx: RobotContext, before: Any) -> float:
    now = ctx.snapshot()
    return float(max(abs(a - b) for a, b in zip(now.positions, before.positions, strict=True)))


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
    run_together(ctx, [race], max_ticks=MAX_TICKS)
    out: dict[str, Any] = {
        "race": race.report,
        "nothing_moved_before_selection": not ctx.executions and _moved(ctx, before) == 0.0,
    }
    selection = race.selection
    assert selection is not None and selection.candidate.drop is not None
    planner, checker = arms[selection.candidate.label]
    look = Lookahead()
    picked = _drive(
        ctx,
        grasp(
            ctx,
            planner,
            selection.candidate.grasp,
            check_planner=checker,
            lookahead=look,
            pregrasp=selection.pregrasp,
        ),
    )
    out["grasp"] = _report(picked)
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
    picked_b = _drive(
        ctx,
        grasp(
            ctx,
            planner,
            chosen.candidate.grasp,
            check_planner=checker,
            lookahead=later,
            pregrasp=chosen.pregrasp,
        ),
    )
    out["grasp_b"] = _report(picked_b)
    if picked_b.succeeded:
        dropped_b = drop(
            ctx, planner, chosen.candidate.drop, check_planner=checker, lookahead=later
        )
        out["drop_b"] = _report(_drive(ctx, dropped_b))
    return out


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
            grasp(
                ctx,
                planner,
                selection.candidate.grasp,
                check_planner=checker,
                lookahead=look,
                pregrasp=selection.pregrasp,
            ),
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
    scenarios = {
        "plain": _session(scene, "plain", _pick_and_drop(None)),
        "lookahead": json.loads(first),
        "mismatch": _session(scene, "mismatch", _pick_and_drop(Lookahead(arrival_tolerance=1e-9))),
        "scene_change": _session(scene, "scene_change", _scene_changed),
        "race": _session(scene, "race", _raced),
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
    timing: dict[str, dict[str, float]] = {}
    for c in calls:
        row = timing.setdefault(c["scenario"], {"context_thread_s": 0.0, "workers_s": 0.0})
        row["context_thread_s" if c["context_thread"] else "workers_s"] += c["s"]
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
    assert raced["nothing_moved_before_selection"]
    outcomes = {c["label"]: c for c in raced["race"]["candidates"]}
    assert raced["race"]["selected"] == "right"
    left = outcomes["left"]
    assert left["outcome"] == "failed" and left["detail"].startswith("no transfer")
    assert {p["stage"] for p in left["plans"] if p["status"] == "succeeded"} >= {
        "pregrasp", "approach", "lift",
    }  # fmt: skip
    assert raced["grasp"]["state"] == raced["drop"]["state"] == SkillState.SUCCEEDED

    race_b = raced["race_b"]
    assert race_b["basis"]["on"] == "prediction"
    assert race_b["basis"]["prediction"]["held"] == []  # released, not carried on
    assert race_b["basis"]["prediction"]["releases"] == ["can"]
    assert [s["name"] for s in raced["drop"]["stages"]] == [
        "preflight", "transfer", "release", "dropped",
    ]  # fmt: skip
    feasible = [c for c in race_b["candidates"] if c["outcome"] == "succeeded"]
    rule = min(feasible, key=lambda c: (-c["priority"], c["cost_s"]))
    assert race_b["selected"] == rule["label"] == "right"
    pregrasp = _entry(raced["grasp_b"], "pregrasp")
    assert pregrasp["during"] == raced["transfer"]  # planned while the can was carried
    assert pregrasp["decision"] == "continued", pregrasp
    assert raced["grasp_b"]["state"] == SkillState.SUCCEEDED
    assert raced["drop_b"]["state"] == SkillState.SUCCEEDED

    # Only the free-space stages keep the clearance, whichever path planned them: the
    # race's pregrasp, the approach planned ahead and continued, the lift, the transfer.
    cleared = scenarios["clearance"]
    assert cleared["raced_pregrasp_clearance_m"] == CLEARANCE_M
    assert cleared["grasp"]["state"] == cleared["drop"]["state"] == SkillState.SUCCEEDED
    assert cleared["compiled_clearance_m"] == [CLEARANCE_M, 0.0, 0.0, CLEARANCE_M]
