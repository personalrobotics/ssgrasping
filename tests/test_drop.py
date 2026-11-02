"""Geodude picks a can off its work surface and drops it into the recycling bin beside it.

The bin is a fixture, a primitive copy of prl_assets' ``recycle_bin`` walls, standing on
the floor where Geodude's recycling demo puts the bin on the right arm's side. The can,
its grasp, and the planners are ``test_grasp_place``'s. The drop region is the legacy
demo's container drop zone (mj_manipulator's ``_generate_container_drop_tsrs``), from
the bin's prl_assets ``geometric_properties`` and placement policy, as an application
would supply it: the hand pointing straight down over the bin's opening, its position
anywhere over the interior less the policy's margin, ``max(0.25, can height + 0.10)``
above the rim (+2 cm / -5 cm), at any yaw.

Each scenario is one contract case, recorded in ``drop.json`` with every stage's time,
scene revision, held and attached state, evidence kind, and, for a failed stage, its
reason and cause:
- ``success``: grasp, then drop on a manual clock: the can is released over the bin, the
  drop ends after its retreat with ``landed`` unobserved, and ``landing`` then scores
  that against the can's privileged pose, for evaluation only, finding it inside the
  bin; run twice, byte-identical.
- ``unconfirmed_release``: the gripper opens only halfway, so ``release`` is not
  confirmed and the can stays attached.
- ``pulled``: the harness pulls the can down at ``PULL_N`` for ``PULL_NS``,
  ``PULL_AFTER_NS`` into the transfer, through the runtime's MuJoCo data, and it leaves
  the fingers. They close past its width, the transfer's opening guard stops the arm,
  ``retained`` is refuted, and the transfer fails with the can reported not held and
  still declared attached; the attachment monitor agrees.
- ``monitor_only``: the attachment monitor alone reports the can leaving the fingers
  during the transfer (injected; the can stays in hand). Privileged state decides no
  stage, so the drop succeeds; the evaluation records that the monitor disagreed with
  the estimate.
- ``registered``: ``success`` in a context with registered object poses
  (``ObjectPoses.REGISTERED``), the can registered where its privileged pose has it
  before anything runs, and never read again for control. Released, its pose is
  unknown, and the test's policy removes it (``released``), as an application that
  assumes it landed would, so the retreat can plan: the same stages, estimates and
  evaluations.
Transfers are planned at ``CARRY_SCALING`` of the limits, as ``test_grasp_place``'s.
The environment the artifact was made in is recorded with it; byte-identity is claimed
within one environment.

Reproduce with ``uv run pytest tests/test_drop.py``.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

from ssrobot import AttachmentViolation, Planner, Pose, RobotContext, RuntimeUpdate, Tsr

from ssgrasping import (
    DropSpec,
    LandingSpec,
    Reason,
    SkillResult,
    SkillState,
    Transition,
    TransitionEstimate,
    drop,
    grasp,
    landing,
)
from tests.test_grasp_place import (
    CAN_XY,
    CARRY_SCALING,
    HEIGHT,
    MAX_TICKS,
    RADIUS,
    SURFACE_Z,
    UP,
    _drive,
    _environment,
    _failed,
    _grasp_spec,
    _reasons,
    _report,
    _session,
    _states,
)

BIN = (0.85, -0.35, 0.01)  # the base of the recycling demo's right-side bin
# prl_assets' recycle_bin: geometric_properties and placement policy.
BIN_OUTER = (0.25, 0.25, 0.80)
BIN_WALL = 0.003
DROP_MARGIN = 0.05
# Its MJCF: 20 mm walls with inner faces at +-0.119 m, and a 6 mm floor.
BIN_INNER_HALF = 0.119
BIN_FLOOR = 0.006
DOWN = (0.0, 1.0, 0.0, 0.0)  # x along the world's, z pointing down
# ssgrasping's transition study's external pull (evidence/transitions/): down on the can,
# well past what the grip holds, briefly, partway into the transfer.
PULL_N = 200.0
PULL_NS = 100_000_000
PULL_AFTER_NS = 500_000_000


def _scene() -> str:
    """``test_grasp_place``'s 50 g can, and the bin as a fixture."""
    walls = [
        ("bottom", (0.122, 0.122, 0.003), (0, 0, 0.003)),
        ("front", (0.132, 0.01, 0.397), (0, 0.129, 0.40)),
        ("back", (0.132, 0.01, 0.397), (0, -0.129, 0.40)),
        ("left", (0.01, 0.129, 0.397), (-0.129, 0, 0.40)),
        ("right", (0.01, 0.129, 0.397), (0.129, 0, 0.40)),
    ]
    geoms = "\n".join(
        f'      <geom name="bin_{n}" type="box" size="{" ".join(map(str, s))}" '
        f'pos="{" ".join(map(str, p))}" friction="0.8 0.01 0.001"/>'
        for n, s, p in walls
    )
    return f"""<mujoco>
  <worldbody>
    <body name="can" pos="{CAN_XY[0]} {CAN_XY[1]} {SURFACE_Z + HEIGHT / 2}">
      <freejoint/>
      <geom type="cylinder" size="{RADIUS} {HEIGHT / 2}" mass="0.05"
            friction="0.6 0.005 0.0001"/>
    </body>
    <body name="recycle_bin" pos="{BIN[0]} {BIN[1]} {BIN[2]}">
{geoms}
    </body>
  </worldbody>
</mujoco>
"""


def _drop_spec(**changes: Any) -> DropSpec:
    """The legacy demo's drop zone over the bin."""
    hx = BIN_OUTER[0] / 2 - BIN_WALL - DROP_MARGIN
    hy = BIN_OUTER[1] / 2 - BIN_WALL - DROP_MARGIN
    clearance = max(0.25, HEIGHT + 0.10)
    zone = Tsr(
        origin=Pose(position=(BIN[0], BIN[1], BIN[2] + BIN_OUTER[2] + clearance), quat_wxyz=DOWN),
        offset=Pose(position=(0.0, 0.0, 0.0), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
        position_bounds=((-hx, hx), (-hy, hy), (-0.02, 0.05)),
        rotation_bounds=((0.0, 0.0), (0.0, 0.0), (-math.pi, math.pi)),
    )
    spec = DropSpec(
        manipulator="right_arm",
        end_effector="right_hand",
        gripper="right_gripper",
        opening_channel="right_gripper_opening",
        object="can",
        container="recycle_bin",
        normal=UP,
        regions=[zone],
        retreat_m=0.08,
        carry_scaling=CARRY_SCALING,
    )
    return dataclasses.replace(spec, **changes)


def _landing_spec() -> LandingSpec:
    """The bin's interior, as where the can must come to rest."""
    interior = Tsr(
        origin=Pose(position=BIN, quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
        offset=Pose(position=(0.0, 0.0, 0.0), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
        position_bounds=(
            (-BIN_INNER_HALF, BIN_INNER_HALF),
            (-BIN_INNER_HALF, BIN_INNER_HALF),
            (BIN_FLOOR, BIN_OUTER[2]),
        ),
        rotation_bounds=((-math.pi, math.pi),) * 3,
    )
    return LandingSpec(object="can", inside=[interior])


def _kind(entry: dict[str, Any]) -> tuple[Any, ...]:
    """A transition or evaluation, without the readings that vary between runs."""
    keys = ("transition", "stage", "outcome", "reason", "estimate", "reference", "source")
    return tuple(entry.get(k) for k in keys)


def _landed(dropped: SkillResult) -> TransitionEstimate:
    """The drop's ``landed`` estimate, for ``landing`` to score."""
    return next(t for t in dropped.transitions if t.transition is Transition.LANDED)


def _pick_and_drop(changes: dict[str, Any] | None = None, remove: bool = False) -> Any:
    def run(ctx: RobotContext, planner: Planner, checker: Planner, can: Pose) -> dict[str, Any]:
        def assumed_landed(estimate: TransitionEstimate) -> None:
            ctx.remove("can", source="assumed_landed")

        picked = _drive(ctx, grasp(ctx, planner, _grasp_spec(can), check_planner=checker))
        report = {"grasp": _report(picked)}
        if picked.succeeded:
            spec = _drop_spec(**(changes or {}))
            released = assumed_landed if remove else None
            dropping = drop(ctx, planner, spec, check_planner=checker, released=released)
            dropped = _drive(ctx, dropping)
            report["drop"] = _report(dropped)
            if dropped.succeeded:
                observing = landing(ctx, _landing_spec(), _landed(dropped))
                report["landing"] = _report(_drive(ctx, observing))
        return report

    return run


class SlipDuringTransfer:
    """A test runtime: Geodude in MuJoCo whose attachment monitor reports the can leaving
    the fingers once ``slip`` is set, as a slip during a transfer would report it."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.slip = False
        self._reported = False

    def poll(self) -> RuntimeUpdate:
        update: RuntimeUpdate = self._inner.poll()
        if self._reported:
            # A monitor reports an object leaving the fingers once. MuJoCo, which never
            # saw the injected slip, may still report the can's own swing in hand.
            events = tuple(
                e
                for e in update.events
                if not (isinstance(e, AttachmentViolation) and e.object == "can")
            )
            return dataclasses.replace(update, events=events)
        if not self.slip:
            return update
        self._reported = True
        left = AttachmentViolation(
            object="can", stamp=update.stamp, position_error=0.05, rotation_error=0.0
        )
        return dataclasses.replace(update, events=(*update.events, left))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _pulled(runtimes: list[Any]) -> Any:
    def run(ctx: RobotContext, planner: Planner, checker: Planner, can: Pose) -> dict[str, Any]:
        picked = _drive(ctx, grasp(ctx, planner, _grasp_spec(can), check_planner=checker))
        execution = drop(ctx, planner, _drop_spec(), check_planner=checker)
        runtime = runtimes[-1]
        body = runtime._model.body("can").id
        started = None
        for _ in range(MAX_TICKS):
            execution.advance()
            if execution.done:
                break
            if started is None and execution.stage == "transfer" and execution.active is not None:
                started = ctx.now.time_ns
            since = None if started is None else ctx.now.time_ns - started
            pulling = since is not None and PULL_AFTER_NS <= since < PULL_AFTER_NS + PULL_NS
            runtime._data.xfrc_applied[body, 2] = -PULL_N if pulling else 0.0
            ctx.step()
        runtime._data.xfrc_applied[body, 2] = 0.0
        assert execution.result is not None
        return {"grasp": _report(picked), "drop": _report(execution.result)}

    return run


def _slipped(runtimes: list[SlipDuringTransfer]) -> Any:
    def run(ctx: RobotContext, planner: Planner, checker: Planner, can: Pose) -> dict[str, Any]:
        picked = _drive(ctx, grasp(ctx, planner, _grasp_spec(can), check_planner=checker))
        execution = drop(ctx, planner, _drop_spec(), check_planner=checker)
        for _ in range(MAX_TICKS):
            execution.advance()
            if execution.done:
                break
            if execution.stage == "transfer" and execution.active is not None:
                runtimes[-1].slip = True
            ctx.step()
        assert execution.result is not None
        return {"grasp": _report(picked), "drop": _report(execution.result)}

    return run


def _wrapping(runtimes: list[SlipDuringTransfer]) -> Any:
    """Wrap a session's runtime in ``SlipDuringTransfer``, kept in ``runtimes``."""

    def wrap(runtime: Any) -> SlipDuringTransfer:
        runtimes.append(SlipDuringTransfer(runtime))
        return runtimes[-1]

    return wrap


def _keeping(runtimes: list[Any]) -> Any:
    """Keep a session's runtime in ``runtimes``, unwrapped, for the harness to reach."""

    def keep(runtime: Any) -> Any:
        runtimes.append(runtime)
        return runtime

    return keep


def test_geodude_drops_a_can_into_its_bin(artifacts: Path, tmp_path: Path) -> None:
    scene = tmp_path / "can_and_bin.xml"
    scene.write_text(_scene())
    slips: list[SlipDuringTransfer] = []
    pulls: list[Any] = []

    first = json.dumps(_session(scene, _pick_and_drop()), indent=2)
    again = json.dumps(_session(scene, _pick_and_drop()), indent=2)
    scenarios = {
        "success": json.loads(first),
        "unconfirmed_release": _session(scene, _pick_and_drop({"open_opening": 0.5})),
        "pulled": _session(scene, _pulled(pulls), wrap=_keeping(pulls)),
        "monitor_only": _session(scene, _slipped(slips), wrap=_wrapping(slips)),
        "registered": _session(scene, _pick_and_drop(remove=True), registered=True),
    }
    report = {
        "environment": _environment(),
        "repeat_identical": first == again,
        "scenarios": scenarios,
    }
    (artifacts / "drop.json").write_text(json.dumps(report, indent=2) + "\n")

    assert first == again, "two runs in one environment differ"
    _reasons(scenarios)

    success = scenarios["success"]
    assert success["grasp"]["state"] == SkillState.SUCCEEDED
    assert success["drop"]["state"] == SkillState.SUCCEEDED, success["drop"]["stages"]
    assert [n for n, _ in _states(success, "drop")] == [
        "preflight", "transfer", "release", "retreat",
    ]  # fmt: skip
    assert success["drop"]["attachment"] is None
    # The drop ends with the robot free, and landed unobserved: no portable signal tells
    # where the can came to rest. The landing scores that against the can's privileged
    # pose, for evaluation only: it finds the can in the bin.
    assert success["landing"]["state"] == SkillState.SUCCEEDED, success["landing"]["stages"]
    assert [(s["name"], s["evidence"]) for s in success["landing"]["stages"]] == [
        ("observed", "privileged"),
    ]  # fmt: skip
    assert success["landing"]["evaluations"] == [
        {
            "transition": "landed",
            "stage": "release",
            "estimate": "unobserved",
            "reference": "confirmed",
            "source": "can.pose",
            "agrees": None,
        }
    ]

    unconfirmed = scenarios["unconfirmed_release"]["drop"]
    assert unconfirmed["state"] == SkillState.FAILED, unconfirmed["stages"]
    assert _failed(unconfirmed) == ("release", Reason.RELEASE_UNCONFIRMED, None, None, True)
    assert unconfirmed["attachment"] is not None

    # Registered object poses: the can is registered once, where its privileged pose has
    # it, and nothing reads its pose again for control. The same stages succeed, with the
    # same estimates, and the same evaluations, though the monitor's reference now comes
    # from the runtime's report alone (the registered scene keeps the can held).
    registered = scenarios["registered"]
    for part in ("grasp", "drop", "landing"):
        assert registered[part]["state"] == SkillState.SUCCEEDED, registered[part]["stages"]
        assert [n for n, _ in _states(registered, part)] == [n for n, _ in _states(success, part)]
        for key in ("transitions", "evaluations"):
            assert [_kind(e) for e in registered[part][key]] == [
                _kind(e) for e in success[part][key]
            ], (part, key)

    # The transfer confirms retained from the opening alone, as the monitor does.
    carried, landed = success["drop"]["transitions"]
    assert (carried["stage"], carried["outcome"], carried["reason"]) == (
        "transfer", "confirmed", "opening_held",
    )  # fmt: skip
    assert (landed["transition"], landed["stage"], landed["outcome"], landed["reason"]) == (
        "landed", "release", "unobserved", "no_portable_signal",
    )  # fmt: skip
    assert [e["agrees"] for e in success["drop"]["evaluations"]] == [True]

    # A can pulled out of the fingers mid-transfer: the opening guard stops the arm there.
    pulled = scenarios["pulled"]["drop"]
    assert pulled["state"] == SkillState.FAILED, pulled["stages"]
    assert _failed(pulled) == ("transfer", Reason.TRANSITION_REFUTED, "retained", False, True)
    assert pulled["stages"][-1]["evidence"] == "portable"
    (lost,) = pulled["transitions"]
    assert (lost["stage"], lost["outcome"], lost["reason"]) == (
        "transfer",
        "refuted",
        "opening_fell",
    )
    below = next(c for c in lost["criteria"] if c["name"] == "below")
    assert lost["evidence"][0]["value"][0] < below["value"]
    assert pulled["evaluations"] == [
        {
            "transition": "retained",
            "stage": "transfer",
            "estimate": "refuted",
            "reference": "refuted",
            "source": "attachment_monitor",
            "agrees": True,
        }
    ]
    stopped = pulled["stages"][-1]["time_ns"]
    arrived = next(s for s in success["drop"]["stages"] if s["name"] == "transfer")["time_ns"]
    assert stopped < arrived  # stopped short of the drop zone, not judged at its end
    assert "released_opening" not in pulled["values"]  # the gripper was never opened

    # The monitor alone never decides a stage: the can stayed in hand, and the drop
    # succeeds; the evaluation records the disagreement.
    monitored = scenarios["monitor_only"]["drop"]
    assert monitored["state"] == SkillState.SUCCEEDED, monitored["stages"]
    assert monitored["evaluations"] == [
        {
            "transition": "retained",
            "stage": "transfer",
            "estimate": "confirmed",
            "reference": "refuted",
            "source": "attachment_monitor",
            "agrees": False,
        }
    ]
