"""Geodude picks a can off its work surface and drops it into the recycling bin beside it.

The bin is a fixture, a primitive copy of prl_assets' ``recycle_bin`` walls, standing on
the floor where Geodude's recycling demo puts the bin on the right arm's side. The can,
its grasp, and the planners are ``test_grasp_place``'s. The drop region is for the can,
as a drop's regions are (personalrobotics/ssgrasping#53), as sstsr's ``OpeningPlacer``
gives them for a container's opening: the can upright over the bin, at any yaw, its
lowest point 2 to 10 cm above the rim, its side ``RELEASE_MARGIN_M`` inside the walls
(``_upright``). ``drop`` composes it with how the side grasp holds the can, so the hand
releases it level, the can standing.

Each scenario is one contract case, recorded in ``drop.json`` with every stage's time,
scene revision, held and attached state, evidence kind, and, for a failed stage, its
reason and cause:
- ``success``: grasp, then drop on a manual clock: the can is released upright over the
  bin, where the region puts it (``values["released_at"]``, by the measured joints and
  the attachment), the drop ends after its retreat with ``landed`` unobserved, and
  ``landing`` then scores that against the can's privileged pose, for evaluation only,
  finding it inside the bin; run twice, byte-identical.
- ``unconfirmed_release``: the gripper opens only halfway, so ``release`` is not
  confirmed and the can stays attached.
- ``beside_bracket``: a bracket is fixed over the bin, and the drop may release at one
  of two zones with the hand's pose fixed, given as regions for the can as the grasp
  holds it (``_holding``), which ``drop`` composes back into exactly those hand poses:
  one beside the bracket, where the fingers would touch it opened though not holding the
  can, and one clear of it. Every transfer that
  ends beside the bracket is rejected, naming what the opened fingers would touch, and
  the can is released at the clear zone.
- ``blocked_release``: the zone beside the bracket alone. No transfer ends where the hand
  can open, so the transfer fails ``not_planned`` before the arm moves, with the can held
  and attached and the gripper never commanded.
- ``fall_blocked``: the can's one region has it lying over the bin, turned about its axis
  so the fingers holding it close vertically (``_lying``): one finger is under it, in its
  fall. Every transfer reaches it, and every one is rejected, naming the finger as
  ``fall/...``; the transfer fails ``not_planned`` with the can held and attached.
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
    Candidate,
    Candidates,
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
    run_together,
)
from ssgrasping._poses import compose, inside, invert, rotate
from tests.test_grasp_place import (
    CAN_MASS,
    CAN_XY,
    CARRY_SCALING,
    CLEARANCE_M,
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
# Its MJCF: 20 mm walls with inner faces at +-0.119 m, and a 6 mm floor.
BIN_INNER_HALF = 0.119
BIN_FLOOR = 0.006
DOWN = (0.0, 1.0, 0.0, 0.0)  # x along the world's, z pointing down
# ssgrasping's transition study's external pull (evidence/transitions/): down on the can,
# well past what the grip holds, briefly, partway into the transfer.
PULL_N = 200.0
PULL_NS = 100_000_000
PULL_AFTER_NS = 500_000_000
RIM_Z = BIN[2] + 0.797  # the top of the bin's walls
LOWEST = (0.02, 0.10)  # how far above the rim the released can's lowest point may be
# Inside the walls by the can's radius and this: the 5 mm clearance Geodude keeps and the
# 16 mm a released object strays sideways before it falls past the rim (Geodude's
# calibrate_release.py, 83 free releases).
RELEASE_MARGIN_M = 0.021
LYING_ABOVE_M = 0.15
FALL_M = 0.20  # beyond how far the 2F-140, holding the can, reaches below it
# A bracket fixed over the bin, beside where a fixed-yaw drop zone puts the hand (``_beside``):
# a 10 mm plate whose inner face is 85 mm from the hand's axis, along the fingers' opening,
# at their height. Holding the can, the 2F-140's outer finger links are 61 mm from the axis
# there; opened, 95 mm (geodude_assets e9538a2's declared positions). So only a release
# touches it.
SPAN = 0.10  # how far apart in x the two fixed-yaw zones are, either side of the bin's centre
BRACKET = ((BIN[0] - SPAN / 2, BIN[1] - 0.090, 1.01), (0.02, 0.005, 0.03))


def _scene(bracket: bool = False) -> str:
    """``test_grasp_place``'s full can, and the bin as a fixture; with ``bracket``, the
    bracket too."""
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
      <geom type="cylinder" size="{RADIUS} {HEIGHT / 2}" mass="{CAN_MASS}"
            friction="0.6 0.005 0.0001"/>
    </body>
    <body name="recycle_bin" pos="{BIN[0]} {BIN[1]} {BIN[2]}">
{geoms}
    </body>{_bracket() if bracket else ""}
  </worldbody>
</mujoco>
"""


def _bracket() -> str:
    (x, y, z), (hx, hy, hz) = BRACKET
    return f"""
    <body name="bracket" pos="{x} {y} {z}">
      <geom type="box" size="{hx} {hy} {hz}"/>
    </body>"""


def _beside(*xs: float) -> list[Tsr]:
    """Drop zones over the bin at each x offset from its centre, at the legacy zone's
    height, the hand pointing straight down at yaw 0: the fingers open along the world's
    y, toward the bracket."""
    z = BIN[2] + BIN_OUTER[2] + max(0.25, HEIGHT + 0.10)
    return [
        Tsr(
            origin=Pose(position=(BIN[0] + x, BIN[1], z), quat_wxyz=DOWN),
            offset=Pose(position=(0.0, 0.0, 0.0), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
            position_bounds=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
            rotation_bounds=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        )
        for x in xs
    ]


def _holding(hand: list[Tsr]) -> Any:
    """Regions for the object that put the hand exactly in ``hand``, for the can as the
    grasp holds it: ``drop`` composes them back into ``hand`` (``for_hand``)."""

    def regions(held: Pose) -> list[Tsr]:
        return [dataclasses.replace(r, offset=compose(r.offset, held)) for r in hand]

    return regions


def _upright() -> Tsr:
    """The can upright over the bin's opening, at any yaw: its axis vertical, its lowest
    point ``LOWEST`` above the rim, and its side ``RELEASE_MARGIN_M`` inside the walls."""
    half = BIN_INNER_HALF - RADIUS - RELEASE_MARGIN_M
    return Tsr(
        origin=Pose(position=(BIN[0], BIN[1], RIM_Z + HEIGHT / 2), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
        offset=Pose(position=(0.0, 0.0, 0.0), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
        position_bounds=((-half, half), (-half, half), LOWEST),
        rotation_bounds=((0.0, 0.0), (0.0, 0.0), (-math.pi, math.pi)),
    )


def _lying(held: Pose) -> list[Tsr]:
    """The can lying over the bin's centre, ``LYING_ABOVE_M`` above the rim so the hand
    clears it, its axis along the world's x, turned about it so the fingers holding it,
    as ``held`` has them, close vertically: one finger under it."""
    # The fingers close along the hand's y: in the can's frame, perpendicular to its axis
    # for a side grasp.
    closing = rotate(invert(held).quat_wxyz, (0.0, 1.0, 0.0))
    angle = math.atan2(closing[1], closing[0])  # the closing direction about the can's axis
    # Turn the can about its axis so that direction is its x; then lay it down, x up.
    spin = (math.cos(-angle / 2), 0.0, 0.0, math.sin(-angle / 2))
    lay = (math.cos(-math.pi / 4), 0.0, math.sin(-math.pi / 4), 0.0)  # z to x, x to z
    lying = compose(
        Pose(position=(0.0, 0.0, 0.0), quat_wxyz=lay),
        Pose(position=(0.0, 0.0, 0.0), quat_wxyz=spin),
    )
    return [
        Tsr(
            origin=Pose(
                position=(BIN[0], BIN[1], RIM_Z + RADIUS + LYING_ABOVE_M), quat_wxyz=lying.quat_wxyz
            ),
            offset=Pose(position=(0.0, 0.0, 0.0), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
            position_bounds=((0.0, 0.0),) * 3,
            rotation_bounds=((0.0, 0.0),) * 3,
        )
    ]


def _drop_spec(**changes: Any) -> DropSpec:
    """The can released upright over the bin (``_upright``)."""
    spec = DropSpec(
        manipulator="right_arm",
        end_effector="right_hand",
        gripper="right_gripper",
        opening_channel="right_gripper_opening",
        object="can",
        container="recycle_bin",
        normal=UP,
        regions=[_upright()],
        retreat_m=0.08,
        fall_m=FALL_M,
        carry_scaling=CARRY_SCALING,
        clearance_m=CLEARANCE_M,
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
            attachment = ctx.scene.attachment("can")
            assert attachment is not None
            # A change given as a function of how the can is held is made from it.
            spec = _drop_spec(
                **{
                    k: v(attachment.transform) if callable(v) else v
                    for k, v in (changes or {}).items()
                }
            )
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
    bracketed = tmp_path / "can_bin_and_bracket.xml"
    bracketed.write_text(_scene(bracket=True))
    scenarios = {
        "success": json.loads(first),
        "beside_bracket": _session(
            bracketed, _pick_and_drop({"regions": _holding(_beside(-SPAN / 2, SPAN / 2))})
        ),
        "blocked_release": _session(
            bracketed, _pick_and_drop({"regions": _holding(_beside(-SPAN / 2))})
        ),
        "fall_blocked": _session(scene, _pick_and_drop({"regions": _lying})),
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

    # A release is chosen where the opened hand fits: a transfer whose fingers would open
    # into the bracket is rejected like an infeasible plan, and another is kept.
    beside = scenarios["beside_bracket"]["drop"]
    assert beside["state"] == SkillState.SUCCEEDED, beside["stages"]
    tried = beside["values"]["transfer_attempts"]
    assert tried[-1]["opened"] == [], tried
    rejected = [a for a in tried[:-1] if a["opened"]]
    assert rejected, tried  # the zone beside the bracket was tried, and rejected
    for attempt in rejected:
        assert all(c.startswith("bracket/right_ur5e/gripper/") for c in attempt["opened"]), tried

    # With nowhere the hand can open, the drop fails before the arm moves: the can is
    # still held and attached, and the gripper was never commanded.
    blocked = scenarios["blocked_release"]["drop"]
    assert blocked["state"] == SkillState.FAILED, blocked["stages"]
    assert _failed(blocked) == ("transfer", Reason.NOT_PLANNED, None, True, True)
    assert all(a["opened"] for a in blocked["values"]["transfer_attempts"])
    assert "gripper" not in blocked["values"] and "released_opening" not in blocked["values"]

    # The can is let go upright, where its region puts it: the hand level, holding it from
    # the side, as no hand-down drop zone could.
    position, quat = success["drop"]["values"]["released_at"]
    released = Pose(position=tuple(position), quat_wxyz=tuple(quat))
    assert inside(released, _upright(), (2e-3, 0.05)), released
    assert rotate(released.quat_wxyz, (0.0, 0.0, 1.0))[2] > math.cos(0.05), released

    # A release whose fall a finger is in is never chosen.
    fallen = scenarios["fall_blocked"]["drop"]
    assert fallen["state"] == SkillState.FAILED, fallen["stages"]
    assert _failed(fallen) == ("transfer", Reason.NOT_PLANNED, None, True, True)
    attempts = fallen["values"]["transfer_attempts"]
    assert all(a["transfer"] == "succeeded" for a in attempts), attempts  # each reached it
    for attempt in attempts:
        assert attempt["opened"], attempts
        assert all(c.startswith("fall/right_ur5e/gripper/") for c in attempt["opened"]), attempts
    assert "gripper" not in fallen["values"]

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


def _held(ctx: RobotContext, planner: Planner, checker: Planner, can: Pose) -> dict[str, Any]:
    """How the grasp holds the can, once picked: its attachment transform. The grasp and
    the race plan in the same scene with the same seeds, so the race's predicts it."""
    picked = _drive(ctx, grasp(ctx, planner, _grasp_spec(can), check_planner=checker))
    attachment = ctx.scene.attachment("can")
    assert picked.succeeded and attachment is not None, picked.stages
    return {"held": attachment.transform}


def _race(regions: list[Tsr]) -> Any:
    """A race of the pinned grasp alone, leading to a drop into ``regions``."""

    def run(ctx: RobotContext, planner: Planner, checker: Planner, can: Pose) -> dict[str, Any]:
        candidate = Candidate(
            label="right",
            grasp=_grasp_spec(can),
            planner=planner,
            checker=checker,
            drop=_drop_spec(regions=regions),
        )
        race = Candidates(ctx, [candidate])
        run_together(ctx, [race], max_ticks=MAX_TICKS)
        assert race.report is not None
        return {"selected": race.selection is not None, "report": race.report}

    return run


def test_the_race_checks_the_drop_that_runs(artifacts: Path, tmp_path: Path) -> None:
    """``Candidates`` races a drop as ``drop`` runs it: its regions, for the object,
    composed with how the grasp would hold it, and each transfer kept only where the
    release would be clear (``release_clear``). Raced into the can lying with a finger
    under it (``_lying``, from how the grasp holds it, measured by picking it once), every
    transfer is reached and ``rejected``, and the candidate fails in the race, its detail
    naming the finger, before anything is held; raced into the can upright over the bin
    (``_upright``), it is selected. The evidence is ``race.json``."""
    scene = tmp_path / "can_and_bin.xml"
    scene.write_text(_scene())
    held = _session(scene, _held)["held"]
    report = {
        "lying": _session(scene, _race(_lying(held))),
        "upright": _session(scene, _race([_upright()])),
    }
    (artifacts / "race.json").write_text(json.dumps(report, indent=2, default=str) + "\n")

    lying = report["lying"]
    assert not lying["selected"], lying["report"]
    (candidate,) = lying["report"]["candidates"]
    transfers = [p for p in candidate["plans"] if p["stage"] == "transfer"]
    assert transfers and all(p["status"] == "rejected" for p in transfers), transfers
    assert "release not clear: fall/right_ur5e/gripper/" in candidate["detail"], candidate
    assert report["upright"]["selected"], report["upright"]["report"]
