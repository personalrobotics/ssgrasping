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
scene revision, held and attached state, and evidence kind:
- ``success``: grasp, then drop on a manual clock: the can is released over the bin and
  comes to rest inside it, by its privileged pose; run twice, byte-identical.
- ``unconfirmed_release``: the gripper opens only halfway, so ``release`` is not
  confirmed and the can stays attached.
- ``slipped``: the attachment monitor reports the can leaving the fingers during the
  transfer (injected, as a slip would report it; the can stays in hand). ``release``
  fails before the gripper is commanded, the attachment is kept, and the can is
  reported held: it may still be in hand.
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

from ssgrasping import DropSpec, SkillState, drop, grasp
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
    _grasp_spec,
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
    """The legacy demo's drop zone over the bin, and its interior as where the can must
    come to rest."""
    hx = BIN_OUTER[0] / 2 - BIN_WALL - DROP_MARGIN
    hy = BIN_OUTER[1] / 2 - BIN_WALL - DROP_MARGIN
    clearance = max(0.25, HEIGHT + 0.10)
    zone = Tsr(
        origin=Pose(position=(BIN[0], BIN[1], BIN[2] + BIN_OUTER[2] + clearance), quat_wxyz=DOWN),
        offset=Pose(position=(0.0, 0.0, 0.0), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
        position_bounds=((-hx, hx), (-hy, hy), (-0.02, 0.05)),
        rotation_bounds=((0.0, 0.0), (0.0, 0.0), (-math.pi, math.pi)),
    )
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
        inside=[interior],
        carry_scaling=CARRY_SCALING,
    )
    return dataclasses.replace(spec, **changes)


def _pick_and_drop(changes: dict[str, Any] | None = None) -> Any:
    def run(ctx: RobotContext, planner: Planner, checker: Planner, can: Pose) -> dict[str, Any]:
        picked = _drive(ctx, grasp(ctx, planner, _grasp_spec(can), check_planner=checker))
        report = {"grasp": _report(picked)}
        if picked.succeeded:
            spec = _drop_spec(**(changes or {}))
            report["drop"] = _report(_drive(ctx, drop(ctx, planner, spec, check_planner=checker)))
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
        if not self.slip or self._reported:
            return update
        self._reported = True
        left = AttachmentViolation(
            object="can", stamp=update.stamp, position_error=0.05, rotation_error=0.0
        )
        return dataclasses.replace(update, events=(*update.events, left))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


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


def test_geodude_drops_a_can_into_its_bin(artifacts: Path, tmp_path: Path) -> None:
    scene = tmp_path / "can_and_bin.xml"
    scene.write_text(_scene())
    slips: list[SlipDuringTransfer] = []

    first = json.dumps(_session(scene, _pick_and_drop()), indent=2)
    again = json.dumps(_session(scene, _pick_and_drop()), indent=2)
    scenarios = {
        "success": json.loads(first),
        "unconfirmed_release": _session(scene, _pick_and_drop({"open_opening": 0.5})),
        "slipped": _session(scene, _slipped(slips), wrap=_wrapping(slips)),
    }
    report = {
        "environment": _environment(),
        "repeat_identical": first == again,
        "scenarios": scenarios,
    }
    (artifacts / "drop.json").write_text(json.dumps(report, indent=2) + "\n")

    assert first == again, "two runs in one environment differ"

    success = scenarios["success"]
    assert success["grasp"]["state"] == SkillState.SUCCEEDED
    assert success["drop"]["state"] == SkillState.SUCCEEDED, success["drop"]["stages"]
    assert [n for n, _ in _states(success, "drop")] == [
        "preflight", "transfer", "release", "retreat", "dropped",
    ]  # fmt: skip
    assert success["drop"]["stages"][-1]["evidence"] == "privileged"
    assert success["drop"]["attachment"] is None

    unconfirmed = scenarios["unconfirmed_release"]["drop"]
    assert unconfirmed["state"] == SkillState.FAILED, unconfirmed["stages"]
    assert unconfirmed["stages"][-1]["name"] == "release"
    assert unconfirmed["stages"][-1]["held"] is None
    assert unconfirmed["attachment"] is not None

    slipped = scenarios["slipped"]["drop"]
    assert slipped["state"] == SkillState.FAILED, slipped["stages"]
    assert [(s["name"], s["succeeded"]) for s in slipped["stages"]][-2:] == [
        ("transfer", True), ("release", False),
    ]  # fmt: skip
    assert slipped["stages"][-1]["evidence"] == "privileged"
    assert slipped["stages"][-1]["held"] is True  # conservative: it may still be in hand
    assert "released_opening" not in slipped["values"]  # the gripper was never opened
    assert slipped["attachment"] is not None and slipped["attachment"]["held"] is False
