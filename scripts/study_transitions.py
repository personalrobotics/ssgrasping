"""Which of Geodude's portable signals observe each manipulation transition (#28).

An observability study, not a test: every tick of a grasp and its place or drop, it
records the right arm's portable signals and, separately, the simulator's truth, so that
each transition (``acquired``, ``retained``, ``released``, ``placed``, ``landed``) can be
scored against what physically happened. Nothing here changes a skill.

- **Portable:** the gripper opening, the wrist wrench (raw, untared, in the sensor's frame),
  the arm's joint positions, velocities, and efforts, and the hand pose, read with
  ``portable_only=True`` so ssrobot itself refuses anything privileged.
- **Truth:** the can's pose, contact and normal force between the can and each finger's pad
  and follower, and the attachment monitor's report.

Each case is one contract case of #28, run over seeds that perturb the can's start by up
to 1 mm and 0.05 rad. Faults the skills cannot cause are made by the harness, through the
runtime's MuJoCo data, and only here: an external pull on the can mid-carry, and a weld
that keeps the can on a pad when the fingers open.

    uv run python scripts/study_transitions.py record CASE SEED OUT
    uv run python scripts/study_transitions.py run OUT [--jobs N]
    uv run python scripts/study_transitions.py score OUT SUMMARY

``record`` writes ``OUT/<case>-<seed>.npz`` (every tick) and ``.json`` (the skills' results
and the recording's sha256); ``run`` records every case and seed into an empty ``OUT``, and
writes ``OUT/provenance.json`` only if every one succeeded; ``score`` reduces them to
``SUMMARY``. ``score`` refuses an ``OUT`` without that provenance, with any case and seed
missing, repeated or unknown, or with a recording whose sha256 no longer matches.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # the test fixtures this study reuses

from ssrobot import ObservationRequest, Pose, RobotContext, load_installed_package  # noqa: E402
from ssrobot.mujoco import MujocoRuntime  # noqa: E402

from ssgrasping import SkillExecution, drop, grasp, place  # noqa: E402
from tests.test_drop import _drop_spec  # noqa: E402
from tests.test_drop import _scene as _bin_scene  # noqa: E402
from tests.test_grasp_place import (  # noqa: E402
    CAN_XY,
    ELSEWHERE,
    HEIGHT,
    MAX_TICKS,
    RADIUS,
    SURFACE_Z,
    SWING_TOLERANCE_RAD,
    _environment,
    _grasp_spec,
    _place_spec,
    _planners,
    _report,
)

ARM = "right"
PORTABLE = (
    f"{ARM}_gripper_opening",
    f"{ARM}_wrist_wrench",
    f"{ARM}_arm_with_lift_q",
    f"{ARM}_arm_with_lift_qd",
    f"{ARM}_arm_with_lift_qf",
    f"{ARM}_hand_pose",
)
FINGERS = tuple(
    f"{ARM}_ur5e/gripper/{side}_{part}"
    for side in ("left", "right")
    for part in ("pad", "follower")
)
TRUTH = ("can.pose", *(f"contact.can.{f}" for f in FINGERS))
SETTLE_NS = 2_000_000_000
"""How long recording continues after the last skill ends, so the can comes to rest."""
SEEDS = range(10)
PERTURB_M, PERTURB_RAD = 0.001, 0.05
HEAVIEST_KG = 0.411
"""The heaviest of Geodude's recycling objects in prl_assets (``cracker_box``,
``geometric_properties.mass``), carried here in the test can's shape."""
PULL_N = 200.0
PULL_NS = 100_000_000
"""The external slip: a pull straight down on the can, mid-carry, for 100 ms. A fault
injection, not a model parameter: large enough to tear the can from the grip."""
OUTSIDE_SHIFT_M = 0.35
"""The outside-the-bin drop: the drop zone moved this far along +y, over the floor."""
WORKTOP_EDGE_X = 0.65
"""Where the worktop ends in +x: ``vention_base``'s collision box, 0.65 m half-wide about
its origin (geodude_assets ``models/geodude/geodude.xml``)."""
OVERHANG_M = 0.02
"""The outside-the-region place: the target's centre this far past the worktop's edge,
so the can is set down overhanging it and tips off."""


@dataclasses.dataclass(frozen=True)
class Case:
    then: str  # "place" or "drop"
    mass: float = 0.05
    post: bool = False
    missed: bool = False
    grasp: dict[str, Any] = dataclasses.field(default_factory=dict)
    after: dict[str, Any] = dataclasses.field(default_factory=dict)
    carry_scaling: float | None = None
    pull: bool = False
    stuck: bool = False
    outside_bin: bool = False
    overhang: bool = False


CASES: dict[str, Case] = {
    "place": Case("place"),
    "drop": Case("drop"),
    "place_heavy": Case("place", mass=HEAVIEST_KG),
    "drop_heavy": Case("drop", mass=HEAVIEST_KG),
    "missed": Case("drop", missed=True),
    "obstructed": Case("drop", missed=True, post=True),
    "lost": Case("drop", mass=20.0),
    "thrown": Case("drop", carry_scaling=1.0),
    "pulled": Case("drop", pull=True),
    "pulled_heavy": Case("drop", mass=HEAVIEST_KG, pull=True),
    "partial_open_drop": Case("drop", after={"open_opening": 0.5}),
    "partial_open_place": Case("place", after={"open_opening": 0.5}),
    "stuck": Case("drop", stuck=True),
    "outside_bin": Case("drop", outside_bin=True),
    "outside_region": Case("place", overhang=True),
}


def _start(seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    dx, dy = rng.uniform(-PERTURB_M, PERTURB_M, 2)
    yaw = rng.uniform(-PERTURB_RAD, PERTURB_RAD)
    return CAN_XY[0] + float(dx), CAN_XY[1] + float(dy), float(yaw)


def _scene(case: Case, seed: int) -> str:
    """The bin scene, with the can of ``case.mass`` at ``seed``'s start, a post where a
    missed grasp aims if ``case.post``, and the weld ``case.stuck`` switches on."""
    x, y, yaw = _start(seed)
    xml = _bin_scene()
    xml = xml.replace(
        f'<body name="can" pos="{CAN_XY[0]} {CAN_XY[1]} {SURFACE_Z + HEIGHT / 2}">',
        f'<body name="can" pos="{x} {y} {SURFACE_Z + HEIGHT / 2}" euler="0 0 {yaw}">',
    ).replace('mass="0.05"', f'mass="{case.mass}"')
    if case.post:
        at = f"{CAN_XY[0]} {CAN_XY[1] + ELSEWHERE} {SURFACE_Z + HEIGHT / 2}"
        xml = xml.replace(
            "  </worldbody>",
            f"""    <body name="post" pos="{at}">
      <geom type="cylinder" size="{RADIUS} {HEIGHT / 2}"/>
    </body>
  </worldbody>""",
        )
    return xml


class Recorder:
    """Every tick's portable signals and truth, with the stage in progress."""

    def __init__(self, ctx: RobotContext) -> None:
        self.ctx = ctx
        self.portable = ObservationRequest(channels=PORTABLE, portable_only=True)
        self.truth = ObservationRequest(channels=TRUTH)
        self.rows: dict[str, list[Any]] = {k: [] for k in ("t", "stage", "portable", "truth")}
        self.stages: list[str] = []

    def __call__(self, stage: str) -> None:
        if stage not in self.stages:
            self.stages.append(stage)
        portable = self.ctx.observe(self.portable).readings
        truth = self.ctx.observe(self.truth).readings
        attachment = self.ctx.scene.attachment("can")
        held = -1.0 if attachment is None else float(attachment.held)
        self.rows["t"].append(self.ctx.now.time_ns)
        self.rows["stage"].append(self.stages.index(stage))
        self.rows["portable"].append([v for r in portable for v in _flat(r.value)])
        self.rows["truth"].append([v for r in truth for v in _flat(r.value)] + [held])

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "t": np.asarray(self.rows["t"], dtype=np.int64),
            "stage": np.asarray(self.rows["stage"], dtype=np.int16),
            "portable": np.asarray(self.rows["portable"], dtype=np.float64),
            "truth": np.asarray(self.rows["truth"], dtype=np.float64),
        }


def _flat(value: Any) -> tuple[float, ...]:
    """A reading's numbers: every channel recorded here is a tuple of floats."""
    assert isinstance(value, tuple), value
    return value


class Faults:
    """The harness's injections, through the runtime's MuJoCo data."""

    def __init__(self, runtime: MujocoRuntime, case: Case) -> None:
        import mujoco

        self.case, self.runtime = case, runtime
        model = runtime._model
        self.can = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "can")
        self.pad = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, FINGERS[2])
        self.pull_from: int | None = None
        self.welded = False
        self.events: dict[str, int] = {}

    def __call__(self, stage: str, now: int, active: bool) -> None:
        data = self.runtime._data
        if self.case.pull and stage == "transfer" and active and self.pull_from is None:
            self.pull_from = now + 500_000_000  # well into the carry
        if self.pull_from is not None:
            on = self.pull_from <= now < self.pull_from + PULL_NS
            data.xfrc_applied[self.can, 2] = -PULL_N if on else 0.0
            if on:
                self.events.setdefault("pull_ns", now)
        if self.case.stuck and stage == "release" and not self.welded:
            self._weld(data)
            self.events["weld_ns"] = now

    def _weld(self, data: Any) -> None:
        """Keep the can where it is on the right pad: a weld at their current relative
        pose, switched on."""
        import mujoco

        model = self.runtime._model
        weld = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "stuck")
        p_pad, q_pad = data.xpos[self.pad], data.xquat[self.pad]
        p_can, q_can = data.xpos[self.can], data.xquat[self.can]
        inv = np.zeros(4)
        mujoco.mju_negQuat(inv, q_pad)
        rel_p = np.zeros(3)
        mujoco.mju_rotVecQuat(rel_p, p_can - p_pad, inv)
        rel_q = np.zeros(4)
        mujoco.mju_mulQuat(rel_q, inv, q_can)
        # eq_data for a weld: anchor (3), relpose position (3) and quaternion (4), torquescale
        model.eq_data[weld, 0:3] = 0.0
        model.eq_data[weld, 3:6] = rel_p
        model.eq_data[weld, 6:10] = rel_q
        model.eq_data[weld, 10] = 1.0
        data.eq_active[weld] = 1
        self.welded = True


def _drive(ctx: RobotContext, execution: SkillExecution, record: Recorder, faults: Faults) -> Any:
    for _ in range(MAX_TICKS):
        execution.advance()
        if execution.done:
            break
        stage = f"{execution.skill}.{execution.stage}"
        record(stage)
        faults(execution.stage or "", ctx.now.time_ns, execution.active is not None)
        ctx.step()
    assert execution.result is not None, "the skill did not end"
    return execution.result


def _outside_bin_spec() -> Any:
    spec = _drop_spec()
    zone = spec.regions[0]
    origin = zone.origin
    moved = Pose(
        position=(origin.position[0], origin.position[1] + OUTSIDE_SHIFT_M, origin.position[2]),
        quat_wxyz=origin.quat_wxyz,
    )
    return dataclasses.replace(spec, regions=[dataclasses.replace(zone, origin=moved)])


def _overhanging(spec: Any) -> Any:
    """``spec``'s region moved in x so its centre is ``OVERHANG_M`` past the worktop's edge."""
    region = spec.regions[0]
    _, y, z = region.origin.position
    moved = Pose(position=(WORKTOP_EDGE_X + OVERHANG_M, y, z), quat_wxyz=region.origin.quat_wxyz)
    return dataclasses.replace(spec, regions=[dataclasses.replace(region, origin=moved)])


def record(name: str, seed: int, out: Path) -> Path:
    case = CASES[name]
    xml = _scene(case, seed)
    if case.stuck:
        xml = xml.replace(
            "</mujoco>",
            f'  <equality>\n    <weld name="stuck" body1="can" body2="{FINGERS[2]}" '
            'active="false"/>\n  </equality>\n</mujoco>',
        )
    scene = out / f"{name}-{seed}.xml"
    scene.write_text(xml)
    package = load_installed_package("geodude_assets")
    planner, checker = _planners(package)
    runtime = MujocoRuntime(
        package,
        scene=scene,
        attach_tolerance_rad=SWING_TOLERANCE_RAD,
        contacts=[("can", f) for f in FINGERS],
    )
    results: dict[str, Any] = {}
    with RobotContext(package.description, runtime) as ctx:
        rec = Recorder(ctx)
        faults = Faults(runtime, case)
        can = next(o.pose for o in ctx.snapshot().objects if o.name == "can")
        aim = can
        if case.missed:
            aim = Pose(
                position=(can.position[0], can.position[1] + ELSEWHERE, can.position[2]),
                quat_wxyz=can.quat_wxyz,
            )
        spec = _grasp_spec(aim, **case.grasp)
        picked = _drive(ctx, grasp(ctx, planner, spec, check_planner=checker), rec, faults)
        results["grasp"] = _report(picked)
        if picked.succeeded:
            scaling = {} if case.carry_scaling is None else {"carry_scaling": case.carry_scaling}
            if case.then == "place":
                pspec = _place_spec(**scaling, **case.after)
                if case.overhang:
                    pspec = _overhanging(pspec)
                then = place(ctx, planner, pspec, check_planner=checker)
            else:
                dspec = _outside_bin_spec() if case.outside_bin else _drop_spec()
                dspec = dataclasses.replace(dspec, **scaling, **case.after)
                then = drop(ctx, planner, dspec, check_planner=checker)
            results[case.then] = _report(_drive(ctx, then, rec, faults))
        ended = ctx.now.time_ns
        while ctx.now.time_ns < ended + SETTLE_NS:
            rec("settle")
            ctx.step()
        rec("settle")
    arrays = rec.arrays()
    path = out / f"{name}-{seed}.npz"
    np.savez_compressed(path, **arrays)  # type: ignore[arg-type]
    meta = {
        "case": name,
        "seed": seed,
        "spec": dataclasses.asdict(case),
        "start": _start(seed),
        "stages": rec.stages,
        "portable": list(PORTABLE),
        "truth": [*TRUTH, "attachment_held"],
        "events": faults.events,
        "results": results,
        "sha256": _digest(arrays),
    }
    (out / f"{name}-{seed}.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    return path


def _digest(arrays: Any) -> str:
    """A recording's sha256, over its arrays' bytes in the order they were saved."""
    return hashlib.sha256(
        b"".join(np.ascontiguousarray(arrays[k]).tobytes() for k in arrays)
    ).hexdigest()


# -- Scoring --------------------------------------------------------------------------

# Columns of the recorded arrays, in PORTABLE's and TRUTH's order.
OPENING, FORCE, Q, QD, QF, HAND = 0, slice(1, 4), slice(7, 14), slice(14, 21), slice(21, 28), 28
CAN, CONTACTS, HELD = slice(0, 7), slice(7, 15), 15
G = 9.81
SLIPPED_M = 0.01
"""Retained, by truth: through the carry, the can stays within this of where it was in the
hand when the carry began: ssrobot's default attachment tolerance (``MujocoRuntime``'s
``attach_tolerance_m``). Finger contact alone flickers as a heavy can shifts a few
millimetres and stays held, so it is not the criterion."""
LEFT_HAND_M = 0.05
"""Released, by truth: at the end, the can is no farther than this from the hand."""


def _sensor_noise() -> dict[str, Any]:
    """The wrist sensor's precision as Geodude's model declares it: the MJCF ``noise`` of
    the right wrist's force and torque sensors, from the UR5e datasheet (+-3.5 N, +-0.2
    N m), which MuJoCo does not apply."""
    import xml.etree.ElementTree as ET

    package = load_installed_package("geodude_assets")
    model = Path(package.root) / "models" / "geodude" / "geodude.xml"
    sensors: dict[str | None, float] = {
        e.get("name"): float(e.get("noise", "nan"))
        for e in ET.parse(model).iter()
        if e.tag in ("force", "torque")
    }
    return {
        "force_n": sensors[f"{ARM}_ur5e/ft_sensor_force"],
        "torque_nm": sensors[f"{ARM}_ur5e/ft_sensor_torque"],
        "source": "geodude_assets models/geodude/geodude.xml (UR5e F/T datasheet precision)",
    }


def _in_hand(truth: np.ndarray) -> np.ndarray:
    """Each tick: both fingers, pad or follower, touch the can."""
    touch = truth[:, CONTACTS][:, ::2] > 0  # left pad, left follower, right pad, right follower
    return (touch[:, 0] | touch[:, 1]) & (touch[:, 2] | touch[:, 3])


def _in_hand_frame(portable: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Each tick: the can's position in the hand's frame (the hand pose is portable; the
    can's is truth)."""
    w, x, y, z = (portable[:, HAND + 3 + i] for i in range(4))
    r = np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )
    offset = truth[:, :3] - portable[:, HAND : HAND + 3]
    return np.asarray(np.einsum("nji,nj->ni", r, offset))


def _stage(meta: dict[str, Any], d: Any, name: str) -> np.ndarray:
    return (
        np.asarray(d["stage"] == meta["stages"].index(name))
        if name in meta["stages"]
        else (np.zeros(len(d["t"]), dtype=bool))
    )


def _force(portable: np.ndarray, mask: np.ndarray) -> tuple[float, float] | None:
    """The wrist force's magnitude over the ticks of ``mask``: its mean and standard
    deviation. The arm is never quite still while the gripper moves, so the deviation
    is reported with every static reading."""
    if not mask.any():
        return None
    norm = np.linalg.norm(portable[mask][:, FORCE], axis=1)
    return float(norm.mean()), float(norm.std())


def _gripper(meta: dict[str, Any], skill: str, stage: str) -> str | None:
    result = meta["results"].get(skill)
    if result is None:
        return None
    gripper = result["values"].get("gripper", {}).get(stage)
    return None if gripper is None else gripper["termination"]


def _run_scores(meta: dict[str, Any], d: Any) -> dict[str, Any]:
    """One run's truth and portable features at each transition's decision point."""
    P, T, t = d["portable"], d["truth"], d["t"]
    held = _in_hand(T)
    then = CASES[meta["case"]].then
    scores: dict[str, Any] = {}
    close, acquired = _stage(meta, d, "grasp.close"), _stage(meta, d, "grasp.acquired")
    empty = _force(P, close & ~held)  # the closing hand, before it touches the can
    if close.any():
        end = np.flatnonzero(close)[-1]
        # The can rose with the hand: at least half the lift (``_grasp_spec``'s 0.06 m).
        rose = acquired.any() and T[np.flatnonzero(acquired)[-1], 2] > T[0, 2] + 0.03
        lifted = _force(P, acquired)
        scores["acquired"] = {
            "truth": bool(rose),
            "close_termination": _gripper(meta, "grasp", "close"),
            "close_opening": float(P[end, OPENING]),
            "reclose_termination": _gripper(meta, "grasp", "acquired"),
            "payload_n": None if empty is None or lifted is None else lifted[0] - empty[0],
            "payload_spread_n": None
            if empty is None or lifted is None
            else max(empty[1], lifted[1]),
        }
    carry = _stage(meta, d, f"{then}.transfer")
    if carry.any() and held[np.flatnonzero(carry)[0]]:
        ticks = np.flatnonzero(carry)
        where = _in_hand_frame(P[ticks], T[ticks])
        lost = ticks[np.linalg.norm(where - where[0], axis=1) > SLIPPED_M]
        start_opening = float(P[ticks[0], OPENING])
        force = np.linalg.norm(P[ticks][:, FORCE], axis=1)
        drop_at = None
        if len(lost):
            after = P[lost[0] :, OPENING]
            fell = np.flatnonzero(after < start_opening / 2)
            drop_at = None if not len(fell) else int(t[lost[0] + fell[0]] - t[lost[0]])
        scores["retained"] = {
            "truth": not len(lost),
            "lost_ns": None if not len(lost) else int(t[lost[0]] - t[ticks[0]]),
            "opening_fall": float(start_opening - P[ticks][:, OPENING].min()),
            "force_range_n": [float(force.min()), float(force.max())],
            "opening_half_latency_ns": drop_at,
        }
    release = _stage(meta, d, f"{then}.release")
    if release.any():
        last = np.flatnonzero(release)[-1]
        hand_to_can = float(np.linalg.norm(T[-1, :3] - P[-1, HAND : HAND + 3]))
        stuck = "weld_ns" in meta["events"]  # the harness's weld: the can stays on the pad
        after = _force(P, _stage(meta, d, "settle"))  # the hand, held still, at the end
        scores["released"] = {
            "truth": hand_to_can > LEFT_HAND_M and not held[-1] and not stuck,
            "release_termination": _gripper(meta, then, "release"),
            "release_opening": float(P[last, OPENING]),
            "load_after_n": None if after is None or empty is None else after[0] - empty[0],
            "load_spread_n": None if after is None or empty is None else max(empty[1], after[1]),
        }
    final = T[-1, CAN]
    pose = Pose(position=tuple(final[:3]), quat_wxyz=tuple(final[3:7]))
    from ssgrasping._poses import inside
    from tests.test_drop import _landing_spec

    if then == "place" and release.any():
        spec = _place_spec()
        if CASES[meta["case"]].overhang:
            spec = _overhanging(spec)
        scores["placed"] = {
            "truth": any(inside(pose, r, spec.region_tolerance) for r in spec.regions),
            "portable": "none: the object's pose after release is not a portable signal",
        }
    if then == "drop" and release.any():
        landing = _landing_spec()
        scores["landed"] = {
            "truth": any(inside(pose, r, landing.region_tolerance) for r in landing.inside),
            "portable": "none: the object's pose after release is not a portable signal",
        }
    return scores


def _separation(rows: list[dict[str, Any]], feature: str) -> dict[str, Any] | None:
    """How ``feature`` falls in runs whose transition truly happened and in runs whose did
    not: each side's range, and the gap between them (negative: they overlap)."""
    yes = [r[feature] for r in rows if r["truth"] and isinstance(r.get(feature), (int, float))]
    no = [r[feature] for r in rows if not r["truth"] and isinstance(r.get(feature), (int, float))]
    if not yes or not no:
        return {"true": _span(yes), "false": _span(no), "gap": None}
    gap = max(min(yes) - max(no), min(no) - max(yes))
    return {"true": _span(yes), "false": _span(no), "gap": gap}


def _span(values: list[float]) -> list[float] | None:
    return [min(values), max(values)] if values else None


class Refused(Exception):
    """The study's inputs are not one complete, unaltered run of every case and seed."""


def score(out: Path, summary: Path) -> dict[str, Any]:
    provenance = out / "provenance.json"
    if not provenance.exists():
        raise Refused(f"{provenance} is missing: `run` did not finish with every case and seed")
    expected = {(c, s) for c in CASES for s in SEEDS}
    seen: set[tuple[str, int]] = set()
    runs = []
    for meta_path in sorted(out.glob("*.json")):
        if meta_path == provenance:
            continue
        meta = json.loads(meta_path.read_text())
        pair = (meta.get("case"), meta.get("seed"))
        if pair not in expected or meta_path.stem != f"{pair[0]}-{pair[1]}":
            raise Refused(f"{meta_path} is not a recording of this study's cases and seeds")
        if pair in seen:
            raise Refused(f"{meta_path} repeats {pair}")
        seen.add(pair)
        d = np.load(meta_path.with_suffix(".npz"))
        if _digest(d) != meta["sha256"]:
            raise Refused(f"{meta_path.with_suffix('.npz')} does not match its recorded sha256")
        runs.append(
            {
                "case": meta["case"],
                "seed": meta["seed"],
                "sha256": meta["sha256"],
                **_run_scores(meta, d),
            }
        )
    if seen != expected:
        raise Refused(f"{out} is missing {sorted(expected - seen)}")
    by: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        for transition in ("acquired", "retained", "released", "placed", "landed"):
            if transition in run:
                by.setdefault(transition, []).append({"case": run["case"], **run[transition]})
    features = {
        "acquired": ("close_opening", "payload_n"),
        "retained": ("opening_fall",),
        "released": ("release_opening", "load_after_n"),
    }
    noise = _sensor_noise()
    report = {
        "recorded": json.loads(provenance.read_text()),
        "runs": runs,
        "sensor_noise": noise,
        "transitions": {
            name: {
                "cases": sorted({r["case"] for r in rows}),
                "true": sum(r["truth"] for r in rows),
                "false": sum(not r["truth"] for r in rows),
                "separation": {f: _separation(rows, f) for f in features.get(name, ())},
                "by_case": _by_case(rows, features.get(name, ())),
                "terminations": _terminations(rows),
            }
            for name, rows in by.items()
        },
    }
    summary.write_text(json.dumps(report, indent=1) + "\n")
    return report


def _by_case(rows: list[dict[str, Any]], features: tuple[str, ...]) -> dict[str, Any]:
    """Per case: how many runs the transition truly happened in, and each feature's range."""
    out: dict[str, Any] = {}
    for case in sorted({r["case"] for r in rows}):
        mine = [r for r in rows if r["case"] == case]
        out[case] = {
            "true": sum(r["truth"] for r in mine),
            "false": sum(not r["truth"] for r in mine),
        }
        extra = ("close_termination", "reclose_termination", "release_termination")
        for f in (*features, *extra, "opening_half_latency_ns"):
            values: list[Any] = [r[f] for r in mine if r.get(f) is not None]
            if values and isinstance(values[0], str):
                out[case][f] = sorted(set(values))
            elif values:
                out[case][f] = _span(values)
    return out


def _terminations(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """For a gripper decision: how often each terminal state came with the truth."""
    out: dict[str, dict[str, int]] = {}
    for r in rows:
        term = r.get("close_termination") or r.get("release_termination")
        if term is not None:
            row = out.setdefault(term, {"true": 0, "false": 0})
            row["true" if r["truth"] else "false"] += 1
    return out


def run(out: Path, jobs: int) -> None:
    """Record every case and seed, each in its own interpreter, into an empty ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise Refused(f"{out} is not empty: record each run into its own directory")
    work = [(c, s) for c in CASES for s in SEEDS]
    failed = []
    with ProcessPoolExecutor(jobs) as pool:
        for case, seed, code in pool.map(_one, work, [out] * len(work)):
            print(case, seed, "ok" if code == 0 else f"exit {code}", flush=True)
            if code != 0:
                failed.append(f"{case}-{seed}")
    if failed:
        raise Refused(f"{len(failed)} recordings failed (see their logs in {out}): {failed}")
    provenance = {
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip(),
        "dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--", "src", "scripts", "tests"],
                cwd=ROOT,
                capture_output=True,
                text=True,
            ).stdout
        ),
        "environment": _environment(),
        "host": platform.platform(),
        "cpus": os.cpu_count(),
        "openblas_coretype": os.environ.get("OPENBLAS_CORETYPE"),
        "command": sys.argv,
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=1) + "\n")


def _one(work: tuple[str, int], out: Path) -> tuple[str, int, int]:
    case, seed = work
    log = (out / f"{case}-{seed}.log").open("w")
    code = subprocess.run(
        [sys.executable, __file__, "record", case, str(seed), str(out)],
        stdout=log,
        stderr=subprocess.STDOUT,
    ).returncode
    return case, seed, code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("record")
    one.add_argument("case", choices=sorted(CASES))
    one.add_argument("seed", type=int)
    one.add_argument("out", type=Path)
    every = sub.add_parser("run")
    every.add_argument("out", type=Path)
    every.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    scored = sub.add_parser("score")
    scored.add_argument("out", type=Path)
    scored.add_argument("summary", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "record":
            args.out.mkdir(parents=True, exist_ok=True)
            print(record(args.case, args.seed, args.out))
        elif args.command == "run":
            run(args.out, args.jobs)
        else:
            score(args.out, args.summary)
    except Refused as refused:
        print(f"refused: {refused}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
