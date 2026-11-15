"""Whether the geometry the planner holds covers ADA's cylinder through a grasp and carry
(#47).

A study script, not a test. ADA's bare hand (``ada_assets.models``) grasps the 4 cm by
12 cm cylinder of ssrobot's ADA reference scene with ``grasp``, on registered object
poses, then carries it as ``place`` would: a transfer planned to the upright placement
raised by its preplace distance, at the carry scaling, and a straight descent that stops
short of the table. ``place`` itself does not run: on ADA it needs an effort guard (A4).

Two departures from ``grasp`` as it stands, both what #47 would declare for the JACO2:
- the close on this cylinder ends ``reached_target``, the fingers wrapped round it once
  it has rolled into the palm, which ``grasp`` today takes for a miss (and, on the
  re-close after the lift, for a loss). The study substitutes the JACO2's declaration,
  that ``reached_target`` does not refute acquisition (``_JACO2_SEMANTICS``);
- the roll-in takes about 12 s, so the gripper's deadline is ``GRIPPER_DEADLINE_NS``, a
  bound on the close, not the default 5 s.

From the tick the cylinder is attached, the planner represents what moves as the hand's
collision geoms at the measured joints plus the cylinder at the attachment transform
ssrobot stored at ``attach``. MuJoCo, and so the planner, collides a mesh as its convex
hull. At every tick the study measures how far the actual cylinder, at its simulated pose,
reaches outside that union: the largest over the cylinder's solid of its signed distance
to the union (negative inside), less the clearance the validator applies to that motion.
The simulated pose is privileged evaluation data: it decides nothing here, and nothing
in the run reads it.

The solid is sampled on a cylindrical grid. The signed distance to a union is the least
of the members' signed distances, and so 1-Lipschitz; every point of the solid lies
within ``bound_m`` of a sample, so the true largest protrusion lies within
``[sampled, sampled + bound_m]``. The planned cylinder's signed distance is exact; each
hull's is MuJoCo's ``mj_geomDistance`` from a probe sphere.

Recorded separately: the registration, the stored attachment transform, and the
evaluated live pose, at attach and at each stage's witness tick. Per stage: the largest
protrusion, its tick, and its witness point. Once the cylinder first touches no hand geom
(MuJoCo's contact condition, all margins zero, evaluated in the study's own copy of the
model), it has left the hand: later ticks are recorded under ``<stage>:released``, and the
headline ``largest`` is over the ticks before.

Reproduce with ``uv run python scripts/study_held_geometry.py OUT.json``; it refuses an
existing output.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from sscbirrt import CBiRRTConfig
from ssrobot import (
    ObjectPoses,
    ObservationRequest,
    PlanResult,
    Pose,
    RobotContext,
    Tsr,
    load_installed_package,
)
from ssrobot.mujoco import MujocoRuntime
from ssrobot.planning import CBiRRT
from tsr import KinovaJaco2TwoFinger
from tsr.placement.stable_placer import StablePlacer

from ssgrasping import GraspObject, GraspRegion, Primitive, grasp, grasp_regions, sstsr
from ssgrasping._poses import invert
from ssgrasping.place import _for_hand, _moved

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = "ada_assets.models"
MANIPULATOR = "arm"
HAND_ROOT = "j2n6s200_link_6"  # the hand: this body and every body below it
# ssrobot's references/scenes/ada-cylinder.xml at 1d3cb77: the over-bed table's collision
# top and the cylinder where that scene stands it. The visual meshes, the columns, the
# floor and the seated user are left out: they are far from every motion here.
RADIUS, HALF_HEIGHT = 0.02, 0.06
TABLE_TOP_Z = 0.735
CYLINDER_POSE = ((0.3082, -0.6416, 0.7950), (0.270668, 0.0, 0.0, 0.962673))
SCENE = f"""<mujoco model="ada-cylinder-study">
  <worldbody>
    <body name="ada_table" pos="0.06 -0.48 0">
      <geom name="ada_table_top" type="box" pos="0 0 0.714" size="0.926 0.371 0.021"
            friction="0.8 0.005 0.0001" mass="10"/>
    </body>
    <body name="cylinder" pos="{" ".join(map(str, CYLINDER_POSE[0]))}"
          quat="{" ".join(map(str, CYLINDER_POSE[1]))}">
      <freejoint/>
      <geom type="cylinder" size="{RADIUS} {HALF_HEIGHT}" mass="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""
UP = (0.0, 0.0, 1.0)
APPROACH_M = 0.05
LIFT_M = 0.06
TARGET_XY = (0.20, -0.55)  # on the table, 14 cm from the cylinder
PREPLACE_M = 0.04
STOP_ABOVE_M = 0.01  # the descent stops this far above where the placement would rest
CARRY_SCALING = 0.5  # place's default
CLEARANCE_M = 0.0  # what every motion here plans with: grasp's and place's defaults
MAX_TICKS = 60_000
GRIPPER_DEADLINE_NS = 30_000_000_000
SERIES_EVERY = 10  # ticks: one row of the series every 20 ms


class _JACO2_SEMANTICS:
    """In ``grasp``'s place of ``GripperTermination``: a termination that never equals
    ``REACHED_TARGET``, so neither the close nor the re-close is taken for a miss."""

    REACHED_TARGET = object()


# The sampling grid over the solid cylinder.
RINGS, SECTORS, LEVELS = 8, 64, 48
PROBE_M = 1e-6
DISTMAX_M = 10.0  # beyond any distance here: mj_geomDistance reports at most this


def _matrix(pose: Pose) -> np.ndarray:
    T = np.eye(4)
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(pose.quat_wxyz, dtype=float))
    T[:3, :3] = mat.reshape(3, 3)
    T[:3, 3] = pose.position
    return T


def _pose(value: Any) -> Pose:
    x, y, z, w, i, j, k = value
    return Pose(position=(x, y, z), quat_wxyz=(w, i, j, k))


def _samples() -> tuple[np.ndarray, float]:
    """Points over the solid cylinder in its own frame, and the radius within which every
    point of the solid has one."""
    dr, dtheta, dz = RADIUS / RINGS, 2 * math.pi / SECTORS, 2 * HALF_HEIGHT / LEVELS
    z = np.linspace(-HALF_HEIGHT, HALF_HEIGHT, LEVELS + 1)
    points = [np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)]
    for r in np.linspace(dr, RADIUS, RINGS):
        for theta in np.arange(SECTORS) * dtheta:
            points.append(
                np.stack(
                    [np.full_like(z, r * math.cos(theta)), np.full_like(z, r * math.sin(theta)), z],
                    1,
                )
            )
    # A point at (r, theta, z) has a node within dr/2 radially, dtheta/2 in angle (an arc
    # of at most RADIUS * dtheta / 2), and dz/2 axially.
    bound = math.hypot(dr / 2 + RADIUS * dtheta / 2, dz / 2)
    return np.concatenate(points), bound


def _cylinder_sdf(points: np.ndarray) -> np.ndarray:
    """The exact signed distance of points, in a cylinder's frame, to that cylinder."""
    radial = np.hypot(points[:, 0], points[:, 1]) - RADIUS
    axial = np.abs(points[:, 2]) - HALF_HEIGHT
    outside = np.hypot(np.maximum(radial, 0), np.maximum(axial, 0))
    signed: np.ndarray = outside + np.minimum(np.maximum(radial, axial), 0)
    return signed


class Hand:
    """The hand's collision geoms at given joints, in the package's own model, with a probe
    sphere to measure a point's signed distance to each."""

    def __init__(self, package: Any, frame: str) -> None:
        manifest = package.manifest
        (canonical,) = [m for m in manifest.models if m.name == manifest.canonical_model]
        spec = mujoco.MjSpec.from_file(str(package.root / canonical.path))
        probe = spec.worldbody.add_body(name="probe", mocap=True)
        probe.add_geom(
            name="probe",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[PROBE_M, 0, 0],
            contype=0,
            conaffinity=0,
        )
        live = spec.worldbody.add_body(name="live", mocap=True)
        live.add_geom(
            name="live",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[RADIUS, HALF_HEIGHT, 0],
            contype=0,
            conaffinity=0,
        )
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.probe = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "probe")
        self.live = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "live")
        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, frame)
        root = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, HAND_ROOT)

        def in_hand(body: int) -> bool:
            while body:
                if body == root:
                    return True
                body = int(self.model.body_parentid[body])
            return False

        m = self.model
        self.geoms = [
            g
            for g in range(m.ngeom)
            if in_hand(int(m.geom_bodyid[g])) and (m.geom_contype[g] or m.geom_conaffinity[g])
        ]
        self.names = [
            f"{m.body(int(m.geom_bodyid[g])).name}/{m.geom(g).name or g}" for g in self.geoms
        ]
        self.fromto = np.zeros(6)

    def pose(self, joints: tuple[str, ...], positions: tuple[float, ...]) -> np.ndarray:
        """Set the joints; the end effector frame in the world."""
        for name, q in zip(joints, positions, strict=True):
            j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.data.qpos[self.model.jnt_qposadr[j]] = q
        mujoco.mj_kinematics(self.model, self.data)
        T = np.eye(4)
        T[:3, :3] = self.data.site_xmat[self.site].reshape(3, 3)
        T[:3, 3] = self.data.site_xpos[self.site]
        return T

    def touching(self, live: np.ndarray) -> bool:
        """Whether the cylinder at ``live`` touches a hand geom: MuJoCo's own contact
        condition, every margin here being zero."""
        self.data.geom_xpos[self.live] = live[:3, 3]
        self.data.geom_xmat[self.live] = live[:3, :3].reshape(9)
        return any(
            mujoco.mj_geomDistance(self.model, self.data, self.live, g, DISTMAX_M, self.fromto) <= 0
            for g in self.geoms
        )

    def distances(self, point: np.ndarray, below: float) -> tuple[float, int]:
        """The least signed distance from ``point`` to a hand geom, and which; stops at
        the first at or below ``below``."""
        self.data.geom_xpos[self.probe] = point
        least, which = math.inf, -1
        for i, g in enumerate(self.geoms):
            d = mujoco.mj_geomDistance(self.model, self.data, self.probe, g, DISTMAX_M, self.fromto)
            d += PROBE_M
            if d < least:
                least, which = d, i
                if d <= below:
                    break
        return least, which


def _protrusion(
    hand: Hand, samples: np.ndarray, live: np.ndarray, planned: np.ndarray
) -> tuple[float, np.ndarray, str]:
    """The largest signed distance over ``samples`` (in the cylinder's frame), placed at
    ``live``, to the union of the hand and the cylinder at ``planned``: its value, its
    witness in the world, and the member nearest the witness. Branch and bound: the
    planned cylinder's exact distance bounds the union's from above."""
    world = samples @ live[:3, :3].T + live[:3, 3]
    in_planned = (world - planned[:3, 3]) @ planned[:3, :3]
    cylinder = _cylinder_sdf(in_planned)
    best, witness, member = -math.inf, world[0], "planned_cylinder"
    for i in np.argsort(-cylinder, kind="stable"):
        if cylinder[i] <= best:
            break
        d, which = hand.distances(world[i], best)
        value = min(float(cylinder[i]), d)
        if value > best:
            best, witness = value, world[i]
            member = "planned_cylinder" if cylinder[i] <= d else hand.names[which]
    return best, witness, member


def _environment() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    names = ("ssgrasping", "ssrobot", "sscbirrt", "sstsr", "mujoco", "numpy", "ada-assets")
    return {
        "commit": git("rev-parse", "HEAD"),
        "clean": git("status", "--porcelain") == "",
        "command": ["scripts/study_held_geometry.py", "OUT.json"],
        "host": platform.node(),
        "python": platform.python_version(),
        "versions": {n: importlib.metadata.version(n) for n in names},
    }


def study() -> dict[str, Any]:
    package = load_installed_package(PACKAGE)
    description = package.description
    (end_effector,) = description.end_effectors
    assert end_effector.gripper is not None
    planner = CBiRRT.from_package(package, MANIPULATOR, config=CBiRRTConfig(edge_resolution=0.01))
    profile = sstsr.profile(
        KinovaJaco2TwoFinger(),
        manipulator=MANIPULATOR,
        end_effector=end_effector.name,
        gripper=end_effector.gripper,
        opening_channel="gripper_opening",
    )
    hand = Hand(package, end_effector.frame)
    samples, bound = _samples()
    with tempfile.TemporaryDirectory() as tmp:
        scene = Path(tmp) / "scene.xml"
        scene.write_text(SCENE)
        runtime = MujocoRuntime(package, scene=scene)
        with RobotContext(description, runtime, object_poses=ObjectPoses.REGISTERED) as ctx:
            pose_request = ObservationRequest(channels=("cylinder.pose",))
            opening_request = ObservationRequest(channels=("gripper_opening",))

            def live() -> Pose:
                return _pose(ctx.observe(pose_request).readings[0].value)

            observation = ctx.observe(pose_request)
            registration = _pose(observation.readings[0].value)
            ctx.register("cylinder", registration, stamp=observation.stamp, source="pose_channel")
            regions = grasp_regions(
                GraspObject(
                    "cylinder",
                    registration,
                    (GraspRegion("body", Primitive("cylinder", (RADIUS, 2 * HALF_HEIGHT))),),
                ),
                profile,
                sstsr.SstsrTemplates(KinovaJaco2TwoFinger()),
                approach_m=APPROACH_M,
            )
            spec = dataclasses.replace(
                regions.spec(support="ada_table", normal=UP, lift_m=LIFT_M),
                gripper_timeout_ns=GRIPPER_DEADLINE_NS,
            )

            stages: dict[str, dict[str, Any]] = {}
            series: list[list[Any]] = []
            stored: list[Pose] = []
            ticks = [0]
            attached_at: dict[str, Any] = {}
            released: list[int] = []  # the first tick the cylinder touches no hand geom

            def record(stage: str) -> None:
                ticks[0] += 1
                attachment = ctx.scene.attachment("cylinder")
                if attachment is None:
                    return
                if not stored:
                    stored.append(attachment.transform)
                # The planner's transform never changes while held.
                assert attachment.transform == stored[0], (attachment.transform, stored[0])
                snapshot = ctx.snapshot()
                effector = hand.pose(snapshot.joints, snapshot.positions)
                planned = effector @ _matrix(stored[0])
                actual = live()
                L = _matrix(actual)
                if not attached_at:
                    attached_at.update(
                        time_ns=ctx.now.time_ns,
                        live_pose=[list(actual.position), list(actual.quat_wxyz)],
                        planned_pose=[planned[:3, 3].tolist(), _quat(planned)],
                    )
                if not released and not hand.touching(L):
                    released.append(ctx.now.time_ns)
                if released:
                    stage = f"{stage}:released"
                value, witness, member = _protrusion(hand, samples, L, planned)
                best = stages.get(stage)
                if best is None or value > best["protrusion_m"]:
                    stages[stage] = {
                        "protrusion_m": value,
                        "time_ns": ctx.now.time_ns,
                        "witness_world_m": [float(v) for v in witness],
                        "nearest_member": member,
                        "live_pose": [list(actual.position), list(actual.quat_wxyz)],
                        "planned_pose": [planned[:3, 3].tolist(), _quat(planned)],
                    }
                stages[stage]["ticks"] = stages[stage].get("ticks", 0) + 1
                if ticks[0] % SERIES_EVERY == 0:
                    opening = ctx.observe(opening_request).readings[0].value
                    assert isinstance(opening, tuple)
                    offset = float(np.linalg.norm(L[:3, 3] - planned[:3, 3]))
                    tilt = math.acos(min(1.0, abs(float(L[:3, 2] @ planned[:3, 2]))))
                    series.append(
                        [
                            ctx.now.time_ns,
                            stage,
                            round(value, 6),
                            round(offset, 6),
                            round(tilt, 6),
                            round(float(L[2, 3]), 6),
                            round(float(opening[0]), 6),
                        ]
                    )

            vars(sys.modules["ssgrasping.grasp"])["GripperTermination"] = _JACO2_SEMANTICS
            execution = grasp(ctx, planner, spec)
            execution.advance()
            while not execution.done and ticks[0] < MAX_TICKS:
                stage = execution.stage
                ctx.step()
                record(stage or "none")
                execution.advance()
            assert execution.result is not None
            result = execution.result
            outcome: dict[str, Any] = {
                "grasp": result.state.value,
                "grasp_stages": [
                    {
                        "name": s.name,
                        "succeeded": s.succeeded,
                        "time_ns": s.time_ns,
                        "reason": None if s.reason is None else s.reason.value,
                        "cause": s.cause,
                        "detail": s.detail,
                    }
                    for s in result.stages
                ],
            }
            if result.succeeded:
                attachment = ctx.scene.attachment("cylinder")
                assert attachment is not None
                outcome["carry"] = _carry(ctx, planner, attachment, record)
    if not stored:
        raise SystemExit(f"the cylinder was never attached: {outcome}")
    held = [s for s in stages if not s.endswith(":released")]
    worst_stage = max(held, key=lambda s: stages[s]["protrusion_m"])
    worst = stages[worst_stage]
    return {
        "provenance": _environment(),
        "scene": SCENE,
        "clearance_m": CLEARANCE_M,
        "sampling": {
            "points": len(samples),
            "rings": RINGS,
            "sectors": SECTORS,
            "levels": LEVELS,
            "bound_m": bound,
            "claim": "true largest protrusion in [sampled, sampled + bound_m]",
        },
        "hand_geoms": hand.names,
        "registration": [list(registration.position), list(registration.quat_wxyz)],
        "stored_transform": [list(stored[0].position), list(stored[0].quat_wxyz)],
        "at_attach": attached_at,
        "released_ns": released[0] if released else None,
        "outcome": outcome,
        "stages": stages,
        "series_columns": [
            "time_ns",
            "stage",
            "protrusion_m",
            "centre_offset_m",
            "axis_tilt_rad",
            "live_z_m",
            "gripper_opening",
        ],
        "series": series,
        "largest": {
            "over": "every tick from attach until the cylinder first touches no hand geom",
            "stage": worst_stage,
            "protrusion_m": worst["protrusion_m"],
            "upper_bound_m": worst["protrusion_m"] + bound,
            "exceeds_clearance": worst["protrusion_m"] > CLEARANCE_M,
            "time_ns": worst["time_ns"],
            "witness_world_m": worst["witness_world_m"],
        },
    }


def _quat(T: np.ndarray) -> list[float]:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, T[:3, :3].reshape(9).copy())
    return q.tolist()


def _carry(ctx: RobotContext, planner: Any, attachment: Any, record: Any) -> dict[str, Any]:
    """The transfer and descent ``place`` would plan, without its guard: to the upright
    placement raised by the preplace distance, then down, stopping short of the table."""
    placer = StablePlacer(table_x=0.06, table_y=0.06)
    upright = next(
        t
        for t in placer.place_cylinder(RADIUS, 2 * HALF_HEIGHT, subject="cylinder")
        if t.variant == "-z"
    )
    surface = np.eye(4)
    surface[:3, 3] = (*TARGET_XY, TABLE_TOP_Z)
    t = upright.instantiate(surface)
    rows = [tuple(float(v) for v in row) for row in t.Bw]
    region = Tsr(
        origin=sstsr.pose(t.T0_w),
        offset=sstsr.pose(t.Tw_e),
        position_bounds=tuple(rows[:3]),  # type: ignore[arg-type]
        rotation_bounds=tuple(rows[3:]),  # type: ignore[arg-type]
    )
    raised = _moved(_for_hand(region, invert(attachment.transform)), UP, PREPLACE_M)
    arm = ctx.manipulator(MANIPULATOR)
    report: dict[str, Any] = {}
    steps: list[tuple[str, Callable[[], PlanResult]]] = [
        (
            "transfer",
            lambda: arm.plan_to_tsr(
                [raised],
                planner=planner,
                velocity_scaling=CARRY_SCALING,
                acceleration_scaling=CARRY_SCALING,
                clearance_m=CLEARANCE_M,
            ),
        ),
        (
            "descend",
            lambda: arm.plan_to_end_effector_offset(
                (0.0, 0.0, -1.0), PREPLACE_M - STOP_ABOVE_M, planner=planner
            ),
        ),
    ]
    for name, plan in steps:
        result = plan()
        report[name] = result.status.value
        if not result.succeeded:
            break
        execution = ctx.execute(result, planner=planner)
        while not execution.done:
            ctx.step()
            record(name)
        report[f"{name}_state"] = execution.status.state.value
        for _ in range(250):  # half a second held still after each motion
            ctx.step()
            record(f"{name}_hold")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; never overwritten")
    text = json.dumps(study(), indent=2) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=args.out.parent, prefix=f".{args.out.name}.")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, args.out)
    except BaseException:
        os.unlink(tmp)
        raise
    digest = hashlib.sha256(text.encode()).hexdigest()
    print(f"wrote {args.out} (sha256 {digest})")


if __name__ == "__main__":
    main()
