"""Geodude grasps a mug by its body region, whatever the mug collides as (#46).

The mug's grasp regions are prl_assets' (objects/ycbv/mug/meta.yaml): a ``body``
cylinder and a ``handle`` box, each offset from the mug's frame. Its mass is the YCB
reference mass prl_assets cites, 0.118 kg. Two scenes put the same mug, at the same pose,
on Geodude's work surface: in ``plain`` it collides as one cylinder; in ``handled`` as
that cylinder and a box for its handle, with the same total mass. Recorded in
``regions.json``:
- ``collider``: the body region's candidates are the same in both scenes, made from the
  mug's observed pose and its regions alone, and the grasp and lift by them succeed in
  both. The grasp records the region it reached before its approach: ``body/...``.
- ``edit``: a handle region with other dimensions changes the handle's candidates and
  not the body's, and neither scene file changes.
- ``instances``: two mugs at different poses have different candidates, each region
  placed at its mug's pose composed with the region's origin, and each mug named in its
  provenance.
The scene files' hashes are recorded with the candidates and the outcomes.

Reproduce with ``uv run pytest tests/test_regions.py``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from ssrobot import ObjectPoses, ObservationRequest, Pose, RobotContext, load_installed_package
from ssrobot.mujoco import MujocoRuntime

from ssgrasping import GraspObject, GraspRegion, Primitive, grasp, grasp_regions, sstsr
from ssgrasping._poses import compose
from tests.test_grasp_place import (
    HAND,
    PROFILE,
    STANDOFF,
    SURFACE_Z,
    SWING_TOLERANCE_RAD,
    UP,
    _drive,
    _environment,
    _planners,
    _report,
)

MASS = 0.118  # the YCB mug's reference mass (Calli et al. 2015), as prl_assets cites it
MUG_XY = (0.35, -0.40)
BODY_RADIUS, MUG_HEIGHT = 0.0395, 0.0814  # the scan's body and height (prl_assets)
_YAW = math.atan2(-0.04840964125460411, 0.9988275660160769)
BODY = GraspRegion(
    "body",
    Primitive("cylinder", (0.039, 0.074)),
    Pose(
        position=(-0.011774273067372281, -0.00016874629554637638, 0.0),
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    ),
    ("side",),
)
HANDLE = GraspRegion(
    "handle",
    Primitive("box", (0.028, 0.012, 0.056)),
    Pose(
        position=(0.045, -0.0031024567164179098, 0.0),
        quat_wxyz=(math.cos(_YAW / 2), 0.0, 0.0, math.sin(_YAW / 2)),
    ),
    ("face_x", "face_y"),
)


def _scene(handled: bool) -> str:
    """The mug at ``MUG_XY`` on the work surface, its frame at its centre: collided as its
    body's cylinder, or as that and a box for its handle, of the same total mass."""
    bx, by, _ = BODY.origin.position
    hx, hy, _ = HANDLE.origin.position
    handle_mass = 0.010 if handled else 0.0
    quat = " ".join(map(str, HANDLE.origin.quat_wxyz))
    handle = (
        f"""
      <geom type="box" size="0.014 0.006 0.028" pos="{hx} {hy} 0" quat="{quat}"
            mass="{handle_mass}" friction="0.6 0.005 0.0001"/>"""
        if handled
        else ""
    )
    return f"""<mujoco>
  <worldbody>
    <body name="mug" pos="{MUG_XY[0]} {MUG_XY[1]} {SURFACE_Z + MUG_HEIGHT / 2}">
      <freejoint/>
      <geom type="cylinder" size="{BODY_RADIUS} {MUG_HEIGHT / 2}" pos="{bx} {by} 0"
            mass="{MASS - handle_mass}" friction="0.6 0.005 0.0001"/>{handle}
    </body>
  </worldbody>
</mujoco>
"""


def _mug(name: str, pose: Pose, handle: GraspRegion = HANDLE) -> GraspObject:
    return GraspObject(name, pose, (BODY, handle))


def _pick(scene: Path) -> dict[str, Any]:
    """Make the mug's body candidates from its observed pose, then grasp and lift it."""
    package = load_installed_package("geodude_assets")
    planner, checker = _planners(package)
    runtime = MujocoRuntime(package, scene=scene, attach_tolerance_rad=SWING_TOLERANCE_RAD)
    with RobotContext(package.description, runtime, object_poses=ObjectPoses.LIVE) as ctx:
        observed = next(o.known_pose() for o in ctx.snapshot().objects if o.name == "mug")
        regions = grasp_regions(
            _mug("mug", observed), PROFILE, sstsr.SstsrTemplates(HAND), approach_m=STANDOFF
        )
        body = grasp_regions(
            _mug("mug", observed),
            PROFILE,
            sstsr.SstsrTemplates(HAND),
            approach_m=STANDOFF,
            regions=("body",),
        )
        spec = body.spec(support="vention_base", normal=UP, lift_m=0.06)
        result = _drive(ctx, grasp(ctx, planner, spec, check_planner=checker))
        lifted = ctx.observe(ObservationRequest(channels=("mug.pose",))).readings[0].value
    return {
        "observed": observed,
        "candidates": regions,
        "grasp": _report(result),
        "pregrasp_region": result.values.get("pregrasp_region"),
        "lifted_z": float(lifted[2]),  # type: ignore[index]
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_geodude_grasps_a_mug_by_its_body_region(artifacts: Path, tmp_path: Path) -> None:
    scenes = {}
    for name, with_handle in (("plain", False), ("handled", True)):
        scenes[name] = tmp_path / f"{name}.xml"
        scenes[name].write_text(_scene(with_handle))
    hashes = {name: _sha(path) for name, path in scenes.items()}
    picks = {name: _pick(path) for name, path in scenes.items()}

    # collider: the same candidates, from the same pose and regions, and a grasp in both.
    plain, handled = picks["plain"], picks["handled"]
    assert plain["observed"] == handled["observed"]
    assert plain["candidates"].region("body").tsrs == handled["candidates"].region("body").tsrs
    for name, pick in picks.items():
        assert pick["grasp"]["state"] == "succeeded", (name, pick["grasp"]["stages"])
        reached = pick["pregrasp_region"]
        assert reached and all(label.startswith("body/") for label in reached), (name, reached)
        rest_z = SURFACE_Z + MUG_HEIGHT / 2
        assert pick["lifted_z"] > rest_z + 0.03, (name, pick["lifted_z"])  # it rose with the hand

    # edit: another handle changes the handle's candidates only; the scenes are untouched.
    observed = plain["observed"]
    wider = dataclasses.replace(HANDLE, primitive=Primitive("box", (0.030, 0.014, 0.060)))
    edited = grasp_regions(
        _mug("mug", observed, wider), PROFILE, sstsr.SstsrTemplates(HAND), approach_m=STANDOFF
    )
    before = plain["candidates"]
    assert edited.region("body").tsrs == before.region("body").tsrs
    assert edited.region("handle").tsrs != before.region("handle").tsrs
    assert {name: _sha(path) for name, path in scenes.items()} == hashes

    # instances: two mugs, two poses, two candidate sets, each placed at its own mug.
    turned = math.radians(70)
    second = Pose(
        position=(0.20, -0.55, observed.position[2]),
        quat_wxyz=(math.cos(turned / 2), 0.0, 0.0, math.sin(turned / 2)),
    )
    instances = {
        name: grasp_regions(
            _mug(name, pose), PROFILE, sstsr.SstsrTemplates(HAND), approach_m=STANDOFF
        )
        for name, pose in (("mug_a", observed), ("mug_b", second))
    }
    assert instances["mug_a"].tsrs != instances["mug_b"].tsrs
    for name, pose in (("mug_a", observed), ("mug_b", second)):
        candidates = instances[name]
        assert candidates.provenance["object"] == name
        for region in (BODY, HANDLE):
            frame = candidates.region(region.name).frame
            expected = compose(pose, region.origin)
            assert all(
                math.isclose(a, b, abs_tol=1e-12)
                for a, b in zip(
                    (*frame.position, *frame.quat_wxyz),
                    (*expected.position, *expected.quat_wxyz),
                    strict=True,
                )
            ), (name, region.name)

    report = {
        "environment": _environment(),
        "scenes": hashes,
        "collider": {
            name: {
                "candidates": pick["candidates"].provenance,
                "labels": list(pick["candidates"].labels),
                "grasp": pick["grasp"]["state"],
                "pregrasp_region": pick["pregrasp_region"],
                "lifted_z": pick["lifted_z"],
            }
            for name, pick in picks.items()
        },
        "edit": edited.provenance,
        "instances": {name: c.provenance for name, c in instances.items()},
    }
    (artifacts / "regions.json").write_text(json.dumps(report, indent=2) + "\n")
