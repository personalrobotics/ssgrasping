"""The same geometry/profile API makes correct cylinder candidates for ADA (#3).

ADA's bare-hand package (``ada_assets.models``, the arm without the articutool) is
selected with sstsr's ``KinovaJaco2TwoFinger``. Recorded in ``ada_candidates.json``:
- ``frame``: ADA's end effector, ``hand``, is ``palm_grasp_site``, and in the compiled
  model it sits on ``j2n6s200_link_6`` at the palm sstsr's JACO2 templates are framed in:
  ``link_6 + [-0.0025, 0, -0.125]``, turned 180 degrees about x (z the approach, y the
  finger opening). Candidates bound to another frame would be wrong however correct
  their mathematics.
- ``candidates``: a 60 mm cup at a translated pose rotated about a tilted axis, through
  the profile built from ADA's own names. Every region equals the one sstsr instantiates
  for the JACO2 at the cup's bottom face, backed off along the approach, and poses
  sampled from each lie in the other.
- ``limits``: an 80 mm cylinder is wider than the JACO2's 73 mm aperture, and its box
  grasps are not verified by sstsr: both are ordinary results with a reason.

This is candidate generation only. It does not show that ADA picks the cup, nor that its
grasps can be observed acquired or retained; those are separate work.

Reproduce with ``uv run pytest tests/test_ada_candidates.py``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import tsr
from scipy.spatial.transform import Rotation
from ssrobot import Pose, RobotContext, load_installed_package
from ssrobot.mujoco import MujocoRuntime
from tsr import KinovaJaco2TwoFinger

from ssgrasping import GeometryReason, GraspObject, Primitive, grasp_regions, sstsr

APPROACH = 0.05
CUP = (0.030, 0.095)  # a 60 mm cup, 95 mm tall
_AXIS = np.array([-0.4, 0.2, 0.9]) / np.linalg.norm([-0.4, 0.2, 0.9])
_X, _Y, _Z = (float(c) for c in math.sin(0.6) * _AXIS)
POSE = Pose(position=(0.30, -0.22, 0.78), quat_wxyz=(math.cos(0.6), _X, _Y, _Z))


def _transform(p: Pose) -> np.ndarray:
    T = np.eye(4)
    w, x, y, z = p.quat_wxyz
    T[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    T[:3, 3] = p.position
    return T


def test_ada_cylinder_candidates_bind_its_palm(artifacts: Path) -> None:
    package = load_installed_package("ada_assets.models")
    description = package.description
    (end_effector,) = description.end_effectors
    (manipulator,) = description.manipulators
    runtime = MujocoRuntime(package)
    with RobotContext(description, runtime) as ctx:
        channels = list(ctx.info.channels)
        model = runtime._model
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, end_effector.frame)
        parent = model.body(model.site_bodyid[site]).name
        palm = model.site_pos[site].tolist(), model.site_quat[site].tolist()

    hand = KinovaJaco2TwoFinger()
    # The palm sstsr's JACO2 templates are framed in, in link_6 (its class docstring).
    expected = ([hand.PALM_X_IN_LINK_6, 0.0, -hand.PALM_OFFSET_FROM_LINK_6], [0.0, 1.0, 0.0, 0.0])
    assert end_effector.frame == "palm_grasp_site" and parent == "j2n6s200_link_6"
    assert np.allclose(palm[0], expected[0], atol=1e-9), palm
    assert np.allclose(np.abs(np.dot(palm[1], expected[1])), 1.0, atol=1e-9), palm

    assert "gripper_opening" in channels
    assert end_effector.gripper is not None
    profile = sstsr.profile(
        hand,
        manipulator=manipulator.name,
        end_effector=end_effector.name,
        gripper=end_effector.gripper,
        opening_channel="gripper_opening",
    )
    regions = grasp_regions(
        GraspObject("cup", Primitive("cylinder", CUP), POSE, revision=1),
        profile,
        sstsr.SstsrTemplates(hand),
        approach_m=APPROACH,
    )
    assert regions.ok, regions.detail

    bottom = _transform(POSE) @ np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -CUP[1] / 2], [0, 0, 0, 1]]
    )
    back = np.eye(4)
    back[2, 3] = -APPROACH
    oracle = {
        t.name: tsr.TSR(T0_w=bottom @ t.T_ref_tsr, Tw_e=t.Tw_e @ back, Bw=t.Bw)
        for t in hand.grasp_cylinder(cylinder_radius=CUP[0], cylinder_height=CUP[1])
    }
    assert sorted(regions.families) == sorted(oracle)
    rng = np.random.default_rng(1)
    worst = 0.0
    for family, region in zip(regions.families, regions.tsrs, strict=True):
        bounds = np.array([*region.position_bounds, *region.rotation_bounds])
        ours = tsr.TSR(T0_w=_transform(region.origin), Tw_e=_transform(region.offset), Bw=bounds)
        theirs = oracle[family]
        worst = max(
            worst,
            float(np.abs(ours.T0_w - theirs.T0_w).max()),
            float(np.abs(ours.Tw_e - theirs.Tw_e).max()),
        )
        for _ in range(5):
            assert ours.contains(theirs.sample(rng=rng)), family
            assert theirs.contains(ours.sample(rng=rng)), family
    assert worst < 1e-9, worst

    wide = grasp_regions(
        GraspObject("jar", Primitive("cylinder", (0.040, 0.10)), POSE),
        profile,
        sstsr.SstsrTemplates(hand),
        approach_m=APPROACH,
    )
    box = grasp_regions(
        GraspObject("block", Primitive("box", (0.03, 0.04, 0.05)), POSE),
        profile,
        sstsr.SstsrTemplates(hand),
        approach_m=APPROACH,
    )
    assert wide.reason is GeometryReason.WIDER_THAN_APERTURE, wide.detail
    assert box.reason is GeometryReason.UNSUPPORTED_BY_HAND, box.detail

    report: dict[str, Any] = {
        "package": "ada_assets.models",
        "frame": {"end_effector": end_effector.frame, "parent": parent, "pose_in_parent": palm},
        "profile": {
            "hand": profile.hand,
            "manipulator": profile.manipulator,
            "end_effector": profile.end_effector,
            "gripper": profile.gripper,
            "opening_channel": profile.opening_channel,
            "max_aperture_m": profile.max_aperture_m,
        },
        "candidates": {
            "provenance": regions.provenance,
            "families": list(regions.families),
            "strategies": list(regions.region_strategies),
            "max_transform_difference": worst,
        },
        "limits": {r.object.name: {"reason": r.reason, "detail": r.detail} for r in (wide, box)},
    }
    (artifacts / "ada_candidates.json").write_text(json.dumps(report, indent=2) + "\n")
