"""ADA's candidates bind its palm, through the same geometry/profile API (#3).

ADA's bare-hand package (``ada_assets.models``, the arm without the articutool) is
selected with sstsr's ``KinovaJaco2TwoFinger``. That the adapter's JACO2 cylinder regions
equal sstsr's own is ``test_objects``' oracle, run for this hand too. What is ADA's,
recorded in ``ada_candidates.json``:
- ``frame``: ADA's end effector, ``hand``, is ``palm_grasp_site``, and in the package's
  compiled model it sits on ``j2n6s200_link_6`` at the palm sstsr frames the JACO2's
  templates in: ``link_6 + [-0.0025, 0, -0.125]``, turned 180 degrees about x (z the
  approach, y the finger opening). Regions bound to another frame would be wrong however
  correct their mathematics.
- ``candidates``: a 60 mm cup, through a profile made from ADA's own names, has regions;
  an 80 mm jar is wider than the JACO2's 73 mm aperture, an ordinary result.

This is candidate generation only. It does not show that ADA picks the cup, nor that its
grasps can be observed acquired or retained; those are separate work.

Reproduce with ``uv run pytest tests/test_ada_candidates.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from ssrobot import Pose, load_installed_package
from tsr import KinovaJaco2TwoFinger

from ssgrasping import GeometryReason, GraspObject, Primitive, grasp_regions, sstsr

POSE = Pose(position=(0.30, -0.22, 0.78), quat_wxyz=(1.0, 0.0, 0.0, 0.0))


def test_ada_candidates_bind_its_palm(artifacts: Path) -> None:
    package = load_installed_package("ada_assets.models")
    description = package.description
    (end_effector,) = description.end_effectors
    (manipulator,) = description.manipulators
    assert end_effector.gripper is not None
    manifest = package.manifest
    (canonical,) = [m for m in manifest.models if m.name == manifest.canonical_model]
    model = mujoco.MjModel.from_xml_path(str(package.root / canonical.path))
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, end_effector.frame)
    parent = model.body(model.site_bodyid[site]).name
    palm = model.site_pos[site].tolist(), model.site_quat[site].tolist()

    hand = KinovaJaco2TwoFinger()
    # The palm sstsr's JACO2 templates are framed in, in link_6 (its class docstring).
    expected = ([hand.PALM_X_IN_LINK_6, 0.0, -hand.PALM_OFFSET_FROM_LINK_6], [0.0, 1.0, 0.0, 0.0])
    assert end_effector.frame == "palm_grasp_site" and parent == "j2n6s200_link_6"
    assert np.allclose(palm[0], expected[0], atol=1e-9), palm
    assert np.isclose(abs(float(np.dot(palm[1], expected[1]))), 1.0, atol=1e-9), palm

    profile = sstsr.profile(
        hand,
        manipulator=manipulator.name,
        end_effector=end_effector.name,
        gripper=end_effector.gripper,
        opening_channel="gripper_opening",
    )
    cup, jar = (
        grasp_regions(
            GraspObject(name, Primitive("cylinder", size), POSE),
            profile,
            sstsr.SstsrTemplates(hand),
            approach_m=0.05,
        )
        for name, size in (("cup", (0.030, 0.095)), ("jar", (0.040, 0.10)))
    )
    assert cup.ok and cup.tsrs, cup.detail
    assert jar.reason is GeometryReason.WIDER_THAN_APERTURE and jar.tsrs == (), jar.detail

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
        "cup": {"provenance": cup.provenance, "regions": len(cup.tsrs)},
        "jar": {"reason": jar.reason, "detail": jar.detail},
    }
    (artifacts / "ada_candidates.json").write_text(json.dumps(report, indent=2) + "\n")
