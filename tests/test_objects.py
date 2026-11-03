"""Grasp regions from explicit object geometry and a selected gripper profile (#3).

Scenarios, recorded in ``objects.json`` with each request's regions and provenance:
- ``frames``: a cylinder, a box, and a sphere at a translated pose rotated about a tilted
  axis. Every region ``grasp_regions`` makes through ``ssgrasping.sstsr`` equals the one
  sstsr's own ``instantiate`` makes at the primitive's bottom face (cylinder and box) or
  centre (sphere), backed off along the approach, and poses sampled from each lie in the
  other. That includes the templates rolled 180 degrees, whose quaternions cannot be
  read from the rotation matrix's antisymmetric part.
- ``no_regions``: a 0.20 m box is wider than the 2F-140's 0.128 m aperture, and the
  JACO2 hand's grasps are verified for cylinders only: both are ordinary results with a
  reason and no regions, before anything is planned.
- ``refused``: malformed geometry, profiles, and requests fail at construction.
The end-to-end grasp through this API is ``test_grasp_place``'s.

Reproduce with ``uv run pytest tests/test_objects.py``.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tsr
from scipy.spatial.transform import Rotation
from ssrobot import Pose, ValidationError
from tsr import KinovaJaco2TwoFinger, Robotiq2F140

from ssgrasping import (
    GeometryReason,
    GraspingProfile,
    GraspObject,
    Primitive,
    grasp_regions,
    sstsr,
)

APPROACH = 0.08
# Translated, and rotated 0.9 rad about a tilted axis: no axis of the object frame is
# aligned with the world's.
_AXIS = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
POSE = Pose(
    position=(0.42, -0.31, 0.87),
    quat_wxyz=(math.cos(0.45), *(math.sin(0.45) * _AXIS)),
)
CASES = {
    "cylinder": (0.033, 0.123),
    "box": (0.05, 0.07, 0.11),
    "sphere": (0.035,),
}
_SSTSR = {
    "cylinder": ("grasp_cylinder", ("cylinder_radius", "cylinder_height")),
    "box": ("grasp_box", ("box_x", "box_y", "box_z")),
    "sphere": ("grasp_sphere", ("object_radius",)),
}


def _transform(p: Pose) -> np.ndarray:
    """``p`` as a 4x4 transform by scipy, independently of ``ssgrasping.sstsr``'s own
    conversions, which the oracle must not share."""
    T = np.eye(4)
    w, x, y, z = p.quat_wxyz
    T[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    T[:3, 3] = p.position
    return T


def _profile(hand: Any) -> GraspingProfile:
    return sstsr.profile(
        hand,
        manipulator="right_arm",
        end_effector="right_hand",
        gripper="right_gripper",
        opening_channel="right_gripper_opening",
    )


def _expected(kind: str, dimensions: tuple[float, ...]) -> dict[str, Any]:
    """sstsr's regions for the primitive at ``POSE``, instantiated at sstsr's own
    reference frame and backed off by the approach, by template name: the independent
    oracle."""
    name, arguments = _SSTSR[kind]
    templates = getattr(Robotiq2F140(), name)(**dict(zip(arguments, dimensions, strict=True)))
    reference = _transform(POSE)
    if kind != "sphere":  # sstsr frames cylinders and boxes at their bottom face
        reference = reference @ np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -dimensions[-1] / 2], [0, 0, 0, 1]]
        )
    back = np.eye(4)
    back[2, 3] = -APPROACH
    return {
        t.name: tsr.TSR(T0_w=reference @ t.T_ref_tsr, Tw_e=t.Tw_e @ back, Bw=t.Bw)
        for t in templates
    }


def _as_sstsr(region: Any) -> Any:
    bounds = np.array([*region.position_bounds, *region.rotation_bounds])
    return tsr.TSR(T0_w=_transform(region.origin), Tw_e=_transform(region.offset), Bw=bounds)


def test_regions_are_sstsrs_own_at_any_object_pose(artifacts: Path) -> None:
    hand = Robotiq2F140()
    report: dict[str, Any] = {}
    rng = np.random.default_rng(3)
    for kind, dimensions in CASES.items():
        regions = grasp_regions(
            GraspObject(kind, Primitive(kind, dimensions), POSE, revision=7),
            _profile(hand),
            sstsr.SstsrTemplates(hand),
            approach_m=APPROACH,
        )
        assert regions.ok, regions.detail
        expected = _expected(kind, dimensions)
        assert sorted(regions.families) == sorted(expected) and expected, kind
        worst = 0.0
        for family, ours in zip(regions.families, regions.tsrs, strict=True):
            theirs = expected[family]
            mine = _as_sstsr(ours)
            worst = max(
                worst,
                float(np.abs(mine.T0_w - theirs.T0_w).max()),
                float(np.abs(mine.Tw_e - theirs.Tw_e).max()),
            )
            for _ in range(5):
                assert mine.contains(theirs.sample(rng=rng)), (kind, ours)
                assert theirs.contains(mine.sample(rng=rng)), (kind, ours)
        assert worst < 1e-9, (kind, worst)
        report[kind] = {
            "provenance": regions.provenance,
            "families": list(regions.families),
            "max_transform_difference": worst,
            "regions": [
                {
                    "origin": [list(r.origin.position), list(r.origin.quat_wxyz)],
                    "offset": [list(r.offset.position), list(r.offset.quat_wxyz)],
                    "position_bounds": [list(b) for b in r.position_bounds],
                    "rotation_bounds": [list(b) for b in r.rotation_bounds],
                }
                for r in regions.tsrs
            ],
        }
        assert set(regions.region_strategies) == set(regions.strategies), kind
        assert hash(regions) == hash(dataclasses.replace(regions)), kind
        assert regions.provenance["revision"] == 7
        assert regions.provenance["hand"] == "Robotiq2F140"
        assert regions.provenance["end_effector"] == "right_hand"
        assert regions.provenance["approach_m"] == APPROACH
    (artifacts / "objects.json").write_text(json.dumps(report, indent=2) + "\n")


def test_unreachable_geometry_is_an_ordinary_result(artifacts: Path) -> None:
    wide = grasp_regions(
        GraspObject("crate", Primitive("box", (0.20, 0.30, 0.25)), POSE),
        _profile(Robotiq2F140()),
        sstsr.SstsrTemplates(Robotiq2F140()),
        approach_m=APPROACH,
    )
    assert wide.reason is GeometryReason.WIDER_THAN_APERTURE and wide.tsrs == ()
    assert "0.2000 m" in wide.detail and "0.1280 m" in wide.detail, wide.detail
    jaco = KinovaJaco2TwoFinger()
    unsupported = grasp_regions(
        GraspObject("block", Primitive("box", (0.03, 0.04, 0.05)), POSE),
        _profile(jaco),
        sstsr.SstsrTemplates(jaco),
        approach_m=APPROACH,
    )
    assert unsupported.reason is GeometryReason.UNSUPPORTED_BY_HAND and unsupported.tsrs == ()
    cylinder = grasp_regions(
        GraspObject("cup", Primitive("cylinder", (0.03, 0.09)), POSE),
        _profile(jaco),
        sstsr.SstsrTemplates(jaco),
        approach_m=APPROACH,
        strategies=("side",),
    )
    assert cylinder.ok and cylinder.provenance["hand"] == "KinovaJaco2TwoFinger"
    # A flat box's thinnest extent is no help to a top grasp, which closes across x or y.
    flat = grasp_regions(
        GraspObject("tray", Primitive("box", (0.20, 0.20, 0.02)), POSE),
        _profile(Robotiq2F140()),
        sstsr.SstsrTemplates(Robotiq2F140()),
        approach_m=APPROACH,
        strategies=("top",),
    )
    assert flat.reason is GeometryReason.WIDER_THAN_APERTURE, flat.detail
    # A hand without verified box grasps is reported as such, whatever the box's width.
    for size in ((0.03, 0.04, 0.05), (0.08, 0.09, 0.10)):
        jaco_box = grasp_regions(
            GraspObject("block", Primitive("box", size), POSE),
            _profile(jaco),
            sstsr.SstsrTemplates(jaco),
            approach_m=APPROACH,
        )
        assert jaco_box.reason is GeometryReason.UNSUPPORTED_BY_HAND, (size, jaco_box.detail)
    # Dimensions from numerical code are accepted as the numbers they are.
    single = tuple(np.array([0.033, 0.123], dtype=np.float32))
    assert np.allclose(Primitive("cylinder", single).dimensions, (0.033, 0.123), atol=1e-6)
    with pytest.raises(ValueError, match="no pregrasp regions"):
        wide.spec(support="table", normal=(0.0, 0.0, 1.0), lift_m=0.05)
    (artifacts / "no_regions.json").write_text(
        json.dumps(
            {
                r.object.name: {"reason": r.reason, "detail": r.detail}
                for r in (wide, unsupported, flat)
            },
            indent=2,
        )
        + "\n"
    )


IDENTITY = Pose(position=(0.0, 0.0, 0.0), quat_wxyz=(1.0, 0.0, 0.0, 0.0))
CAN = Primitive("cylinder", (0.033, 0.123))


@pytest.mark.parametrize(
    "build",
    [
        lambda: Primitive("cone", (0.1, 0.2)),
        lambda: Primitive("cylinder", (0.033,)),
        lambda: Primitive("cylinder", (0.033, 0.123, 0.1)),
        lambda: Primitive("cylinder", (0.0, 0.123)),
        lambda: Primitive("cylinder", (-0.033, 0.123)),
        lambda: Primitive("cylinder", (math.nan, 0.123)),
        lambda: Primitive("cylinder", (math.inf, 0.123)),
        lambda: Primitive("cylinder", (True, 0.123)),
        lambda: Primitive("cylinder", ("0.033", 0.123)),  # type: ignore[arg-type]
        lambda: GraspObject("", CAN, IDENTITY),
        lambda: GraspObject("can", CAN, IDENTITY, frame="right_hand"),
        lambda: GraspObject("can", CAN, IDENTITY, revision=True),
        lambda: Pose(position=(0.0, 0.0, 0.0), quat_wxyz=(1.0, 1.0, 0.0, 0.0)),
        lambda: GraspingProfile("hand", "arm", "ee", "gripper", "opening", 0.0),
        lambda: GraspingProfile("hand", "arm", "", "gripper", "opening", 0.1),
        lambda: grasp_regions(
            GraspObject("can", CAN, IDENTITY),
            _profile(Robotiq2F140()),
            sstsr.SstsrTemplates(Robotiq2F140()),
            approach_m=0.0,
        ),
        lambda: grasp_regions(
            GraspObject("can", CAN, IDENTITY),
            _profile(Robotiq2F140()),
            sstsr.SstsrTemplates(Robotiq2F140()),
            approach_m=APPROACH,
            strategies=("face_x",),
        ),
        lambda: grasp_regions(
            GraspObject("can", CAN, IDENTITY),
            _profile(Robotiq2F140()),
            sstsr.SstsrTemplates(Robotiq2F140()),
            approach_m=APPROACH,
            strategies="side",
        ),
    ],
)
def test_malformed_input_is_refused_at_construction(build: Any) -> None:
    with pytest.raises(ValidationError):
        build()
