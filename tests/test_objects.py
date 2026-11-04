"""Grasp regions from explicit object geometry and a selected gripper profile (#3, #46).

Scenarios, recorded in ``objects.json`` and ``framed.json`` with each request's regions
and provenance:
- ``frames``, for each hand: the primitives sstsr verifies for it (a cylinder, a box, and
  a sphere for the Robotiq 2F-140; a cylinder for the Kinova JACO2), each the one region
  of an object at a translated pose rotated about a tilted axis. Every region
  ``grasp_regions`` makes through ``ssgrasping.sstsr`` equals, origin, offset, and
  bounds, the one sstsr's own ``instantiate`` makes at the primitive's bottom face
  (cylinder and box) or centre (sphere), backed off along the approach, and poses
  sampled from each lie in the other. That includes the templates rolled 180 degrees,
  whose quaternions cannot be read from the rotation matrix's antisymmetric part.
- ``framed``: prl_assets' mug, its ``body`` cylinder and ``handle`` box each offset from
  the mug's frame and the handle turned about z, on the same tilted pose. Each region's
  candidates equal sstsr's at ``T_world_object @ T_object_region``, then sstsr's own
  frame convention, and nothing else: no second half-height shift. Their labels tell the
  two sets apart. With the JACO2 the body is wider than the aperture while the handle
  has candidates: outcomes are per region.
- ``no_regions``: a 0.20 m box is wider than the 2F-140's 0.128 m aperture, and the
  JACO2 hand's grasps are verified for cylinders only: both are ordinary results with a
  reason and no regions, before anything is planned.
- ``refused``: malformed geometry, regions, profiles, and requests fail at construction.
The end-to-end grasps through this API are ``test_grasp_place``'s and ``test_regions``'.

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
    GraspRegion,
    GraspSpec,
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
HANDS: dict[str, tuple[Any, tuple[str, ...], dict[str, tuple[float, ...]]]] = {
    # Every primitive sstsr verifies for each hand: the JACO2's are cylinders only.
    "Robotiq2F140": (Robotiq2F140, ("cylinder", "box", "sphere"), CASES),
    "KinovaJaco2TwoFinger": (KinovaJaco2TwoFinger, ("cylinder",), {"cylinder": (0.030, 0.095)}),
}
_SSTSR = {
    "cylinder": ("grasp_cylinder", ("cylinder_radius", "cylinder_height")),
    "box": ("grasp_box", ("box_x", "box_y", "box_z")),
    "sphere": ("grasp_sphere", ("object_radius",)),
}
_SSTSR_STRATEGY = {
    ("cylinder", "side"): "grasp_cylinder_side",
    ("cylinder", "top"): "grasp_cylinder_top",
    ("box", "face_x"): "grasp_box_face_x",
    ("box", "face_y"): "grasp_box_face_y",
}
# prl_assets' mug (objects/ycbv/mug/meta.yaml, grasp_regions version 1), its regions'
# frames as quaternions of the rotations it declares.
_HANDLE_YAW = math.atan2(-0.04840964125460411, 0.9988275660160769)
MUG = (
    GraspRegion(
        "body",
        Primitive("cylinder", (0.039, 0.074)),
        Pose(
            position=(-0.011774273067372281, -0.00016874629554637638, 0.0),
            quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        ),
        ("side", "top"),
    ),
    GraspRegion(
        "handle",
        Primitive("box", (0.028, 0.012, 0.056)),
        Pose(
            position=(0.045, -0.0031024567164179098, 0.0),
            quat_wxyz=(math.cos(_HANDLE_YAW / 2), 0.0, 0.0, math.sin(_HANDLE_YAW / 2)),
        ),
        ("face_x", "face_y"),
    ),
)


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


def _one(name: str, primitive: Primitive, pose: Pose = POSE, **options: Any) -> GraspObject:
    """An object of one shape: one region, ``body``, at the object's frame."""
    strategies = options.pop("strategies", None)
    return GraspObject(
        name, pose, (GraspRegion("body", primitive, strategies=strategies),), **options
    )


def _oracle(
    templates: Any, kind: str, dimensions: tuple[float, ...], region_frame: np.ndarray
) -> dict[str, Any]:
    """sstsr's regions instantiated at its own reference frame in ``region_frame`` (the
    bottom face of a cylinder or box, a sphere's centre), backed off by the approach, by
    template name."""
    reference = region_frame
    if kind != "sphere":
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


def _agrees(families: Any, tsrs: Any, expected: dict[str, Any], rng: Any) -> float:
    """The largest difference between our regions and sstsr's, by family, after checking
    poses sampled from each lie in the other."""
    assert sorted(families) == sorted(expected) and expected
    worst = 0.0
    for family, ours in zip(families, tsrs, strict=True):
        theirs = expected[family]
        mine = _as_sstsr(ours)
        worst = max(
            worst,
            float(np.abs(mine.T0_w - theirs.T0_w).max()),
            float(np.abs(mine.Tw_e - theirs.Tw_e).max()),
            float(np.abs(mine.Bw - theirs.Bw).max()),
        )
        for _ in range(5):
            assert mine.contains(theirs.sample(rng=rng)), family
            assert theirs.contains(mine.sample(rng=rng)), family
    return worst


def _summary(r: Any) -> list[dict[str, Any]]:
    return [
        {
            "origin": [list(t.origin.position), list(t.origin.quat_wxyz)],
            "offset": [list(t.offset.position), list(t.offset.quat_wxyz)],
            "position_bounds": [list(b) for b in t.position_bounds],
            "rotation_bounds": [list(b) for b in t.rotation_bounds],
        }
        for t in r.tsrs
    ]


@pytest.mark.parametrize("hand_name", sorted(HANDS))
def test_regions_are_sstsrs_own_at_any_object_pose(artifacts: Path, hand_name: str) -> None:
    make, kinds, cases = HANDS[hand_name]
    hand = make()
    report: dict[str, Any] = {}
    rng = np.random.default_rng(3)
    for kind in kinds:
        dimensions = cases[kind]
        regions = grasp_regions(
            _one(kind, Primitive(kind, dimensions), revision=7),
            _profile(hand),
            sstsr.SstsrTemplates(hand),
            approach_m=APPROACH,
        )
        assert regions.ok, regions.provenance
        name, arguments = _SSTSR[kind]
        templates = getattr(hand, name)(**dict(zip(arguments, dimensions, strict=True)))
        expected = _oracle(templates, kind, dimensions, _transform(POSE))
        worst = _agrees(regions.families, regions.tsrs, expected, rng)
        assert worst < 1e-9, (kind, worst)
        body = regions.region("body")
        assert set(body.tsr_strategies) == set(body.region.selected), kind
        assert regions.labels == tuple(f"body/{s}" for s in body.tsr_strategies)
        assert hash(regions) == hash(dataclasses.replace(regions)), kind
        provenance = regions.provenance
        assert (provenance["revision"], provenance["hand"]) == (7, hand_name)
        assert provenance["end_effector"] == "right_hand"
        assert provenance["approach_m"] == APPROACH
        report[kind] = {
            "provenance": provenance,
            "families": list(regions.families),
            "max_difference": worst,
            "regions": _summary(regions),
        }
    (artifacts / "objects.json").write_text(json.dumps(report, indent=2) + "\n")


def test_framed_regions_compose_the_object_and_region_frames(artifacts: Path) -> None:
    hand = Robotiq2F140()
    mug = GraspObject("mug", POSE, MUG, revision=4)
    regions = grasp_regions(mug, _profile(hand), sstsr.SstsrTemplates(hand), approach_m=APPROACH)
    assert regions.ok and all(r.ok for r in regions.regions), regions.provenance
    rng = np.random.default_rng(46)
    report: dict[str, Any] = {"provenance": regions.provenance, "regions": {}}
    for region in MUG:
        candidates = regions.region(region.name)
        frame = _transform(POSE) @ _transform(region.origin)
        assert np.allclose(_transform(candidates.frame), frame, atol=1e-12), region.name
        kind, dimensions = region.primitive.kind, region.primitive.dimensions
        args = dict(zip(_SSTSR[kind][1], dimensions, strict=True))
        templates = [
            t
            for s in region.strategies or ()
            for t in getattr(hand, _SSTSR_STRATEGY[(kind, s)])(**args)
        ]
        worst = _agrees(
            candidates.families, candidates.tsrs, _oracle(templates, kind, dimensions, frame), rng
        )
        assert worst < 1e-9, (region.name, worst)
        # Each label is its region and one of its strategies (sstsr makes no face_y grasp of
        # a handle this small, by its own limits).
        allowed = {f"{region.name}/{s}" for s in region.strategies or ()}
        assert candidates.labels and set(candidates.labels) <= allowed, candidates.labels
        report["regions"][region.name] = {"max_difference": worst, "tsrs": _summary(candidates)}
    # The two regions' candidates are told apart by label, and a spec carries the labels.
    labels = regions.labels
    assert len(labels) == len(regions.tsrs) and {lab.split("/")[0] for lab in labels} == {
        "body",
        "handle",
    }
    spec = regions.spec(support="table", normal=(0.0, 0.0, 1.0), lift_m=0.05)
    assert spec.pregrasp_labels == labels and tuple(spec.pregrasps) == regions.tsrs
    # Each label is paired with a region of its own region's candidates.
    assert spec.pregrasp_labels is not None
    for label, pregrasp in zip(spec.pregrasp_labels, spec.pregrasps, strict=True):
        assert pregrasp in regions.region(label.split("/")[0]).tsrs, label
    # The caller's order of regions is the candidates' order.
    reordered = grasp_regions(
        mug,
        _profile(hand),
        sstsr.SstsrTemplates(hand),
        approach_m=APPROACH,
        regions=("handle", "body"),
    )
    assert [r.region.name for r in reordered.regions] == ["handle", "body"]
    assert reordered.labels[0].startswith("handle/")
    # Only the selected region is made.
    handle_only = grasp_regions(
        mug, _profile(hand), sstsr.SstsrTemplates(hand), approach_m=APPROACH, regions=("handle",)
    )
    assert [r.region.name for r in handle_only.regions] == ["handle"]
    assert handle_only.tsrs == regions.region("handle").tsrs
    # Outcomes are per region: the JACO2 cannot open around the body, but can grasp the
    # handle across its 12 mm. Its grasps are verified for cylinders only, so the handle is
    # given to it as a 12 mm cylinder.
    jaco = KinovaJaco2TwoFinger()
    jaco_mug = GraspObject(
        "mug",
        POSE,
        (
            MUG[0],
            dataclasses.replace(
                MUG[1], primitive=Primitive("cylinder", (0.006, 0.056)), strategies=("side",)
            ),
        ),
    )
    per_region = grasp_regions(
        jaco_mug, _profile(jaco), sstsr.SstsrTemplates(jaco), approach_m=APPROACH
    )
    assert per_region.region("body").reason is GeometryReason.WIDER_THAN_APERTURE
    assert per_region.region("handle").ok and per_region.ok
    assert per_region.tsrs == per_region.region("handle").tsrs
    report["per_region"] = per_region.provenance
    (artifacts / "framed.json").write_text(json.dumps(report, indent=2) + "\n")


def test_unreachable_geometry_is_an_ordinary_result(artifacts: Path) -> None:
    wide = grasp_regions(
        _one("crate", Primitive("box", (0.20, 0.30, 0.25))),
        _profile(Robotiq2F140()),
        sstsr.SstsrTemplates(Robotiq2F140()),
        approach_m=APPROACH,
    )
    crate = wide.region("body")
    assert not wide.ok and crate.reason is GeometryReason.WIDER_THAN_APERTURE and wide.tsrs == ()
    assert "0.2000 m" in crate.detail and "0.1280 m" in crate.detail, crate.detail
    jaco = KinovaJaco2TwoFinger()
    unsupported = grasp_regions(
        _one("block", Primitive("box", (0.03, 0.04, 0.05))),
        _profile(jaco),
        sstsr.SstsrTemplates(jaco),
        approach_m=APPROACH,
    )
    assert unsupported.region("body").reason is GeometryReason.UNSUPPORTED_BY_HAND
    assert unsupported.tsrs == ()
    cylinder = grasp_regions(
        _one("cup", Primitive("cylinder", (0.03, 0.09)), strategies=("side",)),
        _profile(jaco),
        sstsr.SstsrTemplates(jaco),
        approach_m=APPROACH,
    )
    assert cylinder.ok and cylinder.provenance["hand"] == "KinovaJaco2TwoFinger"
    # A flat box's thinnest extent is no help to a top grasp, which closes across x or y.
    flat = grasp_regions(
        _one("tray", Primitive("box", (0.20, 0.20, 0.02)), strategies=("top",)),
        _profile(Robotiq2F140()),
        sstsr.SstsrTemplates(Robotiq2F140()),
        approach_m=APPROACH,
    )
    assert flat.region("body").reason is GeometryReason.WIDER_THAN_APERTURE
    # A hand without verified box grasps is reported as such, whatever the box's width.
    for size in ((0.03, 0.04, 0.05), (0.08, 0.09, 0.10)):
        jaco_box = grasp_regions(
            _one("block", Primitive("box", size)),
            _profile(jaco),
            sstsr.SstsrTemplates(jaco),
            approach_m=APPROACH,
        )
        assert jaco_box.region("body").reason is GeometryReason.UNSUPPORTED_BY_HAND, size
    # Each strategy is checked against the spans it closes across: a 0.05 x 0.20 x 0.20 m
    # box's top and y-face grasps close across 0.05 m, its x-face grasps across 0.20 m.
    slab = grasp_regions(
        _one("slab", Primitive("box", (0.05, 0.20, 0.20))),
        _profile(Robotiq2F140()),
        sstsr.SstsrTemplates(Robotiq2F140()),
        approach_m=APPROACH,
    )
    slab_body = slab.region("body")
    assert slab_body.ok and slab_body.dropped == ("face_x",), slab_body.dropped
    assert "face_x" not in slab_body.tsr_strategies and slab_body.tsrs
    assert slab.complete and not wide.complete
    # Dimensions from numerical code are accepted as the numbers they are.
    single = tuple(np.array([0.033, 0.123], dtype=np.float32))
    assert np.allclose(Primitive("cylinder", single).dimensions, (0.033, 0.123), atol=1e-6)
    with pytest.raises(ValueError, match="no pregrasp regions"):
        wide.spec(support="table", normal=(0.0, 0.0, 1.0), lift_m=0.05)
    (artifacts / "no_regions.json").write_text(
        json.dumps(
            {r.object.name: r.provenance["regions"] for r in (wide, unsupported, flat)},
            indent=2,
        )
        + "\n"
    )


IDENTITY = Pose(position=(0.0, 0.0, 0.0), quat_wxyz=(1.0, 0.0, 0.0, 0.0))
CAN = Primitive("cylinder", (0.033, 0.123))
BODY = GraspRegion("body", CAN)


def _pregrasps(labels: Any) -> GraspSpec:
    one = grasp_regions(
        _one("can", CAN),
        _profile(Robotiq2F140()),
        sstsr.SstsrTemplates(Robotiq2F140()),
        approach_m=APPROACH,
    )
    return dataclasses.replace(
        one.spec(support="table", normal=(0.0, 0.0, 1.0), lift_m=0.05), pregrasp_labels=labels
    )


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
        lambda: GraspRegion("Body", CAN),
        lambda: GraspRegion("2nd", CAN),
        lambda: GraspRegion("body", CAN, strategies=("face_x",)),
        lambda: GraspRegion("body", CAN, strategies=()),
        lambda: GraspRegion("body", CAN, strategies="side"),  # type: ignore[arg-type]
        lambda: GraspRegion(
            "body", CAN, Pose(position=(math.nan, 0.0, 0.0), quat_wxyz=(1.0, 0.0, 0.0, 0.0))
        ),
        lambda: GraspObject("", IDENTITY, (BODY,)),
        lambda: GraspObject("can", IDENTITY, ()),
        lambda: GraspObject("can", IDENTITY, (BODY, BODY)),
        lambda: GraspObject("can", IDENTITY, (CAN,)),  # type: ignore[arg-type]
        lambda: GraspObject("can", IDENTITY, (BODY,), frame="right_hand"),
        lambda: GraspObject("can", IDENTITY, (BODY,), revision=True),
        lambda: Pose(position=(0.0, 0.0, 0.0), quat_wxyz=(1.0, 1.0, 0.0, 0.0)),
        lambda: GraspingProfile("hand", "arm", "ee", "gripper", "opening", 0.0),
        lambda: GraspingProfile("hand", "arm", "", "gripper", "opening", 0.1),
        lambda: grasp_regions(
            GraspObject("can", IDENTITY, (BODY,)),
            _profile(Robotiq2F140()),
            sstsr.SstsrTemplates(Robotiq2F140()),
            approach_m=0.0,
        ),
        lambda: grasp_regions(
            GraspObject("can", IDENTITY, (BODY,)),
            _profile(Robotiq2F140()),
            sstsr.SstsrTemplates(Robotiq2F140()),
            approach_m=APPROACH,
            regions=("handle",),
        ),
        lambda: grasp_regions(
            GraspObject("can", IDENTITY, (BODY,)),
            _profile(Robotiq2F140()),
            sstsr.SstsrTemplates(Robotiq2F140()),
            approach_m=APPROACH,
            regions="body",
        ),
        lambda: _pregrasps(("body/side",)),
        lambda: _pregrasps(""),
    ],
)
def test_malformed_input_is_refused_at_construction(build: Any) -> None:
    with pytest.raises(ValidationError):
        build()
