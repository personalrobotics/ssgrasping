"""Grasp templates from sstsr's hands, for ``ssgrasping.objects``.

Requires the ``sstsr`` extra (``ssgrasping[sstsr]``). ``import ssgrasping`` never
imports this module or sstsr.

Templates are generated through sstsr's ``default_registry``, which records the
primitives each hand's grasps have been verified for: a primitive the registry does not
list for the hand (the Kinova JACO2's boxes and spheres) is ``UnsupportedPrimitive``,
not a set of unverified regions. sstsr frames cylinders and boxes at their bottom face
and spheres at their centre; each template is moved here into the primitive's centred
frame, so callers never shift a pose by half a height.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from ssrobot import Pose
from tsr import KinovaJaco2TwoFinger, Robotiq2F85, Robotiq2F140
from tsr.hands.registry import default_registry

from ssgrasping.objects import GraspingProfile, Primitive, RegionTemplate, UnsupportedPrimitive

REGISTRY_KEYS: dict[type, str] = {
    Robotiq2F140: "robotiq_2f140",
    Robotiq2F85: "robotiq_2f85",
    KinovaJaco2TwoFinger: "kinova_jaco2_two_finger",
}
"""sstsr's registry name for each of its hands that has one. Another hand, such as
``FrankaHand``, needs its registry name given explicitly."""

_FACTORY_ARGUMENTS = {
    "cylinder": ("cylinder_radius", "cylinder_height"),
    "box": ("box_x", "box_y", "box_z"),
    "sphere": ("object_radius",),
}
_FACTORIES = {
    ("cylinder", "side"): "grasp_cylinder_side",
    ("cylinder", "top"): "grasp_cylinder_top",
    ("cylinder", "bottom"): "grasp_cylinder_bottom",
    ("box", "top"): "grasp_box_top",
    ("box", "bottom"): "grasp_box_bottom",
    ("box", "face_x"): "grasp_box_face_x",
    ("box", "face_y"): "grasp_box_face_y",
    ("sphere", "all"): "grasp_sphere",
}
"""The sstsr factory for each strategy: together, a primitive's are its registered
generator (``grasp_cylinder`` is side, top, and bottom)."""


def pose(T: Any) -> Pose:
    """A 4x4 homogeneous transform as an ssrobot ``Pose``.

    Shepperd's method: the quaternion is recovered from the largest of its four
    components, so it is exact for every rotation. Recovering each component's sign from
    off-diagonal differences instead fails at 180 degrees, where those differences
    vanish: half of sstsr's side grasps are rolled 180 degrees."""
    T = np.asarray(T, dtype=float)
    R = T[:3, :3]
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > max(R[0, 0], R[1, 1], R[2, 2]):
        s = 2.0 * math.sqrt(1.0 + trace)
        q = (s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    elif R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = ((R[2, 1] - R[1, 2]) / s, s / 4, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s)
    elif R[1, 1] >= R[2, 2]:
        s = 2.0 * math.sqrt(1.0 - R[0, 0] + R[1, 1] - R[2, 2])
        q = ((R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, s / 4, (R[1, 2] + R[2, 1]) / s)
    else:
        s = 2.0 * math.sqrt(1.0 - R[0, 0] - R[1, 1] + R[2, 2])
        q = ((R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, s / 4)
    n = math.sqrt(sum(c * c for c in q))
    w, x, y, z = (float(c) / n for c in q)
    p = T[:3, 3]
    return Pose(position=(float(p[0]), float(p[1]), float(p[2])), quat_wxyz=(w, x, y, z))


def matrix(p: Pose) -> np.ndarray:
    """An ssrobot ``Pose`` as a 4x4 homogeneous transform."""
    w, x, y, z = p.quat_wxyz
    T = np.eye(4)
    T[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
    T[:3, 3] = p.position
    return T


class SstsrTemplates:
    """A ``TemplateSource`` from an sstsr hand: the grasps of each selected strategy,
    in the primitive's centred frame, for the primitives sstsr's registry lists for the
    hand."""

    def __init__(self, hand: Any, *, registry_key: str | None = None) -> None:
        key = registry_key if registry_key is not None else REGISTRY_KEYS.get(type(hand))
        if key is None:
            raise ValueError(
                f"{type(hand).__name__} has no sstsr registry name here; pass registry_key"
            )
        self.hand = hand
        self.registry_key = key

    def __call__(self, primitive: Primitive, strategies: Sequence[str]) -> Sequence[RegionTemplate]:
        if (self.registry_key, primitive.kind, "grasp") not in default_registry:
            raise UnsupportedPrimitive(
                f"sstsr registers no {primitive.kind} grasp for {self.registry_key}"
            )
        arguments = dict(zip(_FACTORY_ARGUMENTS[primitive.kind], primitive.dimensions, strict=True))
        # sstsr's reference frame for a cylinder or box is its bottom face, half the
        # height below the centred frame along z; for a sphere it is the centre.
        shift = np.eye(4)
        if primitive.kind in ("cylinder", "box"):
            shift[2, 3] = -primitive.dimensions[-1] / 2
        templates = []
        generated = [
            t
            for s in strategies
            for t in getattr(self.hand, _FACTORIES[(primitive.kind, s)])(**arguments)
        ]
        for t in generated:
            bounds = [(float(lo), float(hi)) for lo, hi in t.Bw]
            templates.append(
                RegionTemplate(
                    family=t.name,
                    origin=pose(shift @ t.T_ref_tsr),
                    offset=pose(t.Tw_e),
                    position_bounds=(bounds[0], bounds[1], bounds[2]),
                    rotation_bounds=(bounds[3], bounds[4], bounds[5]),
                )
            )
        return templates


def profile(
    hand: Any, *, manipulator: str, end_effector: str, gripper: str, opening_channel: str
) -> GraspingProfile:
    """A ``GraspingProfile`` for an sstsr hand: its class as the hand's identity and its
    ``max_aperture`` as the aperture."""
    return GraspingProfile(
        hand=type(hand).__name__,
        manipulator=manipulator,
        end_effector=end_effector,
        gripper=gripper,
        opening_channel=opening_channel,
        max_aperture_m=float(hand.max_aperture),
    )


__all__ = ["REGISTRY_KEYS", "SstsrTemplates", "matrix", "pose", "profile"]
