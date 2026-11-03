"""Grasp regions from explicit object geometry and an explicit gripper profile.

An application states what it is grasping and with what, as data: a ``GraspObject``
(a primitive approximation of the object and its observed pose) and a
``GraspingProfile`` (the hand and the robot's names for its manipulator, end effector,
gripper, and opening channel). ``grasp_regions`` turns them into the world pregrasp
regions a ``GraspSpec`` takes, through a ``TemplateSource`` that supplies the hand's
grasp templates in the object's frame; ``ssgrasping.sstsr`` is one, from sstsr's hands.
Nothing is inferred from names, meshes, or the robot.

Every primitive's frame has its origin at its geometric centre, with z the cylinder's
axis and the box's local z. A ``RegionTemplate`` is expressed in that frame, so a source
whose library frames primitives differently (sstsr frames cylinders and boxes at their
bottom face) converts once, in the source, and no caller shifts poses by half a height.

The pregrasp regions are backed off from the grasp along the end effector's approach
axis (its z) by ``approach_m``, the distance the grasp's approach then moves: the two
come from one number, so they cannot disagree.

A request that is well formed but cannot produce a region is an ordinary result, with a
reason, not an exception: an object wider than the hand opens, a primitive the hand's
templates do not cover, or templates that all fall outside the hand's limits.
Malformed input is refused at construction.
"""

from __future__ import annotations

import dataclasses
import enum
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ssrobot import Pose, Tsr

from ssgrasping._poses import compose
from ssgrasping._skill import check_spec
from ssgrasping.grasp import GraspSpec

DIMENSIONS = {
    "cylinder": ("radius", "height"),
    "box": ("x", "y", "z"),
    "sphere": ("radius",),
}
"""Each primitive's dimensions, in metres, in order. A box's are its full extents."""

STRATEGIES = {
    "cylinder": ("side", "top", "bottom"),
    "box": ("top", "bottom", "face_x", "face_y"),
    "sphere": ("all",),
}
"""The grasp families a request may select for each primitive: around a cylinder's
side, or at its top or bottom face; at a box's top or bottom face, or across its x or y
faces. The names are prl_assets' ``grasp_regions`` strategies."""

Bounds = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]


def _length(value: Any, path: str) -> float:
    check_spec(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"a number of metres, not {value!r}",
        path,
    )
    check_spec(math.isfinite(value) and value > 0, f"finite and positive, not {value!r}", path)
    return float(value)


def _name(value: Any, path: str) -> str:
    check_spec(isinstance(value, str) and value.strip() != "", "a non-empty name", path)
    return str(value)


@dataclass(frozen=True, slots=True)
class Primitive:
    """A primitive approximation of an object, in its centred frame."""

    kind: str
    """``cylinder``, ``box``, or ``sphere``."""
    dimensions: tuple[float, ...]
    """In metres, ordered as ``DIMENSIONS[kind]``: a cylinder's radius and height, a
    box's full x, y, z extents, a sphere's radius."""

    def __post_init__(self) -> None:
        check_spec(self.kind in DIMENSIONS, f"one of {sorted(DIMENSIONS)}", "kind")
        names = DIMENSIONS[self.kind]
        check_spec(
            isinstance(self.dimensions, (tuple, list)) and len(self.dimensions) == len(names),
            f"{len(names)} dimensions: {', '.join(names)}",
            "dimensions",
        )
        values = tuple(
            _length(v, f"dimensions[{i}] ({n})")
            for i, (v, n) in enumerate(zip(self.dimensions, names, strict=True))
        )
        object.__setattr__(self, "dimensions", values)

    @property
    def width(self) -> float:
        """The narrowest span a parallel-jaw hand could close across."""
        if self.kind == "box":
            return min(self.dimensions)
        return 2.0 * self.dimensions[0]


@dataclass(frozen=True, slots=True)
class GraspObject:
    """An object to grasp: which scene object, its primitive approximation, and the pose
    of that primitive's centred frame where it was observed."""

    name: str
    """The scene object, as the runtime names it: the grasp's ``object``."""
    primitive: Primitive
    pose: Pose
    """The primitive's centred frame in ``frame``."""
    frame: str = "world"
    """The frame ``pose`` is expressed in. Pregrasps are world regions, so only
    ``world`` is accepted."""
    revision: str | int | None = None
    """The observation revision the pose came from, recorded with the regions."""

    def __post_init__(self) -> None:
        _name(self.name, "name")
        check_spec(isinstance(self.primitive, Primitive), "a Primitive", "primitive")
        check_spec(isinstance(self.pose, Pose), "an ssrobot Pose", "pose")
        check_spec(
            all(math.isfinite(c) for c in (*self.pose.position, *self.pose.quat_wxyz)),
            "finite",
            "pose",
        )
        check_spec(self.frame == "world", "a pose in the world frame", "frame")
        check_spec(
            self.revision is None
            or (isinstance(self.revision, (str, int)) and not isinstance(self.revision, bool)),
            "a string, an integer, or None",
            "revision",
        )


@dataclass(frozen=True, slots=True)
class GraspingProfile:
    """The hand an application selected, and the robot's names for what carries it.

    Nothing here is inferred from the robot: the application chooses the profile."""

    hand: str
    """The hand model's identity, recorded with the regions."""
    manipulator: str
    end_effector: str
    """The end effector whose frame the regions bind: the grasp frame, with z the
    approach and y the finger opening."""
    gripper: str
    opening_channel: str
    max_aperture_m: float
    """The widest object the open hand clears."""

    def __post_init__(self) -> None:
        for name in ("hand", "manipulator", "end_effector", "gripper", "opening_channel"):
            _name(getattr(self, name), name)
        object.__setattr__(self, "max_aperture_m", _length(self.max_aperture_m, "max_aperture_m"))


@dataclass(frozen=True, slots=True)
class RegionTemplate:
    """A grasp region in a primitive's centred frame: ``origin`` places the region
    frame in the object frame, ``offset`` the end effector in the region frame, as in
    ssrobot's ``Tsr``."""

    family: str
    """Which of the hand's grasps this is, as the source names it."""
    origin: Pose
    offset: Pose
    position_bounds: Bounds
    rotation_bounds: Bounds


class TemplateSource(Protocol):
    """A hand's grasp templates for a primitive, in its centred frame, of the selected
    ``STRATEGIES``. Raises ``UnsupportedPrimitive`` for a primitive the hand's
    templates do not cover."""

    def __call__(
        self, primitive: Primitive, strategies: Sequence[str]
    ) -> Sequence[RegionTemplate]: ...


class UnsupportedPrimitive(Exception):
    """The source has no grasp templates for this primitive with this hand."""


class GeometryReason(enum.StrEnum):
    """Why well-formed geometry produced no pregrasp region."""

    WIDER_THAN_APERTURE = "wider_than_aperture"
    """The object's narrowest span exceeds the hand's aperture."""
    UNSUPPORTED_BY_HAND = "unsupported_by_hand"
    """The hand's templates do not cover this primitive."""
    NO_TEMPLATES = "no_templates"
    """The hand's templates for this primitive all fall outside its limits."""


@dataclass(frozen=True, slots=True)
class GraspRegions:
    """The world pregrasp regions for one object and profile, or the reason there are
    none, with what they were made from."""

    object: GraspObject
    profile: GraspingProfile
    approach_m: float
    tsrs: tuple[Tsr, ...] = ()
    families: tuple[str, ...] = ()
    """Each region's grasp family, in the order of ``tsrs``."""
    reason: GeometryReason | None = None
    detail: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    """What the regions were made from: the object's name, primitive, pose, frame, and
    revision; the hand and end effector; and the back-off. JSON-ready."""

    @property
    def ok(self) -> bool:
        return self.reason is None

    def spec(
        self, *, support: str, normal: tuple[float, float, float], lift_m: float, **options: Any
    ) -> GraspSpec:
        """A ``GraspSpec`` over these regions: the profile's names, the object, and the
        approach the regions are backed off by. ``options`` are any other ``GraspSpec``
        fields. Refuses when there are no regions."""
        if self.reason is not None:
            raise ValueError(f"no pregrasp regions for {self.object.name}: {self.detail}")
        return GraspSpec(
            manipulator=self.profile.manipulator,
            end_effector=self.profile.end_effector,
            gripper=self.profile.gripper,
            opening_channel=self.profile.opening_channel,
            object=self.object.name,
            support=support,
            normal=normal,
            pregrasps=self.tsrs,
            approach_m=self.approach_m,
            lift_m=lift_m,
            **options,
        )


def grasp_regions(
    obj: GraspObject,
    profile: GraspingProfile,
    source: TemplateSource,
    *,
    approach_m: float,
    strategies: Sequence[str] | None = None,
) -> GraspRegions:
    """The world pregrasp regions for grasping ``obj`` with ``profile``'s hand, backed
    off by ``approach_m`` along the end effector's approach axis. ``strategies`` selects
    the grasp families (default: all of the primitive's ``STRATEGIES``); a can resting
    on a support, for example, takes only ``side`` and ``top``."""
    check_spec(isinstance(obj, GraspObject), "a GraspObject", "obj")
    check_spec(isinstance(profile, GraspingProfile), "a GraspingProfile", "profile")
    approach_m = _length(approach_m, "approach_m")
    primitive = obj.primitive
    vocabulary = STRATEGIES[primitive.kind]
    if strategies is None:
        selected = vocabulary
    else:
        check_spec(
            not isinstance(strategies, str) and len(strategies) > 0,
            "a non-empty sequence of strategy names",
            "strategies",
        )
        selected = tuple(strategies)
        check_spec(
            all(s in vocabulary for s in selected) and len(set(selected)) == len(selected),
            f"distinct {primitive.kind} strategies from {list(vocabulary)}",
            "strategies",
        )
    provenance: dict[str, Any] = {
        "object": obj.name,
        "primitive": {"kind": primitive.kind, "dimensions_m": list(primitive.dimensions)},
        "pose": {"position": list(obj.pose.position), "quat_wxyz": list(obj.pose.quat_wxyz)},
        "frame": obj.frame,
        "revision": obj.revision,
        "strategies": list(selected),
        "hand": profile.hand,
        "end_effector": profile.end_effector,
        "approach_m": approach_m,
    }
    base = GraspRegions(object=obj, profile=profile, approach_m=approach_m, provenance=provenance)

    if primitive.width > profile.max_aperture_m:
        return dataclasses.replace(
            base,
            reason=GeometryReason.WIDER_THAN_APERTURE,
            detail=(
                f"{obj.name}'s narrowest span, {primitive.width:.4f} m, exceeds "
                f"{profile.hand}'s aperture, {profile.max_aperture_m:.4f} m"
            ),
        )
    try:
        templates = tuple(source(primitive, selected))
    except UnsupportedPrimitive as error:
        return dataclasses.replace(
            base, reason=GeometryReason.UNSUPPORTED_BY_HAND, detail=str(error)
        )
    if not templates:
        return dataclasses.replace(
            base,
            reason=GeometryReason.NO_TEMPLATES,
            detail=(
                f"{profile.hand} has no {'/'.join(selected)} {primitive.kind} grasp "
                "within its limits"
            ),
        )
    back_off = Pose(position=(0.0, 0.0, -approach_m), quat_wxyz=(1.0, 0.0, 0.0, 0.0))
    tsrs = tuple(
        Tsr(
            origin=compose(obj.pose, t.origin),
            offset=compose(t.offset, back_off),
            position_bounds=t.position_bounds,
            rotation_bounds=t.rotation_bounds,
        )
        for t in templates
    )
    return dataclasses.replace(base, tsrs=tsrs, families=tuple(t.family for t in templates))
