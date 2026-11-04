"""Grasp regions from explicit object geometry and an explicit gripper profile.

An application states what it is grasping and with what, as data: a ``GraspObject``
(its observed pose and the named ``GraspRegion``s it may be grasped by, each a
primitive placed in the object's frame) and a ``GraspingProfile`` (the hand and the
robot's names for its manipulator, end effector, gripper, and opening channel).
``grasp_regions`` turns them into the world pregrasp regions a ``GraspSpec`` takes,
region by region, through a ``TemplateSource`` that supplies the hand's grasp templates
in a primitive's frame; ``ssgrasping.sstsr`` is one, from sstsr's hands. Nothing is
inferred from names, meshes, or the robot.

Every primitive's frame has its origin at its geometric centre, with z the cylinder's
axis and the box's local z, and a region's ``origin`` places it in the object frame:
``T_world_region = T_world_object @ T_object_region``. A ``RegionTemplate`` is expressed
in the primitive's frame, so a source whose library frames primitives differently
(sstsr frames cylinders and boxes at their bottom face) converts once, in the source,
and no caller shifts poses by half a height.

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
import numbers
import re
from collections.abc import Sequence
from dataclasses import dataclass
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

SPANS = {
    ("cylinder", "side"): (0,),
    ("cylinder", "top"): (0,),
    ("cylinder", "bottom"): (0,),
    ("box", "top"): (0, 1),
    ("box", "bottom"): (0, 1),
    ("box", "face_x"): (1, 2),
    ("box", "face_y"): (0, 2),
    ("sphere", "all"): (0,),
}
"""Which of a primitive's spans each strategy's fingers may close across, as indices
into ``Primitive.spans``: a box's top grasp closes across its x or y extent, an x-face
grasp across y or z."""

Bounds = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]


def _length(value: Any, path: str) -> float:
    check_spec(
        isinstance(value, numbers.Real) and not isinstance(value, bool),
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
        dimensions: Any = self.dimensions
        check_spec(
            not isinstance(dimensions, (str, bytes)) and hasattr(dimensions, "__len__"),
            f"{len(names)} dimensions: {', '.join(names)}",
            "dimensions",
        )
        check_spec(
            len(dimensions) == len(names),
            f"{len(names)} dimensions: {', '.join(names)}",
            "dimensions",
        )
        values = tuple(
            _length(v, f"dimensions[{i}] ({n})")
            for i, (v, n) in enumerate(zip(dimensions, names, strict=True))
        )
        object.__setattr__(self, "dimensions", values)

    @property
    def spans(self) -> tuple[float, ...]:
        """The widths fingers could close across: a box's x, y, z extents; a cylinder's
        or sphere's diameter."""
        if self.kind == "box":
            return self.dimensions
        return (2.0 * self.dimensions[0],)

    def width(self, strategies: Sequence[str]) -> float:
        """The narrowest span any of ``strategies`` closes across."""
        return min(self.spans[i] for s in strategies for i in SPANS[(self.kind, s)])


IDENTITY = Pose(position=(0.0, 0.0, 0.0), quat_wxyz=(1.0, 0.0, 0.0, 0.0))
_REGION_NAME = re.compile(r"[a-z][a-z0-9_]*")


def _finite_pose(value: Any, path: str) -> None:
    check_spec(isinstance(value, Pose), "an ssrobot Pose", path)
    check_spec(all(math.isfinite(c) for c in (*value.position, *value.quat_wxyz)), "finite", path)


@dataclass(frozen=True, slots=True)
class GraspRegion:
    """A named part of an object to grasp by: a primitive in its own centred frame, placed
    in the object's frame by ``origin``. A mug may have a cylindrical ``body`` and a box
    ``handle`` offset from it. A region is a grasp approximation: it need not enclose the
    object, nor match its collision geometry."""

    name: str
    """Unique within the object: lower case letters, digits, and underscores."""
    primitive: Primitive
    origin: Pose = IDENTITY
    """The region frame in the object frame (``T_object_region``): the world region
    frame is ``T_world_object @ T_object_region``."""
    strategies: tuple[str, ...] | None = None
    """The grasp families to make (default: all of the primitive's ``STRATEGIES``); a
    mug's body, standing on a support, takes ``side`` and ``top``."""

    def __post_init__(self) -> None:
        check_spec(
            isinstance(self.name, str) and _REGION_NAME.fullmatch(self.name) is not None,
            "lower case letters, digits, and underscores, starting with a letter",
            "name",
        )
        check_spec(isinstance(self.primitive, Primitive), "a Primitive", "primitive")
        _finite_pose(self.origin, "origin")
        vocabulary = STRATEGIES[self.primitive.kind]
        if self.strategies is None:
            object.__setattr__(self, "strategies", vocabulary)
            return
        check_spec(
            not isinstance(self.strategies, str) and len(self.strategies) > 0,
            "a non-empty sequence of strategy names",
            "strategies",
        )
        selected = tuple(self.strategies)
        check_spec(
            all(s in vocabulary for s in selected) and len(set(selected)) == len(selected),
            f"distinct {self.primitive.kind} strategies from {list(vocabulary)}",
            "strategies",
        )
        object.__setattr__(self, "strategies", selected)


@dataclass(frozen=True, slots=True)
class GraspObject:
    """An object to grasp: which scene object, where it was observed, and the regions it
    may be grasped by, in its frame."""

    name: str
    """The scene object, as the runtime names it: the grasp's ``object``."""
    pose: Pose
    """The object frame in ``frame``."""
    regions: tuple[GraspRegion, ...]
    """At least one, with distinct names. An object of one shape is one region at the
    identity."""
    frame: str = "world"
    """The frame ``pose`` is expressed in. Pregrasps are world regions, so only
    ``world`` is accepted."""
    revision: str | int | None = None
    """The observation revision the pose came from, recorded with the regions."""

    def __post_init__(self) -> None:
        _name(self.name, "name")
        _finite_pose(self.pose, "pose")
        check_spec(
            not isinstance(self.regions, (str, bytes)) and hasattr(self.regions, "__len__"),
            "a sequence of GraspRegion",
            "regions",
        )
        regions = tuple(self.regions)
        check_spec(len(regions) > 0, "at least one region", "regions")
        check_spec(
            all(isinstance(r, GraspRegion) for r in regions), "GraspRegion values", "regions"
        )
        names = [r.name for r in regions]
        check_spec(len(set(names)) == len(names), f"distinct region names, not {names}", "regions")
        object.__setattr__(self, "regions", regions)
        check_spec(self.frame == "world", "a pose in the world frame", "frame")
        check_spec(
            self.revision is None
            or (isinstance(self.revision, (str, int)) and not isinstance(self.revision, bool)),
            "a string, an integer, or None",
            "revision",
        )

    def region(self, name: str) -> GraspRegion:
        """The region named ``name``."""
        for r in self.regions:
            if r.name == name:
                return r
        raise KeyError(name)


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
    strategy: str
    """The ``STRATEGIES`` entry it belongs to."""
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


def _pose_json(p: Pose) -> dict[str, list[float]]:
    return {"position": list(p.position), "quat_wxyz": list(p.quat_wxyz)}


@dataclass(frozen=True, slots=True)
class RegionCandidates:
    """One region's world pregrasp regions, or the reason it has none."""

    region: GraspRegion
    frame: Pose
    """The region frame in the world: the object's pose composed with the region's
    origin."""
    tsrs: tuple[Tsr, ...] = ()
    families: tuple[str, ...] = ()
    """Each of ``tsrs``' grasp, as the template source names it."""
    strategies: tuple[str, ...] = ()
    """Each of ``tsrs``' strategy."""
    reason: GeometryReason | None = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.reason is None

    @property
    def labels(self) -> tuple[str, ...]:
        """Each of ``tsrs`` as ``region/strategy``."""
        return tuple(f"{self.region.name}/{s}" for s in self.strategies)


@dataclass(frozen=True, slots=True)
class GraspRegions:
    """The world pregrasp regions for one object and profile, region by region, with what
    they were made from."""

    object: GraspObject
    profile: GraspingProfile
    approach_m: float
    regions: tuple[RegionCandidates, ...]
    """One per selected region, in the object's order."""

    @property
    def ok(self) -> bool:
        """Whether any region has pregrasp regions."""
        return any(r.ok for r in self.regions)

    @property
    def tsrs(self) -> tuple[Tsr, ...]:
        """Every region's pregrasp regions, in order."""
        return tuple(t for r in self.regions for t in r.tsrs)

    @property
    def labels(self) -> tuple[str, ...]:
        """Each of ``tsrs`` as ``region/strategy``."""
        return tuple(label for r in self.regions for label in r.labels)

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(f for r in self.regions for f in r.families)

    def region(self, name: str) -> RegionCandidates:
        """The candidates of the region named ``name``."""
        for r in self.regions:
            if r.region.name == name:
                return r
        raise KeyError(name)

    @property
    def provenance(self) -> dict[str, Any]:
        """What the regions were made from, JSON-ready: the object's name, pose, frame,
        and revision; the hand and end effector; the back-off; and for each region its
        name, primitive, origin in the object, frame in the world, strategies, and
        outcome. Each call returns a new dict."""
        obj = self.object
        return {
            "object": obj.name,
            "pose": _pose_json(obj.pose),
            "frame": obj.frame,
            "revision": obj.revision,
            "hand": self.profile.hand,
            "end_effector": self.profile.end_effector,
            "approach_m": self.approach_m,
            "regions": [
                {
                    "name": r.region.name,
                    "primitive": {
                        "kind": r.region.primitive.kind,
                        "dimensions_m": list(r.region.primitive.dimensions),
                    },
                    "origin": _pose_json(r.region.origin),
                    "frame": _pose_json(r.frame),
                    "strategies": list(r.region.strategies or ()),
                    "candidates": len(r.tsrs),
                    "reason": r.reason,
                    "detail": r.detail,
                }
                for r in self.regions
            ],
        }

    def spec(
        self, *, support: str, normal: tuple[float, float, float], lift_m: float, **options: Any
    ) -> GraspSpec:
        """A ``GraspSpec`` over these regions, each labelled ``region/strategy``: the
        profile's names, the object, and the approach the regions are backed off by.
        ``options`` are any other ``GraspSpec`` fields. Refuses when no region has
        pregrasp regions."""
        if not self.ok:
            details = "; ".join(f"{r.region.name}: {r.detail}" for r in self.regions)
            raise ValueError(f"no pregrasp regions for {self.object.name}: {details}")
        return GraspSpec(
            manipulator=self.profile.manipulator,
            end_effector=self.profile.end_effector,
            gripper=self.profile.gripper,
            opening_channel=self.profile.opening_channel,
            object=self.object.name,
            support=support,
            normal=normal,
            pregrasps=self.tsrs,
            pregrasp_labels=self.labels,
            approach_m=self.approach_m,
            lift_m=lift_m,
            **options,
        )


def _candidates(
    obj: GraspObject,
    region: GraspRegion,
    profile: GraspingProfile,
    source: TemplateSource,
    back_off: Pose,
) -> RegionCandidates:
    frame = compose(obj.pose, region.origin)
    base = RegionCandidates(region=region, frame=frame)
    primitive, strategies = region.primitive, region.strategies or ()
    # Whether the hand covers the primitive at all comes first: a hand without box grasps
    # is not one whose aperture a smaller box would fit.
    try:
        templates = tuple(source(primitive, strategies))
    except UnsupportedPrimitive as error:
        return dataclasses.replace(
            base, reason=GeometryReason.UNSUPPORTED_BY_HAND, detail=str(error)
        )
    width = primitive.width(strategies)
    if width > profile.max_aperture_m:
        return dataclasses.replace(
            base,
            reason=GeometryReason.WIDER_THAN_APERTURE,
            detail=(
                f"{obj.name}'s {region.name}'s narrowest span for {'/'.join(strategies)}, "
                f"{width:.4f} m, exceeds {profile.hand}'s aperture, "
                f"{profile.max_aperture_m:.4f} m"
            ),
        )
    if not templates:
        return dataclasses.replace(
            base,
            reason=GeometryReason.NO_TEMPLATES,
            detail=(
                f"{profile.hand} has no {'/'.join(strategies)} {primitive.kind} grasp of "
                f"{obj.name}'s {region.name} within its limits"
            ),
        )
    return dataclasses.replace(
        base,
        tsrs=tuple(
            Tsr(
                origin=compose(frame, t.origin),
                offset=compose(t.offset, back_off),
                position_bounds=t.position_bounds,
                rotation_bounds=t.rotation_bounds,
            )
            for t in templates
        ),
        families=tuple(t.family for t in templates),
        strategies=tuple(t.strategy for t in templates),
    )


def grasp_regions(
    obj: GraspObject,
    profile: GraspingProfile,
    source: TemplateSource,
    *,
    approach_m: float,
    regions: Sequence[str] | None = None,
) -> GraspRegions:
    """The world pregrasp regions for grasping ``obj`` with ``profile``'s hand, region by
    region, backed off by ``approach_m`` along the end effector's approach axis.
    ``regions`` selects the object's regions by name (default: all). Each region's world
    frame is the object's pose composed with the region's origin, and its templates are
    placed there; one region may have regions while another is too wide for the hand."""
    check_spec(isinstance(obj, GraspObject), "a GraspObject", "obj")
    check_spec(isinstance(profile, GraspingProfile), "a GraspingProfile", "profile")
    approach_m = _length(approach_m, "approach_m")
    known = [r.name for r in obj.regions]
    if regions is None:
        selected = tuple(obj.regions)
    else:
        check_spec(
            not isinstance(regions, str) and len(regions) > 0,
            "a non-empty sequence of region names",
            "regions",
        )
        names = tuple(regions)
        check_spec(
            all(n in known for n in names) and len(set(names)) == len(names),
            f"distinct names of {obj.name}'s regions, {known}",
            "regions",
        )
        selected = tuple(r for r in obj.regions if r.name in names)
    back_off = Pose(position=(0.0, 0.0, -approach_m), quat_wxyz=(1.0, 0.0, 0.0, 0.0))
    return GraspRegions(
        object=obj,
        profile=profile,
        approach_m=approach_m,
        regions=tuple(_candidates(obj, r, profile, source, back_off) for r in selected),
    )
