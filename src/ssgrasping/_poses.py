"""Rigid transforms on ssrobot ``Pose`` values: compose, invert, and read axes.

Pure Python, so the base package needs nothing beyond ssrobot's neutral core.
"""

from __future__ import annotations

import math

from ssrobot import Pose, Tsr

Quaternion = tuple[float, float, float, float]
Vector = tuple[float, float, float]


def _multiply(a: Quaternion, b: Quaternion) -> Quaternion:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _conjugate(q: Quaternion) -> Quaternion:
    w, x, y, z = q
    return (w, -x, -y, -z)


def rotate(q: Quaternion, v: Vector) -> Vector:
    """``v`` rotated by the unit quaternion ``q``."""
    _, x, y, z = _multiply(_multiply(q, (0.0, *v)), _conjugate(q))
    return (x, y, z)


def _normalized(q: Quaternion) -> Quaternion:
    n = math.sqrt(sum(c * c for c in q))
    w, x, y, z = (c / n for c in q)
    return (w, x, y, z) if w >= 0.0 else (-w, -x, -y, -z)


def compose(a: Pose, b: Pose) -> Pose:
    """``a`` then ``b``: the pose of frame ``b`` (given in ``a``) in ``a``'s parent."""
    t = rotate(a.quat_wxyz, b.position)
    position = (a.position[0] + t[0], a.position[1] + t[1], a.position[2] + t[2])
    return Pose(position=position, quat_wxyz=_normalized(_multiply(a.quat_wxyz, b.quat_wxyz)))


def invert(a: Pose) -> Pose:
    """The inverse transform of ``a``."""
    q = _conjugate(a.quat_wxyz)
    t = rotate(q, a.position)
    return Pose(position=(-t[0], -t[1], -t[2]), quat_wxyz=_normalized(q))


def axis(pose: Pose, index: int) -> Vector:
    """The world direction of ``pose``'s x (0), y (1), or z (2) axis."""
    unit = [0.0, 0.0, 0.0]
    unit[index] = 1.0
    x, y, z = rotate(pose.quat_wxyz, (unit[0], unit[1], unit[2]))
    return (x, y, z)


def roll_pitch_yaw(q: Quaternion) -> Vector:
    """The roll, pitch, and yaw (fixed-axis x, y, z) of ``q``, in radians."""
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return (roll, pitch, yaw)


def _region_local(pose: Pose, region: Tsr) -> Pose:
    """``pose`` as the region's ``T_w``: ``inv(origin) ∘ pose ∘ inv(offset)``."""
    return compose(compose(invert(region.origin), pose), invert(region.offset))


def _angle_within(angle: float, low: float, high: float, tolerance: float) -> bool:
    """Whether ``angle``, wrapped by whole turns, lies in ``[low, high]`` ± ``tolerance``."""
    if high - low + 2 * tolerance >= 2 * math.pi:
        return True
    start = low - tolerance
    return start + (angle - start) % (2 * math.pi) <= high + tolerance


def _rotation_within(local: Pose, region: Tsr, tolerance: float) -> bool:
    angles = roll_pitch_yaw(local.quat_wxyz)
    return all(
        _angle_within(a, low, high, tolerance)
        for a, (low, high) in zip(angles, region.rotation_bounds, strict=True)
    )


def inside(pose: Pose, region: Tsr, tolerance: tuple[float, float]) -> bool:
    """Whether ``pose`` is in ``region``, within ``(metres, radians)`` of every one of
    its six bounds. Roll, pitch, and yaw are the region's fixed-axis angles, wrapped
    into their bounds by whole turns."""
    local = _region_local(pose, region)
    metres, radians = tolerance
    return all(
        low - metres <= v <= high + metres
        for v, (low, high) in zip(local.position, region.position_bounds, strict=True)
    ) and _rotation_within(local, region, radians)


def offset_interval(
    pose: Pose, region: Tsr, direction: Vector, tolerance: tuple[float, float]
) -> tuple[float, float] | None:
    """The distances ``d`` for which ``pose`` moved by ``-d`` along the world unit
    ``direction`` is in ``region`` (as ``inside``), or None if there are none or its
    rotation is outside. Each position bound is linear in ``d``, so this is exact."""
    local = _region_local(pose, region)
    metres, radians = tolerance
    if not _rotation_within(local, region, radians):
        return None
    step = rotate(invert(region.origin).quat_wxyz, direction)  # d's effect, region frame
    lo, hi = -math.inf, math.inf
    for v, m, (low, high) in zip(local.position, step, region.position_bounds, strict=True):
        # low - metres <= v - d m <= high + metres
        if abs(m) < 1e-12:
            if not low - metres <= v <= high + metres:
                return None
            continue
        a, b = (v - high - metres) / m, (v - low + metres) / m
        lo, hi = max(lo, min(a, b)), min(hi, max(a, b))
    return (lo, hi) if lo <= hi else None
