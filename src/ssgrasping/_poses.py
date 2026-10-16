"""Rigid transforms on ssrobot ``Pose`` values: compose, invert, and read axes.

Pure Python, so the base package needs nothing beyond ssrobot's neutral core.
"""

from __future__ import annotations

import math

from ssrobot import Pose

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
