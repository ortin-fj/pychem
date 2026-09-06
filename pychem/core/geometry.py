"""Small geometric measurements between points (pure NumPy).

The functions take plain Cartesian coordinates (as length-3 sequences, in Å)
and return values in Å or degrees, so they are independent of the viewer and
easy to test. They back the interactive "Measurements" section of the GUI
(Bonds / Angles / Dihedrals).
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-9


def _vec(point) -> np.ndarray:
    return np.asarray(point, dtype=float)


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def bond_length(p1, p2) -> float:
    """Distance between ``p1`` and ``p2`` (Å)."""
    return _norm(_vec(p2) - _vec(p1))


def angle_degrees(p1, vertex, p2) -> float:
    """Angle at ``vertex`` between ``p1`` and ``p2`` (degrees, 0..180)."""
    u = _vec(p1) - _vec(vertex)
    v = _vec(p2) - _vec(vertex)
    nu, nv = _norm(u), _norm(v)
    if nu <= _EPS or nv <= _EPS:
        return 0.0
    cos = float(np.dot(u, v)) / (nu * nv)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def dihedral_degrees(p1, p2, p3, p4) -> float:
    """Signed dihedral of ``p1-p2-p3-p4`` about the central bond (degrees).

    The sign follows the standard convention used by most molecular viewers:
    positive when ``p4`` is on the clockwise side of the ``p1-p2-p3`` plane
    when looking along ``p2 -> p3``.
    """
    a, b, c, d = (_vec(p) for p in (p1, p2, p3, p4))
    b0 = b - a  # along the first "arm"
    b1 = c - b  # central bond
    b2 = d - c  # along the last "arm"
    nb1 = _norm(b1)
    if nb1 <= _EPS:
        return 0.0
    b1 = b1 / nb1

    # Project the arms onto the plane perpendicular to the central bond.
    v = b0 - b1 * float(np.dot(b0, b1))
    w = b2 - b1 * float(np.dot(b2, b1))
    nv, nw = _norm(v), _norm(w)
    if nv <= _EPS or nw <= _EPS:
        return 0.0
    v = v / nv
    w = w / nw
    x = float(np.dot(v, w))
    y = float(np.dot(np.cross(b1, v), w))
    return float(np.degrees(np.arctan2(y, x)))


def bond_anchor(p1, p2) -> np.ndarray:
    """Suggested spot for a bond label: the segment midpoint."""
    return 0.5 * (_vec(p1) + _vec(p2))


def angle_anchor(p1, vertex, p2) -> np.ndarray:
    """Suggested spot for an angle label: a bit along the angle bisector."""
    b = _vec(vertex)
    direction = (_vec(p1) - b) + (_vec(p2) - b)
    if _norm(direction) <= _EPS:
        return b + np.array([0.0, 0.0, 0.5])
    return b + 0.5 * (direction / _norm(direction))


def dihedral_anchor(p1, p2, p3, p4) -> np.ndarray:
    """Suggested spot for a dihedral label: near the central bond, outward."""
    b, c = _vec(p2), _vec(p3)
    middle = 0.5 * (b + c)
    outward = 0.5 * (_vec(p1) + _vec(p4)) - middle
    if _norm(outward) <= _EPS:
        outward = np.array([0.0, 0.0, 1.0])
    return middle + 0.45 * (outward / _norm(outward))
