"""Structure-editing helpers (pure NumPy, viewer-independent).

These back the GUI "Edit" section:

- :func:`toggle_bond` connects/disconnects two atoms in an explicit bond list.
- :class:`UmbrellaGroup` captures a *reference* atom, an *axis* atom and a
  set of *interest* atoms, and can rotate the interests on cones around the
  reference->axis direction while keeping every reference-interest distance
  constant.

The umbrella motion is the classic "umbrella inversion" (Walden inversion):
every interest atom keeps its distance ``r`` to the reference and its azimuth
around the axis, while its polar angle ``theta`` (measured from the axis) is
driven to the same absolute value. With a symmetric group (azimuths 120 deg
apart) the interests stay an equilateral triangle; at ``theta = 90`` they lie
exactly in the plane through the reference. When ``theta`` reaches ``0`` (or
``180``) every atom collapses onto the same point of the axis
(``reference + r * axis``), which is the requested "over-move" behaviour.

All functions work in Angstroms and degrees and never touch the viewer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pychem.core.molecule import Molecule

_EPS = 1e-9


def sorted_unique_pairs(bonds) -> np.ndarray:
    """Normalise a bond list to ``(m, 2)`` int64 rows with ``i < j``."""
    if bonds is None:
        return np.empty((0, 2), dtype=np.int64)
    arr = np.asarray(bonds, dtype=np.int64)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    arr = arr.reshape(-1, 2)
    arr = np.sort(arr, axis=1)
    arr = arr[arr[:, 0] != arr[:, 1]]
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    arr = np.unique(arr, axis=0)
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    return arr[order].astype(np.int64)


def has_bond(bonds, i: int, j: int) -> bool:
    """True if the pair ``(i, j)`` is in the bond list."""
    arr = sorted_unique_pairs(bonds)
    if arr.size == 0:
        return False
    a, b = sorted((int(i), int(j)))
    return bool(np.any((arr[:, 0] == a) & (arr[:, 1] == b)))


def toggle_bond(bonds, i: int, j: int) -> np.ndarray:
    """Add the bond ``(i, j)`` if missing, remove it if present."""
    arr = sorted_unique_pairs(bonds)
    a, b = sorted((int(i), int(j)))
    if a == b:
        return arr
    match = (arr[:, 0] == a) & (arr[:, 1] == b) if arr.size else None
    if match is not None and bool(match.any()):
        return arr[~match]
    return sorted_unique_pairs(np.vstack([arr, np.array([[a, b]])]))


@dataclass(frozen=True)
class UmbrellaGroup:
    """A frozen snapshot of an umbrella-rotation group.

    ``positions(angle_deg)`` gives the new coordinates of the interest atoms
    for an *absolute* polar angle (measured from the reference->axis
    direction). Because the snapshot stores the original radii, polar angles
    and azimuths, repeated calls never drift and every reference-interest
    distance is preserved exactly.
    """

    reference: int
    axis_atom: int
    interests: tuple[int, ...]
    origin: np.ndarray  # reference position, (3,)
    axis: np.ndarray  # unit reference->axis direction, (3,)
    basis1: np.ndarray  # first in-plane unit vector, (3,)
    basis2: np.ndarray  # second in-plane unit vector, (3,)
    radii: np.ndarray  # (n,) reference-interest distances
    thetas: np.ndarray  # (n,) polar angles (radians, 0..pi)
    phis: np.ndarray  # (n,) azimuths (radians)

    @property
    def theta_deg(self) -> float:
        """Current mean polar angle (degrees); the slider's start value."""
        return float(np.degrees(np.mean(self.thetas)))

    @classmethod
    def capture(
        cls,
        molecule: Molecule,
        reference: int,
        axis_atom: int,
        interests,
    ) -> "UmbrellaGroup":
        """Snapshot the group geometry from ``molecule``.

        Raises ``ValueError`` for degenerate definitions (repeated atoms, an
        axis atom on top of the reference, or no interest atoms).
        """
        reference = int(reference)
        axis_atom = int(axis_atom)
        interest_list = tuple(int(i) for i in interests)
        if not interest_list:
            raise ValueError("an umbrella group needs at least one interest atom")
        if axis_atom == reference:
            raise ValueError("the axis atom must differ from the reference atom")
        for idx in (reference, axis_atom, *interest_list):
            if not 0 <= idx < molecule.n_atoms:
                raise ValueError(f"atom index out of range: {idx}")
        if reference in interest_list or axis_atom in interest_list:
            raise ValueError("interest atoms must differ from reference/axis atoms")

        coords = np.asarray(molecule.coords, dtype=float)
        origin = coords[reference].copy()
        axis_vec = coords[axis_atom] - origin
        length = float(np.linalg.norm(axis_vec))
        if length <= _EPS:
            raise ValueError("the axis atom coincides with the reference atom")
        axis = axis_vec / length

        # Orthonormal in-plane basis (any pair perpendicular to the axis).
        helper = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(helper, axis))) > 0.9:
            helper = np.array([1.0, 0.0, 0.0])
        basis1 = helper - float(np.dot(helper, axis)) * axis
        basis1 = basis1 / float(np.linalg.norm(basis1))
        basis2 = np.cross(axis, basis1)

        n = len(interest_list)
        radii = np.empty(n, dtype=float)
        thetas = np.empty(n, dtype=float)
        phis = np.empty(n, dtype=float)
        # ``thetas`` keeps each atom's original polar angle so the slider can
        # start at its mean; ``positions`` then drives them all to the same
        # absolute angle (so the planar state is exact even for a slightly
        # asymmetric group such as an optimized SN2 transition state).
        for k, idx in enumerate(interest_list):
            vec = coords[idx] - origin
            radii[k] = float(np.linalg.norm(vec))
            parallel = float(np.dot(vec, axis))
            radial = vec - parallel * axis
            rho = float(np.linalg.norm(radial))
            thetas[k] = float(np.arctan2(rho, parallel))
            phis[k] = (
                float(np.arctan2(float(np.dot(radial, basis2)), float(np.dot(radial, basis1))))
                if rho > _EPS
                else 0.0
            )
        return cls(
            reference=reference,
            axis_atom=axis_atom,
            interests=interest_list,
            origin=origin,
            axis=axis,
            basis1=basis1,
            basis2=basis2,
            radii=radii,
            thetas=thetas,
            phis=phis,
        )

    def positions(self, angle_deg: float) -> np.ndarray:
        """Coordinates of the interest atoms at an absolute angle (degrees).

        Every interest atom is placed at the same polar angle (its radius and
        azimuth are preserved), so ``90`` puts them all in the plane through
        the reference and ``0``/``180`` collapse them onto the axis.
        """
        theta = np.radians(float(angle_deg))
        sin_t, cos_t = np.sin(theta), np.cos(theta)
        out = np.empty((len(self.interests), 3), dtype=float)
        for k in range(len(self.interests)):
            phi = float(self.phis[k])
            direction = (
                sin_t * (np.cos(phi) * self.basis1 + np.sin(phi) * self.basis2)
                + cos_t * self.axis
            )
            out[k] = self.origin + float(self.radii[k]) * direction
        return out

    def apply(self, molecule: Molecule, angle_deg: float) -> None:
        """Write the interest coordinates for ``angle_deg`` into ``molecule``."""
        molecule.coords[list(self.interests)] = self.positions(angle_deg)

    def angle_from_displacement(self, angle_deg: float, displacement) -> float:
        """Angle after pushing the group along the axis by ``displacement``.

        Used by the drag-in-the-viewer control: the screen-plane displacement
        of the dragged atom is projected onto the axis, and the resulting
        change of the parallel component is turned into a new polar angle.
        """
        radius = float(np.mean(self.radii))
        if radius <= _EPS:
            return float(angle_deg)
        parallel = radius * np.cos(np.radians(float(angle_deg)))
        parallel += float(np.dot(np.asarray(displacement, dtype=float), self.axis))
        return float(np.degrees(np.arccos(np.clip(parallel / radius, -1.0, 1.0))))
