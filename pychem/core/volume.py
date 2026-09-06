"""Volumetric data model (scalar fields on a regular orthogonal grid).

Used for electron densities, orbitals and other 3D scalar fields read from
formats such as Gaussian cube. The grid is described by an origin, the
number of points per axis and a diagonal spacing (orthogonal grids).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class VolumeData:
    """A scalar field sampled on a regular orthogonal 3D grid."""

    values: np.ndarray  # shape (nx, ny, nz)
    origin: np.ndarray  # shape (3,), Å
    spacing: np.ndarray  # shape (3,), Å (step along each axis)
    label: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        origin = np.asarray(self.origin, dtype=float)
        spacing = np.asarray(self.spacing, dtype=float)
        if values.ndim != 3:
            raise ValueError(f"volume values must be 3D; got shape {values.shape}")
        if origin.shape != (3,) or spacing.shape != (3,):
            raise ValueError("origin and spacing must have shape (3,)")
        self.values = values
        self.origin = origin
        self.spacing = spacing

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.values.shape

    @property
    def max_abs(self) -> float:
        return float(np.abs(self.values).max()) if self.values.size else 0.0

    @property
    def extent(self) -> tuple[float, float, float, float, float, float]:
        """Bounding box ``(xmin, xmax, ymin, ymax, zmin, zmax)`` in Å."""
        nx, ny, nz = self.shape
        xyz = [
            self.origin[i] + j * self.spacing[i]
            for i, n in enumerate((nx, ny, nz))
            for j in (0, n - 1)
        ]
        return (xyz[0], xyz[1], xyz[2], xyz[3], xyz[4], xyz[5])

    def suggest_isovalue(self, target_fraction: float = 0.02) -> float:
        """Suggest an absolute isovalue that keeps a small fraction of the grid.

        The level is the |value| above which only ``target_fraction`` of the
        grid points lie. This adapts to each field's scale, giving an
        isosurface that is visible but does not fill the whole box.
        """
        flat = np.abs(self.values).ravel()
        if flat.size == 0:
            return 1.0
        maximum = float(flat.max())
        if maximum <= 0.0:
            return 1.0
        level = float(np.quantile(flat, 1.0 - target_fraction))
        return max(level, maximum * 1e-6)

    def __repr__(self) -> str:  # pragma: no cover - debugging
        return (
            f"VolumeData(label={self.label!r}, shape={self.shape}, "
            f"max_abs={self.max_abs:.3g})"
        )
