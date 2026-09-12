"""Volumetric data model (scalar fields on a regular orthogonal grid).

Used for electron densities, orbitals and other 3D scalar fields read from
formats such as Gaussian cube. The grid is described by an origin, the
number of points per axis and a diagonal spacing (orthogonal grids).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Electron-density envelope used to build electrostatic-potential (ESP) maps.
# Quantum-chemistry codes report the density in e/Bohr^3, where ~1e-3 wraps the
# molecule closely without hiding its shape; that shell is then coloured by
# the ESP (see pychem/rendering/esp_renderer.py).
DENSITY_ENVELOPE_LEVEL = 1.0e-3


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

    def suggest_density_isovalue(
        self, target: float = DENSITY_ENVELOPE_LEVEL
    ) -> float:
        """Envelope level for the *density* of an electrostatic-potential map.

        Unlike :meth:`suggest_isovalue` (a quantile, which for a density is
        dominated by the huge values near the nuclei), this returns the small
        physical level that wraps the molecule. The target is capped for
        fields stored on a very different scale so the surface neither
        vanishes nor fills the box.
        """
        ceiling = self.max_abs * 1e-2
        if ceiling <= 0.0:
            return float(target)
        return float(min(target, ceiling))

    def sample_at(self, points: np.ndarray) -> np.ndarray:
        """Trilinearly interpolate the field at arbitrary positions (Å).

        ``points`` is an ``(n, 3)`` array of positions in the same Å frame as
        ``origin``; the result is an ``(n,)`` array of field values. Positions
        outside the grid are clamped to its boundary, so callers always get a
        finite value (used to colour a surface that may slightly overhang the
        sampling grid instead of leaving holes).

        Trilinear interpolation is exact for the linear fields typical of a
        smooth potential, and reduces to the node value at grid points.
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"points must have shape (n, 3); got {pts.shape}")

        sizes = np.asarray(self.shape, dtype=float)
        last = sizes - 1.0
        # Continuous index of every point along each axis, clamped to the grid.
        frac = (pts - self.origin) / self.spacing
        idx = np.clip(frac, 0.0, last)
        lo = np.floor(idx).astype(int)
        hi = np.minimum(lo + 1, (sizes - 1).astype(int))
        weight = idx - lo  # 0 at ``lo``, 1 at ``hi``

        out = np.zeros(len(idx))
        for dx in (0, 1):
            wx = weight[:, 0] if dx else 1.0 - weight[:, 0]
            ix = hi[:, 0] if dx else lo[:, 0]
            for dy in (0, 1):
                wy = weight[:, 1] if dy else 1.0 - weight[:, 1]
                iy = hi[:, 1] if dy else lo[:, 1]
                for dz in (0, 1):
                    wz = weight[:, 2] if dz else 1.0 - weight[:, 2]
                    iz = hi[:, 2] if dz else lo[:, 2]
                    out += wx * wy * wz * self.values[ix, iy, iz]
        return out

    def __repr__(self) -> str:  # pragma: no cover - debugging
        return (
            f"VolumeData(label={self.label!r}, shape={self.shape}, "
            f"max_abs={self.max_abs:.3g})"
        )
