"""Isosurface rendering of volumetric fields with PyVista.

The volume is turned into a :class:`pyvista.ImageData` (orthogonal grid) and
contoured at ``±level``: one surface for the positive region and one for the
negative region (typical for orbitals). Surfaces are semi-transparent.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv

from pychem.core.volume import VolumeData

# Level used when none is given, as a fraction of the field maximum.
_DEFAULT_ISO_FRACTION = 0.08
_POS_COLOR = (0.25, 0.65, 1.0)  # cyan-blue
_NEG_COLOR = (1.0, 0.35, 0.2)  # orange-red
_OPACITY = 0.55

# "High" quality subdivides the contour with the *Loop* scheme. Loop is
# approximating: it both increases the vertex count and rounds the facets into
# a clean, smooth ("fluid") surface. (Butterfly was tried first because it is
# interpolating, but its fixed stencil produced wrinkle-like ripples on these
# marching-cubes meshes.) Two iterations give a nicely rounded result on
# typical meshes; very large meshes get one to keep the scene light.
_HIGH_SUBDIVIDE_NSUB = 2
_HIGH_SUBDIVIDE_NSUB_LARGE = 1
_HIGH_SUBDIVIDE_POINT_LIMIT = 20_000


def _to_image_data(volume: VolumeData) -> pv.ImageData:
    nx, ny, nz = volume.shape
    grid = pv.ImageData(
        dimensions=(nx, ny, nz),
        origin=tuple(volume.origin),
        spacing=tuple(volume.spacing),
    )
    # volume.values is canonical with index (x, y, z); VTK stores x fastest.
    grid.point_data["scalars"] = volume.values.ravel(order="F")
    return grid


class VolumeRenderer:
    """Manages the isosurface actors of the currently shown volume."""

    def __init__(self, plotter) -> None:
        self.plotter = plotter
        self._actors: list[object] = []

    def clear(self) -> None:
        for actor in self._actors:
            try:
                self.plotter.remove_actor(actor)
            except Exception:  # pragma: no cover - already removed
                pass
        self._actors = []

    def show(
        self,
        volume: VolumeData,
        level: float | None = None,
        quality: str = "medium",
    ) -> None:
        """Replace the scene volume with the isosurface of ``volume``.

        ``quality`` is ``"medium"`` or ``"high"`` and mainly controls how
        smooth ("fluid") the surface looks: High Loop-subdivides the contour
        so its facets are rounded; Medium keeps the raw contour.
        """
        self.clear()
        max_abs = volume.max_abs
        if max_abs <= 0.0:
            return
        if level is None:
            level = _DEFAULT_ISO_FRACTION * max_abs
        level = float(level)

        grid = _to_image_data(volume)
        try:
            positive = grid.contour(isosurfaces=[level])
        except Exception:  # pragma: no cover - no such level
            positive = None
        try:
            negative = grid.contour(isosurfaces=[-level])
        except Exception:  # pragma: no cover - no such level
            negative = None

        if positive is not None and positive.n_points:
            positive = self._refine(positive, quality)
            self._actors.append(
                self.plotter.add_mesh(
                    positive,
                    color=_POS_COLOR,
                    opacity=_OPACITY,
                    smooth_shading=True,
                    name="isosurface-positive",
                    reset_camera=False,
                )
            )
        if negative is not None and negative.n_points:
            negative = self._refine(negative, quality)
            self._actors.append(
                self.plotter.add_mesh(
                    negative,
                    color=_NEG_COLOR,
                    opacity=_OPACITY,
                    smooth_shading=True,
                    name="isosurface-negative",
                    reset_camera=False,
                )
            )

    @staticmethod
    def _refine(mesh: pv.PolyData, quality: str) -> pv.PolyData:
        """Adjust the mesh so ``quality`` mostly controls the smoothness.

        A marching-cubes contour is coarse: all of its triangles lie in the
        original level set, so re-tessellating it *linearly* would only make
        it denser without looking any smoother. Instead:

        - ``"medium"`` keeps the raw contour;
        - ``"high"`` applies an approximating Loop subdivision, which rounds
          the facets into a clean, more fluid surface (Loop was chosen over
          the interpolating butterfly scheme, whose fixed stencil produced
          wrinkle-like ripples on these meshes);
        - ``"low"`` is kept for API compatibility (a light decimation).
        """
        if quality == "low":
            # Fewer triangles: decimate when the mesh is not tiny.
            if mesh.n_points > 2000:
                mesh = mesh.decimate(0.5)
        elif quality == "high":
            nsub = (
                _HIGH_SUBDIVIDE_NSUB
                if mesh.n_points <= _HIGH_SUBDIVIDE_POINT_LIMIT
                else _HIGH_SUBDIVIDE_NSUB_LARGE
            )
            mesh = mesh.subdivide(nsub=nsub, subfilter="loop")
        return mesh
