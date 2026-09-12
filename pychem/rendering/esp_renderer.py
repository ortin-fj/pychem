"""Electrostatic-potential (ESP) maps with PyVista.

An ESP map is the standard two-field picture: the *electron density* is
contoured at a small envelope level (the shape of the molecule's electron
cloud) and that semi-transparent surface is coloured with the
*electrostatic potential* sampled on it. A diverging colormap then reads
directly as "lower potential" (red, electron-rich regions that attract a
positive probe) versus "higher potential" (blue, electron-poor regions).

The potential is sampled with :meth:`VolumeData.sample_at`, so the density
and the potential cubes do not even need to share the same grid.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv

from pychem.core.volume import VolumeData
from pychem.rendering.volume_renderer import refine_mesh, to_image_data

# Diverging map for the potential: low (red) -> zero (pale) -> high (blue).
CMAP = "coolwarm_r"
OPACITY = 0.6
SCALAR_BAR_TITLE = "ESP (a.u.)"

# Quality used for the envelope surface (there is no quality selector in the
# ESP menu entry, so the smoother setting is always used).
QUALITY = "high"


class ElectrostaticRenderer:
    """Draws the electron-density envelope coloured by the potential."""

    def __init__(self, plotter) -> None:
        self.plotter = plotter
        self._actors: list[object] = []
        self._scalar_bar = SCALAR_BAR_TITLE
        # Last drawn state (kept for the status bar and for tests).
        self.level: float | None = None
        self.clim: tuple[float, float] | None = None
        self.surface: pv.PolyData | None = None

    def clear(self) -> None:
        """Remove the ESP surface and its colour bar."""
        for actor in self._actors:
            try:
                self.plotter.remove_actor(actor)
            except Exception:  # pragma: no cover - already removed
                pass
        self._actors = []
        try:
            self.plotter.remove_scalar_bar(self._scalar_bar)
        except Exception:  # pragma: no cover - nothing to remove
            pass
        self.level = None
        self.clim = None
        self.surface = None

    def show(
        self,
        density: VolumeData,
        potential: VolumeData,
        *,
        level: float | None = None,
        quality: str = QUALITY,
        cmap: str = CMAP,
        opacity: float = OPACITY,
    ) -> bool:
        """Replace the ESP overlay with the map of ``density``/``potential``.

        Returns ``False`` (drawing nothing) when the density does not reach
        the requested envelope level, i.e. there is no surface to colour.
        """
        self.clear()

        if level is None:
            level = density.suggest_density_isovalue()
        level = float(level)
        if level <= 0.0 or density.max_abs <= 0.0:
            return False

        try:
            surface = to_image_data(density).contour(isosurfaces=[level])
        except Exception:  # pragma: no cover - no such level
            return False
        if surface is None or surface.n_points == 0:
            return False

        surface = refine_mesh(surface, quality)
        values = np.asarray(potential.sample_at(surface.points), dtype=float)
        surface.point_data[SCALAR_BAR_TITLE] = values

        low = float(values.min())
        high = float(values.max())
        if not np.isfinite(low) or not np.isfinite(high):
            low, high = 0.0, 0.0
        if low == high:  # a constant colour still needs a valid range
            low, high = low - 1e-9, high + 1e-9

        self._actors.append(
            self.plotter.add_mesh(
                surface,
                scalars=SCALAR_BAR_TITLE,
                cmap=cmap,
                clim=(low, high),
                opacity=opacity,
                smooth_shading=True,
                show_scalar_bar=False,
                reset_camera=False,
                name="esp",
            )
        )
        try:
            self.plotter.add_scalar_bar(
                title=SCALAR_BAR_TITLE,
                n_labels=5,
                fmt="%.3f",
                vertical=True,
            )
        except Exception:  # pragma: no cover - offscreen without a renderer
            pass

        self.level = level
        self.clim = (low, high)
        self.surface = surface
        return True
