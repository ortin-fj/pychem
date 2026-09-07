"""Molecular renderer: atoms as spheres and bonds as cylinders.

Atoms are grouped by element: each group is drawn with a single ``glyph`` of
spheres scaled to the element's van der Waals radius, in its CPK color. Bonds
are guessed from covalent radii and drawn as a single tube (cylinder) actor.

The renderer accepts any *plotter* compatible with the PyVista API
(``add_mesh``, ``remove_actor``, ``reset_camera``): either a pyvistaqt
``QtInteractor`` or a ``pv.Plotter(off_screen=True)``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import pyvista as pv

from pychem.core.bonds import guess_bonds
from pychem.core.elements import element
from pychem.core.molecule import Molecule

if TYPE_CHECKING:
    from pyvista import Plotter

_SPHERE_THETA = 24
_SPHERE_PHI = 16
_TUBE_RESOLUTION = 12

# Lower tessellation for the selection halo: fewer, subtler lines.
_SELECTION_THETA = 16
_SELECTION_PHI = 10

# Measurement overlay visuals (Bonds/Angles/Dihedrals).
_MEAS_RING_COLOR = (0.9, 0.62, 0.0)  # gold wireframe around the involved atoms
_MEAS_CENTRAL_COLOR = (0.95, 0.12, 0.55)  # translucent fill on central atoms
_MEAS_CENTRAL_OPACITY = 0.55
_MEAS_LABEL_FONT_SIZE = 16

# Atom ID labels (Tools > Show ID label): plain text (no background box). The
# green matches the side-panel titles (#87D68D); atoms whose CPK colour is too
# light (e.g. white hydrogen) get black text instead, so the number stays
# readable on top of the sphere.
_ID_LABEL_FONT_SIZE = 12
_ID_LABEL_COLOR = (0.53, 0.84, 0.55)  # ~ #87D68D
_ID_LABEL_BLACK = (0.0, 0.0, 0.0)
_ID_LABEL_LIGHT_CPK = 190.0  # CPK luminance above which green is unreadable

# Maximum movement (px) between press and release to be considered a "click"
# (above this it is a drag to rotate the camera).
_CLICK_TOLERANCE = 6


def _is_click(
    press: tuple[int, int],
    release: tuple[int, int],
    tolerance: int = _CLICK_TOLERANCE,
) -> bool:
    """True if press/release are within ``tolerance`` pixels of each other."""
    dx = abs(release[0] - press[0])
    dy = abs(release[1] - press[1])
    return dx <= tolerance and dy <= tolerance


def _rgb(color: tuple[int, int, int]) -> tuple[float, float, float]:
    return tuple(c / 255.0 for c in color)  # type: ignore[return-value]


@dataclass
class RenderOptions:
    """Visual settings for the representation of a molecule."""

    atom_scale: float = 0.30  # fraction of the van der Waals radius
    bond_radius: float = 0.10  # Å
    bond_color: tuple[float, float, float] = (0.65, 0.65, 0.65)
    background: tuple[float, float, float] = (1.0, 1.0, 1.0)
    show_bonds: bool = True
    smooth_shading: bool = True

    # Selection halo: subtle thin wireframe around the selected atom.
    selection_scale: float = 1.25  # multiplier over the atom radius
    selection_color: tuple[float, float, float] = (0.1, 0.1, 0.1)
    selection_line_width: int = 1


class MolecularRenderer:
    """Draws a molecule on a plotter and manages its actors."""

    def __init__(self, plotter: "Plotter", options: RenderOptions | None = None) -> None:
        self.plotter = plotter
        self.options = options or RenderOptions()
        self._actors: list[object] = []
        self._atom_actors: list[object] = []
        self._molecule: Molecule | None = None
        self._bonds: np.ndarray | None = None
        self._selection_actor: object | None = None
        self.selected_index: int | None = None
        self._meas_actors: list[object] = []
        self._id_label_actors: list[object] = []
        self._show_id_labels = False
        self._pick_callback = None
        self._empty_callback = None
        self._iren = None
        self._press_observer: int | None = None
        self._release_observer: int | None = None
        self._press_xy: tuple[int, int] | None = None

    # -- actor management ---------------------------------------------------

    def clear(self) -> None:
        """Remove all actors created by this renderer."""
        self.clear_measurement()
        self.clear_id_labels()
        for actor in self._actors:
            try:
                self.plotter.remove_actor(actor)
            except Exception:  # pragma: no cover - actor already removed
                pass
        if self._selection_actor is not None:
            try:
                self.plotter.remove_actor(self._selection_actor)
            except Exception:  # pragma: no cover
                pass
        self._actors = []
        self._atom_actors = []
        self._selection_actor = None
        self.selected_index = None
        self._molecule = None
        self._bonds = None

    def show(
        self,
        molecule: Molecule,
        options: RenderOptions | None = None,
        *,
        bonds: np.ndarray | None = None,
    ) -> None:
        """Clear the scene and show ``molecule``. Resets the camera.

        ``bonds`` optionally overrides the connectivity (e.g. computed with
        hysteresis during trajectory playback); when ``None`` the bonds are
        guessed from the current geometry.
        """
        opts = replace(self.options, **{}) if options is None else options
        self.clear()
        self.add_molecule(molecule, opts, bonds=bonds)
        if self.options.background is not None:
            try:
                self.plotter.set_background(self.options.background)
            except Exception:  # pragma: no cover
                pass

    # -- actor construction -------------------------------------------------

    def add_molecule(
        self,
        molecule: Molecule,
        options: RenderOptions | None = None,
        *,
        bonds: np.ndarray | None = None,
    ) -> None:
        opts = options or self.options
        symbols = molecule.symbols
        coords = molecule.coords
        self._molecule = molecule

        self._atom_actors = self._add_atoms(symbols, coords, opts)
        self._actors.extend(self._atom_actors)

        self._bonds = bonds
        if bonds is None and opts.show_bonds and molecule.n_atoms > 1:
            bonds = guess_bonds(molecule)
            self._bonds = bonds
        if bonds is not None and bonds.size:
            self._actors.append(self._add_bonds(coords, bonds, opts))

        # Rebuild the atom-ID overlay for the new molecule/pose when enabled.
        if self._show_id_labels:
            self._refresh_id_labels()

    def _add_atoms(
        self,
        symbols: list[str],
        coords: np.ndarray,
        opts: RenderOptions,
    ) -> list[object]:
        """One actor per chemical element (grouped with a sphere glyph)."""
        sphere = pv.Sphere(
            theta_resolution=_SPHERE_THETA,
            phi_resolution=_SPHERE_PHI,
            radius=1.0,
        )
        actors: list[object] = []
        for symbol in sorted(set(symbols)):
            mask = np.array([s == symbol for s in symbols])
            pts = pv.PolyData(np.asarray(coords[mask], dtype=float))
            radius = element(symbol).vdw_radius * opts.atom_scale
            pts["scale"] = np.full(pts.n_points, radius, dtype=float)
            glyph = pts.glyph(
                orient=False, scale=True, factor=1.0, geom=sphere
            )
            color = _rgb(element(symbol).color)
            actor = self.plotter.add_mesh(
                glyph,
                color=color,
                smooth_shading=opts.smooth_shading,
                name=f"atoms-{symbol}",
                # Do not reframe the camera when (re)building actors, e.g.
                # during trajectory playback; camera reset is explicit.
                reset_camera=False,
            )
            actors.append(actor)
        return actors

    def _add_bonds(
        self,
        coords: np.ndarray,
        bonds: np.ndarray,
        opts: RenderOptions,
    ) -> object:
        """Cylinders between bonded pairs as a single tube actor."""
        line_cells = np.empty((bonds.shape[0], 3), dtype=np.int64)
        line_cells[:, 0] = 2
        line_cells[:, 1:] = bonds
        poly = pv.PolyData(coords, lines=line_cells)
        tubes = poly.tube(radius=opts.bond_radius, n_sides=_TUBE_RESOLUTION)
        return self.plotter.add_mesh(
            tubes,
            color=opts.bond_color,
            smooth_shading=opts.smooth_shading,
            name="bonds",
            reset_camera=False,
        )

    # -- atom ID labels ------------------------------------------------------

    def set_id_labels(self, visible: bool) -> None:
        """Show/hide the per-atom ID numbers (persists across molecule swaps)."""
        self._show_id_labels = bool(visible)
        self._refresh_id_labels()

    def clear_id_labels(self) -> None:
        """Remove the atom-ID label actors."""
        for actor in self._id_label_actors:
            try:
                self.plotter.remove_actor(actor)
            except Exception:  # pragma: no cover - already removed
                pass
        self._id_label_actors = []

    def _refresh_id_labels(self) -> None:
        """(Re)draw the atom-ID labels over the current molecule, if enabled.

        Plain numbers (no text box) placed at every atom. The text is green
        like the side-panel titles except on very light atoms (white
        hydrogen...) where black guarantees legibility. Two label actors keep
        the two colours.
        """
        self.clear_id_labels()
        if not self._show_id_labels or self._molecule is None:
            return
        green: list[int] = []
        black: list[int] = []
        for i, symbol in enumerate(self._molecule.symbols):
            cpk = element(symbol).color
            lum = 0.299 * cpk[0] + 0.587 * cpk[1] + 0.114 * cpk[2]
            (black if lum >= _ID_LABEL_LIGHT_CPK else green).append(i)
        coords = self._molecule.coords
        for indices, color, name in (
            (green, _ID_LABEL_COLOR, "id-labels-green"),
            (black, _ID_LABEL_BLACK, "id-labels-black"),
        ):
            if not indices:
                continue
            actor = self.plotter.add_point_labels(
                np.asarray(coords[indices], dtype=float),
                [str(i + 1) for i in indices],
                font_size=_ID_LABEL_FONT_SIZE,
                bold=True,
                text_color=color,
                show_points=False,
                always_visible=True,
                # No background shape: only the plain number must be drawn
                # (pyvista would otherwise add a rounded box behind the text).
                shape=None,
                name=name,
                reset_camera=False,
            )
            # Keep the numbers out of the atom picking loop.
            set_pickable = getattr(actor, "SetPickable", None)
            if callable(set_pickable):
                try:
                    set_pickable(0)
                except Exception:  # pragma: no cover
                    pass
            self._id_label_actors.append(actor)

    # -- atom selection -----------------------------------------------------

    @property
    def molecule(self) -> Molecule | None:
        return self._molecule

    def enable_atom_picking(self, callback, empty_callback=None) -> None:
        """Enable atom picking with the left mouse button.

        It does not replace the camera rotation: it adds left-click
        observers. ``callback(index)`` receives the picked atom index and
        ``empty_callback()`` is called when a *click* (no drag) lands on
        empty space or on a bond.

        To tell a "click" from a "drag" (camera rotation) the press/release
        events are observed: if the mouse barely moved between them it is a
        click; if it moved, it is a drag and the selection is left untouched.
        """
        self._pick_callback = callback
        self._empty_callback = empty_callback
        try:
            self.plotter.enable_surface_point_picking(
                callback=self._on_surface_point_picked,
                show_message=False,
                show_point=False,
                left_clicking=True,
                pickable_window=False,
            )
        except Exception:  # pragma: no cover - unsupported plotters
            self._pick_callback = None
            self._empty_callback = None
            return

        self._iren = getattr(self.plotter, "iren", None)
        if self._iren is not None:
            self._press_observer = self._iren.add_observer(
                "LeftButtonPressEvent", self._on_left_button_press
            )
            self._release_observer = self._iren.add_observer(
                "LeftButtonReleaseEvent", self._on_left_button_release
            )

    def disable_atom_picking(self) -> None:
        """Disable picking and remove the click observers."""
        self._pick_callback = None
        self._empty_callback = None
        if self._iren is not None:
            if self._press_observer is not None:
                try:
                    self._iren.remove_observer(self._press_observer)
                except Exception:  # pragma: no cover
                    pass
            if self._release_observer is not None:
                try:
                    self._iren.remove_observer(self._release_observer)
                except Exception:  # pragma: no cover
                    pass
        self._press_observer = None
        self._release_observer = None
        self._press_xy = None
        disable = getattr(self.plotter, "disable_picking", None)
        if callable(disable):
            disable()

    # -- click vs drag ------------------------------------------------------

    @staticmethod
    def _event_position(iren):
        """Current (x, y) pointer position in the interactor."""
        for name in ("get_event_position", "GetEventPosition"):
            getter = getattr(iren, name, None)
            if callable(getter):
                try:
                    pos = getter()
                except Exception:  # pragma: no cover
                    continue
                return tuple(int(v) for v in pos)
        return None

    def _on_left_button_press(self, *_args) -> None:
        iren = self._iren
        if iren is None:
            return
        pos = self._event_position(iren)
        if pos is not None:
            self._press_xy = pos

    def _on_left_button_release(self, *_args) -> None:
        """On release: if it was a click on empty space, clear the selection.

        To know whether the click landed on an atom, ``picked_actor`` (left by
        the press picker, reliable) is used instead of doing a second pick on
        release (which can fail and wrongly clear a just-made selection).
        """
        if self._press_xy is None:
            return
        press_x, press_y = self._press_xy
        self._press_xy = None

        iren = self._iren
        if iren is None:
            return
        pos = self._event_position(iren)
        release_x, release_y = pos if pos is not None else (press_x, press_y)

        # If there was movement this is not a click but a drag (rotate the
        # camera): in that case the selection is kept.
        if not _is_click((press_x, press_y), (release_x, release_y)):
            return

        # It is a click: if the press was NOT on an atom (background/bond),
        # clear the selection. If it was on an atom, the picking callback
        # already made the selection, so nothing is done here.
        actor = getattr(self.plotter, "picked_actor", None)
        if (actor is None or actor not in self._atom_actors) and self._empty_callback is not None:
            self._empty_callback()

    def _on_surface_point_picked(self, point) -> None:
        actor = getattr(self.plotter, "picked_actor", None)
        if point is None or self._molecule is None:
            return
        # Only select when pressing on an atom (not on bonds/background)
        if actor not in self._atom_actors:
            return
        index = self.nearest_atom(point)
        if index is not None and self._pick_callback is not None:
            self._pick_callback(int(index))

    def nearest_atom(self, point, tolerance: float | None = None) -> int | None:
        """Index of the atom closest to ``point`` (or ``None`` if too far).

        The default threshold is ~1.5 times the visual radius of the closest
        atom, so clicking in the gap between atoms selects nothing.
        """
        if self._molecule is None:
            return None
        p = np.asarray(point, dtype=float)
        d2 = np.sum((self._molecule.coords - p) ** 2, axis=1)
        index = int(np.argmin(d2))
        symbol = self._molecule.symbols[index]
        visual_radius = element(symbol).vdw_radius * self.options.atom_scale
        limit = tolerance if tolerance is not None else max(0.6, visual_radius * 1.6)
        return index if float(np.sqrt(d2[index])) <= limit else None

    def neighbor_indices(self, index: int) -> np.ndarray:
        """Indices of the atoms bonded to ``index`` (per the connectivity)."""
        if self._bonds is None or self._bonds.size == 0:
            return np.empty(0, dtype=np.int64)
        rows = self._bonds[(self._bonds == index).any(axis=1)]
        partners = np.where(rows == index, rows[:, ::-1], rows)[:, 1]
        return np.asarray(partners, dtype=np.int64)

    def select_atom(self, index: int | None) -> None:
        """Mark an atom with a subtle selection halo, or clear if None."""
        if self._selection_actor is not None:
            try:
                self.plotter.remove_actor(self._selection_actor)
            except Exception:  # pragma: no cover
                pass
            self._selection_actor = None
        self.selected_index = None
        if index is None or self._molecule is None:
            return

        opts = self.options
        atom = self._molecule.get_atom(index)
        radius = element(atom.symbol).vdw_radius * opts.atom_scale * opts.selection_scale
        halo = pv.Sphere(
            center=atom.position,
            radius=max(0.25, radius),
            theta_resolution=_SELECTION_THETA,
            phi_resolution=_SELECTION_PHI,
        )
        self._selection_actor = self.plotter.add_mesh(
            halo,
            style="wireframe",
            color=opts.selection_color,
            line_width=opts.selection_line_width,
            name="selection",
            # Never reframe the camera for the halo, otherwise selecting a
            # new atom would subtly shift the whole molecule on screen.
            reset_camera=False,
        )
        self.selected_index = index

    # -- measurement overlays ------------------------------------------------

    def clear_measurement(self) -> None:
        """Remove the measurement markers/label actors."""
        for actor in self._meas_actors:
            try:
                self.plotter.remove_actor(actor)
            except Exception:  # pragma: no cover - already removed
                pass
        self._meas_actors = []

    def set_measurement(
        self,
        indices: list[int],
        central: tuple[int, ...] = (),
        label: str | None = None,
        anchor: np.ndarray | None = None,
    ) -> None:
        """Draw markers (and an optional label) for an active measurement.

        ``indices`` are the involved atoms in selection order; ``central``
        lists the ones highlighted as central (the angle/dihedral pivots). The
        markers are simple wireframe spheres like the Atom Properties
        selection halo, but coloured: gold on every involved atom and magenta
        on the central ones, which additionally get a translucent fill so they
        stand out. When ``label`` and ``anchor`` are given, the text is drawn
        in the viewer, anchored to the world point ``anchor`` and always kept
        readable.
        """
        self.clear_measurement()
        molecule = self._molecule
        if molecule is None or not indices:
            return
        opts = self.options
        central_set = {int(i) for i in central}
        for i in indices:
            i = int(i)
            if not 0 <= i < molecule.n_atoms:
                self.clear_measurement()
                return
            atom = molecule.get_atom(i)
            is_central = i in central_set
            # Same geometry as the Atom Properties selection halo (kept thin
            # and simple); only the colour differs per role.
            ring_radius = max(
                0.25,
                element(atom.symbol).vdw_radius
                * opts.atom_scale
                * opts.selection_scale,
            )
            color = _MEAS_CENTRAL_COLOR if is_central else _MEAS_RING_COLOR
            ring = pv.Sphere(
                center=atom.position,
                radius=ring_radius,
                theta_resolution=_SELECTION_THETA,
                phi_resolution=_SELECTION_PHI,
            )
            self._meas_actors.append(
                self.plotter.add_mesh(
                    ring,
                    style="wireframe",
                    color=color,
                    line_width=opts.selection_line_width,
                    name=f"meas-ring-{i}",
                    reset_camera=False,
                )
            )
            if is_central:
                # A subtle translucent fill makes the pivot atoms unmistakable.
                fill_radius = max(
                    0.22,
                    element(atom.symbol).vdw_radius * opts.atom_scale,
                )
                core = pv.Sphere(
                    center=atom.position,
                    radius=fill_radius,
                    theta_resolution=_SPHERE_THETA,
                    phi_resolution=_SPHERE_PHI,
                )
                self._meas_actors.append(
                    self.plotter.add_mesh(
                        core,
                        color=_MEAS_CENTRAL_COLOR,
                        opacity=_MEAS_CENTRAL_OPACITY,
                        smooth_shading=True,
                        name=f"meas-central-{i}",
                        reset_camera=False,
                    )
                )
        if label is not None and anchor is not None:
            actor = self.plotter.add_point_labels(
                np.asarray([anchor], dtype=float),
                [str(label)],
                font_size=_MEAS_LABEL_FONT_SIZE,
                bold=True,
                text_color=(0.08, 0.08, 0.08),
                shape="rounded_rect",
                shape_color=(1.0, 0.98, 0.87),
                fill_shape=True,
                margin=5,
                show_points=False,
                always_visible=True,
                name="measurement-label",
                reset_camera=False,
            )
            self._meas_actors.append(actor)
