"""Molecular renderer: atoms as spheres and bonds as cylinders.

Atoms are grouped by element: each group is drawn with a single ``glyph`` of
spheres scaled to the element's van der Waals radius, in its CPK color. Bonds
are guessed from covalent radii and drawn as a single tube (cylinder) actor.

The renderer accepts any *plotter* compatible with the PyVista API
(``add_mesh``, ``remove_actor``, ``reset_camera``): either a pyvistaqt
``QtInteractor`` or a ``pv.Plotter(off_screen=True)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import pyvista as pv
import vtk

from pychem.core.bonds import guess_bonds
from pychem.core.elements import element
from pychem.core.molecule import Molecule

if TYPE_CHECKING:
    from pyvista import Plotter

_SPHERE_THETA = 24
_SPHERE_PHI = 16
# Large structures fall back to a lighter sphere: with thousands of atoms the
# shading cost dominates the silhouette, and the GPU draws one instance per
# atom (see ``_sphere_resolution``).
_SPHERE_LOW_THETA = 12
_SPHERE_LOW_PHI = 8
_SPHERE_LOW_ATOM_COUNT = 2000
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

# Acceptance rule of ``nearest_atom``: an atom is considered "hit" when the
# point is within this factor over its drawn radius (or a small minimum, so
# tiny atoms stay reachable).
_NEAREST_ATOM_MIN_RADIUS = 0.6
_NEAREST_ATOM_RADIUS_FACTOR = 1.6

# ``pick_at`` first intersects the ray with the *drawn* spheres, so the result
# always matches what is on screen. Only when the ray misses everything does
# it fall back to picking the front-most atom whose projected centre is
# within this many pixels, which keeps small atoms clickable without
# inventing hits on top of a neighbouring atom.
_PICK_PIXEL_RADIUS = 8.0

_EPS = 1e-9


def _is_click(
    press: tuple[int, int],
    release: tuple[int, int],
    tolerance: int = _CLICK_TOLERANCE,
) -> bool:
    """True if press/release are within ``tolerance`` pixels of each other."""
    dx = abs(release[0] - press[0])
    dy = abs(release[1] - press[1])
    return dx <= tolerance and dy <= tolerance


def screen_delta_to_world(
    direction,
    view_up,
    world_per_pixel: float,
    dx_pixels: float,
    dy_pixels: float,
) -> np.ndarray:
    """Map a screen-space drag (pixels) to a world displacement (Å).

    ``direction`` is the camera's direction of projection and ``view_up`` its
    up vector: the atom moves in the plane perpendicular to ``direction``.
    Display coordinates grow upwards, so ``dy_pixels`` is positive when the
    pointer moves up (the displacement follows it).
    """
    d = np.asarray(direction, dtype=float)
    dnorm = float(np.linalg.norm(d))
    if dnorm > _EPS:
        d = d / dnorm
    up = np.asarray(view_up, dtype=float)
    right = np.cross(d, up)
    rnorm = float(np.linalg.norm(right))
    if rnorm <= _EPS:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / rnorm
    up_screen = np.cross(right, d)
    return right * (dx_pixels * world_per_pixel) + up_screen * (
        dy_pixels * world_per_pixel
    )


def _sphere_resolution(n_atoms: int) -> tuple[int, int]:
    """Sphere tessellation ``(theta, phi)`` suited to the number of atoms."""
    if n_atoms > _SPHERE_LOW_ATOM_COUNT:
        return _SPHERE_LOW_THETA, _SPHERE_LOW_PHI
    return _SPHERE_THETA, _SPHERE_PHI


def ray_sphere_first_hit(
    origin,
    direction,
    centers: np.ndarray,
    radii: np.ndarray,
) -> int | None:
    """Index of the sphere a ray hits first, or ``None`` when it misses all.

    ``direction`` must be a unit vector. The quadratic is solved for every
    sphere at once, which makes the pick independent from VTK's pickers: on
    an 11000-atom structure this takes ~0.2 ms, whereas the ``vtkCellPicker``
    that PyVista's surface picking installs needed ~0.57 s per click (it
    builds a cell locator over the whole triangle mesh).

    Only hits in front of the ray origin are considered; should the eye sit
    inside a sphere, that sphere's far intersection is used instead.
    """
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if centers.size == 0:
        return None
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    offset = origin - centers
    b = offset @ direction
    c = np.einsum("ij,ij->i", offset, offset) - radii * radii
    discriminant = b * b - c
    inside = discriminant >= 0.0
    if not inside.any():
        return None
    root = np.sqrt(np.maximum(discriminant, 0.0))
    near = -b - root
    far = -b + root
    distance = np.where(near > 0.0, near, far)
    distance = np.where(inside & (distance > 0.0), distance, np.inf)
    index = int(np.argmin(distance))
    return index if np.isfinite(distance[index]) else None


def _rgb(color: tuple[int, int, int]) -> tuple[float, float, float]:
    return tuple(c / 255.0 for c in color)  # type: ignore[return-value]


def _make_unpickable(actor) -> None:
    """Keep an overlay actor out of the picking loop.

    Highlights (selection halo, measurement rings/fills, labels) sit on top of
    the atoms; if they were pickable they would swallow clicks aimed at the
    atom below them.
    """
    setter = getattr(actor, "SetPickable", None)
    if callable(setter):
        try:
            setter(0)
        except Exception:  # pragma: no cover - actor without pickability
            pass


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
        self._atom_radii: np.ndarray | None = None
        self._molecule: Molecule | None = None
        self._bonds: np.ndarray | None = None
        self._selection_actor: object | None = None
        self.selected_index: int | None = None
        self._meas_actors: list[object] = []
        self._id_label_actors: list[object] = []
        self._show_id_labels = False
        self._pick_callback = None
        self._empty_callback = None
        self._click_callback = None
        self._last_pick_index: int | None = None
        self._iren = None
        self._press_observer: int | None = None
        self._release_observer: int | None = None
        self._press_xy: tuple[int, int] | None = None
        # Atom-dragging interaction (Edit section). While it is active the
        # renderer aborts the left press on a movable atom, so the camera does
        # not rotate, and translates it following the pointer.
        self._move_callback = None
        self._move_select = None
        self._move_click = None
        self._move_finish = None
        self._movable: set[int] | None = None
        self._drag_index: int | None = None
        self._drag_origin: np.ndarray | None = None
        self._drag_press: tuple[int, int] | None = None
        self._move_candidate: int | None = None
        self._raw_iren = None
        self._move_press_observer: int | None = None
        self._move_motion_observer: int | None = None
        self._move_release_observer: int | None = None
        self._move_release_style = None
        self._move_release_style_observer: int | None = None

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
        self._atom_radii = None
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

        # An explicit connectivity on the model wins over guessing; the
        # ``bonds`` argument still overrides it (e.g. dynamic trajectory
        # bonds). This is what lets the Edit section hand-author bonds.
        if bonds is None:
            bonds = molecule.bonds
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
        """Every atom as a *single* GPU-instanced sphere actor.

        ``vtkGlyph3DMapper`` uploads the sphere once and has the GPU draw it
        per atom, so the geometry is never expanded: an 11000-atom structure
        used to become a ~7.4M-triangle, ~440 MB merged mesh, and now stays a
        single sphere plus one scale/colour per point. Atom colours are
        per-point data, which replaces the previous actor-per-element
        grouping.
        """
        n_atoms = len(symbols)
        if n_atoms == 0:  # pragma: no cover - degenerate input
            self._atom_radii = np.empty(0, dtype=float)
            return []
        theta, phi = _sphere_resolution(n_atoms)
        coords = np.asarray(coords, dtype=float)
        self._atom_radii = np.array(
            [element(s).vdw_radius * opts.atom_scale for s in symbols],
            dtype=float,
        )

        points = pv.PolyData(coords)
        points["scale"] = self._atom_radii
        points["colors"] = np.array(
            [_rgb(element(s).color) + (1.0,) for s in symbols], dtype=float
        )

        sphere = pv.Sphere(
            theta_resolution=theta,
            phi_resolution=phi,
            radius=1.0,
        )
        mapper = vtk.vtkGlyph3DMapper()
        mapper.SetInputData(points)
        mapper.SetSourceData(sphere)
        mapper.ScalingOn()
        mapper.SetScaleArray("scale")
        mapper.SetScaleModeToScaleByMagnitude()
        mapper.SetColorModeToDirectScalars()
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray("colors")
        mapper.Update()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        if opts.smooth_shading:
            actor.GetProperty().SetInterpolationToPhong()
        else:  # pragma: no cover - only used if the option is turned off
            actor.GetProperty().SetInterpolationToFlat()

        # NOTE: ``Plotter.add_actor`` returns ``(actor, property)``, not the
        # actor; keep the first element so ``clear`` can remove it again.
        added, _properties = self.plotter.add_actor(
            actor,
            name="atoms",
            # Do not reframe the camera when (re)building actors, e.g.
            # during trajectory playback; camera reset is explicit.
            reset_camera=False,
        )
        return [added]

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
            _make_unpickable(actor)
            self._id_label_actors.append(actor)

    # -- atom selection -----------------------------------------------------

    @property
    def molecule(self) -> Molecule | None:
        return self._molecule

    @property
    def bonds(self) -> np.ndarray | None:
        """Currently drawn connectivity (effective, after guessing)."""
        return self._bonds

    def enable_atom_picking(
        self, callback, empty_callback=None, click_callback=None
    ) -> None:
        """Enable atom picking with the left mouse button.

        It does not replace the camera rotation: it adds left-click
        observers. ``callback(index)`` receives the picked atom index and
        ``empty_callback()`` is called when a *click* (no drag) lands on
        empty space or on a bond.

        To tell a "click" from a "drag" (camera rotation) the press/release
        events are observed: if the mouse barely moved between them it is a
        click; if it moved, it is a drag and the selection is left untouched.

        ``click_callback(index)`` (if given) fires only on a genuine click on
        an atom, once the selection has been applied; it lets the Edit section
        react to clicks without interfering with camera drags.

        The atom under the pointer is found analytically (see
        :meth:`pick_at`) instead of through PyVista's
        ``enable_surface_point_picking``: that helper installs a
        ``vtkCellPicker``, which builds a cell locator over the whole atom
        mesh and cost ~0.57 s per click on a large structure, freezing the UI
        for half a second on every press.
        """
        self._pick_callback = callback
        self._empty_callback = empty_callback
        self._click_callback = click_callback
        self._iren = getattr(self.plotter, "iren", None)
        if self._iren is None:  # pragma: no cover - plotter without iren
            self._pick_callback = None
            self._empty_callback = None
            return
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
        self._click_callback = None
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
        self._last_pick_index = None
        disable = getattr(self.plotter, "disable_picking", None)
        if callable(disable):  # pragma: no cover - nothing to disable now
            try:
                disable()
            except Exception:
                pass

    # -- analytic picking ----------------------------------------------------

    def _camera_frame(self):
        """Active camera plus its screen basis, or ``None`` if unusable.

        Returns ``(camera, eye, direction, right, up, width, height)`` where
        ``direction`` points from the eye into the scene and all three axes
        are unit vectors. Reads the poked renderer so the maths stays correct
        if the molecule is ever shown in a sub-plot.
        """
        renderer = None
        iren = getattr(self.plotter, "iren", None)
        getter = getattr(iren, "get_poked_renderer", None)
        if callable(getter):
            try:
                renderer = getter()
            except Exception:  # pragma: no cover - no poked renderer
                renderer = None
        if renderer is None:
            renderer = self.plotter.renderer
        camera = renderer.GetActiveCamera()
        width, height = (float(v) for v in renderer.GetSize())
        if width <= 0.0 or height <= 0.0:  # pragma: no cover - no viewport
            return None
        direction = np.asarray(camera.GetDirectionOfProjection(), dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm <= _EPS:  # pragma: no cover - degenerate camera
            return None
        direction = direction / norm
        up = np.asarray(camera.GetViewUp(), dtype=float)
        right = np.cross(direction, up)
        rnorm = float(np.linalg.norm(right))
        if rnorm <= _EPS:  # pragma: no cover - degenerate camera
            return None
        right = right / rnorm
        up = np.cross(right, direction)
        eye = np.asarray(camera.GetPosition(), dtype=float)
        return camera, eye, direction, right, up, width, height

    def _ray_through_pixel(self, x: float, y: float):
        """World-space ray ``(origin, unit direction)`` through a pixel.

        Pixel coordinates follow the VTK convention: origin at the
        bottom-left corner of the render window, y growing upwards. Both
        perspective and parallel projections are supported.
        """
        frame = self._camera_frame()
        if frame is None:  # pragma: no cover - no usable camera
            return None
        camera, eye, direction, right, up, width, height = frame

        # Normalised device coordinates in [-1, 1], y upwards.
        ndc_x = 2.0 * float(x) / width - 1.0
        ndc_y = 2.0 * float(y) / height - 1.0
        distance = float(camera.GetDistance())
        if camera.GetParallelProjection():
            half_h = float(camera.GetParallelScale())
            half_w = half_h * width / height
            origin = eye + right * (ndc_x * half_w) + up * (ndc_y * half_h)
            return origin, direction
        half_h = (
            math.tan(math.radians(float(camera.GetViewAngle())) / 2.0) * distance
        )
        half_w = half_h * width / height
        target = (
            eye
            + direction * distance
            + right * (ndc_x * half_w)
            + up * (ndc_y * half_h)
        )
        vector = target - eye
        vnorm = float(np.linalg.norm(vector))
        if vnorm <= _EPS:  # pragma: no cover - degenerate camera
            return None
        return eye, vector / vnorm

    def _pick_nearest_projected(self, origin, direction) -> int | None:
        """Front-most atom whose screen projection is near the ray.

        Fallback used only when the ray misses every drawn sphere: it keeps
        very small atoms clickable on large structures. Distances are measured
        in pixels (so the threshold does not depend on zoom or depth) and the
        front-most candidate wins, so an atom hidden behind another is never
        preferred.
        """
        frame = self._camera_frame()
        if frame is None or self._molecule is None:  # pragma: no cover
            return None
        camera, _eye, _direction, _right, _up, _width, height = frame
        offset = np.asarray(self._molecule.coords, dtype=float) - origin
        depth = offset @ direction
        in_front = depth > _EPS
        perpendicular = np.linalg.norm(offset - depth[:, None] * direction, axis=1)
        # World size of one pixel at each atom's depth.
        if camera.GetParallelProjection():
            world_per_pixel = 2.0 * float(camera.GetParallelScale()) / height
        else:
            world_per_pixel = (
                2.0
                * math.tan(math.radians(float(camera.GetViewAngle())) / 2.0)
                * depth
                / height
            )
        pixel_distance = perpendicular / np.maximum(world_per_pixel, _EPS)
        candidates = in_front & (pixel_distance <= _PICK_PIXEL_RADIUS)
        if not candidates.any():
            return None
        nearest_front = np.argmin(np.where(candidates, depth, np.inf))
        return int(nearest_front)

    def pick_at(self, x: float, y: float) -> int | None:
        """Index of the atom under pixel ``(x, y)``, or ``None``.

        Two stages, both analytic and needing no VTK picker (~0.2 ms with
        11000 atoms):

        1. the ray is intersected with the spheres that are *actually drawn*,
           so hitting the visible surface of an atom always selects that atom
           (and never a neighbour in front of it);
        2. only when the ray misses everything, the front-most atom whose
           projected centre lies within a few pixels is used, so tiny atoms
           stay clickable.
        """
        if self._molecule is None or self._molecule.n_atoms == 0:
            return None
        if self._atom_radii is None:  # pragma: no cover - nothing drawn yet
            return None
        ray = self._ray_through_pixel(x, y)
        if ray is None:  # pragma: no cover - no usable camera
            return None
        origin, direction = ray
        index = ray_sphere_first_hit(
            origin, direction, self._molecule.coords, self._atom_radii
        )
        if index is not None:
            return index
        return self._pick_nearest_projected(origin, direction)

    # -- atom dragging (Edit section) ---------------------------------------

    def enable_atom_moving(
        self,
        move_callback,
        *,
        movable=None,
        select_callback=None,
        click_callback=None,
        finish_callback=None,
    ) -> None:
        """Let the user drag atoms with the left mouse button.

        ``move_callback(index, new_position)`` is called on every pointer move
        while dragging ``index``; the renderer never modifies the molecule
        itself. Only the atoms in ``movable`` (all of them when ``None``) can
        be grabbed. ``select_callback(index)`` fires when a grab starts,
        ``click_callback(index)`` on a click that did not move the atom, and
        ``finish_callback()`` once a drag ends.

        The left press is aborted for a grabbed atom so the camera does not
        rotate while moving it; pressing empty space (or a fixed atom) still
        rotates the view normally.
        """
        self.disable_atom_moving()
        self._move_callback = move_callback
        self._move_select = select_callback
        self._move_click = click_callback
        self._move_finish = finish_callback
        self._movable = None if movable is None else {int(i) for i in movable}
        self._raw_iren = getattr(
            getattr(self.plotter, "iren", None), "interactor", None
        )
        if self._raw_iren is None:  # pragma: no cover - plotter without iren
            self._move_callback = None
            return
        self._move_press_observer = self._raw_iren.AddObserver(
            "LeftButtonPressEvent", self._on_move_press, 1.0
        )
        self._move_motion_observer = self._raw_iren.AddObserver(
            "MouseMoveEvent", self._on_move_motion, 1.0
        )
        self._move_release_observer = self._raw_iren.AddObserver(
            "LeftButtonReleaseEvent", self._on_move_release, 1.0
        )
        # Some interactor styles swallow the release event on the interactor
        # (the press and move events do arrive), so a plain click would never
        # be confirmed. The release is therefore also observed on the style
        # itself, exactly like PyVista does for its own release observers.
        # ``_on_move_release`` clears its state before acting, so receiving
        # the event twice is harmless.
        self._move_release_style = None
        self._move_release_style_observer = None
        style = getattr(getattr(self.plotter, "iren", None), "style", None)
        if style is not None:
            try:
                self._move_release_style_observer = style.AddObserver(
                    "LeftButtonReleaseEvent", self._on_move_release, 1.0
                )
                self._move_release_style = style
            except Exception:  # pragma: no cover - style without observers
                self._move_release_style_observer = None

    def set_movable(self, movable) -> None:
        """Change which atoms can be grabbed, keeping the observers in place.

        Used when the umbrella selection changes while its mode is active;
        re-registering the VTK observers from inside an event callback would
        be fragile, so only the index set is swapped.
        """
        if self._move_callback is None:
            return
        self._movable = None if movable is None else {int(i) for i in movable}

    def disable_atom_moving(self) -> None:
        """Stop the atom-dragging interaction and drop its observers."""
        raw = self._raw_iren
        if raw is not None:
            for tag in (
                self._move_press_observer,
                self._move_motion_observer,
                self._move_release_observer,
            ):
                if tag is not None:
                    try:
                        raw.RemoveObserver(tag)
                    except Exception:  # pragma: no cover - already gone
                        pass
        style = self._move_release_style
        if style is not None and self._move_release_style_observer is not None:
            try:
                style.RemoveObserver(self._move_release_style_observer)
            except Exception:  # pragma: no cover - already gone
                pass
        self._raw_iren = None
        self._move_press_observer = None
        self._move_motion_observer = None
        self._move_release_observer = None
        self._move_release_style = None
        self._move_release_style_observer = None
        self._move_callback = None
        self._move_select = None
        self._move_click = None
        self._move_finish = None
        self._movable = None
        self._drag_index = None
        self._drag_origin = None
        self._drag_press = None
        self._move_candidate = None

    @staticmethod
    def _abort_event(raw, tag) -> None:
        """Abort the in-flight VTK event so the camera style ignores it."""
        try:
            command = raw.GetCommand(tag) if tag is not None else None
            if command is not None:
                command.SetAbortFlag(1)
        except Exception:  # pragma: no cover - best effort
            pass

    def _on_move_press(self, obj, _event) -> None:  # noqa: ARG002 (VTK API)
        raw = self._raw_iren
        if raw is None or self._move_callback is None:
            return
        try:
            x, y = (int(v) for v in raw.GetEventPosition())
        except Exception:  # pragma: no cover
            return
        index = self.pick_at(x, y)
        self._drag_press = (x, y)
        self._move_candidate = None if index is None else int(index)
        if index is None:
            return
        if self._movable is not None and int(index) not in self._movable:
            # A fixed atom: the camera keeps rotating, but remember the click.
            return
        self._drag_index = int(index)
        self._drag_origin = np.array(self._molecule.coords[index], dtype=float)
        if self._move_select is not None:
            self._move_select(int(index))
        self._abort_event(raw, self._move_press_observer)

    def _world_per_pixel(self, origin: np.ndarray) -> float:
        """World length (Å) covered by one screen pixel at the atom depth."""
        renderer = self.plotter.renderer
        camera = renderer.GetActiveCamera()
        height = float(renderer.GetSize()[1]) or 1.0
        if camera.GetParallelProjection():
            return 2.0 * float(camera.GetParallelScale()) / height
        direction = np.asarray(camera.GetDirectionOfProjection(), dtype=float)
        offset = np.asarray(origin, dtype=float) - np.asarray(
            camera.GetPosition(), dtype=float
        )
        depth = abs(float(np.dot(offset, direction)))
        return (
            2.0
            * math.tan(math.radians(float(camera.GetViewAngle())) / 2.0)
            * depth
            / height
        )

    def _on_move_motion(self, obj, _event) -> None:  # noqa: ARG002 (VTK API)
        raw = self._raw_iren
        if raw is None or self._drag_index is None or self._drag_origin is None:
            return
        try:
            x, y = (int(v) for v in raw.GetEventPosition())
        except Exception:  # pragma: no cover
            return
        dx = x - self._drag_press[0]
        dy = y - self._drag_press[1]
        camera = self.plotter.renderer.GetActiveCamera()
        world_per_pixel = self._world_per_pixel(self._drag_origin)
        displacement = screen_delta_to_world(
            camera.GetDirectionOfProjection(),
            camera.GetViewUp(),
            world_per_pixel,
            dx,
            dy,
        )
        if self._move_callback is not None:
            self._move_callback(int(self._drag_index), self._drag_origin + displacement)

    def _on_move_release(self, obj, _event) -> None:  # noqa: ARG002 (VTK API)
        raw = self._raw_iren
        if raw is None:
            return
        press = self._drag_press
        was_drag = self._drag_index is not None
        index = self._drag_index if was_drag else self._move_candidate
        self._drag_index = None
        self._drag_origin = None
        self._drag_press = None
        self._move_candidate = None
        if press is None or index is None:
            return
        try:
            x, y = (int(v) for v in raw.GetEventPosition())
        except Exception:  # pragma: no cover
            x, y = press
        if _is_click(press, (x, y)) and self._move_click is not None:
            self._move_click(int(index))
        if was_drag and self._move_finish is not None:
            self._move_finish()

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
        """Record the press position and pick the atom under the pointer.

        Picking on press (instead of on release) applies the selection as
        soon as the button goes down, and leaves the release handler with a
        single job: deciding whether the gesture was a click or a camera
        drag.
        """
        iren = self._iren
        if iren is None:
            return
        pos = self._event_position(iren)
        if pos is None:
            return
        self._press_xy = pos
        index = self.pick_at(*pos)
        self._last_pick_index = index
        if index is not None and self._pick_callback is not None:
            self._pick_callback(int(index))

    def _on_left_button_release(self, *_args) -> None:
        """On release: if it was a click on empty space, clear the selection.

        Whether the press landed on an atom was already decided by
        :meth:`pick_at` on press, so no second pick is made here (a fresh
        pick on release can fail and wrongly clear a just-made selection).
        """
        # While atom-dragging is active it owns the whole interaction.
        if self._move_callback is not None:
            return
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
        # clear the selection. If it was on an atom, the selection was already
        # applied on press; the optional click callback is notified here (once
        # the click is confirmed, so it never fires for a camera drag).
        if self._last_pick_index is not None:
            if self._click_callback is not None:
                self._click_callback(int(self._last_pick_index))
            return
        if self._empty_callback is not None:
            self._empty_callback()

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
        limit = (
            tolerance
            if tolerance is not None
            else max(
                _NEAREST_ATOM_MIN_RADIUS,
                visual_radius * _NEAREST_ATOM_RADIUS_FACTOR,
            )
        )
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
        _make_unpickable(self._selection_actor)
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
            ring_actor = self.plotter.add_mesh(
                ring,
                style="wireframe",
                color=color,
                line_width=opts.selection_line_width,
                name=f"meas-ring-{i}",
                reset_camera=False,
            )
            # The highlight must not intercept clicks aimed at its atom.
            _make_unpickable(ring_actor)
            self._meas_actors.append(ring_actor)
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
                fill_actor = self.plotter.add_mesh(
                    core,
                    color=_MEAS_CENTRAL_COLOR,
                    opacity=_MEAS_CENTRAL_OPACITY,
                    smooth_shading=True,
                    name=f"meas-central-{i}",
                    reset_camera=False,
                )
                _make_unpickable(fill_actor)
                self._meas_actors.append(fill_actor)
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
            _make_unpickable(actor)
            self._meas_actors.append(actor)
