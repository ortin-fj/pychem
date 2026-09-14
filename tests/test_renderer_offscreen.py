"""Tests for the molecular renderer using an offscreen plotter (no window)."""

import os
from pathlib import Path

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from pychem.io import get_reader  # noqa: E402
from pychem.rendering import MolecularRenderer, RenderOptions  # noqa: E402

DATA = Path(__file__).parent / "data" / "xyz"
VOLUME_DATA = Path(__file__).parent / "data" / "volumetric"

requires_display = pytest.mark.skipif(
    not os.environ.get("DISPLAY"), reason="a display is required for VTK"
)


def test_density_envelope_is_centred_on_the_molecule():
    """Regression: the isosurface must wrap the molecule, not sit beside it.

    Only the scalar-field *values* used to be tested, never where they land in
    space; a transposed voxel grid slipped through and the surface came out
    shifted. This checks the geometry: pure VTK filters, no window needed.
    """
    from pychem.io.volumetric import read_cube
    from pychem.rendering.volume_renderer import to_image_data

    molecule, volumes = read_cube(VOLUME_DATA / "opt.eldens.cube")
    density = volumes[0]
    surface = to_image_data(density).contour(
        isosurfaces=[density.suggest_density_isovalue()]
    )
    assert surface.n_points > 0

    points = np.asarray(surface.points)
    # Its centre of mass sits on the molecule ...
    np.testing.assert_allclose(
        points.mean(axis=0), molecule.coords.mean(axis=0), atol=0.3
    )
    # ... with every atom inside it, at a plausible (van der Waals-like)
    # distance; a transposed surface leaves some atom far away.
    diameters = np.array(
        [
            np.sqrt(((points - centre) ** 2).sum(axis=1)).min()
            for centre in molecule.coords
        ]
    )
    assert diameters.min() > 0.8
    assert diameters.max() < 3.0
    assert diameters.max() / diameters.min() < 2.0


@requires_display
def test_renderer_builds_expected_actors():
    water = get_reader(DATA / "water.xyz")
    plotter = pv.Plotter(off_screen=True, window_size=(320, 240))
    renderer = MolecularRenderer(plotter, RenderOptions())
    renderer.show(water)

    # One GPU-instanced actor for ALL atoms (colours are per-point data) + 1
    # actor for the bonds. It used to be one actor per element.
    assert len(renderer._atom_actors) == 1
    assert len(renderer._actors) == 2
    plotter.close()


@requires_display
def test_atom_moving_observes_the_release_on_the_interactor_style():
    """Regression: real windows swallow the interactor's release event.

    The press and move events reach the interactor, but the release does not,
    so a plain click was never confirmed and the umbrella roles were never
    assigned. The release is therefore observed on the interactor style too.
    """
    water = get_reader(DATA / "water.xyz")
    plotter = pv.Plotter(off_screen=True, window_size=(320, 240))
    renderer = MolecularRenderer(plotter, RenderOptions())
    renderer.show(water)
    renderer.enable_atom_moving(lambda index, position: None)

    assert renderer._move_press_observer is not None
    assert renderer._move_release_observer is not None
    # The fallback that makes clicks work in a real window.
    assert renderer._move_release_style_observer is not None
    assert renderer._move_release_style is plotter.iren.style

    renderer.disable_atom_moving()
    assert renderer._move_release_observer is None
    assert renderer._move_release_style_observer is None
    plotter.close()


@requires_display
def test_overlay_actors_do_not_intercept_atom_clicks():
    """Highlights sit on top of the atoms: they must not be pickable."""
    water = get_reader(DATA / "water.xyz")
    plotter = pv.Plotter(off_screen=True, window_size=(320, 240))
    renderer = MolecularRenderer(plotter, RenderOptions())
    renderer.show(water)

    renderer.select_atom(0)
    assert renderer._selection_actor.GetPickable() == 0

    renderer.set_measurement([0, 1, 2], central=(0,))
    assert renderer._meas_actors
    assert all(actor.GetPickable() == 0 for actor in renderer._meas_actors)

    plotter.close()


@requires_display
def test_renderer_screenshot_output(tmp_path):
    benzene = get_reader(DATA / "benzene.xyz")
    plotter = pv.Plotter(off_screen=True, window_size=(400, 300))
    renderer = MolecularRenderer(plotter, RenderOptions())
    renderer.show(benzene)
    plotter.reset_camera()
    out = tmp_path / "benzene.png"
    plotter.screenshot(str(out))
    assert out.exists() and out.stat().st_size > 0
    plotter.close()


@requires_display
def test_atom_selection_halo_and_neighbors():
    water = get_reader(DATA / "water.xyz")
    plotter = pv.Plotter(off_screen=True, window_size=(320, 240))
    renderer = MolecularRenderer(plotter, RenderOptions())
    renderer.show(water)

    # No selection yet
    assert renderer.selected_index is None
    assert renderer._selection_actor is None

    # Select the oxygen (index 0)
    renderer.select_atom(0)
    assert renderer.selected_index == 0
    assert renderer._selection_actor is not None

    # Oxygen neighbors: the two hydrogens
    neigh = set(renderer.neighbor_indices(0).tolist())
    assert neigh == {1, 2}
    assert renderer.neighbor_indices(1).tolist() == [0]

    # Clear the selection
    renderer.select_atom(None)
    assert renderer.selected_index is None
    assert renderer._selection_actor is None
    plotter.close()


@requires_display
def test_nearest_atom_mapping():
    water = get_reader(DATA / "water.xyz")
    plotter = pv.Plotter(off_screen=True, window_size=(320, 240))
    renderer = MolecularRenderer(plotter, RenderOptions())
    renderer.show(water)

    # The exact center of an atom maps to itself
    assert renderer.nearest_atom(water.coords[0]) == 0
    assert renderer.nearest_atom(water.coords[2]) == 2

    # A very distant point maps to no atom
    assert renderer.nearest_atom(np.array([100.0, 100.0, 100.0])) is None
    plotter.close()


@requires_display
def test_click_on_atom_without_move_keeps_selection():
    """Regression: press and release on an atom without moving must not clear.

    Whether the press landed on an atom is decided on press by the analytic
    pick (``_last_pick_index``); the release handler only has to tell a click
    from a camera drag.
    """
    water = get_reader(DATA / "water.xyz")
    plotter = pv.Plotter(off_screen=True, window_size=(320, 240))
    renderer = MolecularRenderer(plotter, RenderOptions())
    renderer.show(water)

    events: list[str] = []
    renderer._empty_callback = lambda: events.append("empty")

    class FakeIren:
        def __init__(self, pos):
            self.pos = pos

        def get_event_position(self):
            return self.pos

    # Press and release WITHOUT moving on an atom -> not cleared
    renderer._last_pick_index = 0
    renderer._iren = FakeIren((120, 120))
    renderer._press_xy = (120, 120)
    renderer._on_left_button_release()
    assert events == []

    # Click (no move) on empty space -> cleared
    renderer._last_pick_index = None
    renderer._iren = FakeIren((120, 120))
    renderer._press_xy = (120, 120)
    renderer._on_left_button_release()
    assert events == ["empty"]

    # Drag (with movement) over empty space -> NOT cleared
    events.clear()
    renderer._last_pick_index = None
    renderer._iren = FakeIren((220, 180))
    renderer._press_xy = (120, 120)
    renderer._on_left_button_release()
    assert events == []
    plotter.close()


@requires_display
def test_electrostatic_renderer_colours_the_density_envelope():
    """ESP map: density envelope coloured by the potential."""
    from pychem.io.volumetric import read_esp_pair
    from pychem.rendering.esp_renderer import (
        SCALAR_BAR_TITLE,
        ElectrostaticRenderer,
    )

    pair = read_esp_pair(
        [VOLUME_DATA / "0.esp.cube", VOLUME_DATA / "opt.eldens.cube"]
    )
    plotter = pv.Plotter(off_screen=True, window_size=(320, 240))
    renderer = ElectrostaticRenderer(plotter)

    assert renderer.show(pair.density, pair.potential) is True
    assert len(renderer._actors) == 1

    surface = renderer.surface
    assert surface is not None and surface.n_points > 0
    values = np.asarray(surface.point_data[SCALAR_BAR_TITLE], dtype=float)
    assert values.shape == (surface.n_points,)
    # A colormap only reads if the values actually vary across the surface.
    assert renderer.clim is not None
    assert renderer.clim[0] < renderer.clim[1]
    assert renderer.clim[0] >= pair.potential.values.min() - 1e-9
    assert renderer.clim[1] <= pair.potential.values.max() + 1e-9

    renderer.clear()
    assert renderer._actors == []
    assert renderer.surface is None
    plotter.close()


@requires_display
def test_electrostatic_renderer_needs_a_surface_to_draw():
    from pychem.io.volumetric import read_esp_pair
    from pychem.rendering.esp_renderer import ElectrostaticRenderer

    pair = read_esp_pair(
        [VOLUME_DATA / "opt.eldens.cube", VOLUME_DATA / "0.esp.cube"]
    )
    plotter = pv.Plotter(off_screen=True, window_size=(320, 240))
    renderer = ElectrostaticRenderer(plotter)

    # A level above the density maximum encloses nothing: no actor at all.
    assert renderer.show(pair.density, pair.potential, level=1.0e6) is False
    assert renderer._actors == []
    plotter.close()


# -- rendering performance regressions --------------------------------------
#
# A 11070-atom PDB used to take ~3.6 s to draw (a merged mesh of ~7.4M
# triangles, ~440 MB) and froze for ~0.57 s on every left click, because
# pyvista's surface picking installs a vtkCellPicker that builds a cell
# locator over that whole mesh. Atoms are now one GPU-instanced actor and the
# pick is analytic.


def test_sphere_resolution_is_lighter_for_large_structures():
    from pychem.rendering.molecular_renderer import _sphere_resolution

    assert _sphere_resolution(100) == (24, 16)
    assert _sphere_resolution(11070) == (12, 8)
    # Fewer triangles per sphere for a big system.
    assert np.prod(_sphere_resolution(11070)) < np.prod(_sphere_resolution(100))


def test_ray_sphere_first_hit_picks_the_nearest_sphere():
    from pychem.rendering.molecular_renderer import ray_sphere_first_hit

    centres = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, 10.0]])
    radii = np.array([1.0, 1.0])
    origin = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])

    # The near sphere wins even though both are hit.
    assert ray_sphere_first_hit(origin, direction, centres, radii) == 0
    # A ray beside the spheres misses both.
    assert ray_sphere_first_hit(
        np.array([5.0, 0.0, 0.0]), direction, centres, radii
    ) is None
    # A ray that starts inside a sphere still reports it.
    assert ray_sphere_first_hit(
        np.array([0.0, 0.0, 9.5]), direction, centres, radii
    ) == 1
    # No spheres at all.
    assert ray_sphere_first_hit(
        origin, direction, np.empty((0, 3)), np.empty(0)
    ) is None


def _screen_radius_px(renderer, camera, height, index):
    """Radius of the drawn sphere ``index`` in pixels."""
    coords = renderer.molecule.coords
    depth = abs(
        float(
            np.dot(
                coords[index] - np.asarray(camera.GetPosition()),
                np.asarray(camera.GetDirectionOfProjection()),
            )
        )
    )
    if camera.GetParallelProjection():
        world_per_pixel = 2.0 * float(camera.GetParallelScale()) / height
    else:
        world_per_pixel = (
            2.0 * np.tan(np.radians(camera.GetViewAngle()) / 2.0) * depth / height
        )
    return renderer._atom_radii[index] / world_per_pixel


def _projector(plotter):
    """World -> display projection using VTK's own transform."""
    render_target = plotter.renderer

    def project(point):
        render_target.SetWorldPoint(
            float(point[0]), float(point[1]), float(point[2]), 1.0
        )
        render_target.WorldToDisplay()
        return np.array(render_target.GetDisplayPoint()[:2], dtype=float)

    return project


@requires_display
def test_atoms_are_drawn_as_one_gpu_instanced_actor():
    """Atoms are instanced on the GPU: the geometry must not grow with n."""
    import vtk

    molecule = get_reader(DATA / "4P00.pdb")
    plotter = pv.Plotter(off_screen=True, window_size=(320, 240))
    renderer = MolecularRenderer(plotter, RenderOptions())
    renderer.show(molecule)

    assert len(renderer._atom_actors) == 1
    mapper = renderer._atom_actors[0].GetMapper()
    assert isinstance(mapper, vtk.vtkGlyph3DMapper)
    # The input is the atom cloud (one point per atom) ...
    assert mapper.GetInput().GetNumberOfPoints() == molecule.n_atoms
    # ... and a single sphere is uploaded, whatever the atom count. The old
    # CPU-expanded glyph would report ~7.4M cells here instead.
    assert mapper.GetSource().GetNumberOfCells() < 1000
    plotter.close()


@requires_display
def test_rebuilding_the_scene_does_not_leak_actors():
    """Trajectory playback rebuilds the scene per frame: no actor may leak.

    ``Plotter.add_actor`` returns ``(actor, property)``; storing the tuple
    instead of the actor made the removal silently fail and pile up actors.
    """
    water = get_reader(DATA / "water.xyz")
    plotter = pv.Plotter(off_screen=True, window_size=(320, 240))
    renderer = MolecularRenderer(plotter, RenderOptions())
    renderer.show(water)
    after_first = len(plotter.renderer.actors)
    renderer.show(water)
    renderer.show(water)
    assert len(plotter.renderer.actors) == after_first
    plotter.close()


@requires_display
def test_pick_at_selects_the_atom_under_its_projected_centre():
    """The analytic ray must agree with VTK's own projection.

    The projected centre comes from ``WorldToDisplay``, so an axis or y-flip
    mistake in the ray would show up as a pick that lands somewhere else.
    """
    from pychem.rendering.molecular_renderer import _PICK_PIXEL_RADIUS

    molecule = get_reader(DATA / "benzene.xyz")
    plotter = pv.Plotter(off_screen=True, window_size=(640, 480))
    renderer = MolecularRenderer(plotter, RenderOptions())
    renderer.show(molecule)
    plotter.reset_camera()
    plotter.render()

    project = _projector(plotter)
    camera = plotter.renderer.GetActiveCamera()
    height = plotter.renderer.GetSize()[1]

    for index in range(molecule.n_atoms):
        px, py = project(molecule.coords[index])
        found = renderer.pick_at(int(round(px)), int(round(py)))
        assert found is not None, f"atom {index} is not pickable at its centre"
        gap = float(np.linalg.norm(project(molecule.coords[found]) - (px, py)))
        allowed = (
            _PICK_PIXEL_RADIUS + _screen_radius_px(renderer, camera, height, found) + 1.0
        )
        assert gap <= allowed, f"picked atom {found} {gap:.1f} px away (max {allowed:.1f})"
    plotter.close()


@requires_display
def test_pick_at_ignores_far_background_and_stays_fast():
    """Background clicks select nothing, and a pick must stay cheap."""
    import time

    from pychem.rendering.molecular_renderer import _PICK_PIXEL_RADIUS

    molecule = get_reader(DATA / "4P00.pdb")
    plotter = pv.Plotter(off_screen=True, window_size=(640, 480))
    renderer = MolecularRenderer(plotter, RenderOptions())
    renderer.show(molecule)
    plotter.reset_camera()
    plotter.render()

    image = plotter.screenshot(return_img=True)
    height, width = image.shape[:2]
    project = _projector(plotter)
    projected = np.array([project(c) for c in molecule.coords])

    # Pick a *pure background* pixel that is far from every atom, so the
    # small screen-space fallback cannot legitimately claim it.
    background = np.argwhere(np.all(image[:, :, :3] > 252, axis=-1))
    assert background.size, "expected background pixels around the molecule"
    distances = []
    for row, col in background[:: max(1, len(background) // 200)]:
        nearest = float(
            np.min(np.hypot(projected[:, 0] - col, projected[:, 1] - (height - 1 - row)))
        )
        distances.append((nearest, int(col), height - 1 - int(row)))
    nearest, x, y = max(distances)
    assert nearest > _PICK_PIXEL_RADIUS + 5.0
    assert renderer.pick_at(x, y) is None

    start = time.perf_counter()
    for _ in range(50):
        renderer.pick_at(width // 2, height // 2)
    per_pick = (time.perf_counter() - start) / 50
    # ~0.2 ms in practice; the old VTK cell picker took ~570 ms.
    assert per_pick < 0.05, f"pick took {per_pick * 1000:.1f} ms"
    plotter.close()
