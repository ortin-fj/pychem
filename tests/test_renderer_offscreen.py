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

    # 2 atom actors (O, H) + 1 bond actor
    assert len(renderer._actors) == 3
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

    Reproduces the release-handler sequence with an atom actor in
    ``picked_actor`` (what the press picker leaves behind).
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

    def set_picked(actor):
        # In PyVista 0.48 the picked actor lives in the picking component.
        component = getattr(plotter, "picking", None)
        if component is not None:
            component._picked_actor = actor
        else:
            plotter._picked_actor = actor

    # Press and release WITHOUT moving on an atom -> not cleared
    atom_actor = renderer._atom_actors[0]
    set_picked(atom_actor)
    renderer._iren = FakeIren((120, 120))
    renderer._press_xy = (120, 120)
    renderer._on_left_button_release()
    assert events == []

    # Click (no move) on empty space -> cleared
    set_picked(None)
    renderer._iren = FakeIren((120, 120))
    renderer._press_xy = (120, 120)
    renderer._on_left_button_release()
    assert events == ["empty"]

    # Drag (with movement) over empty space -> NOT cleared
    events.clear()
    set_picked(None)
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
