"""Tests for the molecular renderer using an offscreen plotter (no window)."""

import os
from pathlib import Path

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from pychem.io import get_reader  # noqa: E402
from pychem.rendering import MolecularRenderer, RenderOptions  # noqa: E402

DATA = Path(__file__).parent / "data" / "xyz"

requires_display = pytest.mark.skipif(
    not os.environ.get("DISPLAY"), reason="a display is required for VTK"
)


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
