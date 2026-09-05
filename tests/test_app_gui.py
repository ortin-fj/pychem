"""Smoke tests for the main window (construction and file loading).

The GUI runs on the Qt "offscreen" platform, so the tests never open a real
window.
"""

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pyside6 = pytest.importorskip("PySide6")

DATA = Path(__file__).parent / "data" / "xyz"


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def test_main_window_loads_molecule(qapp):
    from pychem.app import MainWindow

    window = MainWindow(filename=DATA / "water.xyz")
    qapp.processEvents()

    assert window._molecule is not None
    assert window._molecule.n_atoms == 3
    assert window.windowTitle().startswith("PyChem")

    window.close()
    qapp.processEvents()


def test_main_window_without_file_shows_hint(qapp):
    from pychem.app import MainWindow

    window = MainWindow()
    qapp.processEvents()
    assert window._molecule is None
    window.close()
    qapp.processEvents()


def test_about_dialog_ok_button_closes(qapp):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from pychem.app import MainWindow

    window = MainWindow()
    qapp.processEvents()

    window.show_about()
    qapp.processEvents()

    dialog = window._about_dialog
    assert dialog is not None and dialog.isVisible()

    ok_button = next(b for b in dialog.buttons() if b.text() == "OK")
    QTest.mouseClick(ok_button, Qt.MouseButton.LeftButton)
    qapp.processEvents()
    # With WA_DeleteOnClose, finished() resets the reference
    assert window._about_dialog is None

    window.close()
    qapp.processEvents()


def test_atom_selection_updates_panel(qapp):
    from pychem.app import MainWindow

    window = MainWindow(filename=DATA / "water.xyz")
    qapp.processEvents()

    # Simulate a click on the oxygen atom (index 0)
    window._on_atom_picked(0)
    qapp.processEvents()
    assert window.renderer.selected_index == 0
    assert "ID 1 · O" in window._props_label.text()

    # Second click on the same atom -> deselect
    window._on_atom_picked(0)
    qapp.processEvents()
    assert window.renderer.selected_index is None

    window.close()
    qapp.processEvents()


def test_properties_dock_is_fixed_and_limited(qapp):
    from PySide6.QtWidgets import QDockWidget

    from pychem.app import MainWindow

    window = MainWindow(filename=DATA / "water.xyz")
    qapp.processEvents()

    # Fixed dock: no close/float buttons, and it cannot be undocked/moved
    assert (
        window._props_dock.features()
        == QDockWidget.DockWidgetFeature.NoDockWidgetFeatures
    )

    # The properties section must not exceed the top quarter of the panel
    panel = window._props_panel
    panel.resize(240, 800)
    # Hidden offscreen widgets do not get a resizeEvent; send one manually.
    from PySide6.QtCore import QCoreApplication, QSize
    from PySide6.QtGui import QResizeEvent

    QCoreApplication.sendEvent(panel, QResizeEvent(panel.size(), QSize(0, 0)))
    qapp.processEvents()
    assert panel._scroll.maximumHeight() <= int(panel.height() * 0.25) + 1

    window.close()
    qapp.processEvents()


def test_trajectory_opens_with_playback_controls(qapp):
    import numpy as np

    from pychem.app import MainWindow

    window = MainWindow(filename=DATA / "water_breathing.xyz")
    qapp.processEvents()

    # Detected as a trajectory and the playback controls are shown
    assert window._trajectory is not None
    assert window._trajectory.n_frames == 8
    assert window._slider.maximum() == 7
    # isVisible() is False in offscreen (parent not shown); check explicit hide
    assert not window._traj_box.isHidden()

    # Linear interpolation halfway between frames 0 and 1
    window._playback_t = 0.5
    window._render_pose()
    qapp.processEvents()
    expected = (window._trajectory.frames[0] + window._trajectory.frames[1]) / 2.0
    assert np.allclose(window.renderer.molecule.coords, expected)

    window.close()
    qapp.processEvents()


def test_selected_atom_bonds_update_with_frame(qapp):
    import numpy as np

    from pychem.app import MainWindow

    window = MainWindow(filename=DATA / "water_breathing.xyz")
    qapp.processEvents()

    # Select an H (index 1), which is bonded to the O (index 0).
    window._on_atom_picked(1)
    qapp.processEvents()
    assert window.renderer.selected_index == 1
    assert "Bonds (1)" in window._props_label.text()

    def oh_distance():
        mol = window.renderer.molecule
        neighbors = window.renderer.neighbor_indices(1)
        return float(np.linalg.norm(mol.coords[1] - mol.coords[neighbors[0]]))

    d_short = oh_distance()  # frame 0: O-H = 0.70 Å

    # Move to a stretched pose (around frame 4.5, ~0.925 Å).
    window._playback_t = 4.5
    window._render_pose()
    qapp.processEvents()

    d_long = oh_distance()
    assert d_long > d_short
    # Selection persists across poses and the panel shows the new distance.
    assert window.renderer.selected_index == 1
    assert f"{d_long:.3f}" in window._props_label.text()

    window.close()
    qapp.processEvents()
