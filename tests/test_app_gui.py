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

    # The properties section must not exceed half of the panel height
    panel = window._props_panel
    panel.resize(240, 800)
    # Hidden offscreen widgets do not get a resizeEvent; send one manually.
    from PySide6.QtCore import QCoreApplication, QSize
    from PySide6.QtGui import QResizeEvent

    QCoreApplication.sendEvent(panel, QResizeEvent(panel.size(), QSize(0, 0)))
    qapp.processEvents()
    assert panel._scroll.maximumHeight() <= int(panel.height() * 0.5) + 1

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


def test_dynamic_bonds_appear_and_break_with_frame(qapp):
    from pychem.app import MainWindow

    window = MainWindow(filename=DATA / "h2_dissociation.xyz")
    qapp.processEvents()

    # Dynamic bonds are enabled by default.
    assert window._dynamic_chk.isChecked()

    def nbonds_at(t):
        window._playback_t = t
        window._render_pose()
        qapp.processEvents()
        bonds = window.renderer._bonds
        return 0 if bonds is None else len(bonds)

    # frame 0 (0.58 Å, forms) -> 1 bond
    assert nbonds_at(0.0) == 1
    # frame 1 (0.72 Å, kept by hysteresis) -> 1 bond
    assert nbonds_at(1.0) == 1
    # frame 2 (0.82 Å, broken) -> 0 bonds
    assert nbonds_at(2.0) == 0
    # frame 3 (0.62 Å, reformed) -> 1 bond
    assert nbonds_at(3.0) == 1

    # With "Dynamic bonds" off, the frame-0 connectivity is kept fixed.
    window._dynamic_chk.setChecked(False)
    assert nbonds_at(2.0) == 1

    window.close()
    qapp.processEvents()


def test_volumetric_cube_plot_flow(qapp):
    from pychem.app import MainWindow

    cube = Path(__file__).parent / "data" / "volumetric" / "single_point.mo1a.cube"
    window = MainWindow(filename=cube)
    qapp.processEvents()

    # Detected as volumetric; nothing plotted yet; panel shown with the button.
    assert window._volfile is not None
    assert not window._vol_box.isHidden()
    assert window._vol_list.isHidden()
    # Isovalue/Quality rows stay hidden until the first plot.
    assert window._iso_row.isHidden()
    assert window._qual_row.isHidden()

    window._on_plot_isosurface()
    qapp.processEvents()

    assert window._vol_list.count() == 1
    assert not window._vol_list.isHidden()
    # The Isovalue/Quality controls appear only after the first plot.
    assert not window._iso_row.isHidden()
    assert not window._qual_row.isHidden()
    # The isosurface actors were added to the viewer.
    assert len(window.vol_renderer._actors) >= 1

    # A suggested absolute isovalue is proposed automatically.
    assert window._iso_spin.value() > 0.0
    assert window._iso_manual is False
    # Quality selector exists with a Medium default ("Low" was dropped).
    assert [
        window._quality_combo.itemText(i)
        for i in range(window._quality_combo.count())
    ] == ["Medium", "High"]
    assert window._quality_combo.currentText() == "Medium"

    # Changing the isovalue marks it as manual and re-plots.
    window._iso_spin.setValue(0.05)
    qapp.processEvents()
    assert window._iso_manual is True
    assert window._plotted_index == 0

    # The Volumetric section is limited to about half the panel area.
    panel_future = window._props_panel.future
    assert window._vol_box.maximumHeight() <= max(120, panel_future.height() // 2) + 1

    window.close()
    qapp.processEvents()


def test_measurements_flow(qapp):
    from pychem.app import MainWindow

    water = Path(__file__).parent / "data" / "xyz" / "water.xyz"
    window = MainWindow(filename=water)
    qapp.processEvents()

    # The Measurements section is shown for a molecule; no mode active yet.
    assert not window._meas_box.isHidden()
    assert window._meas_mode is None
    assert all(not b.isChecked() for b in window._meas_buttons.values())

    # Activate the Bonds mode through its button.
    window._meas_buttons["bonds"].setChecked(True)
    qapp.processEvents()
    assert window._meas_mode == "bonds"

    # Picked atoms go to the measurement, not to the single-atom selection.
    window._on_atom_picked(0)
    window._on_atom_picked(1)
    qapp.processEvents()
    assert window._meas_atoms == [0, 1]
    assert window.renderer.selected_index is None
    # 2 gold rings + 1 value label.
    assert len(window.renderer._meas_actors) == 3

    # Switching to Angles drops the previous measurement and selection.
    window._meas_buttons["angles"].setChecked(True)
    qapp.processEvents()
    assert window._meas_mode == "angles"
    assert window._meas_atoms == []
    assert len(window.renderer._meas_actors) == 0
    assert not window._meas_buttons["bonds"].isChecked()

    # 3 atoms complete the angle: rings + central sphere + label.
    window._on_atom_picked(1)
    window._on_atom_picked(0)
    window._on_atom_picked(2)
    qapp.processEvents()
    assert window._meas_atoms == [1, 0, 2]
    assert len(window.renderer._meas_actors) == 5

    # Turning the mode off clears the measurement and the buttons.
    window._meas_buttons["angles"].setChecked(False)
    qapp.processEvents()
    assert window._meas_mode is None
    assert len(window.renderer._meas_actors) == 0
    assert all(not b.isChecked() for b in window._meas_buttons.values())

    window.close()
    qapp.processEvents()

