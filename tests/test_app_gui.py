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
VOLUME_DATA = Path(__file__).parent / "data" / "volumetric"


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


def test_side_sections_collapsible_and_resizable(qapp):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QDockWidget, QSplitter

    from pychem.app import MainWindow

    window = MainWindow(filename=DATA / "water.xyz")
    qapp.processEvents()

    # Fixed dock: no close/float buttons, and it cannot be undocked/moved
    assert (
        window._props_dock.features()
        == QDockWidget.DockWidgetFeature.NoDockWidgetFeatures
    )

    # The sections live in a vertical splitter (draggable separators).
    assert isinstance(window._side_splitter, QSplitter)
    assert window._side_splitter.orientation() == Qt.Orientation.Vertical
    # Atom Properties, Measurements and Volumetric are siblings in it.
    assert window._side_splitter.indexOf(window._props_box) >= 0
    assert window._side_splitter.indexOf(window._meas_box) >= 0
    assert window._side_splitter.indexOf(window._vol_box) >= 0

    # Atom Properties is framed like the other sections (collapsible box).
    props_box = window._props_box
    assert props_box.title() == "Atom Properties"
    # No native title check box anymore: each header exposes a right-aligned
    # window-style button that swaps between minimize (—) and expand (▢).
    assert not props_box.isCheckable()
    assert not props_box.is_collapsed()
    assert window._props_scroll is not None  # its (scrollable) body exists

    from PySide6.QtWidgets import QToolButton

    assert isinstance(props_box.toggle_button, QToolButton)
    assert props_box.toggle_button.text() == props_box._MIN_GLYPH  # expanded

    # Clicking the header button minimizes a section (body hidden) and the
    # glyph swaps to the expand one; clicking again restores it.
    assert not window._meas_box.isHidden()  # molecule loaded -> visible
    window._meas_box.toggle_button.click()
    qapp.processEvents()
    assert window._meas_box.is_collapsed()
    assert window._meas_buttons["bonds"].isHidden()
    assert window._meas_box.toggle_button.text() == window._meas_box._EXPAND_GLYPH
    assert window._meas_box.maximumHeight() <= window._meas_box.header_height()

    window._meas_box.toggle_button.click()
    qapp.processEvents()
    assert not window._meas_box.is_collapsed()
    assert not window._meas_buttons["bonds"].isHidden()
    assert window._meas_box.toggle_button.text() == window._meas_box._MIN_GLYPH
    # Expanding frees the height again so the body is not squeezed.
    assert window._meas_box.maximumHeight() > window._meas_box.header_height()

    window.close()
    qapp.processEvents()


def test_section_header_click_toggles(qapp):
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    from pychem.app import MainWindow

    window = MainWindow(filename=DATA / "water.xyz")
    qapp.processEvents()

    box = window._props_box
    assert not box.is_collapsed()

    def header_click(x=30.0, y=6.0):
        press = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(x, y),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        release = QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            QPointF(x, y),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        box.mousePressEvent(press)
        box.mouseReleaseEvent(release)

    # A click on the title strip minimizes the section...
    header_click()
    qapp.processEvents()
    assert box.is_collapsed()
    assert window._props_scroll.isHidden()

    # ...and a second click on the (now header-only) box expands it again.
    header_click()
    qapp.processEvents()
    assert not box.is_collapsed()
    assert not window._props_scroll.isHidden()

    # Clicks below the header (over the body) must NOT toggle the section.
    press = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(30.0, 300.0),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    release = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        QPointF(30.0, 300.0),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    box.mousePressEvent(press)
    box.mouseReleaseEvent(release)
    qapp.processEvents()
    assert not box.is_collapsed()

    window.close()
    qapp.processEvents()


def test_collapsing_all_keeps_side_panel_width(qapp):
    from PySide6.QtCore import Qt

    from pychem.app import MainWindow

    window = MainWindow(filename=DATA / "water_breathing.xyz")
    window.show()
    window.resize(1200, 800)
    qapp.processEvents()

    # Give the side panel a comfortable, user-like width.
    window.resizeDocks([window._props_dock], [320], Qt.Orientation.Horizontal)
    qapp.processEvents()
    qapp.processEvents()
    width_before = window._props_dock.width()
    assert width_before >= 300

    # Collapsing every visible section must not narrow the side panel...
    for box in (window._props_box, window._meas_box, window._traj_box):
        box.set_collapsed(True)
    qapp.processEvents()
    qapp.processEvents()
    assert window._props_dock.width() == width_before

    # ...but the width must not be stuck either: while collapsed the panel can
    # still be dragged (much) narrower again.
    window.resizeDocks([window._props_dock], [170], Qt.Orientation.Horizontal)
    qapp.processEvents()
    qapp.processEvents()
    assert window._props_dock.width() < width_before

    # Expanding the sections again keeps whatever width was last set.
    for box in (window._props_box, window._meas_box, window._traj_box):
        box.set_collapsed(False)
    qapp.processEvents()
    qapp.processEvents()
    assert window._props_dock.width() < width_before

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

    # The Volumetric section is an expandable sibling inside the side panel.
    assert window._side_splitter.indexOf(window._vol_box) >= 0
    assert not window._vol_box.is_collapsed()

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


# ---------------------------------------------------------------------------
# Tools > Visualize atomic orbitals (Molden only)
# ---------------------------------------------------------------------------


def test_ao_menu_action_enabled_only_for_molden(qapp):
    from pychem.app import MainWindow

    # A plain XYZ molecule: no basis, so the action is disabled.
    window = MainWindow(filename=DATA / "water.xyz")
    qapp.processEvents()
    assert window._act_visualize_ao is not None
    assert not window._act_visualize_ao.isEnabled()
    assert not window._act_visualize_ao.isChecked()
    window.close()
    qapp.processEvents()

    # A Molden file: enabled, but the panel still starts on the MO page.
    molden = Path(__file__).parent / "data" / "volumetric" / "single_point.molden"
    window = MainWindow(filename=molden)
    qapp.processEvents()
    assert window._act_visualize_ao.isEnabled()
    assert not window._act_visualize_ao.isChecked()
    assert window._vol_stack.currentWidget() is window._mo_page
    window.close()
    qapp.processEvents()


def test_atomic_orbitals_panel_flow(qapp):
    from pychem.app import MainWindow

    molden = Path(__file__).parent / "data" / "volumetric" / "single_point.molden"
    window = MainWindow(filename=molden)
    qapp.processEvents()

    # Entering the AO mode through the Tools action swaps the panel page.
    window._act_visualize_ao.trigger()
    qapp.processEvents()
    assert window._act_visualize_ao.isChecked()
    assert window._vol_stack.currentWidget() is window._ao_page
    # The section title announces atomic orbitals and the Plot button is
    # available to start the session.
    assert window._vol_box.title() == "Volumetric: AO"
    assert not window._ao_plot_btn.isHidden()

    # 16 atoms listed; the first one is a carbon with s/p shells selected.
    assert window._ao_atom_combo.count() == 16
    assert window._ao_atom_combo.currentText().startswith("1 · C")
    types = [
        window._ao_type_combo.itemText(i)
        for i in range(window._ao_type_combo.count())
    ]
    assert types == ["s", "p"]
    assert window._ao_orb_combo.count() > 0

    # Isovalue/Quality appear only after the first plot.
    assert window._ao_iso_row.isHidden()
    assert window._ao_qual_row.isHidden()

    window._on_plot_atomic_orbital()
    qapp.processEvents()
    assert len(window.vol_renderer._actors) >= 1
    assert not window._ao_iso_row.isHidden()
    assert not window._ao_qual_row.isHidden()
    assert window._ao_iso_spin.value() > 0.0
    assert window._ao_manual_iso is False
    # After the first plot the button disappears: selections re-plot on their own.
    assert window._ao_plot_btn.isHidden()

    # A hydrogen only exposes s functions (its two contracted s shells); the
    # new selection is plotted automatically.
    window._ao_atom_combo.setCurrentIndex(3)
    qapp.processEvents()
    types = [
        window._ao_type_combo.itemText(i)
        for i in range(window._ao_type_combo.count())
    ]
    assert types == ["s"]
    assert window._ao_orb_combo.count() == 2
    assert len(window.vol_renderer._actors) >= 1

    # A manual isovalue marks the value as user-set.
    window._ao_iso_spin.setValue(0.05)
    qapp.processEvents()
    assert window._ao_manual_iso is True

    # Toggling the action off returns to the molecular-orbitals page.
    window._act_visualize_ao.trigger()
    qapp.processEvents()
    assert not window._act_visualize_ao.isChecked()
    assert window._vol_stack.currentWidget() is window._mo_page
    assert window._vol_box.title() == "Volumetric: MO"

    window.close()
    qapp.processEvents()


def test_second_atomic_orbital_overlay_flow(qapp):
    from pychem.app import MainWindow

    molden = Path(__file__).parent / "data" / "volumetric" / "single_point.molden"
    window = MainWindow(filename=molden)
    qapp.processEvents()
    window._act_visualize_ao.trigger()
    qapp.processEvents()
    assert window._vol_stack.currentWidget() is window._ao_page

    # The first plot button was renamed to "Plot AO".
    assert window._ao_plot_btn.text() == "Plot AO"
    # "Plot another AO" only makes sense once the first AO is on screen.
    assert window._ao_plot_another_btn.isHidden()
    # The button lives in the AO page only (the MO page has no such control)
    # and the "Volumetric: AO" title is not duplicated by the second box.
    assert window._ao_plot_another_btn.parent() is window._ao_page
    assert window._vol_box.title() == "Volumetric: AO"

    window._on_plot_atomic_orbital()
    qapp.processEvents()
    n_first = len(window.vol_renderer._actors)
    assert n_first >= 1
    assert window._ao_plotted
    assert window._ao_plot_btn.isHidden()
    # The offer to plot a second AO appears now...
    assert not window._ao_plot_another_btn.isHidden()
    # ...while the second picker itself is still hidden.
    assert window._ao2_box.isHidden()

    # Clicking "Plot another AO" reveals the second picker box and removes the
    # button, so a third AO can never be requested.
    window._ao_plot_another_btn.click()
    qapp.processEvents()
    assert not window._ao2_box.isHidden()
    assert window._ao_plot_another_btn.isHidden()
    assert window._ao2_atom_combo.count() == window._ao_atom_combo.count()

    # Pick a different atom for the second AO and plot it.
    window._ao2_atom_combo.setCurrentIndex(7)
    qapp.processEvents()
    window._ao2_plot_btn.click()
    qapp.processEvents()
    assert window._ao2_plotted
    assert window._ao2_plot_btn.isHidden()
    # Two AO overlays are now visible (more actors than with a single AO).
    assert len(window.vol_renderer._actors) > n_first

    # Editing either picker re-renders the pair (two overlays stay).
    window._ao2_orb_combo.setCurrentIndex(0)
    qapp.processEvents()
    assert len(window.vol_renderer._actors) > n_first
    window._ao_atom_combo.setCurrentIndex(3)
    qapp.processEvents()
    assert len(window.vol_renderer._actors) > n_first
    assert window._ao2_plotted

    # Leaving and re-entering the AO mode resets the whole session: the second
    # picker is hidden again and a fresh "Plot AO" button is available.
    window._act_visualize_ao.trigger()
    qapp.processEvents()
    window._act_visualize_ao.trigger()
    qapp.processEvents()
    assert window._vol_stack.currentWidget() is window._ao_page
    assert window._ao2_box.isHidden()
    assert not window._ao2_plotted
    assert not window._ao_plotted
    assert not window._ao_plot_btn.isHidden()

    window.close()
    qapp.processEvents()


def test_ao_roundtrip_through_mo_restores_second_offer(qapp):
    from pychem.app import MainWindow

    molden = Path(__file__).parent / "data" / "volumetric" / "single_point.molden"
    window = MainWindow(filename=molden)
    qapp.processEvents()

    # Plot two atomic orbitals.
    window._act_visualize_ao.trigger()
    qapp.processEvents()
    window._on_plot_atomic_orbital()
    qapp.processEvents()
    window._on_plot_another_ao()
    qapp.processEvents()
    window._ao2_atom_combo.setCurrentIndex(5)
    qapp.processEvents()
    window._ao2_plot_btn.click()
    qapp.processEvents()
    assert window._ao2_plotted
    assert not window._ao2_box.isHidden()

    # Leave AO mode (Tools off): the second picker must not stay open.
    window._act_visualize_ao.trigger()
    qapp.processEvents()
    assert window._vol_stack.currentWidget() is window._mo_page
    assert window._ao2_box.isHidden()
    assert not window._ao2_plotted

    # Plot a molecular orbital while on the MO page.
    window._on_plot_isosurface()
    qapp.processEvents()
    assert window._plotted_index == 0

    # Come back to AO mode and plot the first AO again: the offer to add a
    # second atomic orbital must be back.
    window._act_visualize_ao.trigger()
    qapp.processEvents()
    assert window._vol_stack.currentWidget() is window._ao_page
    assert window._ao2_box.isHidden()
    window._on_plot_atomic_orbital()
    qapp.processEvents()
    assert window._ao_plotted
    assert not window._ao_plot_another_btn.isHidden()

    window.close()
    qapp.processEvents()


def test_show_id_label_toggle(qapp):
    from pychem.app import MainWindow

    # Without a molecule there is nothing to label.
    window = MainWindow()
    qapp.processEvents()
    assert window._act_show_ids is not None
    assert not window._act_show_ids.isEnabled()
    assert not window._act_show_ids.isChecked()
    window.close()
    qapp.processEvents()

    # A loaded molecule enables the action; toggling it draws the numbers.
    window = MainWindow(filename=DATA / "water.xyz")
    qapp.processEvents()
    assert window._act_show_ids.isEnabled()
    assert len(window.renderer._id_label_actors) == 0

    window._act_show_ids.trigger()
    qapp.processEvents()
    assert window._act_show_ids.isChecked()
    assert len(window.renderer._id_label_actors) > 0

    # It coexists with an atom selection.
    window._on_atom_picked(0)
    qapp.processEvents()
    assert window.renderer.selected_index == 0
    assert len(window.renderer._id_label_actors) > 0

    # Untoggling removes the numbers again.
    window._act_show_ids.trigger()
    qapp.processEvents()
    assert not window._act_show_ids.isChecked()
    assert len(window.renderer._id_label_actors) == 0
    window.close()
    qapp.processEvents()


def test_show_id_labels_follow_trajectory_and_measurement(qapp):
    from pychem.app import MainWindow

    window = MainWindow(filename=DATA / "water_breathing.xyz")
    qapp.processEvents()
    window._act_show_ids.trigger()
    qapp.processEvents()
    assert len(window.renderer._id_label_actors) > 0

    # A new pose redraws the molecule and keeps the labels on the atoms.
    window._playback_t = 4.5
    window._render_pose()
    qapp.processEvents()
    assert len(window.renderer._id_label_actors) > 0

    # A geometric measurement does not remove them either.
    window._meas_buttons["bonds"].setChecked(True)
    qapp.processEvents()
    window._on_atom_picked(0)
    window._on_atom_picked(1)
    qapp.processEvents()
    assert len(window.renderer._meas_actors) > 0
    assert len(window.renderer._id_label_actors) > 0

    window.close()
    qapp.processEvents()


# ---------------------------------------------------------------------------
# Tools > Visualize Electrostatic Potential
# ---------------------------------------------------------------------------


def test_esp_action_needs_a_loaded_structure(qapp):
    from pychem.app import MainWindow

    # Empty GUI: there is no structure to paint a potential around.
    window = MainWindow()
    qapp.processEvents()
    assert window._act_visualize_esp is not None
    assert not window._act_visualize_esp.isEnabled()
    assert not window._act_visualize_esp.isChecked()
    window.close()
    qapp.processEvents()

    window = MainWindow(filename=DATA / "sn2_opt.xyz")
    qapp.processEvents()
    assert window._act_visualize_esp.isEnabled()
    assert not window._act_visualize_esp.isChecked()
    window.close()
    qapp.processEvents()


def test_electrostatic_potential_flow(qapp):
    from pychem.app import MainWindow

    density = VOLUME_DATA / "opt.eldens.cube"
    potential = VOLUME_DATA / "0.esp.cube"
    window = MainWindow(filename=DATA / "sn2_opt.xyz")
    qapp.processEvents()

    # Loading the pair (what the file dialog ends up calling) paints the
    # potential on the density envelope and checks the Tools action.
    window._load_esp_cubes([potential, density])
    qapp.processEvents()
    assert window._act_visualize_esp.isChecked()
    assert window._esp_pair is not None
    assert window._esp_pair.density_path == density
    assert window._esp_pair.potential_path == potential

    # One semi-transparent actor, coloured by the potential (a real range).
    assert len(window.esp_renderer._actors) == 1
    surface = window.esp_renderer.surface
    assert surface is not None and surface.n_points > 0
    values = surface.point_data["ESP (a.u.)"]
    assert len(values) == surface.n_points
    assert values.min() < values.max()
    assert "Electrostatic potential" in window.statusBar().currentMessage()

    # Unchecking the entry drops the overlay and forgets the pair, so the
    # cubes are asked for again the next time.
    window._act_visualize_esp.trigger()
    qapp.processEvents()
    assert not window._act_visualize_esp.isChecked()
    assert window.esp_renderer._actors == []
    assert window._esp_pair is None

    window.close()
    qapp.processEvents()


def test_esp_requires_both_cubes(qapp):
    from pychem.app import MainWindow

    window = MainWindow(filename=DATA / "sn2_opt.xyz")
    qapp.processEvents()

    # Two densities: the potential is missing, so nothing is drawn.
    with pytest.raises(ValueError, match="Electrostatic Potential"):
        window._load_esp_cubes([VOLUME_DATA / "opt.eldens.cube"] * 2)
    assert window._esp_pair is None
    assert window.esp_renderer._actors == []
    assert not window._act_visualize_esp.isChecked()

    window.close()
    qapp.processEvents()


def test_esp_and_orbital_isosurfaces_are_alternatives(qapp):
    """The ESP map and the MO/AO isosurfaces never show at the same time."""
    from pychem.app import MainWindow

    window = MainWindow(filename=VOLUME_DATA / "single_point.molden")
    qapp.processEvents()

    # Plot an atomic orbital: the Volumetric panel moves to the AO page.
    window._act_visualize_ao.trigger()
    window._on_plot_atomic_orbital()
    qapp.processEvents()
    assert window._vol_stack.currentWidget() is window._ao_page
    assert len(window.vol_renderer._actors) >= 1

    # Showing the ESP map leaves AO mode and clears the orbital isosurface.
    window._load_esp_cubes(
        [VOLUME_DATA / "0.esp.cube", VOLUME_DATA / "opt.eldens.cube"]
    )
    qapp.processEvents()
    assert not window._act_visualize_ao.isChecked()
    assert window._vol_stack.currentWidget() is window._mo_page
    assert window._vol_box.title() == "Volumetric: MO"
    assert window.vol_renderer._actors == []
    assert len(window.esp_renderer._actors) == 1

    # Plotting an orbital (a molecular one here) drops the ESP map again.
    window._on_plot_isosurface()
    qapp.processEvents()
    assert window.esp_renderer._actors == []
    assert window._esp_pair is None
    assert not window._act_visualize_esp.isChecked()
    assert len(window.vol_renderer._actors) >= 1

    window.close()
    qapp.processEvents()

