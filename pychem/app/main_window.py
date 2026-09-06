"""Main window of PyChem (PySide6 + pyvistaqt).

Layout:

- Central 3D viewer (QtInteractor).
- "File" menu: open a structure and export an image.
- "View" menu: reset the camera.
- Status bar with a summary of the loaded molecule.

Future milestones will add here: a scene-object tree, a properties panel for
the selected atom, a trajectory playback bar and isosurfaces.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from PySide6.QtCore import QLocale, QTimer, Qt
from PySide6.QtGui import QTextDocument
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from pychem.core import geometry
from pychem.core.bonds import guess_bonds, update_dynamic_bonds
from pychem.core.molecule import Molecule
from pychem.core.trajectory import Trajectory
from pychem.core.volume import VolumeData
from pychem.io import get_reader
from pychem.io.volumetric import VolumetricFile, load_volume_file
from pychem.rendering import MolecularRenderer, RenderOptions
from pychem.rendering.volume_renderer import VolumeRenderer

# Extensions accepted by the "Open" dialog. Extended as formats are added.
_OPEN_FILTER = "Structures (*.xyz);;All files (*)"

# Trajectory playback tuning.
_PLAYBACK_UPDATE_MS = 16  # refresh of the playback timer (~60 Hz)
_PLAYBACK_BASE_FPS = 30  # stored frames per second at speed 1
_FPS_LOW = 15  # below this, auto interpolation switches off
_FPS_HIGH = 40  # above this, auto interpolation switches on

# Vertical separation (px) between the rows of the Trajectory group.
_TRAJ_LAYOUT_SPACING = 6  # half of the previous 12 px
_TRAJ_LAYOUT_MARGIN = 12

# Interpolation modes of the trajectory combo box.
_INTERP_AUTO = "Auto"
_INTERP_LINEAR = "Linear"
_INTERP_OFF = "Off"

# Measurement modes: (internal key, button label, required atoms).
_MEAS_MODES = (
    ("bonds", "Bonds"),
    ("angles", "Angles"),
    ("dihedrals", "Dihedrals"),
)
_MEAS_NEEDED = {"bonds": 2, "angles": 3, "dihedrals": 4}


class _IsoSpinBox(QDoubleSpinBox):
    """Isovalue spin box with a few ergonomic tweaks.

    - It shows the decimal point ``'.'`` regardless of the system locale.
    - It does not re-pad the value to the configured number of decimals: a
      typed ``0.004`` stays ``0.004`` instead of becoming ``0.004000``.
    - The up/down arrows move by one unit of the *least significant digit
      that is currently shown*: from ``0.003`` they step by ``0.001`` (to
      ``0.004``), from ``0.1`` by ``0.1`` (to ``0.2``) and from ``0.000345``
      by ``0.000001``.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setDecimals(6)
        self.setLocale(QLocale.c())  # '.' as the decimal separator

    def textFromValue(self, value: float) -> str:  # noqa: N802 (Qt API)
        text = super().textFromValue(value)
        # Drop the trailing zeros of the fractional part (and the separator
        # when the fraction becomes empty), e.g. "0.005000" -> "0.005".
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text

    def stepBy(self, steps: int) -> None:  # noqa: N802 (Qt API)
        """Step by one unit of the least significant digit on screen."""
        if steps == 0:
            return
        # Read the visible text so a value that was just typed is honoured
        # even before it has been committed by pressing Enter.
        text = self.lineEdit().text().strip()
        try:
            current = float(text.replace(",", "."))
        except ValueError:
            current = self.value()
        # Count the fractional digits actually shown (e.g. "0.003" -> 3).
        frac = ""
        for sep in (".", ","):
            if sep in text:
                frac = text.rsplit(sep, 1)[1]
                break
        unit = 10.0 ** (-len(frac)) if frac else 1.0
        if frac:
            value = round(current + steps * unit, len(frac))
        else:
            value = current + steps * unit
        value = min(max(value, self.minimum()), self.maximum())
        self.setValue(value)


class _PropertiesPanel(QWidget):
    """Side-panel contents.

    The selected-atom properties section occupies at most half of the panel;
    the remaining space is reserved for future controls (actions on the
    molecule).
    """

    # Maximum height of the properties area, as a fraction of the panel
    # height. Set to half on purpose so the selected-atom information has
    # room to breathe.
    _MAX_HEIGHT_FRACTION = 0.5

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.on_panel_resized = None  # optional callback (set by the owner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Properties section (scrollable, capped at half the panel height).
        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        label = QLabel()
        label.setWordWrap(True)
        label.setMargin(8)
        label.setTextFormat(Qt.TextFormat.RichText)
        self._scroll.setWidget(label)
        self.label = label
        # The properties area only takes the room its content needs (the cap is
        # applied in resizeEvent), so the sections below sit right underneath
        # it instead of leaving a large blank band between them.
        outer.addWidget(self._scroll, stretch=0)

        # Bottom area reserved for future actions (Trajectory / Volumetric).
        self.future = QWidget(self)
        self.future_layout = QVBoxLayout(self.future)
        self.future_layout.setContentsMargins(6, 6, 6, 6)
        outer.addWidget(self.future, stretch=3)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        self._scroll.setMaximumHeight(
            max(60, int(self.height() * self._MAX_HEIGHT_FRACTION))
        )
        # Let the owner re-fit the lower sections (e.g. the Volumetric box)
        # once the panel has reached its final size.
        if self.on_panel_resized is not None:
            self.on_panel_resized()
        super().resizeEvent(event)


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self, filename: str | Path | None = None) -> None:
        super().__init__()
        self._molecule: Molecule | None = None
        self._about_dialog: QMessageBox | None = None
        self._file_status: str = ""
        self._trajectory: Trajectory | None = None
        self._playing = False
        self._loop = True
        self._speed = 1.0
        self._playback_t = 0.0
        self._auto_linear = True
        self._fps_ticks = 0
        self._fps_start = 0.0
        self._refresh_ms = _PLAYBACK_UPDATE_MS
        self._prev_dynamic_bonds: np.ndarray | None = None
        self._fixed_bonds: np.ndarray | None = None
        self._volfile: VolumetricFile | None = None
        self._volume_cache: dict[int, VolumeData] = {}
        self._plotted_index: int | None = None
        self._iso_manual = False
        self._props_fit_scheduled = False
        self._meas_mode: str | None = None
        self._meas_atoms: list[int] = []
        self._meas_buttons: dict[str, QPushButton] = {}

        self.setWindowTitle("PyChem")
        self.resize(1100, 800)

        self._build_central_widget()
        self._build_menu()
        self._build_properties_dock()
        self._build_measurement_controls()
        self._setup_picking()
        self._build_trajectory_controls()
        self._build_volumetric_controls()
        # Bottom stretch: whichever section is visible (Trajectory or
        # Volumetric) stays pinned to the top, right below Atom Properties,
        # and any leftover space collects at the very bottom of the panel.
        # Future controls must be inserted before this stretch.
        self._future_layout.addStretch(1)
        self.statusBar().showMessage("Ready. Use File > Open to load a structure.")

        if filename is not None:
            self.open_path(filename)

    # -- UI construction -----------------------------------------------------

    def _build_central_widget(self) -> None:
        frame = QFrame()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(frame)
        self.renderer = MolecularRenderer(self.plotter, RenderOptions())
        self.vol_renderer = VolumeRenderer(self.plotter)
        layout.addWidget(self.plotter.interactor)
        self.setCentralWidget(frame)

    def _build_properties_dock(self) -> None:
        """Fixed side panel (no close/float buttons).

        The properties section is limited to half the panel height; the rest
        is reserved for future controls.
        """
        dock = QDockWidget("Atom Properties", self)
        # Permanent panel: no close/float button, and it cannot be undocked
        # or moved.
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        panel = _PropertiesPanel(self)
        panel.label.setText(self._properties_html(None))
        dock.setWidget(panel)
        self._props_panel = panel
        self._props_label = panel.label
        self._future_layout = panel.future_layout
        self._props_dock = dock
        # Re-fit the side-panel sections after a resize (their heights depend
        # on the final panel size, only known once it is laid out).
        panel.on_panel_resized = self._schedule_sections_fit
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _setup_picking(self) -> None:
        """Enable atom selection with the left mouse button.

        Clicking an atom selects it; clicking empty space deselects it; a
        drag (camera rotation) does not change the selection.
        """
        self.renderer.enable_atom_picking(
            self._on_atom_picked, empty_callback=self._on_empty_click
        )

    def _on_empty_click(self) -> None:
        """Click on empty space/bonds: reset the current interaction."""
        if self._meas_mode is not None:
            # In a measurement mode an empty click restarts the selection.
            if self._meas_atoms:
                self._meas_atoms = []
                self._draw_measurement()
            return
        if self.renderer.selected_index is None:
            return
        self.renderer.select_atom(None)
        self._update_properties(None)
        self.statusBar().showMessage(self._file_status or "Ready.")

    def _build_menu(self) -> None:
        menu = self.menuBar()

        file_menu = menu.addMenu("&File")
        act_open = file_menu.addAction("&Open structure...")
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self.open_dialog)

        act_export = file_menu.addAction("&Export image...")
        act_export.setShortcut("Ctrl+E")
        act_export.triggered.connect(self.export_image)

        file_menu.addSeparator()
        act_exit = file_menu.addAction("E&xit")
        act_exit.setShortcut("Ctrl+Q")
        act_exit.triggered.connect(self.close)

        view_menu = menu.addMenu("&View")
        act_reset = view_menu.addAction("Reset &camera")
        act_reset.setShortcut("Ctrl+R")
        act_reset.triggered.connect(self.reset_camera)

        help_menu = menu.addMenu("&Help")
        act_about = help_menu.addAction("&About PyChem")
        act_about.triggered.connect(self.show_about)

    # -- actions -------------------------------------------------------------

    def open_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open structure", "", _OPEN_FILTER)
        if path:
            self.open_path(path)

    def _load(self, path: str | Path) -> Molecule | Trajectory | VolumetricFile:
        """Load a file as Molecule, Trajectory or VolumetricFile."""
        p = Path(path)
        suffix = p.suffix.lower()
        if suffix in (".cube", ".molden"):
            return load_volume_file(p)
        if suffix == ".xyz":
            from pychem.io.xyz import read_xyz_frames

            frames = read_xyz_frames(p)
            if len(frames) > 1:
                return Trajectory.from_molecules(frames, name=p.stem)
            molecule = frames[0]
            molecule.name = p.stem
            return molecule
        return get_reader(p)

    def open_path(self, path: str | Path) -> None:
        """Load a structure/trajectory/volume and show it in the viewer."""
        path = Path(path)
        try:
            data = self._load(path)
        except Exception as exc:  # noqa: BLE001 - reader errors to the UI
            QMessageBox.critical(self, "Error opening file", str(exc))
            return

        self._set_playing(False)
        self.vol_renderer.clear()
        self._volume_cache = {}
        self._plotted_index = None
        self._iso_manual = False
        self._reset_measurement_mode()

        trajectory = data if isinstance(data, Trajectory) else None
        volfile = data if isinstance(data, VolumetricFile) else None
        molecule = (
            volfile.molecule
            if volfile is not None
            else (data.get_frame(0) if trajectory is not None else data)
        )

        self._trajectory = trajectory
        self._volfile = volfile
        self._molecule = molecule
        self.renderer.show(molecule)
        self.plotter.reset_camera()
        self.renderer.select_atom(None)
        self._update_properties(None)

        if trajectory is not None:
            self._activate_trajectory(trajectory, path)
            self._set_volumetric_controls_visible(False)
        elif volfile is not None:
            self._set_trajectory_controls_visible(False)
            self._activate_volumetric(volfile, path)
        else:
            self._set_trajectory_controls_visible(False)
            self._set_volumetric_controls_visible(False)

        self._set_measurement_controls_visible(molecule is not None)

        if trajectory is not None:
            self.setWindowTitle(
                f"PyChem — {path.name}  ({trajectory.n_frames} frames)"
            )
            self._file_status = (
                f"{trajectory.n_frames} frames · {trajectory.n_atoms} atoms · "
                f"{path.name}"
            )
        else:
            self.setWindowTitle(f"PyChem — {path.name}  ({molecule.formula})")
            self._file_status = (
                f"{molecule.n_atoms} atoms · {molecule.formula} · {path.name}"
            )
        self.statusBar().showMessage(self._file_status)

    def _activate_volumetric(self, volfile: VolumetricFile, path: Path) -> None:
        """Show the volumetric panel (nothing plotted yet)."""
        self._iso_manual = False
        self._vol_info.setText(
            f"{path.name}\n{len(volfile.labels)} orbital/field(s) available"
        )
        self._vol_list.clear()
        self._vol_list.setVisible(False)
        # The Isovalue/Quality rows appear only after the first plot.
        self._iso_row.setVisible(False)
        self._qual_row.setVisible(False)
        self._set_volumetric_controls_visible(True)
        self._limit_volumetric_height()

    def _activate_trajectory(self, trajectory: Trajectory, path: Path) -> None:
        """Set up the playback state and controls for ``trajectory``."""
        self._playback_t = 0.0
        self._speed = 1.0
        self._loop = True
        self._auto_linear = True
        self._prev_dynamic_bonds = None
        self._fixed_bonds = guess_bonds(trajectory.get_frame(0))

        self._dynamic_chk.setChecked(True)
        self._speed_slider.blockSignals(True)
        self._speed_slider.setValue(self._slider_from_speed(self._speed))
        self._speed_slider.blockSignals(False)
        self._speed_label.setText(self._format_speed(self._speed))
        self._loop_chk.setChecked(True)
        self._interp_combo.setCurrentText(_INTERP_AUTO)
        self._slider.setRange(0, trajectory.n_frames - 1)
        self._slider.setValue(0)
        self._set_trajectory_controls_visible(True)
        self._render_pose()

    # -- atom selection ------------------------------------------------------

    def _on_atom_picked(self, index: int) -> None:
        """Receive a picked atom index and update the selection/measurement."""
        if self.renderer.molecule is None:
            return
        if self._meas_mode is not None:
            self._on_measurement_atom(index)
            return
        # Click on the already-selected atom -> deselect
        target = None if self.renderer.selected_index == index else index
        self.renderer.select_atom(target)
        self._update_properties(target)

    def _properties_html(self, index: int | None) -> str:
        """(HTML) text of the properties panel (reduced information)."""
        mol = self.renderer.molecule
        if mol is None:
            return "<i>No structure loaded.</i>"
        if index is None:
            return "<i>Click an atom to see its properties.</i>"

        atom = mol.get_atom(index)
        lines = [
            f"<b>ID {index + 1} · {atom.symbol}</b>",
            "<b>Coordinates (Å):</b>",
            "<table cellspacing='0' cellpadding='1'>"
            f"<tr><td><b>x</b></td><td>{atom.x:.4f}</td></tr>"
            f"<tr><td><b>y</b></td><td>{atom.y:.4f}</td></tr>"
            f"<tr><td><b>z</b></td><td>{atom.z:.4f}</td></tr>"
            "</table>",
        ]

        neighbors = self.renderer.neighbor_indices(index)
        if neighbors.size:
            lines.append(f"<b>Bonds ({neighbors.size}):</b>")
            for j in neighbors:
                d = float(np.linalg.norm(mol.coords[index] - mol.coords[j]))
                lines.append(
                    f"&nbsp;&nbsp;{j + 1:>3d} {mol.symbols[j]} — {d:.3f} Å"
                )
        else:
            lines.append("<i>No bonds.</i>")
        return "<br>".join(lines)

    def _update_properties(self, index: int | None, update_status: bool = True) -> None:
        self._props_label.setText(self._properties_html(index))
        # Re-fit the Atom Properties area to its new content on the next tick.
        self._schedule_properties_fit()
        if update_status and self.renderer.molecule is not None and index is not None:
            atom = self.renderer.molecule.get_atom(index)
            self.statusBar().showMessage(f"Selected atom: {index + 1} ({atom.symbol})")

    # -- measurements --------------------------------------------------------

    def _build_measurement_controls(self) -> None:
        """Build the Measurements section: three exclusive mode buttons."""
        box = QGroupBox("Measurements")
        layout = QVBoxLayout(box)
        layout.setSpacing(4)
        layout.setContentsMargins(10, 8, 10, 8)
        for key, label in _MEAS_MODES:
            btn = QPushButton(label)
            btn.setCheckable(True)
            tip = (
                "Measure the interatomic distance between 2 atoms"
                if key == "bonds"
                else (
                    "Measure the angle formed by 3 atoms (central highlighted)"
                    if key == "angles"
                    else "Measure the dihedral angle formed by 4 atoms"
                )
            )
            btn.setToolTip(tip)
            btn.toggled.connect(
                lambda checked, k=key: self._on_measurement_toggled(k, checked)
            )
            layout.addWidget(btn)
            self._meas_buttons[key] = btn
        self._future_layout.addWidget(box)
        self._meas_box = box
        box.setVisible(False)

    def _set_measurement_controls_visible(self, visible: bool) -> None:
        """Show/hide the Measurements section (dropping any active mode)."""
        if not hasattr(self, "_meas_box"):
            return
        if not visible:
            self._exit_measurement_mode()
        self._meas_box.setVisible(visible)

    def _measurement_name(self, key: str) -> str:
        return dict(_MEAS_MODES)[key]

    def _measurement_prompt(self) -> str:
        key = self._meas_mode
        needed = _MEAS_NEEDED[key]
        return (
            f"{self._measurement_name(key)}: select {needed} atoms "
            f"({len(self._meas_atoms)}/{needed})"
        )

    def _on_measurement_toggled(self, key: str, checked: bool) -> None:
        """Handle a measurement mode button being toggled."""
        if checked:
            # The mode buttons are exclusive: silently uncheck the others.
            for other, btn in self._meas_buttons.items():
                if other != key and btn.isChecked():
                    btn.blockSignals(True)
                    btn.setChecked(False)
                    btn.blockSignals(False)
            # Entering a mode drops any previous measurement and selection.
            self._meas_mode = None
            self._meas_atoms = []
            self.renderer.clear_measurement()
            self._clear_atom_selection_ui()
            self._meas_mode = key
            self.statusBar().showMessage(self._measurement_prompt())
        elif self._meas_mode == key:
            self._exit_measurement_mode()

    def _reset_measurement_mode(self) -> None:
        """Drop any active measurement (mode, atoms, overlay and buttons)."""
        self._meas_mode = None
        self._meas_atoms = []
        if hasattr(self, "renderer"):
            self.renderer.clear_measurement()
        for btn in self._meas_buttons.values():
            if btn.isChecked():
                btn.blockSignals(True)
                btn.setChecked(False)
                btn.blockSignals(False)

    def _exit_measurement_mode(self) -> None:
        """Deactivate the active measurement mode and clear its visuals."""
        self._reset_measurement_mode()
        self._clear_atom_selection_ui()
        self.statusBar().showMessage(self._file_status or "Ready.")

    def _clear_atom_selection_ui(self) -> None:
        """Deselect the current atom and reset the properties panel."""
        if self.renderer.selected_index is not None:
            self.renderer.select_atom(None)
        self._update_properties(None)

    def _on_measurement_atom(self, index: int) -> None:
        """An atom was clicked while a measurement mode is active."""
        molecule = self.renderer.molecule
        key = self._meas_mode
        if molecule is None or key is None:
            return
        needed = _MEAS_NEEDED[key]
        if index in self._meas_atoms:
            # Clicking a picked atom removes it again.
            self._meas_atoms.remove(index)
        elif len(self._meas_atoms) >= needed:
            # A completed measurement: starting a new one with this atom.
            self._meas_atoms = [index]
        else:
            self._meas_atoms.append(index)
        self._draw_measurement()

    def _draw_measurement(self) -> None:
        """Refresh the measurement overlay for the current selection/mode."""
        molecule = self.renderer.molecule
        key = self._meas_mode
        if molecule is None or key is None:
            self.renderer.clear_measurement()
            return
        if not self._meas_atoms:
            self.renderer.clear_measurement()
            self.statusBar().showMessage(self._measurement_prompt())
            return
        indices = list(self._meas_atoms)
        if len(indices) < _MEAS_NEEDED[key]:
            # Partial selection: mark the chosen atoms, no label yet.
            self.renderer.set_measurement(indices)
            self.statusBar().showMessage(self._measurement_prompt())
            return
        coords = molecule.coords
        pts = coords[indices]
        if key == "bonds":
            label = f"{geometry.bond_length(pts[0], pts[1]):.3f} Å"
            anchor = geometry.bond_anchor(pts[0], pts[1])
            central: tuple[int, ...] = ()
        elif key == "angles":
            label = f"{geometry.angle_degrees(pts[0], pts[1], pts[2]):.2f}°"
            anchor = geometry.angle_anchor(pts[0], pts[1], pts[2])
            central = (indices[1],)
        else:  # dihedrals
            label = f"{geometry.dihedral_degrees(*pts):.1f}°"
            anchor = geometry.dihedral_anchor(*pts)
            central = (indices[1], indices[2])
        self.renderer.set_measurement(
            indices, central=central, label=label, anchor=anchor
        )
        self.statusBar().showMessage(
            f"{self._measurement_name(key)}: {label}"
        )

    # -- trajectory playback -------------------------------------------------

    def _build_trajectory_controls(self) -> None:
        """Build the playback controls in the lower part of the side panel."""
        self._playback_timer = QTimer(self)
        self._playback_timer.timeout.connect(self._traj_tick)

        box = QGroupBox("Trajectory")
        layout = QVBoxLayout(box)
        # Vertical gap between rows and inner margins of the group; tweak
        # ``_TRAJ_LAYOUT_SPACING`` / ``_TRAJ_LAYOUT_MARGIN`` at the top of
        # this file to your taste.
        layout.setSpacing(_TRAJ_LAYOUT_SPACING)
        layout.setContentsMargins(
            _TRAJ_LAYOUT_MARGIN,
            _TRAJ_LAYOUT_MARGIN,
            _TRAJ_LAYOUT_MARGIN,
            _TRAJ_LAYOUT_MARGIN,
        )

        self._traj_info = QLabel("")
        self._traj_info.setWordWrap(True)
        layout.addWidget(self._traj_info)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        layout.addWidget(self._slider)

        transport = QWidget()
        t_layout = QHBoxLayout(transport)
        t_layout.setContentsMargins(0, 0, 0, 0)
        self._play_btn = QPushButton("Play")
        self._play_btn.clicked.connect(self._toggle_play)
        stop_btn = QPushButton("Stop")
        stop_btn.clicked.connect(self._stop_playback)
        self._loop_chk = QCheckBox("Loop")
        self._loop_chk.setChecked(True)
        self._loop_chk.toggled.connect(self._on_loop_toggled)
        t_layout.addWidget(self._play_btn)
        t_layout.addWidget(stop_btn)
        t_layout.addWidget(self._loop_chk)
        layout.addWidget(transport)

        dyn_row = QWidget()
        dyn_layout = QHBoxLayout(dyn_row)
        dyn_layout.setContentsMargins(0, 0, 0, 0)
        self._dynamic_chk = QCheckBox("Dynamic bonds")
        self._dynamic_chk.setChecked(True)
        dyn_layout.addWidget(self._dynamic_chk)
        dyn_layout.addStretch(1)
        layout.addWidget(dyn_row)

        speed_row = QWidget()
        speed_layout = QHBoxLayout(speed_row)
        speed_layout.setContentsMargins(0, 0, 0, 0)
        speed_layout.addWidget(QLabel("Speed"))
        self._speed_slider = QSlider(Qt.Orientation.Horizontal)
        self._speed_slider.setRange(0, 100)
        self._speed_slider.setValue(self._slider_from_speed(1.0))
        self._speed_slider.valueChanged.connect(self._on_speed_slider)
        speed_layout.addWidget(self._speed_slider, 1)
        self._speed_label = QLabel(self._format_speed(1.0))
        speed_layout.addWidget(self._speed_label)
        layout.addWidget(speed_row)

        interp_row = QWidget()
        interp_layout = QHBoxLayout(interp_row)
        interp_layout.setContentsMargins(0, 0, 0, 0)
        interp_layout.addWidget(QLabel("Interp."))
        self._interp_combo = QComboBox()
        self._interp_combo.addItems([_INTERP_AUTO, _INTERP_LINEAR, _INTERP_OFF])
        self._interp_combo.setCurrentText(_INTERP_AUTO)
        self._interp_combo.currentTextChanged.connect(self._on_interp_changed)
        interp_layout.addWidget(self._interp_combo, 1)
        self._fps_label = QLabel("")
        interp_layout.addWidget(self._fps_label)
        layout.addWidget(interp_row)

        # Push all the rows to the top of the group box: the "Frame ..." line
        # stays at the very top and any leftover height collects at the bottom.
        layout.addStretch(1)

        self._slider.sliderMoved.connect(self._on_slider_moved)
        self._slider.valueChanged.connect(self._on_slider_value)

        self._future_layout.addWidget(box)
        self._traj_box = box
        box.setVisible(False)

    def _on_loop_toggled(self, checked: bool) -> None:
        self._loop = bool(checked)

    # Speed: the slider maps position 0..100 to 0.1x..5.0x with a quadratic
    # curve, so the left half gives fine control over slow speeds.
    _SPEED_MIN = 0.1
    _SPEED_MAX = 5.0

    @staticmethod
    def _speed_from_slider(value: int) -> float:
        t = value / 100.0
        return MainWindow._SPEED_MIN + t * t * (MainWindow._SPEED_MAX - MainWindow._SPEED_MIN)

    @staticmethod
    def _slider_from_speed(speed: float) -> int:
        t = (max(float(speed), MainWindow._SPEED_MIN) - MainWindow._SPEED_MIN) / (
            MainWindow._SPEED_MAX - MainWindow._SPEED_MIN
        )
        return int(round(math.sqrt(max(t, 0.0)) * 100.0))

    def _format_speed(self, value: float) -> str:
        return f"{value:.2f}×"

    def _on_speed_slider(self, value: int) -> None:
        self._speed = self._speed_from_slider(value)
        self._speed_label.setText(self._format_speed(self._speed))

    def _on_interp_changed(self, _text: str) -> None:
        if self._interp_combo.currentText() == _INTERP_AUTO:
            self._auto_linear = True

    def _on_slider_moved(self, value: int) -> None:
        # Pause while scrubbing to avoid fighting the timer.
        if self._playing:
            self._set_playing(False)
        self._playback_t = float(value)
        self._render_pose()

    def _on_slider_value(self, value: int) -> None:
        if not self._playing:
            self._playback_t = float(value)
            self._render_pose()

    def _set_trajectory_controls_visible(self, visible: bool) -> None:
        if not hasattr(self, "_traj_box"):
            return
        self._traj_box.setVisible(visible)

    def _toggle_play(self) -> None:
        if self._trajectory is None:
            return
        if self._playing:
            self._set_playing(False)
            return
        if self._playback_t >= self._trajectory.n_frames - 1:
            self._playback_t = 0.0
        self._set_playing(True)

    def _set_playing(self, on: bool) -> None:
        self._playing = on
        if on:
            self._fps_ticks = 0
            self._fps_start = 0.0
            self._playback_timer.start(self._refresh_ms)
        else:
            self._playback_timer.stop()
            self._fps_label.setText("")
        self._play_btn.setText("Pause" if on else "Play")

    def _stop_playback(self) -> None:
        self._set_playing(False)
        if self._trajectory is None:
            return
        self._playback_t = 0.0
        self._render_pose()

    def _traj_tick(self) -> None:
        trajectory = self._trajectory
        if trajectory is None:
            return
        n = trajectory.n_frames
        if n > 1:
            step = self._speed * _PLAYBACK_BASE_FPS * (self._refresh_ms / 1000.0)
            self._playback_t += step
            if self._playback_t >= n - 1 - 1e-9:
                if self._loop:
                    self._playback_t = 0.0
                else:
                    self._playback_t = float(n - 1)
                    self._set_playing(False)
        self._render_pose()
        self._account_fps()

    def _render_pose(self) -> None:
        """Show the pose at ``self._playback_t`` (interpolated if enabled)."""
        trajectory = self._trajectory
        if trajectory is None:
            return
        self._playback_t = min(
            max(self._playback_t, 0.0), float(trajectory.n_frames - 1)
        )
        # Keep the selected atom across poses (trajectory topology is fixed).
        selected = self.renderer.selected_index
        pose = trajectory.sample(
            self._playback_t, interpolate=self._interpolation_enabled()
        )

        # Bonds for this pose: dynamic (with hysteresis) or fixed from frame 0.
        bonds: np.ndarray | None
        if self._dynamic_chk.isChecked():
            bonds = update_dynamic_bonds(pose, self._prev_dynamic_bonds)
            self._prev_dynamic_bonds = bonds
        else:
            bonds = self._fixed_bonds

        self.renderer.show(pose, bonds=bonds)
        if selected is not None and selected < pose.n_atoms:
            # Redraw the halo on the new pose and refresh the bonds panel.
            self.renderer.select_atom(selected)
            self._update_properties(selected, update_status=False)
        # Keep the measurement overlay on the current pose when active.
        if self._meas_mode is not None and self._meas_atoms:
            self._draw_measurement()

        index = int(round(self._playback_t))
        self._slider.blockSignals(True)
        self._slider.setValue(index)
        self._slider.blockSignals(False)
        self._traj_info.setText(
            f"Frame {index + 1}/{trajectory.n_frames} · "
            f"{trajectory.n_atoms} atoms"
        )
        if self.isVisible():
            self.plotter.render()

    def _interpolation_enabled(self) -> bool:
        mode = self._interp_combo.currentText()
        if mode == _INTERP_LINEAR:
            return True
        if mode == _INTERP_OFF:
            return False
        return self._auto_linear  # Auto: decided by the FPS watchdog

    def _account_fps(self) -> None:
        """Count rendered poses and (in Auto mode) toggle interpolation."""
        now = time.monotonic()
        self._fps_ticks += 1
        if self._fps_start <= 0.0:
            self._fps_start = now
            return
        elapsed = now - self._fps_start
        if elapsed >= 1.0:
            fps = self._fps_ticks / elapsed
            self._fps_ticks = 0
            self._fps_start = now
            if self._interp_combo.currentText() == _INTERP_AUTO:
                # Hysteresis avoids flickering near the thresholds.
                if fps >= _FPS_HIGH and not self._auto_linear:
                    self._auto_linear = True
                elif fps < _FPS_LOW and self._auto_linear:
                    self._auto_linear = False
            self._fps_label.setText(f"{fps:.0f} fps")

    # -- volumetric isosurfaces ---------------------------------------------

    def _build_volumetric_controls(self) -> None:
        """Build the volumetric panel in the lower part of the side panel."""
        box = QGroupBox("Volumetric")
        layout = QVBoxLayout(box)
        layout.setSpacing(6)

        self._vol_info = QLabel("")
        self._vol_info.setWordWrap(True)
        layout.addWidget(self._vol_info)

        self._plot_btn = QPushButton("Plot isosurface")
        # Give the label some vertical breathing room inside the button.
        self._plot_btn.setMinimumHeight(32)
        self._plot_btn.clicked.connect(self._on_plot_isosurface)
        layout.addWidget(self._plot_btn)

        # Isovalue/Quality rows: kept hidden until the first "Plot isosurface"
        # so the section starts compact (just the info line and the button).
        iso_row = QWidget()
        iso_layout = QHBoxLayout(iso_row)
        iso_layout.setContentsMargins(0, 0, 0, 0)
        iso_layout.addWidget(QLabel("Isovalue"))
        self._iso_spin = _IsoSpinBox()
        self._iso_spin.setRange(1e-9, 1e6)
        self._iso_spin.setValue(0.05)
        self._iso_spin.valueChanged.connect(self._on_iso_changed)
        iso_layout.addWidget(self._iso_spin, 1)
        layout.addWidget(iso_row)
        iso_row.setVisible(False)
        self._iso_row = iso_row

        qual_row = QWidget()
        qual_layout = QHBoxLayout(qual_row)
        qual_layout.setContentsMargins(0, 0, 0, 0)
        qual_layout.addWidget(QLabel("Quality"))
        self._quality_combo = QComboBox()
        # Low is omitted on purpose: at this scale it is visually almost
        # identical to Medium.
        self._quality_combo.addItems(["Medium", "High"])
        self._quality_combo.setCurrentText("Medium")
        self._quality_combo.currentTextChanged.connect(self._on_quality_changed)
        qual_layout.addWidget(self._quality_combo, 1)
        layout.addWidget(qual_row)
        qual_row.setVisible(False)
        self._qual_row = qual_row

        self._vol_list = QListWidget()
        self._vol_list.itemSelectionChanged.connect(
            self._on_vol_selection_changed
        )
        layout.addWidget(self._vol_list)

        self._future_layout.addWidget(box)
        self._vol_box = box
        box.setVisible(False)

    def _set_volumetric_controls_visible(self, visible: bool) -> None:
        if hasattr(self, "_vol_box"):
            self._vol_box.setVisible(visible)

    def _on_iso_changed(self, _value: float) -> None:
        self._iso_manual = True
        if self._plotted_index is not None:
            self._plot_volume(self._plotted_index)

    def _on_quality_changed(self, _text: str) -> None:
        if self._plotted_index is not None:
            self._plot_volume(self._plotted_index)

    def _schedule_sections_fit(self) -> None:
        """Re-fit the side-panel sections after a resize of the panel."""
        QTimer.singleShot(0, self._limit_volumetric_height)
        self._schedule_properties_fit()

    def _schedule_properties_fit(self) -> None:
        """Fit the Atom Properties height to its text on the next idle tick."""
        if self._props_fit_scheduled:
            return
        self._props_fit_scheduled = True
        QTimer.singleShot(0, self._apply_properties_fit)

    def _apply_properties_fit(self) -> None:
        """Give the properties scroll the height of its text content (capped)."""
        self._props_fit_scheduled = False
        if not hasattr(self, "_props_panel"):
            return
        panel = self._props_panel
        cap = max(60, int(panel.height() * panel._MAX_HEIGHT_FRACTION))
        needed = self._properties_content_height(panel.label)
        # Pin the scroll area (min == max) to the measured text height: it
        # grows with the content up to the cap and shrinks back when the
        # content gets shorter. Pinning both bounds is required because the
        # scroll area does not auto-size to its child.
        height = min(max(60, needed), cap)
        panel._scroll.setMinimumHeight(height)
        panel._scroll.setMaximumHeight(height)

    def _properties_content_height(self, label: QLabel) -> int:
        """Text height of ``label`` wrapped at its current width."""
        doc = QTextDocument()
        doc.setDefaultFont(label.font())
        doc.setHtml(label.text())
        margin = label.margin()
        doc.setTextWidth(max(1, label.width() - 2 * margin))
        return int(doc.size().height()) + 2 * margin

    def _limit_volumetric_height(self) -> None:
        """Keep the Volumetric section to about half of the free panel area."""
        if not (hasattr(self, "_props_panel") and hasattr(self, "_vol_box")):
            return
        available = self._props_panel.future.height()
        self._vol_box.setMaximumHeight(max(120, available // 2))

    def _on_plot_isosurface(self) -> None:
        """Plot the first orbital/field and show the list of fields."""
        volfile = self._volfile
        if volfile is None or not volfile.labels:
            return
        # First plot: reveal the Isovalue/Quality controls.
        self._iso_row.setVisible(True)
        self._qual_row.setVisible(True)
        self._vol_list.blockSignals(True)
        self._vol_list.clear()
        self._vol_list.addItems(volfile.labels)
        self._vol_list.setVisible(True)
        self._vol_list.blockSignals(False)
        self._limit_volumetric_height()
        self._plot_volume(0)
        self._vol_list.setCurrentRow(0)

    def _on_vol_selection_changed(self) -> None:
        row = self._vol_list.currentRow()
        if row >= 0:
            self._plot_volume(row)

    def _plot_volume(self, index: int) -> None:
        volfile = self._volfile
        if volfile is None:
            return
        volume = self._volume_cache.get(index)
        if volume is None:
            try:
                volume = volfile.volume(index)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.critical(self, "Error computing isosurface", str(exc))
                return
            self._volume_cache[index] = volume
        if not self._iso_manual:
            # Suggest an absolute level that makes the first surface visible
            # without filling the whole box (user can then override it).
            suggested = volume.suggest_isovalue()
            self._iso_spin.blockSignals(True)
            self._iso_spin.setValue(suggested)
            self._iso_spin.blockSignals(False)
        level = self._iso_spin.value()
        quality = self._quality_combo.currentText().lower()
        self.vol_renderer.show(volume, level=level if level > 0 else None, quality=quality)
        label = (
            volfile.labels[index]
            if 0 <= index < len(volfile.labels)
            else f"field {index + 1}"
        )
        self._plotted_index = index
        self._vol_info.setText(f"Showing {label}")
        self.statusBar().showMessage(f"Isosurface: {label}")
        if self.isVisible():
            self.plotter.render()

    def export_image(self) -> None:
        if self._molecule is None:
            QMessageBox.information(self, "Export", "Load a structure first.")
            return
        default = f"{self._molecule.name or 'pychem'}.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export image", default, "PNG images (*.png)"
        )
        if not path:
            return
        try:
            self.plotter.screenshot(str(path))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Error exporting", str(exc))
            return
        self.statusBar().showMessage(f"Image saved: {path}")

    def reset_camera(self) -> None:
        self.plotter.reset_camera()

    def show_about(self) -> None:
        """Non-blocking "About" dialog.

        A hand-built ``QMessageBox`` shown with ``open()`` is used instead of
        ``QMessageBox.about`` (which starts a nested modal loop over the
        OpenGL viewer). On some desktops that nested loop never activates the
        dialog window and the first click is lost: only the keyboard (Enter)
        worked. With ``open()`` + forced focus the dialog accepts clicks from
        the start.
        """
        if self._about_dialog is not None:  # already open; just bring to front
            self._about_dialog.raise_()
            self._about_dialog.activateWindow()
            return

        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Information)
        dialog.setWindowTitle("About PyChem")
        dialog.setText(
            "PyChem 0.1.0\n\n"
            "Modular molecular visualization tool in Python.\n"
            "PySide6 + PyVista + RDKit."
        )
        ok_button = dialog.addButton("OK", QMessageBox.ButtonRole.AcceptRole)
        dialog.setDefaultButton(ok_button)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.finished.connect(self._about_finished)
        self._about_dialog = dialog

        # Center over the main window (same screen).
        dialog.adjustSize()
        center = self.frameGeometry().center()
        dialog.move(center.x() - dialog.width() // 2, center.y() - dialog.height() // 2)

        dialog.open()

        # Force activation and focus: the first click must press the button.
        def _activate() -> None:
            dialog.raise_()
            dialog.activateWindow()
            ok_button.setFocus()

        QTimer.singleShot(0, _activate)

    def _about_finished(self, _result: int) -> None:
        self._about_dialog = None
