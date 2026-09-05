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

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from pychem.core.molecule import Molecule
from pychem.core.trajectory import Trajectory
from pychem.io import get_reader
from pychem.rendering import MolecularRenderer, RenderOptions

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


class _PropertiesPanel(QWidget):
    """Side-panel contents.

    The selected-atom properties section occupies at most the top quarter of
    the panel; the remaining space is reserved for future controls (actions
    on the molecule).
    """

    _QUARTER = 0.25

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Properties section (limited to the top quarter), scrollable.
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
        outer.addWidget(self._scroll, stretch=1)

        # Bottom area reserved for future actions.
        self.future = QWidget(self)
        self.future_layout = QVBoxLayout(self.future)
        self.future_layout.setContentsMargins(6, 6, 6, 6)
        outer.addWidget(self.future, stretch=3)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        self._scroll.setMaximumHeight(
            max(60, int(self.height() * self._QUARTER))
        )
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

        self.setWindowTitle("PyChem")
        self.resize(1100, 800)

        self._build_central_widget()
        self._build_menu()
        self._build_properties_dock()
        self._setup_picking()
        self._build_trajectory_controls()
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
        layout.addWidget(self.plotter.interactor)
        self.setCentralWidget(frame)

    def _build_properties_dock(self) -> None:
        """Fixed side panel (no close/float buttons).

        The properties section is limited to the top quarter; the rest is
        reserved for future controls.
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
        """Click on empty space/bonds: remove the halo and clear the panel."""
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

    def _load(self, path: str | Path) -> Molecule | Trajectory:
        """Load a file as a Molecule or, for multi-frame XYZ, a Trajectory."""
        p = Path(path)
        if p.suffix.lower() == ".xyz":
            from pychem.io.xyz import read_xyz_frames

            frames = read_xyz_frames(p)
            if len(frames) > 1:
                return Trajectory.from_molecules(frames, name=p.stem)
            molecule = frames[0]
            molecule.name = p.stem
            return molecule
        return get_reader(p)

    def open_path(self, path: str | Path) -> None:
        """Load a structure/trajectory and show it in the viewer."""
        path = Path(path)
        try:
            data = self._load(path)
        except Exception as exc:  # noqa: BLE001 - reader errors to the UI
            QMessageBox.critical(self, "Error opening file", str(exc))
            return

        self._set_playing(False)
        trajectory = data if isinstance(data, Trajectory) else None
        molecule = data if trajectory is None else data.get_frame(0)

        self._trajectory = trajectory
        self._molecule = molecule
        self.renderer.show(molecule)
        self.plotter.reset_camera()
        self.renderer.select_atom(None)
        self._update_properties(None)

        if trajectory is not None:
            self._activate_trajectory(trajectory, path)
        else:
            self._set_trajectory_controls_visible(False)

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

    def _activate_trajectory(self, trajectory: Trajectory, path: Path) -> None:
        """Set up the playback state and controls for ``trajectory``."""
        self._playback_t = 0.0
        self._speed = 1.0
        self._loop = True
        self._auto_linear = True

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
        """Receive a picked atom index and update the selection and panel."""
        if self.renderer.molecule is None:
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
        if update_status and self.renderer.molecule is not None and index is not None:
            atom = self.renderer.molecule.get_atom(index)
            self.statusBar().showMessage(f"Selected atom: {index + 1} ({atom.symbol})")

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
        self.renderer.show(pose)
        if selected is not None and selected < pose.n_atoms:
            # Redraw the halo on the new pose and refresh the bonds panel.
            self.renderer.select_atom(selected)
            self._update_properties(selected, update_status=False)

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
