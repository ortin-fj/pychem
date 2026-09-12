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

from collections.abc import Sequence

import math
import time
from pathlib import Path

import numpy as np

from PySide6.QtCore import QLocale, QTimer, Qt
from PySide6.QtGui import QMouseEvent, QResizeEvent
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
    QSizePolicy,
    QSlider,
    QSplitter,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from pychem import __version__ as PYCHEN_VERSION
from pychem.core import geometry
from pychem.core.bonds import guess_bonds, update_dynamic_bonds
from pychem.core.molecule import Molecule
from pychem.core.trajectory import Trajectory
from pychem.core.volume import VolumeData
from pychem.io import get_reader
from pychem.io.volumetric import (
    DENSITY_LABEL,
    POTENTIAL_LABEL,
    EspPair,
    VolumetricFile,
    atomic_orbital_label,
    atomic_orbitals_for_atom,
    evaluate_atomic_orbital,
    load_volume_file,
    read_esp_pair,
)
from pychem.rendering import MolecularRenderer, RenderOptions
from pychem.rendering.esp_renderer import ElectrostaticRenderer
from pychem.rendering.volume_renderer import VolumeRenderer

# Extensions accepted by the "Open" dialog. Extended as formats are added.
_OPEN_FILTER = "Structures (*.xyz *.pdb);;All files (*)"

# Tools > Visualize Electrostatic Potential. An ESP map is built from the
# electron-density and the electrostatic-potential cubes of the same
# structure, so both are picked at once and told apart by the cube's second
# line (see pychem/io/volumetric.py). The action tooltip carries the full
# method to generate the files; the (short) status hint and the error dialog
# repeat the essentials.
_ESP_FILTER = "Cube files (*.cube);;All files (*)"
_ESP_CAPTION = (
    "Select the 2 cubes written by orca_plot "
    f"({DENSITY_LABEL!r} + {POTENTIAL_LABEL!r})"
)
_ESP_STATUS = (
    "Select the density and potential cubes written by orca_plot for this "
    "structure (order does not matter)."
)
_ESP_METHOD = (
    "Paint the electrostatic potential on the electron-density envelope.\n"
    "An ESP map needs two Gaussian cubes of the same structure, both written "
    "by orca_plot from the same .gbw file (orca_plot job.gbw -i): the electron "
    "density and the electrostatic potential. orca_plot records the contents on "
    "the second line of each cube ("
    f"{DENSITY_LABEL!r} and {POTENTIAL_LABEL!r}); PyChem reads that line to "
    "tell the two files apart, so the selection order does not matter."
)

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


class _CollapsibleBox(QGroupBox):
    """A group box that minimizes/expands from its header strip.

    The header holds the section title (left) and a small window-style button
    on the right that swaps between a minimize glyph (``—``) while the section
    is expanded and an expand glyph (``▢``) while it is collapsed. Clicking
    the header strip itself (title included) toggles the section too, instead
    of the old native title check box.

    While collapsed the body widgets are hidden and the box is capped to the
    header height, so the owning panel can pack the sections from the top.
    """

    # Glyphs of the header toggle button (window-style minimize/maximize).
    _MIN_GLYPH = "\u2014"  # shown while expanded: click to minimize
    _EXPAND_GLYPH = "\u25a2"  # shown while collapsed: click to expand
    # Extra vertical space above the text that makes up the header strip.
    _HEADER_PAD = 26

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        self._collapsed = False
        self._header_pressed = False
        self._open_height: int | None = None  # body height restored on expand
        # The native title check box is not used: the toggle lives in the
        # header button (and in the header strip itself).
        self.setCheckable(False)
        # Bold title; the panel's QSS resets the body widgets to normal weight
        # so only the section header is emphasised.
        font = self.font()
        font.setBold(True)
        self.setFont(font)

        # Small window-style button pinned to the right of the header strip.
        self.toggle_button = QToolButton(self)
        self.toggle_button.setObjectName("sectionToggleButton")
        self.toggle_button.setAutoRaise(True)
        self.toggle_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.toggle_button.setFixedSize(20, 18)
        self.toggle_button.clicked.connect(self.toggle)
        self._refresh_button()

    # -- state ---------------------------------------------------------------

    def is_collapsed(self) -> bool:
        return self._collapsed

    def set_collapsed(self, collapsed: bool) -> None:
        """Collapse (hide the body) or expand (show it again)."""
        collapsed = bool(collapsed)
        if collapsed == self._collapsed:
            return
        if collapsed:
            # Remember the current full height (the splitter may have given the
            # section more room than its content) so expanding can restore it.
            if self.height() > self.header_height():
                self._open_height = self.height()
        self._collapsed = collapsed
        self._apply_state()
        # Let the owning panel repack the sections from the top.
        self.toggled.emit(not collapsed)

    def toggle(self) -> None:
        self.set_collapsed(not self._collapsed)

    # -- presentation --------------------------------------------------------

    def header_height(self) -> int:
        """Height of the header strip (the whole box when collapsed)."""
        text_h = max(self.fontMetrics().height(), self.toggle_button.height())
        return text_h + self._HEADER_PAD

    def _in_header(self, y: float) -> bool:
        return y <= self.header_height()

    def _refresh_button(self) -> None:
        btn = self.toggle_button
        if self._collapsed:
            btn.setText(self._EXPAND_GLYPH)
            btn.setToolTip("Expand")
        else:
            btn.setText(self._MIN_GLYPH)
            btn.setToolTip("Minimize")

    def _apply_state(self) -> None:
        self._set_body_visible(not self._collapsed)
        self._refresh_button()

    def _set_body_visible(self, visible: bool) -> None:
        lay = self.layout()
        if lay is not None:
            for i in range(lay.count()):
                item = lay.itemAt(i)
                widget = item.widget()
                if widget is not None:
                    widget.setVisible(visible)
        if visible:
            # Free the height again: the splitter redistributes the space.
            self.setMaximumHeight(16777215)
        else:
            # Only the header remains; cap the section to its height so the
            # splitter cannot stretch the collapsed frame to fill the gap.
            self.setMaximumHeight(self.header_height())

    # -- header interaction --------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt API)
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._in_header(event.position().y())
        ):
            self._header_pressed = True
            event.accept()
            return
        self._header_pressed = False
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt API)
        if (
            self._header_pressed
            and event.button() == Qt.MouseButton.LeftButton
            and self._in_header(event.position().y())
        ):
            self._header_pressed = False
            self.toggle()
            event.accept()
            return
        self._header_pressed = False
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 (Qt API)
        super().resizeEvent(event)
        self._place_button()

    def _place_button(self) -> None:
        """Pin the toggle button to the top-right corner of the header."""
        btn = self.toggle_button
        btn.move(max(0, self.width() - btn.width() - 6), 3)
        btn.raise_()


class _PropertiesPanel(QWidget):
    """Right side panel: vertically stacked collapsible sections.

    The sections (Atom Properties, Measurements, Trajectory and Volumetric)
    live in a vertical ``QSplitter``: drag the separator between two sections
    to resize them, or use a section header (the title or the window-style
    button on its right) to minimize it and expand it again.
    """

    # Section titles: bold, accent color (see also the group boxes' QSS).
    # The descendant rule keeps the section *bodies* at normal weight (they
    # must not inherit the bold used for the headers).
    TITLE_STYLE = (
        "QGroupBox QWidget { font-weight: normal; }"
        "QGroupBox::title { color: #87D68D; font-weight: bold;"
        " subcontrol-origin: margin; left: 12px; padding: 0 3px; }"
        "QToolButton#sectionToggleButton { border: none; background: transparent;"
        " color: #87D68D; font-weight: bold; }"
        "QToolButton#sectionToggleButton:hover { color: #b3edba; }"
        "QFrame#aoSecondBox { border: 1px solid #c3cfc3; border-radius: 4px; }"
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(0)
        self.splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(6)
        outer.addWidget(self.splitter)
        self.label: QLabel | None = None  # Atom Properties rich label
        self.setStyleSheet(self.TITLE_STYLE)


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
        # Atomic orbital visualization (Tools > Visualize atomic orbitals).
        self._act_visualize_ao = None
        self._ao_cache: dict[int, VolumeData] = {}
        self._ao_plotted = False
        self._ao_manual_iso = False
        self._ao_filling = False
        self._ao2_plotted = False
        # Electrostatic-potential map (Tools > Visualize Electrostatic
        # Potential): the loaded density/potential cube pair.
        self._act_visualize_esp = None
        self._esp_pair: EspPair | None = None
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
        self._build_side_stretch()
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
        self.esp_renderer = ElectrostaticRenderer(self.plotter)
        layout.addWidget(self.plotter.interactor)
        self.setCentralWidget(frame)

    def _build_properties_dock(self) -> None:
        """Fixed side panel: collapsible/resizable sections in a splitter.

        The Atom Properties section is framed like the other sections and
        every section can be collapsed from its header and resized by
        dragging the splitter separators.
        """
        dock = QDockWidget("", self)
        # Permanent panel: no close/float button, and it cannot be undocked
        # or moved.
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        panel = _PropertiesPanel(self)

        # --- Atom Properties section ---------------------------------------
        props_box = _CollapsibleBox("Atom Properties")
        props_box.toggled.connect(self._on_section_toggled)
        props_layout = QVBoxLayout(props_box)
        props_layout.setContentsMargins(8, 12, 8, 8)
        scroll = QScrollArea(props_box)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        label = QLabel()
        label.setWordWrap(True)
        label.setMargin(8)
        label.setTextFormat(Qt.TextFormat.RichText)
        scroll.setWidget(label)
        props_layout.addWidget(scroll)
        panel.splitter.addWidget(props_box)

        dock.setWidget(panel)
        self._props_panel = panel
        self._props_label = label
        panel.label = label
        self._props_scroll = scroll
        self._props_box = props_box
        self._props_dock = dock
        self._side_splitter = panel.splitter
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        panel.label.setText(self._properties_html(None))

    def _build_side_stretch(self) -> None:
        """Append a stretch cell to the side splitter.

        The ``QSplitter`` cannot be taller than the sum of its children's
        maximum heights, so when every section is minimized it shrinks and the
        surrounding layout would center it in the panel. A trailing spacer
        that is allowed to grow absorbs the leftover vertical space instead,
        which keeps the sections (minimized or not) packed at the top of the
        side panel.
        """
        spacer = QWidget(self._side_splitter)
        policy = spacer.sizePolicy()
        policy.setVerticalPolicy(QSizePolicy.Policy.Expanding)
        spacer.setSizePolicy(policy)
        spacer.setObjectName("sideStretch")
        self._side_splitter.addWidget(spacer)
        self._side_spacer = spacer

    def _on_section_toggled(self, _checked: bool) -> None:
        """A section header was collapsed/expanded: keep width, repack top."""
        self._stabilize_panel_width()
        self._pack_sections()

    def _stabilize_panel_width(self) -> None:
        """Keep the side panel width when every visible section is collapsed.

        Hiding the sections drops the content-driven minimum width of the
        dock, so a dock whose width was never set by the user would auto-shrink
        to that new minimum (e.g. 270 -> 160 px). Committing the current width
        back into the dock layout at that moment preserves it, without pinning
        any minimum: the user can still drag the panel narrower afterwards,
        down to the (smaller) collapsed content minimum.
        """
        dock = self._props_dock
        if not dock.isVisible():
            return
        splitter = self._side_splitter
        visible = [
            splitter.widget(i)
            for i in range(splitter.count())
            if isinstance(splitter.widget(i), _CollapsibleBox)
            and not splitter.widget(i).isHidden()
        ]
        if not visible or not all(box.is_collapsed() for box in visible):
            return
        width = dock.width()
        if width > 0:
            self.resizeDocks([dock], [width], Qt.Orientation.Horizontal)

    def _pack_sections(self) -> None:
        """Repack the visible side sections from the top of the panel.

        Collapsed sections stay capped at their header height, expanded ones
        keep the height they had before the toggle (or their natural size on
        the first toggle) and the trailing spacer absorbs every pixel they do
        not use. The headers therefore always line up at the top of the panel
        after a minimize/expand.
        """
        splitter = self._side_splitter
        count = splitter.count()
        current = splitter.sizes()
        spacer = getattr(self, "_side_spacer", None)
        # Remember the current real size of every expanded section (this also
        # follows manual separator drags) so a collapse/expand cycle can give
        # each section back the height it had.
        for i in range(count):
            w = splitter.widget(i)
            if (
                isinstance(w, _CollapsibleBox)
                and not w.isHidden()
                and not w.is_collapsed()
                and current[i] > w.header_height()
            ):
                w._open_height = current[i]
        # Visible children that take vertical room: shown sections + the spacer.
        visible = [
            (i, splitter.widget(i))
            for i in range(count)
            if (isinstance(splitter.widget(i), _CollapsibleBox) and not splitter.widget(i).isHidden())
            or splitter.widget(i) is spacer
        ]
        sizes = list(current)
        used = 0
        for i, w in visible:
            if not isinstance(w, _CollapsibleBox):
                continue  # the spacer: filled with the leftover below
            if w.is_collapsed():
                sizes[i] = w.header_height()
            else:
                open_h = w._open_height
                if open_h is None:
                    open_h = max(w.sizeHint().height(), w.header_height())
                sizes[i] = max(open_h, w.header_height())
            used += sizes[i]
        # Give the leftover vertical space to the spacer at the bottom.
        if spacer is not None:
            handles = max(0, len(visible) - 1) * splitter.handleWidth()
            avail = splitter.height() - handles
            idx = -1
            for i in range(count):
                if splitter.widget(i) is spacer:
                    idx = i
                    break
            if idx >= 0:
                sizes[idx] = max(0, avail - used)
        splitter.setSizes(sizes)

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

        tools_menu = menu.addMenu("&Tools")
        act_ao = tools_menu.addAction("&Visualize atomic orbitals")
        # Only meaningful for Molden files, which carry the basis functions.
        act_ao.setCheckable(True)
        act_ao.setEnabled(False)
        act_ao.triggered.connect(self._on_visualize_ao_toggled)
        self._act_visualize_ao = act_ao

        act_esp = tools_menu.addAction("Visualize &Electrostatic Potential")
        # Needs a structure on screen to paint the potential around.
        act_esp.setCheckable(True)
        act_esp.setEnabled(False)
        # The tooltip tells the user how to generate the two cubes.
        act_esp.setToolTip(_ESP_METHOD)
        act_esp.triggered.connect(self._on_visualize_esp_toggled)
        self._act_visualize_esp = act_esp
        tools_menu.setToolTipsVisible(True)

        act_ids = tools_menu.addAction("&Show ID label")
        # Overlays the atom number on every atom; works for any molecule on
        # screen (plain structures, trajectories and volumetric files).
        act_ids.setCheckable(True)
        act_ids.setEnabled(False)
        act_ids.triggered.connect(self._on_show_ids_toggled)
        self._act_show_ids = act_ids

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
        self._deactivate_esp()
        self._volume_cache = {}
        self._plotted_index = None
        self._iso_manual = False
        self._ao_cache = {}
        self._ao_plotted = False
        self._ao_manual_iso = False
        self._ao2_plotted = False
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

        # Atomic-orbital mode is only available for Molden files and the
        # volumetric panel always restarts on the molecular-orbitals page.
        has_basis = volfile is not None and getattr(volfile, "basis", None) is not None
        if self._act_visualize_ao is not None:
            self._act_visualize_ao.setEnabled(has_basis)
            if not has_basis and self._act_visualize_ao.isChecked():
                self._act_visualize_ao.blockSignals(True)
                self._act_visualize_ao.setChecked(False)
                self._act_visualize_ao.blockSignals(False)
        self._exit_ao_mode()

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
        # "Show ID label" needs an actual molecule on screen.
        if self._act_show_ids is not None:
            self._act_show_ids.setEnabled(molecule is not None)
            if molecule is None and self._act_show_ids.isChecked():
                self._act_show_ids.blockSignals(True)
                self._act_show_ids.setChecked(False)
                self._act_show_ids.blockSignals(False)
                self.renderer.set_id_labels(False)
        # "Visualize Electrostatic Potential" also needs a structure on screen.
        if self._act_visualize_esp is not None:
            self._act_visualize_esp.setEnabled(molecule is not None)
        self._stabilize_panel_width()

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
            f"ID {index + 1} · {atom.symbol}",
            "Coordinates (Å):",
            "<table cellspacing='0' cellpadding='1'>"
            f"<tr><td>x</td><td>{atom.x:.4f}</td></tr>"
            f"<tr><td>y</td><td>{atom.y:.4f}</td></tr>"
            f"<tr><td>z</td><td>{atom.z:.4f}</td></tr>"
            "</table>",
        ]

        neighbors = self.renderer.neighbor_indices(index)
        if neighbors.size:
            lines.append(f"Bonds ({neighbors.size}):")
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

    # -- measurements --------------------------------------------------------

    def _build_measurement_controls(self) -> None:
        """Build the Measurements section: three exclusive mode buttons."""
        box = _CollapsibleBox("Measurements")
        box.toggled.connect(self._on_section_toggled)
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
        self._side_splitter.addWidget(box)
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

        box = _CollapsibleBox("Trajectory")
        box.toggled.connect(self._on_section_toggled)
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

        self._side_splitter.addWidget(box)
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
        """Build the volumetric panel in the lower part of the side panel.

        The group box holds two interchangeable pages (a ``QStackedWidget``):

        - Molecular orbitals: an info line, a "Plot isosurface" button and a
          list of the fields/orbitals.
        - Atomic orbitals (Tools menu, Molden files): pick an atom, the
          angular-momentum type (s/p/...) and one of its basis functions.

        Both pages mirror the same Isovalue/Quality rows (only revealed after
        the first plot) and drive the shared :class:`VolumeRenderer`.
        """
        box = _CollapsibleBox("Volumetric")
        box.toggled.connect(self._on_section_toggled)
        outer = QVBoxLayout(box)
        outer.setSpacing(6)
        self._vol_stack = QStackedWidget(box)
        outer.addWidget(self._vol_stack)
        # The title reflects which page is active (molecular/atomic orbitals).
        box.setTitle("Volumetric: MO")

        # ---- page: molecular orbitals ------------------------------------
        mo_page = QWidget()
        layout = QVBoxLayout(mo_page)
        layout.setContentsMargins(0, 0, 0, 0)
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

        self._mo_page = mo_page
        self._vol_stack.addWidget(mo_page)

        # ---- page: atomic orbitals ----------------------------------------
        ao_page = QWidget()
        ao_layout = QVBoxLayout(ao_page)
        ao_layout.setContentsMargins(0, 0, 0, 0)
        ao_layout.setSpacing(6)

        atom_row = QWidget()
        atom_layout = QHBoxLayout(atom_row)
        atom_layout.setContentsMargins(0, 0, 0, 0)
        atom_layout.addWidget(QLabel("Atom"))
        self._ao_atom_combo = QComboBox()
        self._ao_atom_combo.currentIndexChanged.connect(self._on_ao_atom_changed)
        atom_layout.addWidget(self._ao_atom_combo, 1)
        ao_layout.addWidget(atom_row)

        type_row = QWidget()
        type_layout = QHBoxLayout(type_row)
        type_layout.setContentsMargins(0, 0, 0, 0)
        type_layout.addWidget(QLabel("Type"))
        self._ao_type_combo = QComboBox()
        self._ao_type_combo.currentIndexChanged.connect(self._on_ao_type_changed)
        type_layout.addWidget(self._ao_type_combo, 1)
        ao_layout.addWidget(type_row)

        orb_row = QWidget()
        orb_layout = QHBoxLayout(orb_row)
        orb_layout.setContentsMargins(0, 0, 0, 0)
        orb_layout.addWidget(QLabel("Orbital"))
        self._ao_orb_combo = QComboBox()
        self._ao_orb_combo.currentIndexChanged.connect(
            self._on_ao_orbital_changed
        )
        orb_layout.addWidget(self._ao_orb_combo, 1)
        ao_layout.addWidget(orb_row)

        self._ao_plot_btn = QPushButton("Plot AO")
        self._ao_plot_btn.setMinimumHeight(32)
        self._ao_plot_btn.clicked.connect(self._on_plot_atomic_orbital)
        ao_layout.addWidget(self._ao_plot_btn)

        # Isovalue/Quality rows mirroring the molecular-orbital ones: hidden
        # until the first plot, then an editable, auto-suggested isovalue.
        ao_iso_row = QWidget()
        ao_iso_layout = QHBoxLayout(ao_iso_row)
        ao_iso_layout.setContentsMargins(0, 0, 0, 0)
        ao_iso_layout.addWidget(QLabel("Isovalue"))
        self._ao_iso_spin = _IsoSpinBox()
        self._ao_iso_spin.setRange(1e-9, 1e6)
        self._ao_iso_spin.setValue(0.05)
        self._ao_iso_spin.valueChanged.connect(self._on_ao_iso_changed)
        ao_iso_layout.addWidget(self._ao_iso_spin, 1)
        ao_layout.addWidget(ao_iso_row)
        ao_iso_row.setVisible(False)
        self._ao_iso_row = ao_iso_row

        ao_qual_row = QWidget()
        ao_qual_layout = QHBoxLayout(ao_qual_row)
        ao_qual_layout.setContentsMargins(0, 0, 0, 0)
        ao_qual_layout.addWidget(QLabel("Quality"))
        self._ao_quality_combo = QComboBox()
        self._ao_quality_combo.addItems(["Medium", "High"])
        self._ao_quality_combo.setCurrentText("Medium")
        self._ao_quality_combo.currentTextChanged.connect(
            self._on_ao_quality_changed
        )
        ao_qual_layout.addWidget(self._ao_quality_combo, 1)
        ao_layout.addWidget(ao_qual_row)
        ao_qual_row.setVisible(False)
        self._ao_qual_row = ao_qual_row

        # "Plot another AO": once the first atomic orbital is on screen the
        # user can add a second one. It is a separate box (no duplicated
        # "Volumetric: AO" title) and at most two AOs can be shown at once.
        self._ao_plot_another_btn = QPushButton("Plot another AO")
        self._ao_plot_another_btn.setMinimumHeight(32)
        self._ao_plot_another_btn.clicked.connect(self._on_plot_another_ao)
        ao_layout.addWidget(self._ao_plot_another_btn)
        self._ao_plot_another_btn.setVisible(False)

        # Second atomic-orbital picker: Atom/Type/Orbital + its own "Plot AO"
        # button. It reuses the Isovalue/Quality rows above and is revealed by
        # "Plot another AO".
        ao2_box = QFrame(ao_page)
        ao2_box.setObjectName("aoSecondBox")
        ao2_layout = QVBoxLayout(ao2_box)
        ao2_layout.setContentsMargins(8, 8, 8, 8)
        ao2_layout.setSpacing(6)

        ao2_atom_row = QWidget()
        ao2_atom_layout = QHBoxLayout(ao2_atom_row)
        ao2_atom_layout.setContentsMargins(0, 0, 0, 0)
        ao2_atom_layout.addWidget(QLabel("Atom"))
        self._ao2_atom_combo = QComboBox()
        self._ao2_atom_combo.currentIndexChanged.connect(
            self._on_ao2_atom_changed
        )
        ao2_atom_layout.addWidget(self._ao2_atom_combo, 1)
        ao2_layout.addWidget(ao2_atom_row)

        ao2_type_row = QWidget()
        ao2_type_layout = QHBoxLayout(ao2_type_row)
        ao2_type_layout.setContentsMargins(0, 0, 0, 0)
        ao2_type_layout.addWidget(QLabel("Type"))
        self._ao2_type_combo = QComboBox()
        self._ao2_type_combo.currentIndexChanged.connect(
            self._on_ao2_type_changed
        )
        ao2_type_layout.addWidget(self._ao2_type_combo, 1)
        ao2_layout.addWidget(ao2_type_row)

        ao2_orb_row = QWidget()
        ao2_orb_layout = QHBoxLayout(ao2_orb_row)
        ao2_orb_layout.setContentsMargins(0, 0, 0, 0)
        ao2_orb_layout.addWidget(QLabel("Orbital"))
        self._ao2_orb_combo = QComboBox()
        self._ao2_orb_combo.currentIndexChanged.connect(
            self._on_ao2_orbital_changed
        )
        ao2_orb_layout.addWidget(self._ao2_orb_combo, 1)
        ao2_layout.addWidget(ao2_orb_row)

        self._ao2_plot_btn = QPushButton("Plot AO")
        self._ao2_plot_btn.setMinimumHeight(32)
        self._ao2_plot_btn.clicked.connect(self._on_plot_ao2)
        ao2_layout.addWidget(self._ao2_plot_btn)

        ao_layout.addWidget(ao2_box)
        ao2_box.setVisible(False)
        self._ao2_box = ao2_box

        self._ao_page = ao_page
        self._vol_stack.addWidget(ao_page)
        self._vol_stack.setCurrentWidget(mo_page)

        self._side_splitter.addWidget(box)
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
        # An orbital isosurface replaces a previously shown ESP map.
        self._deactivate_esp()
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

    # -- atomic orbitals ----------------------------------------------------

    @staticmethod
    def _ao_type_name(l: int) -> str:
        """Spectroscopic letter for an angular momentum (0=s, 1=p, 2=d...)."""
        return {0: "s", 1: "p", 2: "d", 3: "f", 4: "g"}.get(l, str(l))

    def _on_show_ids_toggled(self, checked: bool) -> None:
        """Tools > Show ID label: overlay the per-atom ID numbers."""
        if self.renderer.molecule is None:
            # Nothing to label; keep the action tidy.
            self.renderer.set_id_labels(False)
            if self._act_show_ids is not None and self._act_show_ids.isChecked():
                self._act_show_ids.blockSignals(True)
                self._act_show_ids.setChecked(False)
                self._act_show_ids.blockSignals(False)
            return
        self.renderer.set_id_labels(bool(checked))
        if self.isVisible():
            self.plotter.render()

    def _on_visualize_ao_toggled(self, checked: bool) -> None:
        """Tools > Visualize atomic orbitals: swap the volumetric panel page."""
        if checked:
            self._enter_ao_mode()
            self.statusBar().showMessage(
                "Atomic orbitals: pick an atom and a basis function, then Plot."
            )
        else:
            self._exit_ao_mode()
            self.statusBar().showMessage(self._file_status or "Ready.")

    def _enter_ao_mode(self) -> None:
        """Switch the volumetric panel to the atomic-orbitals page."""
        volfile = self._volfile
        if volfile is None or volfile.basis is None:
            # The action is disabled in this case; keep the check state tidy.
            if self._act_visualize_ao is not None and self._act_visualize_ao.isChecked():
                self._act_visualize_ao.blockSignals(True)
                self._act_visualize_ao.setChecked(False)
                self._act_visualize_ao.blockSignals(False)
            return
        # Atomic orbitals replace a previously shown ESP map.
        self._deactivate_esp()
        self._reset_ao_session()
        self._vol_box.setTitle("Volumetric: AO")
        self._populate_ao_controls()
        self._vol_stack.setCurrentWidget(self._ao_page)

    def _reset_ao_session(self) -> None:
        """Put the atomic-orbital pickers back to a fresh, empty session.

        Nothing is plotted, the Isovalue/Quality rows are hidden, the first
        "Plot AO" button is ready and the second-AO picker (and its offer) is
        tucked away. It runs when entering AO mode and again when leaving it,
        so the second AO can never stay "open" across a switch to the MO page
        (which would swallow the "Plot another AO" button on return).
        """
        self._ao_plotted = False
        self._ao_manual_iso = False
        self._ao_cache = {}
        self._ao2_plotted = False
        if not hasattr(self, "_ao_plot_another_btn"):
            return
        self._ao_iso_row.setVisible(False)
        self._ao_qual_row.setVisible(False)
        self._ao_plot_btn.setVisible(True)
        self._ao_plot_another_btn.setVisible(False)
        self._ao2_box.setVisible(False)
        self._ao2_plot_btn.setVisible(True)

    def _exit_ao_mode(self) -> None:
        """Return to the MO page and reset the atomic-orbital session."""
        if hasattr(self, "_vol_stack") and hasattr(self, "_mo_page"):
            self._vol_stack.setCurrentWidget(self._mo_page)
            self._vol_box.setTitle("Volumetric: MO")
        # Leaving AO mode clears both pickers: re-entering is always a clean
        # session, so after replotting the first AO the "Plot another AO"
        # offer is there again.
        if hasattr(self, "_ao_plot_another_btn"):
            self._reset_ao_session()

    def _populate_ao_controls(self) -> None:
        """Fill the Atom combo and select the first atom (and its orbitals)."""
        volfile = self._volfile
        if volfile is None or volfile.basis is None:
            return
        # Adding the first item auto-selects it (emitting currentIndexChanged);
        # block signals so the selection below is the one that drives the
        # Type/Orbital population.
        self._ao_atom_combo.blockSignals(True)
        self._ao_atom_combo.clear()
        for i, symbol in enumerate(volfile.molecule.symbols):
            self._ao_atom_combo.addItem(f"{i + 1} · {symbol}", i)
        self._ao_atom_combo.blockSignals(False)
        if self._ao_atom_combo.count():
            # Populate Type/Orbital for the (silently selected) first atom.
            self._on_ao_atom_changed()

    def _on_ao_atom_changed(self) -> None:
        """An atom was chosen: rebuild its Type/Orbital lists and re-plot."""
        if self._ao_filling:
            return
        volfile = self._volfile
        if volfile is None or volfile.basis is None:
            return
        atom = self._ao_atom_combo.currentData()
        if atom is None:
            return
        groups = atomic_orbitals_for_atom(volfile.basis, atom)
        self._ao_filling = True
        self._ao_type_combo.clear()
        for l, _aos in groups.items():
            self._ao_type_combo.addItem(self._ao_type_name(l), l)
        self._ao_filling = False
        self._populate_ao_orbital_list()
        self._maybe_ao_replot()

    def _on_ao_type_changed(self) -> None:
        """A type was chosen: rebuild the Orbital list and re-plot."""
        if self._ao_filling:
            return
        self._populate_ao_orbital_list()
        self._maybe_ao_replot()

    def _on_ao_orbital_changed(self) -> None:
        if not self._ao_filling:
            self._maybe_ao_replot()

    def _populate_ao_orbital_list(self) -> None:
        """Fill the Orbital combo with the basis functions of the current type."""
        volfile = self._volfile
        if volfile is None or volfile.basis is None:
            return
        atom = self._ao_atom_combo.currentData()
        l = self._ao_type_combo.currentData()
        if atom is None or l is None:
            return
        aos = atomic_orbitals_for_atom(volfile.basis, atom).get(l, [])
        self._ao_filling = True
        self._ao_orb_combo.clear()
        for ordinal, ao in enumerate(aos, start=1):
            text = atomic_orbital_label(ao)
            if len(aos) > 1:
                text = f"{text}  #{ordinal}"
            self._ao_orb_combo.addItem(text, ao)
        self._ao_filling = False
        if self._ao_orb_combo.count():
            self._ao_orb_combo.setCurrentIndex(0)

    def _maybe_ao_replot(self) -> None:
        """Re-plot after a selection change once a first plot exists."""
        if self._ao_plotted:
            self._plot_current_ao()

    def _on_plot_atomic_orbital(self) -> None:
        self._plot_current_ao()

    def _on_ao_iso_changed(self, _value: float) -> None:
        self._ao_manual_iso = True
        if self._ao_plotted:
            self._render_ao_scene()

    def _on_ao_quality_changed(self, _text: str) -> None:
        if self._ao_plotted:
            self._render_ao_scene()

    def _plot_current_ao(self) -> None:
        """Start/refresh the first atomic orbital and render the AO scene."""
        if self._volfile is None or self._volfile.basis is None:
            return
        ao = self._ao_orb_combo.currentData()
        if ao is None:
            return
        if not self._ao_plotted:
            # First plot: reveal the Isovalue/Quality controls, hide this Plot
            # button (selection changes re-plot from now on) and offer adding
            # a second atomic orbital (never more than two on screen).
            self._ao_plotted = True
            self._ao_iso_row.setVisible(True)
            self._ao_qual_row.setVisible(True)
            self._ao_plot_btn.setVisible(False)
            self._refresh_ao2_offer()
            if not self._ao_manual_iso:
                volume = self._ao_volume(ao)
                if volume is not None:
                    # Suggest a level that shows the surface without filling
                    # the box (the user can override it: manual isovalue).
                    suggested = volume.suggest_isovalue()
                    self._ao_iso_spin.blockSignals(True)
                    self._ao_iso_spin.setValue(suggested)
                    self._ao_iso_spin.blockSignals(False)
        self._render_ao_scene()

    def _ao_volume(self, ao) -> VolumeData | None:
        """Return the cached (or newly computed) grid of an atomic orbital."""
        volfile = self._volfile
        if ao is None or volfile is None or volfile.basis is None:
            return None
        volume = self._ao_cache.get(ao.index)
        if volume is None:
            try:
                volume = evaluate_atomic_orbital(volfile.basis, ao)
            except Exception as exc:  # noqa: BLE001 - errors to the UI
                QMessageBox.critical(
                    self, "Error computing atomic orbital", str(exc)
                )
                return None
            self._ao_cache[ao.index] = volume
        return volume

    def _render_ao_scene(self) -> None:
        """(Re)render the plotted atomic orbitals: AO1 plus AO2 when plotted.

        Both orbitals share the Isovalue/Quality rows and are added as
        separate overlays, so changing either one redraws the whole pair.
        """
        if self._volfile is None or self._volfile.basis is None:
            return
        level = self._ao_iso_spin.value()
        quality = self._ao_quality_combo.currentText().lower()
        level = level if level > 0 else None
        self.vol_renderer.clear()
        shown: list[str] = []
        if self._ao_plotted:
            ao1 = self._ao_orb_combo.currentData()
            if ao1 is not None:
                volume = self._ao_volume(ao1)
                if volume is not None:
                    self.vol_renderer.add(
                        volume, level=level, quality=quality, name="ao-1"
                    )
                    shown.append(self._ao_orb_combo.currentText())
        if getattr(self, "_ao2_plotted", False):
            ao2 = self._ao2_orb_combo.currentData()
            if ao2 is not None:
                volume = self._ao_volume(ao2)
                if volume is not None:
                    self.vol_renderer.add(
                        volume, level=level, quality=quality, name="ao-2"
                    )
                    shown.append(self._ao2_orb_combo.currentText())
        if shown:
            self.statusBar().showMessage("Isosurfaces: " + " · ".join(shown))
        self._refresh_ao2_offer()
        if self.isVisible():
            self.plotter.render()

    def _refresh_ao2_offer(self) -> None:
        """Keep the 'Plot another AO' offer consistent with the state.

        The button is visible whenever the first AO is on screen and the
        second picker has not been revealed yet; it disappears as soon as the
        second picker is shown (at most two AOs). Recomputing it from the
        state (instead of only on the first plot) restores it after replots
        and after round trips through the MO page.
        """
        if not hasattr(self, "_ao_plot_another_btn"):
            return
        self._ao_plot_another_btn.setVisible(
            self._ao_plotted and self._ao2_box.isHidden()
        )

    # -- second atomic orbital (at most two at once) -----------------------

    def _on_plot_another_ao(self) -> None:
        """Reveal the second atomic-orbital picker (never a third one)."""
        box = getattr(self, "_ao2_box", None)
        if box is None or not box.isHidden():
            return
        self._ao2_plot_btn.setVisible(True)
        box.setVisible(True)
        self._ao_plot_another_btn.setVisible(False)
        self._populate_ao2_controls()

    def _populate_ao2_controls(self) -> None:
        """Fill the second AO's Atom combo and select the first atom."""
        volfile = self._volfile
        if volfile is None or volfile.basis is None:
            return
        self._ao2_atom_combo.blockSignals(True)
        self._ao2_atom_combo.clear()
        for i, symbol in enumerate(volfile.molecule.symbols):
            self._ao2_atom_combo.addItem(f"{i + 1} · {symbol}", i)
        self._ao2_atom_combo.blockSignals(False)
        if self._ao2_atom_combo.count():
            self._on_ao2_atom_changed()

    def _on_ao2_atom_changed(self) -> None:
        """An atom was chosen for the second AO: rebuild its Type/Orbital."""
        if self._ao_filling:
            return
        volfile = self._volfile
        if volfile is None or volfile.basis is None:
            return
        atom = self._ao2_atom_combo.currentData()
        if atom is None:
            return
        groups = atomic_orbitals_for_atom(volfile.basis, atom)
        self._ao_filling = True
        self._ao2_type_combo.clear()
        for l, _aos in groups.items():
            self._ao2_type_combo.addItem(self._ao_type_name(l), l)
        self._ao_filling = False
        self._populate_ao2_orbital_list()
        self._maybe_ao2_replot()

    def _on_ao2_type_changed(self) -> None:
        if self._ao_filling:
            return
        self._populate_ao2_orbital_list()
        self._maybe_ao2_replot()

    def _on_ao2_orbital_changed(self) -> None:
        if not self._ao_filling:
            self._maybe_ao2_replot()

    def _populate_ao2_orbital_list(self) -> None:
        """Fill the second AO's Orbital combo for the current atom/type."""
        volfile = self._volfile
        if volfile is None or volfile.basis is None:
            return
        atom = self._ao2_atom_combo.currentData()
        l = self._ao2_type_combo.currentData()
        if atom is None or l is None:
            return
        aos = atomic_orbitals_for_atom(volfile.basis, atom).get(l, [])
        self._ao_filling = True
        self._ao2_orb_combo.clear()
        for ordinal, ao in enumerate(aos, start=1):
            text = atomic_orbital_label(ao)
            if len(aos) > 1:
                text = f"{text}  #{ordinal}"
            self._ao2_orb_combo.addItem(text, ao)
        self._ao_filling = False
        if self._ao2_orb_combo.count():
            self._ao2_orb_combo.setCurrentIndex(0)

    def _maybe_ao2_replot(self) -> None:
        """Re-plot the pair after a second-AO change once it is on screen."""
        if getattr(self, "_ao2_plotted", False):
            self._render_ao_scene()

    def _on_plot_ao2(self) -> None:
        """Plot the second atomic orbital (it joins the first on screen)."""
        if self._volfile is None or self._volfile.basis is None:
            return
        ao2 = self._ao2_orb_combo.currentData()
        if ao2 is None:
            return
        if not self._ao2_plotted:
            self._ao2_plotted = True
            self._ao2_plot_btn.setVisible(False)
        self._render_ao_scene()

    # -- electrostatic potential ---------------------------------------------

    def _on_visualize_esp_toggled(self, checked: bool) -> None:
        """Tools > Visualize Electrostatic Potential.

        Checking it asks for the density and potential cubes (once — the pair
        is then reused) and paints the potential on the density envelope;
        unchecking it drops the overlay.
        """
        if not checked:
            # Forget the pair as well: next time the cubes are asked for
            # again, so the user can pick a different map.
            self._clear_esp_overlay()
            self.statusBar().showMessage(self._file_status or "Ready.")
            return
        if self.renderer.molecule is None:
            self._uncheck_esp_action()
            return
        if self._esp_pair is None and not self._choose_esp_cubes():
            self._uncheck_esp_action()
            return
        self._draw_esp()

    def _replace_volumetric_overlays(self) -> None:
        """Leave MO/AO isosurfaces so the ESP map is the only volume shown."""
        self.vol_renderer.clear()
        if self._act_visualize_ao is not None and self._act_visualize_ao.isChecked():
            # setChecked does not emit ``triggered``, so leave the AO mode
            # explicitly instead of relying on its handler.
            self._act_visualize_ao.blockSignals(True)
            self._act_visualize_ao.setChecked(False)
            self._act_visualize_ao.blockSignals(False)
            self._exit_ao_mode()

    def _uncheck_esp_action(self) -> None:
        """Uncheck the Tools action without running its handler again."""
        if self._act_visualize_esp is None:
            return
        self._act_visualize_esp.blockSignals(True)
        self._act_visualize_esp.setChecked(False)
        self._act_visualize_esp.blockSignals(False)

    def _deactivate_esp(self) -> None:
        """Uncheck the ESP action and drop its overlay (no dialog)."""
        self._uncheck_esp_action()
        self._clear_esp_overlay()

    def _clear_esp_overlay(self) -> None:
        """Remove the ESP surface/scalar bar and forget the cube pair."""
        self._esp_pair = None
        if not hasattr(self, "esp_renderer"):
            return
        self.esp_renderer.clear()
        if self.isVisible():
            self.plotter.render()

    def _choose_esp_cubes(self) -> bool:
        """Ask for the two cubes and load them; ``True`` when drawn."""
        dialog = QFileDialog(self, _ESP_CAPTION)
        # ExistingFiles enables selecting the density and the potential at once.
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setNameFilter(_ESP_FILTER)
        dialog.setLabelText(QFileDialog.DialogLabel.Accept, "Load ESP map")
        self.statusBar().showMessage(_ESP_STATUS)
        if dialog.exec() != QFileDialog.DialogCode.Accepted:
            self.statusBar().showMessage(self._file_status or "Ready.")
            return False
        try:
            self._load_esp_cubes(dialog.selectedFiles())
        except Exception as exc:  # noqa: BLE001 - reader errors to the UI
            QMessageBox.critical(
                self, "Error loading ESP cubes", f"{exc}\n\n{_ESP_METHOD}"
            )
            return False
        return True

    def _load_esp_cubes(self, paths: Sequence[str | Path]) -> None:
        """Load the density/potential pair and draw the ESP map.

        This is the single entry point for showing an ESP map (the file dialog
        goes through it), so the overlay bookkeeping — replacing the MO/AO
        isosurfaces, keeping the Tools action in sync — happens here. Kept
        separate from the dialog so it can be driven directly by tests.
        """
        pair = read_esp_pair(paths)
        level = pair.density.suggest_density_isovalue()
        if not self.esp_renderer.show(pair.density, pair.potential, level=level):
            raise ValueError(
                "The density cube has no surface at its envelope level "
                f"({level:g}); it may be empty or stored on a different scale."
            )
        self._esp_pair = pair
        # Only once the map is on screen: an ESP map and an orbital isosurface
        # are alternative volumetric overlays.
        self._replace_volumetric_overlays()
        if (
            self._act_visualize_esp is not None
            and not self._act_visualize_esp.isChecked()
        ):
            self._act_visualize_esp.blockSignals(True)
            self._act_visualize_esp.setChecked(True)
            self._act_visualize_esp.blockSignals(False)
        self._draw_esp()

    def _draw_esp(self) -> None:
        """Report the loaded map (including its colour range) and refresh."""
        pair = self._esp_pair
        if pair is None:
            return
        text = (
            "Electrostatic potential on the density envelope "
            f"({Path(pair.density_path).name} + {Path(pair.potential_path).name})"
        )
        clim = self.esp_renderer.clim
        if clim is not None:
            text += f" · {clim[0]:+.3f} … {clim[1]:+.3f} a.u."
        self.statusBar().showMessage(text)
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
            f"PyChem {PYCHEN_VERSION}\n\n"
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
