# PyChem

A **modular** Python tool for molecular visualization and analysis, aiming to
grow into a Molden-like program (structures, trajectories and isosurfaces:
orbitals, electron densities, etc.).

**Version 1.0.0.** PyChem is a regular Python package that installs a `pychem`
command, so it can be launched from any terminal as `pychem file.xyz` (see
[Install](#install)).

## Project status

Current phase — Version **1.0.0** (installable
command-line program).

Done:

- `pychem/` package split into `core` / `io` / `rendering` / `app`
  (models independent of file format and viewer).
- `Atom` and `Molecule` models (coordinates in Å, Hill formula, safe copy).
- `Trajectory` model: frames as one array `(n_frames, n_atoms, 3)`, optional
  energies/times, and linear interpolation between frames (`sample()`).
- Basic atomic data (mass, radii, CPK colors) in `pychem/core/elements.py`.
- Covalent-radius bond guessing (`pychem/core/bonds.py`) on a spatial grid:
  only neighbouring cells are compared, so large structures (an 11000-atom
  PDB) are handled in ~0.1 s and a few MB instead of O(n²) time and memory.
- **XYZ** and **extended XYZ** reader/writer with `Lattice=...`; multi-frame
  reading to `Molecule` list or a `Trajectory`.
- **PDB** reader (`pychem/io/pdb.py`): `ATOM`/`HETATM` records become a
  `Molecule` (structure only — multi-model files use the first model).
- Extensible format registry (`pychem/io/registry.py`).
- Reusable **PyVista** molecular renderer (GPU-instanced spheres with
  per-atom CPK colours + bond cylinders): works both in a window and in
  *offscreen* mode. Atoms are drawn by a single `vtkGlyph3DMapper` actor, so
  the geometry never grows with the atom count (an 11000-atom structure stays
  one sphere instead of a ~7.4M-triangle mesh).
- **Atom selection**: left-click an atom (compatible with camera rotation),
  subtle selection halo, click again to deselect, click empty space to clear.
  The atom under the pointer is found analytically (ray against the drawn
  spheres, ~0.2 ms), so no VTK cell picker is involved.
- **Trajectory playback**: open a multi-frame XYZ, play/pause/stop, frame
  slider, speed, loop, and **linear interpolation** with an **Auto mode**
  (FPS watchdog with hysteresis) or manual On/Off.
- **Dynamic bonds**: bonds appear/disappear with the distance between frames,
  using **hysteresis** (break > form thresholds) to avoid flicker; toggleable
  (off = keep the frame-0 connectivity).
- **Volumetric data / isosurfaces (step 5, part 1)**: Gaussian **cube** reader
  (including **multi-block** cubes) and **Molden** reader (s/p basis; orbitals
  evaluated on a grid on demand). When such a file is opened, the structure is
  shown as usual and the side panel offers a **"Plot isosurface"** button;
  plotting shows the field and a list of the orbitals/fields, each selectable
  with a click. Isosurface settings: **absolute isovalue** (auto-suggested on
  first plot, editable), **quality** (Medium/High), and the list of
  fields/orbitals.
- **Measurements**: a side-panel section with three exclusive modes — Bonds,
  Angles and Dihedrals. Pick 2/3/4 atoms (left-click) and the value appears as
  an anchored, always-readable label in the 3D viewer near the measured atoms;
  in Angles/Dihedrals the central atom(s) are highlighted distinctly (magenta)
  from the involved endpoints (gold rings). Switching modes, clicking the same
  mode again or loading a new file clears the measurement and the selection.
- **Atomic orbitals (Molden)**: a **Tools → Visualize atomic orbitals** menu
  entry (enabled only when a `.molden` file is open) swaps the Volumetric panel
  to an atomic-orbital mode: pick an atom (ID · symbol), the angular-momentum
  type (s/p/...) and the specific basis function (e.g. one of several
  contracted s functions on an atom). A first **"Plot AO"** click starts the
  session — the button then disappears and any selection re-plots
  automatically. Once an AO is plotted a **"Plot another AO"** button (below
  Quality) reveals a second, separate picker (a box of its own, without
  duplicating the section title) so a **second AO can be shown at the same
  time**; at most two AOs are allowed and the button is removed as soon as the
  second picker appears. The section title reads **Volumetric: AO** while
  atomic orbitals are shown and **Volumetric: MO** otherwise (this two-AO
  overlay is only for atomic orbitals). Isosurfaces use the same **isovalue**
  (auto-suggested, editable) and **quality** (Medium/High) controls as the
  molecular orbitals.
- **Atom ID labels**: a checkable **Tools → Show ID label** entry overlays the
  plain atom number on every atom (no text box). It stays compatible with atom
  selection, geometric measurements, trajectory playback (labels follow each
  pose) and volumetric/orbital isosurfaces. The numbers use the same green as
  the side-panel titles (`#87D68D`), falling back to **black** on very light
  atoms (e.g. white hydrogen) so they stay readable.
- **Electrostatic-potential (ESP) maps**: a checkable **Tools → Visualize
  Electrostatic Potential** entry, enabled only once a structure is on screen.
  It opens a file dialog that picks the **two cubes of that structure at
  once** — the electron density and the electrostatic potential — and draws a
  **semi-transparent isosurface of the density envelope** (ρ ≈ 1e-3 e/Bohr³)
  coloured with the potential sampled on it, through a diverging **colormap**
  (red = lower potential, electron-rich; blue = higher, electron-poor) and a
  colour bar showing the value range in a.u. The selection order does not
  matter: the two files are told apart by the **second line** of each cube,
  which states what it holds (`Total electron density` /
  `Electrostatic Potential`). Both cubes are written by **`orca_plot`** from
  the same `.gbw` file (`orca_plot job.gbw -i`, electron density and
  electrostatic potential, Gaussian-cube format); `orca_plot` is what records
  that second line, which is why the entry reminds you of it while you pick
  the files (its tooltip carries the full recipe). The ESP map and the
  orbital/AO isosurfaces are alternatives: showing one clears the other.
- **Structure editing (Edit section)**: available while a single structure
  (XYZ/PDB) is open — trajectories and volumetric files are not editable.
  Three exclusive modes, mirroring the Measurements section:
  **Move atoms** (drag an atom with the left button; dragging empty space
  still rotates the camera), **Bonds** (click two atoms to create or delete
  the bond between them, which hands the connectivity to the model instead
  of guessing it) and **Umbrella** (pick a *reference* atom, an *axis* atom
  and any number of *interest* atoms, then drive the group with a slider or
  by dragging one of its atoms). The umbrella motion keeps every
  reference-interest distance constant and preserves the azimuth of each
  interest atom around the reference->axis line: at 90° they are exactly
  coplanar with the reference (an equilateral triangle when they start 120°
  apart, the SN2 transition state), 109.47°/70.53° reproduce the reactant and
  product pyramids and 0°/180° collapse them onto the axis. Leaving a mode
  (clicking its button again) or switching to another one clears its atom
  markers. **Undo** (Ctrl+Z)
  and **Return to initial structure** restore geometry and connectivity, and
  the edited structure is written out with **File → Export structure
  (XYZ)...** (plain XYZ stores the coordinates only, not the bonds).
- **PySide6 GUI** (`pychem/app/`): open a structure, export a PNG image,
  reset the camera, a fixed side panel with atom properties and trajectory
  controls, status bar.
- **Collapsible side panel**: Atom Properties is framed like the other
  sections (Measurements, Trajectory, Volumetric); every section has a **bold
  colored title** (accent `#87D68D`) and is **minimized/expanded** from its
  header (click the title or the window-style button at its right, which
  swaps between a minimize `—` and an expand `▢` glyph). The sections are
  placed in a vertical splitter so each one can be **resized by dragging its
  border**, and they always pack from the top of the panel (collapsed headers
  line up at the top, free space stays at the bottom).
- Installable command-line program: console script `pychem` (entry point
  `pychem.cli:main`, legacy `main.py` launcher and `python -m pychem` kept).
  Modes: GUI (default), *offscreen* render (`--screenshot`) and text summary
  (`--info`). With no file argument the GUI opens empty — use File > Open
  structure... to load a file.
- Tests with pytest (113 tests).

In progress / next:

- More formats (Cube refinements, Molden with d/f shells, analysis of quantum-mechanical outputs...).
- ESP map refinements (user-selectable isovalue/colormap range, slice planes).

## Layout

```text
pychem/
├── core/        # Scientific models (Atom, Molecule, bonds, elements)
├── io/          # File-format readers/writers + registry
├── rendering/   # Reusable PyVista viewer (molecules, ...)
├── app/         # PySide6 interface
└── analysis/    # (future) geometry, trajectories, volumes, NEB...
tests/
```

## Install

Requires **Python ≥ 3.12** and [uv](https://docs.astral.sh/uv/).

### Editable install (development, from the source checkout)

```bash
uv sync                  # create .venv and install the project (editable)
uv run pytest            # run the test suite
uv run pychem file.xyz   # run the command inside the project
```

### Global command (available from any directory)

```bash
uv tool install --editable .   # install the `pychem` command globally
pychem file.xyz                # now usable anywhere
```

(Use `uv tool install .` instead for a regular, non-editable global install;
re-run it after pulling changes.) To uninstall later: `uv tool uninstall pychem`.

## Usage

```bash
pychem                          # open the GUI empty; then File > Open structure... (Ctrl+O)
pychem file.xyz                 # open the GUI with a structure/trajectory/volume
pychem file.xyz --info          # text summary of the molecule
pychem file.xyz --screenshot out.png  # headless render to PNG and exit
```

For developers the same entry point can be run from the checkout without a
global install: `uv run python main.py [file]` or `uv run python -m pychem [file]`.
