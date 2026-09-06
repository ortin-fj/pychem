# PyChem

A **modular** Python tool for molecular visualization and analysis, aiming to
grow into a Molden-like program (structures, trajectories and isosurfaces:
orbitals, electron densities, etc.).

## Project status

Current phase —  (step 5/14 of the plan)**.

Done:

- `pychem/` package split into `core` / `io` / `rendering` / `app`
  (models independent of file format and viewer).
- `Atom` and `Molecule` models (coordinates in Å, Hill formula, safe copy).
- `Trajectory` model: frames as one array `(n_frames, n_atoms, 3)`, optional
  energies/times, and linear interpolation between frames (`sample()`).
- Basic atomic data (mass, radii, CPK colors) in `pychem/core/elements.py`.
- Covalent-radius bond guessing (`pychem/core/bonds.py`).
- **XYZ** and **extended XYZ** reader/writer with `Lattice=...`; multi-frame
  reading to `Molecule` list or a `Trajectory`.
- Extensible format registry (`pychem/io/registry.py`).
- Reusable **PyVista** molecular renderer (spheres per element + bond
  cylinders): works both in a window and in *offscreen* mode.
- **Atom selection**: left-click an atom (compatible with camera rotation),
  subtle selection halo, click again to deselect, click empty space to clear.
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
  first plot, editable), **quality** (Low/Medium/High), and the Volumetric
  section is kept to about half of the panel area.
- **Measurements**: a side-panel section with three exclusive modes — Bonds,
  Angles and Dihedrals. Pick 2/3/4 atoms (left-click) and the value appears as
  an anchored, always-readable label in the 3D viewer near the measured atoms;
  in Angles/Dihedrals the central atom(s) are highlighted distinctly (magenta)
  from the involved endpoints (gold rings). Switching modes, clicking the same
  mode again or loading a new file clears the measurement and the selection.
- **PySide6 GUI** (`pychem/app/`): open a structure, export a PNG image,
  reset the camera, a fixed side panel with atom properties and trajectory
  controls, status bar.
- `main.py` with three modes: GUI, *offscreen* render and text summary.
- Tests with pytest (46 tests).

In progress / next:

- More formats (PDB, Cube refinements, Molden with d/f shells...).

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

## Usage

```bash
uv sync                    # install dependencies
uv run pytest              # run the test suite

uv run python main.py                  # open the GUI with guanine.xyz
uv run python main.py file.xyz         # open the GUI with a file
uv run python main.py --info           # text summary of the molecule
uv run python main.py file.xyz --screenshot out.png  # headless PNG
```
