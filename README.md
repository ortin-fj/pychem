# PyChem

A **modular** Python tool for molecular visualization and analysis, aiming to
grow into a Molden-like program (structures, trajectories and isosurfaces:
orbitals, electron densities, etc.).

## Project status

Current phase — **trajectories, part 1 (step 4/14 of the plan)**.

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
- **PySide6 GUI** (`pychem/app/`): open a structure, export a PNG image,
  reset the camera, a fixed side panel with atom properties and trajectory
  controls, status bar.
- `main.py` with three modes: GUI, *offscreen* render and text summary.
- Tests with pytest (38 tests).

In progress / next:

- Dynamic bonds (appear/disappear by distance) with hysteresis.
- More formats (PDB, Cube, Molden...).

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
