"""PyChem — application entry point.

Modes:

    uv run python main.py [file]                    # opens the GUI (default)
    uv run python main.py [file] --info             # prints a text summary
    uv run python main.py file --screenshot out.png # headless render
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _summary(path: str) -> None:
    from pychem.io import get_reader

    molecule = get_reader(path)
    print(f"File    : {path}")
    print(f"Name    : {molecule.name}")
    print(f"Formula : {molecule.formula}")
    print(f"Atoms   : {molecule.n_atoms}")
    for atom in molecule.iter_atoms():
        print(
            f"  {atom.index + 1:>3d} {atom.symbol:<2s}"
            f"  {atom.x:9.4f} {atom.y:9.4f} {atom.z:9.4f}"
        )


def _screenshot(path: str, output: str) -> None:
    import pyvista as pv

    from pychem.io import get_reader
    from pychem.rendering import MolecularRenderer, RenderOptions

    molecule = get_reader(path)
    plotter = pv.Plotter(off_screen=True, window_size=(1000, 800))
    renderer = MolecularRenderer(plotter, RenderOptions())
    renderer.show(molecule)
    plotter.reset_camera()
    plotter.screenshot(output)
    print(f"Image saved: {output}")


def _gui(path: str | None) -> int:
    import os

    # Force the "offscreen" platform only when no display is available, so
    # tests can run in headless environments.
    if os.environ.get("DISPLAY", "") == "" and os.environ.get("QT_QPA_PLATFORM", "") == "":
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

    from PySide6.QtWidgets import QApplication

    from pychem.app import MainWindow

    app = QApplication(sys.argv)
    window = MainWindow(filename=path)
    window.show()
    return app.exec()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pychem", description=__doc__)
    parser.add_argument("file", nargs="?", default=None, help="file to open")
    parser.add_argument("--info", action="store_true", help="text summary")
    parser.add_argument(
        "--screenshot", metavar="OUT.png", help="offscreen render to PNG and exit"
    )
    args = parser.parse_args(argv)

    default_file = "guanine.xyz"
    path = args.file or (default_file if Path(default_file).exists() else None)

    if args.screenshot:
        if not path:
            parser.error("--screenshot requires a file")
        _screenshot(path, args.screenshot)
        return 0

    if args.info:
        if not path:
            parser.error("--info requires a file")
        _summary(path)
        return 0

    return _gui(path)


if __name__ == "__main__":
    raise SystemExit(main())


