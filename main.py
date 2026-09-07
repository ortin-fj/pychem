"""PyChem — legacy root launcher (kept for source checkouts).

The installed console command is ``pychem`` (see ``pyproject.toml``
``[project.scripts]``), implemented in ``pychem/cli.py``. This file lets the
program also be launched from a checkout without installing it:

    uv run python main.py [file]

or, equivalently, from anywhere inside the installed project:

    python -m pychem [file]
"""

from pychem.cli import main

if __name__ == "__main__":
    raise SystemExit(main())


