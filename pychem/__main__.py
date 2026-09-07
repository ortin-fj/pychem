"""Allow running the package as ``python -m pychem [file]``."""

from pychem.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
