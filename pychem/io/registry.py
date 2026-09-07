"""Registry of file-format readers/writers.

Each format is registered against one or more extensions. Readers always
return core models (``Molecule``, ``Trajectory``...), so adding a new format
does not require touching the viewer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pychem.core.molecule import Molecule


@dataclass(frozen=True)
class Format:
    """Format descriptor with optional read/write functions."""

    extensions: tuple[str, ...]
    description: str = ""
    # Callables; typed with Protocol in later versions if needed.
    read: object | None = None
    write: object | None = None

    def can_read(self) -> bool:
        return self.read is not None

    def can_write(self) -> bool:
        return self.write is not None


class Registry:
    """Maps file extensions to format implementations."""

    def __init__(self) -> None:
        self._formats: dict[str, Format] = {}

    def register(self, fmt: Format) -> None:
        for ext in fmt.extensions:
            ext = ext.lower().lstrip(".")
            if ext in self._formats:
                raise ValueError(f"Extension '{ext}' is already registered")
            self._formats[ext] = fmt

    def get(self, ext: str) -> Format | None:
        return self._formats.get(ext.lower().lstrip("."))

    def supported_extensions(self) -> list[str]:
        return sorted(self._formats)

    def read(self, path: str | Path):
        """Read a file using the reader registered for its extension."""
        p = Path(path)
        fmt = self.get(p.suffix)
        if fmt is None or fmt.read is None:
            raise ValueError(
                f"No reader registered for '{p.suffix}' "
                f"(supported: {self.supported_extensions()})"
            )
        return fmt.read(p)


# Global registry; format modules are imported to register themselves.
_registry = Registry()

from pychem.io import pdb, xyz  # noqa: E402  (register .pdb/.ent and .xyz)

# Future formats auto-register by adding their import here:
# from pychem.io import cube, molden  # noqa: E402


def get_reader(path: str | Path):
    """Read a file with the reader of its extension (single-use format)."""
    return _registry.read(path)


def get_writer(fmt: Format, *args, **kwargs):
    if fmt.write is None:
        raise ValueError(f"Format {fmt} does not support writing")
    return fmt.write(*args, **kwargs)


def supported_extensions() -> list[str]:
    return _registry.supported_extensions()
