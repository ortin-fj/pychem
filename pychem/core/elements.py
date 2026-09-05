"""Basic atomic data used by the core and the rendering.

Light table of elements commonly used in computational chemistry. The
covalent radii are used to guess bonds by default, and the van der Waals
radii / colors drive the visual representation.
"""

from __future__ import annotations

from dataclasses import dataclass

# (atomic number, symbol, mass u, covalent radius Å, vdW radius Å, RGB color)
# Radii in Å (approximate Pyykko/Cordero); simplified CPK-style colors.
_ELEMENTS: dict[str, tuple[int, float, float, float, tuple[int, int, int]]] = {
    "H":  (1,  1.008,  0.31, 1.20, (255, 255, 255)),
    "He": (2,  4.003,  0.28, 1.40, (217, 255, 255)),
    "Li": (3,  6.941,  1.28, 1.82, (204, 128, 255)),
    "Be": (4,  9.012,  0.96, 1.53, (194, 255, 0)),
    "B":  (5,  10.811, 0.84, 1.92, (255, 181, 181)),
    "C":  (6,  12.011, 0.76, 1.70, (144, 144, 144)),
    "N":  (7,  14.007, 0.71, 1.55, (48, 80, 248)),
    "O":  (8,  15.999, 0.66, 1.52, (255, 13, 13)),
    "F":  (9,  18.998, 0.57, 1.47, (144, 224, 80)),
    "Ne": (10, 20.180, 0.58, 1.54, (179, 227, 245)),
    "Na": (11, 22.990, 1.66, 2.27, (171, 92, 242)),
    "Mg": (12, 24.305, 1.41, 1.73, (138, 255, 0)),
    "Al": (13, 26.982, 1.21, 1.84, (191, 166, 166)),
    "Si": (14, 28.086, 1.11, 2.10, (153, 153, 153)),
    "P":  (15, 30.974, 1.07, 1.80, (255, 128, 0)),
    "S":  (16, 32.065, 1.05, 1.80, (255, 200, 50)),
    "Cl": (17, 35.453, 1.02, 1.75, (31, 240, 40)),
    "Ar": (18, 39.948, 1.06, 1.88, (128, 209, 227)),
    "K":  (19, 39.098, 2.03, 2.75, (143, 64, 212)),
    "Ca": (20, 40.078, 1.76, 2.31, (61, 255, 0)),
    "Cr": (24, 51.996, 1.39, 2.07, (138, 153, 199)),
    "Mn": (25, 54.938, 1.39, 2.05, (155, 199, 199)),
    "Fe": (26, 55.845, 1.32, 2.04, (224, 102, 51)),
    "Co": (27, 58.933, 1.26, 2.00, (240, 144, 160)),
    "Ni": (28, 58.693, 1.24, 1.97, (80, 208, 80)),
    "Cu": (29, 63.546, 1.32, 1.96, (200, 128, 51)),
    "Zn": (30, 65.380, 1.22, 2.01, (125, 128, 176)),
    "Br": (35, 79.904, 1.20, 1.85, (166, 41, 41)),
    "Ag": (47, 107.868, 1.45, 2.02, (192, 192, 192)),
    "I":  (53, 126.904, 1.39, 1.98, (148, 0, 148)),
    "W":  (74, 183.840, 1.62, 2.10, (0, 153, 153)),
    "Pt": (78, 195.084, 1.36, 1.75, (128, 128, 144)),
    "Au": (79, 196.967, 1.36, 1.66, (255, 209, 35)),
    "Hg": (80, 200.590, 1.32, 1.55, (184, 184, 208)),
    "Pb": (82, 207.200, 1.46, 2.02, (87, 89, 97)),
}

_DEFAULT = (0, 0.0, 1.0, 2.0, (255, 0, 255))  # unknown element (magenta)


@dataclass(frozen=True)
class Element:
    """Basic properties of a chemical element."""

    symbol: str
    number: int
    mass: float  # u
    covalent_radius: float  # Å
    vdw_radius: float  # Å
    color: tuple[int, int, int]  # RGB 0-255

    @property
    def radius(self) -> float:
        """Van der Waals radius used by default in rendering."""
        return self.vdw_radius


def element(symbol: str) -> Element:
    """Return element data, or a neutral element if the symbol is unknown."""
    data = _ELEMENTS.get(symbol.capitalize(), _DEFAULT)
    return Element(symbol.capitalize(), *data)


def is_known(symbol: str) -> bool:
    return symbol.capitalize() in _ELEMENTS
