"""3D rendering of molecules, trajectories and volumes with PyVista.

Renderers are agnostic as to whether the *plotter* is an interactive window
(PySide6) or an *offscreen* render (images without opening a window).
"""

from pychem.rendering.molecular_renderer import MolecularRenderer, RenderOptions

__all__ = ["MolecularRenderer", "RenderOptions"]
