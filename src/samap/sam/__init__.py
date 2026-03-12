"""Vendored SAM (Self-Assembling Manifold) algorithm.

Originally from the sc-sam package (samalg module). Vendored into SAMap
to eliminate the external dependency and enable targeted optimizations.
"""

from .core import SAM

__all__ = ["SAM"]
