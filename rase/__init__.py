"""RASE core package.

Submodules intentionally avoid eager imports so the package remains importable
across the isolated SmolVLA, OFT, and RL environments.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
