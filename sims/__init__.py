"""
sims
====

Lightweight simulation utilities.

Public API
----------
- RigidBodySim: Rigid-body simulation and visualization helpers.
"""

from __future__ import annotations

# --- Version ----
try:
    # Python 3.8+: read the installed package version
    from importlib.metadata import version as _pkg_version, PackageNotFoundError  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    _pkg_version = None  # type: ignore[assignment]
    PackageNotFoundError = Exception  # type: ignore[misc]


def _get_version() -> str:
    if _pkg_version is None:
        return "0.0.0"
    try:
        return _pkg_version("sims")
    except PackageNotFoundError:
        # Fallback for editable installs or source-tree usage
        return "0.0.0"


__version__ = _get_version()


# --- Public exports ---
from .rigid_body_sim import RigidBodySim  # re-export for sims.RigidBodySim
from .kalman_filter import LinearKF, LinearGaussianSystemSyms

__all__ = [
    "RigidBodySim",
    "LinearKF",
    "LinearGaussianSystemSyms",
    "__version__",
]


def about() -> str:
    """Return a short, human-readable package blurb with version."""
    return f"sims v{__version__} — simulation utilities (RigidBodySim, etc.)"
