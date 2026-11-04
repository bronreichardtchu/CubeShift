"""Locating external instrument data (JWST filter throughputs, dispersion tables).

These files are not shipped with the package (they come from JDox). Point
CubeShift at a directory containing them, either per call (``data_dir=...``) or
once via the ``CUBESHIFT_DATA_DIR`` environment variable.  Expected layout::

    <data_dir>/
        jwst_nirspec_<filter>_trans.fits   (e.g. jwst_nirspec_f170lp_trans.fits)
        jwst_nirspec_<grating>_disp.fits   (e.g. jwst_nirspec_g235h_disp.fits)

Sub-directories are also accepted: ``<data_dir>/Transmission Filters`` and
``<data_dir>/Dispersion Filters`` (the original layout) are searched as well.
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "CUBESHIFT_DATA_DIR"

_SUBDIRS = {
    "filters": ("Transmission Filters", "filters"),
    "dispersion": ("Dispersion Filters", "dispersion"),
}


def get_data_dir(kind: str, data_dir: str | os.PathLike | None = None) -> Path:
    """Return the directory holding instrument files of ``kind``.

    Parameters
    ----------
    kind : {'filters', 'dispersion'}
        Which set of instrument files is needed.
    data_dir : path-like, optional
        Explicit directory.  Overrides the ``CUBESHIFT_DATA_DIR`` variable.

    Raises
    ------
    FileNotFoundError
        If no directory is configured or it does not exist.
    """
    if kind not in _SUBDIRS:
        raise ValueError(f"kind must be one of {list(_SUBDIRS)}, not {kind!r}")
    root = data_dir if data_dir is not None else os.environ.get(ENV_VAR)
    if root is None:
        raise FileNotFoundError(
            f"No instrument data directory given. Pass data_dir=... or set ${ENV_VAR} "
            "to the folder holding the JWST jwst_nirspec_*_trans.fits / *_disp.fits files."
        )
    root = Path(root).expanduser()
    for sub in _SUBDIRS[kind]:
        if (root / sub).is_dir():
            return root / sub
    if root.is_dir():
        return root
    raise FileNotFoundError(f"Instrument data directory does not exist: {root}")
