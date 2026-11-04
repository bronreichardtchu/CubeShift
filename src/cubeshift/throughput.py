"""JWST/NIRSpec filter throughput."""
from __future__ import annotations

import os

import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d

from .config import get_data_dir
from .instruments import NIRSPEC_THROUGHPUT_RANGES


def filter_path(mode, filter_dir):
    """Path of the throughput table for a NIRSpec filter ``mode`` (e.g. 'f170lp')."""
    return os.path.join(filter_dir, f"jwst_nirspec_{mode}_trans.fits")


def load_filter_transmission(fits_path):
    """Read (wavelength [micron], throughput) from a JWST ``*_trans.fits`` table."""
    with fits.open(fits_path) as hdul:
        data = hdul[1].data
        return np.asarray(data["WAVELENGTH"]), np.asarray(data["THROUGHPUT"])


def select_best_nirspec_filter(lambda_obs_micron):
    """NIRSpec filter covering ``lambda_obs_micron`` with the smallest bandwidth, or None."""
    candidates = [(mode, hi - lo) for mode, (lo, hi) in NIRSPEC_THROUGHPUT_RANGES.items()
                  if lo <= lambda_obs_micron <= hi]
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[1])
    return candidates[0][0]


def apply_nirspec_throughput(cube, z, line_rest_wavelength, filter_dir):
    """Multiply a cube by the throughput of the NIRSpec filter covering a redshifted line.

    The cube is modified **in place** and also returned.

    Parameters
    ----------
    cube : mpdaf.obj.Cube
        Wavelengths in Angstrom.
    z : float
        Redshift of the line.
    line_rest_wavelength : float
        Rest wavelength in Angstrom (e.g. 5007 for [OIII]).
    filter_dir : str
        Directory containing ``jwst_nirspec_<mode>_trans.fits``.

    Returns
    -------
    cube : mpdaf.obj.Cube
    cube_waves_um : np.ndarray
        Cube wavelengths in micron.
    throughput : np.ndarray
        Throughput interpolated onto the cube wavelengths.
    filter_filename : str
    mode : str
    """
    lambda_obs = (line_rest_wavelength * (1 + z)) / 1e4
    mode = select_best_nirspec_filter(lambda_obs)
    if mode is None:
        raise ValueError(f"No NIRSpec filter mode covers lambda = {lambda_obs:.3f} um")

    path = filter_path(mode, filter_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Transmission filter file not found: {path}")
    wave_file, trans_file = load_filter_transmission(path)
    print(f"Applying filter {mode} for line at {lambda_obs:.3f} um")

    cube_waves_um = cube.wave.coord() / 1e4
    throughput = interp1d(wave_file, trans_file, bounds_error=False, fill_value=0.0)(cube_waves_um)

    # Known issue (review #10): calibrated JWST cubes should not be multiplied by throughput.
    cube.data *= throughput[:, np.newaxis, np.newaxis]
    return cube, cube_waves_um, throughput, os.path.basename(path), mode


def apply_transmission_filter(cube, z, line_rest_wavelength, data_dir=None):
    """Apply the appropriate NIRSpec filter throughput (in place).

    Parameters
    ----------
    data_dir : path-like, optional
        Instrument data directory (see :mod:`cubeshift.config`).

    Returns
    -------
    filtered_cube : mpdaf.obj.Cube
    filter_name : str
        Short filter name, e.g. 'f170lp'.
    """
    filter_dir = get_data_dir("filters", data_dir)
    filtered_cube, _, _, _, mode = apply_nirspec_throughput(cube, z, line_rest_wavelength, filter_dir)
    return filtered_cube, mode
