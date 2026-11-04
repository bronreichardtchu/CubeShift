"""Spectral axis: NIRSpec disperser choice, resampling and resolution matching."""
from __future__ import annotations

import os

import numpy as np
import numpy.ma as ma
from astropy.io import fits
from mpdaf.obj import Cube, WaveCoord
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

from .config import get_data_dir
from .instruments import NIRSPEC_DISPERSER_RANGES


def create_wavecoord_from_axis(new_wave_axis):
    """mpdaf WaveCoord (Angstrom) from a linear wavelength array."""
    return WaveCoord(cdelt=new_wave_axis[1] - new_wave_axis[0], crval=new_wave_axis[0], crpix=1,
                     cunit="Angstrom")


def crop_cube_to_wavelength_range(cube, lambda_min, lambda_max):
    """Keep only wavelengths within [lambda_min, lambda_max] (cube units).

    Note: variance and mask are not carried over.
    """
    wave_axis = cube.wave.coord()
    keep = (wave_axis >= lambda_min) & (wave_axis <= lambda_max)
    if np.sum(keep) == 0:
        raise ValueError("No wavelengths fall within the specified range.")
    return Cube(data=cube.data[keep, :, :], wave=create_wavecoord_from_axis(wave_axis[keep]), wcs=cube.wcs)


def select_best_disperser_with_partial(cube, lam_obs_min, lam_obs_max, crop_cube_if_needed=True):
    """Choose a NIRSpec disperser for the observed range [lam_obs_min, lam_obs_max] (micron).

    Walks :data:`~cubeshift.instruments.NIRSPEC_DISPERSER_RANGES` in order.  The
    first disperser covering >50% of the range (but not all of it) is returned
    with the cube cropped to the overlap.  Otherwise the narrowest disperser with
    full coverage is used, falling back to the PRISM.

    Returns
    -------
    (dict, mpdaf.obj.Cube)
        Disperser entry and the (possibly cropped) cube.
    """
    full_matches = []
    for combo in NIRSPEC_DISPERSER_RANGES:
        lambda_min, lambda_max = combo["lambda_min"], combo["lambda_max"]

        if lam_obs_min >= lambda_min and lam_obs_max <= lambda_max:
            full_matches.append((combo, lambda_max - lambda_min))
            continue

        overlap = max(0, min(lam_obs_max, lambda_max) - max(lam_obs_min, lambda_min))
        coverage_fraction = overlap / (lam_obs_max - lam_obs_min)

        # Known issue (review #11): returns the *first* >50% match, not the best one.
        if coverage_fraction > 0.5 and crop_cube_if_needed:
            lam_overlap_min = max(lambda_min, lam_obs_min)
            lam_overlap_max = min(lambda_max, lam_obs_max)
            if lam_overlap_max > lam_overlap_min:
                print(f"Cropping cube to {combo['name']} range {lam_overlap_min:.3f}-{lam_overlap_max:.3f} um")
                return combo, crop_cube_to_wavelength_range(cube, lam_overlap_min * 1e4, lam_overlap_max * 1e4)

    if full_matches:
        return sorted(full_matches, key=lambda m: m[1])[0][0], cube

    for combo in NIRSPEC_DISPERSER_RANGES:
        if combo["name"].lower() == "prism":
            print("No good match found. Falling back to PRISM.")
            return combo, cube

    raise ValueError("No suitable disperser found.")


def load_dispersion_file(filter_name, dispersion_dir):
    """Load a JWST NIRSpec dispersion table.

    Parameters
    ----------
    filter_name : str
        Disperser name, e.g. 'g235h'.
    dispersion_dir : str
        Folder with ``jwst_nirspec_<name>_disp.fits``.

    Returns
    -------
    np.ndarray
        (N, 2) array: wavelength [micron], R.
    """
    file_path = os.path.join(dispersion_dir, f"jwst_nirspec_{filter_name.lower()}_disp.fits")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Dispersion FITS file not found: {file_path}")
    with fits.open(file_path) as hdul:
        data = hdul[1].data
        return np.column_stack([data["WAVELENGTH"], data["R"]])


def to_nan_array(arr):
    """Masked array -> float array with NaN where masked."""
    if isinstance(arr, ma.MaskedArray):
        return arr.filled(np.nan)
    return arr


def fill_nans_spectrum(spectrum):
    """Linearly interpolate over NaNs in a 1D spectrum (zeros if <2 good points)."""
    idx = np.arange(len(spectrum))
    good = ~np.isnan(spectrum)
    if good.sum() < 2:
        return np.nan_to_num(spectrum)
    return interp1d(idx[good], spectrum[good], bounds_error=False, fill_value="extrapolate")(idx)


def precomputed_match_spectral_resolution_variable_kernel(cube, R_input_interpolated, R_target):
    """Degrade spectral resolution with a wavelength-dependent Gaussian kernel.

    Kernel FWHM(lambda) = sqrt((lambda/R_target)**2 - (lambda/R_input(lambda))**2),
    zero where the input is already coarser.  A kernel bank is precomputed and
    rolled to each wavelength.

    Parameters
    ----------
    cube : mpdaf.obj.Cube
        Linear wavelength grid assumed.
    R_input_interpolated : np.ndarray
        Input resolving power per wavelength bin.
    R_target : float
        Target resolving power.

    Notes
    -----
    Known issues (review #5): the pipeline passes the NIRSpec R as the *input*
    (the source LSF is ignored), and ``np.roll`` wraps kernels around the band edges.
    """
    new_cube = cube.copy()
    wave = cube.wave.coord()
    delta_lambda = wave[1] - wave[0]
    n_wave = cube.shape[0]

    fwhm_input = wave / R_input_interpolated
    fwhm_target = wave / R_target
    fwhm_kernel = np.sqrt(np.maximum(fwhm_target ** 2 - fwhm_input ** 2, 0))
    sigma_kernel_pix = fwhm_kernel / delta_lambda / 2.355

    kernel_bank = []
    for sigma in sigma_kernel_pix:
        delta = np.zeros(n_wave)
        delta[n_wave // 2] = 1.0
        if sigma == 0 or np.isnan(sigma):
            kernel_bank.append(delta)
        else:
            kernel_bank.append(gaussian_filter1d(delta, sigma=sigma, mode="constant"))

    for y in range(cube.shape[1]):
        for x in range(cube.shape[2]):
            spectrum = fill_nans_spectrum(to_nan_array(cube.data[:, y, x]))
            smoothed = np.zeros_like(spectrum)
            for i in range(n_wave):
                kernel = np.roll(kernel_bank[i], i - n_wave // 2)
                smoothed[i] = np.nansum(spectrum * kernel)
            new_cube.data[:, y, x] = smoothed

    return new_cube


def resample_spectral_axis(cube, new_wave_axis):
    """Linearly interpolate every spectrum onto ``new_wave_axis``.

    Points outside the original range are dropped.  Variance and mask are not
    carried over.

    Notes
    -----
    Known issue (review #6): point sampling, not flux-conserving.
    """
    ny, nx = cube.shape[1], cube.shape[2]
    new_data = np.zeros((len(new_wave_axis), ny, nx))
    old_wave = cube.wave.coord()

    for y in range(ny):
        for x in range(nx):
            new_data[:, y, x] = interp1d(old_wave, cube.data[:, y, x], bounds_error=False,
                                         fill_value=0)(new_wave_axis)

    keep = (new_wave_axis >= old_wave[0]) & (new_wave_axis <= old_wave[-1])
    return Cube(data=new_data[keep, :, :], wave=create_wavecoord_from_axis(new_wave_axis[keep]),
                wcs=cube.wcs)


def apply_dispersion_pipeline(redshifted_cube, data_dir=None):
    """Choose a NIRSpec disperser, resample onto its grid and match its resolution.

    Parameters
    ----------
    redshifted_cube : mpdaf.obj.Cube
        Cube already at the simulated redshift (wavelengths in Angstrom).
    data_dir : path-like, optional
        Instrument data directory (see :mod:`cubeshift.config`).

    Returns
    -------
    blurred_cube : mpdaf.obj.Cube
    disperser : str
        Name of the disperser used, e.g. 'g235h'.
    """
    dispersion_dir = get_data_dir("dispersion", data_dir)

    wave_um = redshifted_cube.wave.coord() / 1e4
    best_disperser, cube_cropped = select_best_disperser_with_partial(
        redshifted_cube, wave_um.min(), wave_um.max())

    dispersion_data = load_dispersion_file(best_disperser["name"], dispersion_dir)

    resampled_cube = resample_spectral_axis(cube_cropped, dispersion_data[:, 0] * 1e4)

    wave_um = resampled_cube.wave.coord() / 1e4
    R_input = np.clip(np.interp(wave_um, dispersion_data[:, 0], dispersion_data[:, 1]), 10, 10000)
    target_R = np.median(dispersion_data[:, 1])

    blurred_cube = precomputed_match_spectral_resolution_variable_kernel(resampled_cube, R_input, target_R)
    return blurred_cube, best_disperser["name"]
