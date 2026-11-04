"""Spectral axis: NIRSpec disperser choice, resolution matching and resampling.

Both operations are linear maps along the wavelength axis, built once as
sparse matrices and applied to every spaxel at once:

* :func:`lsf_matrix` - variable-width Gaussian convolution in the "scatter"
  form (each input pixel spreads its flux with its own LSF width), with
  pixel-integrated (erf) weights.  Flux-conserving; flux that falls off the
  band edges is lost rather than wrapped around.
* :func:`rebin_matrix` - flux-conserving rebinning (piecewise-constant
  spectrum integrated over each output pixel).

Variance is propagated as M**2 @ var (pixel covariance is ignored).
Masked/NaN pixels are excluded; output pixels with less than half their weight
from good input pixels are masked.
"""
from __future__ import annotations

import os

import numpy as np
from astropy.io import fits
from mpdaf.obj import Cube, WaveCoord
from scipy import sparse
from scipy.special import erf

from .config import get_data_dir
from .instruments import NIRSPEC_DISPERSER_RANGES

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def create_wavecoord_from_axis(new_wave_axis):
    """mpdaf WaveCoord (Angstrom) from a linear wavelength array."""
    return WaveCoord(cdelt=new_wave_axis[1] - new_wave_axis[0], crval=new_wave_axis[0], crpix=1,
                     cunit="Angstrom")


def pixel_edges(centers):
    """Pixel edges for a monotonic array of pixel centres."""
    c = np.asarray(centers, float)
    mid = 0.5 * (c[1:] + c[:-1])
    return np.concatenate([[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]])


def crop_cube_to_wavelength_range(cube, lambda_min, lambda_max):
    """Keep only wavelengths within [lambda_min, lambda_max] (cube units), with var/mask/WCS."""
    wave_axis = cube.wave.coord()
    idx = np.flatnonzero((wave_axis >= lambda_min) & (wave_axis <= lambda_max))
    if idx.size == 0:
        raise ValueError("No wavelengths fall within the specified range.")
    return cube[idx[0]:idx[-1] + 1, :, :]


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


# ---------------------------------------------------------------------------
# linear operators along the wavelength axis
# ---------------------------------------------------------------------------
def lsf_matrix(wave, sigma, nsig=6.0):
    """Sparse (n, n) matrix convolving with a Gaussian of width ``sigma[j]`` at input pixel j.

    Column j holds the fraction of input pixel j's flux landing in each output
    pixel (integrated over the output pixel).  ``sigma == 0`` gives the identity.
    """
    wave = np.asarray(wave, float)
    sigma = np.broadcast_to(np.asarray(sigma, float), wave.shape)
    edges = pixel_edges(wave)
    n = wave.size
    rows, cols, vals = [], [], []
    for j in range(n):
        s = sigma[j]
        if not np.isfinite(s) or s <= 0:
            rows.append(j); cols.append(j); vals.append(1.0)
            continue
        lo = np.searchsorted(edges, wave[j] - nsig * s, side="right") - 1
        hi = np.searchsorted(edges, wave[j] + nsig * s, side="left")
        lo, hi = max(lo, 0), min(hi, n)
        cdf = 0.5 * (1 + erf((edges[lo:hi + 1] - wave[j]) / (np.sqrt(2) * s)))
        w = np.diff(cdf)
        rows.extend(range(lo, hi)); cols.extend([j] * (hi - lo)); vals.extend(w)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))


def rebin_matrix(old_wave, new_wave):
    """Sparse (n_new, n_old) flux-conserving rebinning matrix (flux densities in, out).

    Row i holds overlap(new_i, old_j) / width(new_i).  New pixels only partly
    covered by the old grid get row sums < 1.
    """
    oe, ne = pixel_edges(old_wave), pixel_edges(new_wave)
    rows, cols, vals = [], [], []
    for i in range(len(new_wave)):
        lo, hi = ne[i], ne[i + 1]
        j0 = max(np.searchsorted(oe, lo, side="right") - 1, 0)
        j1 = min(np.searchsorted(oe, hi, side="left"), len(old_wave))
        for j in range(j0, j1):
            ov = min(hi, oe[j + 1]) - max(lo, oe[j])
            if ov > 0:
                rows.append(i); cols.append(j); vals.append(ov / (hi - lo))
    return sparse.csr_matrix((vals, (rows, cols)), shape=(len(new_wave), len(old_wave)))


def _apply_spectral_operator(cube, M, new_wave_axis):
    """Apply a wavelength-axis linear map to data and variance; returns a new Cube."""
    nz, ny, nx = cube.shape
    data = np.ma.getdata(cube.data).reshape(nz, -1).astype(float)
    good = np.isfinite(data) & ~np.ma.getmaskarray(cube.data).reshape(nz, -1)
    data = np.where(good, data, 0.0)

    full = np.asarray(M @ np.ones(nz)).ravel()[:, None]
    cover = np.asarray(M @ good.astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(full > 0, cover / full, 0.0)       # share of weight from good pixels
        out = np.asarray(M @ data) / np.where(frac > 0, frac, 1.0)
    out_mask = (frac < 0.5) | (full <= 0)

    var = None
    if cube.var is not None:
        v = np.where(good, np.ma.getdata(cube.var).reshape(nz, -1), 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            var = np.asarray(M.multiply(M) @ v) / np.where(frac > 0, frac, 1.0) ** 2
        var = var.reshape(-1, ny, nx)

    n_new = len(new_wave_axis)
    out_cube = Cube(data=out.reshape(n_new, ny, nx), var=var, mask=out_mask.reshape(n_new, ny, nx),
                    wcs=cube.wcs, wave=create_wavecoord_from_axis(new_wave_axis), copy=False)
    out_cube.primary_header = cube.primary_header.copy()
    out_cube.data_header.extend([c for c in cube.data_header.cards if c.keyword.startswith(("CSANG", "PAD"))])
    return out_cube


def match_spectral_resolution(cube, R_target, R_input=None):
    """Degrade the spectral resolution of a cube to ``R_target``.

    Kernel FWHM(lambda) = sqrt((lambda/R_target)**2 - (lambda/R_input)**2), zero
    where the input is already coarser.  Flux-conserving, no edge wrapping.

    Parameters
    ----------
    cube : mpdaf.obj.Cube
    R_target : float, array (per wavelength bin) or callable(wavelength_angstrom)
    R_input : float, array or callable, optional
        Resolution of the input cube; None means infinitely resolved.
    """
    wave = cube.wave.coord()

    def _r(R):
        return R(wave) if callable(R) else np.broadcast_to(np.asarray(R, float), wave.shape)

    fwhm2 = (wave / _r(R_target)) ** 2
    if R_input is not None:
        fwhm2 = fwhm2 - (wave / _r(R_input)) ** 2
    sigma = np.sqrt(np.maximum(fwhm2, 0.0)) * FWHM_TO_SIGMA
    return _apply_spectral_operator(cube, lsf_matrix(wave, sigma), wave)


def resample_spectral_axis(cube, new_wave_axis):
    """Flux-conserving rebin of every spectrum onto the pixel centres ``new_wave_axis``.

    New pixels not fully inside the original range are dropped.  Variance is
    propagated; masked/NaN input pixels are excluded.
    """
    new_wave_axis = np.asarray(new_wave_axis, float)
    old_edges = pixel_edges(cube.wave.coord())
    new_edges = pixel_edges(new_wave_axis)
    keep = (new_edges[:-1] >= old_edges[0]) & (new_edges[1:] <= old_edges[-1])
    new_wave_axis = new_wave_axis[keep]
    if new_wave_axis.size < 2:
        raise ValueError("New wavelength grid does not overlap the cube.")
    return _apply_spectral_operator(cube, rebin_matrix(cube.wave.coord(), new_wave_axis), new_wave_axis)


def _linear_grid(wave):
    """The table grid if it is linear, else a linear grid with its median step over the same range."""
    step = np.diff(wave)
    if np.allclose(step, np.median(step), rtol=1e-3):
        return wave
    return np.arange(wave[0], wave[-1], np.median(step))


def apply_dispersion_pipeline(redshifted_cube, source_resolution=None, data_dir=None):
    """Observe a cube with the best NIRSpec disperser.

    1. choose the disperser (and crop to it if needed);
    2. on the cube's native wavelength grid, convolve from the source
       resolution to the disperser's R(lambda) (redshifting does not change R);
    3. rebin, flux-conservingly, onto the disperser's wavelength grid.

    Parameters
    ----------
    redshifted_cube : mpdaf.obj.Cube
        Cube already at the simulated redshift (wavelengths in Angstrom).  Its
        sampling should be finer than the disperser's.
    source_resolution : float or callable, optional
        Resolving power of the source instrument (e.g. 4000 for KCWI); None
        treats the input as infinitely resolved.
    data_dir : path-like, optional
        Instrument data directory (see :mod:`cubeshift.config`).

    Returns
    -------
    observed_cube : mpdaf.obj.Cube
    disperser : str
        Name of the disperser used, e.g. 'g235h'.
    """
    dispersion_dir = get_data_dir("dispersion", data_dir)

    wave_um = redshifted_cube.wave.coord() / 1e4
    best_disperser, cube_cropped = select_best_disperser_with_partial(
        redshifted_cube, wave_um.min(), wave_um.max())

    table = load_dispersion_file(best_disperser["name"], dispersion_dir)
    table_wave_aa, table_R = table[:, 0] * 1e4, table[:, 1]

    def R_nirspec(lam_aa):
        return np.interp(lam_aa, table_wave_aa, table_R)

    if np.median(np.diff(cube_cropped.wave.coord())) > np.median(np.diff(table_wave_aa)):
        print("Warning: the cube is more coarsely sampled than the disperser grid.")

    blurred = match_spectral_resolution(cube_cropped, R_nirspec, source_resolution)
    observed = resample_spectral_axis(blurred, _linear_grid(table_wave_aa))
    return observed, best_disperser["name"]
