"""Spatial PSF matching."""
from __future__ import annotations

import numpy as np
from astropy.cosmology import Planck18
from scipy.ndimage import correlate1d

from .cosmology import angular_scale_ratio

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def matching_kernel_fwhm(fwhm_source_arcsec, fwhm_target_arcsec, z_obs, z_sim, cosmo=Planck18,
                         distance_obs_mpc=None):
    """FWHM (arcsec at z_sim) of the Gaussian that turns the source PSF into the target PSF.

    The source PSF (arcsec at z_obs) shrinks by D_A(z_obs)/D_A(z_sim);
    kernel**2 = target**2 - (f_ang * source)**2, and 0 where the shrunk source
    PSF is already broader than the target.

    Parameters
    ----------
    fwhm_source_arcsec : float
        Source PSF FWHM (e.g. seeing) at z_obs.
    fwhm_target_arcsec : float or array
        Target PSF FWHM, e.g. one value per wavelength.
    """
    fwhm_sim = fwhm_source_arcsec * angular_scale_ratio(z_obs, z_sim, cosmo, distance_obs_mpc)
    return np.sqrt(np.maximum(np.asarray(fwhm_target_arcsec, float) ** 2 - fwhm_sim ** 2, 0.0))


def convolve_to_match_psf(cube, fwhm_source_arcsec, fwhm_target_arcsec, z_obs, z_sim, cosmo=Planck18,
                          distance_obs_mpc=None):
    """Convolve each wavelength slice so the PSF matches the target instrument.

    Parameters
    ----------
    cube : mpdaf.obj.Cube
        Its spatial WCS must be at the angular scale of z_sim (as produced by
        :func:`cubeshift.resample.resample_cube_to_telescope_and_redshift`).
    fwhm_source_arcsec : float
        Source PSF FWHM in arcsec at z_obs (e.g. ``source_telescope.spatial_fwhm``).
    fwhm_target_arcsec : float or callable
        Target PSF FWHM in arcsec, or a function of wavelength (Angstrom), e.g.
        ``target_telescope.psf_fwhm``.
    z_obs, z_sim : float
        Original and simulated redshift.

    Returns
    -------
    mpdaf.obj.Cube
        Convolved copy of the cube; the variance is convolved with the squared
        kernel (pixel covariance ignored).
    """
    if callable(fwhm_target_arcsec):
        if cube.wave is None:
            raise ValueError("cube has no wavelength axis; pass the target FWHM as a number")
        target = np.asarray(fwhm_target_arcsec(cube.wave.coord()), float)
    else:
        target = np.full(cube.shape[0], float(fwhm_target_arcsec))
    kernel_fwhm = matching_kernel_fwhm(fwhm_source_arcsec, target, z_obs, z_sim, cosmo, distance_obs_mpc)

    dy_arcsec, dx_arcsec = np.abs(cube.wcs.get_step(unit="arcsec"))
    sigma_x_pix = kernel_fwhm * FWHM_TO_SIGMA / dx_arcsec
    sigma_y_pix = kernel_fwhm * FWHM_TO_SIGMA / dy_arcsec
    print(f"PSF matching: kernel FWHM {kernel_fwhm.min():.4f}-{kernel_fwhm.max():.4f}\" "
          f"(sigma {sigma_x_pix.min():.2f}-{sigma_x_pix.max():.2f} pix in x)")

    convolved_cube = cube.copy()
    if np.all(kernel_fwhm == 0):
        print("Source PSF at z_sim is broader than the target everywhere; no convolution.")
        return convolved_cube
    data = np.ma.getdata(cube.data).astype(float)
    mask = np.ma.getmaskarray(cube.data)
    good = np.isfinite(data) & ~mask
    data = np.where(good, data, 0.0)
    var = None
    if cube.var is not None:
        var = np.where(good, np.ma.getdata(cube.var).astype(float), 0.0)

    def conv(img, ky, kx):
        return correlate1d(correlate1d(img, ky, axis=0, mode="reflect"), kx, axis=1, mode="reflect")

    for i in range(cube.shape[0]):
        if kernel_fwhm[i] > 0:
            ky, kx = gaussian_kernel1d(sigma_y_pix[i]), gaussian_kernel1d(sigma_x_pix[i])
            # normalised convolution: bad pixels get no weight, and stay masked
            w = conv(good[i].astype(float), ky, kx)
            wsafe = np.where(w > 0, w, 1.0)
            data[i] = conv(data[i], ky, kx) / wsafe
            if var is not None:
                # variance of a weighted sum: sum(k**2 v) (pixel covariance ignored)
                var[i] = conv(var[i], ky ** 2, kx ** 2) / wsafe ** 2
    convolved_cube.data = np.ma.array(data, mask=mask | ~good)
    if var is not None:
        convolved_cube.var = var
    return convolved_cube


def gaussian_kernel1d(sigma, truncate=4.0):
    """Normalised, sampled 1D Gaussian (as used by scipy.ndimage.gaussian_filter)."""
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()
