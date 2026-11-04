"""Spatial PSF matching."""
from __future__ import annotations

import numpy as np
from astropy.cosmology import Planck18
from scipy.ndimage import gaussian_filter

from .cosmology import angular_scale_ratio

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def matching_kernel_fwhm(fwhm_source_arcsec, fwhm_target_arcsec, z_obs, z_sim, cosmo=Planck18):
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
    fwhm_sim = fwhm_source_arcsec * angular_scale_ratio(z_obs, z_sim, cosmo)
    return np.sqrt(np.maximum(np.asarray(fwhm_target_arcsec, float) ** 2 - fwhm_sim ** 2, 0.0))


def convolve_to_match_psf(cube, fwhm_source_arcsec, fwhm_target_arcsec, z_obs, z_sim, cosmo=Planck18):
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
        Convolved copy of the cube (variance is not convolved yet).
    """
    wave = cube.wave.coord()
    target = fwhm_target_arcsec(wave) if callable(fwhm_target_arcsec) else np.full(wave.shape, fwhm_target_arcsec)
    kernel_fwhm = matching_kernel_fwhm(fwhm_source_arcsec, target, z_obs, z_sim, cosmo)

    dy_arcsec, dx_arcsec = np.abs(cube.wcs.get_step(unit="arcsec"))
    sigma_x_pix = kernel_fwhm * FWHM_TO_SIGMA / dx_arcsec
    sigma_y_pix = kernel_fwhm * FWHM_TO_SIGMA / dy_arcsec
    print(f"PSF matching: kernel FWHM {kernel_fwhm.min():.4f}-{kernel_fwhm.max():.4f}\" "
          f"(sigma {sigma_x_pix.min():.2f}-{sigma_x_pix.max():.2f} pix in x)")

    convolved_cube = cube.copy()
    if np.all(kernel_fwhm == 0):
        print("Source PSF at z_sim is broader than the target everywhere; no convolution.")
        return convolved_cube
    for i in range(cube.shape[0]):
        if kernel_fwhm[i] > 0:
            convolved_cube.data[i] = gaussian_filter(np.ma.getdata(cube.data[i]),
                                                     sigma=(sigma_y_pix[i], sigma_x_pix[i]))
    return convolved_cube
