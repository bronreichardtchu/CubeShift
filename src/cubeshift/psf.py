"""Spatial PSF matching."""
from __future__ import annotations

import astropy.units as u
import numpy as np
from astropy.cosmology import Planck18
from scipy.ndimage import gaussian_filter

from .cosmology import angular_scale_ratio


def convolve_to_match_psf(cube, fwhm_real_x_arcsec, fwhm_real_y_arcsec, fwhm_target_arcsec,
                          z_obs, z_sim, cosmo=Planck18):
    """Convolve each wavelength slice with a Gaussian so the PSF matches a target.

    The source PSF is first shrunk by D_A(z_obs)/D_A(z_sim); if it is then still
    broader than the target nothing is done.  Otherwise the kernel FWHM is
    sqrt(target**2 - sim**2), converted to pixels with the cube's WCS step.

    Parameters
    ----------
    cube : mpdaf.obj.Cube
    fwhm_real_x_arcsec, fwhm_real_y_arcsec : float
        Source PSF FWHM along x and y, in arcsec at z_obs.
    fwhm_target_arcsec : float
        Target instrument PSF FWHM in arcsec.
    z_obs, z_sim : float
        Original and simulated redshift.

    Returns
    -------
    mpdaf.obj.Cube
        Convolved copy of the cube.

    Notes
    -----
    Known issue (review #4): the pipeline currently passes the *binned pixel
    scale* as the source FWHM, and the kernel is converted to pixels with a WCS
    that is not at the z_sim angular scale.
    """
    ratio = angular_scale_ratio(z_obs, z_sim, cosmo)
    fwhm_sim_x = fwhm_real_x_arcsec * ratio
    fwhm_sim_y = fwhm_real_y_arcsec * ratio

    if fwhm_target_arcsec <= max(fwhm_sim_x, fwhm_sim_y):
        print("Target PSF is finer than simulated real PSF — skipping convolution.")
        return cube.copy()

    fwhm_kernel_x = np.sqrt(max(fwhm_target_arcsec ** 2 - fwhm_sim_x ** 2, 0))
    fwhm_kernel_y = np.sqrt(max(fwhm_target_arcsec ** 2 - fwhm_sim_y ** 2, 0))

    sigma_x_arcsec = fwhm_kernel_x / 2.355
    sigma_y_arcsec = fwhm_kernel_y / 2.355

    # mpdaf returns (dy, dx), and dx is negative for RA
    dy_arcsec, dx_arcsec = np.abs(cube.wcs.get_axis_increments(unit=u.arcsec))
    sigma_x_pix = sigma_x_arcsec / dx_arcsec
    sigma_y_pix = sigma_y_arcsec / dy_arcsec

    print(f"PSF sim (x, y) arcsec: ({fwhm_sim_x:.4f}, {fwhm_sim_y:.4f})")
    print(f"Matching kernel FWHM (x, y) arcsec: ({fwhm_kernel_x:.4f}, {fwhm_kernel_y:.4f}) "
          f"-> sigma pixels: ({sigma_x_pix:.2f}, {sigma_y_pix:.2f})")

    convolved_cube = cube.copy()
    for i in range(cube.shape[0]):
        convolved_cube.data[i] = gaussian_filter(cube.data[i], sigma=(sigma_y_pix, sigma_x_pix))

    return convolved_cube
