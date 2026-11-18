"""Broadband imaging: re-observe an image at another redshift with the same filter.

The image is wrapped as a one-slice cube (wavelength = the filter's pivot
wavelength) so it goes through the same spatial steps as the IFU pipeline:

1. dim the flux (same filter, no K-correction: flat f_nu by default);
2. shrink the angular size to z_sim, convert to surface brightness;
3. bin to the target pixels at z_sim (optionally pad to the target FOV);
4. convert back to flux per pixel;
5. convolve from the source PSF (shrunk to z_sim) to the target PSF.

Pixel values stay in the input units (e.g. electrons/s for HST), which is
consistent because the filter does not change.  No noise is added (see
ROADMAP.md, #8): the original noise is dimmed with the signal.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import astropy.units as u
from astropy.io import fits
import numpy as np
from astropy.cosmology import Planck18
from mpdaf.obj import Cube, Image, WaveCoord

from .cosmology import angular_scale_ratio, flux_dimming_factor, scale_cube
from .io import read_hst_image
from .pipeline import _as_telescope
from .psf import convolve_to_match_psf
from .resample import get_spaxel_area, pixel_scale_xy, resample_cube_to_telescope_and_redshift, shrink_to_redshift

#: data_header keywords added to simulated images
SIM_KEYS = ("CSZOBS", "CSZSIM", "CSDIST", "CSSED", "CSDIM", "CSANGSCL", "CSBINX", "CSBINY")


@dataclass
class ImageSimulationResult:
    """Output of :func:`simulate_image_observation`."""

    image: Image
    """Simulated image (input units, WCS at the angular scale of z_sim)."""
    bin_factors: tuple
    """Integer binning (x, y) applied to the input pixels."""
    pixel_scale: tuple
    """(dx, dy) of the output pixels in arcsec at z_sim."""
    angular_scale_ratio: float
    """D_A(z_obs)/D_A(z_sim)."""
    dimming_factor: float
    """Factor applied to the flux density."""
    path: str | None = None


def _pivot_wavelength(image, pivot_wavelength):
    if pivot_wavelength is not None:
        return float(pivot_wavelength)
    for key in ("PHOTPLAM", "PIVOT"):
        if key in image.data_header:
            return float(image.data_header[key])
    raise ValueError("No PHOTPLAM in the header; pass pivot_wavelength (Angstrom) for the PSF")


def _image_to_cube(image, pivot):
    data = np.ma.getdata(image.data).astype(float)[None]
    var = None if image.var is None else np.ma.getdata(image.var).astype(float)[None]
    cube = Cube(data=np.ma.array(data, mask=np.ma.getmaskarray(image.data)[None]), var=var,
                wcs=image.wcs.copy(), wave=WaveCoord(crval=pivot, cdelt=1.0, crpix=1, cunit=u.AA, shape=1))
    cube.primary_header = image.primary_header.copy()
    return cube


def simulate_image_observation(image, z_obs, z_sim, source_telescope, target_telescope, *,
                               distance_obs_mpc=None, sed="flat_fnu", pivot_wavelength=None,
                               pad_to_fov=False, noise_scale=1.0, rng=None, output_dir=None,
                               galaxy_name=None, cosmo=Planck18):
    """Simulate how a broadband image would look at z_sim with the same filter.

    Parameters
    ----------
    image : mpdaf.obj.Image or str
        Input image, or a path to an HST drizzled file (read with
        :func:`cubeshift.io.read_hst_image`).
    z_obs, z_sim : float
        Observed and simulated redshift.
    source_telescope, target_telescope : Telescope or str
        E.g. ``"HST_ACS_WFC"`` for both.  Only ``spatial_fwhm``/``psf_fwhm`` and
        the pixel scale are used.
    distance_obs_mpc : float, optional
        Measured distance of the observed galaxy in Mpc (recommended for nearby
        galaxies, whose redshift is dominated by peculiar velocity).
    sed : {'flat_fnu', 'flat_flambda'}
        Spectral shape assumed for the dimming, since the filter is not shifted
        (no K-correction).  Flat f_nu: surface brightness dims as (1+z)^-3.
    pivot_wavelength : float, optional
        Filter pivot wavelength in Angstrom (default: PHOTPLAM); sets the target
        PSF if it is wavelength dependent.
    pad_to_fov : bool
        Pad to the target field of view with background noise.  Default False.
    output_dir, galaxy_name : str, optional
        If ``output_dir`` is given the image is written to
        ``<output_dir>/<galaxy_name or 'image'>_z_<z_sim>.fits``.

    Returns
    -------
    ImageSimulationResult
    """
    source_telescope = _as_telescope(source_telescope)
    target_telescope = _as_telescope(target_telescope)
    if not isinstance(image, Image):
        image = read_hst_image(image)
    pivot = _pivot_wavelength(image, pivot_wavelength)

    cube = _image_to_cube(image, pivot)
    dim = flux_dimming_factor(z_obs, z_sim, cosmo=cosmo, distance_obs_mpc=distance_obs_mpc, sed=sed)
    scale_cube(cube, dim)
    cube = shrink_to_redshift(cube, z_obs, z_sim, cosmo=cosmo, distance_obs_mpc=distance_obs_mpc)
    scale_cube(cube, 1.0 / get_spaxel_area(cube))

    binned, bin_factors, _, _, _ = resample_cube_to_telescope_and_redshift(
        cube, target_telescope, z_obs, z_sim, noise_scale=noise_scale, rng=rng, cosmo=cosmo,
        pad_to_fov=pad_to_fov)
    scale_cube(binned, get_spaxel_area(binned))

    # (mpdaf drops a length-1 wavelength axis when slicing, so evaluate the PSF at the pivot here)
    target_fwhm = float(np.asarray(target_telescope.psf_fwhm(pivot)))
    convolved = convolve_to_match_psf(binned, source_telescope.spatial_fwhm, target_fwhm,
                                      z_obs, z_sim, cosmo=cosmo, distance_obs_mpc=distance_obs_mpc)

    out = convolved[0, :, :]
    out.primary_header = image.primary_header.copy()
    hdr = out.data_header
    for card in image.data_header.cards:
        if card.keyword not in hdr and not card.keyword.startswith(("CD", "CRPIX", "CRVAL", "CTYPE",
                                                                    "CUNIT", "CDELT", "PC", "NAXIS",
                                                                    "WCSAXES", "LTV", "LTM")):
            hdr.append(card)
    f_ang = angular_scale_ratio(z_obs, z_sim, cosmo, distance_obs_mpc)
    hdr["CSZOBS"] = (z_obs, "CubeShift: observed redshift")
    hdr["CSZSIM"] = (z_sim, "CubeShift: simulated redshift")
    hdr["CSDIST"] = (distance_obs_mpc if distance_obs_mpc is not None else "z_obs",
                     "CubeShift: distance of observed galaxy [Mpc]")
    hdr["CSSED"] = (sed, "CubeShift: SED assumed for dimming (no K-corr)")
    hdr["CSDIM"] = (dim, "CubeShift: flux dimming factor applied")
    hdr["CSANGSCL"] = (f_ang, "D_A(z_obs)/D_A(z_sim) applied to spatial WCS")
    hdr["CSBINX"], hdr["CSBINY"] = bin_factors
    for key in ("PADX0", "PADY0", "PADNX", "PADNY"):
        if key in binned.data_header:
            hdr[key] = binned.data_header[key]

    path = None
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{galaxy_name or 'image'}_z_{z_sim}.fits")
        write_image(out, path)

    return ImageSimulationResult(image=out, bin_factors=bin_factors, pixel_scale=pixel_scale_xy(out),
                                 angular_scale_ratio=f_ang, dimming_factor=dim, path=path)


def write_image(image, path):
    """Write an mpdaf Image, restoring BUNIT from CSBUNIT (mpdaf blanks units it can't parse)."""
    image.write(path)
    unit = image.data_header.get("CSBUNIT")
    if unit:
        with fits.open(path, mode="update") as hdul:
            for hdu in hdul[1:]:
                if hdu.header.get("EXTNAME", "DATA") == "DATA":
                    hdu.header["BUNIT"] = unit
