"""Resampling a cube onto a target telescope's spaxels at the simulated redshift."""
from __future__ import annotations

import astropy.units as u
from astropy.cosmology import Planck18

from .cosmology import angular_scale_ratio
from .cropping import trim_empty_edges


def pixel_scale_xy(cube):
    """(dx, dy) spaxel size in arcsec, both positive.

    mpdaf's ``wcs.get_step`` returns (dy, dx) in numpy axis order; this helper
    exists so callers can't unpack it the wrong way round.
    """
    dy, dx = cube.wcs.get_step(unit="arcsec")
    return float(abs(dx)), float(abs(dy))


def get_spaxel_area(cube):
    """Spaxel area in arcsec**2 from the cube's WCS."""
    dx_arcsec, dy_arcsec = pixel_scale_xy(cube)
    return dx_arcsec * dy_arcsec


#: Data-header keyword recording that the WCS was shrunk to the simulated redshift.
ANGSCALE_KEY = "CSANGSCL"


def shrink_to_redshift(cube, z_source, z_target, cosmo=Planck18):
    """Copy of ``cube`` whose spatial WCS is at the angular scale of z_target.

    Every spaxel keeps its physical size, so its angular size shrinks by
    f_ang = D_A(z_source) / D_A(z_target).  The data are not changed.  The
    factor is recorded in the data header (``CSANGSCL``); calling this on a cube
    that already has it raises ValueError.
    """
    if ANGSCALE_KEY in cube.data_header:
        raise ValueError(f"cube already shrunk to the simulated redshift ({ANGSCALE_KEY} set)")
    f_ang = angular_scale_ratio(z_source, z_target, cosmo)
    out = cube.copy()
    dy, dx = out.wcs.get_step(unit=u.arcsec)
    out.wcs.set_step([dy * f_ang, dx * f_ang], unit=u.arcsec)
    out.data_header[ANGSCALE_KEY] = (f_ang, "D_A(z_obs)/D_A(z_sim) applied to spatial WCS")
    return out


def resampling_factors(cube, simulated_telescope, z_source=None, z_target=None, cosmo=Planck18):
    """Raw (non-integer) spatial binning factors (s_x, s_y) = p_target / p_cube(z_target).

    If the cube's WCS is still at the source angular scale (no ``CSANGSCL``
    keyword), its pixels are first scaled by D_A(z_source)/D_A(z_target), i.e.
    s = p_target / (p_src * f_ang).  (Equivalently the "Giavalisco" form
    (1+z_s)^2/(1+z_t)^2 * D_L(z_t)/D_L(z_s) * p_target / p_src.)
    """
    dx, dy = pixel_scale_xy(cube)
    if ANGSCALE_KEY not in cube.data_header:
        if z_source is None or z_target is None:
            raise ValueError("cube is at the source angular scale; z_source and z_target are needed")
        f_ang = angular_scale_ratio(z_source, z_target, cosmo)
        dx, dy = dx * f_ang, dy * f_ang
    return simulated_telescope.pixel_scale_x / dx, simulated_telescope.pixel_scale_y / dy


def resample_cube_to_telescope_and_redshift(cube, simulated_telescope, z_source, z_target, trim=True,
                                            noise_scale=1.0, rng=None, cosmo=Planck18,
                                            fwhm_source_arcsec=None, fwhm_target_arcsec=None,
                                            background="auto", phase="center"):
    """Blur to the target PSF and bin to the target spaxels at z_target, in one padded pass.

    Expects surface-brightness data (the binning averages the summed pixels).
    If the cube is still at the source angular scale it is first shrunk with
    :func:`shrink_to_redshift` and its surface brightness converted to per
    z_target arcsec^2, so the output WCS is always at the true angular scale of
    z_target (binned pixel = bin * p_src * f_ang, which differs from the target
    pixel by the integer rounding of the bin factor) and multiplying by
    :func:`get_spaxel_area` gives the flux per binned spaxel.

    The cube is padded once (blur margin, multiple of the bin factor, and the
    target field of view when that is cheap), blurred on the fine grid, then
    binned: see :func:`cubeshift.psf.psf_match_and_bin`.  Nothing is cropped
    except, at the end, data larger than the field of view.

    Parameters
    ----------
    cube : mpdaf.obj.Cube
    simulated_telescope : cubeshift.instruments.Telescope
        Its ``fov_x``/``fov_y`` (if set) define the output field.
    z_source, z_target : float
        Observed and simulated redshift.
    trim : bool
        Remove all-NaN/masked borders of the input before padding.
    noise_scale : float
        Multiplier on the padding noise (the original code used 0.5 for CGCG453).
    rng : numpy.random.Generator, optional
        Random generator for the padding noise; default uses ``np.random``.
    fwhm_source_arcsec, fwhm_target_arcsec : optional
        Source PSF (arcsec at z_source) and target PSF (arcsec, or a function of
        wavelength).  Without them the cube is only binned.
    background : 'auto', 'zero', number or BackgroundEstimate
        Level and noise of the padding; see :func:`cubeshift.padding.estimate_background`.
    phase : 'center' or int
        Where the bin grid falls; see :func:`cubeshift.psf.psf_match_and_bin`.

    Returns
    -------
    cube_resampled : mpdaf.obj.Cube
    bin_factors : (int, int)
        (bin_x, bin_y)
    wave_step : float
        Wavelength step of the cube.
    dx_arcsec_new, dy_arcsec_new : float
        Pixel scale after binning at z_target (arcsec/pix).
    """
    from .psf import psf_match_and_bin

    # make sure the cube is at the angular scale of z_target, in SB per z_target arcsec^2
    if ANGSCALE_KEY not in cube.data_header:
        cube = shrink_to_redshift(cube, z_source, z_target, cosmo)
        f2 = cube.data_header[ANGSCALE_KEY] ** 2
        cube.data = cube.data / f2
        if cube.var is not None:
            cube.var = cube.var / f2 ** 2

    # integer bin factors from the target pixel size
    s_x, s_y = resampling_factors(cube, simulated_telescope)
    bin_factor_x = max(1, int(round(s_x)))
    bin_factor_y = max(1, int(round(s_y)))
    print(f"Raw resampling factors: s_x = {s_x:.3f}, s_y = {s_y:.3f}; bins {bin_factor_x} x {bin_factor_y}")

    # drop empty (all NaN/masked) input borders, then pad once, blur and bin
    if trim:
        cube = trim_empty_edges(cube)
    fov = None
    if simulated_telescope.fov_x is not None and simulated_telescope.fov_y is not None:
        fov = (simulated_telescope.fov_x, simulated_telescope.fov_y)
    cube_resampled = psf_match_and_bin(
        cube, bin_factor_x, bin_factor_y, fwhm_source_arcsec, fwhm_target_arcsec, z_source, z_target,
        background=background, fov=fov, phase=phase, noise_scale=noise_scale, rng=rng, cosmo=cosmo)

    dx_arcsec_new, dy_arcsec_new = pixel_scale_xy(cube_resampled)
    wave_step = cube.wave.get_step() if cube.wave is not None else None
    return cube_resampled, (bin_factor_x, bin_factor_y), wave_step, dx_arcsec_new, dy_arcsec_new
