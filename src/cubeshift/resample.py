"""Resampling a cube onto a target telescope's spaxels at the simulated redshift."""
from __future__ import annotations

import astropy.units as u
import numpy as np
from astropy.cosmology import Planck18
from mpdaf.obj import Cube

from .binning import bin_data
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


def resampling_factors(cube, target_telescope, z_source=None, z_target=None, cosmo=Planck18):
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
    return target_telescope.pixel_scale_x / dx, target_telescope.pixel_scale_y / dy


def resample_cube_to_telescope_and_redshift(cube, target_telescope, z_source, z_target, trim=True,
                                            noise_scale=1.0, rng=None, cosmo=Planck18):
    """Bin a cube to the target telescope's spaxels at z_target, then pad to its FOV.

    Expects surface-brightness data (the binning averages the summed pixels).
    If the cube is still at the source angular scale it is first shrunk with
    :func:`shrink_to_redshift` and its surface brightness converted to per
    z_target arcsec^2, so the output WCS is always at the true angular scale of
    z_target (binned pixel = bin * p_src * f_ang, which differs from the target
    pixel by the integer rounding of the bin factor) and multiplying by
    :func:`get_spaxel_area` gives the flux per binned spaxel.

    Parameters
    ----------
    cube : mpdaf.obj.Cube
    target_telescope : cubeshift.instruments.Telescope
        Must define ``fov_x``/``fov_y``.
    z_source, z_target : float
        Observed and simulated redshift.
    trim : bool
        Remove all-NaN spatial borders after binning.
    noise_scale : float
        Multiplier on the background standard deviation used for the padding
        noise (the original code used 0.5 for CGCG453).
    rng : numpy.random.Generator, optional
        Random generator for the padding noise; default uses ``np.random``.

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
    if ANGSCALE_KEY not in cube.data_header:
        # surface brightness per source arcsec^2 -> per z_target arcsec^2
        cube = shrink_to_redshift(cube, z_source, z_target, cosmo)
        f2 = cube.data_header[ANGSCALE_KEY] ** 2
        cube.data = cube.data / f2
        if cube.var is not None:
            cube.var = cube.var / f2 ** 2
    s_x, s_y = resampling_factors(cube, target_telescope)
    print(f"Raw resampling factors: s_x = {s_x:.3f}, s_y = {s_y:.3f}")

    bin_factor_x = max(1, int(round(s_x)))
    bin_factor_y = max(1, int(round(s_y)))
    print(f"Integer bin factors: {bin_factor_x}, {bin_factor_y}")

    cube_binned = bin_data(bin_factor_x, bin_factor_y, cube, surface_brightness=True)

    if trim:
        cube_binned = trim_empty_edges(cube_binned)

    ny_binned, nx_binned = cube_binned.shape[1], cube_binned.shape[2]

    dx_arcsec_new, dy_arcsec_new = pixel_scale_xy(cube_binned)

    target_nx = int(np.round(target_telescope.fov_x / target_telescope.pixel_scale_x))
    target_ny = int(np.round(target_telescope.fov_y / target_telescope.pixel_scale_y))

    needs_pad_x = nx_binned < target_nx
    needs_pad_y = ny_binned < target_ny
    if needs_pad_x or needs_pad_y:
        cube_resampled = pad_cube_with_background(
            cube_binned,
            target_nx if needs_pad_x else nx_binned,
            target_ny if needs_pad_y else ny_binned,
            noise_scale=noise_scale,
            rng=rng,
        )
    else:
        cube_resampled = cube_binned

    wave_step = cube.wave.get_step()
    return cube_resampled, (bin_factor_x, bin_factor_y), wave_step, dx_arcsec_new, dy_arcsec_new


def pad_cube_with_background(cube, target_nx, target_ny, noise_scale=1.0, rng=None):
    """Centre a cube in a larger field filled with noisy background.

    Each wavelength slice of the padding is drawn from a normal distribution
    with the slice's background level and ``noise_scale`` x its std.  WCS,
    wavelength axis, variance and mask are carried over.

    The padding is valid (unmasked) simulated background; inside the original
    footprint the input mask is kept.  The footprint is recorded in the data
    header as PADX0/PADY0 (0-based offsets) and PADNX/PADNY (size).  The
    variance of the padding is that of the noise drawn into it,
    (noise_scale * std)**2.

    Parameters
    ----------
    cube : mpdaf.obj.Cube
    target_nx, target_ny : int
        Output spatial size in pixels.
    noise_scale : float
        Multiplier on the background standard deviation.
    rng : numpy.random.Generator, optional
        Default uses the global ``np.random`` state.
    """
    normal = np.random.normal if rng is None else rng.normal
    nz, ny, nx = cube.shape
    x0 = (target_nx - nx) // 2
    y0 = (target_ny - ny) // 2

    padded_data = np.zeros((nz, target_ny, target_nx), dtype=float)
    padded_var = None
    padded_mask = None
    if cube.var is not None:
        padded_var = np.zeros((nz, target_ny, target_nx), dtype=cube.var.dtype)
    if cube.mask is not None:
        padded_mask = np.zeros((nz, target_ny, target_nx), dtype=bool)

    for k in range(nz):
        bkg, bkg_std = cube[k].background()
        padded_data[k] = normal(loc=bkg, scale=bkg_std * noise_scale, size=(target_ny, target_nx))
        padded_data[k, y0:y0 + ny, x0:x0 + nx] = cube[k].data
        if padded_var is not None:
            padded_var[k] = (bkg_std * noise_scale) ** 2
            padded_var[k, y0:y0 + ny, x0:x0 + nx] = cube.var[k, :, :]
        if padded_mask is not None:
            padded_mask[k, y0:y0 + ny, x0:x0 + nx] = cube.mask[k, :, :]

    cube_padded = Cube(data=padded_data, var=padded_var, wcs=cube.wcs, wave=cube.wave,
                       mask=padded_mask, copy=False)
    if getattr(cube_padded, "wave", None) is None and getattr(cube, "wave", None) is not None:
        cube_padded.wave = cube.wave.copy()
    hdr = cube_padded.data_header
    if ANGSCALE_KEY in cube.data_header:
        hdr[ANGSCALE_KEY] = (cube.data_header[ANGSCALE_KEY], cube.data_header.comments[ANGSCALE_KEY])
    hdr["PADX0"], hdr["PADY0"] = (x0, "x offset of unpadded data (0-based)"), (y0, "y offset of unpadded data")
    hdr["PADNX"], hdr["PADNY"] = (nx, "x size of unpadded data"), (ny, "y size of unpadded data")
    return cube_padded
