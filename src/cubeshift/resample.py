"""Resampling a cube onto a target telescope's spaxels at the simulated redshift."""
from __future__ import annotations

import astropy.units as u
import numpy as np
from astropy.cosmology import Planck18
from mpdaf.obj import Cube

from .binning import bin_data
from .cosmology import angular_scale_ratio
from .cropping import trim_empty_edges


def get_spaxel_area(cube):
    """Spaxel area in arcsec**2 from the cube's WCS."""
    dx_arcsec, dy_arcsec = cube.wcs.get_step(unit="arcsec")
    return abs(dx_arcsec * dy_arcsec)


def resampling_factors(cube, target_telescope, z_source, z_target, scaling="Giavalisco", cosmo=Planck18):
    """Raw (non-integer) spatial resampling factors (s_x, s_y).

    Parameters
    ----------
    scaling : {'Giavalisco', 'DA'}
        'DA': s = D_A(z_src)/D_A(z_tgt) * p_src / p_target.
        'Giavalisco': s = (1+z_src)**2/(1+z_tgt)**2 * D_L(z_tgt)/D_L(z_src) * p_target / p_src,
        which equals p_target / (p_src * D_A(z_src)/D_A(z_tgt)).
    """
    # KNOWN BUG (review #13): mpdaf get_step returns (dy, dx) but is unpacked as
    # (dx, dy).  Harmless for square spaxels; kept to preserve outputs.
    dx_arcsec_src, dy_arcsec_src = (float(abs(s)) for s in cube.wcs.get_step(unit="arcsec"))

    scaling = scaling.lower()
    if scaling == "da":
        f_ang = angular_scale_ratio(z_source, z_target, cosmo)
        s_x = f_ang * (dx_arcsec_src / target_telescope.pixel_scale_x)
        s_y = f_ang * (dy_arcsec_src / target_telescope.pixel_scale_y)
    elif scaling == "giavalisco":
        l_source = cosmo.luminosity_distance(z_source).to(u.Mpc).value
        l_target = cosmo.luminosity_distance(z_target).to(u.Mpc).value
        k = ((1 + z_source) ** 2 / (1 + z_target) ** 2) * (l_target / l_source)
        s_x = k * (target_telescope.pixel_scale_x / dx_arcsec_src)
        s_y = k * (target_telescope.pixel_scale_y / dy_arcsec_src)
    else:
        raise ValueError(f"Unknown scaling method: {scaling}")
    return s_x, s_y


def resample_cube_to_telescope_and_redshift(cube, target_telescope, z_source, z_target, trim=True,
                                            scaling="Giavalisco", noise_scale=1.0, rng=None,
                                            cosmo=Planck18):
    """Bin a cube to the target telescope's spaxels at z_target, then pad to its FOV.

    Expects surface-brightness data (the binning averages the summed pixels).

    Parameters
    ----------
    cube : mpdaf.obj.Cube
    target_telescope : cubeshift.instruments.Telescope
        Must define ``fov_x``/``fov_y``.
    z_source, z_target : float
        Observed and simulated redshift.
    trim : bool
        Remove all-NaN spatial borders after binning.
    scaling : {'Giavalisco', 'DA'}
        See :func:`resampling_factors`.
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
        Pixel scale after binning, read from the binned WCS (arcsec/pix).
        Known issue (review #3): this is the source-frame scale, not the true
        scale at z_target.
    """
    s_x, s_y = resampling_factors(cube, target_telescope, z_source, z_target, scaling, cosmo)
    print(f"Scaling method: {scaling}; raw resampling factors: s_x = {s_x:.3f}, s_y = {s_y:.3f}")

    bin_factor_x = max(1, int(round(s_x)))
    bin_factor_y = max(1, int(round(s_y)))
    print(f"Integer bin factors: {bin_factor_x}, {bin_factor_y}")

    # KNOWN BUG (review #13): bin_data takes (x, y) but is given (y, x).
    # Harmless for square bins; fixed in Phase 2.
    cube_binned = bin_data(bin_factor_y, bin_factor_x, cube, surface_brightness=True)

    if trim:
        cube_binned = trim_empty_edges(cube_binned)

    ny_binned, nx_binned = cube_binned.shape[1], cube_binned.shape[2]

    dx_arcsec_new, dy_arcsec_new = (float(abs(s)) for s in cube_binned.wcs.get_step(unit="arcsec"))

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
    header as PADX0/PADY0 (0-based offsets) and PADNX/PADNY (size); the variance
    of the padding is zero (no noise model yet, review #8).

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

    padded_data = np.zeros((nz, target_ny, target_nx), dtype=np.float32)
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
            padded_var[k, y0:y0 + ny, x0:x0 + nx] = cube.var[k, :, :]
        if padded_mask is not None:
            padded_mask[k, y0:y0 + ny, x0:x0 + nx] = cube.mask[k, :, :]

    cube_padded = Cube(data=padded_data, var=padded_var, wcs=cube.wcs, wave=cube.wave,
                       mask=padded_mask, copy=False)
    if getattr(cube_padded, "wave", None) is None and getattr(cube, "wave", None) is not None:
        cube_padded.wave = cube.wave.copy()
    hdr = cube_padded.data_header
    hdr["PADX0"], hdr["PADY0"] = (x0, "x offset of unpadded data (0-based)"), (y0, "y offset of unpadded data")
    hdr["PADNX"], hdr["PADNY"] = (nx, "x size of unpadded data"), (ny, "y size of unpadded data")
    return cube_padded
