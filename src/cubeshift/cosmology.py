"""Cosmological scalings between the observed and simulated redshift.

All functions take a ``cosmo`` argument (an astropy cosmology); the default is
Planck18, which is what the pipeline uses.
"""
from __future__ import annotations

import astropy.units as u
import numpy as np
from astropy.cosmology import Planck18


def flux_dimming_factor(z_obs, z_sim, cosmo=Planck18):
    """(D_L(z_obs)/D_L(z_sim))**2 * (1+z_obs)/(1+z_sim): the f_lambda dimming factor."""
    dl_obs = cosmo.luminosity_distance(z_obs).to("cm").value
    dl_sim = cosmo.luminosity_distance(z_sim).to("cm").value
    return (dl_obs / dl_sim) ** 2 * ((1.0 + z_obs) / (1.0 + z_sim))


def rescale_flux(data, z_obs, z_sim, cosmo=Planck18):
    """Dim flux densities f_lambda from z_obs to z_sim (array in, array out).

    f_lambda -> f_lambda * (D_L(z_obs)/D_L(z_sim))**2 * (1+z_obs)/(1+z_sim).
    Combined with the stretched wavelength step this conserves luminosity, i.e.
    the integrated line flux scales as D_L**-2.  For a Cube use
    ``scale_cube(cube, flux_dimming_factor(z_obs, z_sim))`` so the variance
    is scaled too.
    """
    return data * flux_dimming_factor(z_obs, z_sim, cosmo)


def scale_cube(cube, factor):
    """Multiply a cube's data by ``factor`` and its variance by ``factor**2`` (in place).

    Parameters
    ----------
    cube : mpdaf.obj.Cube
    factor : float or np.ndarray
        Scalar, or one value per wavelength (length nz).

    Returns
    -------
    The same cube.
    """
    f = np.asarray(factor, dtype=float)
    if f.ndim == 1:
        f = f[:, None, None]
    cube.data = cube.data * f
    if cube.var is not None:
        cube.var = cube.var * f ** 2
    return cube


def angular_scale_ratio(z_obs, z_sim, cosmo=Planck18):
    """D_A(z_obs) / D_A(z_sim): factor by which angular sizes shrink."""
    return (cosmo.angular_diameter_distance(z_obs) / cosmo.angular_diameter_distance(z_sim)).value


def angular_diameter_distance_pc(z, cosmo=Planck18):
    """Angular diameter distance at z, in parsec (Quantity)."""
    return cosmo.angular_diameter_distance(z).to(u.pc)


def kpc_per_arcsec(z, cosmo=Planck18):
    """Proper physical scale at z, in kpc/arcsec (Quantity)."""
    return cosmo.kpc_proper_per_arcmin(z).to(u.kpc / u.arcsec)


def arcsec_to_physical(arcsec, scale):
    """Convert an angle in arcsec to a physical length using ``scale`` (kpc/arcsec)."""
    return arcsec * u.arcsec * scale
