"""Cosmological scalings between the observed and simulated redshift.

All functions take a ``cosmo`` argument (an astropy cosmology); the default is
Planck18, which is what the pipeline uses.

For very nearby galaxies the redshift is a poor distance indicator (peculiar
velocities).  Functions that need the source distance accept
``distance_obs_mpc``: the luminosity distance of the observed galaxy in Mpc
(for such galaxies effectively the measured distance).  It replaces
D_L(z_obs), and D_A(z_obs) becomes distance_obs_mpc / (1 + z_obs)**2; z_obs
still sets the (1 + z) factors.
"""
from __future__ import annotations

import astropy.units as u
import numpy as np
from astropy.cosmology import Planck18

SEDS = ("flat_flambda", "flat_fnu")


def luminosity_distance_obs_mpc(z_obs, distance_obs_mpc=None, cosmo=Planck18):
    """D_L of the observed galaxy in Mpc: ``distance_obs_mpc`` if given, else D_L(z_obs)."""
    if distance_obs_mpc is not None:
        return float(distance_obs_mpc)
    return cosmo.luminosity_distance(z_obs).to(u.Mpc).value


def angular_diameter_distance_obs_mpc(z_obs, distance_obs_mpc=None, cosmo=Planck18):
    """D_A of the observed galaxy in Mpc: distance_obs_mpc/(1+z_obs)^2 if given, else D_A(z_obs)."""
    if distance_obs_mpc is not None:
        return float(distance_obs_mpc) / (1.0 + z_obs) ** 2
    return cosmo.angular_diameter_distance(z_obs).to(u.Mpc).value


def flux_dimming_factor(z_obs, z_sim, cosmo=Planck18, distance_obs_mpc=None, sed="flat_flambda"):
    """Factor by which flux densities dim when the galaxy is moved from z_obs to z_sim.

    Parameters
    ----------
    sed : {'flat_flambda', 'flat_fnu'}
        'flat_flambda' (default): (D_L,obs/D_L,sim)^2 (1+z_obs)/(1+z_sim).
        This is exact for f_lambda at the *redshifted* wavelength, i.e. for a cube
        whose wavelength axis has been stretched.  For a broadband image in a
        fixed filter it corresponds to a flat f_lambda spectrum (no K-correction).
        'flat_fnu': (D_L,obs/D_L,sim)^2 (1+z_sim)/(1+z_obs), for a fixed filter and
        a flat f_nu spectrum (constant AB colour; no K-correction).
    distance_obs_mpc : float, optional
        Luminosity distance of the observed galaxy (see module docstring).
    """
    if sed not in SEDS:
        raise ValueError(f"sed must be one of {SEDS}, not {sed!r}")
    if distance_obs_mpc is None:
        dl_obs = cosmo.luminosity_distance(z_obs).to("cm").value
    else:
        dl_obs = (distance_obs_mpc * u.Mpc).to("cm").value
    dl_sim = cosmo.luminosity_distance(z_sim).to("cm").value
    if sed == "flat_fnu":
        return (dl_obs / dl_sim) ** 2 * ((1.0 + z_sim) / (1.0 + z_obs))
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


def angular_scale_ratio(z_obs, z_sim, cosmo=Planck18, distance_obs_mpc=None):
    """D_A(z_obs) / D_A(z_sim): factor by which angular sizes shrink.

    With ``distance_obs_mpc``, D_A(z_obs) = distance_obs_mpc / (1 + z_obs)**2.
    """
    if distance_obs_mpc is None:
        return (cosmo.angular_diameter_distance(z_obs) / cosmo.angular_diameter_distance(z_sim)).value
    da_obs = angular_diameter_distance_obs_mpc(z_obs, distance_obs_mpc, cosmo)
    return da_obs / cosmo.angular_diameter_distance(z_sim).to(u.Mpc).value


def angular_diameter_distance_pc(z, cosmo=Planck18):
    """Angular diameter distance at z, in parsec (Quantity)."""
    return cosmo.angular_diameter_distance(z).to(u.pc)


def kpc_per_arcsec(z, cosmo=Planck18):
    """Proper physical scale at z, in kpc/arcsec (Quantity)."""
    return cosmo.kpc_proper_per_arcmin(z).to(u.kpc / u.arcsec)


def arcsec_to_physical(arcsec, scale):
    """Convert an angle in arcsec to a physical length using ``scale`` (kpc/arcsec)."""
    return arcsec * u.arcsec * scale
