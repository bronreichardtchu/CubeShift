"""Cosmological scalings between the observed and simulated redshift.

All functions take a ``cosmo`` argument (an astropy cosmology); the default is
Planck18, which is what the pipeline uses.
"""
from __future__ import annotations

import astropy.units as u
from astropy.cosmology import Planck18


def rescale_flux(data, z_obs, z_sim, cosmo=Planck18):
    """Dim flux densities f_lambda from z_obs to z_sim.

    f_lambda -> f_lambda * (D_L(z_obs)/D_L(z_sim))**2 * (1+z_obs)/(1+z_sim).
    Combined with the stretched wavelength step this conserves luminosity, i.e.
    the integrated line flux scales as D_L**-2.

    Parameters
    ----------
    data : array-like
        Flux density array (e.g. ``cube.data``).
    z_obs, z_sim : float
        Original and simulated redshift.
    """
    dl_obs = cosmo.luminosity_distance(z_obs).to("cm").value
    dl_sim = cosmo.luminosity_distance(z_sim).to("cm").value
    scale = (dl_obs / dl_sim) ** 2 * ((1.0 + z_obs) / (1.0 + z_sim))
    return data * scale


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
