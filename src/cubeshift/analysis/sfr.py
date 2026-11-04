"""Star-formation rates from (extinction-corrected) Hbeta fluxes."""
import numpy as np
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo
from astropy.wcs.utils import proj_plane_pixel_scales

# -------------------------------
# Calibration constants
# -------------------------------
CALIBRATIONS = {
    "reichardtchu2025": {
        "C_Halpha": 5.5335e-42,  # Msol/yr / erg/s
        "Halpha_Hbeta_ratio": 2.87
    },
    "hao2011": {
        "C_Halpha": 10**-41.257,  # Msol/yr / erg/s
        "Halpha_Hbeta_ratio": 2.87
    }
}

#: Default (Reichardt Chu et al. 2025) Halpha SFR calibration, Msun yr^-1 per (erg s^-1)
C_Halpha = CALIBRATIONS["reichardtchu2025"]["C_Halpha"]
#: Default intrinsic Halpha/Hbeta flux ratio (dimensionless)
F_Halpha_over_F_Hbeta = CALIBRATIONS["reichardtchu2025"]["Halpha_Hbeta_ratio"]


def flux_to_Halpha_luminosity(hbeta_flux_map, F_Halpha_over_F_Hbeta=F_Halpha_over_F_Hbeta, z=None, dl_cm=None):
    """
    Convert Hbeta flux map (erg s^-1 cm^-2) to Halpha luminosity map (erg s^-1).

    Parameters
    ----------
    hbeta_flux_map : array_like
        Hbeta flux per spaxel (erg s^-1 cm^-2).
    F_Halpha_over_F_Hbeta : float
        Halpha/Hbeta flux ratio used to convert Hbeta to Halpha.
    z : float, optional
        Redshift, used to compute the luminosity distance (Planck18) if
        ``dl_cm`` is not given.
    dl_cm : float, optional
        Luminosity distance in cm.

    Returns
    -------
    L_halpha : numpy.ndarray
        Halpha luminosity per spaxel (erg s^-1).
    """
    if dl_cm is None:
        if z is None:
            raise ValueError("Provide either z or dl_cm")
        dl = cosmo.luminosity_distance(z).to(u.cm).value
    else:
        dl = float(dl_cm)

    halpha_flux_map = F_Halpha_over_F_Hbeta * np.array(hbeta_flux_map, dtype=float)
    L_halpha = 4.0 * np.pi * (dl ** 2) * halpha_flux_map
    return L_halpha


def compute_sfr_from_hbeta_flux(hbeta_flux_map, Ahbeta_map,
                                C_Halpha=C_Halpha,
                                F_Halpha_over_F_Hbeta=F_Halpha_over_F_Hbeta,
                                z=None, dl_cm=None):
    """
    Compute SFR map (Msun yr^-1) from Hbeta flux map and extinction at Hbeta.

    Spaxels with non-finite or non-positive Hbeta flux are set to 0.

    Parameters
    ----------
    hbeta_flux_map : array_like
        Observed Hbeta flux per spaxel (erg s^-1 cm^-2).
    Ahbeta_map : array_like or float
        Extinction at Hbeta (mag).
    C_Halpha : float
        Halpha SFR calibration constant (Msun yr^-1 per erg s^-1).
    F_Halpha_over_F_Hbeta : float
        Halpha/Hbeta flux ratio.
    z, dl_cm : float, optional
        Redshift or luminosity distance in cm (see
        :func:`flux_to_Halpha_luminosity`).

    Returns
    -------
    sfr_map : numpy.ndarray
        SFR per spaxel (Msun yr^-1).
    """
    hbeta_flux_map = np.array(hbeta_flux_map, dtype=float)
    Ahbeta_map = np.array(Ahbeta_map, dtype=float)

    bad_mask = ~np.isfinite(hbeta_flux_map) | (hbeta_flux_map <= 0)

    # Correct Hbeta flux for dust at Hbeta
    corrected_hbeta_flux = hbeta_flux_map * (10.0 ** (0.4 * Ahbeta_map))

    # Convert corrected Hbeta flux -> Halpha luminosity
    L_halpha = flux_to_Halpha_luminosity(corrected_hbeta_flux,
                                         F_Halpha_over_F_Hbeta=F_Halpha_over_F_Hbeta,
                                         z=z, dl_cm=dl_cm)

    # SFR (Msun/yr) from Halpha luminosity
    sfr_map = C_Halpha * L_halpha

    sfr_map[bad_mask] = 0.0
    return sfr_map


def _pixel_area_arcsec2(wcs):
    """Spatial pixel area (arcsec^2) from an MPDAF or astropy WCS."""
    if hasattr(wcs, "get_step"):
        # mpdaf.obj.WCS: get_step returns (dy, dx)
        dy, dx = np.abs(wcs.get_step(unit=u.arcsec))
        return dy * dx
    spatial_wcs = wcs.celestial if wcs.naxis > 2 else wcs
    scales = proj_plane_pixel_scales(spatial_wcs) * 3600.0
    return scales[0] * scales[1]


def compute_sfr_from_Hbeta(
    cube,
    F_Hbeta,
    z,
    A_Hbeta,
    calibration="reichardtchu2025",
    wcs=None
):
    """
    Compute star formation rates from Hβ fluxes.

    Parameters
    ----------
    cube : mpdaf.obj.Cube or None
        The cube the fluxes were measured from. Only used as a fallback source
        of the spatial WCS when ``wcs`` is ``True``.
    F_Hbeta : 2D array
        Observed Hβ flux per spaxel (erg/s/cm^2).
    z : float
        Redshift of the galaxy.
    A_Hbeta : 2D array or float
        Extinction at Hβ (mag).
    calibration : str
        Choice of calibration constants ("reichardtchu2025" or "hao2011").
    wcs : mpdaf.obj.WCS, astropy.wcs.WCS or True, optional
        WCS for the ΣSFR calculation (``True`` means use ``cube.wcs``). If
        None, no surface density is computed.

    Returns
    -------
    sfr_map : 2D array
        Star formation rate per spaxel (Msol/yr).
    total_sfr : float
        Total star formation rate (Msol/yr).
    sfr_surface_density : 2D array or None
        SFR surface density map (Msol/yr/kpc^2), if WCS is provided.
    """
    cal = CALIBRATIONS[calibration]

    # Same physics as compute_sfr_from_hbeta_flux, without the bad-pixel zeroing:
    # SFR = C * (Ha/Hb) * 10^(0.4 A_Hb) * 4 pi d_L^2 F_Hb
    corrected = np.asarray(F_Hbeta, dtype=float) * (10**(0.4 * np.asarray(A_Hbeta, dtype=float)))
    L_Halpha_corr = flux_to_Halpha_luminosity(
        corrected, F_Halpha_over_F_Hbeta=cal["Halpha_Hbeta_ratio"], z=z
    )
    sfr_map = cal["C_Halpha"] * L_Halpha_corr

    total_sfr = np.nansum(sfr_map)

    # Optional ΣSFR (requires WCS)
    sfr_surface_density = None
    if wcs is not None:
        if wcs is True:
            wcs = cube.wcs
        kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(z).value / 60.0
        pix_area_kpc2 = _pixel_area_arcsec2(wcs) * kpc_per_arcsec**2
        sfr_surface_density = sfr_map / pix_area_kpc2

    return sfr_map, total_sfr, sfr_surface_density
