"""Redshifting the wavelength axis of a cube, and wavelength-frame conversions."""
from __future__ import annotations

import numpy as np
from astropy.io import fits
from mpdaf.obj import WCS as MPDAF_WCS
from mpdaf.obj import Cube, WaveCoord

from .io import read_in_data


def redshift_wavelength_axis(filename, z_obs, z_sim, pixscale_arcsec=None, keep_ra_negative=True):
    """Redshift the wavelength axis of a cube from z_obs to z_sim.

    lambda_new = lambda_old * (1 + z_sim) / (1 + z_obs).  Data values are not
    changed (see :func:`cubeshift.cosmology.rescale_flux` for the dimming).

    Parameters
    ----------
    filename : str or mpdaf.obj.Cube
        FITS path or Cube.
    z_obs : float
        Original redshift of the source.
    z_sim : float
        Redshift to simulate.
    pixscale_arcsec : float or (float, float), optional
        Spatial pixel scale (arcsec/pix), used ONLY if the header lacks a
        spatial WCS.  A single float is applied to both axes.
    keep_ra_negative : bool
        If building a WCS, use the conventional negative CDELT1 for RA.

    Returns
    -------
    redshifted_cube : mpdaf.obj.Cube
    lam_new : np.ndarray
    """
    lam_old, data, var, header = read_in_data(filename)
    if header is None:
        header = fits.Header()

    lam_new = lam_old * (1.0 + z_sim) / (1.0 + z_obs)

    # Wavelength unit from the header if present, else Angstrom
    wave_unit = header.get("CUNIT3") or header.get("WCSDIM3") or "Angstrom"
    wave = WaveCoord(crval=lam_new[0], cdelt=np.median(np.diff(lam_new)), crpix=1,
                     cunit=wave_unit, ctype="WAVE")

    # Spatial WCS: reuse the header's if it has one (keeps CD/PC matrix, rotation)
    has_spatial = (("CDELT1" in header or "CD1_1" in header or "PC1_1" in header) and
                   ("CDELT2" in header or "CD2_2" in header or "PC2_2" in header))

    if has_spatial:
        wcs_spatial = MPDAF_WCS(header)
    else:
        if pixscale_arcsec is None:
            raise ValueError(
                "Input header has no spatial WCS (no CDELT/CD/PC keywords). "
                "Provide pixscale_arcsec (float or (x,y)) to build a WCS."
            )
        if isinstance(pixscale_arcsec, (tuple, list, np.ndarray)):
            px_x_arcsec, px_y_arcsec = float(pixscale_arcsec[0]), float(pixscale_arcsec[1])
        else:
            px_x_arcsec = px_y_arcsec = float(pixscale_arcsec)

        cdelt1_deg = -(px_x_arcsec / 3600.0) if keep_ra_negative else (px_x_arcsec / 3600.0)
        cdelt2_deg = px_y_arcsec / 3600.0
        # mpdaf takes (y, x) = (Dec, RA) order; deg=True gives a RA---TAN/DEC--TAN WCS in degrees
        wcs_spatial = MPDAF_WCS(
            crval=(header.get("CRVAL2", 0.0), header.get("CRVAL1", 0.0)),
            cdelt=(cdelt2_deg, cdelt1_deg),
            crpix=(header.get("CRPIX2", 1.0), header.get("CRPIX1", 1.0)),
            shape=np.shape(data)[1:],
            deg=True,
        )

    redshifted_cube = Cube(data=data, var=var, wcs=wcs_spatial, wave=wave)
    redshifted_cube.data_header["REDSHIFT"] = (z_sim, "Simulated redshift applied to wavelength axis")
    return redshifted_cube, lam_new


def air_to_vac(wavelength: np.ndarray) -> np.ndarray:
    """Convert air wavelengths to vacuum (Greisen et al. 2006, eqns 64-65).

    The error in the index of refraction is ~1e-9, below that of the empirical
    formula.  Adapted from ``specutils.utils.wcs_utils``.  Whether a file is in
    air or vacuum is given by CTYPE3 (AWAV/AWAV-LOG = air, WAVE/WAVE-LOG = vacuum).

    Parameters
    ----------
    wavelength : np.ndarray
        Air wavelength(s) in Angstrom.

    Returns
    -------
    np.ndarray
        Vacuum wavelength(s) in Angstrom.
    """
    wlum = wavelength / 10000
    return (1 + 1e-6 * (287.6155 + 1.62887 / wlum ** 2 + 0.01360 / wlum ** 4)) * wavelength
