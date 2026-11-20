"""Redshifting the wavelength axis of a cube, and wavelength-frame conversions."""
from __future__ import annotations

import warnings

import astropy.units as u
import numpy as np
from astropy.io import fits
from mpdaf.obj import WCS as MPDAF_WCS
from mpdaf.obj import Cube, WaveCoord
from mpdaf.tools import fix_unit_read

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
        Spatial pixel scale (arcsec/pix), used ONLY if there is no spatial WCS
        (``cube.wcs`` for a Cube, the header for a file).  A single float is
        applied to both axes.
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

    # every wavelength is stretched by the same factor
    stretch = (1.0 + z_sim) / (1.0 + z_obs)
    lam_new = lam_old * stretch

    if isinstance(filename, Cube):
        # Cube input: its WaveCoord already knows the reference pixel and unit, so
        # scaling CRVAL and CDELT by the stretch gives exactly lam_old * stretch
        wave = filename.wave.copy()
        wave.set_crval(filename.wave.get_crval() * stretch)
        wave.set_step(filename.wave.get_step() * stretch)
    else:
        # FITS input: build a linear axis from the (CRPIX3-aware) wavelengths,
        # with the unit from the header if present, else Angstrom
        wave_unit = header.get("CUNIT3") or header.get("WCSDIM3") or "Angstrom"
        wave = WaveCoord(crval=lam_new[0], cdelt=np.median(np.diff(lam_new)), crpix=1,
                         cunit=wave_unit, ctype="WAVE")

    # Spatial WCS for a Cube: its own WCS object (review v2 M6).  mpdaf keeps the
    # WCS in the data extension, so the primary header of a MUSE file or of a
    # cube built in memory has none.  A Cube made without a WCS has wcs None.
    cube_wcs = filename.wcs if isinstance(filename, Cube) else None

    # Spatial WCS for a file: reuse the header's if it has one (keeps CD/PC matrix, rotation)
    has_spatial = (("CDELT1" in header or "CD1_1" in header or "PC1_1" in header) and
                   ("CDELT2" in header or "CD2_2" in header or "PC2_2" in header))

    if cube_wcs is not None:
        wcs_spatial = cube_wcs.copy()
    elif has_spatial:
        wcs_spatial = MPDAF_WCS(header)
    else:
        # no WCS anywhere: build a simple TAN WCS from the given pixel scale
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

    # Units and headers (review v2 M14): keep the flux unit and the primary
    # header (observation metadata), which were dropped before
    if isinstance(filename, Cube):
        # a Cube already holds its unit; earlier CubeShift steps may have left a
        # CSBUNIT card for a unit mpdaf could not parse
        unit = filename.unit
        csbunit = filename.data_header.get("CSBUNIT")
        primary = filename.primary_header
    else:
        # a file: parse BUNIT of the data HDU, and read the primary header
        unit, csbunit = _unit_from_bunit(header.get("BUNIT"))
        primary = fits.getheader(filename, 0)

    redshifted_cube = Cube(data=data, var=var, wcs=wcs_spatial, wave=wave, unit=unit,
                           primary_header=_primary_metadata(primary))
    # keep the original BUNIT string when mpdaf cannot represent it
    if csbunit:
        redshifted_cube.data_header["CSBUNIT"] = (csbunit, "original BUNIT of the input cube")
    redshifted_cube.data_header["REDSHIFT"] = (z_sim, "Simulated redshift applied to wavelength axis")
    return redshifted_cube, lam_new


def _unit_from_bunit(bunit):
    """(astropy unit, raw string to keep) for a FITS BUNIT value.

    Returns (unit, None) when the string parses, (dimensionless, bunit) when it
    does not (e.g. 'ELECTRONS/S'), so the caller can store it as CSBUNIT, and
    (dimensionless, None) when there is no BUNIT (mpdaf's own default).
    """
    if not bunit:
        return u.dimensionless_unscaled, None
    try:
        # fix_unit_read handles the spellings mpdaf itself accepts on reading
        return u.Unit(fix_unit_read(str(bunit))), None
    except ValueError:
        warnings.warn(f"BUNIT {bunit!r} is not a unit astropy understands; the cube is "
                      "treated as dimensionless and the string is kept as CSBUNIT")
        return u.dimensionless_unscaled, str(bunit)


# Primary-header keywords that describe an HDU's array or WCS rather than the
# observation.  They are dropped when the primary header is carried over: the
# output's arrays and WCS live in its DATA/STAT extensions and have changed
# (wavelengths redshifted, pixels shrunk), so stale copies would mislead.
_STRUCTURE_KEYS = {"SIMPLE", "BITPIX", "EXTEND", "XTENSION", "PCOUNT", "GCOUNT", "EXTNAME",
                   "BSCALE", "BZERO", "BUNIT", "WCSAXES", "LONPOLE", "LATPOLE", "CHECKSUM",
                   "DATASUM"}
_STRUCTURE_PREFIXES = ("NAXIS", "CRVAL", "CRPIX", "CDELT", "CTYPE", "CUNIT", "CROTA",
                       "CD1_", "CD2_", "CD3_", "PC1_", "PC2_", "PC3_")


def _primary_metadata(header):
    """Copy of a primary header without the array-structure and WCS keywords."""
    out = fits.Header()
    for card in header.cards:
        key = card.keyword
        # skip structure/WCS cards; keep everything else (incl. COMMENT/HISTORY)
        if key in _STRUCTURE_KEYS or key.startswith(_STRUCTURE_PREFIXES):
            continue
        out.append(card)
    return out


def air_to_vac(wavelength: np.ndarray) -> np.ndarray:
    """Convert air wavelengths to vacuum (IAU standard: dry air at 15 C, 1 atm).

    Uses the inversion by N. Piskunov (VALD) of the Morton (2000) / Ciddor
    (1996) refractive index, which is what the IAU standard air wavelengths
    are defined with: Halpha 6562.80 -> 6564.61 A, [OIII] 5006.84 -> 5008.24 A.
    It round-trips with the Morton (2000) vac -> air formula to ~1e-8 A.
    Valid from ~2000 A into the near-IR.

    The formula used before (Greisen et al. 2006 eq. 65) is the refractivity
    of air at 0 C, ~5% too high for the 15 C standard: +0.1 A (~5 km/s) at
    Halpha.

    Whether a file is in air or vacuum is given by CTYPE3 (AWAV/AWAV-LOG = air,
    WAVE/WAVE-LOG = vacuum).

    Parameters
    ----------
    wavelength : np.ndarray
        Air wavelength(s) in Angstrom.

    Returns
    -------
    np.ndarray
        Vacuum wavelength(s) in Angstrom.
    """
    wavelength = np.asarray(wavelength, dtype=float)
    # squared wavenumber in 1/micron^2 (computed from the air wavelength)
    s2 = (1e4 / wavelength) ** 2
    # refractive index n(s) of standard air; lambda_vac = n * lambda_air
    n = 1.0 + 8.336624212083e-5 + 2.408926869968e-2 / (130.1065924522 - s2) \
        + 1.599740894897e-4 / (38.92568793293 - s2)
    return n * wavelength
