"""Synthetic cubes and mock instrument files for tests and quick experiments."""
from __future__ import annotations

import os

import astropy.units as u
import numpy as np
from astropy.io import fits
from mpdaf.obj import WCS as MPDAF_WCS
from mpdaf.obj import Cube, WaveCoord

OIII_5007 = 5006.84


def make_test_cube(nx=30, ny=30, pixscale_arcsec=0.29, z=0.025, lam0=5080.0, dlam=0.5, nw=200,
                   line_rest=OIII_5007, line_sigma=1.0, sigma_pix=3.0, continuum=0.05, peak=1e-17,
                   with_var=True, wcs_in_primary=True, nan_cols=0):
    """Gaussian blob with one emission line on a flat continuum (KCWI-like defaults).

    Parameters
    ----------
    nx, ny, nw : int
        Cube size.
    pixscale_arcsec : float
        Spaxel size.
    z : float
        Redshift of the line.
    lam0, dlam : float
        First wavelength and step (Angstrom).
    line_rest, line_sigma : float
        Rest wavelength and observed Gaussian sigma (Angstrom) of the line.
    sigma_pix : float
        Spatial Gaussian sigma in pixels.
    continuum, peak : float
        Continuum level relative to the line peak, and the line peak value.
    with_var : bool
        Add a smooth, non-uniform variance cube.
    wcs_in_primary : bool
        Copy the spatial WCS to the primary header (as in KCWI files).
    nan_cols : int
        Set the first ``nan_cols`` columns to NaN.
    """
    lam = lam0 + dlam * np.arange(nw)
    yy, xx = np.mgrid[0:ny, 0:nx]
    spat = np.exp(-((xx - (nx - 1) / 2) ** 2 + (yy - (ny - 1) / 2) ** 2) / (2 * sigma_pix ** 2))
    spec = peak * (continuum + np.exp(-0.5 * ((lam - line_rest * (1 + z)) / line_sigma) ** 2))
    data = spec[:, None, None] * spat[None]
    var = None
    if with_var:
        var = (0.01 * data.max()) ** 2 * (1 + xx / nx + 2 * yy / ny)[None] * np.ones((nw, 1, 1))
    if nan_cols:
        data[:, :, :nan_cols] = np.nan
    cube = Cube(data=data, var=var,
                wcs=MPDAF_WCS(crval=(10.0, 10.0), cdelt=(pixscale_arcsec / 3600, -pixscale_arcsec / 3600),
                              crpix=(ny / 2, nx / 2), shape=(ny, nx), deg=True),
                wave=WaveCoord(crval=lam0, cdelt=dlam, crpix=1, cunit=u.AA))
    if wcs_in_primary:
        for k, v in cube.wcs.to_header().items():
            cube.primary_header[k] = v
    return cube


def _spatial_wcs(nx, ny, pixscale_arcsec):
    h = fits.Header()
    h["NAXIS"] = 2
    h["NAXIS1"], h["NAXIS2"] = nx, ny
    h["CTYPE1"], h["CTYPE2"] = "RA---TAN", "DEC--TAN"
    h["CRVAL1"], h["CRVAL2"] = 0.0, 0.0
    h["CRPIX1"], h["CRPIX2"] = (nx + 1) / 2.0, (ny + 1) / 2.0
    h["CDELT1"] = h["CDELT2"] = pixscale_arcsec / 3600.0
    h["CUNIT1"] = h["CUNIT2"] = "deg"
    return MPDAF_WCS(h)


def make_sersic_cube(nx=100, ny=100, nw=50, lam_min=5000.0, lam_max=5100.0, line_sigma_pix=2.0 / 2.355,
                     pixscale_arcsec=0.03, sersic_n=1.0, r_eff=10.0, normalize="peak"):
    """Gaussian line (centred on the band) in every spaxel, scaled by a Sersic profile.

    Parameters
    ----------
    line_sigma_pix : float
        Gaussian sigma of the line in spectral pixels.
    normalize : {'peak', 'total'}
        Scale the Sersic profile to a peak of 1 or a total of 1.
    """
    from astropy.modeling.models import Sersic2D

    cdelt = (lam_max - lam_min) / (nw - 1)
    wave = WaveCoord(crval=lam_min, cdelt=cdelt, crpix=1, ctype="WAVE", cunit="Angstrom")
    spec_pix = np.arange(nw)
    spectrum = np.exp(-0.5 * ((spec_pix - (nw - 1) / 2.0) / line_sigma_pix) ** 2)

    y, x = np.mgrid[0:ny, 0:nx]
    sersic = Sersic2D(amplitude=1.0, r_eff=r_eff, n=sersic_n, x_0=(nx - 1) / 2, y_0=(ny - 1) / 2)(x, y)
    sersic /= sersic.max() if normalize == "peak" else sersic.sum()

    cube = Cube(data=spectrum[:, None, None] * sersic[None], wave=wave,
                wcs=_spatial_wcs(nx, ny, pixscale_arcsec))
    hdr = cube.data_header
    hdr["CRVAL3"], hdr["CDELT3"], hdr["CRPIX3"] = lam_min, cdelt, 1
    hdr["CTYPE3"], hdr["CUNIT3"] = "WAVE", "Angstrom"
    return cube


def write_mock_instrument_files(directory):
    """Write stand-in NIRSpec filter-throughput and dispersion tables.

    Not real JWST data (smooth throughput curves, R proportional to lambda);
    use them to run the pipeline without the JDox files.
    """
    os.makedirs(directory, exist_ok=True)
    for mode, (lo, hi) in {"f170lp": (1.7, 3.2), "f290lp": (2.9, 5.3),
                           "f100lp": (1.0, 1.9), "f070lp": (0.7, 1.3)}.items():
        wl = np.linspace(lo - 0.1, hi + 0.1, 400)
        thr = 0.6 * np.exp(-0.5 * ((wl - (lo + hi) / 2) / (hi - lo)) ** 2)
        thr[(wl < lo) | (wl > hi)] = 0.0
        fits.BinTableHDU.from_columns([
            fits.Column(name="WAVELENGTH", format="D", array=wl),
            fits.Column(name="THROUGHPUT", format="D", array=thr),
        ]).writeto(os.path.join(directory, f"jwst_nirspec_{mode}_trans.fits"), overwrite=True)
    for name, lo, hi, r0 in [("g235h", 1.66, 3.05, 2700), ("g235m", 1.66, 3.07, 1000)]:
        wl = np.arange(lo, hi, 3.96e-4)
        fits.BinTableHDU.from_columns([
            fits.Column(name="WAVELENGTH", format="D", array=wl),
            fits.Column(name="R", format="D", array=r0 * wl / 2.36),
        ]).writeto(os.path.join(directory, f"jwst_nirspec_{name}_disp.fits"), overwrite=True)
    return directory


def fit_gaussian_line(wave, spectrum, center=None, window=None):
    """Fit amp*exp(-(x-mu)^2/2sigma^2)+offset; returns dict with mu, sigma, fwhm, amp, offset."""
    from scipy.optimize import curve_fit

    wave = np.asarray(wave, float)
    spectrum = np.asarray(spectrum, float)
    if center is not None and window is not None:
        sel = np.abs(wave - center) < window
        wave, spectrum = wave[sel], spectrum[sel]
    p0 = [spectrum.max() - np.median(spectrum), wave[np.argmax(spectrum)],
          3 * np.median(np.diff(wave)), np.median(spectrum)]
    popt, _ = curve_fit(lambda x, a, mu, s, b: a * np.exp(-0.5 * ((x - mu) / s) ** 2) + b,
                        wave, spectrum, p0=p0, maxfev=5000)
    amp, mu, sigma, offset = popt
    return {"amp": amp, "mu": mu, "sigma": abs(sigma), "fwhm": 2.3548200450309493 * abs(sigma),
            "offset": offset}


def make_hst_like_image(nx=400, ny=420, pixscale_arcsec=0.05, rot_deg=8.0, sigma_pix=20.0,
                        peak=1.0, background=0.0, point_source=False):
    """Synthetic HST/ACS-like image: Gaussian galaxy in electrons/s with a rotated CD matrix.

    Parameters
    ----------
    sigma_pix : float
        Gaussian sigma of the galaxy in pixels (ignored if ``point_source``).
    point_source : bool
        Put all the flux in the central pixel instead.

    Returns
    -------
    mpdaf.obj.Image
        With BUNIT, PHOTFLAM, PHOTPLAM (F555W-like) in ``data_header``.
    """
    from mpdaf.obj import Image

    yy, xx = np.mgrid[0:ny, 0:nx]
    if point_source:
        data = np.full((ny, nx), background, float)
        data[ny // 2, nx // 2] += peak
    else:
        data = background + peak * np.exp(-((xx - (nx - 1) / 2) ** 2 + (yy - (ny - 1) / 2) ** 2)
                                          / (2 * sigma_pix ** 2))
    s = pixscale_arcsec / 3600.0
    th = np.radians(rot_deg)
    h = fits.Header()
    h["NAXIS"], h["NAXIS1"], h["NAXIS2"] = 2, nx, ny
    h["CTYPE1"], h["CTYPE2"] = "RA---TAN", "DEC--TAN"
    h["CRVAL1"], h["CRVAL2"] = 13.73, -37.68
    h["CRPIX1"], h["CRPIX2"] = (nx + 1) / 2, (ny + 1) / 2
    h["CD1_1"], h["CD1_2"] = -s * np.cos(th), -s * np.sin(th)
    h["CD2_1"], h["CD2_2"] = -s * np.sin(th), s * np.cos(th)
    image = Image(data=data, wcs=MPDAF_WCS(h))
    image.data_header["BUNIT"] = "ELECTRONS/S"
    image.data_header["PHOTFLAM"] = 1.93985e-19
    image.data_header["PHOTPLAM"] = 5360.24
    return image


def write_hst_like_drc(path, uncovered_corner=40, **kwargs):
    """Write :func:`make_hst_like_image` as a _drc-style file (PRIMARY, SCI, WHT).

    The WHT is zero in a ``uncovered_corner`` x ``uncovered_corner`` corner.
    """
    image = make_hst_like_image(**kwargs)
    sci = fits.ImageHDU(np.asarray(image.data, dtype=np.float32), header=image.wcs.to_header(), name="SCI")
    for k in ("BUNIT", "PHOTFLAM", "PHOTPLAM"):
        sci.header[k] = image.data_header[k]
    wht = np.full(image.shape, 1080.0, np.float32)
    wht[:uncovered_corner, :uncovered_corner] = 0
    primary = fits.PrimaryHDU()
    primary.header.update(TELESCOP="HST", INSTRUME="ACS", DETECTOR="WFC", FILTER1="F555W",
                          FILTER2="CLEAR2L", TARGNAME="TEST", EXPTIME=1080.0)
    fits.HDUList([primary, sci, fits.ImageHDU(wht, header=image.wcs.to_header(), name="WHT")]).writeto(
        path, overwrite=True)
    return path
