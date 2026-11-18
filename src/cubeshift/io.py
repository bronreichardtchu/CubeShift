"""Reading IFU data into mpdaf objects or plain arrays."""
from __future__ import annotations

import warnings

import numpy as np
from astropy.io import fits
from mpdaf.obj import Cube, Image


def _check_num_axes(filename: str) -> int:
    """Number of data axes in a FITS file (primary HDU, else first extension)."""
    ndim = fits.getheader(filename, ext=0)["NAXIS"]
    if ndim == 0:
        ndim = fits.getheader(filename, ext=1)["NAXIS"]
    return ndim


def read_in_datacube(filename: str, ext=None) -> Cube:
    """Read a FITS cube into an mpdaf Cube.

    Parameters
    ----------
    filename : str
        Path to the FITS cube.
    ext : int or str or (int|str, int|str), optional
        Data extension, or (data, variance) extensions.  By default mpdaf
        chooses (DATA/STAT or the first data HDU).
    """
    return Cube(filename) if ext is None else Cube(filename, ext=ext)


def read_in_dataim(filename: str, ext=None) -> Image:
    """Read a FITS image into an mpdaf Image (see :func:`read_in_datacube`)."""
    return Image(filename) if ext is None else Image(filename, ext=ext)


def read_fits(filename: str, ext=None) -> Cube | Image:
    """Open a FITS file as an mpdaf Cube or Image depending on its dimensions."""
    if not isinstance(filename, str):
        raise TypeError(f"filename must be str, not {type(filename)}")
    ndim = _check_num_axes(filename)
    if ndim == 3:
        return read_in_datacube(filename, ext=ext)
    if ndim == 2:
        return read_in_dataim(filename, ext=ext)
    raise ValueError(f"Expected a 2D image or 3D cube, {filename} has {ndim} axes")


def read_cube(cube_or_path, ext=None) -> Cube:
    """Return an mpdaf Cube, reading it from disk if a path is given."""
    if isinstance(cube_or_path, Cube):
        return cube_or_path
    return read_in_datacube(str(cube_or_path), ext=ext)


def _wavelength_axis(header) -> np.ndarray | None:
    """Wavelength of every pixel along FITS axis 3, or None if the header has no axis 3.

    Uses the linear FITS WCS rule  lambda_i = CRVAL3 + (i + 1 - CRPIX3) * CDELT3
    for 0-based pixel index i.  CRPIX3 defaults to 1 (the FITS default), so files
    whose reference pixel is not the first one are no longer shifted, and the
    axis always has exactly NAXIS3 elements (``np.arange`` with a float step can
    return NAXIS3 + 1 through rounding).
    """
    # the reference value and the number of pixels are required
    if "CRVAL3" not in header or "NAXIS3" not in header:
        return None
    # the step: CDELT3, else the CD matrix element (no rotation along the spectral axis)
    cdelt = header.get("CDELT3", header.get("CD3_3"))
    if cdelt is None:
        return None
    # reference pixel (1-based); FITS says it defaults to 1 when absent
    crpix = header.get("CRPIX3", 1.0)
    naxis = int(header["NAXIS3"])
    # integer pixel indices -> exactly naxis wavelengths, with the reference pixel at CRVAL3
    return header["CRVAL3"] + (np.arange(naxis) + 1.0 - crpix) * cdelt


# EXTNAMEs recognised as uncertainties, by what they hold (compared upper-case):
#   variance sigma^2 (mpdaf/MUSE 'STAT'), standard deviation sigma (KCWI DRP
#   'UNCERT', JWST 'ERR'), or inverse variance 1/sigma^2 (SDSS/MaNGA 'IVAR')
_VARIANCE_NAMES = ("STAT", "VAR", "VARIANCE")
_SIGMA_NAMES = ("ERR", "ERROR", "SIGMA", "STDDEV", "UNCERT", "UNCERTAINTY", "NOISE")
_IVAR_NAMES = ("IVAR", "INVVAR")
_UNCERTAINTY_NAMES = _VARIANCE_NAMES + _SIGMA_NAMES + _IVAR_NAMES
# extensions that are never the science data (uncertainties, masks, flags)
_NOT_DATA_NAMES = _UNCERTAINTY_NAMES + ("MASK", "FLAGS", "DQ", "BADPIX")


def _extname(hdu) -> str:
    """EXTNAME of an HDU in upper case ('' if it has none)."""
    return str(hdu.header.get("EXTNAME", "")).strip().upper()


def _variance_from_hdulist(hdulist, data_hdu, filename) -> np.ndarray | None:
    """Variance (sigma^2) from the first extension named as an uncertainty.

    Only EXTNAME is trusted: positional guesses picked the DATA extension of
    MUSE-layout files and the MASK (or sigma) of KCWI files.  Returns None, with
    a warning listing the unused extensions, when no uncertainty is named.
    """
    unused = []
    for hdu in hdulist:
        # skip the science data itself and empty HDUs (e.g. an empty primary)
        if hdu is data_hdu or hdu.data is None:
            continue
        name = _extname(hdu)
        # an uncertainty must cover the data pixel for pixel
        if name in _UNCERTAINTY_NAMES and hdu.data.shape != data_hdu.data.shape:
            warnings.warn(f"{filename}: extension {name} has shape {hdu.data.shape}, "
                          f"data has {data_hdu.data.shape}; not used as variance")
            continue
        # sigma^2 already: copy out of the (closing) file
        if name in _VARIANCE_NAMES:
            return np.array(hdu.data, dtype=float)
        # standard deviation: square it
        if name in _SIGMA_NAMES:
            return np.array(hdu.data, dtype=float) ** 2
        # inverse variance: 1/ivar, with ivar <= 0 (no information) -> NaN
        if name in _IVAR_NAMES:
            ivar = np.array(hdu.data, dtype=float)
            with np.errstate(divide="ignore"):
                return np.where(ivar > 0, 1.0 / ivar, np.nan)
        # anything else (MASK, FLAGS, unnamed) is reported, never used as variance
        unused.append(name or f"HDU {hdulist.index(hdu)}")
    if unused:
        warnings.warn(f"{filename}: no variance extension found by EXTNAME "
                      f"(looked for {_UNCERTAINTY_NAMES}); "
                      f"ignored extensions {unused}. Returning var=None.")
    return None


def read_in_data(input_cube):
    """Read a cube into plain arrays.

    Parameters
    ----------
    input_cube : str or mpdaf.obj.Cube
        Filename of a FITS cube, or an mpdaf Cube.

    Returns
    -------
    lamdas : np.ndarray
        Wavelength vector, in the cube's units (normally Angstrom).
    data : np.ndarray
        Data cube (nz, ny, nx).
    var : np.ndarray or None
        Variance cube (sigma^2).  For a file it is read only from an extension
        whose EXTNAME marks it as an uncertainty: STAT/VAR/VARIANCE (used as is),
        ERR/ERROR/SIGMA/STDDEV/UNCERT/UNCERTAINTY/NOISE (squared) or IVAR/INVVAR
        (inverted).  None, with a warning, if there is none.
    header : astropy.io.fits.Header
        For a Cube, its primary header; for a file, the header of the first HDU
        with a spectral axis.
    """
    if isinstance(input_cube, Cube):
        cube = input_cube
        return cube.wave.coord(), cube.data, cube.var, cube.primary_header

    filename = input_cube
    if filename is None:
        raise ValueError("No filename provided to read_in_data.")

    with fits.open(filename) as hdulist:
        data_hdu = None
        lamdas = None
        var = None
        header = None

        # first HDU with data (and preferably a spectral axis), skipping
        # extensions that are named as uncertainties, masks or flags
        for hdu in hdulist:
            if hdu.data is not None and _extname(hdu) not in _NOT_DATA_NAMES:
                data_hdu = hdu
                data = hdu.data
                header = hdu.header
                lamdas = _wavelength_axis(header)
                if lamdas is not None:
                    break

        if data_hdu is None:
            raise ValueError(f"No data found in FITS file {filename}")

        if lamdas is None:
            print("[WARNING] Could not find CRVAL3/CD3_3 keywords; returning dummy wavelength axis")
            lamdas = np.arange(data.shape[0])

        # the variance comes only from an extension *named* as an uncertainty
        # (converted to sigma^2); an unnamed extension may be data, a mask or sigma
        var = _variance_from_hdulist(hdulist, data_hdu, filename)

    return lamdas, data, var, header
