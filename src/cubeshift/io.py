"""Reading IFU data into mpdaf objects or plain arrays."""
from __future__ import annotations

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


#: Primary-header keywords copied onto HST images read with :func:`read_hst_image`.
HST_PRIMARY_KEYS = ("TELESCOP", "INSTRUME", "DETECTOR", "FILTER", "FILTER1", "FILTER2", "TARGNAME",
                    "EXPTIME")


def read_hst_image(path, sci_ext="SCI", wht_ext="WHT", var=None) -> Image:
    """Read an HST drizzled image (``_drz``/``_drc``) into an mpdaf Image.

    Pixels with WHT <= 0 (no coverage) or non-finite values are masked.  The
    photometry keywords of the SCI header (BUNIT, PHOTFLAM, PHOTPLAM, ...) are
    kept in ``data_header``, and the main primary-header keywords (instrument,
    filter, target, exposure time) are copied there too.

    Parameters
    ----------
    path : str
    sci_ext, wht_ext : str or int
        Science and weight extensions; ``wht_ext=None`` skips the weight mask.
    var : np.ndarray, optional
        Variance image, if known.  None by default: a drizzled WHT is usually an
        exposure-time map, not an inverse variance.
    """
    from mpdaf.obj import WCS as MPDAF_WCS

    with fits.open(str(path)) as hdul:
        primary = hdul[0].header.copy()
        sci_header = hdul[sci_ext].header.copy()
        data = np.asarray(hdul[sci_ext].data, dtype=float)
        wht = None if wht_ext is None else np.asarray(hdul[wht_ext].data)
    mask = ~np.isfinite(data)
    if wht is not None:
        mask |= wht <= 0
    # mpdaf cannot represent HST units (e.g. ELECTRONS/S) in FITS, so the data are
    # kept unitless in mpdaf; the original unit is kept as CSBUNIT (mpdaf
    # rewrites BUNIT on writing)
    if "BUNIT" in sci_header:
        sci_header["CSBUNIT"] = (sci_header["BUNIT"], "original BUNIT of the input image")
    image = Image(data=np.ma.array(data, mask=mask), var=var, wcs=MPDAF_WCS(sci_header),
                  data_header=sci_header, primary_header=primary, copy=False)
    for key in HST_PRIMARY_KEYS:
        if key in primary and key not in image.data_header:
            image.data_header[key] = primary[key]
    return image


def read_cube(cube_or_path, ext=None) -> Cube:
    """Return an mpdaf Cube, reading it from disk if a path is given."""
    if isinstance(cube_or_path, Cube):
        return cube_or_path
    return read_in_datacube(str(cube_or_path), ext=ext)


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
        Variance cube if present.
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

        # first HDU with data (and preferably a spectral axis)
        for hdu in hdulist:
            if hdu.data is not None:
                data_hdu = hdu
                data = hdu.data
                header = hdu.header
                try:
                    crval = header["CRVAL3"]
                    cdelt = header.get("CDELT3", header.get("CD3_3"))
                    naxis = header["NAXIS3"]
                    lamdas = np.arange(crval, crval + naxis * cdelt, cdelt)
                    break
                except KeyError:
                    continue

        if data_hdu is None:
            raise ValueError(f"No data found in FITS file {filename}")

        if lamdas is None:
            print("[WARNING] Could not find CRVAL3/CD3_3 keywords; returning dummy wavelength axis")
            lamdas = np.arange(data.shape[0])

        # first later HDU with data is taken as the variance
        for hdu in hdulist[1:]:
            if hdu.data is not None:
                var = hdu.data
                break

    return lamdas, data, var, header
