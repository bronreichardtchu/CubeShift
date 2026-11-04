
from astropy.wcs import WCS
import numpy as np
from astropy.io import fits
from mpdaf.obj import Cube

def read_in_data(input_cube):
    """
    Reads in a data cube from an MPDAF Cube object or a FITS filename.
    Handles variance cubes, spectral axis info, and missing keywords robustly.

    Parameters
    ----------
    input_cube : str or mpdaf.obj.Cube
        Either the filename of the FITS cube or an MPDAF Cube object.

    Returns
    -------
    lamdas : np.ndarray
        Wavelength vector (1D) in the same units as the cube (typically Ångström).
    data : np.ndarray
        Data cube (nz, ny, nx).
    var : np.ndarray or None
        Variance cube, if available.
    header : astropy.io.fits.Header
        Header of the HDU containing the data.
    """
    # --- Case 1: input is already an MPDAF Cube ---
    if isinstance(input_cube, Cube):
        cube = input_cube
        data = cube.data
        lamdas = cube.wave.coord()
        var = cube.var if cube.var is not None else None
        header = cube.primary_header
        print(f"[DEBUG] Loaded MPDAF Cube: {data.shape[0]} spectral channels, λ range {lamdas[0]/1e4:.3f} – {lamdas[-1]/1e4:.3f} μm")
        return (lamdas, data, var, header) if var is not None else (lamdas, data, header)

    # --- Case 2: input is a FITS filename ---
    filename = input_cube
    if filename is None:
        raise ValueError("No filename provided to read_in_data.")

    with fits.open(filename) as hdulist:
        data_hdu = None
        lamdas = None
        var = None
        header = None

        # search for the first HDU with data
        for hdu in hdulist:
            if hdu.data is not None:
                data_hdu = hdu
                data = hdu.data
                header = hdu.header

                # attempt to build wavelength axis from header keywords
                try:
                    crval = header['CRVAL3']
                    cdelt = header.get('CDELT3', header.get('CD3_3'))
                    naxis = header['NAXIS3']
                    lamdas = np.arange(crval, crval + naxis*cdelt, cdelt)
                    break  # success
                except KeyError:
                    continue

        if data_hdu is None:
            raise ValueError(f"No data found in FITS file {filename}")

        if lamdas is None:
            print("[WARNING] Could not find CRVAL3/CD3_3 keywords; returning dummy λ axis")
            lamdas = np.arange(data.shape[0])

        # check for variance extension in other HDUs
        for hdu in hdulist[1:]:
            if hdu.data is not None:
                var = hdu.data
                break

    print(f"[DEBUG] Loaded FITS cube: {data.shape[0]} spectral channels, λ range {lamdas[0]/1e4:.3f} – {lamdas[-1]/1e4:.3f} μm")
    return (lamdas, data, var, header) 
