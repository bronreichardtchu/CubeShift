from ioTools import read_in_data

def old_redshift_wavelength_axis(filename, z_obs, z_sim):
    """
    Redshift the wavelength axis of a cube from z_old to z_new,
    returning a new cube with unchanged data, but redshifted wavelengths.
    Works for rectangular pixels.

    Parameters
    ----------
    filename : str
        Path to the FITS cube.
    z_obs : float
        Original redshift of the source.
    z_sim : float
        New redshift to shift the cube to.

    Returns
    -------
    redshifted_cube : mpdaf.obj.Cube
        Cube with same data, mask, var, but redshifted wavelengths.
    lam_new : np.ndarray
        New wavelength array.
    """

    # --- load data & wavelength axis from FITS ---
    lam_old, data, *rest = read_in_data_fits(filename)
    if len(rest) == 2:
        var, header = rest
    else:
        header = rest[0]
        var = None

    # --- redshift wavelength axis ---
    lam_new = lam_old * (1 + z_sim) / (1 + z_obs)
    wave = WaveCoord(
        crval=lam_new[0],
        cdelt=lam_new[1] - lam_new[0],
        crpix=1,
        cunit='Angstrom',
        ctype='WAVE'
    )

    # --- build Cube (only spectral WCS is rebuilt) ---
    redshifted_cube = Cube(data=data, var=var, header=header, wave=wave)

    # --- update wavelength keywords ---
    hdr = redshifted_cube.data_header
    hdr["CRVAL3"] = lam_new[0]
    hdr["CDELT3"] = lam_new[1] - lam_new[0]
    hdr["CTYPE3"] = "WAVE" # vaccum or air?
    hdr["CUNIT3"] = "Angstrom"
    hdr["REDSHIFT"] = z_sim

    # --- preserve spatial WCS keywords from input header ---
    for key in ["CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2"]:
        if key in header:
            hdr[key] = header[key]

    # --- get pixel scales from input header ---
    # FITS WCS stores them in degrees/pixel
    dx_deg = abs(header.get("CDELT1", -0.0002777))  # default = 1 arcsec in deg
    dy_deg = abs(header.get("CDELT2",  0.0002777))  # default = 1 arcsec in deg

    dx_arcsec = dx_deg * 3600.0
    dy_arcsec = dy_deg * 3600.0

    # --- store arcsec scales in cube object for convenience ---
    redshifted_cube.dx_arcsec = dx_arcsec
    redshifted_cube.dy_arcsec = dy_arcsec

    # --- write pixel scales back to FITS header (degrees/pixel) ---
    hdr['CDELT1'] = -dx_deg   # RA axis should be negative
    hdr['CDELT2'] =  dy_deg
    hdr['CUNIT1'] = 'deg'
    hdr['CUNIT2'] = 'deg'


    return redshifted_cube, lam_new