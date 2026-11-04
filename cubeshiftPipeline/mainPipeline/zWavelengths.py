
import numpy as np
from mpdaf.obj import WaveCoord
from mpdaf.obj import Cube
from astropy.io import fits
from mpdaf.obj import WaveCoord
import matplotlib.pyplot as plt
import numpy as np
from mpdaf.obj import Cube, WaveCoord
from scipy.interpolate import interp1d
from mpdaf.obj import Cube, WaveCoord
import numpy as np
from copy import deepcopy
from astropy.io import fits
from mpdaf.obj import WCS as mpdaf_WCS
from mpdaf.obj import Cube, WaveCoord
from astropy.wcs import WCS
import numpy as np
from ioTools import read_in_data

from mpdaf.obj import Cube, WCS as MPDAF_WCS, WaveCoord
import numpy as np
import astropy.units as u
from astropy.io import fits

def redshift_wavelength_axis(filename, z_obs, z_sim,
                             pixscale_arcsec=None,         # float or (x,y); used only if header lacks spatial WCS
                             keep_ra_negative=True):       # conventional RA CDELT1 < 0
    """
    Redshift the wavelength axis of a cube from z_obs to z_sim and return a new MPDAF Cube.
    Spatial WCS is preserved from the input when available; otherwise you must pass pixscale_arcsec.

    Parameters
    ----------
    filename : str
        Path to the FITS cube.
    z_obs : float
        Original redshift of the source.
    z_sim : float
        New redshift to shift the cube to.
    pixscale_arcsec : float or (float,float), optional
        Spatial pixel scale (arcsec/pix). Used ONLY if the input header lacks spatial WCS.
        If a single float, it is applied to both axes.
    keep_ra_negative : bool
        If True, enforce negative CDELT1 for RA (standard FITS convention).

    Returns
    -------
    redshifted_cube : mpdaf.obj.Cube
    lam_new : np.ndarray
    """

    # --- Load data and header via helper ---
    lam_old, data, *rest = read_in_data(filename)

    header = None
    var = None
    if len(rest) == 2:
        var, header = rest
    elif len(rest) == 1:
        header = rest[0]
    else:
        header = fits.Header()

    # --- Redshift wavelength axis (units preserved relative to lam_old) ---
    lam_new = lam_old * (1.0 + z_sim) / (1.0 + z_obs)

    # Determine wave units from header if present, else default to Angstrom
    wave_unit = (header.get('CUNIT3') or header.get('WCSDIM3') or 'Angstrom')
    # MPDAF WaveCoord wants a unit string like 'Angstrom', 'nm', 'um'
    # We assume lam_old already in `wave_unit`
    wave = WaveCoord(
        crval=lam_new[0],
        cdelt=np.median(np.diff(lam_new)),
        crpix=1,
        cunit=wave_unit,
        ctype='WAVE'
    )

    # --- Build spatial WCS ---
    # If header has a valid spatial WCS, use it as-is (keeps CD/PC matrix, rotation, etc.)
    has_spatial = (
        ('CDELT1' in header or 'CD1_1' in header or 'PC1_1' in header) and
        ('CDELT2' in header or 'CD2_2' in header or 'PC2_2' in header)
    )

    if has_spatial:
        # Let MPDAF parse the spatial part from the header
        wcs_spatial = MPDAF_WCS(header)
    else:
        # Need user-provided pixel scale
        if pixscale_arcsec is None:
            raise ValueError(
                "Input header has no spatial WCS (no CDELT/CD/PC keywords). "
                "Provide pixscale_arcsec (float or (x,y)) to build a WCS."
            )

        if isinstance(pixscale_arcsec, (tuple, list, np.ndarray)):
            px_x_arcsec, px_y_arcsec = float(pixscale_arcsec[0]), float(pixscale_arcsec[1])
        else:
            px_x_arcsec = px_y_arcsec = float(pixscale_arcsec)

        # FITS convention: RA axis is axis 1 (x), DEC is axis 2 (y)
        cdelt1_deg = -(px_x_arcsec / 3600.0) if keep_ra_negative else (px_x_arcsec / 3600.0)
        cdelt2_deg =  (px_y_arcsec / 3600.0)

        # Reasonable defaults for missing world values
        crval1 = header.get('CRVAL1', 0.0)
        crval2 = header.get('CRVAL2', 0.0)
        crpix1 = header.get('CRPIX1', 1.0)
        crpix2 = header.get('CRPIX2', 1.0)
        ctype1 = header.get('CTYPE1', 'RA---TAN')
        ctype2 = header.get('CTYPE2', 'DEC--TAN')

        wcs_spatial = MPDAF_WCS(
            crval=(crval1, crval2),
            cdelt=(cdelt1_deg, cdelt2_deg),
            crpix=(crpix1, crpix2),
            cunit=('deg', 'deg'),
            ctype=(ctype1, ctype2)
        )

    # --- Build the new MPDAF cube with explicit WCS and Wave ---
    redshifted_cube = Cube(data=data, var=var, wcs=wcs_spatial, wave=wave)

    # --- Update a few informative header keywords (do NOT overwrite WCS again) ---
    hdr = redshifted_cube.data_header
    hdr['REDSHIFT'] = (z_sim, 'Simulated redshift applied to wavelength axis')

    return redshifted_cube, lam_new



def check_redshifted_cube(redshifted_cube):
    print("\n--- Redshifted Cube Check ---")
    print(f"Cube shape (λ, y, x): {redshifted_cube.data.shape}")

    wcs_obj = getattr(redshifted_cube, 'wcs', None)

    if wcs_obj is None:
        print("WCS is None")
    else:
        # Try to get CDELT, CRVAL, CRPIX from WCS header
        hdr = redshifted_cube.data_header
        for i, ax in enumerate(['x', 'y', 'λ']):
            crval = hdr.get(f'CRVAL{i+1}', None)
            crpix = hdr.get(f'CRPIX{i+1}', None)
            cdelt = hdr.get(f'CDELT{i+1}', None)
            cunit = hdr.get(f'CUNIT{i+1}', None)
            print(f"{ax.upper()}: CRVAL={crval}, CRPIX={crpix}, CDELT={cdelt}, CUNIT={cunit}")

    dx = getattr(redshifted_cube, 'dx_arcsec', None)
    dy = getattr(redshifted_cube, 'dy_arcsec', None)
    print(f"Stored pixel scales: dx_arcsec={dx}, dy_arcsec={dy}")

    if redshifted_cube.data.shape[1] < 2 or redshifted_cube.data.shape[2] < 2:
        print("Warning: Spatial dimensions are very small. Check WCS / CDELT values!")



if __name__ == "__main__":

    # Set redshift parameters


    # Load a test cube

    galaxy_name = "CGCG453"
    file_path = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/cgcg453_red_mosaic.fits"
    z_obs = 0.025
    z_sim = 3
    

    # galaxy_name = "IRAS08"
    # file_path = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/IRAS08/IRAS08_combined_final_metacube.fits"
    # z_obs = 0.019 
    # z_sim = 3

    output_path = f"/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/Output_cubes/{galaxy_name}/{galaxy_name}_redshifted_cube_{z_sim}.fits"
    
    lam_old, data, _ = read_in_data(file_path)


    # Run redshift function
    redshifted_cube, lam_new = redshift_wavelength_axis(file_path, z_obs, z_sim)

    # # Print wavelength range before/after
    # print(f"Original λ range: {lam_old[0]:.2f} – {lam_old[-1]:.2f}")
    # print(f"Redshifted λ range: {lam_new[0]:.2f} – {lam_new[-1]:.2f}")
    # check_redshifted_cube(redshifted_cube)
    # print(f"--- Redshifted Cube Check ---")
    # print(f"Cube shape (λ, y, x): {redshifted_cube.shape}")
    # print("Min flux:", np.nanmin(data))
    # print("Max flux:", np.nanmax(data))

    # hdr = redshifted_cube.data_header
    # print(f"X: CRVAL={hdr.get('CRVAL1')}, CRPIX={hdr.get('CRPIX1')}, "
    #     f"CDELT={hdr.get('CDELT1')}, CUNIT={hdr.get('CUNIT1')}")
    # print(f"Y: CRVAL={hdr.get('CRVAL2')}, CRPIX={hdr.get('CRPIX2')}, "
    #     f"CDELT={hdr.get('CDELT2')}, CUNIT={hdr.get('CUNIT2')}")
    # print(f"Λ: CRVAL={hdr.get('CRVAL3')}, CRPIX={hdr.get('CRPIX3')}, "
    #     f"CDELT={hdr.get('CDELT3')}, CUNIT={hdr.get('CUNIT3')}")



    # Plot spectrum before and after
    y, x = 12, 17  # Pick a spaxel
    original_spectrum = data[:, y, x]
    redshifted_spectrum = redshifted_cube.data[:, y, x]

    plt.figure(figsize=(10, 4))
    plt.plot(lam_old, original_spectrum, label='Original', alpha=0.7)
    plt.plot(lam_new, redshifted_spectrum, label='Redshifted', alpha=0.7)
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux')
    plt.title('Spectrum Before and After Redshifting')
    plt.legend()
    plt.tight_layout()
    plt.show()

    #  write redshifted cube to file
    redshifted_cube.write(output_path)
    print(f"redshifted cube has been stored in: {output_path}")
    data = fits.getdata(output_path)
    print("Cube shape:", data.shape)
    print("Min flux:", np.nanmin(data))
    print("Max flux:", np.nanmax(data))

