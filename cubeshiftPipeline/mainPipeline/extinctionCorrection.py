from zWavelengths import redshift_wavelength_axis
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

from dustmaps.sfd import SFDQuery
import requests
from bs4 import BeautifulSoup
import pandas as pd
import io, re

from mpdaf.obj import Cube, WaveCoord, WCS as MPDAF_WCS


NED_EXTINCT_URL = "https://ned.ipac.caltech.edu/cgi-bin/nph-calc_extinct"



def get_A_lambda_at_wavelength(ra, dec, wavelength_um):
    """
    Return interpolated A_lambda (mag) at a specific wavelength (microns).
    Also returns the dataframe of NED band info for inspection.
    """
    df = fetch_ned_extinction_table(ra, dec)
    # sort by wavelength
    df = df.sort_values("wavelength_um").reset_index(drop=True)

    lams = df["wavelength_um"].values
    Avals = df["A_lambda_mag"].values

    # If the requested wavelength is outside the table range, np.interp will
    # clamp to the endpoint values. Optionally raise a warning instead.
    A_interp = float(np.interp(wavelength_um, lams, Avals))
    return A_interp, df


def compute_A_lambda(lamdas, Av=0.2511, Rv=3.1):
    """
    Compute A_lambda extinction curve from Cardelli et al. (1989).
    lamdas must be in Angstroms.
    """
    lam_um = lamdas / 1e4
    y = lam_um**(-1) - 1.82
    a_x = (1.0 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 +
           0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7)
    b_x = (1.41338*y + 2.28305*y**2 + 1.07233*y**3 -
           5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7)
    return (a_x + b_x/Rv) * Av

def milky_way_extinction_correction(lamdas, data, Av=0.2511, undo=False):
    """
    Corrects for the extinction caused by light travelling through the dust and
    gas of the Milky Way, as described in Cardelli et al. 1989.

    Parameters
    ----------
    lamdas : :obj:'~numpy.ndarray'
        wavelength vector

    data : :obj:'~numpy.ndarray'
        3D cube of data

    Av : float 
        The extinction from the Milky Way, found using NED which uses the Schlafly
        & Finkbeiner (2011) recalibration of the Schlegel, Finkbeiner & Davis (1998)
        extinction map based on dust emission measured by COBE/DIRBE and IRAS/ISSA.
        Default is 0.2511, which is the value for IRAS 08339+6517.

    Returns
    -------
    data : :obj:'~numpy.ndarray'
        the data corrected for extinction
    """
    #convert lamdas from Angstroms into micrometers
    lamdas = lamdas/10000

    #define the equations from the paper
    y = lamdas**(-1) - 1.82
    a_x = 1.0 + 0.17699*y - 0.50447*(y**2) - 0.02427*(y**3) + 0.72085*(y**4) + 0.01979*(y**5) - 0.77530*(y**6) + 0.32999*(y**7)
    b_x = 1.41338*y + 2.28305*(y**2) + 1.07233*(y**3) - 5.38434*(y**4) - 0.62251*(y**5) + 5.30260*(y**6) - 2.09002*(y**7)

    #define the constants
    Rv = 3.1

    #find A(lambda)
    A_lam = (a_x + b_x/Rv)*Av
    #return A_lam

    #apply to the data
    # if 
    if undo:
        data = data / (10**(0.4*A_lam[:, None, None]))  # put extinction back
    else:
        data = (10**(0.4*A_lam[:, None, None])) * data  # correct extinction

    print(f" Extinction correction applied with Av of {Av} undo = {undo}")
    return data, A_lam




def milky_way_extinction_correction_cube(
    filename, Av=0.2511, undo=False,
    output_filename=None, return_a_lam=False
):
    """
    Apply Milky Way extinction correction (Cardelli et al. 1989) to an MPDAF Cube.

    Parameters
    ----------
    filename : str
        Path to FITS cube file.
    Av : float
        Milky Way extinction (mag).
    undo : bool
        If True, undo the extinction correction instead of applying it.
    output_filename : str, optional
        If provided, save the corrected cube here.

    Returns
    -------
    Cube
        Extinction-corrected MPDAF Cube with WCS and header preserved.
    """
    import numpy as np
    from mpdaf.obj import Cube

    # Load full cube with WCS + headers intact
    cube = Cube(filename)

    # Wavelength axis in Angstroms
    lam = cube.wave.coord()
    lam_microns = lam / 10000.0
    y = lam_microns**(-1) - 1.82

    # Cardelli extinction law
    a_x = (1.0 + 0.17699*y - 0.50447*(y**2) - 0.02427*(y**3) +
           0.72085*(y**4) + 0.01979*(y**5) - 0.77530*(y**6) +
           0.32999*(y**7))
    b_x = (1.41338*y + 2.28305*(y**2) + 1.07233*(y**3) -
           5.38434*(y**4) - 0.62251*(y**5) + 5.30260*(y**6) -
           2.09002*(y**7))
    Rv = 3.1
    A_lam = (a_x + b_x/Rv) * Av

    # Apply correction along spectral axis
    if undo:
        corrected_data = cube.data / (10**(0.4 * A_lam[:, None, None]))
    else:
        corrected_data = cube.data * (10**(0.4 * A_lam[:, None, None]))

    print(f"Extinction correction applied with Av={Av}, undo={undo}")

    # Create new Cube while preserving WCS + header
    corrected_cube = Cube(data=corrected_data,
                          wave=cube.wave,
                          var=cube.var,
                          mask=cube.mask,
                          wcs=cube.wcs,
                          copy=True)
    corrected_cube.primary_header = cube.primary_header.copy()
    corrected_cube.data_header = cube.data_header.copy()
    corrected_cube.data_header['EXTCORR'] = (not undo, 'Milky Way extinction correction applied')

    # Save if requested
    if output_filename is not None:
        corrected_cube.write(output_filename, overwrite=True)
        print(f"Saved corrected cube to {output_filename}")
    if return_a_lam:
        return corrected_cube, A_lam
    return corrected_cube






def preRedshiftExtCor(cube_path):
    
    print(f"Go to https://ned.ipac.caltech.edu/byname")
    print("Step 1: enter the galaxy name")
    print("Step 2: copy/paste the RA and Dec coordinates into the extinction calculator: https://ned.ipac.caltech.edu/extinction_calculator")
    print(f"Cube path is: {cube_path}")
    cube = Cube(cube_path)
    print(f"Cube path after making cube is: {cube_path}")
    lamdas = cube.wave.coord()    
    lam_central_um = lamdas.mean() / 1e4 
    print(f" Your central wavelength is {lam_central_um}")
    print(f" Step 3: Filter for => {lam_central_um} and =< {lam_central_um} in the Central Wavelength column ")



    # --- Step 1: Get Av from NED ---
    # Using Sloan g': 0.385 (central wavelength 0.4925)
    Av = float(input( "Step 4: Enter the Galactic Extinction (mag) number from the extinction calculator: "))

    # Step 2: 
    # Edited: DO THE CORRECTION
    cube_with_corr, A_lam = milky_way_extinction_correction_cube(cube_path, Av, undo=False, return_a_lam=True)
    cube_with_corr.write("/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/Output_cubes/cube_without_corr.fits")
    print(f" Milky Way correction at observed redshift: {A_lam}")
    return cube_with_corr

def postPipelineExtCor(final_pipeline_cube_path):
    final_pipeline_cube = Cube(final_pipeline_cube_path)
    lamdas_post = final_pipeline_cube.wave.coord() 
    lam_central_um = lamdas_post.mean() / 1e4 
    print(f" Your new central wavelength is {lam_central_um}")
    print(f" Filter for => {lam_central_um} and =< {lam_central_um} in the Central Wavelength column ")
    # Using UKIRT J: 0.072 (central wavelength 1.2483)
    Av_z = float(input( " Enter the new Galactic Extinction (mag) number from the extinction calculator: "))

    # Step 4: 
    # UNDO CORRECTION
    cube_without_corr = milky_way_extinction_correction_cube(final_pipeline_cube_path, Av_z, undo=True)
    return cube_without_corr



if __name__ == "__main__":
    
    cube_path = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/cgcg453_red_mosaic.fits"
    cube = Cube(cube_path)

    lamdas = cube.wave.coord()    
    data = cube.data     

    print("---------- Testing modular functions --------")
    cube_with_corr = preRedshiftExtCor(cube_path)
    final_pipeline_cube_path = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/Output_cubes/CGCG453/CGCG453_z_3_f170lp_g235h_lsf.fits"
    final_pipeline_cube = Cube(final_pipeline_cube_path)
    postPipelineExtCor(final_pipeline_cube)
    print(" --------- Modular function tests finished --------")



    print(f"Go to https://ned.ipac.caltech.edu/byname")
    print("Step 1: enter the galaxy name")
    print("Step 2: copy/paste the RA and Dec coordinates into the extinction calculator: https://ned.ipac.caltech.edu/extinction_calculator")
    ra_deg = 346.235578   
    dec_deg = 19.552296   
    lam_central_um = lamdas.mean() / 1e4 
    print(f" Your central wavelength is {lam_central_um}")
    print(f" Filter for => {lam_central_um} and =< {lam_central_um} in the Central Wavelength column ")



    # --- Step 1: Get Av from NED ---
    # Using Sloan g': 0.385 (central wavelength 0.4925)
    Av = float(input( " Enter the Galactic Extinction (mag) number from the extinction calculator: "))

    # Step 2: 
    # Edited: DO THE CORRECTION
    data_with_ext = milky_way_extinction_correction(lamdas, data, Av, undo=False)

    # Step 3: Redshift the cube
    # REDSHIFT
    # recalc central wavelength for redshifted cube
    # input new av
    redshifted_cube, lam_new = redshift_wavelength_axis(cube, 0.025, 1.5)
    lamdas_z = redshifted_cube.wave.coord()    
    data_z = redshifted_cube.data 
    lam_central_um = lamdas_z.mean() / 1e4 
    print(f" Your redshifted central wavelength is {lam_central_um}")
    print(f" Filter for => {lam_central_um} and =< {lam_central_um} in the Central Wavelength column ")
    # Using UKIRT J: 0.072 (central wavelength 1.2483)
    Av_z = float(input( " Enter the new Galactic Extinction (mag) number from the extinction calculator: "))


    # Step 4: 
    # UNDO CORRECTION
    data_final = milky_way_extinction_correction(lamdas_z, data_z, Av_z, undo=True)

    # --- CHECKS: spectra & extinction curve plots ---
    ny, nx = data.shape[1:]
    y0, x0 = ny // 2, nx // 2
    spec_orig = data[:, y0, x0]
    spec_with_ext = data_with_ext[:, y0, x0]
    spec_final = data_final[:, y0, x0]


    plt.figure(figsize=(10, 6))
    plt.plot(lamdas, spec_orig, label="Original")
    plt.plot(lamdas, spec_with_ext, label="Original with correction")
    plt.plot(lamdas_z, spec_final, label="Redshifted + un-corrected")
    plt.xlabel("Wavelength [Å]")
    plt.ylabel("Flux [cube units]")
    plt.legend()
    plt.title("Check extinction correction + redshift pipeline (central spaxel)")
    plt.show()

    A_lam = compute_A_lambda(lamdas, Av)
    plt.figure(figsize=(8, 5))
    plt.plot(lamdas, A_lam)
    plt.xlabel("Wavelength [Å]")
    plt.ylabel("A(λ) [mag]")
    plt.title("Milky Way Extinction Curve")
    plt.show()
