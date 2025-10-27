
from mpdaf.obj import Cube
from astropy import units as u
import matplotlib.pyplot as plt
from matchSpectral import precomputed_match_spectral_resolution_variable_kernel, resample_spectral_axis, apply_dispersion_pipeline
from zWavelengths import redshift_wavelength_axis
from binData import resample_cube_to_telescope_and_redshift, bin_cube, get_spaxel_area, Telescope, telescope_specs
from simulateObs import scale_luminosity_for_redshift, convolve_to_match_psf, rescale_flux
from reprojectBinData import reproject_cube_preserve_wcs
import numpy as np
from astropy.wcs import WCS as AstropyWCS
from astropy.wcs import WCS
from cropCube import auto_crop_cube, trim_empty_edges
from applyFilter import apply_transmission_filter
from scipy.ndimage import gaussian_filter1d
from mpdaf.obj import Image, plot_rgb
from astropy.wcs.utils import proj_plane_pixel_scales
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.patches import Patch
from hBetaHgammaRatio import plot_hbeta_hgamma_ratio, plot_hbeta_hgamma_ratio_amp, plot_hbeta_hgamma_ratio_amp_soft
from astropy.io import fits
from ioTools import read_in_data
from oldVersions import old_redshift_wavelength_axis
from extinctionCorrection import preRedshiftExtCor, postPipelineExtCor



# The main pipeline function:

def simulate_observation(cube_file_name, telescope_name, z_obs, z_sim, source_telescope, target_telescope, galaxy_name=None, check_numbers=False, return_final_cube_path=False):
    """
    Simulates how a cube would appear if observed with a different telescope at a different redshift.
    Pipeline:
        1. Redshift wavelength axis
        2. Spatially rebin to match resolution
        3. Scale luminosity for redshift effects
        4. Convolve spatially to match PSF
        5. Match spectral resolution (convolve in lambda)
        6. Match spatial sampling (reproject to new pixel grid) - REMOVED
        7. Match spectral sampling (resample lambda axis)

    Parameters
    ----------
    cube : mpdaf.obj.Cube
        Original data cube (e.g. from Keck/KCWI).
    telescope_name : str
        Key to look up specs from the telescope_specs dictionary.
    z_obs : float, optional
        Original redshift. If None, assumed 0.

    Returns
    -------
    cube_sim : mpdaf.obj.Cube
        The transformed cube simulating the target telescope's observation.
    """
    z = z_sim # for file naming 
    telescope = telescope_specs[telescope_name]
    # test galaxy used to verify calculations when check_numbers = True
    TEST_FILE = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/Output_cubes/gauss_test_cube.fits"  
    TEST_CUBE = Cube(TEST_FILE)
    print(f"Simulating observation as seen by {telescope.name} at z = {z_sim}")

    # STEP 1: Redshift wavelength axis
    redshifted_cube, lam_new = redshift_wavelength_axis(cube_file_name, z_obs, z_sim)
    print("WCS after redshifting:", redshifted_cube.wcs)
    print("Cube shape:", redshifted_cube.data.shape)

    if check_numbers:
  
        expected_wavelengths = [7425.7, 7574.3]

        # Run the redshift function on the known test case
        test_redshifted_cube, test_lam = redshift_wavelength_axis(TEST_FILE, 0.01, 0.5)

        # Print what we expect vs. what we got
        print("\n[Redshift Function Numbers Check (using gauss_test_cube.fits)]")
        print("Expected wavelengths:", expected_wavelengths)
        print("Actual wavelengths:  ", [round(test_lam[0], 1), round(test_lam[-1], 1)])

        # Confirm match
        if (round(test_lam[0], 1) == expected_wavelengths[0] and 
            round(test_lam[-1], 1) == expected_wavelengths[1]):
            print(" Redshift function passed the check.")
        else:
            print(" Redshift function failed the check.")
    
    output_path = f"/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/Output_cubes/z_{z}_redshifted.fits"
    if galaxy_name is not None:
        output_path = f"/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/Output_cubes/{galaxy_name}/{galaxy_name}_z_{z}_redshifted.fits"
    redshifted_cube.write(output_path)
    print(f"✔ Redshifted cube saved to: {output_path}")

    # STEP 2: Flux rescaling
    scaled_data = rescale_flux(redshifted_cube.data, z_obs, z_sim)
    redshifted_cube.data = scaled_data

    # STEP 3: Flux → Surface Brightness
    spaxel_area = get_spaxel_area(redshifted_cube)
    redshifted_cube.data = redshifted_cube.data / spaxel_area

    output_path = f"/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/Output_cubes/z_{z}_fluxscaled.fits"
    if galaxy_name is not None:
        output_path = f"/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/Output_cubes/{galaxy_name}/{galaxy_name}_z_{z}_fluxscaled.fits"
    redshifted_cube.write(output_path)
    print(f"✔ Flux-scaled (SB) cube saved to: {output_path}")


    # STEP 4: Apply transmission filter
    line_rest_wavelength = 5007  # [OIII]
    transmission_cube, trans_filter_name = apply_transmission_filter(redshifted_cube, z_sim, line_rest_wavelength)
    if check_numbers:
        test_transmission_cube, _ = apply_transmission_filter(test_redshifted_cube, 0.5, line_rest_wavelength)


    # STEP 5: Rebin spatially
    cube_resampled, bin_factors, wave_step, dx_arcsec, dy_arcsec = resample_cube_to_telescope_and_redshift(
        transmission_cube,
        target_telescope,
        z_obs,
        z_sim
    )

    print(f" Cube resampled wave (should not be none): {cube_resampled.wave}")  # Should not be None
    print("Wave shape: ", cube_resampled.wave.shape)

    if check_numbers:
        expected_f_ang = 0.03
        expected_s = 0.01  # Need to redo this math for the redshifted and transmission filtered cube

        # Run the binning function on the known test case
        test_cube_resampled, bin_factors, wave_step, test_f_ang, test_s_x, test_s_y = resample_cube_to_telescope_and_redshift(
            test_transmission_cube,
            target_telescope,
            0.01,
            0.5,
            trim=True,
            check_numbers=True
        )

        # Safely unpack if values are tuples
        def _as_float(val):
            if isinstance(val, tuple):
                return float(val[0])
            return float(val)

        sx_val = _as_float(test_s_x)
        sy_val = _as_float(test_s_y)

        print("\n[Binning Function Numbers Check (using gauss_test_cube.fits)]")
        print("Expected F ang:", expected_f_ang)
        print("Actual F ang:  ", round(test_f_ang, 2))

        print(" Expected pixel resampling:", expected_s) 
        print(f" Actual pixel resampling (x,y) : ({round(sx_val, 2)}, {round(sy_val, 2)})")

        # Confirm match
        if (round(test_f_ang, 2) == expected_f_ang and 
            round(sx_val, 2) == expected_s and
            round(sy_val, 2) == expected_s):
            print(" Binning function passed the check.")
        else:
            print(" Binning function failed the check.")


    # STEP 6: Surface Brightness → Flux (using new pixel area)
    new_spaxel_area = get_spaxel_area(cube_resampled)
    cube_resampled.data = cube_resampled.data * new_spaxel_area

    output_path = f"/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/Output_cubes/z_{z}_rebinned_fluxscaled.fits"
    if galaxy_name is not None:
        output_path = f"/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/Output_cubes/{galaxy_name}/{galaxy_name}_z_{z}_rebinned_fluxscaled.fits"
    cube_resampled.write(output_path)
    print(f"✔ Rebinned & flux-scaled cube saved to: {output_path}")


    # STEP 7: Convolve spatially to match target PSF
    cube_psf = convolve_to_match_psf(
        cube_resampled,
        dx_arcsec, dy_arcsec,
        fwhm_target_arcsec=telescope.spatial_fwhm,
        z_obs=z_obs,
        z_sim=z_sim
    )

    print(f" Cube psf wave (should not be none):", cube_psf.wave)  # Should not be None
    print(cube_psf.wave.shape)

    output_path = f"/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/Output_cubes/z_{z}_{trans_filter_name}_psf.fits"
    if galaxy_name is not None:
        output_path = f"/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/Output_cubes/{galaxy_name}/{galaxy_name}_z_{z}_{trans_filter_name}_psf.fits"
    cube_psf.write(output_path)
    print(f"✔ PSF convolved cube saved to: {output_path}")

    # STEP 8: Apply dispersion filter and spectral resolution matching

    blurred_cube, best_disperser = apply_dispersion_pipeline(cube_psf, z_obs, z_sim)

    # Removed unecessary reproject step and fixed binning 

    # Step 7 
    telescope = telescope_specs[telescope_name]  # e.g., "JWST_NIRCam"
    R = telescope.spectral_resolution           
    lam_center = np.median(blurred_cube.wave.coord())
    bin_width = lam_center / R

    wavelengths = blurred_cube.wave.coord()
    start = wavelengths[0]
    end = wavelengths[-1]

    new_wave_axis = np.arange(start, end + bin_width, bin_width)
    cube = resample_spectral_axis(blurred_cube, new_wave_axis)

    output_path = f"/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/Output_cubes/z_{z}_{trans_filter_name}_{best_disperser}_lsf.fits"
    if galaxy_name is not None:
        output_path = f"/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/Output_cubes/{galaxy_name}/{galaxy_name}_z_{z}_{trans_filter_name}_{best_disperser}_lsf.fits"
    cube.write(output_path)
    print(f"✔ LSF spectral resolution matched cube saved to: {output_path}")

    if return_final_cube_path and galaxy_name is not None:
        return cube, output_path
    return cube, trans_filter_name, best_disperser

def test_Keck_to_JWST_full_pipeline(file_path, z_obs, z_sim, galaxy_name):
    

    source_telescope = telescope_specs["Keck_KCWI"]
    target_telescope = telescope_specs["JWST_NIRSpec"]
    ext_corr_cube = preRedshiftExtCor(file_path)
    #ext_corr_cube = Cube(file_path)
    final_pipeline_cube, final_pipeline_cube_path = simulate_observation(ext_corr_cube, "JWST_NIRSpec", z_obs, z_sim, source_telescope, target_telescope, galaxy_name = galaxy_name, return_final_cube_path=True)
    ext_inverted_cube = postPipelineExtCor(final_pipeline_cube_path)
    final_output_path_ext_invert = f"/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/Output_cubes/{galaxy_name}/z_{z_sim}_final_pipeline_with_ext_{galaxy_name}_cube.fits"
    ext_inverted_cube.write(final_output_path_ext_invert)
    print(" Final cube with extinction inverted saved to: ", final_output_path_ext_invert)
    print(f"Done!")



if __name__ == "__main__":

    #  run pipeline 

    galaxy_name = "CGCG453"
    file_path = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/cgcg453_red_mosaic.fits"
    z_obs = 0.025  # Original observed redshift
    z_sim = 3 # Simulated redshift

    # galaxy_name = "UGC10099"
    # file_path = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/UGC10099/J155636_red_mosaic.fits"
    # z_obs = 0.035 # from NED: 0.034713 (heliocentric)
    # z_sim = 3

    # galaxy_name = "IRAS08"
    # file_path = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/IRAS08/IRAS08_combined_final_metacube.fits"
    # z_obs = 0.019 
    # z_sim = 3
    
    # galaxy_name = "TEST"
    # file_path = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/Output_cubes/gauss_test_cube.fits"
    # z_obs = 0.01
    # z_sim = 0.5
    
    test_Keck_to_JWST_full_pipeline(file_path, z_obs, z_sim, galaxy_name)
    

    
 