# CubeShift
A Python code to artificially redshift IFU data

==== Description ====
This project provides a Python pipeline for simulating how low-redshift IFU (Integral Field Unit) data cubes would appear if observed with different telescopes at higher redshifts. The code supports spatial resampling, resolution degradation, luminosity scaling, and PSF convolution to replicate the observational characteristics of instruments such as JWST NIRCam.

The pipeline is modular and designed for easy extension to other telescopes by adding new specifications to the Telescope class.

=== Requirements ===

Python 3.9 or higher

numpy

scipy

astropy

mpdaf

matplotlib

It is recommended to use a virtual environment. 
You can install the required packages using:

pip install -r requirements.txt

==== Usage ====
The entire redshifting pipeline is applied via the simulate_obs function in main.py. Example workflow:

Define the telescope instrument. Either use an existing one from the telescope dictionary in binData, or create your own telescope using the Telescope class. 

Run the simulation on your IFU data cube at the desired redshift by calling simulate_observation with 
    1. the file path to the IFU cube
    2. the name of the telescope you wish to simulate (formatted as dictionary key)
    3. the observed redshift
    4. the simulated redshift
    5. source telescope
    6. target telescope
    7. optional: galaxy name (for file structure purposes)
    8. optional: check numbers (boolean)
    9. optional: return the file path of the final cube (boolean)


for convienince, this program includes a helper function that converts quickly from KECK KCWI to JWST NIRSpec by presetting many of the parameters, including telescope types, and applies milky way extinction correction in the correct order. It may be helpful to create your own version of this template if you are using the same source and target telescopes for a variety of different galaxies.

Template

    1. Set source and target telescope
    2. use preRedshiftExtCor function from extinctionCorrection.py to apply the milkyway correction the cube
    3. then, apply the main redshifting pipeline using simulate_observation to the cube
    4. finally, use postPipelineExtCor to undo the mikly way correction 
    5. save the cube

Applied template:

    test_Keck_to_JWST_full_pipeline(file_path, z_obs, z_sim, galaxy_name):
        # 1
        source_telescope = telescope_specs["Keck_KCWI"]
        target_telescope = telescope_specs["JWST_NIRSpec"]
        
        # 2
        ext_corr_cube = preRedshiftExtCor(file_path)
        
        # 3
        final_pipeline_cube, final_pipeline_cube_path = simulate_observation(ext_corr_cube, "JWST_NIRSpec", z_obs, z_sim, source_telescope, target_telescope, galaxy_name = galaxy_name, return_final_cube_path=True)
        
        # 4
        ext_inverted_cube = postPipelineExtCor(final_pipeline_cube_path)
        
        # 5
        final_output_path_ext_invert = f"/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/Output_cubes/{galaxy_name}/z_{z_sim}_final_pipeline_with_ext_{galaxy_name}_cube.fits"

        ext_inverted_cube.write(final_output_path_ext_invert)
        print(" Final cube saved to: ", final_output_path_ext_invert)



then, you can quickly define the essential information from you galaxy.
for example:

    galaxy_name = "CGCG453"
    file_path = "/Users/ ... /cgcg453_red_mosaic.fits" # your file path here 
    z_obs = 0.025  # Original observed redshift
    z_sim = 3 # Simulated redshift

and call test_Keck_to_JWST_full_pipeline(file_path, z_obs, z_sim, galaxy_name).
