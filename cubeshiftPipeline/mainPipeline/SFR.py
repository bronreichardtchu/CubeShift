from fluxMeasurements import measure_hbeta_flux, measure_hgamma_flux, compute_Ahbeta, flux_to_Halpha_luminosity, compute_sfr_from_hbeta_flux, compute_sfr_from_Hbeta

from mpdaf.obj import Cube
from astropy import units as u
import matplotlib.pyplot as plt

from extinctionCorrection import preRedshiftExtCor
import numpy as np

def run_SFR_pipeline():

    """
        All steps:
        1. Correct for MW extinction
        2. Fit and subtract absorption around H𝛽
        3. Integrate over H𝛽 emission line to get flux
        4. Subtract continuum and integrate over H𝛾 to get flux
        5. Derive internal galaxy extinction using H𝛽/H𝛾 ratio (this gives us
        𝐴𝐻 𝛽 )
        6. Put into the equation below to get SFR, see Reichardt Chu et al.
        (2025) for values of the constants
        7. Divide by spaxel area to get ΣSFR
        8. Make maps for each galaxy at each redshift of the H𝛽 flux and
        extinction values
        9. Make violin plots of the H𝛽 flux, SFR and ΣSFR (and extinction
        maybe?) for each galaxy at each redshift
    """
    # units: solar masses per year at the end 

    galaxy_name = "CGCG453"
    z_obs = 0.025  # Original observed redshift
    z_sim = 3      # Simulated redshift

    final_output_path_ext_invert = (
        f"/Users/janev/Library/CloudStorage/"
        f"OneDrive-Queen'sUniversity/MNU 2025/Output_cubes/{galaxy_name}/"
        f"z_{z_sim}_final_pipeline_with_ext_{galaxy_name}_cube.fits"
    )

    # Load cube
    cube = Cube(final_output_path_ext_invert)

    # Step 1: MW correction 
    # (UKIRT H, 1.6732, Av = 0.058 for CGCCG)
    mw_corr_cube = preRedshiftExtCor(final_output_path_ext_invert)

    mw_corr_cube.write(f"/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/Output_cubes/{galaxy_name}/mw_corr_{galaxy_name}_z_{z_sim}.fits")

    lam = mw_corr_cube.wave.coord()
    print("Cube wavelength coverage:", lam.min(), "-", lam.max())

    # Step 2 & 3: Fit absorption & measure Hβ
    print("[DEBUG] Starting Hβ flux measurement...")
    hbeta_flux_map = measure_hbeta_flux(mw_corr_cube, z_sim, debug=True, debug_coords=(17, 10))
    print(f"[DEBUG] Hβ map stats: min={np.nanmin(hbeta_flux_map):.3e}, "
        f"max={np.nanmax(hbeta_flux_map):.3e}, "
        f"nans={np.isnan(hbeta_flux_map).sum()}")

    # Step 4: Measure Hγ
    print("[DEBUG] Starting Hγ flux measurement...")
    hgamma_flux_map = measure_hgamma_flux(mw_corr_cube, z_sim, debug=True, debug_coords=(24, 12))
    print(f"[DEBUG] Hγ map stats: min={np.nanmin(hgamma_flux_map):.3e}, "
        f"max={np.nanmax(hgamma_flux_map):.3e}, "
        f"nans={np.isnan(hgamma_flux_map).sum()}")

    # Step 5: Internal extinction → A_Hβ
    print("[DEBUG] Computing A_Hβ...")
    Ahbeta_map = compute_Ahbeta(hbeta_flux_map, hgamma_flux_map)

    # Step 6: Compute SFR (Reichardt Chu 2025 by default)
    sfr_map, total_sfr, sfr_surface_density = compute_sfr_from_Hbeta(
        cube=mw_corr_cube,
        F_Hbeta=hbeta_flux_map,
        z=z_sim,
        A_Hbeta=Ahbeta_map,
        calibration="reichardtchu2025",
        wcs=cube.wcs
    )

    print(f"Total SFR for {galaxy_name} at z={z_sim}: {total_sfr:.2f} Msol/yr")

    # Step 7: ΣSFR is already returned above as sfr_surface_density





if __name__ == "__main__":
    run_SFR_pipeline()




    
