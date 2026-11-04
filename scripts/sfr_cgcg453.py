"""Hβ-based SFR maps for a simulated (pipeline output) cube, e.g. CGCG453 at z=3.

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

Units: solar masses per year at the end.

Example
-------
python scripts/sfr_cgcg453.py Output_cubes/CGCG453/z_3_final_pipeline_with_ext_CGCG453_cube.fits \
    --z-sim 3 --av 0.058 --mw-output mw_corr_CGCG453_z_3.fits

The Galactic extinction ``--av`` can be looked up with the NED extinction
calculator (https://ned.ipac.caltech.edu/extinction_calculator) for the band
nearest the cube's central wavelength (e.g. UKIRT H, 1.6732 um, Av = 0.058 for
CGCG453 at z=3).
"""
import argparse

import numpy as np

from cubeshift.analysis.lines import compute_Ahbeta, measure_hbeta_flux, measure_hgamma_flux
from cubeshift.analysis.sfr import CALIBRATIONS, compute_sfr_from_Hbeta
from cubeshift.extinction import milky_way_extinction_correction_cube


def run_SFR_pipeline(cube_path, z_sim, av, galaxy_name="CGCG453", mw_output=None,
                     calibration="reichardtchu2025", debug=True):
    """Run steps 1-7 above and return ``(sfr_map, total_sfr, sfr_surface_density)``."""
    # Step 1: MW correction
    mw_corr_cube = milky_way_extinction_correction_cube(
        cube_path, av, undo=False, output_filename=mw_output
    )

    lam = mw_corr_cube.wave.coord()
    print("Cube wavelength coverage:", lam.min(), "-", lam.max())

    # Step 2 & 3: Fit absorption & measure Hβ
    print("[DEBUG] Starting Hβ flux measurement...")
    hbeta_flux_map = measure_hbeta_flux(mw_corr_cube, z_sim, debug=debug, debug_coords=(17, 10))
    print(f"[DEBUG] Hβ map stats: min={np.nanmin(hbeta_flux_map):.3e}, "
        f"max={np.nanmax(hbeta_flux_map):.3e}, "
        f"nans={np.isnan(hbeta_flux_map).sum()}")

    # Step 4: Measure Hγ
    print("[DEBUG] Starting Hγ flux measurement...")
    hgamma_flux_map = measure_hgamma_flux(mw_corr_cube, z_sim, debug=debug, debug_coords=(24, 12))
    print(f"[DEBUG] Hγ map stats: min={np.nanmin(hgamma_flux_map):.3e}, "
        f"max={np.nanmax(hgamma_flux_map):.3e}, "
        f"nans={np.isnan(hgamma_flux_map).sum()}")

    # Step 5: Internal extinction → A_Hβ
    print("[DEBUG] Computing A_Hβ...")
    Ahbeta_map = compute_Ahbeta(hbeta_flux_map, hgamma_flux_map)

    # Step 6: Compute SFR (Reichardt Chu 2025 by default)
    # Step 7: ΣSFR is returned as sfr_surface_density
    sfr_map, total_sfr, sfr_surface_density = compute_sfr_from_Hbeta(
        cube=mw_corr_cube,
        F_Hbeta=hbeta_flux_map,
        z=z_sim,
        A_Hbeta=Ahbeta_map,
        calibration=calibration,
        wcs=mw_corr_cube.wcs
    )

    print(f"Total SFR for {galaxy_name} at z={z_sim}: {total_sfr:.2f} Msol/yr")
    return sfr_map, total_sfr, sfr_surface_density


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cube", help="Pipeline output cube (FITS)")
    parser.add_argument("--z-sim", type=float, default=3.0, help="Redshift of the simulated cube (default 3)")
    parser.add_argument("--av", type=float, required=True,
                        help="Milky Way A_V (mag) towards the galaxy (NED 'Landolt V' row, or 3.1*E(B-V))")
    parser.add_argument("--galaxy-name", default="CGCG453")
    parser.add_argument("--mw-output", default=None, help="Optional path to save the MW-corrected cube")
    parser.add_argument("--calibration", default="reichardtchu2025", choices=sorted(CALIBRATIONS))
    parser.add_argument("--no-debug", action="store_true", help="Skip per-spaxel debug output/plots")
    args = parser.parse_args(argv)

    run_SFR_pipeline(args.cube, args.z_sim, args.av, galaxy_name=args.galaxy_name,
                     mw_output=args.mw_output, calibration=args.calibration,
                     debug=not args.no_debug)


if __name__ == "__main__":
    main()
