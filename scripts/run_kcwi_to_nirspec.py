"""Simulate a Keck/KCWI cube as seen by JWST/NIRSpec at a higher redshift.

Replaces ``main.test_Keck_to_JWST_full_pipeline``:

1. correct the input cube for Milky Way extinction (A_V at the observed frame),
2. run :func:`cubeshift.simulate_observation`,
3. put Milky Way extinction back for the simulated frame,
4. write the result.

Get A_V from https://ned.ipac.caltech.edu/extinction_calculator (see the note in
cubeshift.extinction about A_V vs band A_lambda).

Example (the original CGCG453 run)::

    python scripts/run_kcwi_to_nirspec.py cgcg453_red_mosaic.fits \
        --z-obs 0.025 --z-sim 3 --galaxy CGCG453 \
        --av-obs 0.385 --av-sim 0.072 --noise-scale 0.5 \
        --data-dir "MNU 2025" --output-dir "MNU 2025/Output_cubes"

Other galaxies used so far: UGC10099 (J155636_red_mosaic.fits, z_obs=0.035),
IRAS08 (IRAS08_combined_final_metacube.fits, z_obs=0.019).
"""
import argparse
import os

from cubeshift import simulate_observation
from cubeshift.extinction import milky_way_extinction_correction_cube


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cube", help="input KCWI FITS cube")
    p.add_argument("--z-obs", type=float, required=True, help="observed redshift")
    p.add_argument("--z-sim", type=float, required=True, help="simulated redshift")
    p.add_argument("--galaxy", default=None, help="galaxy name (output file names)")
    p.add_argument("--av-obs", type=float, default=None,
                   help="Milky Way A_V to remove from the input (skip if not given)")
    p.add_argument("--av-sim", type=float, default=None,
                   help="Milky Way A_V to re-apply after simulating (skip if not given)")
    p.add_argument("--noise-scale", type=float, default=1.0,
                   help="scale of the background noise used to pad to the NIRSpec FOV (CGCG453 used 0.5)")
    p.add_argument("--data-dir", default=None, help="JWST filter/dispersion files (or set CUBESHIFT_DATA_DIR)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--save-intermediates", action="store_true")
    a = p.parse_args(argv)

    cube = a.cube
    if a.av_obs is not None:
        cube = milky_way_extinction_correction_cube(cube, a.av_obs, undo=False)

    res = simulate_observation(cube, a.z_obs, a.z_sim, "Keck_KCWI", "JWST_NIRSpec",
                               data_dir=a.data_dir, output_dir=a.output_dir, galaxy_name=a.galaxy,
                               save_intermediates=a.save_intermediates, noise_scale=a.noise_scale)
    final = res.cube
    if a.av_sim is not None:
        final = milky_way_extinction_correction_cube(final, a.av_sim, undo=True)

    out_dir = os.path.join(a.output_dir, a.galaxy) if a.galaxy else a.output_dir
    os.makedirs(out_dir, exist_ok=True)
    name = f"z_{a.z_sim}_final_pipeline_with_ext_{a.galaxy}_cube.fits" if a.galaxy else f"z_{a.z_sim}_final_cube.fits"
    out = os.path.join(out_dir, name)
    final.write(out)
    print(f"Filter {res.filter_name}, disperser {res.disperser}, bin factors {res.bin_factors}")
    print(f"Final cube saved to: {out}")


if __name__ == "__main__":
    main()
