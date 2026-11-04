"""Compare Hβ flux distributions of an original cube and a simulated cube.

(Was the ``__main__`` block of the legacy ``hBetaFlux.py``.)

Example
-------
python scripts/hbeta_histogram.py cgcg453_..._cube_cropped.fits \
    Output_cubes/CGCG453/CGCG453_z_3_f170lp_g235h_lsf.fits --z-obs 0.025 --z-sim 3
"""
import argparse

from cubeshift.analysis.flux_histogram import run_hbeta_histogram


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("original_cube", help="Cube at the observed redshift (FITS)")
    parser.add_argument("simulated_cube", help="Simulated cube at z_sim (FITS)")
    parser.add_argument("--z-obs", type=float, default=0.025)
    parser.add_argument("--z-sim", type=float, default=3.0)
    args = parser.parse_args(argv)

    run_hbeta_histogram(args.original_cube, args.simulated_cube, args.z_obs, args.z_sim)


if __name__ == "__main__":
    main()
