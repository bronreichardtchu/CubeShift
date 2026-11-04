"""RGB maps with integrated spectra and highlighted bands for a cube at its
observed redshift and simulated at higher redshifts.

(Was the ``__main__`` block of the legacy ``plotSpectrum.py``.)

Example
-------
python scripts/plot_spectra_with_bands.py cgcg453_red_mosaic_fresh.fits \
    --z-obs 0.025 --redshifts 3 4 5
"""
import argparse

from mpdaf.obj import Cube

from cubeshift.instruments import TELESCOPES
from cubeshift.pipeline import simulate_observation
from cubeshift.plotting.spectra import plot_multiple_spectra_with_bands


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cube", help="Original cube (e.g. KCWI mosaic, FITS)")
    parser.add_argument("--z-obs", type=float, default=0.025, help="Observed redshift of the real galaxy")
    parser.add_argument("--redshifts", type=float, nargs="+", default=[3.0, 4.0, 5.0],
                        help="Simulated redshifts (up to 3; the original is always shown first)")
    parser.add_argument("--source-telescope", default="Keck_KCWI", choices=sorted(TELESCOPES))
    parser.add_argument("--target-telescope", default="JWST_NIRSpec", choices=sorted(TELESCOPES))
    parser.add_argument("--data-dir", default=None, help="Passed to simulate_observation")
    parser.add_argument("--output-dir", default=None, help="Passed to simulate_observation")
    parser.add_argument("--show-filter-bands", action="store_true",
                        help="Shade NIRSpec filter/disperser ranges on the spectra")
    args = parser.parse_args(argv)

    cube = Cube(args.cube)
    z_obs = args.z_obs
    redshifts = [z_obs] + list(args.redshifts)[:3]
    source_telescope = TELESCOPES[args.source_telescope]
    target_telescope = TELESCOPES[args.target_telescope]

    filter_info = {}
    cubes = []

    for z in redshifts:
        if z == z_obs:
            cube_z = cube
            trans_filter_name = None
            disp_filter_name = None
        else:
            result = simulate_observation(
                cube, z_obs, z, source_telescope, target_telescope,
                data_dir=args.data_dir, output_dir=args.output_dir,
            )
            cube_z = result.cube
            trans_filter_name = result.filter_name
            disp_filter = result.disperser
            disp_filter_name = disp_filter["name"] if isinstance(disp_filter, dict) else str(disp_filter)

        cubes.append(cube_z)
        filter_info[z] = (trans_filter_name, disp_filter_name)
        print(f"[FILTER INFO] z={z}: trans={trans_filter_name}, disp={disp_filter_name}")

    plot_multiple_spectra_with_bands(cubes, redshifts, z_obs, filter_info=filter_info,
                                     display_filter_bands=args.show_filter_bands)


if __name__ == "__main__":
    main()
