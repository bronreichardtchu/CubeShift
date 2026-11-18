"""Simulate an HST broadband image as seen by HST (same filter) at higher redshift(s).

Same filter, no K-correction (flat f_nu by default): the galaxy is dimmed, shrunk
to the angular scale of each z_sim, binned to the target pixels and convolved
to the target PSF.  Writes one FITS file per redshift, plus a quick-look PNG.

Example (NGC 300, ACS/WFC F555W)::

    python scripts/run_hst_image.py low8_f555w_drc.fits --z-obs 0.00048 --distance-mpc 2.0 \
        --z-sim 0.01 --z-sim 0.05 --z-sim 0.5 --z-sim 1 --z-sim 3 --galaxy NGC300 --output-dir out
"""
import argparse
import contextlib
import io
import os

import numpy as np

from cubeshift import read_hst_image, simulate_image_observation


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", help="HST drizzled image (_drz/_drc)")
    p.add_argument("--z-obs", type=float, required=True, help="observed redshift")
    p.add_argument("--z-sim", type=float, action="append", required=True, help="simulated redshift (repeatable)")
    p.add_argument("--distance-mpc", type=float, default=None,
                   help="measured distance of the galaxy in Mpc (recommended for nearby galaxies)")
    p.add_argument("--source-telescope", default="HST_ACS_WFC")
    p.add_argument("--target-telescope", default="HST_ACS_WFC")
    p.add_argument("--sed", choices=["flat_fnu", "flat_flambda"], default="flat_fnu")
    p.add_argument("--galaxy", default=None, help="name used in output file names")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args(argv)

    image = read_hst_image(a.image)
    total_in = np.ma.sum(image.data)
    print(f"{a.image}: {image.shape}, {image.data_header.get('FILTER1', '')} "
          f"{image.data_header.get('CSBUNIT', '')}, pixel {np.abs(image.wcs.get_step(unit='arcsec'))}\"")

    results = []
    for z in a.z_sim:
        ctx = contextlib.nullcontext() if a.verbose else contextlib.redirect_stdout(io.StringIO())
        with ctx:
            r = simulate_image_observation(image, a.z_obs, z, a.source_telescope, a.target_telescope,
                                           distance_obs_mpc=a.distance_mpc, sed=a.sed,
                                           output_dir=a.output_dir, galaxy_name=a.galaxy)
        results.append((z, r))
        print(f"z_sim={z}: shape {r.image.shape}, bin {r.bin_factors}, pixel {r.pixel_scale[0]:.4f}\", "
              f"D_A ratio {r.angular_scale_ratio:.3e}, dimming {r.dimming_factor:.3e}, "
              f"counts ratio {np.ma.sum(r.image.data) / total_in:.3e} -> {r.path}")

    if not a.no_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(results) + 1
        fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4))
        step = max(1, max(image.shape) // 800)
        axes[0].imshow(np.arcsinh(np.ma.filled(image.data[::step, ::step], np.nan)), origin="lower", cmap="gray")
        axes[0].set_title(f"input (z={a.z_obs})")
        for ax, (z, r) in zip(axes[1:], results):
            ax.imshow(np.ma.filled(r.image.data, np.nan), origin="lower", cmap="gray")
            ax.set_title(f"z={z}  ({r.image.shape[1]}x{r.image.shape[0]} px)")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        png = os.path.join(a.output_dir, f"{a.galaxy or 'image'}_quicklook.png")
        fig.savefig(png, dpi=120)
        print(f"Quick-look: {png}")


if __name__ == "__main__":
    main()
