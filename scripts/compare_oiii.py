"""Compare [O III] 5007 flux images of a cube at its observed redshift and
after simulating it at higher redshifts (2x2 grid).

Example
-------
python scripts/compare_oiii.py cgcg453_red_mosaic.fits --z-obs 0.025 --redshifts 2.5 3 4
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from mpdaf.obj import Cube
from scipy.ndimage import gaussian_filter1d

from cubeshift.instruments import TELESCOPES
from cubeshift.pipeline import simulate_observation


# === Image extraction function with redshifted width & peak centering ===
def extract_centered_oiii_image(cube, z, rest_wavelength=5007, rest_width=20):
    # Convert to observed frame
    lam_center = rest_wavelength * (1 + z)
    width = rest_width * (1 + z)

    # Collapse cube spatially to get 1D spectrum
    spectrum = cube.sum(axis=(1, 2))  # Wavelength axis is 0
    wave = cube.wave.coord()  # Wavelength array in Angstrom

    # Create mask around expected line center
    window = 100 * (1 + z)
    mask = (wave > lam_center - window) & (wave < lam_center + window)
    if np.sum(mask) < 5:
        raise ValueError(f"[O III] line likely outside cube for z = {z}")

    # Smooth the spectrum and locate line peak
    smooth_spec = gaussian_filter1d(spectrum.data[mask], sigma=3)
    lam_window = wave[mask]
    peak_idx = np.argmax(smooth_spec)
    lam_peak = lam_window[peak_idx]

    # Extract flux image centered on peak wavelength
    # (the legacy script passed (lam_peak, width) as (lmin, lmax), i.e. an empty/
    # inverted window; centre a window of full width `width` on the peak instead)
    img = cube.get_image((lam_peak - width / 2, lam_peak + width / 2))
    return img


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cube", help="Original cube (e.g. KCWI mosaic, FITS)")
    parser.add_argument("--z-obs", type=float, default=0.025, help="Observed redshift of the real galaxy")
    parser.add_argument("--redshifts", type=float, nargs="+", default=[2.5, 3.0, 4.0],
                        help="Simulated redshifts (up to 3; the original is always shown first)")
    parser.add_argument("--source-telescope", default="Keck_KCWI", choices=sorted(TELESCOPES))
    parser.add_argument("--target-telescope", default="JWST_NIRCam", choices=sorted(TELESCOPES))
    parser.add_argument("--data-dir", default=None, help="Passed to simulate_observation")
    parser.add_argument("--output-dir", default=None, help="Passed to simulate_observation")
    args = parser.parse_args(argv)

    cube = Cube(args.cube)
    z_obs = args.z_obs
    redshifts = [z_obs] + list(args.redshifts)[:3]
    source_telescope = TELESCOPES[args.source_telescope]
    target_telescope = TELESCOPES[args.target_telescope]

    # === Create 2x2 plot ===
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    im = None

    for i, z in enumerate(redshifts):
        ax = axes[i]

        try:
            if z == z_obs:
                cube_z = cube
            else:
                cube_z = simulate_observation(
                    cube, z_obs, z, source_telescope, target_telescope,
                    data_dir=args.data_dir, output_dir=args.output_dir,
                ).cube

            img = extract_centered_oiii_image(cube_z, z)
            im = ax.imshow(img.data, origin='lower', cmap='viridis',
                           vmin=0, vmax=np.nanpercentile(img.data, 99))
            ax.set_title(f'[O III] flux map at z = {z}')
        except Exception as e:
            ax.text(0.5, 0.5, f'Error at z={z}\n{str(e)}',
                    ha='center', va='center', fontsize=10)

        ax.axis('off')

    # Add a single colorbar if at least one image worked
    if im is not None:
        fig.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.02)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
