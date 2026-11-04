"""
Script: diagnostic_rbg.py

Description:
    This script generates an RGB image from a spectral cube of a galaxy by mapping:
    - Red channel to Hβ emission (4861 Å),
    - Green channel to [O III] emission (5007 Å),
    - Blue channel to a stellar continuum region (~5300 Å),
    all shifted to the galaxy's observed frame using its redshift (z).

    It uses the MPDAF Cube object to extract narrowband images centered on each wavelength region,
    normalizes each channel, and combines them into a single RGB image. It also plots the spectrum
    at the brightest [O III] pixel to show which spectral regions were used.

Dependencies:
    - numpy
    - matplotlib
    - mpdaf
    - CGCG455mpdaf (custom module containing telescope settings and plotting utilities)

Usage:
    Modify the `file_path` and `z` values under `if __name__ == "__main__"` to run on your own cube.

"""

# put the scaling in log

import numpy as np
import matplotlib.pyplot as plt
from mpdaf.obj import Cube
from CGCG455mpdaf import (
    Telescope,
    plotSpectrumOfPixel,
    plotSpectrumWithEmissionLines,
    makeBroadband,
    makeNarrowbandImage,
    get_brightest_spot_gauss
)

def make_rgb_hbeta_oiii_continuum(cube, z, width=5):
    """
    Create an RGB image from an MPDAF cube using:
    - R: Hβ (4861 Å)
    - G: [O III] (5007 Å)
    - B: Stellar continuum region (~5500 Å)

    Parameters:
        cube : mpdaf.obj.Cube
        z : float
            Redshift of the galaxy
        width : float
            Half-width (in Å) of the narrowband filter
    """
    # Observed wavelengths
    hbeta_obs = 4861 * (1 + z)
    oiii_obs = 5007 * (1 + z)
    wave_min = cube.wave.coord()[0]
    wave_max = cube.wave.coord()[-1]

    # Try using a clean region that is definitely within bounds
    # cover wider region in contunuum (+-100 A)

    if (5300 * (1 + z) < wave_max):
        continuum_center = 5300 * (1 + z)
    else:
        continuum_center = (wave_min + wave_max) / 2  # fallback


    # Print wavelengths used
    print(f"Using wavelength ranges:")
    print(f"  Hβ (R):       {hbeta_obs - width:.1f}–{hbeta_obs + width:.1f} Å")
    print(f"  [O III] (G):  {oiii_obs - width:.1f}–{oiii_obs + width:.1f} Å")
    print(f"  Continuum (B):{continuum_center - width:.1f}–{continuum_center + width:.1f} Å")

    # Extract images
    try:
        r = cube.get_image(wave=(hbeta_obs - width, hbeta_obs + width)).data
        g = cube.get_image(wave=(oiii_obs - width, oiii_obs + width)).data
        b = cube.get_image(wave=(continuum_center - width, continuum_center + width)).data
    except Exception as e:
        print("Error extracting one or more narrowband images.")
        raise e

    # Normalize each channel
    def normalize(img):
        p1, p99 = np.nanpercentile(img, [1, 99])
        return np.clip((img - p1) / (p99 - p1 + 1e-8), 0, 1)

    r_norm = normalize(r)
    g_norm = normalize(g)
    b_norm = normalize(b)

    # Stack into RGB image
    rgb_image = np.dstack((r_norm, g_norm, b_norm))

    # Plot RGB
    plt.figure(figsize=(6, 6))
    plt.imshow(rgb_image, origin='lower')
    plt.title("RGB: Hβ (Red), [O III] (Green), Continuum (Blue)")
    plt.axis('off')
    plt.show()

    # === Plot spectrum from brightest pixel ===
    brightest_coords = np.unravel_index(np.nanargmax(g), g.shape)  # Use [O III] map for brightest
    spectrum = cube[:, brightest_coords[0], brightest_coords[1]]
    wave = spectrum.wave.coord()
    flux = spectrum.data

    plt.figure(figsize=(8, 4))
    plt.plot(wave, flux, color='k', lw=1.2)
    plt.axvspan(hbeta_obs - width, hbeta_obs + width, color='red', alpha=0.3, label='Hβ')
    plt.axvspan(oiii_obs - width, oiii_obs + width, color='green', alpha=0.3, label='[O III]')
    plt.axvspan(continuum_center - width, continuum_center + width, color='blue', alpha=0.3, label='Continuum')
    plt.xlabel("Wavelength (Å)")
    plt.ylabel("Flux")
    plt.title("Spectrum at Brightest [O III] Pixel")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return rgb_image

if __name__ == "__main__":
    # Example usage
    z = 0.025
    file_path = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/cgcg453_red_mosaic.fits"
    cube = Cube(file_path)
    make_rgb_hbeta_oiii_continuum(cube, z)
