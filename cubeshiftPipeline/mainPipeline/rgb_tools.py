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
    - B: Stellar continuum region (e.g., ~5500–5550 Å)
    
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

    # Stellar continuum: choose a relatively clean region (adjust if needed)
    continuum_center = 5500 * (1 + z)

    # Narrowband integration
    r = cube.get_image(wave=(hbeta_obs - width, hbeta_obs + width)).data
    g = cube.get_image(wave=(oiii_obs - width, oiii_obs + width)).data
    b = cube.get_image(wave=(continuum_center - width, continuum_center + width)).data

    # Normalize using 1st–99th percentile stretch
    def normalize(img):
        p1, p99 = np.nanpercentile(img, [1, 99])
        return np.clip((img - p1) / (p99 - p1 + 1e-8), 0, 1)

    r_norm = normalize(r)
    g_norm = normalize(g)
    b_norm = normalize(b)

    # Stack channels
    rgb_image = np.dstack((r_norm, g_norm, b_norm))

    # Plot
    plt.figure(figsize=(6, 6))
    plt.imshow(rgb_image, origin='lower')
    plt.title("RGB: Hβ (Red), [O III] (Green), Continuum (Blue)")
    plt.axis('off')
    plt.show()

    return rgb_image
