"""
Diagnostic RGB image of a galaxy cube.

Maps:
    - Red channel to Hβ emission (4861 Å),
    - Green channel to [O III] emission (5007 Å),
    - Blue channel to a stellar continuum region (~5300 Å by default),
all shifted to the galaxy's observed frame using its redshift (z).

Narrowband images centred on each wavelength region are extracted with MPDAF,
each channel is normalised, and they are combined into a single RGB image.
Optionally the spectrum at the brightest [O III] pixel is plotted to show
which spectral regions were used.

(Merged from the legacy ``diagnostic_rbg.py`` and ``rgb_tools.py``.)
"""

# put the scaling in log

import numpy as np
import matplotlib.pyplot as plt

def make_rgb_hbeta_oiii_continuum(cube, z, width=5, continuum_rest=5300.0,
                                  continuum_fallback=True, plot_spectrum=True):
    """
    Create an RGB image from an MPDAF cube using:
    - R: Hβ (4861 Å)
    - G: [O III] (5007 Å)
    - B: Stellar continuum region (``continuum_rest``, default 5300 Å)

    Parameters
    ----------
    cube : mpdaf.obj.Cube
        Input cube.
    z : float
        Redshift of the galaxy.
    width : float
        Half-width (in Å) of the narrowband filter.
    continuum_rest : float
        Rest-frame centre (Å) of the continuum (blue) band. The legacy
        ``rgb_tools.py`` version used 5500 Å.
    continuum_fallback : bool
        If True and the redshifted continuum centre is beyond the cube's red
        end, use the middle of the cube's wavelength range instead. (The legacy
        ``rgb_tools.py`` version had no fallback.)
    plot_spectrum : bool
        If True, also plot the spectrum at the brightest [O III] pixel with the
        three bands highlighted. (The legacy ``rgb_tools.py`` version did not.)

    Returns
    -------
    rgb_image : numpy.ndarray
        (ny, nx, 3) RGB image with each channel normalised to [0, 1].
    """
    # Observed wavelengths
    hbeta_obs = 4861 * (1 + z)
    oiii_obs = 5007 * (1 + z)
    wave_min = cube.wave.coord()[0]
    wave_max = cube.wave.coord()[-1]

    # Try using a clean region that is definitely within bounds
    # cover wider region in contunuum (+-100 A)

    continuum_center = continuum_rest * (1 + z)
    if continuum_fallback and not (continuum_center < wave_max):
        continuum_center = (wave_min + wave_max) / 2  # fallback


    # Print wavelengths used
    print("Using wavelength ranges:")
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

    if not plot_spectrum:
        return rgb_image

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
