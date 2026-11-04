"""Quick-look plots of a cube: single-spaxel spectra, broadband/narrowband
images and simple emission masks."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def _convolve_to_resolution(spectrum, wavelengths, spectral_resolution, redshift=None):
    """
    Smooth a spectrum with a Gaussian whose sigma is lambda_mean / R (in Å).

    Replaces the legacy private ``Telescope.convolve_spectrum`` of this module
    (same maths: sigma in pixels = (mean(lambda) / R) / mean(dlambda)).

    Parameters
    ----------
    spectrum : numpy.ndarray
        1D flux array.
    wavelengths : numpy.ndarray
        1D wavelength array (Å).
    spectral_resolution : float
        Resolving power R.
    redshift : float, optional
        If given, wavelengths are first multiplied by (1 + redshift).

    Returns
    -------
    numpy.ndarray
        Smoothed spectrum.
    """
    if redshift is not None:
        wavelengths = wavelengths * (1 + redshift)

    lambda_mean = np.nanmean(wavelengths)
    delta_lambda = lambda_mean / spectral_resolution
    d_lambda_pix = np.mean(np.diff(wavelengths))
    sigma_pix = delta_lambda / d_lambda_pix

    return gaussian_filter1d(spectrum, sigma=sigma_pix)


def plotSpectrumOfPixel(cube, x, y, telescope=None, redshift=None):
    """
    Plot the spectrum of one spaxel, optionally smoothed to a telescope's resolution.

    Parameters
    ----------
    cube : mpdaf.obj.Cube
        Input cube (must provide ``get_spectrum(x, y)``).
    x, y : float
        Spaxel coordinates.
    telescope : cubeshift.instruments.Telescope, optional
        If given, the spectrum is smoothed to ``telescope.spectral_resolution``.
    redshift : float, optional
        Redshift applied to the wavelengths before computing the smoothing kernel.
    """
    spec = cube.get_spectrum(x, y)
    wavelengths = spec.wave.coord()
    spectrum = spec.data

    if telescope is not None:
        spectrum = _convolve_to_resolution(spectrum, wavelengths,
                                           telescope.spectral_resolution, redshift)

    label = f"Spectrum at (x={x:.2f}, y={y:.2f})"
    if telescope is not None:
        label += f" ({telescope.name})"

    plt.figure(figsize=(8, 4))
    plt.plot(wavelengths, spectrum)
    plt.xlabel("Wavelength (\u00c5)")
    plt.ylabel("Flux")
    plt.title(label)
    plt.grid(True)
    plt.show()

def plotSpectrumWithEmissionLines(cube, x, y, z=0.0):
    spec = cube.get_spectrum(x, y)
    wavelengths = spec.wave.coord()
    spectrum = spec.data

    plt.figure(figsize=(12, 6))
    plt.plot(wavelengths, spectrum, label='Spectrum', color='black')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux')
    plt.title(f"Spectrum at (x={x:.2f}, y={y:.2f}) with Emission Lines (z = {z})")
    plt.grid(True)

    emission_lines = {
        'Hδ': 4102,
        'Hγ': 4341,
        'Hβ': 4863,
        '[O III]': 5007,
        '[N II] 6548': 6548,
        'Hα': 6563,
        '[N II] 6584': 6584,
        '[S II] 6716': 6716,
        '[S II] 6731': 6731
    }

    y_max = np.nanmax(spectrum)
    y_range = y_max - np.nanmin(spectrum)
    label_offsets = np.linspace(0.05, 0.25, len(emission_lines))

    for i, (name, rest_wave) in enumerate(emission_lines.items()):
        obs_wave = rest_wave * (1 + z)

        if wavelengths.min() < rest_wave < wavelengths.max():
            plt.axvline(x=rest_wave, color='gray', linestyle='--', alpha=0.4)

        if wavelengths.min() < obs_wave < wavelengths.max():
            plt.axvline(x=obs_wave, color='red', linestyle='--', alpha=0.7)
            label_y = y_max + label_offsets[i % len(label_offsets)] * y_range
            plt.text(obs_wave, label_y, name, rotation=90, verticalalignment='bottom',
                     color='red', fontsize=9, ha='center')

    plt.tight_layout()
    plt.show()

def makeBroadband(cube):
    broadband_image = cube.sum(axis=0)

    plt.figure(figsize=(6, 6))
    plt.imshow(broadband_image.data, origin='lower', cmap='inferno')
    plt.colorbar(label='Flux (summed)')
    plt.title('Broadband Image (summed over all wavelengths)')
    plt.xlabel('X (spaxel)')
    plt.ylabel('Y (spaxel)')
    plt.show()

def makeNarrowbandImage(cube, center_wavelength, width=5):
    wl = cube.wave.coord()
    low = center_wavelength - width
    high = center_wavelength + width

    indices = (wl >= low) & (wl <= high)
    if not np.any(indices):
        print(f"No data in wavelength range {low}-{high} Å")
        return

    narrowband_image = cube.data[indices, :, :].sum(axis=0)

    plt.figure(figsize=(6, 6))
    plt.imshow(narrowband_image, origin='lower', cmap='plasma')
    plt.title(f'Narrowband Image: {center_wavelength} ± {width} Å')
    plt.colorbar(label='Flux')
    plt.xlabel('X (spaxel)')
    plt.ylabel('Y (spaxel)')
    plt.show()

def get_brightest_spot_gauss(cube):
    broadband_image = cube.sum(axis=0)
    fit_result = broadband_image.gauss_fit()
    x_bright = fit_result['xcenter'].value
    y_bright = fit_result['ycenter'].value
    print(f"Brightest spot from Gaussian fit at (x={x_bright:.2f}, y={y_bright:.2f})")
    return x_bright, y_bright

def make_hbeta_mask(cube, z=0.0251, width=5, flux_threshold=0.3):
    """
    Create a binary mask over regions with significant Hβ emission.

    Parameters:
        cube : MPDAF Cube
            The MPDAF spectral cube object.
        z : float
            Redshift of the galaxy.
        width : float
            Half-width (in Å) of the narrowband region around Hβ.
        flux_threshold : float
            Threshold (in normalized flux) to include pixels in the mask (0–1 scale).

    Returns:
        mask : 2D numpy array (binary)
            Boolean mask where True indicates strong Hβ emission.
        hbeta_image : 2D numpy array
            The narrowband Hβ flux image used to make the mask.
    """
    # Convert rest-frame Hβ to observed wavelength
    rest_hbeta = 4861  # Å
    hbeta_obs = rest_hbeta * (1 + z)
    
    # Extract narrowband image around Hβ
    hbeta_slice = cube.get_image((hbeta_obs - width, hbeta_obs + width))

    # Normalize for thresholding
    data = hbeta_slice.data
    data_norm = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data) + 1e-8)

    # Create binary mask
    mask = data_norm > flux_threshold

    return mask, hbeta_slice
