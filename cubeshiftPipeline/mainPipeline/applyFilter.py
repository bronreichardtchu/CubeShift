from astropy.io import fits
from scipy.interpolate import interp1d
import numpy as np
from mpdaf.obj import Cube
import matplotlib.pyplot as plt
from mpdaf.obj import Cube, WaveCoord
from zWavelengths import redshift_wavelength_axis
from astropy import units as u
import os



# Bandpass ranges in microns, based on JWST/NIRSpec documentation
NIRSPEC_THROUGHPUT_RANGES = {
    "f140x": (0.8, 2.0),      # Target acquisition
    "f110w": (1.0, 1.3),      # Bright target acquisition
    "f070lp": (0.7, 1.3),     # Used with G140M
    "f100lp": (1.0, 1.9),     # Used with G140M
    "f170lp": (1.7, 3.2),     # Used with G235M
    "f290lp": (2.9, 5.3),     # Used with G395M
    "clear": (0.6, 5.3),      # Used with PRISM or TA
}



def plot_throughput_and_spectrum(cube, wave_filter, thr_filter, x=0, y=0):
    """
    Plots the filter throughput and compares the cube spectrum before and after applying filter at (x,y).

    Parameters:
        cube: MPDAF Cube (already filtered)
        wave_filter: wavelength array of filter (in microns)
        thr_filter: throughput array (0 to 1)
        x, y: spatial pixel coordinates to extract spectrum
    """

    # Convert cube wavelengths from Å to microns
    wave_cube = cube.wave.coord() / 1e4  

    # Extract the filtered spectrum at pixel (x, y)
    spec_filtered = cube.data[:, y, x]  

    # Plot
    plt.figure(figsize=(10, 6))
    
    # Throughput curve
    plt.plot(wave_filter, thr_filter, label='Filter Throughput', color='green', alpha=0.7)
    
    # Filtered spectrum
    plt.plot(wave_cube, spec_filtered / np.max(spec_filtered), label='Filtered Spectrum (norm)', color='blue')

    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Normalized Flux / Throughput")
    plt.title(f"Spectrum at pixel (x={x}, y={y}) with Filter Applied")
    plt.legend()
    plt.grid(True)
    plt.show()



def load_filter_transmission(fits_path):
    with fits.open(fits_path) as hdul:
        data = hdul[1].data
        wl = data['WAVELENGTH']  # μm
        thr = data['THROUGHPUT']
    return wl, thr


def select_nirspec_mode(lambda_obs_micron):
    for mode, (lam_min, lam_max) in NIRSPEC_THROUGHPUT_RANGES.items():
        if lam_min <= lambda_obs_micron <= lam_max:
            return mode
    return None


def select_best_nirspec_filter(lambda_obs_micron):
    """
    Select the NIRSpec filter mode that covers the observed wavelength
    and has the smallest bandwidth (finest resolution).
    """
    candidates = []
    for mode, (lam_min, lam_max) in NIRSPEC_THROUGHPUT_RANGES.items():
        if lam_min <= lambda_obs_micron <= lam_max:
            bandwidth = lam_max - lam_min
            candidates.append((mode, bandwidth))

    if not candidates:
        return None

    # Sort by bandwidth ascending and pick the smallest
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]


def apply_nirspec_throughput(cube, z, line_rest_wavelength, filter_dir):
    """
    Applies the appropriate NIRSpec transmission filter to a cube based on redshifted line.
    
    Parameters:
        cube: MPDAF Cube
        z: redshift of the galaxy
        line_rest_wavelength: rest wavelength in Angstroms (e.g., 5007 for [OIII])
        filter_dir: path to directory containing *_trans.fits files

    Returns:
        MPDAF Cube after throughput correction,
        cube wavelength array (µm),
        interpolated throughput array,
        chosen filter filename.
    """
    lambda_obs = (line_rest_wavelength * (1 + z)) / 1e4  # Convert to microns
    print(f"Observed wavelength: {lambda_obs:.3f} µm at z = {z}")

    chosen_mode = select_best_nirspec_filter(lambda_obs)
    if chosen_mode is None:
        raise ValueError(f"No NIRSpec filter mode covers λ = {lambda_obs:.3f} µm")

    print(f"Chosen filter mode with smallest bandwidth: {chosen_mode}")

    filter_filename = f"jwst_nirspec_{chosen_mode}_trans.fits"
    filter_path = os.path.join(filter_dir, filter_filename)
    if not os.path.exists(filter_path):
        raise FileNotFoundError(f"Transmission filter file not found: {filter_path}")

    with fits.open(filter_path) as hdul:
        wave_file = hdul[1].data['WAVELENGTH']  # µm
        trans_file = hdul[1].data['THROUGHPUT']

    print(f"Applying filter: {filter_filename} (covers {wave_file.min():.2f}–{wave_file.max():.2f} µm)")

    cube_waves_um = cube.wave.coord() / 1e4  # Convert cube wavelengths to µm
    interp = interp1d(wave_file, trans_file, bounds_error=False, fill_value=0.0)
    throughput_interp = interp(cube_waves_um)

    print(f"Throughput range after interpolation: min={throughput_interp.min()}, max={throughput_interp.max()}")

    # Apply throughput correction
    cube.data *= throughput_interp[:, np.newaxis, np.newaxis]

    return cube, cube_waves_um, throughput_interp, filter_filename, chosen_mode


def plot_results(filtered_cube):
    # test that filter is valid
    hdu = fits.open(r"/Users/janev/Library/Group Containers/UBF8T346G9.OneDriveStandaloneSuite/OneDrive - Queen's University.noindex/OneDrive - Queen's University/MNU 2025/Transmission Filters/jwst_nirspec_f070lp_trans.fits")
    filter_data = hdu[1].data  # Assuming extension 1 contains throughput
    lam_filter = filter_data['WAVELENGTH']  # µm
    throughput = filter_data['THROUGHPUT']

    print(f"Filter wavelength range: {lam_filter.min()} - {lam_filter.max()} µm")
    print(f"Throughput min/max: {throughput.min()}, {throughput.max()}")

    plt.plot(lam_filter, throughput)
    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Throughput")
    plt.title("Filter Transmission Curve")
    plt.grid()
    plt.show()

    # Choose a bright pixel 
    x, y = 30, 96 
    spectrum = filtered_cube[:, y, x].data
    wavelengths = filtered_cube.wave.coord()  # in Angstroms

    plt.figure(figsize=(8, 4))
    plt.plot(wavelengths, spectrum, color='blue')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux (filtered)')
    plt.title(f'Spectrum at pixel (y={y}, x={x})')
    plt.grid()
    plt.show()

    wave = filtered_cube.wave.coord()
    center = 1.252 * 1e4  # 1.252 μm in Å
    width = 20  # Narrowband width in Å

    # Find the slice indices within this narrow window
    mask = (wave > center - width/2) & (wave < center + width/2)
    narrowband_image = np.nansum(filtered_cube.data[mask], axis=0)

    plt.figure(figsize=(6, 6))
    plt.imshow(narrowband_image, origin='lower', cmap='inferno')
    plt.colorbar(label='Flux')
    plt.title(f'Narrowband image around [O III] λ = {center:.0f} Å')
    plt.show()

    broadband_image = np.nansum(filtered_cube.data, axis=0)

    plt.figure(figsize=(6, 6))
    plt.imshow(broadband_image, origin='lower', cmap='magma')
    plt.colorbar(label='Flux')
    plt.title('Broadband image after filter applied')
    plt.show()

def apply_transmission_filter(cube, z, line_rest_wavelength):
    """
    Applies the appropriate NIRSpec transmission filter to the cube.

    Returns:
        filtered_cube : mpdaf.obj.Cube
            The cube after throughput correction.
        filter_name : str
            The short filter name, e.g., 'f070lp'.
    """
    filter_dir = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/Transmission Filters"

    # Apply the throughput
    filtered_cube, wave, throughput, filter_filename, chosen_mode = apply_nirspec_throughput(
        cube, z, line_rest_wavelength, filter_dir
    )

    # chosen_filter is expected to be something like 'f070lp'
    filter_path = os.path.join(filter_dir, filter_filename)

    if not os.path.exists(filter_path):
        raise FileNotFoundError(f"Transmission filter file not found: {filter_path}")

    return filtered_cube, chosen_mode

if __name__ == "__main__":
    file_path = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/cgcg453_red_mosaic.fits"
    cube = Cube(file_path)

    z_obs = 0.025
    z_sim = 1.5
    rest_line = 5007  # [OIII]

    redshifted_cube, _ = redshift_wavelength_axis(cube, z_obs, z_sim)

    filter_dir = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/Transmission Filters"
    cube_waves = cube.wave.coord() / 1e4  # in microns
    print(f"Cube wavelength range: {cube_waves.min():.3f} - {cube_waves.max():.3f} µm")

    # Apply throughput correction and unpack
    filtered_cube, wave_filter, trans, filter_filename, chosen_mode = apply_nirspec_throughput(redshifted_cube, z_sim, rest_line, filter_dir)

    # Use the correct variables here:
    plot_throughput_and_spectrum(filtered_cube, wave_filter, trans, x=30, y=96)

    plot_results(filtered_cube)

    filtered_cube.write("/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/filtered_cube_with_throughput.fits")
