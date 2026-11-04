
# matchSpectral.py

import numpy as np
from astropy.convolution import Gaussian1DKernel, convolve
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy.ma as ma
from mpdaf.obj import Cube
import matplotlib.pyplot as plt
from astropy import units as u
from mpdaf.obj import Cube, WaveCoord
import copy
from zWavelengths import redshift_wavelength_axis
import os
from astropy.io import fits




# Wavelengths in microns (μm)
NIRSPEC_DISPERSER_RANGES = [
    {"name": "g140m", "R": 1000, "lambda_min": 0.70, "lambda_max": 1.27},
    {"name": "g235m", "R": 1000, "lambda_min": 1.66, "lambda_max": 3.07},
    {"name": "g395m", "R": 1000, "lambda_min": 2.87, "lambda_max": 5.10},
    {"name": "g140h", "R": 2700, "lambda_min": 0.81, "lambda_max": 1.27},
    {"name": "g235h", "R": 2700, "lambda_min": 1.66, "lambda_max": 3.05},
    {"name": "g395h", "R": 2700, "lambda_min": 2.87, "lambda_max": 5.14},
    {"name": "prism", "R": 100, "lambda_min": 0.60, "lambda_max": 5.30},
]

def crop_cube_to_wavelength_range(cube, lambda_min, lambda_max):
    """
    Crops the MPDAF cube along the spectral axis to only include wavelengths
    within [lambda_min, lambda_max].

    Parameters:
        cube         -- MPDAF Cube object
        lambda_min   -- minimum wavelength to keep (float)
        lambda_max   -- maximum wavelength to keep (float)

    Returns:
        cropped_cube -- new MPDAF Cube containing only the selected spectral range
    """
    wave_axis = cube.wave.coord()
    mask = (wave_axis >= lambda_min) & (wave_axis <= lambda_max)
    print(f"Attempting to crop to {lambda_min:.3f}–{lambda_max:.3f} μm ({lambda_min*1e4:.0f}–{lambda_max*1e4:.0f} Å)")
    print(cube.wave)               # If you're using MPDAF
    print(cube.wave.unit)   # Should return u.um, u.AA, etc.
    wave_axis = cube.wave.coord()
    print(wave_axis[0], wave_axis[-1])



    if np.sum(mask) == 0:
        raise ValueError("No wavelengths fall within the specified range.")

    cropped_data = cube.data[mask, :, :]
    cropped_wave = wave_axis[mask]
    cropped_wavecoord = create_wavecoord_from_axis(cropped_wave)

    return Cube(data=cropped_data, wave=cropped_wavecoord, wcs=cube.wcs)



def select_best_disperser_with_partial(cube, lam_obs_min, lam_obs_max, crop_cube_if_needed=True):
    """
    Selects best disperser for the observed wavelength range.
    If full coverage isn't possible, but a filter covers >50%,
    it crops the cube to that filter's wavelength range.
    
    Returns:
        (selected_filter_dict, possibly_cropped_cube)
    """
    print(f"\nObserved wavelength range: {lam_obs_min:.3f} – {lam_obs_max:.3f} μm")
    full_matches = []

    for combo in NIRSPEC_DISPERSER_RANGES:

        name = combo["name"]
        lambda_min = combo["lambda_min"]
        lambda_max = combo["lambda_max"]

        print(f"\nChecking {name}: {lambda_min:.3f}–{lambda_max:.3f} μm")
        
        # Full coverage check
        if lam_obs_min >= lambda_min and lam_obs_max <= lambda_max:
            print(f"  Full coverage with {name}")
            full_matches.append((combo, lambda_max - lambda_min))  # Save width for ranking
            continue  # Go to next filter

        # Partial overlap calculation
        overlap = max(0, min(lam_obs_max, lambda_max) - max(lam_obs_min, lambda_min))
        coverage_fraction = overlap / (lam_obs_max - lam_obs_min)
        print(f"Overlap = {overlap:.3f} μm, Coverage fraction = {coverage_fraction:.2%}")

        if coverage_fraction > 0.5 and crop_cube_if_needed:
            print(f" More than 50% falls within {name}. Attempting to crop.")
            lam_overlap_min = max(lambda_min, lam_obs_min)
            lam_overlap_max = min(lambda_max, lam_obs_max)

            if lam_overlap_max > lam_overlap_min:
                print(f" Cropping cube to: {lam_overlap_min:.3f}–{lam_overlap_max:.3f} μm "
                      f"({lam_overlap_min*1e4:.0f}–{lam_overlap_max*1e4:.0f} Å)")
                cube = crop_cube_to_wavelength_range(cube, lam_overlap_min * 1e4, lam_overlap_max * 1e4)
                return combo, cube
            else:
                print("Warning: invalid overlap range. Skipping crop.")

    # Prefer narrowest full coverage if any exist
    if full_matches:
        best_full = sorted(full_matches, key=lambda x: x[1])[0][0]
        print(f"\n Using fully-covered filter: {best_full['name']}")
        return best_full, cube

    # Fallback: PRISM
    for combo in NIRSPEC_DISPERSER_RANGES:

        if combo["name"].lower() == "prism":
            print("\n No good match found. Falling back to PRISM.")
            return combo, cube

    raise ValueError("No suitable disperser found.")



def R_input_func(wave_angstrom):
    """
    wave_angstrom: numpy array of wavelengths in Angstroms
    Returns interpolated R values at those wavelengths.
    """
    wave_micron = wave_angstrom * 1e-4  # convert Angstrom to microns
    
    # known wavelengths and R values from your dispersion data (microns)
    # make sure these are global or passed in some way; here I assume they're available
    
    return np.interp(wave_micron, wave_um, R_vals, left=R_vals[0], right=R_vals[-1])

def load_dispersion_file(filter_name, dispersion_dir):
    """
    Load dispersion data from a JWST NIRSpec FITS file.
    
    Parameters:
        filter_name (str): e.g. "G140H_F100LP"
        dispersion_dir (str): path to folder containing dispersion FITS files

    Returns:
        np.ndarray: Nx2 array, columns = [wavelength (µm), R]
    """
    # Construct expected FITS filename
    fname = dispersion_filename = f"jwst_nirspec_{filter_name.lower()}_disp.fits"

    file_path = os.path.join(dispersion_dir, fname)

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Dispersion FITS file not found: {file_path}")

    # Open FITS file
    with fits.open(file_path) as hdul:
        # Usually dispersion data is in HDU[1] as a table
        data = hdul[1].data
        
        # JWST files typically have columns like 'WAVELENGTH' and 'R'
        # Wavelength is in microns, R is dimensionless
        wavelength = data['WAVELENGTH']  # in µm
        R = data['R']

        return np.column_stack([wavelength, R])


def to_nan_array(arr):
    """Convert a masked array to a float array with NaNs where masked."""
    if isinstance(arr, ma.MaskedArray):
        return arr.filled(np.nan)
    return arr


def fill_nans_spectrum(spectrum):
    """
    Interpolates over NaNs in a 1D spectrum. If too many NaNs, fills with 0.
    """
    idx = np.arange(len(spectrum))
    good = ~np.isnan(spectrum)
    
    if good.sum() < 2:
        return np.nan_to_num(spectrum)  # fallback: fill with zeros

    interp = interp1d(idx[good], spectrum[good], bounds_error=False, fill_value="extrapolate")
    return interp(idx)



def precomputed_match_spectral_resolution_variable_kernel(cube, R_input_interpolated, R_target):
    # Version 3 (final version - )
    """
    Matches spectral resolution by convolving each wavelength bin with 
    a wavelength-dependent Gaussian kernel. Uses precomputed kernel bank. 
    Rolls precomputed kernels per wavelength rather than recalculating Gaussian.

    Parameters:
        cube      -- MPDAF Cube
        R_input   -- original spectral resolution (λ / Δλ)
        R_target  -- desired spectral resolution

    Returns:
        cube      -- resolution-matched cube
    """
    print("Matching precomputed spectral resolution with variable kernel...")



    new_cube = cube.copy()
    wave = cube.wave.coord()


    # Calculate the wavelength step size (assumes constant spacing)
    delta_lambda = wave[1] - wave[0]
    # Get number of wavelength bins (spectral dimension)
    n_wave = cube.shape[0]

    # --- STEP 1: Compute Gaussian kernel widths ---
    # FWHM (Full Width at Half Maximum) of the current spectral resolution
    R_input = R_input_interpolated  # array of same length as wave
    fwhm_input = wave / R_input

    # FWHM of the target (lower) spectral resolution
    fwhm_target = wave / R_target
    # Compute the FWHM of the Gaussian kernel needed to degrade from input to target resolution
    # Use quadrature (if that makes sense?): FWHM_kernel^2 = FWHM_target^2 - FWHM_input^2
    fwhm_kernel = np.sqrt(np.maximum(fwhm_target**2 - fwhm_input**2, 0))
    # Convert FWHM to standard deviation (sigma) in pixel units, since convolution uses sigma
    sigma_kernel_pix = fwhm_kernel / delta_lambda / 2.355

    # --- STEP 2: Precompute all Gaussian kernels ---
    print(" Precomputing kernels...")
    kernel_bank = []

    for sigma in sigma_kernel_pix:
        if sigma == 0 or np.isnan(sigma):
            kernel = np.zeros(n_wave)
            kernel[n_wave // 2] = 1.0  # delta function (no blur)
        else:
            delta = np.zeros(n_wave)
            delta[n_wave // 2] = 1.0
            kernel = gaussian_filter1d(delta, sigma=sigma, mode='constant')
        kernel_bank.append(kernel)




    # --- STEP 3: Convolve each spaxel spectrum using precomputed kernels ---
    for y in range(cube.shape[1]): # loop over spatial Y pixels
        for x in range(cube.shape[2]): # loop over spatial X pixels
            spectrum = to_nan_array(cube.data[:, y, x])
            # Fill in any NaNs with interpolated values
            spectrum = fill_nans_spectrum(spectrum)
            # Create an empty array to store the smoothed spectrum
            smoothed = np.zeros_like(spectrum) # move to outside loop

            # Loop over each wavelength bin
            for i in range(n_wave):
                # Roll the kernel so it's centered at the current wavelength index i
                kernel = np.roll(kernel_bank[i], i - n_wave // 2)
                # Apply the kernel to the spectrum using a dot product
                # This is equivalent to Gaussian smoothing around index i
                smoothed[i] = np.nansum(spectrum * kernel)
            # Store the smoothed spectrum back into the new cube
            new_cube.data[:, y, x] = smoothed

    print(" Spectral resolution matched using variable kernel.")
    return new_cube


def create_wavecoord_from_axis(new_wave_axis):
    """
    Creates a new wavelength coordinate system (WaveCoord) from an array of wavelengths.
    This is needed because MPDAF cubes use a WaveCoord object to define their spectral axis.
    The function uses the wavelength step size, starting wavelength, and assumes the first pixel 
    corresponds to the start.

    Returns:
        WaveCoord -- MPDAF object that defines the spectral axis of a cube
    """
    # Calculate the step size (Δλ) between each wavelength bin
    cdelt = new_wave_axis[1] - new_wave_axis[0] 
    # Set the reference value (starting wavelength)
    crval = new_wave_axis[0] 
    # Set the reference pixel (assume first bin is pixel 1 for simplicity)
    crpix = 1  
    return WaveCoord(cdelt=cdelt, crval=crval, crpix=crpix, cunit='Angstrom')


def resample_spectral_axis(cube, new_wave_axis):
    """
    Resamples (regrids) the spectral axis of the cube by interpolating every spectrum
    onto a new set of wavelength values.

    This is used when we want to change the *sampling* of the spectrum — for example, to match
    the wavelength grid of another instrument, or to downsample after resolution degradation.

    Parameters:
        cube           -- the original MPDAF cube to resample
        new_wave_axis  -- 1D array of new wavelength values to interpolate to

    Returns:
        resampled_cube -- new MPDAF Cube object with data interpolated onto new_wave_axis
    """
    print("Resampling spectral axis...")

    # Get the spatial dimensions of the cube
    ny, nx = cube.shape[1], cube.shape[2]
    # Number of wavelength bins in the new spectral axis
    new_nwave = len(new_wave_axis)
    # Create an empty array to hold the resampled cube data
    new_data = np.zeros((len(new_wave_axis), ny, nx))

    # Get the original wavelength axis from the cube
    old_wave = cube.wave.coord()

    # --- STEP 1: Loop through every spaxel and interpolate its spectrum to the new wavelengths ---
    for y in range(ny):
        for x in range(nx):
            # Extract the 1D spectrum at position (y, x)
            spectrum = cube.data[:, y, x]
             # Create an interpolator from the original spectrum
            # If the new wavelength is outside the old range, fill it with 0
            interp = interp1d(old_wave, spectrum, bounds_error=False, fill_value=0)
            new_data[:, y, x] = interp(new_wave_axis)

    # --- STEP 2: Define a new spectral coordinate system for the resampled axis ---
    new_wave = create_wavecoord_from_axis(new_wave_axis)

    # Determine overlapping spectral range between old and new
    # Overlapping range between old and new wavelengths
    old_min, old_max = old_wave[0], old_wave[-1]
    new_min, new_max = new_wave_axis[0], new_wave_axis[-1]
    min_wave = max(old_min, new_min)
    max_wave = min(old_max, new_max)

    # Mask for overlapping wavelengths
    # --- STEP 3: trim to overlap (optional) ---
    mask = (new_wave_axis >= old_wave[0]) & (new_wave_axis <= old_wave[-1])
    trimmed_wave_axis = new_wave_axis[mask]
    new_data_trimmed    = new_data[mask, :, :]

    # Build a WaveCoord even if Δλ varies:
    new_wave_trimmed = create_wavecoord_from_axis(trimmed_wave_axis)
    resampled_cube = Cube(data=new_data_trimmed,
                          wave=new_wave_trimmed,
                          wcs=cube.wcs)
    print("Spectral axis resampled.")
    return resampled_cube


def match_variable_resolution(cube, R_lambda, R_target):
    """
    Matches spectral resolution by convolving each wavelength bin 
    with a Gaussian kernel to degrade the resolution from R_lambda to R_target.
    
    Parameters:
        cube      : MPDAF Cube object
        R_lambda  : 1D array of input resolving power, one per wavelength bin
        R_target  : Scalar target resolving power to degrade to
        
    Returns:
        A new Cube object with matched spectral resolution
    """
    new_cube = cube.copy()
    data = new_cube.data
    var = new_cube.var
    lam = new_cube.wave.coord()  # in Ångstroms

    # Calculate FWHM and sigma in Å
    lam = np.asarray(lam)  # shape (Nλ,)
    fwhm_input = lam / R_lambda              # input FWHM(λ)
    fwhm_target = lam / R_target             # target FWHM (constant)
    
    # Compute Gaussian kernel widths in σ (Å)
    fwhm_diff_squared = fwhm_target**2 - fwhm_input**2
    sigma_kernel = np.sqrt(np.maximum(fwhm_diff_squared, 0)) / 2.355  # avoid negative under sqrt

    # Convert sigma from Å to pixels along the spectral axis
    delta_lambda = np.median(np.diff(lam))  # assume uniform spacing
    sigma_pixels = sigma_kernel / delta_lambda

    # Apply wavelength-dependent Gaussian smoothing along spectral axis
    smoothed_data = np.empty_like(data)
    smoothed_var = np.empty_like(var)
    for y in range(data.shape[1]):
        for x in range(data.shape[2]):
            spectrum = data[:, y, x]
            variance = var[:, y, x]
            smoothed_data[:, y, x] = gaussian_filter1d(spectrum, sigma_pixels, mode='nearest')
            smoothed_var[:, y, x] = gaussian_filter1d(variance, sigma_pixels, mode='nearest')

    new_cube.data = smoothed_data
    new_cube.var = smoothed_var

    return new_cube


def check_nans_inf(cube, label):
    data = cube.data
    n_nans = np.isnan(data).sum()
    n_infs = np.isinf(data).sum()
    print(f"{label} - NaNs: {n_nans}, Infs: {n_infs}")




def apply_dispersion_pipeline(redshifted_cube, z_obs, z_sim):
    """
    Applies the NIRSpec disperser pipeline:
    - Redshifts the cube
    - Crops to overlapping dispersion range
    - Loads and applies dispersion filter
    - Resamples cube to dispersion wavelength axis
    - Matches spectral resolution using precomputed kernel
    
    Parameters:
        cube (MPDAF Cube): The input observed cube
        z_obs (float): Observed redshift of the cube
        z_sim (float): Target redshift for simulation
        dispersion_dir (str): Directory containing dispersion filter files
    
    Returns:
        blurred_cube (MPDAF Cube): Cube after applying disperser and resolution match
        disperser_info (dict): The dictionary of the disperser used (name, R, range)
    """
    # Path to folder containing NIRSpec dispersion filters 
    dispersion_dir = r"/Users/janev/Library/Group Containers/UBF8T346G9.OneDriveStandaloneSuite/OneDrive - Queen's University.noindex/OneDrive - Queen's University/MNU 2025/Dispersion Filters"


    # Step 1: Redshift the cube --- THIS IS DONE IN THE MAIN PIPELINE OF make_narrowband_map()
    # we assume the cube has already been redshifted at this point
    # redshifted_cube, _ = redshift_wavelength_axis(cube, z_obs, z_sim)
    redshifted_wave_um = redshifted_cube.wave.coord() / 1e4  # Convert to microns
    lam_obs_min, lam_obs_max = redshifted_wave_um.min(), redshifted_wave_um.max()

    # Step 2: Choose best disperser & crop the cube accordingly
    best_disperser, cube_cropped = select_best_disperser_with_partial(redshifted_cube, lam_obs_min, lam_obs_max)

    # Step 3: Load disperser data
    dispersion_data = load_dispersion_file(best_disperser["name"], dispersion_dir)  # [:,0] = µm, [:,1] = R

    # Step 4: Resample cube to disperser's wavelength axis
    new_wave_axis_angstrom = dispersion_data[:, 0] * 1e4  # microns → Angstroms
    resampled_cube = resample_spectral_axis(cube_cropped, new_wave_axis_angstrom)

    # Step 5: Interpolate R input across new wavelength axis
    wave_ang = resampled_cube.wave.coord()
    wave_um = wave_ang / 1e4
    R_vals_interp = np.interp(wave_um, dispersion_data[:, 0], dispersion_data[:, 1])
    R_input = np.clip(R_vals_interp, 10, 10000)
    target_R = np.median(dispersion_data[:, 1])

    # Step 6: Match spectral resolution using precomputed kernel
    blurred_cube = precomputed_match_spectral_resolution_variable_kernel(
        resampled_cube, R_input, target_R
    )

    return blurred_cube, best_disperser["name"]




if __name__ == "__main__":
    
    file_path = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/cgcg453_red_mosaic.fits"
    dispersion_dir = r"/Users/janev/Library/Group Containers/UBF8T346G9.OneDriveStandaloneSuite/OneDrive - Queen's University.noindex/OneDrive - Queen's University/MNU 2025/Dispersion Filters"


    # Load the cube
    cube = Cube(file_path)
    z_obs = 0.025
    z_sim = 1.5

    redshifted_cube, _ = redshift_wavelength_axis(cube, z_obs, z_sim)
    blurred_cube, disperser_info = apply_dispersion_pipeline(redshifted_cube, z_obs, z_sim)
    print(f"Disperser used: {disperser_info['name']} with R ≈ {disperser_info['R']}")

        # After you write the file...
    output_path = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/spectral_matched_cube_with_despersion.fits"
    blurred_cube.write(output_path)

    # Now immediately read it back in and print stats:
    reloaded = Cube(output_path)
    print("Reloaded cube stats:", 
        "min=", np.nanmin(reloaded.data), 
        "max=", np.nanmax(reloaded.data), 
        "mean=", np.nanmean(reloaded.data))


"""

    print(f"Original cube stats: min={cube.data.min()}, max={cube.data.max()}, mean={cube.data.mean()}")
    y, x = cube.shape[1] // 2, cube.shape[2] // 2
    original_spec = cube.data[:, y, x]
    plt.figure()
    plt.plot(cube.wave.coord(), original_spec)
    plt.title("Original cube central spaxel spectrum")
    plt.xlabel("Wavelength (Angstrom)")
    plt.ylabel("Flux")
    plt.show()


    redshifted_cube, _ = redshift_wavelength_axis(cube, z_obs, z_sim)
    redshifted_cube_wave = redshifted_cube.wave.coord() / 1e4  # Convert to microns


    lam_obs_min = redshifted_cube_wave.min()
    lam_obs_max = redshifted_cube_wave.max()

    print(f"lam_obs_min = {lam_obs_min}, lam_obs_max = {lam_obs_max}")
    for combo in NIRSPEC_DISPERSER_RANGES
:
        print(f"{combo['name']} → R={combo['R']}, λ = [{combo['lambda_min']}, {combo['lambda_max']}]")

    best_disperser,  cube_cropped = select_best_disperser_with_partial(redshifted_cube, lam_obs_min, lam_obs_max)
    print("DEBUG: Type of cube after cropping:", type(cube_cropped))
    print("DEBUG: Shape of cube data:", cube_cropped.data.shape)


    print(f"Using disperser: {best_disperser['name']}")

    # Select and load disperser

    filter_name = best_disperser["name"]
    print(f"Selected disperser: {filter_name} with R ≈ {best_disperser['R']}")


    dispersion_data = load_dispersion_file(filter_name, dispersion_dir)  # Returns array with [:,0]=wavelength (µm), [:,1]=R

    # Step 1: Resample to the disperser's wavelength grid
    new_wave_axis = dispersion_data[:, 0] * 1e4  # microns to Angstroms
    target_R = np.median(dispersion_data[:, 1])  # Representative R value
    print("DEBUG: cube_cropped.wave is None?", cube_cropped.wave is None)

    print("Resampling cube to match dispersion wavelength axis...")

    resampled_cube = resample_spectral_axis(cube_cropped, new_wave_axis)

    print(f"After resample: min={resampled_cube.data.min()}, max={resampled_cube.data.max()}, mean={resampled_cube.data.mean()}")
    resampled_spec = resampled_cube.data[:, y, x]
    plt.figure()
    plt.plot(resampled_cube.wave.coord(), resampled_spec)
    plt.title("Resampled cube central spaxel spectrum")
    plt.xlabel("Wavelength (Angstrom)")
    plt.ylabel("Flux")
    plt.show()

    wave_um = dispersion_data[:, 0]  # microns
    R_vals = dispersion_data[:, 1]
    target_R = np.median(R_vals)  # or your chosen target resolution


    # new_wave_axis: the cube's current wavelength array in microns (same length as wave)
    new_wave_axis = resampled_cube.wave.coord() 

    # wave_um and R_vals come from your disperser dispersion data (original)
    # Interpolate R_vals onto new_wave_axis
    wave_ang = resampled_cube.wave.coord()   # Angstroms
    wave_um = wave_ang / 1e4                 # convert to microns
    R_input_interp = np.interp(wave_um, dispersion_data[:, 0], R_vals)
    R_input_interpolated = np.clip(R_input_interp, 10, 10000)


    plt.figure()
    plt.plot(new_wave_axis, R_input_interpolated, label='Input R (interp)')
    plt.axhline(target_R, color='r', linestyle='--', label='Target R')
    plt.title("Spectral Resolution R vs Wavelength")
    plt.xlabel("Wavelength (micron)")
    plt.ylabel("Resolution R")
    plt.legend()
    plt.show()

    print(f"Input R: min={np.min(R_input_interpolated)}, max={np.max(R_input_interpolated)}")
    print(f"Target R: {target_R}")


    print("Before blur: mean =", np.mean(resampled_cube.data))
    print("Interpolated R: min =", np.min(R_input_interpolated), "max =", np.max(R_input_interpolated))
    print("Target R:", target_R)


    # Step 3: Plot comparison at central spaxel
    y, x = blurred_cube.shape[1] // 2, blurred_cube.shape[2] // 2
    original_spec = resampled_cube.data[:, y, x]
    smoothed_spec = blurred_cube.data[:, y, x]

    plt.figure(figsize=(10, 5))
    plt.plot(new_wave_axis, original_spec, label="Resampled", alpha=0.6)
    plt.plot(new_wave_axis, smoothed_spec, label="Matched R", alpha=0.7)
    plt.xlabel("Wavelength (Å)")
    plt.ylabel("Flux")
    plt.title(f"Spectral Matching at Spaxel ({y}, {x})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Before blur: mean = {resampled_cube.data.mean()}")
    blurred_cube = precomputed_match_spectral_resolution_variable_kernel(
        resampled_cube, R_input_interpolated, target_R
    )
    print(f"After blur: mean = {blurred_cube.data.mean()}")

    blurred_spec = blurred_cube.data[:, y, x]
    plt.figure()
    plt.plot(blurred_cube.wave.coord(), blurred_spec, label='Blurred spectrum')
    plt.plot(resampled_cube.wave.coord(), resampled_spec, alpha=0.5, label='Resampled spectrum')
    plt.title("Spectrum Before and After Spectral Resolution Matching")
    plt.xlabel("Wavelength (Angstrom)")
    plt.ylabel("Flux")
    plt.legend()
    plt.show()

    print("Resampling and resolution matching complete.")
    check_nans_inf(cube, "Original cube")
    check_nans_inf(resampled_cube, "Resampled cube")
    check_nans_inf(blurred_cube, "Blurred cube")
"""
