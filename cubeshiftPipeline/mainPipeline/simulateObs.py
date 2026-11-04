"""
simulateObs.py

The simulation includes:
    1. Binning the data cube to a lower spatial resolution (to match the target instrument's pixel scale).
    2. Adjusting the brightness in each pixel to account for redshift effects:
        - Change in angular size (affecting pixel area)
        - Dimming due to luminosity distance
    3. Blurring the image to match the target telescope's point spread function (PSF),
       using a Gaussian kernel that accounts for the redshift-dependent change in resolution.

These steps help us understand how features like star formation, outflows, or emission structure 
might look when viewed at higher redshift, where both resolution and brightness are affected 
by cosmology and the observing instrument.

Main functions:
---------------
- bin_cube(x_factor, y_factor, cube) — Bins the cube to lower spatial resolution.
- scale_luminosity_for_redshift(...) — Scales flux to simulate brightness at higher redshift.
- convolve_to_match_psf(...) — Applies Gaussian blur to simulate PSF at new redshift.
- generate_mock_jwst_cube(...) — Runs the full pipeline from binning to final simulated cube.

Example:
--------
- Load a low-redshift IFU cube.
- Simulate how it would appear at z = 1.0 using JWST's pixel scale and resolution.
- Visualize and compare results before and after simulation.
"""


import numpy as np
from scipy.ndimage import gaussian_filter
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo
from mpdaf.obj import Cube
import matplotlib.pyplot as plt
from binData import bin_cube
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.cosmology import Planck18 as cosmo
import numpy as np

def rescale_flux(data, z_i, z_o):
    """
    Rescale flux values in a cube from redshift z_i to z_o.

    Parameters
    ----------
    data : numpy.ndarray
        Flux data array (e.g., cube.data).
    z_i : float
        Original redshift of the source.
    z_o : float
        Target redshift to simulate.

    Returns
    -------
    new_data : numpy.ndarray
        Flux-rescaled data array.
    """
    Dl_i = cosmo.luminosity_distance(z_i).to("cm").value
    Dl_o = cosmo.luminosity_distance(z_o).to("cm").value
    scale = (Dl_i / Dl_o)**2 * ((1.0 + z_i) / (1.0 + z_o))
    return data * scale


def scale_luminosity_for_redshift(cube, redshift_old, redshift_new, method="angular"):
    """
    Rescale the observed luminosity of each spaxel based on redshift.

    Parameters
    ----------
    cube : mpdaf.obj.Cube
        The data cube observed at redshift_old.
    redshift_old : float
        The original redshift of the observation.
    redshift_new : float
        The target redshift to simulate.
    method : str
        Scaling method to use. Options:
            - "angular": angular size change only (Da)
            - "luminosity": flux dimming only (Dl)
            - "both": includes both effects (Da and Dl)

    Returns
    -------
    new_data : numpy.ndarray
        The new spaxel luminosities as they would appear at redshift_new.
    """
    # get the x and y pixel scales in arcsec (handle rectangular pixels)
    dx_arcsec, dy_arcsec = cube.wcs.get_axis_increments(unit=u.arcsec)

    scaled_data = cube.data.copy()

    if method in ("angular", "both"):
        # proper kpc/arcsec at old and new redshifts
        proper_old = cosmo.kpc_proper_per_arcmin(redshift_old).to(u.kpc/u.arcsec)
        proper_new = cosmo.kpc_proper_per_arcmin(redshift_new).to(u.kpc/u.arcsec)

        # physical area per pixel in kpc^2
        spaxel_area_old = (dx_arcsec * proper_old) * (dy_arcsec * proper_old)
        spaxel_area_new = (dx_arcsec * proper_new) * (dy_arcsec * proper_new)

        # convert brightness to luminosity per kpc^2
        lum_per_area = scaled_data / spaxel_area_old.value

        # rescale to reflect area at new redshift
        scaled_data = lum_per_area * spaxel_area_new.value

    if method in ("luminosity", "both"):
        Dl_old = cosmo.luminosity_distance(redshift_old)
        Dl_new = cosmo.luminosity_distance(redshift_new)

        flux_scaling = (Dl_old / Dl_new)**2
        flux_scaling = flux_scaling.to(u.dimensionless_unscaled).value

        scaled_data *= flux_scaling

    return scaled_data, dx_arcsec, dy_arcsec



def convolve_to_match_psf(cube, fwhm_real_x_arcsec, fwhm_real_y_arcsec, fwhm_target_arcsec, z_obs, z_sim):
    """
    Convolve cube spatially to match target PSF, accounting for redshift scaling
    and potentially rectangular pixels.

    Parameters
    ----------
    cube : mpdaf.obj.Cube
        The cube to be convolved.
    fwhm_real_x_arcsec : float
        Original PSF FWHM along x-axis (arcsec/pixel).
    fwhm_real_y_arcsec : float
        Original PSF FWHM along y-axis (arcsec/pixel).
    fwhm_target_arcsec : float
        The telescope's PSF you want to simulate (arcsec).
    z_obs : float
        Original redshift.
    z_sim : float
        New redshift.

    Returns
    -------
    cube_convolved : mpdaf.obj.Cube
        Cube convolved with matching kernel to simulate PSF at new z.
    """
    
    # Step 1: Scale original PSF to simulate how it would appear at high-z
    Da_old = cosmo.angular_diameter_distance(z_obs)
    Da_new = cosmo.angular_diameter_distance(z_sim)

    fwhm_sim_x = fwhm_real_x_arcsec * (Da_old / Da_new).value
    fwhm_sim_y = fwhm_real_y_arcsec * (Da_old / Da_new).value

    if fwhm_target_arcsec <= max(fwhm_sim_x, fwhm_sim_y):
        print("Target PSF is finer than simulated real PSF — skipping convolution.")
        return cube.copy()

    # Step 2: Compute matching kernel (in arcsec)
    fwhm_kernel_x = np.sqrt(max(fwhm_target_arcsec**2 - fwhm_sim_x**2, 0))
    fwhm_kernel_y = np.sqrt(max(fwhm_target_arcsec**2 - fwhm_sim_y**2, 0))

    # Convert FWHM to sigma (Gaussian: sigma = FWHM / 2.355)
    sigma_x_arcsec = fwhm_kernel_x / 2.355
    sigma_y_arcsec = fwhm_kernel_y / 2.355

    # Convert sigma from arcsec to pixels
    dx_arcsec, dy_arcsec = cube.wcs.get_axis_increments(unit=u.arcsec)
    sigma_x_pix = sigma_x_arcsec / dx_arcsec
    sigma_y_pix = sigma_y_arcsec / dy_arcsec

    print(f"PSF sim (x, y) arcsec: ({fwhm_sim_x:.4f}, {fwhm_sim_y:.4f})")
    print(f"Matching kernel FWHM (x, y) arcsec: ({fwhm_kernel_x:.4f}, {fwhm_kernel_y:.4f}) → σ pixels: ({sigma_x_pix:.2f}, {sigma_y_pix:.2f})")

    # Step 3: Apply 2D Gaussian convolution per wavelength slice
    convolved_cube = cube.copy()
    for i in range(cube.shape[0]):  # over wavelength slices
        convolved_cube.data[i] = gaussian_filter(cube.data[i], sigma=(sigma_y_pix, sigma_x_pix))

    return convolved_cube

def generate_mock_jwst_cube(original_cube, x_factor, y_factor, redshift_old, redshift_new, fwhm_target_arcsec):
    """
    Run full simulation pipeline:
    bin → scale luminosity → convolve PSF
    """

    # Step 1: bin
    binned_cube = bin_cube(x_factor, y_factor, original_cube)

    # Step 2: scale luminosity
    scaled_data, dx_arcsec, dy_arcsec = scale_luminosity_for_redshift(binned_cube, redshift_old, redshift_new)
    binned_cube.data = scaled_data

    # Step 3: convolve to match PSF at new redshift
    convolved_cube = convolve_to_match_psf(
        binned_cube,
        dx_arcsec, dy_arcsec,
        fwhm_target_arcsec=fwhm_target_arcsec,
        z_obs=redshift_old,
        z_sim=redshift_new
    )

    return convolved_cube

def plot_masked_slice(data_cube, slice_index=0, title="Slice"):
    masked = np.ma.masked_invalid(data_cube[slice_index])
    plt.imshow(masked, origin='lower', cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.show()

def print_scaling_factors(z1, z2):
    Da1 = cosmo.angular_diameter_distance(z1)
    Da2 = cosmo.angular_diameter_distance(z2)
    Dl1 = cosmo.luminosity_distance(z1)
    Dl2 = cosmo.luminosity_distance(z2)

    angular_scaling = (Da1 / Da2)**2
    luminosity_scaling = (Dl1 / Dl2)**2
    combined_scaling = angular_scaling * luminosity_scaling

    print(f"Da(z={z1:.3f}) = {Da1:.2f}, Da(z={z2:.3f}) = {Da2:.2f}")
    print(f"Dl(z={z1:.3f}) = {Dl1:.2f}, Dl(z={z2:.3f}) = {Dl2:.2f}")
    print(f"Angular (Da²) scaling factor:     {angular_scaling:.4e}")
    print(f"Luminosity (Dl²) scaling factor:  {luminosity_scaling:.4e}")
    print(f"Combined (Da² * Dl²) scaling:     {combined_scaling:.4e}\n")

if __name__ == "__main__":

    cube = Cube("/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/cgcg453_red_mosaic.fits")
    x_factor = 3
    y_factor = 3

    # Step 1: binning
    binned_cube = bin_cube(x_factor, y_factor, cube)
    print("After binning:", np.isnan(binned_cube.data).sum())
    print("Total NaNs in original cube:", np.isnan(cube.data).sum())

    # Step 2: scale luminosity
    redshift_old = 0.025
    redshift_new = 1.0
    scaled_data, dx_arcsec, dy_arcsec = scale_luminosity_for_redshift(binned_cube, redshift_old, redshift_new)
    binned_cube.data = scaled_data
    print("After luminosity scaling:", np.isnan(binned_cube.data).sum())

    # Step 3: apply PSF convolution
    # Convolve to match simulated PSF
    fwhm_real = 0.29  # KCWI resolution in arcsec
    fwhm_target = 0.07  # JWST NIRCam at ~2 μm
    z1 = redshift_old
    z2 = redshift_new

    cube_blurred = convolve_to_match_psf(
        binned_cube,
        dx_arcsec, dy_arcsec,
        fwhm_target_arcsec=fwhm_target,
        z_obs=z1,
        z_sim=z2
    )

    # Visualize PSF-convolved image
    plt.figure(figsize=(6, 5))
    plt.imshow(cube_blurred.data[0], origin='lower', cmap='viridis')
    plt.title("Cube after PSF Matching")
    plt.colorbar(label="Flux")
    plt.show()


    # Visualize original vs scaled luminosity
    plt.subplot(1, 2, 1)
    plt.title("Original data slice")
    im = plt.imshow(cube.data[0], origin='lower')
    plt.colorbar(im)

    plt.subplot(1, 2, 2)
    plt.title("Scaled data slice")
    im2 = plt.imshow(scaled_data[0], origin='lower')
    plt.colorbar(im2)
    plt.show()
    



    # Test full pipeline
    mock_cube = generate_mock_jwst_cube(
        original_cube=cube,
        x_factor=3,
        y_factor=3,
        redshift_old=0.025,
        redshift_new=1.0,
        fwhm_target_arcsec=0.07
    )


    im3 = plt.imshow(mock_cube.data[0], origin='lower')
    plt.title("Mock JWST simulated cube slice")
    plt.colorbar(im3)
    plt.show()

    print("\n--- Scaling factor comparisons ---")


    print_scaling_factors(redshift_old, redshift_new)

    # # Generate cubes with different scaling methods
    # binned_cube_copy1 = bin_cube(x_factor, y_factor, cube)
    # scaled_ang = scale_luminosity_for_redshift(binned_cube_copy1, redshift_old, redshift_new, method="angular")

    # binned_cube_copy2 = bin_cube(x_factor, y_factor, cube)
    # scaled_lum = scale_luminosity_for_redshift(binned_cube_copy2, redshift_old, redshift_new, method="luminosity")

    # binned_cube_copy3 = bin_cube(x_factor, y_factor, cube)
    # scaled_both = scale_luminosity_for_redshift(binned_cube_copy3, redshift_old, redshift_new, method="both")

    # # Visualize the different scaling methods
    # plt.figure(figsize=(15, 4))
    # plt.subplot(1, 3, 1)
    # plt.title("Angular Scaling Only")
    # im4 = plt.imshow(scaled_ang[0], origin='lower', cmap='viridis')
    # plt.colorbar(im4)

    # plt.subplot(1, 3, 2)
    # plt.title("Luminosity Scaling Only")
    # im5 = plt.imshow(scaled_lum[0], origin='lower', cmap='viridis')
    # plt.colorbar(im5)

    # plt.subplot(1, 3, 3)
    # plt.title("Both Angular + Luminosity")
    # im6 = plt.imshow(scaled_both[0], origin='lower', cmap='viridis')
    # plt.colorbar(im6)

    # plt.tight_layout()
    # plt.show()