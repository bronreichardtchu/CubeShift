import numpy as np
from mpdaf.obj import WaveCoord
from astropy import units as u
import matplotlib.pyplot as plt
from matchSpectral import precomputed_match_spectral_resolution_variable_kernel, resample_spectral_axis, apply_dispersion_pipeline
from zWavelengths import redshift_wavelength_axis
from binData import resample_cube_to_telescope_and_redshift, bin_cube, calculate_spatial_resampling_factor, abs_calculate_spatial_resampling_factor
from simulateObs import scale_luminosity_for_redshift, convolve_to_match_psf
from reprojectBinData import reproject_cube_preserve_wcs
from astropy.wcs import WCS
from cropCube import auto_crop_cube, trim_empty_edges
from applyFilter import apply_transmission_filter
from scipy.ndimage import gaussian_filter1d
from mpdaf.obj import Image, plot_rgb
from astropy.wcs.utils import proj_plane_pixel_scales
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.patches import Patch
from hBetaHgammaRatio import plot_hbeta_hgamma_ratio, plot_hbeta_hgamma_ratio_amp, plot_hbeta_hgamma_ratio_amp_soft
from main import telescope_specs, Telescope
from mpdaf.obj import WCS as MPDAF_WCS
from scipy.optimize import curve_fit
from astropy.io import fits
from astropy.modeling.models import Gaussian2D
from mpdaf.obj import Cube
from astropy.io.fits import Header
from main import simulate_observation


# 2d array (num py to stack) to represent the cube in the x, y direction 
# every spaxel in the cube is exactly the same, it is the same individual gaussian repeated in every spaxel
# multiply every spaxel by sersic so it looks like a real galaxy (blob in the middle)
def make_emission_line_cube(nx=100, ny=100, nw=50,
                            lam_min=5000.0, lam_max=5100.0,
                            fwhm_spatial=2.0, fwhm_spectral=2.0,
                            pixscale_arcsec=0.03):
    # --- spectral axis: use WaveCoord via crval/cdelt/crpix (no array) ---
    cdelt = (lam_max - lam_min) / (nw - 1)  # so last pixel is lam_max
    wave = WaveCoord(crval=lam_min, cdelt=cdelt, crpix=1,
                     ctype='WAVE', cunit='Angstrom')

    # --- synthetic 3D Gaussian line ---
    x = np.arange(nx)
    y = np.arange(ny)
    xv, yv = np.meshgrid(x, y, indexing='xy')
    spatial = np.exp(-((xv - (nx - 1)/2.0)**2 + (yv - (ny - 1)/2.0)**2) /
                     (2.0 * fwhm_spatial**2))

    spec_pix = np.arange(nw)
    spec_center = (nw - 1)/2.0
    spectral = np.exp(-0.5 * ((spec_pix - spec_center)/fwhm_spectral)**2)

    data = np.empty((nw, ny, nx), dtype=float)
    for i in range(nw):
        data[i] = spatial * spectral[i]

    # --- build a FITS header for SPATIAL WCS, then MPDAF WCS from it ---
    h = Header()
    h['NAXIS']  = 2
    h['NAXIS1'] = nx
    h['NAXIS2'] = ny
    h['CTYPE1'] = 'RA---TAN'
    h['CTYPE2'] = 'DEC--TAN'
    h['CRVAL1'] = 0.0
    h['CRVAL2'] = 0.0
    # reference pixel at center in 1-based FITS convention:
    h['CRPIX1'] = (nx + 1)/2.0
    h['CRPIX2'] = (ny + 1)/2.0
    # pixel scale in degrees/pixel (keep positive; sign handling isn’t critical here)
    scale_deg = pixscale_arcsec / 3600.0
    h['CDELT1'] = scale_deg
    h['CDELT2'] = scale_deg
    h['CUNIT1'] = 'deg'
    h['CUNIT2'] = 'deg'

    wcs_spatial = MPDAF_WCS(h)  # <-- MPDAF WCS, not astropy.wcs.WCS

    # --- make the cube ---
    cube = Cube(data=data, wave=wave, wcs=wcs_spatial)

    # ensure spectral keywords exist for your redshift function’s header access
    hdr = cube.data_header
    hdr['CRVAL3'] = lam_min
    hdr['CDELT3'] = cdelt
    hdr['CRPIX3'] = 1
    hdr['CTYPE3'] = 'WAVE'
    hdr['CUNIT3'] = 'Angstrom'

    # simple diagnostic spectrum
    summed = data.sum(axis=(1, 2))
    plt.figure()
    plt.plot(lam_min + cdelt * np.arange(nw), summed)
    plt.xlabel('Wavelength [Angstrom]')
    plt.ylabel('Summed flux')
    plt.title('Synthetic emission line (diagnostic)')
    plt.tight_layout()
    plt.show()

    return cube

def make_gauss_cube(nx=100, ny=100, nw=50,
                            lam_min=5000.0, lam_max=5100.0,
                            fwhm_spectral=2.0,
                            pixscale_arcsec=0.03,
                            sersic_n=1.0, reff=10.0):
    # --- spectral axis ---
    cdelt = (lam_max - lam_min) / (nw - 1)
    wave = WaveCoord(crval=lam_min, cdelt=cdelt, crpix=1,
                     ctype='WAVE', cunit='Angstrom')

    # --- make the 1D Gaussian spectrum ---
    spec_pix = np.arange(nw)
    spec_center = (nw - 1)/2.0
    spectrum = np.exp(-0.5 * ((spec_pix - spec_center)/fwhm_spectral)**2)

    # --- make the 2D Sersic spatial profile ---
    x = np.arange(nx)
    y = np.arange(ny)
    xv, yv = np.meshgrid(x, y, indexing='xy')

    sersic = Sersic2D(amplitude=1.0, r_eff=reff,
                      n=sersic_n, x_0=(nx-1)/2, y_0=(ny-1)/2)
    spatial = sersic(xv, yv)

    # --- normalize spatial profile so total flux = 1 ---
    spatial /= spatial.sum()

    # --- outer product: every spaxel gets the Gaussian scaled by local Sersic flux ---
    data = spectrum[:, None, None] * spatial[None, :, :]



    # --- build FITS WCS header ---
    h = Header()
    h['NAXIS']  = 2
    h['NAXIS1'] = nx
    h['NAXIS2'] = ny
    h['CTYPE1'] = 'RA---TAN'
    h['CTYPE2'] = 'DEC--TAN'
    h['CRVAL1'] = 0.0
    h['CRVAL2'] = 0.0
    h['CRPIX1'] = (nx + 1)/2.0
    h['CRPIX2'] = (ny + 1)/2.0
    scale_deg = pixscale_arcsec / 3600.0
    h['CDELT1'] = scale_deg
    h['CDELT2'] = scale_deg
    h['CUNIT1'] = 'deg'
    h['CUNIT2'] = 'deg'

    wcs_spatial = MPDAF_WCS(h)

    cube = Cube(data=data, wave=wave, wcs=wcs_spatial)

    # add spectral WCS keywords
    hdr = cube.data_header
    hdr['CRVAL3'] = lam_min
    hdr['CDELT3'] = cdelt
    hdr['CRPIX3'] = 1
    hdr['CTYPE3'] = 'WAVE'
    hdr['CUNIT3'] = 'Angstrom'

    # --- diagnostic spectrum (sum over spaxels) ---
    summed = data.sum(axis=(1, 2))
    plt.figure()
    plt.plot(lam_min + cdelt * np.arange(nw), summed)
    plt.xlabel('Wavelength [Angstrom]')
    plt.ylabel('Summed flux')
    plt.title('Summed emission line (Gaussian × Sersic)')
    plt.show()

    return cube



# gaussian model for fit
def gaussian(x, amp, mu, sigma, offset):
    return amp * np.exp(-0.5 * ((x - mu)/sigma)**2) + offset

def measure_sigma_from_cube(cube):
    # get wave axis (works whether cube.wave is WaveCoord or ndarray)
    if hasattr(cube.wave, 'coord'):
        wave_axis = cube.wave.coord()
    else:
        wave_axis = np.asarray(cube.wave)

    data = cube.data  # shape (nw, ny, nx)
    nw = data.shape[0]
    cdelt = wave_axis[1] - wave_axis[0]
    lam_center = wave_axis[int((nw-1)//2)]  # central wavelength

    # expected sigma from how you built the cube
    # your code used fwhm_spectral as sigma_pixels:
    sigma_pix_expected = 2.0
    sigma_lambda_expected = sigma_pix_expected * cdelt
    fwhm_lambda_expected = sigma_lambda_expected * 2.3548200450309493
    sigma_v_expected = (sigma_lambda_expected / lam_center) * 299792.458

    # extract central spaxel spectrum
    iy, ix = data.shape[1]//2, data.shape[2]//2
    spec = data[:, iy, ix].astype(float)

    # initial guesses for fit
    amp0 = spec.max() - np.median(spec)
    mu0 = wave_axis[np.argmax(spec)]
    sigma0 = max(sigma_lambda_expected, cdelt)
    offset0 = np.median(spec)
    p0 = [amp0, mu0, sigma0, offset0]

    # Fit
    try:
        popt, pcov = curve_fit(gaussian, wave_axis, spec, p0=p0, maxfev=5000)
        amp, mu, sigma_lambda_fit, offset = popt
        fwhm_lambda_fit = sigma_lambda_fit * 2.3548200450309493
        sigma_pix_fit = sigma_lambda_fit / cdelt
        sigma_v_fit = (sigma_lambda_fit / mu) * 299792.458

        result = {
            'sigma_lambda_expected_A': sigma_lambda_expected,
            'fwhm_lambda_expected_A': fwhm_lambda_expected,
            'sigma_v_expected_kms': sigma_v_expected,
            'sigma_lambda_fit_A': sigma_lambda_fit,
            'fwhm_lambda_fit_A': fwhm_lambda_fit,
            'sigma_pix_fit': sigma_pix_fit,
            'sigma_v_fit_kms': sigma_v_fit,
            'mu_fit_A': mu,
            'amp_fit': amp,
            'offset_fit': offset,
            'popt': popt,
            'pcov': pcov
        }
    except Exception as e:
        result = {'error': str(e)}
    return result


# Pipeline:

def simulate_observation_return_all_cubes(cube, telescope_name, z_obs, z_sim, source_telescope, target_telescope):
    """
    Simulates how a cube would appear if observed with a different telescope at a different redshift.
    Pipeline:
        1. Redshift wavelength axis
        2. Spatially rebin to match resolution
        3. Scale luminosity for redshift effects
        4. Convolve spatially to match PSF
        5. Match spectral resolution (convolve in lambda)
        6. Match spectral sampling (resample lambda axis)

    Parameters
    ----------
    cube : mpdaf.obj.Cube
        Original data cube (e.g. from Keck/KCWI).
    telescope_name : str
        Key to look up specs from the telescope_specs dictionary.
    z_obs : float, optional
        Original redshift. If None, assumed 0.

    Returns
    -------
    cube_sim : mpdaf.obj.Cube
        The transformed cube simulating the target telescope's observation.
    """
    z = z_sim # for file naming 

    telescope = telescope_specs[telescope_name]

    print(f"Simulating observation as seen by {telescope.name} at z = {z_sim}")

    # STEP 1: Redshift wavelength axis
    redshifted_cube, lam_new = redshift_wavelength_axis(cube, z_obs, z_sim)
    print("WCS after redshifting:", redshifted_cube.wcs)

    output_path = f"/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/Output_cubes/test_z_{z}_redshifted.fits"
    redshifted_cube.write(output_path)
    print(f"✔ Redshifted cube saved to: {output_path}")


    # STEP 2: Apply transmission filter 
    line_rest_wavelength = 5007 # For [OIII], change this for other emission lines!!!!
    transmission_cube, trans_filter_name = apply_transmission_filter(redshifted_cube, z_sim, line_rest_wavelength)

    output_path = f"/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/Output_cubes/test_z_{z}_{trans_filter_name}_transmission.fits"
    transmission_cube.write(output_path)
    print(f"✔ Transmission filter applied cube saved to: {output_path}")


    # STEP 3: Rebin spatially to match physical resolution
    cube_resampled, bin_factors = resample_cube_to_telescope_and_redshift(
        transmission_cube,
        source_telescope,
        target_telescope,
        z_obs,
        z_sim 
    )

    print(f"Final bin factors applied: {bin_factors}")
    output_path = f"/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/Output_cubes/test_z_{z}_rebinned.fits"
    cube_resampled.write(output_path)
    print(f"✔ Rebinned cube saved to: {output_path}")


    # STEP 3: Scale luminosity to account for angular size and flux dimming
    scaled_data = scale_luminosity_for_redshift(cube_resampled, z_obs, z_sim, method='both')
    cube_resampled.data = scaled_data

    # STEP 4: Convolve spatially to match target PSF
    cube_psf = convolve_to_match_psf(
        cube_resampled,
        fwhm_real_arcsec=cube.wcs.get_axis_increments(unit=u.arcsec)[0],
        fwhm_target_arcsec=telescope.spatial_fwhm,
        z_obs=z_obs,
        z_sim=z_sim
    )
    output_path = f"/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/Output_cubes/test_z_{z}_{trans_filter_name}_psf.fits"
    cube_psf.write(output_path)
    print(f"✔ PSF convolved cube saved to: {output_path}")

    # STEP 5: Apply dispersion filter and spectral resolution matching

    blurred_cube, best_disperser = apply_dispersion_pipeline(cube_psf, z_obs, z_sim)

    # Removed unecessary reproject step and fixed binning 

    # STEP 6:  Match spectral sampling (resample lambda axis)
    telescope = telescope_specs[telescope_name]  # e.g., "JWST_NIRCam"
    R = telescope.spectral_resolution           
    lam_center = np.median(blurred_cube.wave.coord())
    bin_width = lam_center / R

    wavelengths = blurred_cube.wave.coord()
    start = wavelengths[0]
    end = wavelengths[-1]

    new_wave_axis = np.arange(start, end + bin_width, bin_width)
    cube = resample_spectral_axis(blurred_cube, new_wave_axis)

    output_path = f"/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/Output_cubes/test_z_{z}_{trans_filter_name}_{best_disperser}_lsf.fits"
    cube.write(output_path)
    print(f"✔ LSF spectral resolution matched cube saved to: {output_path}")

    return redshifted_cube, transmission_cube, cube_resampled, cube_psf, blurred_cube, cube

from astropy.modeling import models, fitting

def measure_emission_line_sigma(cube, line_center_guess, z_sim, region_width=50):
    """
    Measure the sigma of an emission line in the final cube spectrum.
    
    Parameters
    ----------
    cube : mpdaf.obj.Cube
        The processed cube after pipeline.
    line_center_guess : float
        Rest wavelength of the line (e.g. 5007 for [OIII]).
    z_sim : float
        The simulated redshift applied to the cube.
    region_width : float
        Wavelength window around the line (in Angstroms).
    """
    # Extract a spectrum (collapse spatial dimensions)
    spectrum = cube.sum(axis=(1,2))  # collapse to 1D
    lam = spectrum.wave.coord()
    flux = spectrum.data
    
    # Shift line center to observed frame
    line_center_obs = line_center_guess * (1 + z_sim)
    
    # Select region around line
    mask = (lam > line_center_obs - region_width) & (lam < line_center_obs + region_width)
    lam_region = lam[mask]
    flux_region = flux[mask]
    
    # Fit Gaussian to the line
    g_init = models.Gaussian1D(amplitude=flux_region.max(), mean=line_center_obs, stddev=2.0)
    fit_g = fitting.LevMarLSQFitter()
    g_fit = fit_g(g_init, lam_region, flux_region)
    
    sigma = g_fit.stddev.value
    fwhm = 2.355 * sigma
    R_measured = g_fit.mean.value / fwhm
    
    print(f"Measured σ = {sigma:.2f} Å, FWHM = {fwhm:.2f} Å")
    print(f"Effective R from fit = {R_measured:.1f}")
    
    return sigma, fwhm, R_measured



if __name__ == "__main__":


    test_cube = f"/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/Output_cubes/gauss_test_cube.fits"
    # test_cube.write(output_path, overwrite=True)
    # print(f"test cube saved to: {output_path}")

    # # Run pipeline
    # res = measure_sigma_from_cube(test_cube) 
    # for k,v in res.items():
    #     print(f"{k}: {v}")

    source_telescope=telescope_specs["Keck_KCWI"]
    target_telescope=telescope_specs["JWST_NIRSpec"]
    redshifted_cube, _, _ = simulate_observation(
        test_cube, telescope_name="JWST_NIRSpec", z_obs=0.01, z_sim=0.5,
        source_telescope=source_telescope, target_telescope=target_telescope
    )

    # sigma, fwhm, R_measured = measure_emission_line_sigma(final_cube, 5050, z_sim=0.5) 
    # do this for each cube (or are there some other metrics I should be printing? or just look at qfits
