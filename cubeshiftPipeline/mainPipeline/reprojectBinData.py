import copy
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo
from reproject import reproject_interp
from mpdaf.obj import Cube, coords
from astropy.wcs import WCS as AstropyWCS
from astropy.io import fits

def get_spectral_wcs_keywords_from_wave(cube):
    wave_coords = cube.wave.coord()  # 1D numpy array of wavelengths
    nz = len(wave_coords)

    crpix3 = (nz + 1) / 2  # FITS 1-based center pixel
    # Interpolate wavelength value at reference pixel
    from numpy import interp
    crval3 = interp(crpix3 - 1, np.arange(nz), wave_coords)

    # Approximate delta wavelength (per pixel)
    diffs = np.diff(wave_coords)
    cdelt3 = np.median(diffs) if len(diffs) > 0 else 1.0

    ctype3 = 'WAVE'  # or 'AWAV', 'FREQ', etc. Use what fits your data best
    cunit3 = str(cube.wave.unit) if cube.wave.unit else 'Angstrom'

    return {
        'CTYPE3': ctype3,
        'CRVAL3': crval3,
        'CRPIX3': crpix3,
        'CDELT3': cdelt3,
        'CUNIT3': cunit3,
    }


def reproject_cube_preserve_wcs(cube, target_spatial_wcs_astropy, shape_out):
    nz = cube.shape[0]
    ny, nx = shape_out

    new_data = np.empty((nz, ny, nx), dtype=cube.data.dtype)

    # Get original astropy WCS (2D spatial only)
    original_header = cube.wcs.to_header()
    original_wcs_astropy = AstropyWCS(original_header)

    # Get spectral axis keywords from wave
    spectral_wcs_keys = get_spectral_wcs_keywords_from_wave(cube)

    for i in range(nz):
        slice2d = cube.data[i]

        # Build 2D WCS for slice - from 3D WCS drop spectral axis (axis=0)
        # BUT dropaxis can fail if non-zero off-diagonal elements, so build manually:

        # Create a 2D WCS using only spatial axes keys from original_header:
        slice_header = fits.Header()
        for key in original_header.keys():
            if key.endswith('1') or key.endswith('2'):
                slice_header[key] = original_header[key]
        # Also copy WCSAXES=2 etc.
        slice_header['WCSAXES'] = 2

        slice_wcs = AstropyWCS(slice_header)

        # Reproject the 2D slice to target spatial WCS
        reprojected_slice, footprint = reproject_interp(
            (slice2d, slice_wcs),
            target_spatial_wcs_astropy,
            shape_out=shape_out,
        )
        new_data[i] = reprojected_slice

    # Build output WCS header from target spatial WCS header
    target_header = target_spatial_wcs_astropy.to_header()

    # Add spectral axis keywords manually
    for key, val in spectral_wcs_keys.items():
        target_header[key] = val

    target_header['NAXIS'] = 3
    target_header['NAXIS1'] = nx
    target_header['NAXIS2'] = ny
    target_header['NAXIS3'] = nz

    # Create MPDAF WCS from header
    new_wcs = coords.WCS(target_header)

    # Create new cube with reprojected data and new WCS
    new_cube = Cube(data=new_data, wcs=new_wcs, unit=cube.unit)
    new_cube.wave = cube.wave.copy()

    return new_cube

def extract_spatial_wcs(original_wcs):
    """
    Given a 3D Astropy WCS, construct a new 2D spatial WCS from axes 1 and 2,
    ignoring the spectral axis and avoiding non-separable dropaxis issues.

    Parameters:
        original_wcs : astropy.wcs.WCS
            Original 3D WCS object.

    Returns:
        spatial_wcs : astropy.wcs.WCS
            New 2D WCS object containing only spatial axes.
    """
    original_header = original_wcs.to_header()

    spatial_header = fits.Header()

    # Copy all keywords relevant to spatial axes 1 and 2
    # WCS keywords end with 1 or 2 for spatial axes (e.g. CTYPE1, CRVAL2, etc.)
    # Also copy WCS general keywords like WCSNAME, RADESYS, EQUINOX

    for key, value in original_header.items():
        if (
            (key.endswith('1') or key.endswith('2'))
            or key in ['WCSNAME', 'RADESYS', 'EQUINOX', 'LONPOLE', 'LATPOLE']
        ):
            spatial_header[key] = value

    # Set NAXIS to 2 for 2D WCS
    spatial_header['NAXIS'] = 2

    # Sometimes you need to remove or adjust keywords related to spectral axis:
    for key in list(spatial_header.keys()):
        # Remove any keys that explicitly reference axis 3 or spectral axis
        if key.endswith('3'):
            del spatial_header[key]

    spatial_wcs = WCS(spatial_header)

    return spatial_wcs



def construct_target_wcs_mpdaf(cube, x_pixel_scale_arcsec, y_pixel_scale_arcsec):
    """
    Create a new MPDAF WCS object with updated spatial pixel scales.

    Parameters:
        cube: MPDAF Cube object
        x_pixel_scale_arcsec, y_pixel_scale_arcsec: target pixel scale in arcsec

    Returns:
        new MPDAF WCS object with modified pixel scale
    """
    wcs = copy.deepcopy(cube.wcs)
    # MPDAF WCS axis order: 0=spectral, 1=DEC (y), 2=RA (x)

    # RA axis step is negative to match FITS conventions
    wcs.set_step(2, -x_pixel_scale_arcsec * u.arcsec)  # RA axis
    wcs.set_step(1, y_pixel_scale_arcsec * u.arcsec)   # DEC axis

    return wcs




def calculate_spatial_resampling_factor(z_old, z_new, original_pixel_scale_arcsec, target_telescope_resolution_arcsec):
    """
    Calculate spatial resampling factor when simulating observation at a new redshift.

    Returns:
        rebin_factor (float), new_pixel_scale (Quantity with arcsec)
    """
    # Physical size per pixel at old redshift (kpc)
    physical_size_kpc = (original_pixel_scale_arcsec * u.arcsec *
                         cosmo.kpc_proper_per_arcmin(z_old).to(u.kpc / u.arcsec))

    # Pixel scale needed at new redshift to keep physical size fixed
    new_spaxel_size_arcsec = (physical_size_kpc /
                              cosmo.kpc_proper_per_arcmin(z_new).to(u.kpc / u.arcsec))

    # Ensure new pixel scale is not smaller than telescope resolution
    new_spaxel_size_arcsec_value = max(new_spaxel_size_arcsec.value, target_telescope_resolution_arcsec)

    rebin_factor = new_spaxel_size_arcsec_value / original_pixel_scale_arcsec

    return rebin_factor, new_spaxel_size_arcsec_value * u.arcsec


def crop_valid_fov(cube):
    """
    Crop cube to spatial region where data is finite in any spectral layer.
    """
    data = cube.data
    valid_mask = np.isfinite(data).any(axis=0)  # 2D spatial mask
    y_valid, x_valid = np.where(valid_mask)

    y_min, y_max = y_valid.min(), y_valid.max() + 1
    x_min, x_max = x_valid.min(), x_valid.max() + 1

    return cube[:, y_min:y_max, x_min:x_max]



def test_wcs_preservation(input_path, output_path, pixscale_x, pixscale_y):
    """
    Load cube, reproject to new pixel scale, save, and verify WCS header preservation.
    """
    print("\n--- Running WCS Preservation Test ---")

    cube = Cube(input_path)
    original_wcs = cube.wcs

    target_wcs = construct_target_wcs_mpdaf(cube, pixscale_x, pixscale_y)
    shape_out = (cube.shape[1], cube.shape[2])

    cube_resampled = reproject_cube_with_wcs(cube, target_wcs, shape_out=shape_out)
    cube_resampled.write(output_path, savemask='primary')

    check_wcs_headers_match(original_wcs, output_path)


def check_wcs_headers_match(original_wcs, output_fits_path):
    original_header = original_wcs.to_header()
    saved_header = fits.getheader(output_fits_path)

    keys_to_check = [
        'CTYPE1', 'CTYPE2',
        'CRVAL1', 'CRVAL2',
        'CDELT1', 'CDELT2',
        'CRPIX1', 'CRPIX2',
        'CUNIT1', 'CUNIT2'
    ]

    print("Checking WCS header values...")
    for key in keys_to_check:
        val_orig = original_header.get(key)
        val_saved = saved_header.get(key)
        if val_orig is None or val_saved is None:
            print(f"⚠️ Missing key {key} in one of the headers.")
            continue
        if np.allclose(float(val_orig), float(val_saved), rtol=1e-6):
            print(f" {key} matches: {val_orig}")
        else:
            print(f" {key} mismatch: original={val_orig}, saved={val_saved}")
    print("--- Done ---\n")

if __name__ == "__main__":
    input_path = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/cgcg453_red_mosaic.fits"
    output_path = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/output_resampled_cube.fits"

    cube = Cube(input_path)

    print("Original cube WCS type:", type(cube.wcs))
    print("Cube shape:", cube.shape)
    print("Number of dimensions:", len(cube.shape))

    original_header = cube.wcs.to_header()
    print("Original header keys:", list(original_header.keys()))

    original_wcs_astropy = AstropyWCS(original_header)

    # Extract spatial 2D WCS (axes 1 and 2)
    spatial_wcs = original_wcs_astropy.sub([1, 2])

    # Modify pixel scale (degrees)
    target_header = spatial_wcs.to_header()
    target_header['CDELT1'] = -0.03 / 3600.0
    target_header['CDELT2'] = 0.03 / 3600.0

    ny, nx = cube.shape[1], cube.shape[2]
    target_header['NAXIS1'] = nx
    target_header['NAXIS2'] = ny

    target_spatial_wcs_astropy = AstropyWCS(target_header)

    shape_out = (ny, nx)
    cube_reprojected = reproject_cube_preserve_wcs(cube, target_spatial_wcs_astropy, shape_out)

    cube_reprojected.write(output_path, savemask='primary')

    print(f"Reprojected cube saved to {output_path}")