import numpy as np
import matplotlib.pyplot as plt
from reprojectBinData import reproject_cube_preserve_wcs
from astropy.wcs import WCS as AstropyWCS
from mpdaf.obj import Cube


def crop_cube_to_size(cube, nx_target, ny_target):
    """
    Crop a 3D cube (nz, ny, nx) to the target spatial size, centered.
    Only crops spatial axes (y, x), leaves spectral axis unchanged.

    Parameters
    ----------
    cube : np.ndarray
        Input cube with shape (nz, ny, nx).
    nx_target : int
        Desired number of pixels in x (RA).
    ny_target : int
        Desired number of pixels in y (Dec).

    Returns
    -------
    cube_cropped : np.ndarray
        Cropped cube with shape (nz, ny_target, nx_target).
    """
    nz, ny, nx = cube.shape

    # --- Compute cropping start indices ---
    if nx_target > nx or ny_target > ny:
        raise ValueError(f"Target size ({ny_target}, {nx_target}) "
                         f"is larger than cube ({ny}, {nx}). Cropping cannot enlarge.")

    # Center offsets
    x0 = (nx - nx_target) // 2
    y0 = (ny - ny_target) // 2

    # Apply crop
    cube_cropped = cube[:, y0:y0+ny_target, x0:x0+nx_target]

    # Debug
    print(f"[DEBUG] Cropped cube from {ny}×{nx} → {ny_target}×{nx_target}, "
          f"offsets: y0={y0}, x0={x0}")

    return cube_cropped


def auto_crop_cube(cube, sigma_thresh=2, threshold_nan=0.1, buffer=4, min_crop_size=30, debug=False):
    """
    Automatically crop an MPDAF cube to the smallest bounding box containing significant flux.

    Parameters:
        cube            : MPDAF Cube
        sigma_thresh    : Signal threshold above background (in σ)
        threshold_nan   : NaN fraction cutoff to trigger fallback method
        buffer          : Padding in pixels around valid region
        min_crop_size   : Minimum size of cropped region in pixels
        debug           : If True, shows plot of cropping region

    Returns:
        Cropped MPDAF Cube
    """
    img = cube.mean(axis=0)  # less biased than sum
    data = img.data
    nan_fraction = np.isnan(data).sum() / data.size

    print(f"NaN fraction in projected image: {nan_fraction:.3f}")

    if nan_fraction <= threshold_nan:
        print("Cropping using signal above background ...")
        try:
            bkg, std = img.background()
            signal_mask = data > (bkg + sigma_thresh * std)

            if not signal_mask.any():
                raise ValueError("No signal found above background.")

            y_valid, x_valid = np.where(signal_mask)
        except Exception as e:
            print(f"[Warning] Background cropping failed: {e}")
            print("Falling back to NaN-based cropping ...")
            return crop_nan_mask_fallback(cube, buffer, min_crop_size, debug)
    else:
        print("Too many NaNs — cropping based on finite values ...")
        return crop_nan_mask_fallback(cube, buffer, min_crop_size, debug)

    ny, nx = data.shape
    y_min = max(y_valid.min() - buffer, 0)
    y_max = min(y_valid.max() + 1 + buffer, ny)
    x_min = max(x_valid.min() - buffer, 0)
    x_max = min(x_valid.max() + 1 + buffer, nx)

    # Ensure minimum crop size
    if (y_max - y_min) < min_crop_size:
        y_max = min(ny, y_min + min_crop_size)
    if (x_max - x_min) < min_crop_size:
        x_max = min(nx, x_min + min_crop_size)

    print(f"Final cropped FOV: y=[{y_min}:{y_max}], x=[{x_min}:{x_max}]")

    if debug:
        _plot_crop_region(data, x_min, x_max, y_min, y_max, title="Background-based Crop")

    return cube[:, y_min:y_max, x_min:x_max]


def crop_nan_mask_fallback(cube, buffer=4, min_crop_size=30, debug=False):
    """
    Fallback: Crop MPDAF cube based on where any spectral layer is finite.
    """
    data = cube.data
    valid_mask = np.isfinite(data).any(axis=0)
    y_valid, x_valid = np.where(valid_mask)

    if len(y_valid) == 0 or len(x_valid) == 0:
        raise ValueError("No finite spatial pixels found.")

    ny, nx = data.shape[1:]
    y_min = max(y_valid.min() - buffer, 0)
    y_max = min(y_valid.max() + 1 + buffer, ny)
    x_min = max(x_valid.min() - buffer, 0)
    x_max = min(x_valid.max() + 1 + buffer, nx)

    # Ensure minimum crop size
    if (y_max - y_min) < min_crop_size:
        y_max = min(ny, y_min + min_crop_size)
    if (x_max - x_min) < min_crop_size:
        x_max = min(nx, x_min + min_crop_size)

    print(f"[Fallback] Cropped to: y=[{y_min}:{y_max}], x=[{x_min}:{x_max}]")

    if debug:
        img = cube.mean(axis=0).data
        _plot_crop_region(img, x_min, x_max, y_min, y_max, title="NaN-based Crop")

    return cube[:, y_min:y_max, x_min:x_max]

def trim_empty_edges(cube, buffer=0, debug=False):
    """
    Trim spatial borders of an MPDAF cube where all values are NaN.
    
    Parameters:
        cube   : MPDAF Cube
        buffer : Optional number of pixels to keep around valid region
        debug  : If True, plot trimmed region
    
    Returns:
        Trimmed MPDAF Cube
    """
    data = cube.data
    valid_mask = np.isfinite(data).any(axis=0)
    y_valid, x_valid = np.where(valid_mask)

    if len(y_valid) == 0 or len(x_valid) == 0:
        raise ValueError("No finite spatial pixels found.")

    ny, nx = data.shape[1:]

    y_min = max(y_valid.min() - buffer, 0)
    y_max = min(y_valid.max() + 1 + buffer, ny)
    x_min = max(x_valid.min() - buffer, 0)
    x_max = min(x_valid.max() + 1 + buffer, nx)

    if debug:
        from matplotlib import pyplot as plt
        img = cube.sum(axis=0).data
        plt.imshow(img, origin='lower', cmap='gray')
        plt.axhline(y_min, color='red', linestyle='--')
        plt.axhline(y_max, color='red', linestyle='--')
        plt.axvline(x_min, color='red', linestyle='--')
        plt.axvline(x_max, color='red', linestyle='--')
        plt.title("Trimmed Region")
        plt.show()

    return cube[:, y_min:y_max, x_min:x_max]


def _plot_crop_region(image_data, x_min, x_max, y_min, y_max, title=""):
    """
    Helper function to show the cropping region on the 2D image.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(image_data, origin='lower', cmap='viridis',
               vmin=np.nanpercentile(image_data, 5),
               vmax=np.nanpercentile(image_data, 99))
    plt.colorbar(label='Integrated Flux')
    plt.axhline(y_min, color='r', linestyle='--')
    plt.axhline(y_max, color='r', linestyle='--')
    plt.axvline(x_min, color='r', linestyle='--')
    plt.axvline(x_max, color='r', linestyle='--')
    plt.title(f"Cropping Region ({title})")
    plt.xlabel('X [pix]')
    plt.ylabel('Y [pix]')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    file_path = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/cgcg453_red_mosaic.fits"
    output_path = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/cropped_cube.fits"

    cube = Cube(file_path)

    ### Pixel scale logic
    original_pixscale = 0.145  # arcsec/pix
    target_pixscale = 0.031    # arcsec/pix
    scale_factor = original_pixscale / target_pixscale

    # Compute new shape to preserve FOV
    new_shape = (
        int(cube.shape[1] * scale_factor),
        int(cube.shape[2] * scale_factor)
    )

    # Modify WCS for new pixel scale
    original_wcs_astropy = AstropyWCS(cube.wcs.to_header()).sub([1, 2])
    target_header = original_wcs_astropy.to_header()
    target_header['CDELT1'] = -target_pixscale / 3600.0  # arcsec → deg
    target_header['CDELT2'] =  target_pixscale / 3600.0
    target_header['NAXIS1'], target_header['NAXIS2'] = new_shape[1], new_shape[0]
    target_wcs = AstropyWCS(target_header)

    # Reproject and crop
    cube_reprojected = reproject_cube_preserve_wcs(cube, target_wcs, new_shape)
    cropped_cube = trim_empty_edges(cube_reprojected, buffer=2, debug=True)
    cropped_cube.write(output_path, savemask='primary')

    print(f"Cropped cube saved to: {output_path}")
