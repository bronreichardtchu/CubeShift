"""Spatial cropping of mpdaf Cubes."""
from __future__ import annotations

import numpy as np


def _finite_bbox(data, buffer=0):
    """Bounding box (y_min, y_max, x_min, x_max) of spaxels finite and unmasked in any slice."""
    valid_mask = (np.isfinite(np.ma.getdata(data)) & ~np.ma.getmaskarray(data)).any(axis=0)
    y_valid, x_valid = np.where(valid_mask)
    if len(y_valid) == 0 or len(x_valid) == 0:
        raise ValueError("No finite spatial pixels found.")
    ny, nx = data.shape[1:]
    return (max(y_valid.min() - buffer, 0), min(y_valid.max() + 1 + buffer, ny),
            max(x_valid.min() - buffer, 0), min(x_valid.max() + 1 + buffer, nx))


def _enforce_min_size(lo, hi, n, min_size):
    if (hi - lo) < min_size:
        hi = min(n, lo + min_size)
    return lo, hi


def crop_cube_to_size(cube, nx_target, ny_target):
    """Centre-crop the spatial axes of a cube to (ny_target, nx_target)."""
    nz, ny, nx = cube.shape
    if nx_target > nx or ny_target > ny:
        raise ValueError(f"Target size ({ny_target}, {nx_target}) is larger than cube ({ny}, {nx}). "
                         "Cropping cannot enlarge.")
    x0 = (nx - nx_target) // 2
    y0 = (ny - ny_target) // 2
    return cube[:, y0:y0 + ny_target, x0:x0 + nx_target]


def auto_crop_cube(cube, sigma_thresh=2, threshold_nan=0.1, buffer=4, min_crop_size=30, debug=False):
    """Crop to the smallest box containing flux above background.

    Falls back to :func:`crop_nan_mask_fallback` if the collapsed image has more
    than ``threshold_nan`` NaNs or no significant signal.

    Parameters
    ----------
    cube : mpdaf.obj.Cube
    sigma_thresh : float
        Signal threshold above background, in sigma.
    threshold_nan : float
        NaN fraction above which the NaN-based fallback is used.
    buffer : int
        Padding in pixels around the valid region.
    min_crop_size : int
        Minimum size of the cropped region in pixels.
    debug : bool
        Plot the crop region.
    """
    img = cube.mean(axis=0)
    data = img.data
    nan_fraction = np.isnan(data).sum() / data.size

    if nan_fraction > threshold_nan:
        return crop_nan_mask_fallback(cube, buffer, min_crop_size, debug)
    try:
        bkg, std = img.background()
        signal_mask = data > (bkg + sigma_thresh * std)
        if not signal_mask.any():
            raise ValueError("No signal found above background.")
        y_valid, x_valid = np.where(signal_mask)
    except Exception as e:
        print(f"[Warning] Background cropping failed: {e}; falling back to NaN-based cropping")
        return crop_nan_mask_fallback(cube, buffer, min_crop_size, debug)

    ny, nx = data.shape
    y_min, y_max = _enforce_min_size(max(y_valid.min() - buffer, 0), min(y_valid.max() + 1 + buffer, ny),
                                     ny, min_crop_size)
    x_min, x_max = _enforce_min_size(max(x_valid.min() - buffer, 0), min(x_valid.max() + 1 + buffer, nx),
                                     nx, min_crop_size)
    if debug:
        _plot_crop_region(data, x_min, x_max, y_min, y_max, title="Background-based Crop")
    return cube[:, y_min:y_max, x_min:x_max]


def crop_nan_mask_fallback(cube, buffer=4, min_crop_size=30, debug=False):
    """Crop to where any spectral slice is finite, with a minimum size."""
    ny, nx = cube.shape[1:]
    y_min, y_max, x_min, x_max = _finite_bbox(cube.data, buffer)
    y_min, y_max = _enforce_min_size(y_min, y_max, ny, min_crop_size)
    x_min, x_max = _enforce_min_size(x_min, x_max, nx, min_crop_size)
    if debug:
        _plot_crop_region(cube.mean(axis=0).data, x_min, x_max, y_min, y_max, title="NaN-based Crop")
    return cube[:, y_min:y_max, x_min:x_max]


def trim_empty_edges(cube, buffer=0, debug=False):
    """Trim spatial borders where every value is NaN or masked.

    Parameters
    ----------
    cube : mpdaf.obj.Cube
    buffer : int
        Pixels to keep around the valid region.
    debug : bool
        Plot the trimmed region.
    """
    y_min, y_max, x_min, x_max = _finite_bbox(cube.data, buffer)
    if debug:
        _plot_crop_region(cube.sum(axis=0).data, x_min, x_max, y_min, y_max, title="Trimmed Region")
    return cube[:, y_min:y_max, x_min:x_max]


def _plot_crop_region(image_data, x_min, x_max, y_min, y_max, title=""):
    """Show a crop region on a 2D image."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.imshow(image_data, origin="lower", cmap="viridis",
               vmin=np.nanpercentile(image_data, 5), vmax=np.nanpercentile(image_data, 99))
    plt.colorbar(label="Integrated Flux")
    for y in (y_min, y_max):
        plt.axhline(y, color="r", linestyle="--")
    for x in (x_min, x_max):
        plt.axvline(x, color="r", linestyle="--")
    plt.title(f"Cropping Region ({title})")
    plt.xlabel("X [pix]")
    plt.ylabel("Y [pix]")
    plt.tight_layout()
    plt.show()
