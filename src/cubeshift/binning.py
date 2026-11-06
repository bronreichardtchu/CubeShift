"""Spatial binning of mpdaf Cubes/Images by integer factors.

Adapted from ``mpdaf.DataArray._rebin`` to allow summing, averaging or taking
the median, and to return either flux (sum) or surface brightness.
"""
from __future__ import annotations

import numpy as np
from astropy import units
from astropy.cosmology import WMAP9
from mpdaf.obj import Cube, Image

from .cosmology import kpc_per_arcsec

_MARGINS = ("center", "origin", "left", "right")
_METHODS = ("sum", "mean", "median")


def _reduction_factor_to_array(x_factor: int, y_factor: int, data_cube: Cube | Image) -> np.ndarray:
    """Reduction factor per axis; the wavelength axis is never binned."""
    if data_cube.ndim == 3:
        return np.asarray([1, y_factor, x_factor])
    if data_cube.ndim == 2:
        return np.asarray([y_factor, x_factor])
    if data_cube.ndim == 1:
        return np.asarray([x_factor])
    raise ValueError(f"Unsupported data dimensionality: {data_cube.ndim}")


def _compute_num_of_extra_pixels(factor: np.ndarray, data_cube: Cube | Image) -> np.ndarray:
    """Pixels by which each axis exceeds an integer multiple of its factor."""
    return np.mod(data_cube.shape, factor).astype(int)


def _compute_shortened_axis_dimensions(data_cube: Cube | Image, num_extra_pixels: np.ndarray,
                                       margin: str = "center") -> tuple[slice, ...]:
    """Slices that trim each axis to an integer multiple of its factor.

    ``margin`` decides which pixels are dropped: 'origin'/'left' keep the start
    of each axis, 'right' keeps the end, 'center' aligns the centres (within one
    pixel).
    """
    slices = []
    for k in range(data_cube.ndim):
        if margin in ("origin", "left"):
            nstart = 0
        elif margin == "center":
            nstart = num_extra_pixels[k] // 2
        elif margin == "right":
            nstart = num_extra_pixels[k]
        slices.append(slice(nstart, data_cube.shape[k] - num_extra_pixels[k] + nstart))
    return tuple(slices)


def _shorten_data_axis_dimensions(data_cube: Cube | Image, slices: tuple[slice, ...]) -> Cube | Image:
    """Apply ``slices`` in place (so ``inplace=True`` works)."""
    tmp = data_cube[slices]
    data_cube._data = tmp._data
    data_cube._var = tmp._var
    data_cube._mask = tmp._mask
    data_cube.wcs = tmp.wcs
    if hasattr(data_cube, "wave"):
        data_cube.wave = tmp.wave
    return data_cube


def _determine_new_data_shape(data_cube: Cube | Image, factor: np.ndarray) -> np.ndarray:
    """Shape after binning (dimensions are already integer multiples)."""
    return data_cube.shape // factor


def _create_preshaping_array(newshape: np.ndarray, factor: np.ndarray) -> np.ndarray:
    """Interleave new shape and factors, e.g. (nz, 1, ny', fy, nx', fx).

    Reshaping with this places the pixels to be combined on their own axes.
    """
    return np.column_stack((newshape, factor)).ravel()


def _bin_axes(ndim: int) -> tuple[int, ...]:
    """Axes of the pre-shaped array that hold the pixels to be combined."""
    return tuple(range(1, 2 * ndim, 2))


def _count_unmasked_pixels(data_cube: Cube | Image, preshape: np.ndarray) -> np.ndarray:
    """Number of unmasked input pixels contributing to each output pixel."""
    unmasked = ~np.ma.getmaskarray(data_cube.data)
    return unmasked.reshape(preshape).astype(int).sum(axis=_bin_axes(data_cube.ndim))


def _bin_data(data_cube: Cube | Image, newdata: np.ndarray, unmasked: np.ndarray,
              method: str = "sum", surface_brightness: bool = False) -> Cube | Image:
    """Combine each group of pixels by ``method`` (all bin axes at once)."""
    axes = _bin_axes(data_cube.ndim)
    if method == "sum":
        binned = np.nansum(newdata, axis=axes)
        if surface_brightness:
            binned = binned / unmasked
    elif method == "mean":
        binned = np.nanmean(newdata, axis=axes)
    elif method == "median":
        binned = np.nanmedian(newdata, axis=axes)
    data_cube._data = np.ma.getdata(binned)
    return data_cube


def _calc_var_sum(newvar: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    """Variance of a sum: sum(v[i])."""
    return newvar.sum(axis=axes)


def _calc_var_mean(newvar: np.ndarray, axes: tuple[int, ...], unmasked: np.ndarray) -> np.ndarray:
    """Variance of a mean of N unmasked pixels: sum(v[i]) / N**2."""
    return newvar.sum(axis=axes) / unmasked ** 2


def _calc_var_median(newdata: np.ndarray, axes: tuple[int, ...], unmasked: np.ndarray) -> np.ndarray:
    """Estimated variance of a median of N pixels: (1.253 * std(p[i]))**2 / N."""
    return (1.253 * np.nanstd(newdata, axis=axes)) ** 2 / unmasked


def _bin_var(data_cube: Cube | Image, newdata: np.ndarray, preshape: np.ndarray,
             unmasked: np.ndarray, method: str = "sum", surface_brightness: bool = False) -> Cube | Image:
    """Propagate the variance for the chosen binning method.

    With ``surface_brightness=True`` (sum divided by N) this is sum(v[i]) / N**2,
    the same as for the mean.
    """
    axes = _bin_axes(data_cube.ndim)
    newvar = data_cube.var.reshape(preshape)
    if method == "sum" and surface_brightness:
        newvar = _calc_var_mean(newvar, axes, unmasked)
    elif method == "sum":
        newvar = _calc_var_sum(newvar, axes)
    elif method == "mean":
        newvar = _calc_var_mean(newvar, axes, unmasked)
    elif method == "median":
        newvar = _calc_var_median(newdata, axes, unmasked)
    data_cube._var = np.ma.getdata(newvar)
    return data_cube


def _mask_output_array(data_cube: Cube | Image, unmasked: np.ndarray) -> Cube | Image:
    """Mask output pixels that had no unmasked input pixels."""
    data_cube._mask = unmasked < 1
    return data_cube


def _update_wcs(data_cube: Cube | Image, factor: np.ndarray) -> Cube | Image:
    """Rebin the spatial WCS to the new pixel size."""
    if data_cube._has_wcs and data_cube.wcs is not None and data_cube.ndim > 1:
        data_cube.wcs = data_cube.wcs.rebin([factor[-2], factor[-1]])
    return data_cube


def bin_data(x_factor: int, y_factor: int, data_cube: Cube | Image, margin: str = "center",
             method: str = "sum", inplace: bool = False,
             surface_brightness: bool = False) -> Cube | Image:
    """Combine neighbouring pixels to reduce the spatial size by integer factors.

    Parameters
    ----------
    x_factor, y_factor : int
        Integer reduction factors along the image x and y axes.
    data_cube : mpdaf.obj.Cube or mpdaf.obj.Image
    margin : {'center', 'origin', 'left', 'right'}, optional
        If the dimensions are not integer multiples of the factors, the array is
        truncated first; this chooses which pixels are dropped.  Default 'center'.
    method : {'sum', 'mean', 'median'}, optional
        How pixels are combined.  Default 'sum'.
    inplace : bool, optional
        If True, bin the input object in place.  Default False (returns a copy).
    surface_brightness : bool, optional
        Only for ``method='sum'``: divide each output pixel by the number of
        unmasked input pixels, i.e. return surface brightness instead of flux.
        Default False.

    Returns
    -------
    mpdaf.obj.Cube or mpdaf.obj.Image
        The binned object.  Pixels with no unmasked inputs are masked.

    Raises
    ------
    ValueError
        Unsupported ``margin``/``method``, or factors outside 1..shape.
    """
    data_cube = data_cube if inplace else data_cube.copy()

    if margin not in _MARGINS:
        raise ValueError(f"Unsupported margin parameter: {margin}")
    if method not in _METHODS:
        raise ValueError(f"Unknown binning method: {method}")

    factor = _reduction_factor_to_array(x_factor, y_factor, data_cube)
    if np.any(factor < 1) or np.any(factor > data_cube.shape):
        raise ValueError("The reduction factors must be from 1 to shape")

    num_extra_pixels = _compute_num_of_extra_pixels(factor, data_cube)
    if np.any(num_extra_pixels != 0):
        slices = _compute_shortened_axis_dimensions(data_cube, num_extra_pixels, margin=margin)
        data_cube = _shorten_data_axis_dimensions(data_cube, slices)

    newshape = _determine_new_data_shape(data_cube, factor)
    preshape = _create_preshaping_array(newshape, factor)
    unmasked = _count_unmasked_pixels(data_cube, preshape)
    newdata = data_cube.data.reshape(preshape)

    # data first: mpdaf checks the shape of _var against _data
    data_cube = _bin_data(data_cube, newdata, unmasked, method=method,
                          surface_brightness=surface_brightness)
    if data_cube._var is not None:
        data_cube = _bin_var(data_cube, newdata, preshape, unmasked, method=method,
                             surface_brightness=surface_brightness)
    data_cube = _mask_output_array(data_cube, unmasked)
    data_cube = _update_wcs(data_cube, factor)
    return data_cube


def remove_var(data_cube: Cube | Image) -> Cube | Image:
    """Return a copy of the cube/image without its variance extension."""
    data_cube_novar = data_cube.clone()
    data_cube_novar.data = data_cube.data
    return data_cube_novar


_strip_var = remove_var  # the ``remove_var`` argument below shadows the function name


def bin_list_of_cubes(x_factor_list: list[int], y_factor_list: list[int], cube_list: list[str],
                      redshift: float | list[float] | None = None, remove_var: bool = True,
                      cosmo=WMAP9, **kwargs):
    """Bin each cube file by each (x, y) factor pair and write the results.

    Output files are named ``<input>_binned_NNNpc.fits`` when ``redshift`` is
    given (bin size in parsec), otherwise ``<input>_binned_XXXxYYYspax.fits``.

    Parameters
    ----------
    x_factor_list, y_factor_list : list of int
        Binning factors; must be the same length.
    cube_list : list of str
        Filenames of the cubes.
    redshift : float or list of float, optional
        Redshift of all cubes, or one per cube.
    remove_var : bool
        Also write a copy without the variance extension (``*_novar.fits``).
    cosmo : astropy cosmology
        Used for the physical bin size.  Default WMAP9 (as originally written).
    **kwargs
        Passed to :func:`bin_data`.
    """
    assert len(x_factor_list) == len(y_factor_list), "x_factor_list must be same length as y_factor_list"

    for i, file in enumerate(cube_list):
        this_cube = Cube(file)

        if redshift:
            z = redshift[i] if isinstance(redshift, list) else redshift
            scale = kpc_per_arcsec(z, cosmo=cosmo)

        for j in range(len(x_factor_list)):
            binned_cube = bin_data(x_factor_list[j], y_factor_list[j], this_cube, **kwargs)

            if redshift:
                step = binned_cube.wcs.get_axis_increments(units.arcsec)[0] * units.arcsec
                bin_size = (scale * step).to("pc")
                new_filename = binned_cube.filename.split(".fits")[0] + "_binned_{:0>3d}pc.fits".format(int(bin_size.value))
            else:
                new_filename = binned_cube.filename.split(".fits")[0] + "_binned_{:0>3d}x{:0>3d}spax.fits".format(
                    x_factor_list[j], y_factor_list[j])

            binned_cube.write(new_filename, savemask=False)

            if remove_var:
                binned_cube_novar = _strip_var(binned_cube)
                binned_cube_novar.write(new_filename.split(".fits")[0] + "_novar.fits", savemask=False)
