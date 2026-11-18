"""Background estimation and padding/cropping of cubes (WCS kept consistent)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.stats import sigma_clipped_stats
from mpdaf.obj import Cube


@dataclass
class BackgroundEstimate:
    """Per-wavelength background level and pixel noise used to fill padding."""

    mean: np.ndarray
    """Background level per slice (cube units)."""
    noise: np.ndarray
    """Pixel-to-pixel noise (standard deviation) per slice."""
    source: str
    """How it was obtained: 'sky' (border sky pixels), 'diff' (galaxy fills the field:
    mean 0, noise from neighbour differences), 'zero' or 'user'."""

    def header_cards(self):
        """FITS cards summarising the estimate (medians over slices)."""
        return {
            "CSBKGSRC": (self.source, "CubeShift padding background source"),
            "CSBKGMN": (float(np.nanmedian(self.mean)), "median padding background level"),
            "CSBKGNS": (float(np.nanmedian(self.noise)), "median padding pixel noise (pre-binning)"),
        }


def _slices(cube):
    """Data as a float array, plus a mask of usable (finite, unmasked) pixels."""
    data = np.ma.getdata(cube.data).astype(float)
    good = np.isfinite(data) & ~np.ma.getmaskarray(cube.data)
    return data, good


def difference_noise(image, good=None):
    """Robust pixel noise from second differences: 1.4826 * MAD(d2) / sqrt(6).

    d2 = x[i-1] - 2 x[i] + x[i+1] along rows and columns.  Smooth structure,
    including linear gradients of galaxy light, cancels in d2, so this works
    even where a source covers every pixel.
    """
    # set unusable pixels to NaN so they drop out of the differences
    img = np.where(good if good is not None else np.isfinite(image), image, np.nan)

    # second differences along x and y (a constant or a linear gradient gives 0)
    d2x = img[:, :-2] - 2 * img[:, 1:-1] + img[:, 2:]
    d2y = img[:-2, :] - 2 * img[1:-1, :] + img[2:, :]
    diffs = np.concatenate([d2x.ravel(), d2y.ravel()])
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return 0.0

    # MAD -> Gaussian sigma (x1.4826); d2 of independent pixels has sqrt(1+4+1) x the pixel noise
    mad = np.median(np.abs(diffs - np.median(diffs)))
    return float(1.4826 * mad / np.sqrt(6.0))


def estimate_background(cube, mode="auto", border_width=5, min_sky_fraction=0.3, nsigma=3.0):
    """Background level and noise per wavelength slice, for filling padding.

    Parameters
    ----------
    cube : mpdaf.obj.Cube
    mode : {'auto', 'zero'} or float or array
        'auto': the input is assumed sky-subtracted.  The outer ``border_width``
        pixels are tested against zero: if at least ``min_sky_fraction`` of them
        (median over slices) are within ``nsigma`` x the difference noise, the
        galaxy does not fill the field and the mean and noise are measured from
        those sky pixels (source 'sky').  Otherwise the galaxy fills the field:
        mean 0 and noise from neighbour differences (source 'diff').
        'zero': mean 0, noise 0.  A number (or one per slice): that mean, with
        the difference noise.
    border_width : int
        Width in pixels of the border used to look for sky.
    """
    data, good = _slices(cube)
    nz = data.shape[0]

    # 'zero': empty, noiseless padding
    if isinstance(mode, str) and mode == "zero":
        return BackgroundEstimate(np.zeros(nz), np.zeros(nz), "zero")

    # pixel noise per slice from neighbour differences (used by every other mode)
    noise_diff = np.array([difference_noise(data[k], good[k]) for k in range(nz)])

    # user-supplied level (one value, or one per slice)
    if not isinstance(mode, str):
        mean = np.broadcast_to(np.asarray(mode, float), (nz,)).copy()
        return BackgroundEstimate(mean, noise_diff, "user")
    if mode != "auto":
        raise ValueError(f"background must be 'auto', 'zero' or a number, not {mode!r}")

    # mask of the outer border_width pixels of the field
    ny, nx = data.shape[1:]
    bw = max(1, min(int(border_width), ny // 2, nx // 2))
    border = np.zeros((ny, nx), bool)
    border[:bw, :] = border[-bw:, :] = border[:, :bw] = border[:, -bw:] = True

    # per slice: which border pixels are consistent with (sky-subtracted) zero?
    sky_like, frac = [], np.zeros(nz)
    for k in range(nz):
        vals = data[k][border & good[k]]
        ok = np.abs(vals) <= nsigma * noise_diff[k]
        sky_like.append(vals[ok])
        frac[k] = ok.mean() if vals.size else 0.0

    # too little sky in the border: the galaxy fills the field
    if np.median(frac) < min_sky_fraction:
        return BackgroundEstimate(np.zeros(nz), noise_diff, "diff")

    # sky present: clipped mean and std of the sky-like border pixels
    # (slices with too few sky pixels keep mean 0 and the difference noise)
    mean, noise = np.zeros(nz), noise_diff.copy()
    for k in range(nz):
        if sky_like[k].size >= 3:
            m, _, s = sigma_clipped_stats(sky_like[k], sigma=nsigma)
            mean[k], noise[k] = m, s
    return BackgroundEstimate(mean, noise, "sky")


def pad_to_shape(cube, pad_y, pad_x, fill, noise=None, rng=None):
    """Pad a cube spatially, keeping sky positions (CRPIX shifted by the low-side pad).

    Parameters
    ----------
    pad_y, pad_x : (int, int)
        Pixels added on the low and high side of each axis.
    fill : float or array (nz,)
        Value of the padding per slice.
    noise : float or array (nz,), optional
        If given, Gaussian noise of this standard deviation is added to the
        padding, and (if the cube has a variance) its variance is noise**2.
    rng : numpy.random.Generator, optional
        Default: the global ``np.random`` state.

    Returns
    -------
    mpdaf.obj.Cube
        Padded cube.  Padding is unmasked; headers are copied.
    """
    (ylo, yhi), (xlo, xhi) = pad_y, pad_x
    nz, ny, nx = cube.shape
    oy, ox = ny + ylo + yhi, nx + xlo + xhi
    fill = np.broadcast_to(np.asarray(fill, float), (nz,))
    noise_arr = None if noise is None else np.broadcast_to(np.asarray(noise, float), (nz,))
    normal = np.random.normal if rng is None else rng.normal

    # output filled with the background level of each slice, plus noise if requested
    data = np.repeat(fill[:, None, None], oy, axis=1).repeat(ox, axis=2)
    if noise_arr is not None and np.any(noise_arr > 0):
        data = data + normal(size=data.shape) * noise_arr[:, None, None]

    # copy the original data and mask into the middle; the padding is unmasked
    inner = (slice(None), slice(ylo, ylo + ny), slice(xlo, xlo + nx))
    data[inner] = np.ma.getdata(cube.data)
    mask = np.zeros(data.shape, bool)
    mask[inner] = np.ma.getmaskarray(cube.data)

    # variance: noise**2 in the padding (0 if noiseless), the original variance in the middle
    var = None
    if cube.var is not None:
        pad_var = np.zeros(nz) if noise_arr is None else noise_arr ** 2
        var = np.repeat(pad_var[:, None, None], oy, axis=1).repeat(ox, axis=2)
        var[inner] = np.ma.getdata(cube.var)

    # same WCS, with the reference pixel moved by the low-side padding so that
    # every original pixel keeps its sky position
    wcs = cube.wcs.copy()
    wcs.set_crpix1(wcs.get_crpix1() + xlo)
    wcs.set_crpix2(wcs.get_crpix2() + ylo)
    wcs.naxis1, wcs.naxis2 = ox, oy

    return Cube(data=np.ma.array(data, mask=mask), var=var, wcs=wcs, wave=cube.wave, unit=cube.unit,
                data_header=cube.data_header.copy(), primary_header=cube.primary_header.copy(), copy=False)


def crop_to_shape(cube, ny, nx):
    """Centre-crop the spatial axes to (ny, nx).

    mpdaf slicing keeps the WCS consistent.  Returns the cropped cube and the
    (y0, x0) offset of the crop in the input.
    """
    _, cy, cx = cube.shape
    y0, x0 = max((cy - ny) // 2, 0), max((cx - nx) // 2, 0)
    return cube[:, y0:y0 + min(ny, cy), x0:x0 + min(nx, cx)], (y0, x0)
