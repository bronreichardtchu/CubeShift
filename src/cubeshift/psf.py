"""Spatial PSF matching, done on the fine grid before binning.

The PSF is matched *before* integer binning: on the binned grid the matching
kernel is often a fraction of a pixel, so a sampled Gaussian there is close to
a delta function and the PSF is under-applied (review v2, M3).

:func:`psf_match_and_bin` pads the cube once, blurs it on the fine grid, then
bins it.  The single padding serves three purposes:

* a blur margin of ~4 sigma per side, so no light is lost or wrapped around;
* a top-up so each axis is a multiple of the bin factor (nothing is cropped);
* if cheap enough, the target field of view (otherwise the field is padded on
  the coarse grid after binning).

The padding holds the background level with matching noise
(:func:`cubeshift.padding.estimate_background`), and the WCS reference pixel is
shifted so sky positions are unchanged.
"""
from __future__ import annotations

import numpy as np
from astropy.cosmology import Planck18
from scipy.signal import fftconvolve
from scipy.special import erf

from .binning import bin_data
from .cosmology import angular_scale_ratio
from .padding import BackgroundEstimate, crop_to_shape, estimate_background, pad_to_shape

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

#: Above this many fine pixels per axis the field of view is padded on the
#: coarse grid after binning instead of on the fine grid before blurring.
FINE_FOV_LIMIT = 4096


def matching_kernel_fwhm(fwhm_source_arcsec, fwhm_target_arcsec, z_obs, z_sim, cosmo=Planck18):
    """FWHM (arcsec at z_sim) of the Gaussian that turns the source PSF into the target PSF.

    The source PSF (arcsec at z_obs) shrinks by D_A(z_obs)/D_A(z_sim);
    kernel**2 = target**2 - (f_ang * source)**2, and 0 where the shrunk source
    PSF is already broader than the target.
    """
    fwhm_sim = fwhm_source_arcsec * angular_scale_ratio(z_obs, z_sim, cosmo)
    return np.sqrt(np.maximum(np.asarray(fwhm_target_arcsec, float) ** 2 - fwhm_sim ** 2, 0.0))


def gaussian_kernel1d(sigma, truncate=4.0):
    """Normalised, sampled 1D Gaussian (as used by scipy.ndimage.gaussian_filter)."""
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def erf_kernel1d(sigma):
    """Pixel-integrated 1D Gaussian of width ``sigma`` pixels, normalised to 1.

    Exact for any sigma (sigma -> 0 gives [1]).  Radius ceil(4 sigma) + 1.
    """
    if sigma <= 0:
        return np.ones(1)
    radius = int(np.ceil(4 * sigma)) + 1
    x = np.arange(-radius, radius + 1)
    # integral of the Gaussian over each pixel [x - 1/2, x + 1/2]
    k = 0.5 * (erf((x + 0.5) / (np.sqrt(2) * sigma)) - erf((x - 0.5) / (np.sqrt(2) * sigma)))
    return k / k.sum()


def _target_fwhm(cube, fwhm_target_arcsec):
    """Target PSF FWHM per wavelength slice (number or callable of wavelength)."""
    if callable(fwhm_target_arcsec):
        if cube.wave is None:
            raise ValueError("cube has no wavelength axis; pass the target FWHM as a number")
        return np.asarray(fwhm_target_arcsec(cube.wave.coord()), float)
    return np.full(cube.shape[0], float(fwhm_target_arcsec))


def _prebin_factor(b, pixel, sigma_min):
    """Largest divisor q of b with q * pixel <= sigma_min / 2 (b if there is no kernel)."""
    if sigma_min is None:
        return b
    divisors = [q for q in range(1, b + 1) if b % q == 0 and q * pixel <= sigma_min / 2]
    return max(divisors) if divisors else 1


def _pad_amounts(n, margin, b, fov_pixels, phase):
    """Low/high padding for one axis: blur margin, field of view, multiple of b.

    n_need = max(n + 2*margin, fov_pixels); topped up to a multiple of b.
    ``phase`` ('center' or an int in [0, extra]) places the top-up, which sets
    where the bin edges fall.
    """
    n_need = n + 2 * margin
    if fov_pixels is not None:
        n_need = max(n_need, fov_pixels)
    extra = (-n_need) % b
    shift = extra // 2 if phase == "center" else int(phase)
    if not 0 <= shift <= extra:
        raise ValueError(f"phase must be 'center' or an int in [0, {extra}], not {phase!r}")
    lo = (n_need - n) // 2 + shift
    return lo, n_need + extra - n - lo


def _binned_variance(var, good, ky, kx, by, bx):
    """Exact variance of blur-then-bin for independent input pixels.

    Each output pixel is sum_f H(f) x_f, with H the blur kernel averaged over the
    bin (box * kernel / N), so its variance is sum_f H(f)**2 v_f:
    v convolved with H**2 and sampled at the bin anchors.
    """
    # combined blur + bin weights along each axis
    hy = np.convolve(ky, np.ones(by)) / by
    hx = np.convolve(kx, np.ones(bx)) / bx
    v0 = np.where(good, var, 0.0)
    full = fftconvolve(v0, np.outer(hy ** 2, hx ** 2), mode="full")

    # output bin I starts at input pixel I*b; with kernel radius r the matching
    # index in the 'full' convolution is I*b + r + b - 1
    ry, rx = (len(ky) - 1) // 2, (len(kx) - 1) // 2
    ny_out, nx_out = var.shape[0] // by, var.shape[1] // bx
    rows = np.arange(ny_out) * by + ry + by - 1
    cols = np.arange(nx_out) * bx + rx + bx - 1
    return np.clip(full[np.ix_(rows, cols)], 0.0, None)


def psf_match_and_bin(cube, bin_x, bin_y, fwhm_source_arcsec=None, fwhm_target_arcsec=None,
                      z_obs=None, z_sim=None, background="auto", fov=None, phase="center",
                      noise_scale=1.0, rng=None, fine_fov_limit=FINE_FOV_LIMIT, cosmo=Planck18):
    """Pad once, blur to the target PSF on the fine grid, then bin.

    Parameters
    ----------
    cube : mpdaf.obj.Cube
        Surface-brightness cube whose WCS is at the angular scale of z_sim.
    bin_x, bin_y : int
        Integer bin factors.
    fwhm_source_arcsec : float, optional
        Source PSF FWHM in arcsec at z_obs (shrunk to z_sim internally).
    fwhm_target_arcsec : float or callable, optional
        Target PSF FWHM in arcsec, or a function of wavelength (Angstrom).
        If None, no blurring (plain binning with the same padding).
    z_obs, z_sim : float
        Needed with a PSF, to shrink the source PSF.
    background : 'auto', 'zero', number or BackgroundEstimate
        Level and noise of the padding (:func:`cubeshift.padding.estimate_background`).
    fov : (float, float), optional
        Target field of view (x, y) in arcsec.  The output is padded or
        centre-cropped to round(fov / output pixel) pixels.
    phase : 'center' or int
        Position of the bin grid: where the divisibility top-up goes (see
        :func:`_pad_amounts`).  Applies to both axes.
    noise_scale : float
        Multiplier on the padding noise.
    rng : numpy.random.Generator, optional
        For the padding noise; default ``np.random``.
    fine_fov_limit : int
        Pad the field on the fine grid only if it needs at most this many
        (pre-binned) pixels per axis; otherwise pad on the coarse grid.

    Returns
    -------
    mpdaf.obj.Cube
        Binned cube.  Data header: background summary (CSBKG*), and the
        footprint of the original data on the output grid (PADX0, PADY0,
        PADNX, PADNY).
    """
    from .resample import pixel_scale_xy

    nz = cube.shape[0]
    px, py = pixel_scale_xy(cube)

    # --- matching kernel (arcsec) per wavelength slice; zero means no blur
    if fwhm_target_arcsec is None:
        kernel_fwhm = np.zeros(nz)
    else:
        kernel_fwhm = matching_kernel_fwhm(fwhm_source_arcsec or 0.0, _target_fwhm(cube, fwhm_target_arcsec),
                                           z_obs, z_sim, cosmo)
    sigma = kernel_fwhm * FWHM_TO_SIGMA
    sigma_min = sigma[sigma > 0].min() if np.any(sigma > 0) else None

    # --- background level/noise for the padding, measured on the input
    bkg = background if isinstance(background, BackgroundEstimate) else estimate_background(cube, background)
    noise = bkg.noise * noise_scale

    # --- pre-binning factor: bin part of the way first when the fine pixels are
    #     much smaller than the kernel (keeps the FFT small); q divides b
    qx, qy = _prebin_factor(bin_x, px, sigma_min), _prebin_factor(bin_y, py, sigma_min)
    work = cube
    ny0, nx0 = cube.shape[1:]
    lo_y0 = lo_x0 = 0
    if qx > 1 or qy > 1:
        # top up to a multiple of q (centred) so pre-binning crops nothing
        ty, tx = (-ny0) % qy, (-nx0) % qx
        lo_y0, lo_x0 = ty // 2, tx // 2
        if ty or tx:
            work = pad_to_shape(work, (lo_y0, ty - lo_y0), (lo_x0, tx - lo_x0), bkg.mean, noise, rng)
        work = bin_data(qx, qy, work, surface_brightness=True)
        noise = noise / np.sqrt(qx * qy)
    bx, by = bin_x // qx, bin_y // qy
    p2x, p2y = px * qx, py * qy

    # --- kernel width in (pre-binned) pixels; pre-binning already added a box
    #     of (q*p)^2/12 - p^2/12 in variance, so remove it from the kernel
    sig_x = np.sqrt(np.clip(sigma ** 2 - ((p2x ** 2 - px ** 2) / 12.0) * (sigma > 0), 0, None)) / p2x
    sig_y = np.sqrt(np.clip(sigma ** 2 - ((p2y ** 2 - py ** 2) / 12.0) * (sigma > 0), 0, None)) / p2y
    margin_x = len(erf_kernel1d(sig_x.max())) // 2
    margin_y = len(erf_kernel1d(sig_y.max())) // 2

    # --- field of view in output pixels; pad to it on the fine grid if cheap
    n2y, n2x = work.shape[1:]
    nfov_x = nfov_y = None
    fov_on_fine = False
    if fov is not None:
        nfov_x = int(round(fov[0] / (p2x * bx)))
        nfov_y = int(round(fov[1] / (p2y * by)))
        fov_on_fine = max(nfov_x * bx, nfov_y * by) <= fine_fov_limit

    # --- the single pad: blur margin + (field of view) + multiple of the bin factor
    lo_y, hi_y = _pad_amounts(n2y, margin_y, by, nfov_y * by if fov_on_fine else None, phase)
    lo_x, hi_x = _pad_amounts(n2x, margin_x, bx, nfov_x * bx if fov_on_fine else None, phase)
    work = pad_to_shape(work, (lo_y, hi_y), (lo_x, hi_x), bkg.mean, noise, rng)

    # --- blur each slice (normalised convolution: masked/NaN pixels get no weight)
    data = np.ma.getdata(work.data).astype(float)
    good = np.isfinite(data) & ~np.ma.getmaskarray(work.data)
    var = None if work.var is None else np.ma.getdata(work.var).astype(float)
    out_var = None
    if var is not None:
        out_var = np.zeros((nz, data.shape[1] // by, data.shape[2] // bx))
    for k in range(nz):
        ky, kx = erf_kernel1d(sig_y[k]), erf_kernel1d(sig_x[k])
        if len(ky) > 1 or len(kx) > 1:
            kernel = np.outer(ky, kx)
            # convolve relative to the background level, so that beyond the array
            # edge counts as background (not zero) and the level is preserved
            level = bkg.mean[k]
            num = fftconvolve(np.where(good[k], data[k] - level, 0.0), kernel, mode="same")
            # renormalise only for masked/NaN pixels: weight of good pixels relative to
            # the weight of all in-array pixels (= 1 wherever no pixel is bad)
            inside = fftconvolve(np.ones_like(data[k]), kernel, mode="same")
            weight = fftconvolve(good[k].astype(float), kernel, mode="same") / np.where(inside > 0, inside, 1.0)
            data[k] = level + np.where(weight > 1e-12, num / np.where(weight > 1e-12, weight, 1.0), 0.0)
        # exact variance of the blurred-and-binned pixels
        if var is not None:
            out_var[k] = _binned_variance(var[k], good[k], ky, kx, by, bx)
    work.data = np.ma.array(data, mask=~good)

    # --- bin to the output grid (mean of unmasked pixels: surface brightness)
    out = bin_data(bx, by, work, surface_brightness=True)
    if out_var is not None:
        out.var = out_var

    # --- footprint of the original data on the output grid (before FOV pad/crop)
    fy0 = (lo_y + lo_y0 // qy) // by
    fx0 = (lo_x + lo_x0 // qx) // bx
    fny = -(-(lo_y + n2y) // by) - (lo_y // by)
    fnx = -(-(lo_x + n2x) // bx) - (lo_x // bx)

    # --- field of view on the coarse grid: pad (large b) or centre-crop (oversized data)
    if fov is not None:
        ny_out, nx_out = out.shape[1:]
        if ny_out < nfov_y or nx_out < nfov_x:
            # noise of one output pixel made of background only: sqrt(sum H^2) x pixel noise,
            # with H the combined blur + bin weights (as in _binned_variance)
            noise_out = np.zeros(nz)
            for k in range(nz):
                hy = np.convolve(erf_kernel1d(sig_y[k]), np.ones(by)) / by
                hx = np.convolve(erf_kernel1d(sig_x[k]), np.ones(bx)) / bx
                noise_out[k] = noise[k] * np.sqrt((hy ** 2).sum() * (hx ** 2).sum())
            py_, px_ = max(nfov_y - ny_out, 0), max(nfov_x - nx_out, 0)
            out = pad_to_shape(out, (py_ // 2, py_ - py_ // 2), (px_ // 2, px_ - px_ // 2),
                               bkg.mean, noise_out, rng)
            fy0, fx0 = fy0 + py_ // 2, fx0 + px_ // 2
        if out.shape[1] > nfov_y or out.shape[2] > nfov_x:
            out, (cy, cx) = crop_to_shape(out, min(nfov_y, out.shape[1]), min(nfov_x, out.shape[2]))
            fy0, fx0 = fy0 - cy, fx0 - cx

    # --- provenance
    hdr = out.data_header
    for key, card in bkg.header_cards().items():
        hdr[key] = card
    hdr["PADX0"], hdr["PADY0"] = (int(fx0), "x offset of original data (output px)"), (int(fy0), "y offset")
    hdr["PADNX"], hdr["PADNY"] = (int(fnx), "x size of original data (output px)"), (int(fny), "y size")
    return out


def convolve_to_match_psf(cube, fwhm_source_arcsec, fwhm_target_arcsec, z_obs, z_sim, cosmo=Planck18):
    """Convolve each wavelength slice so the PSF matches the target instrument (no binning).

    The cube is zero-padded by the blur margin so the PSF wings are kept (the
    output is larger than the input; the WCS is shifted accordingly).  The
    variance is propagated exactly for independent input pixels.

    Parameters
    ----------
    cube : mpdaf.obj.Cube
        WCS at the angular scale of z_sim.
    fwhm_source_arcsec : float
        Source PSF FWHM in arcsec at z_obs.
    fwhm_target_arcsec : float or callable
        Target PSF FWHM in arcsec, or a function of wavelength (Angstrom).
    z_obs, z_sim : float
        Original and simulated redshift.
    """
    return psf_match_and_bin(cube, 1, 1, fwhm_source_arcsec, fwhm_target_arcsec, z_obs, z_sim,
                             background="zero", cosmo=cosmo)
