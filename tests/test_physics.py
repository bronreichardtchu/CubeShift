"""Physical correctness checks from the code review (cubeshift_review.md).

These encode what the pipeline *should* do.  They are marked ``xfail(strict=True)``
while the corresponding bug is still present: when a fix lands the test starts
passing, strict mode turns that into a failure, and the marker must be removed
in the same commit (together with regenerating any golden files it changes).
"""
import numpy as np
import pytest
from astropy.cosmology import Planck18 as cosmo

from cubeshift import cosmology, extinction, resample, spectral
from cubeshift.instruments import TELESCOPES, Telescope
from cubeshift.redshift import redshift_wavelength_axis
from cubeshift.testing import fit_gaussian_line, make_test_cube

Z_OBS, Z_SIM = 0.025, 3.0
NIRSPEC = TELESCOPES["JWST_NIRSpec"]


def _spatial_chain(quiet, target=NIRSPEC):
    """Steps 1-6 of the pipeline (no throughput) on the default test cube."""
    cube = make_test_cube()
    tot0 = np.nansum(cube.data) * cube.wave.get_step()
    with quiet():
        rc, _ = redshift_wavelength_axis(cube, Z_OBS, Z_SIM)
        cosmology.scale_cube(rc, cosmology.flux_dimming_factor(Z_OBS, Z_SIM))
        cosmology.scale_cube(rc, 1.0 / resample.get_spaxel_area(rc))
        np.random.seed(0)
        cr, bf, _, dx, dy = resample.resample_cube_to_telescope_and_redshift(rc, target, Z_OBS, Z_SIM)
        cosmology.scale_cube(cr, resample.get_spaxel_area(cr))
    n = cube.shape[1] // bf[0]
    o = (cr.shape[1] - n) // 2
    inner = np.zeros(cr.shape[1:], bool)
    inner[o:o + n, o:o + n] = True
    return cube, cr, bf, inner, tot0


def test_luminosity_distance_dimming_of_redshift_and_rescale():
    """Wavelength stretch x f_lambda dimming => integrated flux scales as D_L**-2 (already correct)."""
    cube = make_test_cube()
    rc, _ = redshift_wavelength_axis(cube, Z_OBS, Z_SIM)
    f0 = np.nansum(cube.data) * cube.wave.get_step()
    f1 = np.nansum(cosmology.rescale_flux(rc.data, Z_OBS, Z_SIM)) * rc.wave.get_step()
    dl2 = (cosmo.luminosity_distance(Z_OBS) / cosmo.luminosity_distance(Z_SIM)).value ** 2
    assert f1 / f0 == pytest.approx(dl2, rel=1e-6)


def test_padding_keeps_galaxy_unmasked(quiet):
    """Review #1: the padding mask used to be inverted (galaxy masked, noise not)."""
    _, cr, _, inner, _ = _spatial_chain(quiet)
    mask = np.ma.getmaskarray(cr.data)
    assert mask[:, inner].mean() == 0.0
    assert mask[:, ~inner].mean() == 0.0          # padding is valid simulated background
    h = cr.data_header
    footprint = np.zeros(cr.shape[1:], bool)
    footprint[h["PADY0"]:h["PADY0"] + h["PADNY"], h["PADX0"]:h["PADX0"] + h["PADNX"]] = True
    np.testing.assert_array_equal(footprint, inner)


def test_spatial_resampling_conserves_flux(quiet):
    """Review #1/#2: masked galaxy spaxels were never converted SB -> flux (0.48x)."""
    _, cr, _, inner, tot0 = _spatial_chain(quiet)
    dl2 = (cosmo.luminosity_distance(Z_OBS) / cosmo.luminosity_distance(Z_SIM)).value ** 2
    galaxy = np.ma.getdata(cr.data)[:, inner].sum() * cr.wave.get_step()
    assert galaxy / tot0 == pytest.approx(dl2, rel=0.01)


def test_wcs_is_at_simulated_angular_scale(quiet):
    """Review #3: the WCS used to stay at the source-frame scale (1.45"/spaxel instead of 0.096")."""
    _, cr, bf, _, _ = _spatial_chain(quiet)
    f_ang = cosmology.angular_scale_ratio(Z_OBS, Z_SIM)
    expected = bf[0] * 0.29 * f_ang
    assert np.abs(cr.wcs.get_step(unit="arcsec")) == pytest.approx([expected, expected], rel=0.01)


def _second_moment_fwhm(img, pix):
    _, x = np.indices(img.shape)
    tot = img.sum()
    xc = (img * x).sum() / tot
    sx2 = (img * (x - xc) ** 2).sum() / tot
    return 2.3548 * np.sqrt(sx2) * pix


def test_psf_matches_target_at_each_wavelength(quiet):
    """Review #4: source seeing (shrunk to z_sim) is convolved up to the target's PSF(lambda)."""
    from mpdaf.obj import WCS, Cube, WaveCoord

    from cubeshift.psf import convolve_to_match_psf

    seeing = 0.5                                   # arcsec at z_obs
    f_ang = cosmology.angular_scale_ratio(Z_OBS, Z_SIM)
    pix = 0.004                                    # fine grid, arcsec at z_sim
    n = 121
    yy, xx = np.indices((n, n)) - (n - 1) / 2
    sig = seeing * f_ang / 2.3548 / pix
    img = np.exp(-(xx ** 2 + yy ** 2) / (2 * sig ** 2))
    wave = WaveCoord(crval=17000.0, cdelt=12000.0, crpix=1, cunit="Angstrom")   # 1.7 and 2.9 um
    cube = Cube(data=np.stack([img, img]), wave=wave,
                wcs=WCS(crval=(0, 0), cdelt=(pix / 3600, -pix / 3600), shape=(n, n), deg=True))
    with quiet():
        out = convolve_to_match_psf(cube, seeing, NIRSPEC.psf_fwhm, Z_OBS, Z_SIM)
    for i, lam in enumerate(cube.wave.coord()):
        assert _second_moment_fwhm(np.asarray(out.data[i]), pix) == pytest.approx(
            NIRSPEC.psf_fwhm(lam), rel=0.02)


def test_pipeline_uses_source_seeing(mock_data_dir, quiet):
    """Review #4: the pipeline used to pass the pixel scale as the source PSF."""
    from cubeshift import simulate_observation

    sharp = Telescope("sharp", spatial_fwhm=0.1, pixel_scale_x=0.29, pixel_scale_y=0.29, spectral_resolution=4000)
    blurry = Telescope("blurry", spatial_fwhm=2.0, pixel_scale_x=0.29, pixel_scale_y=0.29, spectral_resolution=4000)
    out = {}
    for t in (sharp, blurry):
        np.random.seed(0)
        with quiet():
            out[t.name] = simulate_observation(make_test_cube(), Z_OBS, Z_SIM, t, "JWST_NIRSpec",
                                               data_dir=mock_data_dir).cube
    # 2" seeing shrinks to 0.13" > NIRSpec PSF: no extra blur; 0.1" -> 0.0066": convolved
    assert not np.allclose(out["sharp"].data, out["blurry"].data, rtol=1e-3, atol=0)


def test_surface_brightness_dimming():
    """Integrated SB per true arcsec^2 dims as (1+z)^-4 (pipeline steps 1-3)."""
    cube = make_test_cube()
    sb0 = np.nansum(cube.data) * cube.wave.get_step() / (cube.shape[1] * cube.shape[2] * 0.29 ** 2)
    rc, _ = redshift_wavelength_axis(cube, Z_OBS, Z_SIM)
    cosmology.scale_cube(rc, cosmology.flux_dimming_factor(Z_OBS, Z_SIM))
    rc = resample.shrink_to_redshift(rc, Z_OBS, Z_SIM)
    cosmology.scale_cube(rc, 1.0 / resample.get_spaxel_area(rc))
    sb1 = np.nansum(rc.data) * rc.wave.get_step() / (rc.shape[1] * rc.shape[2])
    assert sb1 / sb0 == pytest.approx(((1 + Z_OBS) / (1 + Z_SIM)) ** 4, rel=1e-6)
    with pytest.raises(ValueError):
        resample.shrink_to_redshift(rc, Z_OBS, Z_SIM)


def test_rectangular_target_spaxels(quiet):
    """Review #13: x/y were swapped twice (cancelling only for square spaxels)."""
    rect = Telescope("rect", spatial_fwhm=0.07, pixel_scale_x=0.1, pixel_scale_y=0.2,
                     spectral_resolution=1000, fov_x=3.0, fov_y=3.0)
    _, cr, bf, _, _ = _spatial_chain(quiet, target=rect)
    dy, dx = np.abs(cr.wcs.get_step(unit="arcsec"))
    assert bf[1] > bf[0]
    assert dy / dx == pytest.approx(bf[1] / bf[0])


def test_lsf_matches_nirspec_resolution(mock_data_dir, quiet):
    """Review #5: KCWI line (R=4000) observed with G235H (mock R=2291 at 2.0027um).

    FWHM_out^2 = FWHM_in^2 + (lam/R_N)^2 - (lam/R_src)^2 = 5.51^2 + 8.74^2 - 5.01^2 -> 9.04 A,
    plus the 1.95 A pixel the kernel is integrated over (exact check on the fine grid).
    After rebinning to the 3.96 A disperser pixels (~2.3 px per FWHM) a Gaussian fit reads
    ~9.7 A.  The old code used R_NIRSpec as the *input* resolution: the line stayed at 5.5 A.
    """
    c = make_test_cube(line_sigma=0.6, continuum=0.0)[:, 12:18, 12:18]
    with quiet():
        rc, _ = redshift_wavelength_axis(c, Z_OBS, Z_SIM)
    lam0 = 5006.84 * (1 + Z_SIM)
    wave = rc.wave.coord()
    fwhm_in = fit_gaussian_line(wave, np.nansum(np.asarray(rc.data), axis=(1, 2)), lam0, 60)["fwhm"]
    R_n = lambda lam: 2700 * lam / 23600.0                     # the mock G235H table
    expected = np.sqrt(fwhm_in ** 2 + (lam0 / R_n(lam0)) ** 2 - (lam0 / 4000) ** 2)

    fine = spectral.match_spectral_resolution(rc, R_n, 4000)
    pix_box = 2.3548 * np.diff(wave)[0] / np.sqrt(12)
    fit = fit_gaussian_line(wave, np.nansum(np.asarray(fine.data), axis=(1, 2)), lam0, 60)
    assert fit["fwhm"] == pytest.approx(np.hypot(expected, pix_box), rel=0.005)

    with quiet():
        bl, _ = spectral.apply_dispersion_pipeline(rc, source_resolution=4000, data_dir=mock_data_dir)
    fit = fit_gaussian_line(bl.wave.coord(), np.nansum(np.asarray(bl.data), axis=(1, 2)), lam0, 60)
    assert 9.0 < fit["fwhm"] < 10.0


def test_lsf_operator_width_and_flux():
    """A delta-like line convolved to R=2000 has FWHM lambda/2000; flux and edges are conserved."""
    from mpdaf.obj import Cube, WaveCoord

    wave = WaveCoord(crval=19000.0, cdelt=0.5, crpix=1, cunit="Angstrom")
    lam = 19000.0 + 0.5 * np.arange(4000)
    data = np.exp(-0.5 * ((lam - 20000.0) / 0.3) ** 2)[:, None, None] * np.ones((1, 2, 2))
    data[0] = 5.0                                   # spike at the band edge must not wrap around
    c = Cube(data=data, var=np.ones_like(data), wave=wave)
    out = spectral.match_spectral_resolution(c, 2000.0)
    fit = fit_gaussian_line(lam, out.data[:, 0, 0], center=20000.0, window=60)
    assert fit["fwhm"] == pytest.approx(np.sqrt((20000 / 2000) ** 2 + (0.3 * 2.3548) ** 2), rel=0.01)
    line = np.abs(lam - 20000) < 100
    assert out.data[line, 0, 0].sum() == pytest.approx(data[line, 0, 0].sum(), rel=1e-6)
    assert np.all(out.data[-200:, 0, 0] < 1e-12)    # no wrap-around from the edge spike
    assert out.var is not None and np.all(out.var[line] < 1)   # smoothing lowers per-pixel variance


def test_final_spectral_resampling_conserves_line_flux(quiet):
    """Review #6: point-sampling onto a lambda/1000 grid gave 0.03-2.4x the line flux."""
    c = make_test_cube(line_sigma=0.6, continuum=0.0)[:, 14:16, 14:16]
    with quiet():
        rc, _ = redshift_wavelength_axis(c, Z_OBS, Z_SIM)
    w = rc.wave.coord()
    bw = np.median(w) / NIRSPEC.spectral_resolution
    for phase in (0.0, 0.33, 0.66):
        grid = np.arange(w[0] + phase * bw, w[-1] + bw, bw)
        with quiet():
            out = spectral.resample_spectral_axis(rc, grid)
        f_in = np.nansum(rc.data) * rc.wave.get_step()
        f_out = np.nansum(out.data) * out.wave.get_step()
        assert f_out / f_in == pytest.approx(1.0, rel=1e-6)


def test_ccm_infrared_branch():
    """Review #7: the optical polynomial was used in the NIR (A/A_V = 4.8 at 2 um)."""
    # CCM89 IR branch at 2 um: A/A_V = (0.574 - 0.527/3.1) * 0.5**1.61 = 0.132
    assert extinction.cardelli_A_lambda(np.array([20000.0]), Av=1.0)[0] == pytest.approx(0.132, abs=0.001)


def test_ccm_law_shape():
    """A_V normalisation, continuity at the branch boundaries, and the 2175 A bump."""
    ccm = extinction.cardelli_A_lambda
    assert ccm(5494.5, Av=1.0) == pytest.approx(1.0, abs=0.01)          # V band
    for x in (1.1, 3.3, 8.0):                                            # branch edges (1/um)
        lo, hi = ccm(1e4 / (x - 1e-6), 1.0), ccm(1e4 / (x + 1e-6), 1.0)
        assert lo == pytest.approx(hi, rel=0.02)
    lam = np.linspace(1800, 2600, 200)
    assert 2100 < lam[np.argmax(ccm(lam, 1.0))] < 2250                   # UV bump
    with pytest.raises(ValueError):
        ccm(900.0, 1.0)
    with pytest.warns(UserWarning):
        ccm(40000.0, 1.0)


def test_variance_survives_pipeline(mock_data_dir, quiet):
    """Review #9 (part): the spectral resamplers used to drop the variance."""
    from cubeshift import simulate_observation

    np.random.seed(0)
    with quiet():
        res = simulate_observation(make_test_cube(), Z_OBS, Z_SIM, "Keck_KCWI", "JWST_NIRSpec",
                                   data_dir=mock_data_dir)
    assert res.cube.var is not None


# --- review #9: variance propagation ----------------------------------------
def _snr(cube):
    return np.ma.getdata(cube.data) / np.sqrt(np.ma.getdata(cube.var))


def test_flux_scalings_preserve_snr(mock_data_dir, quiet):
    """Dimming, SB conversion, throughput and MW extinction rescale data and variance together."""
    from cubeshift.extinction import milky_way_extinction_correction_cube
    from cubeshift.throughput import apply_transmission_filter

    cube = make_test_cube()
    snr0 = _snr(cube)
    rc, _ = redshift_wavelength_axis(cube, Z_OBS, Z_SIM)
    cosmology.scale_cube(rc, cosmology.flux_dimming_factor(Z_OBS, Z_SIM))
    rc = resample.shrink_to_redshift(rc, Z_OBS, Z_SIM)
    cosmology.scale_cube(rc, 1.0 / resample.get_spaxel_area(rc))
    np.testing.assert_allclose(_snr(rc), snr0, rtol=1e-10)
    with quiet():
        tc, _ = apply_transmission_filter(rc, Z_SIM, 5007, data_dir=mock_data_dir)
    t = np.ma.getdata(tc.data) != 0
    np.testing.assert_allclose(_snr(tc)[t], snr0[t], rtol=1e-10)
    ext = milky_way_extinction_correction_cube(cube, 0.3)
    np.testing.assert_allclose(_snr(ext), snr0, rtol=1e-10)


def test_surface_brightness_binning_variance():
    """Sum / N must have variance sum(v) / N^2 (it was sum(v))."""
    from cubeshift.binning import bin_data

    c = make_test_cube(nx=12, ny=9, nw=3)
    c.var = np.full(c.shape, 4.0)
    b = bin_data(3, 3, c, surface_brightness=True)
    np.testing.assert_allclose(b.var, 4.0 / 9)


def test_psf_variance_matches_noise():
    """Convolved variance equals the empirical variance of convolved white noise."""
    from mpdaf.obj import WCS, Cube, WaveCoord

    from cubeshift.psf import convolve_to_match_psf

    rng = np.random.default_rng(1)
    n = 300
    noise = rng.normal(0, 1, (1, n, n))
    cube = Cube(data=noise, var=np.ones_like(noise),
                wave=WaveCoord(crval=20000.0, cdelt=1.0, crpix=1, cunit="Angstrom"),
                wcs=WCS(crval=(0, 0), cdelt=(0.01 / 3600, -0.01 / 3600), shape=(n, n), deg=True))
    out = convolve_to_match_psf(cube, 0.0, 0.05, Z_OBS, Z_SIM)       # 5 px FWHM kernel
    inner = (slice(None), slice(20, -20), slice(20, -20))
    assert np.var(np.ma.getdata(out.data)[inner]) == pytest.approx(np.median(out.var[inner]), rel=0.05)


def test_padding_variance_matches_padding_noise(quiet):
    _, cr, _, inner, _ = _spatial_chain(quiet)
    pad = ~inner
    k = cr.shape[0] // 2
    assert np.std(np.ma.getdata(cr.data)[k][pad]) == pytest.approx(np.sqrt(np.ma.getdata(cr.var)[k][pad].mean()),
                                                                    rel=0.1)


# --- review v2: PSF before binning (M3), single padding, background (M1), WCS (M2) ------
def _cube_from_image(img, pix_arcsec, nw=1, var=None, crpix=None):
    """Synthetic cube with the same image in every slice, WCS already 'at z_sim' (flagged)."""
    from mpdaf.obj import WCS, Cube, WaveCoord

    ny, nx = img.shape
    data = np.repeat(img[None], nw, axis=0)
    cube = Cube(data=data, var=None if var is None else np.full(data.shape, var),
                wcs=WCS(crval=(10.0, 10.0), cdelt=(pix_arcsec / 3600, -pix_arcsec / 3600),
                        crpix=crpix or (ny / 2, nx / 2), shape=(ny, nx), deg=True),
                wave=WaveCoord(crval=20000.0, cdelt=100.0, crpix=1, cunit="Angstrom", shape=nw))
    # mark the WCS as already at the z_sim angular scale
    cube.data_header[resample.ANGSCALE_KEY] = 1.0
    return cube


def _moments(img):
    """Centroid (y, x) and second-moment FWHM (in pixels, averaged over x and y)."""
    img = np.ma.filled(img, 0.0)
    y, x = np.indices(img.shape)
    tot = img.sum()
    yc, xc = (img * y).sum() / tot, (img * x).sum() / tot
    var = (img * ((x - xc) ** 2 + (y - yc) ** 2)).sum() / tot / 2
    return yc, xc, 2.3548 * np.sqrt(var)


@pytest.mark.parametrize("z_sim", [4.0, 6.0, 8.0])
def test_psf_reaches_target_after_binning(quiet, z_sim):
    """M3: the PSF used to be applied on the binned grid (0.77-0.90 of the target at z = 4-8).

    A source with 1" KCWI seeing is shrunk to z_sim, blurred to the target PSF and binned.
    Its PSF is then max(target, shrunk seeing).  The output is under-sampled (~1 output
    pixel per FWHM), so the second moment is averaged over sub-pixel source positions,
    where the output pixel adds exactly p_out^2/12 to sigma^2.
    """
    from cubeshift.psf import FWHM_TO_SIGMA, psf_match_and_bin

    f_ang = cosmology.angular_scale_ratio(Z_OBS, z_sim)
    pix = 0.29 * f_ang                                   # KCWI pixel at z_sim
    b = int(round(NIRSPEC.pixel_scale_x / pix))
    p_out = pix * b
    target = float(NIRSPEC.psf_fwhm(35000.0))            # 0.113" (wider than the shrunk seeing)
    sig_src = 1.0 * FWHM_TO_SIGMA / 0.29                 # 1" seeing in KCWI pixels

    # average the measured sigma^2 over source positions within one output pixel
    n = 61
    variances = []
    for dy in range(b):
        for dx in range(b):
            yy, xx = np.indices((n, n)) - (n - 1) / 2
            img = np.exp(-((xx - dx) ** 2 + (yy - dy) ** 2) / (2 * sig_src ** 2))
            with quiet():
                out = psf_match_and_bin(_cube_from_image(img, pix), b, b, 1.0, target, Z_OBS, z_sim,
                                        background="zero")
            variances.append((_moments(out.data[0])[2] / 2.3548 * p_out) ** 2)

    # expected: (target PSF or shrunk seeing, whichever is wider) + output pixel box
    psf = max(target, 1.0 * f_ang)
    expected = np.sqrt((psf * FWHM_TO_SIGMA) ** 2 + p_out ** 2 / 12)
    assert np.sqrt(np.mean(variances)) == pytest.approx(expected, rel=0.03)


def test_psf_and_bin_keeps_flux_with_wings(quiet):
    """Zero padding keeps the PSF wings: a source at the field edge loses no flux."""
    from cubeshift.psf import psf_match_and_bin

    # bright source 2 pixels from the corner of a 40x40 field
    img = np.zeros((40, 40))
    img[2, 2] = 1.0
    cube = _cube_from_image(img, 0.01)
    with quiet():
        out = psf_match_and_bin(cube, 4, 4, 0.0, 0.1, Z_OBS, Z_OBS, background="zero")
    # total flux = sum(SB) x pixel area, before and after
    flux_in = img.sum() * 0.01 ** 2
    dx, dy = resample.pixel_scale_xy(out)
    assert np.ma.sum(out.data) * dx * dy == pytest.approx(flux_in, rel=1e-6)


def test_padding_rule():
    """One pad: >= the blur margin per side, a multiple of b, the FOV if given, and the phase."""
    from cubeshift.psf import _pad_amounts

    # margin + divisibility
    lo, hi = _pad_amounts(n=23, margin=5, b=4, fov_pixels=None, phase="center")
    assert (23 + lo + hi) % 4 == 0 and lo >= 5 and hi >= 5
    # field of view larger than the data: pad straight to it
    lo, hi = _pad_amounts(n=23, margin=5, b=4, fov_pixels=120, phase="center")
    assert 23 + lo + hi == 120
    # phase moves the top-up from one side to the other
    lo0, _ = _pad_amounts(n=23, margin=5, b=4, fov_pixels=None, phase=0)
    lo3, _ = _pad_amounts(n=23, margin=5, b=4, fov_pixels=None, phase=3)
    assert lo3 - lo0 == 3
    with pytest.raises(ValueError):
        _pad_amounts(n=23, margin=5, b=4, fov_pixels=None, phase=9)


@pytest.mark.parametrize("fine_limit", [100000, 10])
def test_fov_padding_paths_and_crop(quiet, fine_limit):
    """Small b pads the FOV on the fine grid, large b on the coarse grid: same output size.
    Data larger than the FOV are centre-cropped."""
    from cubeshift.psf import psf_match_and_bin

    img = np.zeros((40, 40))
    img[20, 20] = 1.0
    cube = _cube_from_image(img, 0.02)
    with quiet():
        out = psf_match_and_bin(cube, 5, 5, 0.0, 0.1, Z_OBS, Z_OBS, background="zero",
                                fov=(3.0, 2.0), fine_fov_limit=fine_limit)
    assert out.shape[1:] == (round(2.0 / 0.1), round(3.0 / 0.1))
    with quiet():
        small = psf_match_and_bin(cube, 5, 5, 0.0, 0.1, Z_OBS, Z_OBS, background="zero", fov=(0.4, 0.4))
    assert small.shape[1:] == (4, 4)


def test_background_modes():
    """'auto' finds sky in the border, or falls back to mean 0 + difference noise."""
    from cubeshift.padding import estimate_background

    rng = np.random.default_rng(3)
    yy, xx = np.indices((40, 40)) - 19.5

    # compact galaxy on sky with mean 0.05 and noise 0.1: source 'sky'
    img = 5 * np.exp(-(xx ** 2 + yy ** 2) / (2 * 3 ** 2)) + 0.05 + rng.normal(0, 0.1, (40, 40))
    bkg = estimate_background(_cube_from_image(img, 0.02))
    assert bkg.source == "sky"
    assert bkg.mean[0] == pytest.approx(0.05, abs=0.02) and bkg.noise[0] == pytest.approx(0.1, rel=0.15)

    # galaxy filling the field: source 'diff', mean 0, noise from neighbour differences
    img = 10 * np.exp(-(xx ** 2 + yy ** 2) / (2 * 60 ** 2)) + rng.normal(0, 0.1, (40, 40))
    bkg = estimate_background(_cube_from_image(img, 0.02))
    assert bkg.source == "diff" and bkg.mean[0] == 0
    assert bkg.noise[0] == pytest.approx(0.1, rel=0.1)

    # explicit modes
    assert estimate_background(_cube_from_image(img, 0.02), "zero").noise[0] == 0
    assert estimate_background(_cube_from_image(img, 0.02), 2.5).mean[0] == 2.5


def test_padding_is_background_not_galaxy_light(quiet):
    """M1: the padding mean used to come from background() of galaxy-filled slices
    (2.7x the galaxy's line flux).  Now it is the sky level (here ~0)."""
    rng = np.random.default_rng(4)
    yy, xx = np.indices((30, 30)) - 14.5
    # extended galaxy (reaching ~4 sigma at the field edge) on zero-mean sky noise 0.01
    img = 1.0 * np.exp(-(xx ** 2 + yy ** 2) / (2 * 6 ** 2)) + rng.normal(0, 0.01, (30, 30))
    cube = _cube_from_image(img, 0.0192, var=0.01 ** 2)
    with quiet():
        out, _, _, _, _ = resample.resample_cube_to_telescope_and_redshift(
            cube, NIRSPEC, Z_OBS, Z_SIM, rng=np.random.default_rng(5))

    # padding pixels = outside the original footprint recorded in the header
    h = out.data_header
    pad = np.ones(out.shape[1:], bool)
    pad[h["PADY0"]:h["PADY0"] + h["PADNY"], h["PADX0"]:h["PADX0"] + h["PADNX"]] = False
    data = np.ma.getdata(out.data)[0]

    # padding level within 1 sky sigma of zero, and its total far below the galaxy's
    assert abs(data[pad].mean()) < 0.01
    assert abs(data[pad].sum()) < 0.05 * data[~pad].sum()
    # scatter consistent with the variance assigned to the padding
    assert data[pad].std() == pytest.approx(np.sqrt(np.ma.getdata(out.var)[0][pad].mean()), rel=0.2)


def test_sky_positions_survive_padding_and_binning(quiet):
    """M2: CRPIX used not to be shifted after FOV padding (sky positions off by the pad offset)."""
    from cubeshift.psf import psf_match_and_bin

    # off-centre source on a 30x30 grid
    img = np.zeros((30, 30))
    img[8:10, 20:22] = 1.0
    cube = _cube_from_image(img, 0.0192)
    yc, xc, _ = _moments(img)
    sky_in = cube.wcs.pix2sky([yc, xc])[0]
    with quiet():
        out = psf_match_and_bin(cube, 5, 5, 0.0, 0.07, Z_OBS, Z_OBS, background="zero", fov=(3.0, 3.0))
    yo, xo, _ = _moments(out.data[0])
    # sky position of the output centroid, in output pixels
    yo_exp, xo_exp = out.wcs.sky2pix(sky_in)[0]
    assert abs(yo - yo_exp) < 0.05 and abs(xo - xo_exp) < 0.05


def test_psf_and_bin_variance_monte_carlo(quiet):
    """Blur-then-bin variance (sum_f H^2 v_f) matches the scatter of white-noise realisations,
    including padding-only pixels (noise drawn into the padding before the blur)."""
    from cubeshift.padding import BackgroundEstimate
    from cubeshift.psf import psf_match_and_bin

    rng = np.random.default_rng(6)
    sims = []
    for _ in range(150):
        cube = _cube_from_image(rng.normal(0, 1, (48, 48)), 0.01, var=1.0)
        bkg = BackgroundEstimate(np.zeros(1), np.ones(1), "user")      # padding noise = input noise
        with quiet():
            out = psf_match_and_bin(cube, 4, 4, 0.0, 0.06, Z_OBS, Z_OBS, background=bkg,
                                    fov=(1.0, 1.0), rng=rng)
        sims.append(np.ma.getdata(out.data)[0])
    mc = np.array(sims).var(axis=0)
    prop = np.ma.getdata(out.var)[0]
    assert np.median(mc / prop) == pytest.approx(1.0, rel=0.1)


def test_prebinning_matches_full_resolution(quiet, monkeypatch):
    """Pre-binning (for large bin factors) gives the same FWHM (2%) and flux (1e-6)."""
    import cubeshift.psf as psf_mod

    img = np.zeros((96, 96))
    img[47:49, 47:49] = 1.0
    cube = _cube_from_image(img, 0.002)

    # with pre-binning (q > 1 chosen automatically for a 0.1" kernel on 0.002" pixels)
    with quiet():
        fast = psf_mod.psf_match_and_bin(cube, 24, 24, 0.0, 0.1, Z_OBS, Z_OBS, background="zero")
    # reference: force no pre-binning
    monkeypatch.setattr(psf_mod, "_prebin_factor", lambda b, p, s: 1)
    with quiet():
        ref = psf_mod.psf_match_and_bin(cube, 24, 24, 0.0, 0.1, Z_OBS, Z_OBS, background="zero")

    assert np.ma.sum(fast.data) == pytest.approx(np.ma.sum(ref.data), rel=1e-6)
    assert _moments(fast.data[0])[2] == pytest.approx(_moments(ref.data[0])[2], rel=0.02)
