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
        rc.data = cosmology.rescale_flux(rc.data, Z_OBS, Z_SIM)
        rc.data = rc.data / resample.get_spaxel_area(rc)
        np.random.seed(0)
        cr, bf, _, dx, dy = resample.resample_cube_to_telescope_and_redshift(rc, target, Z_OBS, Z_SIM)
        cr.data = cr.data * resample.get_spaxel_area(cr)
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
    rc.data = cosmology.rescale_flux(rc.data, Z_OBS, Z_SIM)
    rc = resample.shrink_to_redshift(rc, Z_OBS, Z_SIM)
    rc.data = rc.data / resample.get_spaxel_area(rc)
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
