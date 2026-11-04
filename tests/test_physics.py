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


@pytest.mark.xfail(strict=True, reason="review #5: LSF matching ignores the source resolution")
def test_lsf_matches_nirspec_resolution(mock_data_dir, quiet):
    """Unresolved KCWI line (R=4000) observed with G235H (R~2345 at 2.05um) -> FWHM ~9 A."""
    c = make_test_cube(line_sigma=0.6)[:, 12:18, 12:18]
    with quiet():
        rc, _ = redshift_wavelength_axis(c, Z_OBS, Z_SIM)
        bl, _ = spectral.apply_dispersion_pipeline(rc, data_dir=mock_data_dir)
    sp = np.nansum(np.asarray(bl.data), axis=(1, 2))
    fit = fit_gaussian_line(bl.wave.coord(), sp, center=5006.84 * 4, window=60)
    assert fit["fwhm"] == pytest.approx(9.0, abs=1.0)


@pytest.mark.xfail(strict=True, reason="review #6: final spectral resampling is point sampling at lambda/1000")
def test_final_spectral_resampling_conserves_line_flux(quiet):
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
        assert f_out / f_in == pytest.approx(1.0, rel=0.05)


@pytest.mark.xfail(strict=True, reason="review #7: CCM optical polynomial used in the NIR")
def test_ccm_infrared_branch():
    # CCM89 IR branch at 2 um: A/A_V = (0.574 - 0.527/3.1) * 0.5**1.61 = 0.132
    assert extinction.cardelli_A_lambda(np.array([20000.0]), Av=1.0)[0] == pytest.approx(0.132, abs=0.005)


@pytest.mark.xfail(strict=True, reason="review #9: variance dropped by the spectral resamplers")
def test_variance_survives_pipeline(mock_data_dir, quiet):
    from cubeshift import simulate_observation

    np.random.seed(0)
    with quiet():
        res = simulate_observation(make_test_cube(), Z_OBS, Z_SIM, "Keck_KCWI", "JWST_NIRSpec",
                                   data_dir=mock_data_dir)
    assert res.cube.var is not None
