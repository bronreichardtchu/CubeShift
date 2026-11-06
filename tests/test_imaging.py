"""Broadband imaging (HST -> HST, same filter): simulate_image_observation."""
import os

import numpy as np
import pytest
from astropy.cosmology import Planck18 as cosmo

from cubeshift import read_hst_image, simulate_image_observation
from cubeshift.cosmology import angular_scale_ratio, flux_dimming_factor
from cubeshift.testing import make_hst_like_image, write_hst_like_drc

Z_OBS, D_OBS = 0.00048, 2.0          # NGC 300
ACS = "HST_ACS_WFC"


def _sim(image, z_sim, quiet, **kw):
    with quiet():
        return simulate_image_observation(image, Z_OBS, z_sim, ACS, ACS, **kw)


@pytest.mark.parametrize("distance", [None, D_OBS])
@pytest.mark.parametrize("z_sim", [0.01, 0.05])
def test_total_counts_follow_flat_fnu_dimming(quiet, z_sim, distance):
    image = make_hst_like_image(nx=600, ny=640, sigma_pix=60)
    r = _sim(image, z_sim, quiet, distance_obs_mpc=distance)
    dl_obs = distance if distance is not None else cosmo.luminosity_distance(Z_OBS).value
    expected = (dl_obs / cosmo.luminosity_distance(z_sim).value) ** 2 * (1 + z_sim) / (1 + Z_OBS)
    assert r.dimming_factor == pytest.approx(expected, rel=1e-10)
    assert np.ma.sum(r.image.data) / np.ma.sum(image.data) == pytest.approx(expected, rel=0.01)


def test_flat_flambda_option():
    assert flux_dimming_factor(Z_OBS, 1.0, sed="flat_flambda") / flux_dimming_factor(Z_OBS, 1.0, sed="flat_fnu") \
        == pytest.approx(((1 + Z_OBS) / 2.0) ** 2)
    with pytest.raises(ValueError):
        flux_dimming_factor(Z_OBS, 1.0, sed="blackbody")


def test_surface_brightness_dims_as_1pz_cubed(quiet):
    """Flat f_nu, same filter: SB per true arcsec^2 dims as (1+z)^-3."""
    z_sim = 0.05
    image = make_hst_like_image(nx=500, ny=500, sigma_pix=1e6)          # ~uniform
    r = _sim(image, z_sim, quiet)
    sb0 = np.ma.mean(image.data) / 0.05 ** 2
    dx, dy = r.pixel_scale
    sb1 = np.ma.mean(r.image.data) / (dx * dy)
    assert sb1 / sb0 == pytest.approx(((1 + Z_OBS) / (1 + z_sim)) ** 3, rel=1e-3)


@pytest.mark.parametrize("z_sim", [0.01, 0.05])
def test_pixel_scale_and_rotation(quiet, z_sim):
    image = make_hst_like_image(rot_deg=8.0)
    r = _sim(image, z_sim, quiet, distance_obs_mpc=D_OBS)
    f_ang = angular_scale_ratio(Z_OBS, z_sim, distance_obs_mpc=D_OBS)
    assert r.angular_scale_ratio == pytest.approx(f_ang)
    assert r.bin_factors[0] == round(1 / f_ang)
    assert r.pixel_scale == pytest.approx((r.bin_factors[0] * 0.05 * f_ang,) * 2, rel=1e-6)
    assert r.image.wcs.get_rot() == pytest.approx(8.0, abs=1e-6)
    # no padding by default: output covers only the input footprint
    assert r.image.shape == (image.shape[0] // r.bin_factors[1], image.shape[1] // r.bin_factors[0])
    np.testing.assert_allclose(r.image.wcs.get_center(), image.wcs.get_center(), atol=0.05 / 3600)


def _second_moment_fwhm(img):
    img = np.ma.filled(img, 0.0)
    y, x = np.indices(img.shape)
    tot = img.sum()
    xc, yc = (img * x).sum() / tot, (img * y).sum() / tot
    return 2.3548 * np.sqrt((img * ((x - xc) ** 2 + (y - yc) ** 2)).sum() / tot / 2)


def test_point_source_gets_target_psf(quiet):
    """At z_sim the source PSF is ~0.1*f_ang", so a point source ends with the ACS PSF (0.10")."""
    image = make_hst_like_image(nx=1320, ny=1320, point_source=True)
    r = _sim(image, 0.01, quiet, distance_obs_mpc=D_OBS)
    fwhm = _second_moment_fwhm(r.image.data) * r.pixel_scale[0]
    assert fwhm == pytest.approx(0.10, rel=0.05)


def test_drc_reader_and_masking(tmp_path, quiet):
    path = write_hst_like_drc(str(tmp_path / "t_drc.fits"), uncovered_corner=60, sigma_pix=40)
    image = read_hst_image(path)
    assert np.ma.getmaskarray(image.data).sum() == 60 * 60
    assert image.data_header["CSBUNIT"] == "ELECTRONS/S" and image.data_header["FILTER1"] == "F555W"
    r = _sim(path, 0.01, quiet, distance_obs_mpc=D_OBS, output_dir=str(tmp_path), galaxy_name="t")
    bx, by = r.bin_factors
    assert bx == 22
    # bins entirely inside the uncovered 60x60 corner are masked; partly covered ones are not
    mask = np.ma.getmaskarray(r.image.data)
    assert mask[0, 0] and mask.sum() == 4
    from astropy.io import fits

    h = fits.getheader(r.path, 1)
    assert h["BUNIT"] == "ELECTRONS/S" and h["CSZSIM"] == 0.01 and h["CSBINX"] == bx
    assert fits.getheader(r.path, 0)["TARGNAME"] == "TEST"


REAL = os.environ.get("CUBESHIFT_HST_TEST_FILE")


@pytest.mark.slow
@pytest.mark.skipif(not REAL, reason="set CUBESHIFT_HST_TEST_FILE to an HST _drz/_drc file")
@pytest.mark.parametrize("z_sim", [0.01, 0.05, 0.5, 1.0, 3.0])
def test_real_hst_image(quiet, z_sim):
    image = read_hst_image(REAL)
    rot = image.wcs.get_rot()
    r = _sim(image, z_sim, quiet, distance_obs_mpc=D_OBS)
    data = np.ma.getdata(r.image.data)[~np.ma.getmaskarray(r.image.data)]
    assert data.size > 0 and np.all(np.isfinite(data))
    assert r.image.wcs.get_rot() == pytest.approx(rot, abs=1e-6)
    # Counts are conserved per bin, where a partly covered bin (edge of the drizzled
    # footprint) holds its covered pixels' mean x the full bin area (see ROADMAP #12).
    bx, by = r.bin_factors
    ny, nx = image.shape
    y0, x0 = (ny % by) // 2, (nx % bx) // 2
    sl = (slice(y0, y0 + ny // by * by), slice(x0, x0 + nx // bx * bx))
    good = ~np.ma.getmaskarray(image.data)[sl]
    data = np.where(good, np.ma.getdata(image.data)[sl], 0.0)
    shape = (ny // by, by, nx // bx, bx)
    flux = data.reshape(shape).sum(axis=(1, 3))
    cov = good.reshape(shape).mean(axis=(1, 3))
    expected = (flux / np.where(cov > 0, cov, 1)).sum() * r.dimming_factor
    assert np.ma.sum(r.image.data) == pytest.approx(expected, rel=1e-4)
