"""Smoke and unit tests for the package structure and small helpers."""
import importlib
import pkgutil

import numpy as np
import pytest

import cubeshift
from cubeshift import binning, config, io, redshift
from cubeshift.testing import make_test_cube


def _all_modules():
    return [m.name for m in pkgutil.walk_packages(cubeshift.__path__, prefix="cubeshift.")]


@pytest.mark.parametrize("name", _all_modules())
def test_every_module_imports(name):
    importlib.import_module(name)


def test_air_to_vac():
    # Greisen et al. 2006 eq. 65 as implemented gives Halpha 6562.80 A (air) -> 6564.71 A.
    # NB the IAU standard (Morton 1991 / Ciddor) gives 6564.61 A: this formula's
    # refractivity is ~5% high (0.1 A, ~5 km/s, at Halpha).
    assert redshift.air_to_vac(np.array([6562.80]))[0] == pytest.approx(6564.71, abs=0.01)


def test_read_in_data_always_four_values(tmp_path):
    c = make_test_cube(nx=5, ny=4, nw=6)
    assert len(io.read_in_data(c)) == 4
    c_novar = make_test_cube(nx=5, ny=4, nw=6, with_var=False)
    lam, data, var, hdr = io.read_in_data(c_novar)
    assert var is None and data.shape == (6, 4, 5)
    path = tmp_path / "c.fits"
    c.write(str(path))
    lam, data, var, hdr = io.read_in_data(str(path))
    assert data.shape == (6, 4, 5) and var is not None
    np.testing.assert_allclose(lam, c.wave.coord())


def test_read_fits_dispatch(tmp_path):
    c = make_test_cube(nx=5, ny=4, nw=6)
    c.write(str(tmp_path / "c.fits"))
    c[0].write(str(tmp_path / "i.fits"))
    assert type(io.read_fits(str(tmp_path / "c.fits"))).__name__ == "Cube"
    assert type(io.read_fits(str(tmp_path / "i.fits"))).__name__ == "Image"


@pytest.mark.parametrize("method", ["sum", "mean", "median"])
def test_bin_data_shapes_and_flux(method):
    c = make_test_cube(nx=12, ny=9, nw=5)
    b = binning.bin_data(3, 3, c, method=method)
    assert b.shape == (5, 3, 4)
    assert b.var.shape == b.shape
    if method == "sum":
        np.testing.assert_allclose(b.data.sum(), c.data.sum(), rtol=1e-12)
    step_in = np.abs(c.wcs.get_step(unit="arcsec"))
    np.testing.assert_allclose(np.abs(b.wcs.get_step(unit="arcsec")), 3 * step_in)


def test_bin_data_inplace_and_bad_args():
    c = make_test_cube(nx=12, ny=9, nw=5)
    with pytest.raises(ValueError):
        binning.bin_data(2, 2, c, margin="middle")
    with pytest.raises(ValueError):
        binning.bin_data(2, 2, c, method="max")
    with pytest.raises(ValueError):
        binning.bin_data(20, 2, c)
    out = binning.bin_data(3, 3, c, inplace=True)
    assert out is c and c.shape == (5, 3, 4)


def test_data_dir_resolution(tmp_path, monkeypatch):
    monkeypatch.delenv(config.ENV_VAR, raising=False)
    with pytest.raises(FileNotFoundError):
        config.get_data_dir("filters")
    (tmp_path / "Dispersion Filters").mkdir()
    assert config.get_data_dir("dispersion", tmp_path) == tmp_path / "Dispersion Filters"
    monkeypatch.setenv(config.ENV_VAR, str(tmp_path))
    assert config.get_data_dir("filters") == tmp_path


def test_unknown_telescope():
    with pytest.raises(KeyError):
        cubeshift.simulate_observation(make_test_cube(), 0.025, 3.0, "Keck_KCWI", "Hubble")


def test_redshift_builds_wcs_from_pixscale():
    """Header without a spatial WCS + pixscale_arcsec (mpdaf takes (y, x) order, no cunit/ctype)."""
    from mpdaf.obj import Cube, WaveCoord

    c = Cube(data=np.ones((5, 4, 6)), wave=WaveCoord(crval=5000.0, cdelt=1.0, crpix=1, cunit="Angstrom"))
    out, _ = redshift.redshift_wavelength_axis(c, 0.0, 1.0, pixscale_arcsec=(0.2, 0.3))
    dy, dx = out.wcs.get_step(unit="arcsec")
    assert (dy, dx) == pytest.approx((0.3, 0.2))
    assert out.wcs.get_axis_increments(unit="arcsec")[1] < 0       # RA increases to the left
