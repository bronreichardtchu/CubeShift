"""Reading cubes from FITS files and mpdaf objects (review v2 I/O items M5, M6, M13, M14)."""
import numpy as np
import pytest
from astropy.io import fits

from cubeshift import io


def _write_single_hdu_cube(path, nz, crval, cdelt, crpix):
    """A bare FITS cube in the primary HDU with only a spectral WCS on axis 3."""
    hdr = fits.Header()
    hdr["CRVAL3"] = crval
    hdr["CDELT3"] = cdelt
    hdr["CRPIX3"] = crpix
    fits.PrimaryHDU(data=np.zeros((nz, 2, 3)), header=hdr).writeto(path)


# --- M13: wavelength axis --------------------------------------------------------------

def test_wavelength_axis_uses_crpix3(tmp_path):
    """The reference pixel (CRPIX3 = 11) sits at CRVAL3, not the first pixel."""
    path = tmp_path / "crpix.fits"
    _write_single_hdu_cube(path, nz=20, crval=5000.0, cdelt=2.0, crpix=11)
    lam, _, _, _ = io.read_in_data(str(path))
    # pixel 11 (1-based) is index 10
    assert lam[10] == 5000.0
    assert lam[0] == 5000.0 - 10 * 2.0


def test_wavelength_axis_has_naxis3_elements(tmp_path):
    """np.arange(1.0, 1.3, 0.1) has 4 elements; the axis must have NAXIS3 = 3."""
    path = tmp_path / "rounding.fits"
    _write_single_hdu_cube(path, nz=3, crval=1.0, cdelt=0.1, crpix=1)
    lam, data, _, _ = io.read_in_data(str(path))
    assert lam.shape == (3,) == data.shape[:1]
    np.testing.assert_allclose(lam, [1.0, 1.1, 1.2])


# --- M5: variance chosen by EXTNAME ----------------------------------------------------

def _spectral_header():
    """A minimal spectral WCS so read_in_data takes the HDU as the data."""
    hdr = fits.Header()
    hdr["CRVAL3"], hdr["CDELT3"], hdr["CRPIX3"] = 5000.0, 1.0, 1.0
    return hdr


def test_muse_layout_variance_is_stat_not_data(tmp_path):
    """Empty primary, DATA, STAT: the old positional rule returned DATA as the variance."""
    data = np.full((4, 2, 3), 10.0)
    stat = np.full((4, 2, 3), 2.5)
    path = tmp_path / "muse.fits"
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(data, header=_spectral_header(), name="DATA"),
                  fits.ImageHDU(stat, name="STAT")]).writeto(path)
    _, d, var, _ = io.read_in_data(str(path))
    np.testing.assert_array_equal(d, data)
    np.testing.assert_array_equal(var, stat)


def test_kcwi_layout_uncertainty_is_squared_and_mask_skipped(tmp_path):
    """Flux in the primary, then MASK and UNCERT (sigma): var = UNCERT**2."""
    data = np.full((4, 2, 3), 10.0)
    sigma = np.full((4, 2, 3), 3.0)
    path = tmp_path / "kcwi.fits"
    fits.HDUList([fits.PrimaryHDU(data, header=_spectral_header()),
                  fits.ImageHDU(np.zeros((4, 2, 3), dtype=np.int16), name="MASK"),
                  fits.ImageHDU(sigma, name="UNCERT")]).writeto(path)
    _, _, var, _ = io.read_in_data(str(path))
    np.testing.assert_allclose(var, 9.0)


def test_inverse_variance_is_inverted(tmp_path):
    """IVAR -> 1/IVAR; ivar = 0 means no information and becomes NaN."""
    ivar = np.full((4, 2, 3), 0.25)
    ivar[0, 0, 0] = 0.0
    path = tmp_path / "ivar.fits"
    fits.HDUList([fits.PrimaryHDU(np.ones((4, 2, 3)), header=_spectral_header()),
                  fits.ImageHDU(ivar, name="IVAR")]).writeto(path)
    _, _, var, _ = io.read_in_data(str(path))
    assert np.isnan(var[0, 0, 0])
    np.testing.assert_allclose(var.ravel()[1:], 4.0)


def test_unnamed_extension_is_not_taken_as_variance(tmp_path):
    """An extension with no recognised EXTNAME is reported, not guessed."""
    path = tmp_path / "unnamed.fits"
    fits.HDUList([fits.PrimaryHDU(np.ones((4, 2, 3)), header=_spectral_header()),
                  fits.ImageHDU(np.ones((4, 2, 3)))]).writeto(path)
    with pytest.warns(UserWarning, match="no variance extension"):
        _, _, var, _ = io.read_in_data(str(path))
    assert var is None


# --- M6: Cube inputs use cube.wcs ------------------------------------------------------

def test_redshift_uses_cube_wcs_not_primary_header():
    """A Cube whose WCS is only in cube.wcs (MUSE files, in-memory cubes) keeps that WCS.

    The wavelength reference pixel is not the first one, so this also checks
    that the redshifted axis is lam_old * (1 + z_sim) / (1 + z_obs) everywhere.
    """
    from mpdaf.obj import WCS, Cube, WaveCoord

    from cubeshift.redshift import redshift_wavelength_axis

    # empty primary header; spatial WCS only on the object
    wcs = WCS(crval=(-37.7, 13.7), cdelt=(0.2 / 3600, -0.2 / 3600), crpix=(2, 3), shape=(4, 6), deg=True)
    wave = WaveCoord(crval=5000.0, cdelt=1.25, crpix=3, cunit="Angstrom")
    c = Cube(data=np.ones((5, 4, 6)), wcs=wcs, wave=wave)
    assert "CDELT1" not in c.primary_header and "CD1_1" not in c.primary_header

    out, lam_new = redshift_wavelength_axis(c, 0.0, 1.0)
    # same sky position at every pixel, same pixel size
    assert out.wcs.isEqual(c.wcs)
    # the whole axis is stretched by (1 + 1) / (1 + 0) = 2
    np.testing.assert_allclose(out.wave.coord(), 2.0 * c.wave.coord())
    np.testing.assert_allclose(lam_new, 2.0 * c.wave.coord())


# --- M14: flux unit and primary header survive the pipeline ----------------------------

def test_pipeline_keeps_unit_and_primary_header(mock_data_dir, quiet):
    """BUNIT and observation metadata reach the final cube; stale WCS cards do not."""
    import astropy.units as u

    from cubeshift import simulate_observation
    from cubeshift.testing import make_test_cube

    # KCWI-like input: flux unit, an OBJECT card, and the spatial WCS copied
    # into the primary header (which becomes wrong once the pixels shrink)
    flux_unit = u.Unit("1e-16 erg / (s cm2 Angstrom)")
    c = make_test_cube(wcs_in_primary=True)
    c.unit = flux_unit
    c.primary_header["OBJECT"] = "TESTGAL"

    with quiet():
        res = simulate_observation(c, 0.025, 3.0, "Keck_KCWI", "JWST_NIRSpec", data_dir=mock_data_dir)
    # same flux-density unit: per Angstrom per (new) spaxel
    assert res.cube.unit == flux_unit
    # metadata kept, input-grid WCS cards dropped from the primary header
    assert res.cube.primary_header["OBJECT"] == "TESTGAL"
    assert not any(k.startswith(("CD1_", "CDELT1", "CRPIX1")) for k in res.cube.primary_header)


def test_file_bunit_parsed_or_kept_as_csbunit(tmp_path):
    """A parseable BUNIT becomes the cube unit; an unparseable one is kept as CSBUNIT."""
    import astropy.units as u

    from cubeshift.redshift import redshift_wavelength_axis

    def _cube_file(name, bunit):
        # flux in the primary HDU with a minimal 3-axis WCS and the given BUNIT
        hdr = _spectral_header()
        hdr["CDELT1"], hdr["CDELT2"] = -0.29 / 3600, 0.29 / 3600
        hdr["CTYPE1"], hdr["CTYPE2"] = "RA---TAN", "DEC--TAN"
        hdr["BUNIT"] = bunit
        path = tmp_path / name
        fits.PrimaryHDU(np.ones((4, 2, 3)), header=hdr).writeto(path)
        return str(path)

    out, _ = redshift_wavelength_axis(_cube_file("ok.fits", "erg/s/cm2/Angstrom"), 0.0, 1.0)
    assert out.unit == u.Unit("erg / (s cm2 Angstrom)")

    with pytest.warns(UserWarning, match="CSBUNIT"):
        out, _ = redshift_wavelength_axis(_cube_file("odd.fits", "COUNTS/PIXEL/BANANA"), 0.0, 1.0)
    assert out.data_header["CSBUNIT"] == "COUNTS/PIXEL/BANANA"
