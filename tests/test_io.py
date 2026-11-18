"""Reading cubes from FITS files and mpdaf objects (review v2 I/O items M5, M6, M13, M14)."""
import numpy as np
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
