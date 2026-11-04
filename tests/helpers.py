import astropy.units as u
import numpy as np


def snap(cube):
    """Same fields as tests/golden/make_golden.py:snap, without prefixes."""
    d = cube.data
    return {
        "data": np.ma.getdata(d).astype(float),
        "mask": np.ma.getmaskarray(d),
        "var": np.ma.getdata(cube.var).astype(float) if cube.var is not None else np.zeros(0),
        "wave": cube.wave.coord(),
        "wcs_step": np.asarray(cube.wcs.get_step(unit=u.arcsec)),
        "wcs_crpix": np.asarray([cube.wcs.get_crpix1(), cube.wcs.get_crpix2()]),
        "wcs_crval": np.asarray([cube.wcs.get_crval1(), cube.wcs.get_crval2()]),
    }


def assert_matches(cube, golden, prefix, rtol=1e-9, atol=0.0):
    """Compare a cube with the golden record ``prefix``.

    Data/var are compared only where unmasked (masked values are undefined).
    """
    got = snap(cube)
    exp_mask = golden[prefix + "mask"]
    np.testing.assert_array_equal(got["mask"], exp_mask, err_msg=f"{prefix}mask")
    keep = ~exp_mask
    np.testing.assert_allclose(got["data"][keep], golden[prefix + "data"][keep], rtol=rtol, atol=atol,
                               equal_nan=True, err_msg=f"{prefix}data")
    gv = golden[prefix + "var"]
    if gv.size:
        np.testing.assert_allclose(got["var"][keep], gv[keep], rtol=rtol, atol=atol, equal_nan=True,
                                   err_msg=f"{prefix}var")
    else:
        assert got["var"].size == 0, f"{prefix}var: expected no variance"
    for k in ("wave", "wcs_step", "wcs_crpix", "wcs_crval"):
        np.testing.assert_allclose(got[k], golden[prefix + k], rtol=1e-12, err_msg=prefix + k)
