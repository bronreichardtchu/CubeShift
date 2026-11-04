"""Milky Way (foreground) extinction, Cardelli, Clayton & Mathis (1989).

Use **A_V** (not the A_lambda of some band): from NED's extinction calculator
(https://ned.ipac.caltech.edu/extinction_calculator) take the "Landolt V" row,
or A_V = 3.1 * E(B-V).  The foreground dust is the same whatever redshift the
galaxy is simulated at, so the same A_V is removed at the observed wavelengths
and re-applied at the simulated ones; the extinction law supplies the
wavelength dependence.
"""
from __future__ import annotations

import warnings

import numpy as np

from .io import read_cube


def cardelli_A_lambda(lamdas, Av, Rv=3.1):
    """A_lambda (mag) from the CCM89 law, valid 0.1-3.3 um (all four branches).

    x = 1/lambda[um]: infrared 0.3-1.1 (power law), optical/NIR 1.1-3.3
    (polynomial), UV 3.3-8, far-UV 8-10.  Longward of 3.3 um (x < 0.3, e.g.
    NIRSpec G395) the IR power law is extrapolated, with a warning.

    Parameters
    ----------
    lamdas : array-like
        Wavelengths in Angstrom.
    Av : float
        V-band extinction (mag).
    Rv : float
        Total-to-selective extinction ratio.
    """
    x = 1e4 / np.atleast_1d(np.asarray(lamdas, dtype=float))
    if np.any(x > 10):
        raise ValueError("CCM89 is only defined for lambda > 1000 A")
    if np.any(x < 0.3):
        warnings.warn("CCM89 extrapolated beyond 3.3 um (IR power law)", stacklevel=2)
    a = np.empty_like(x)
    b = np.empty_like(x)

    ir = x < 1.1
    a[ir] = 0.574 * x[ir] ** 1.61
    b[ir] = -0.527 * x[ir] ** 1.61

    opt = (x >= 1.1) & (x < 3.3)
    y = x[opt] - 1.82
    a[opt] = (1.0 + 0.17699 * y - 0.50447 * y ** 2 - 0.02427 * y ** 3 +
              0.72085 * y ** 4 + 0.01979 * y ** 5 - 0.77530 * y ** 6 + 0.32999 * y ** 7)
    b[opt] = (1.41338 * y + 2.28305 * y ** 2 + 1.07233 * y ** 3 -
              5.38434 * y ** 4 - 0.62251 * y ** 5 + 5.30260 * y ** 6 - 2.09002 * y ** 7)

    uv = (x >= 3.3) & (x < 8)
    xu = x[uv]
    fa = np.where(xu >= 5.9, -0.04473 * (xu - 5.9) ** 2 - 0.009779 * (xu - 5.9) ** 3, 0.0)
    fb = np.where(xu >= 5.9, 0.2130 * (xu - 5.9) ** 2 + 0.1207 * (xu - 5.9) ** 3, 0.0)
    a[uv] = 1.752 - 0.316 * xu - 0.104 / ((xu - 4.67) ** 2 + 0.341) + fa
    b[uv] = -3.090 + 1.825 * xu + 1.206 / ((xu - 4.62) ** 2 + 0.263) + fb

    fuv = x >= 8
    yf = x[fuv] - 8
    a[fuv] = -1.073 - 0.628 * yf + 0.137 * yf ** 2 - 0.070 * yf ** 3
    b[fuv] = 13.670 + 4.257 * yf - 0.420 * yf ** 2 + 0.374 * yf ** 3

    out = (a + b / Rv) * Av
    return out if np.ndim(lamdas) else out[0]


#: Original name in extinctionCorrection.py.
compute_A_lambda = cardelli_A_lambda


def milky_way_extinction_correction(lamdas, data, Av, undo=False, Rv=3.1):
    """Correct a data cube for Milky Way extinction (or put it back with ``undo``).

    Parameters
    ----------
    lamdas : np.ndarray
        Wavelengths in Angstrom (length nz).
    data : np.ndarray
        Cube (nz, ny, nx).
    Av : float
        V-band extinction (mag).
    undo : bool
        If True, apply the extinction instead of removing it.

    Returns
    -------
    data : np.ndarray
    A_lam : np.ndarray
    """
    A_lam = cardelli_A_lambda(lamdas, Av, Rv)
    factor = 10 ** (0.4 * A_lam[:, None, None])
    data = data / factor if undo else data * factor
    return data, A_lam


def milky_way_extinction_correction_cube(cube, Av, undo=False, output_filename=None,
                                         return_a_lam=False, Rv=3.1):
    """Milky Way extinction correction for an mpdaf Cube.

    Parameters
    ----------
    cube : mpdaf.obj.Cube or str
        Cube or path to a FITS cube.
    Av : float
        V-band extinction (mag).
    undo : bool
        If True, apply the extinction instead of removing it.
    output_filename : str, optional
        Write the result here.
    return_a_lam : bool
        Also return A_lambda.

    Returns
    -------
    mpdaf.obj.Cube (and A_lambda if ``return_a_lam``)
        New cube with header keyword EXTCORR set.  Variance is not rescaled.
    """
    from mpdaf.obj import Cube

    cube = read_cube(cube)
    corrected_data, A_lam = milky_way_extinction_correction(cube.wave.coord(), cube.data, Av, undo, Rv)

    corrected_cube = Cube(data=corrected_data, wave=cube.wave, var=cube.var, mask=cube.mask,
                          wcs=cube.wcs, copy=True)
    corrected_cube.primary_header = cube.primary_header.copy()
    corrected_cube.data_header = cube.data_header.copy()
    corrected_cube.data_header["EXTCORR"] = (not undo, "Milky Way extinction correction applied")

    if output_filename is not None:
        corrected_cube.write(output_filename)
    if return_a_lam:
        return corrected_cube, A_lam
    return corrected_cube
