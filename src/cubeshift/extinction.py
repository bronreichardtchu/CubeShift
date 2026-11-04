"""Milky Way (foreground) extinction, Cardelli, Clayton & Mathis (1989).

Get A_V for a target from NED's extinction calculator
(https://ned.ipac.caltech.edu/extinction_calculator), which uses the
Schlafly & Finkbeiner (2011) recalibration of the SFD (1998) dust map.

Known issues (review #7): only the optical branch (0.3-0.91 um) of CCM is
implemented, but it is evaluated at NIR wavelengths after redshifting; and the
old interactive workflow asked for a NED *band* A_lambda which was then used as A_V.
"""
from __future__ import annotations

import numpy as np

from .io import read_cube


def cardelli_A_lambda(lamdas, Av, Rv=3.1):
    """A_lambda (mag) from the CCM89 optical/NIR polynomial.

    Parameters
    ----------
    lamdas : array-like
        Wavelengths in Angstrom.
    Av : float
        V-band extinction (mag).
    Rv : float
        Total-to-selective extinction ratio.
    """
    lam_um = np.asarray(lamdas) / 1e4
    y = lam_um ** (-1) - 1.82
    a_x = (1.0 + 0.17699 * y - 0.50447 * y ** 2 - 0.02427 * y ** 3 +
           0.72085 * y ** 4 + 0.01979 * y ** 5 - 0.77530 * y ** 6 + 0.32999 * y ** 7)
    b_x = (1.41338 * y + 2.28305 * y ** 2 + 1.07233 * y ** 3 -
           5.38434 * y ** 4 - 0.62251 * y ** 5 + 5.30260 * y ** 6 - 2.09002 * y ** 7)
    return (a_x + b_x / Rv) * Av


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
