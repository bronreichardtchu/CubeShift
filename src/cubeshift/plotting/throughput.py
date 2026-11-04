"""Plots for checking the NIRSpec filter throughput step."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..throughput import load_filter_transmission


def plot_throughput_and_spectrum(cube, wave_filter, thr_filter, x=0, y=0):
    """Filter throughput with the (filtered, normalised) spectrum at spaxel (x, y).

    Parameters
    ----------
    cube : mpdaf.obj.Cube
        Filtered cube (wavelengths in Angstrom).
    wave_filter : np.ndarray
        Filter wavelengths in micron.
    thr_filter : np.ndarray
        Throughput (0-1).
    """
    wave_cube = cube.wave.coord() / 1e4
    spec_filtered = cube.data[:, y, x]

    plt.figure(figsize=(10, 6))
    plt.plot(wave_filter, thr_filter, label="Filter Throughput", color="green", alpha=0.7)
    plt.plot(wave_cube, spec_filtered / np.max(spec_filtered), label="Filtered Spectrum (norm)", color="blue")
    plt.xlabel("Wavelength (um)")
    plt.ylabel("Normalized Flux / Throughput")
    plt.title(f"Spectrum at pixel (x={x}, y={y}) with Filter Applied")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_filter_results(filtered_cube, filter_file, x, y, line_center_angstrom, width_angstrom=20):
    """Filter curve, spectrum at (x, y), narrow-band image around a line, and broad-band image.

    Parameters
    ----------
    filtered_cube : mpdaf.obj.Cube
    filter_file : str
        Path to a ``jwst_nirspec_*_trans.fits`` table.
    x, y : int
        Spaxel to plot.
    line_center_angstrom : float
        Observed line wavelength for the narrow-band image.
    width_angstrom : float
        Narrow-band width.
    """
    lam_filter, throughput = load_filter_transmission(filter_file)
    plt.plot(lam_filter, throughput)
    plt.xlabel("Wavelength (um)")
    plt.ylabel("Throughput")
    plt.title("Filter Transmission Curve")
    plt.grid()
    plt.show()

    wavelengths = filtered_cube.wave.coord()
    plt.figure(figsize=(8, 4))
    plt.plot(wavelengths, filtered_cube[:, y, x].data, color="blue")
    plt.xlabel("Wavelength (A)")
    plt.ylabel("Flux (filtered)")
    plt.title(f"Spectrum at pixel (y={y}, x={x})")
    plt.grid()
    plt.show()

    band = np.abs(wavelengths - line_center_angstrom) < width_angstrom / 2
    plt.figure(figsize=(6, 6))
    plt.imshow(np.nansum(filtered_cube.data[band], axis=0), origin="lower", cmap="inferno")
    plt.colorbar(label="Flux")
    plt.title(f"Narrowband image around {line_center_angstrom:.0f} A")
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.imshow(np.nansum(filtered_cube.data, axis=0), origin="lower", cmap="magma")
    plt.colorbar(label="Flux")
    plt.title("Broadband image after filter applied")
    plt.show()
