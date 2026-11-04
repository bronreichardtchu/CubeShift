"""Line-flux distributions (histograms/PDFs) before and after redshifting."""
import numpy as np
import matplotlib.pyplot as plt

from ..io import read_in_data


def extract_line_flux(filename, line_rest=4861.0, z=0.0, width=20.0):
    """
    Extract Hβ flux map from an IFU cube by integrating over a wavelength window.

    Parameters
    ----------
    filename : str or mpdaf.obj.Cube
        Path to IFU cube FITS file (or an already-loaded Cube).
    line_rest : float
        Rest-frame wavelength of the line (default: Hβ = 4861 Å).
    z : float
        Redshift of the cube.
    width : float
        Integration half-width in Angstroms.

    Returns
    -------
    flux_map : 2D array
        Hβ flux map (integrated flux per spaxel).
    """
    lam, data, _var, _header = read_in_data(filename)
    # If data cube is (nw, ny, nx)
    # Observed wavelength of line
    line_obs = line_rest * (1 + z)
    # Select window around the line
    mask = (lam > line_obs - width) & (lam < line_obs + width)
    flux_map = np.sum(data[mask, :, :], axis=0)
    return flux_map


def flux_distribution(flux_map):
    """
    Flatten flux map into 1D array, removing NaNs/zeros.
    """
    flat = flux_map.flatten()
    return flat[np.isfinite(flat) & (flat > 0)]


def normalized_pdf(cube, bins=100):
    """
    Returns a PDF normalized to the range [0, 1] along the axis.
    """
    # Flatten the cube to 1D for histogram
    data_flat = cube.flatten()

    # Compute histogram
    pdf, bin_edges = np.histogram(data_flat, bins=bins, density=True)

    # Normalize bins to 0–1 or any fixed axis scale (e.g., 0–100)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_centers_norm = (bin_centers - bin_centers.min()) / (bin_centers.max() - bin_centers.min())
    bin_centers_norm *= 100  # optional: scale to 0–100

    return bin_centers_norm, pdf


def plot_flux_histogram(flux1, flux2, label1="Original z", label2="Simulated z"):
    """
    Plot normalized histograms (PDFs) of flux distributions on a log-scaled x-axis.
    """
    plt.hist(flux1, bins=50, density=True, histtype='step', lw=2, label=label1)
    plt.hist(flux2, bins=50, density=True, histtype='step', lw=2, label=label2)
    plt.xscale("log")
    plt.xlabel("Hβ Flux")
    plt.ylabel("Normalized PDF")
    plt.legend()
    plt.show()


def run_hbeta_histogram(original_file, z_file, z_obs, z_sim):
    """
    Compare Hβ flux distributions of an original cube and a simulated cube.

    Parameters
    ----------
    original_file : str or mpdaf.obj.Cube
        Cube at the observed redshift.
    z_file : str or mpdaf.obj.Cube
        Simulated cube at ``z_sim``.
    z_obs, z_sim : float
        Redshifts of the two cubes.
    """
    flux_map_before = extract_line_flux(original_file, line_rest=4861.0, z=z_obs)
    flux_map_after  = extract_line_flux(z_file,  line_rest=4861.0, z=z_sim)

    flux_before = flux_distribution(flux_map_before)
    flux_after  = flux_distribution(flux_map_after)

    plot_flux_histogram(flux_before, flux_after, label1=f"z={z_obs}", label2=f"z={z_sim}")
