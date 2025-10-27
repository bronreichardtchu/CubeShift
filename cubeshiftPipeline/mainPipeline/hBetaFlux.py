import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS


def read_in_data_fits(filename):
    """
    Reads a FITS cube, robustly handling missing CRVAL3/CDELT3 keywords
    and different HDU extensions. Returns wavelength array, data cube, and header.
    
    Parameters
    ----------
    filename : str
        Path to FITS file
    
    Returns
    -------
    lamdas : np.ndarray
        Wavelength array (1D)
    data : np.ndarray
        3D data cube
    header : astropy.io.fits.Header
        Header of the HDU containing the data
    """
    hdul = fits.open(filename)
    
    # Find first HDU with actual data
    for hdu in hdul:
        if hdu.data is not None:
            data = hdu.data
            header = hdu.header
            break
    else:
        raise ValueError("No HDU with data found in FITS file")
    
    # Determine spectral axis
    try:
        spec_axis = 2  # default third axis
        if 'CRVAL3' in header and 'CDELT3' in header and 'NAXIS3' in header:
            lamdas = np.arange(header['CRVAL3'],
                               header['CRVAL3'] + header['NAXIS3'] * header['CDELT3'],
                               header['CDELT3'])
        else:
            # Use WCS to determine spectral axis
            wcs = WCS(header)
            if wcs.wcs.spec is not None:
                spec_axis = wcs.wcs.spec
            Nspec = data.shape[spec_axis]
            pix = np.arange(Nspec)
            # WCS expects pixel coordinates as N x ndim array
            lamdas = wcs.wcs_pix2world(np.moveaxis(np.expand_dims(pix, 1), 1, 0).T, 0)[0]
    except Exception as e:
        raise RuntimeError(f"Could not extract wavelength axis: {e}")
    
    return lamdas, data, header


def extract_line_flux(filename, line_rest=4861.0, z=0.0, width=20.0):
    """
    Extract Hβ flux map from an IFU cube by integrating over a wavelength window.

    Parameters
    ----------
    filename : str
        Path to IFU cube FITS file.
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
    lam, data, *rest = read_in_data_fits(filename)
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

import numpy as np

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
    flux_map_before = extract_line_flux(original_file, line_rest=4861.0, z=z_obs)
    flux_map_after  = extract_line_flux(z_file,  line_rest=4861.0, z=z_sim)

    flux_before = flux_distribution(flux_map_before)
    flux_after  = flux_distribution(flux_map_after)

    plot_flux_histogram(flux_before, flux_after, label1=f"z={z_obs}", label2=f"z={z_sim}")

if __name__ == "__main__":
    original_file = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/cgcg453_red_cont_subtracted_unnormalised_all_corrections_cube_cropped.fits"
    z_file = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/Output_cubes/CGCG453/CGCG453_z_3_f170lp_g235h_lsf.fits"
    z_obs = 0.025
    z_sim = 3


    run_hbeta_histogram(original_file, z_file, z_obs, z_sim)