from mpdaf.obj import Cube

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
from astropy.wcs.utils import proj_plane_pixel_scales


def measure_flux(
    cube, z,
    w_line=4861.0,  # rest wavelength
    line_window=(-15, 25),  # relative span in Å (used for width only)
    line_name="Line",
    search_window=(-30, 30),  # for detecting emission peak
    cont_blue=(-80, -15),
    cont_red=(25, 80),
    debug=False, debug_coords=(19, 13)
):
    lam_center = w_line * (1 + z)

    # initialize flux map
    flux_map = np.full(cube.shape[1:], np.nan)
    print("Spectral pixel scale (Δλ):", np.median(np.diff(cube.wave.coord())))
    lam = cube.wave.coord()  # wavelength axis (Å)
    delta_lambda = np.median(np.diff(lam))  # spectral pixel scale (Å/pixel)




    for j in range(cube.shape[1]):
        for i in range(cube.shape[2]):
            spax = cube[:, j, i]
            if np.all(np.isnan(spax.data)):
                continue

            lam = spax.wave.coord()
            flux = spax.data

            if line_name == "hGamma" and (j, i) == debug_coords:
                print(f"[DEBUG {line_name}] lam_center={lam_center:.1f}, lam_peak={lam_peak:.1f}")
                print(f"[DEBUG {line_name}] band_line={band_line}")
                print(f"[DEBUG {line_name}] band_cont1={band_cont1}, band_cont2={band_cont2}")
                print(f"[DEBUG {line_name}] mask_search points={np.sum(mask_search)}, "
                    f"mask_cont points={np.sum(mask_cont)}, mask_line points={np.sum(mask_line)}")

            
            if debug and (j, i) == debug_coords:
                print(f"[DEBUG {line_name}] Spaxel ({j},{i}) λ range {lam.min():.1f}–{lam.max():.1f}")
                print(f"[DEBUG {line_name}] λ_center={lam_center:.1f}, search_window={search_window}")
                print(f"[DEBUG {line_name}] flux first 5={flux[:5]}")

            # --- Step 1: detect peak within search window ---
            # to deal with the nan issue for hGamma
            if line_name.lower().startswith("hγ") or line_name.lower().startswith("hgamma"):
                search_window = (-100, 100)   # widen for weak line
            mask_search = (lam > lam_center + search_window[0]) & (lam < lam_center + search_window[1])
            if np.sum(mask_search) < 1:
                if debug and (j, i) == debug_coords:
                    print(f"[DEBUG Hγ] Skipping: mask_search too small ({np.sum(mask_search)})")
                    
                continue
                print(f"{np.sum(mask_search)} is less than 1. Continuing.")
            peak_idx = np.argmax(flux[mask_search])
            lam_peak = lam[mask_search][peak_idx]

            # --- Step 2: define line band centered on lam_peak ---
            line_width = line_window[1] - line_window[0]
            half_width = 1.5 * delta_lambda  # ~1.5 pixels wide
            band_line = (lam_peak - half_width, lam_peak + half_width)
            
            
            # --- Step 3: make contiguous continuum bands ---
            band_cont1 = tuple(sorted((lam_peak + cont_blue[0], band_line[0])))
            band_cont2 = tuple(sorted((band_line[1], lam_peak + cont_red[1])))
            band_line  = tuple(sorted(band_line))


            # --- Step 4: build masks ---
            mask_cont = ((lam > band_cont1[0]) & (lam < band_cont1[1])) | \
                        ((lam > band_cont2[0]) & (lam < band_cont2[1]))
            mask_line = (lam > band_line[0]) & (lam < band_line[1])

            if np.sum(mask_cont) < 3 or np.sum(mask_line) < 1:
                continue

            # Quadratic continuum fit only if we have enough points
            if np.sum(mask_cont) >= 3:
                coeffs = np.polyfit(
                    np.ma.filled(lam[mask_cont], np.nan),
                    np.ma.filled(flux[mask_cont], np.nan),
                    2
                )
                cont_fit = np.polyval(coeffs, lam)
                flux_sub = flux - cont_fit
            else:
                continue  # skip spaxel if no reliable continuum

            # Integrate only if line window has valid values
            if np.sum(mask_line) >= 2:
                lam_vals = np.ma.filled(lam[mask_line], 0.0)
                flux_vals = np.ma.filled(flux_sub[mask_line], 0.0)
                flux_map[j, i] = np.trapz(flux_vals, lam_vals)
                
            else:
                continue

            # --- Debugging ---
            if debug and (j, i) == debug_coords:
                print(f"[DEBUG] Spaxel ({j},{i})")
                print(f" Detected peak λ = {lam_peak:.2f}")
                print(f" Line band centered on peak: {band_line}")
                print(f" Continuum bands: {band_cont1}, {band_cont2}")

                if line_name == "hGamma":
                    print(f"[DEBUG Hγ] λ_center={lam_center:.1f}, search_window={search_window}")
                    print(f"[DEBUG Hγ] lam range: {lam.min():.1f}–{lam.max():.1f}")
                    print(f"[DEBUG Hγ] flux sample: {flux[:10]}")

                lam_min = lam_center - 100
                lam_max = lam_center + 100
                mask_plot = (lam > lam_min) & (lam < lam_max)
                print(f"[DEBUG] Spaxel ({j}, {i}) λ range: {lam.min():.1f}–{lam.max():.1f}")
                print(f"[DEBUG] mask_line points: {np.sum(mask_line)}, "f"mask_cont points: {np.sum(mask_cont)}")
                print(f"[DEBUG] Integrated flux = {flux_map[j,i]}")

                # --- Plot A: spectrum + bands ---
                plt.figure(figsize=(8,4))
                plt.plot(lam[mask_plot], flux[mask_plot], "k", lw=1, label="Spectrum")
                plt.axvline(lam_peak, color="g", ls="--", lw=1, alpha=0.8, label="Detected peak")
                plt.axvspan(*band_cont1, color="blue", alpha=0.3, label="Blue continuum")
                plt.axvspan(*band_line,  color="green", alpha=0.3, label="Line region")
                plt.axvspan(*band_cont2, color="red", alpha=0.3, label="Red continuum")
                plt.xlabel("Wavelength (Å)")
                plt.ylabel("Flux")
                plt.title(f"Spectrum at spaxel (y={j}, x={i})")
                plt.legend()
                plt.tight_layout()
                plt.show()

                # --- Plot B: continuum fit + subtraction ---
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9,7), sharex=True)

                # Top panel
                ax1.plot(lam[mask_plot], flux[mask_plot], color="black", lw=1, label="Original spectrum")
                ax1.plot(lam[mask_plot], cont_fit[mask_plot], color="orange", lw=1.5, label="Quadratic continuum")
                ax1.axvspan(*band_cont1, color="blue", alpha=0.3, label="Blue cont.")
                ax1.axvspan(*band_line,  color="green", alpha=0.3, label="Line region")
                ax1.axvspan(*band_cont2, color="red", alpha=0.3, label="Red cont.")
                ax1.axvline(lam_peak, color="g", ls="--", lw=1, alpha=0.8, label="Detected peak")
                ax1.set_ylabel("Flux")
                ax1.legend()

                # Bottom panel
                ax2.plot(lam[mask_plot], flux_sub[mask_plot], "purple", lw=1, label="Continuum-subtracted")
                ax2.axhline(0, color="k", ls="--", lw=1)
                ax2.axvspan(*band_line, color="green", alpha=0.3)
                ax2.set_xlabel("Wavelength (Å)")
                ax2.set_ylabel("Flux (subtracted)")
                ax2.legend()

                plt.tight_layout()
                plt.show()

    return flux_map



def plot_flux_map(flux_map):

    print("Flux map stats:")
    print(" min =", np.nanmin(flux_map))
    print(" max =", np.nanmax(flux_map))
    print(" mean =", np.nanmean(flux_map))
    print(" fraction NaNs =", np.isnan(flux_map).sum() / flux_map.size)

    # Plot with percentile scaling

    vmin = np.nanpercentile(flux_map, 5)
    vmax = np.nanpercentile(flux_map, 95)

    plt.figure(figsize=(8,6))
    im = plt.imshow(flux_map, origin='lower', cmap='inferno',
                    vmin=vmin, vmax=vmax)
    plt.colorbar(im, label=r'H$\beta$ flux')
    plt.title(r'H$\beta$ Flux Map')
    plt.xlabel('x [spaxel]')
    plt.ylabel('y [spaxel]')



    plt.show()

def measure_hgamma_flux(cube, z, debug=False, debug_coords=(19, 13)):
    """
    Wrapper for measure_hbeta_flux to measure Hγ line (4340 Å).

    Parameters
    ----------
    cube : mpdaf.obj.Cube
        The IFU cube.
    z : float
        Source redshift.
    debug : bool, optional
        If True, prints diagnostic info and shows a debug plot for the given spaxel.
    debug_coords : tuple
        (j, i) coordinates of the spaxel to plot when debug=True.

    Returns
    -------
    flux_map : 2D numpy.ndarray
        Map of integrated Hγ fluxes.
    """

    return measure_flux(
        cube, z,
        w_line=4340.0,              # Hγ rest wavelength
        line_name="hGamma",
        line_window=(10, 40),      # tweak if needed
        cont_blue=(-45, 10),       # tweak if needed
        cont_red=(40, 110),          # tweak if needed
        debug=debug,
        debug_coords=debug_coords
    )

def measure_hbeta_flux(cube, z, debug=False, debug_coords=(19, 13)):
    """
    Wrapper for measure_hbeta_flux to measure Hγ line (4340 Å).

    Parameters
    ----------
    cube : mpdaf.obj.Cube
        The IFU cube.
    z : float
        Source redshift.
    debug : bool, optional
        If True, prints diagnostic info and shows a debug plot for the given spaxel.
    debug_coords : tuple
        (j, i) coordinates of the spaxel to plot when debug=True.

    Returns
    -------
    flux_map : 2D numpy.ndarray
        Map of integrated Hγ fluxes.
    """

    return measure_flux(
        cube, z,
        w_line=4861.0,              # Hγ rest wavelength
        line_name="hBeta",
        line_window=(-15, 25),      # tweak if needed
        cont_blue=(-110, -15), 
        cont_red=(25, 80),         # tweak if needed
        debug=debug,
        debug_coords=debug_coords
    )


def compute_Ahbeta(hbeta_flux_map, hgamma_flux_map, law="cardelli"):
    """
    Compute internal extinction at Hbeta using Hbeta/Hgamma ratio.
    
    Parameters
    ----------
    hbeta_flux_map : 2D array
        Measured Hbeta flux map
    hgamma_flux_map : 2D array
        Measured Hgamma flux map
    law : str
        Extinction law to use ("cardelli" default)

    Returns
    -------
    Ahbeta_map : 2D array
        Extinction (mag) at Hbeta for each spaxel
    """
    # observed ratio
    ratio_obs = hbeta_flux_map / hgamma_flux_map

    # intrinsic Case B recombination ratio
    ratio_int = 2.15  

    # extinction law coefficients (Cardelli+89, Rv=3.1)
    k_hbeta = 3.61   # k(4861 Å)
    k_hgamma = 4.08  # k(4341 Å)
    diff_ext = k_hgamma - k_hbeta  # ~0.47

    # compute E(B-V)
    valid = (ratio_obs > 0)
    ebv = np.full_like(ratio_obs, np.nan, dtype=float)
    ebv[valid] = (2.5 / diff_ext) * np.log10(ratio_obs[valid] / ratio_int)


    # compute A(Hbeta)
    Ahbeta = k_hbeta * ebv

    # replace negative values (if any) with 0
    Ahbeta[Ahbeta < 0] = 0

    return Ahbeta


# Consants
C_Halpha = 5.5335e-42         # Msun yr^-1 per (erg s^-1)
F_Halpha_over_F_Hbeta = 2.87  # dimensionless

def flux_to_Halpha_luminosity(hbeta_flux_map, F_Halpha_over_F_Hbeta=F_Halpha_over_F_Hbeta, z=None, dl_cm=None):
    """
    Convert Hbeta flux map (erg s^-1 cm^-2) to Halpha luminosity map (erg s^-1).
    """
    if dl_cm is None:
        if z is None:
            raise ValueError("Provide either z or dl_cm")
        dl = cosmo.luminosity_distance(z).to(u.cm).value
    else:
        dl = float(dl_cm)

    halpha_flux_map = F_Halpha_over_F_Hbeta * np.array(hbeta_flux_map, dtype=float)
    L_halpha = 4.0 * np.pi * (dl ** 2) * halpha_flux_map
    return L_halpha


def compute_sfr_from_hbeta_flux(hbeta_flux_map, Ahbeta_map,
                                C_Halpha=C_Halpha,
                                F_Halpha_over_F_Hbeta=F_Halpha_over_F_Hbeta,
                                z=None, dl_cm=None):
    """
    Compute SFR map (Msun yr^-1) from Hbeta flux map and extinction at Hbeta.
    """
    hbeta_flux_map = np.array(hbeta_flux_map, dtype=float)
    Ahbeta_map = np.array(Ahbeta_map, dtype=float)

    bad_mask = ~np.isfinite(hbeta_flux_map) | (hbeta_flux_map <= 0)

    # Correct Hbeta flux for dust at Hbeta
    corrected_hbeta_flux = hbeta_flux_map * (10.0 ** (0.4 * Ahbeta_map))

    # Convert corrected Hbeta flux -> Halpha luminosity
    L_halpha = flux_to_Halpha_luminosity(corrected_hbeta_flux,
                                         F_Halpha_over_F_Hbeta=F_Halpha_over_F_Hbeta,
                                         z=z, dl_cm=dl_cm)

    # SFR (Msun/yr) from Halpha luminosity
    sfr_map = C_Halpha * L_halpha

    sfr_map[bad_mask] = 0.0
    return sfr_map

import numpy as np
from astropy.cosmology import Planck18 as cosmo
from astropy.wcs import WCS

# -------------------------------
# Calibration constants
# -------------------------------
CALIBRATIONS = {
    "reichardtchu2025": {
        "C_Halpha": 5.5335e-42,  # Msol/yr / erg/s
        "Halpha_Hbeta_ratio": 2.87
    },
    "hao2011": {
        "C_Halpha": 10**-41.257,  # Msol/yr / erg/s
        "Halpha_Hbeta_ratio": 2.87
    }
}

# -------------------------------
# Main SFR calculator
# -------------------------------
def compute_sfr_from_Hbeta(
    cube,
    F_Hbeta,
    z,
    A_Hbeta,
    calibration="reichardtchu2025",
    wcs=None
):
    """
    Compute star formation rates from Hβ fluxes.

    Parameters
    ----------
    F_Hbeta : 2D array
        Observed Hβ flux per spaxel (erg/s/cm^2).
    z : float
        Redshift of the galaxy.
    A_Hbeta : 2D array or float
        Extinction at Hβ (mag).
    calibration : str
        Choice of calibration constants ("reichardtchu2025" or "hao2011").
    wcs : astropy.wcs.WCS, optional
        WCS object for ΣSFR calculation.

    Returns
    -------
    sfr_map : 2D array
        Star formation rate per spaxel (Msol/yr).
    total_sfr : float
        Total star formation rate (Msol/yr).
    sfr_surface_density : 2D array or None
        SFR surface density map (Msol/yr/kpc^2), if WCS is provided.
    """

    # Pick calibration constants
    cal = CALIBRATIONS[calibration]
    C_Halpha = cal["C_Halpha"]
    Halpha_Hbeta_ratio = cal["Halpha_Hbeta_ratio"]

    # Luminosity distance in cm
    d_L = cosmo.luminosity_distance(z).to("cm").value

    # Convert flux to luminosity (per spaxel Hβ luminosity)
    L_Hbeta = 4 * np.pi * d_L**2 * F_Hbeta

    # Apply extinction correction + Hα/Hβ conversion
    L_Halpha_corr = Halpha_Hbeta_ratio * (10**(0.4 * A_Hbeta)) * L_Hbeta

    # Star formation rate per spaxel
    sfr_map = C_Halpha * L_Halpha_corr

    # Total SFR
    total_sfr = np.nansum(sfr_map)

    # Optional ΣSFR (requires WCS)
    sfr_surface_density = None
    if wcs is not None:
        # Pixel scale (arcsec/pixel → kpc/pixel at z)
        # drop wavelength axis
        spatial_wcs = cube.wcs.sub(['celestial'])
        pix_scale_arcsec = proj_plane_pixel_scales(spatial_wcs)[0] * 3600.0
        kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(z).value / 60.0
        pix_area_kpc2 = (pix_scale_arcsec * kpc_per_arcsec)**2

        sfr_surface_density = sfr_map / pix_area_kpc2

    return sfr_map, total_sfr, sfr_surface_density



if __name__ == "__main__": 

    galaxy_name = "CGCG453"
    file_path = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/cgcg453_red_mosaic.fits"
    z_obs = 0.025  # Original observed redshift
    z_sim = 3 # Simulated redshift
    final_output_path_ext_invert = f"/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/Output_cubes/{galaxy_name}/z_{z_sim}_final_pipeline_with_ext_{galaxy_name}_cube.fits"
    
    cube = Cube(final_output_path_ext_invert)

    lam = cube.wave.coord()
    print("Cube wavelength coverage:", lam.min(), "-", lam.max())

    lam_hbeta = 4861.0 * (1 + z_sim)
    print("Expected Hβ center:", lam_hbeta)

    hbeta_flux_map = measure_hbeta_flux(cube, z_sim, debug=True, debug_coords=(17,10))

    lam_hgamma = 4340.0 * (1 + z_sim)
    print("Expected Hγ center:", lam_hgamma)

    hgamma_flux_map = measure_hgamma_flux(cube, z_sim, debug=True, debug_coords=(24,12))
    

    print("\n--- Testing compute_Ahbeta() ---")

    # Case 1: ratio exactly intrinsic (2.15) → extinction should be 0 everywhere
    hbeta_test = np.ones((3, 3)) * 2.15
    hgamma_test = np.ones((3, 3)) * 1.0
    Ahbeta_test = compute_Ahbeta(hbeta_test, hgamma_test)
    print("Case 1 (intrinsic ratio):")
    print(Ahbeta_test)

    # Case 2: ratio larger than intrinsic (dust present) → extinction > 0
    hbeta_test = np.ones((3, 3)) * 4.3   # observed ratio = 4.3 / 1 = 4.3
    hgamma_test = np.ones((3, 3)) * 1.0
    Ahbeta_test = compute_Ahbeta(hbeta_test, hgamma_test)
    print("Case 2 (dusty, ratio > 2.15):")
    print(Ahbeta_test)

    # Case 3: ratio smaller than intrinsic (should floor at 0)
    hbeta_test = np.ones((3, 3)) * 1.0
    hgamma_test = np.ones((3, 3)) * 1.0
    Ahbeta_test = compute_Ahbeta(hbeta_test, hgamma_test)
    print("Case 3 (ratio < 2.15, floored):")
    print(Ahbeta_test)


    print("\n--- End-to-end SFR test  ---")

    # realistic example flux per spaxel (erg s^-1 cm^-2)
    # typical per-spaxel extragalactic emission-line fluxes are ~1e-18 - 1e-15
    hbeta_realistic = np.ones((3, 3)) * 1e-16  # example
    # example extinction map: modest extinction A_Hbeta ~ 1 mag
    Ahbeta_example = np.ones((3, 3)) * 1.0

    sfr_map_example = compute_sfr_from_hbeta_flux(
        hbeta_realistic,
        Ahbeta_example,
        C_Halpha=C_Halpha,
        F_Halpha_over_F_Hbeta=F_Halpha_over_F_Hbeta,
        z=z_sim
    )

    print("Inputs:")
    print("  hbeta flux (erg/s/cm^2):", hbeta_realistic[0, 0])
    print("  A_Hbeta (mag):", Ahbeta_example[0, 0])
    print("\nOutput SFR map (Msun/yr):")
    print(sfr_map_example)
    print("\nNote: use realistic flux values per spaxel — the result scales linearly with input flux.")
