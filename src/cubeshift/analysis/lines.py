"""Emission-line flux measurement and Balmer-decrement extinction maps."""
import numpy as np
import matplotlib.pyplot as plt


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
    """
    Measure a continuum-subtracted emission-line flux map from a cube.

    For each spaxel the line peak is located within ``search_window`` of the
    redshifted line centre, a quadratic continuum is fitted to the blue/red
    bands either side of the peak, and the continuum-subtracted flux is
    integrated over ~1.5 spectral pixels either side of the peak.

    Parameters
    ----------
    cube : mpdaf.obj.Cube
        The IFU cube.
    z : float
        Source redshift.
    w_line : float
        Rest-frame wavelength of the line (Å).
    line_window : tuple of float
        Legacy parameter; currently unused (the integration band is set from
        the spectral pixel scale).
    line_name : str
        Name used in debug output. Names starting with "hgamma"/"hγ" use a
        wider (±100 Å) peak search window.
    search_window : tuple of float
        Offsets (Å) from the observed line centre in which to search for the peak.
    cont_blue, cont_red : tuple of float
        Offsets (Å) from the peak defining the continuum bands (only
        ``cont_blue[0]`` and ``cont_red[1]`` are used; the bands extend to the
        line band edges).
    debug : bool
        If True, print diagnostics and show plots for ``debug_coords``.
    debug_coords : tuple of int
        (j, i) spaxel to debug.

    Returns
    -------
    flux_map : 2D numpy.ndarray
        Integrated line flux per spaxel (NaN where not measured).
    """
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
            peak_idx = np.argmax(flux[mask_search])
            lam_peak = lam[mask_search][peak_idx]

            # --- Step 2: define line band centered on lam_peak ---
            # (line_window is not used: the band is ~1.5 spectral pixels either side of the peak)
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

            # Hγ band/mask diagnostics. This used to sit before Step 1, where it
            # referenced lam_peak/band_*/mask_* before they were defined (stale
            # values from the previous spaxel, or a NameError), so it now runs
            # here once those are computed for this spaxel.
            if debug and line_name == "hGamma" and (j, i) == debug_coords:
                print(f"[DEBUG {line_name}] lam_center={lam_center:.1f}, lam_peak={lam_peak:.1f}")
                print(f"[DEBUG {line_name}] band_line={band_line}")
                print(f"[DEBUG {line_name}] band_cont1={band_cont1}, band_cont2={band_cont2}")
                print(f"[DEBUG {line_name}] mask_search points={np.sum(mask_search)}, "
                    f"mask_cont points={np.sum(mask_cont)}, mask_line points={np.sum(mask_line)}")

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
    """
    Print summary statistics of a flux map and show it with a 5-95 percentile stretch.

    Parameters
    ----------
    flux_map : 2D numpy.ndarray
        Flux map (e.g. from :func:`measure_hbeta_flux`).
    """

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
    Wrapper for measure_flux to measure the Hγ line (4340 Å).

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
    Wrapper for measure_flux to measure the Hβ line (4861 Å).

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
        Map of integrated Hβ fluxes.
    """

    return measure_flux(
        cube, z,
        w_line=4861.0,            # Hγ rest wavelength
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
