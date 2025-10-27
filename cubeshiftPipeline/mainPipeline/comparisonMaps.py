
def test_MUSE_FOV():

    file_path = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/cgcg453_red_mosaic.fits"
    cube = Cube(file_path)

    # Step 1: Redshift wavelength axis
    z_old = 0.025
    telescope = telescope_specs["VLT_MUSE"]
    z_new = 1

    print(f"Redshifting from z = {z_old} to z = {z_new}")
    redshifted_cube, _ = redshift_wavelength_axis(cube, z_old, z_new)

    # Step 2: Reproject spatially to match target pixel scale
    # Get original WCS
    original_header = redshifted_cube.wcs.to_header()
    original_wcs_astropy = AstropyWCS(original_header)

    # Build 2D spatial WCS
    spatial_wcs = original_wcs_astropy.sub([1, 2])
    target_header = spatial_wcs.to_header()
    target_header['CDELT1'] = -telescope.pixel_scale_x / 3600.0
    target_header['CDELT2'] = telescope.pixel_scale_y / 3600.0

    # Set output shape same as input (you can adjust this later if needed)
    ny, nx = redshifted_cube.shape[1:]
    target_header['NAXIS1'] = nx
    target_header['NAXIS2'] = ny

    target_wcs = AstropyWCS(target_header)
    shape_out = (ny, nx)

    print("Reprojecting to match MUSE pixel scale")
    reprojected_cube = reproject_cube_preserve_wcs(redshifted_cube, target_wcs, shape_out)

    # Step 3: Crop to nonzero region (if needed)
    print("Cropping to region with signal...")
    cropped_cube = trim_empty_edges(reprojected_cube, buffer=2, debug=True)

    # Save output
    output_path = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/kcwi_to_muse_prelim.fits"
    cropped_cube.write(output_path)

    print(f" Saved redshifted, reprojected, cropped cube to:\n{output_path}")

def extract_centered_flux_map(cube,
    line_rest,
    z,
    broad_width=30,  # Angstroms for finding peak
    narrow_width=10  # Angstroms for final integration
    ):
    """
    Extracts a flux map centered on the brightest emission region of a spectral line.
    
    Parameters:
    - cube: MPDAF Cube object (assumed to be at redshift z)
    - line_rest: rest-frame wavelength (e.g. 5007 for [O III])
    - z: redshift of the cube
    - broad_width: width (Å) of initial search window
    - narrow_width: width (Å) for final flux map

    Returns:
    - flux_map: 2D numpy array centered on emission
    """
    lam_obs = line_rest * (1 + z)
    lam = cube.wave.coord()

    print(f"[DEBUG] Observed line center: {lam_obs:.2f} Å")
    print(f"[DEBUG] Cube wavelength range: {lam.min():.2f} - {lam.max():.2f} Å")


    # Step 1: Broad wavelength mask around line center
    broad_mask = (lam > lam_obs - broad_width/2) & (lam < lam_obs + broad_width/2)
    cube_broad = cube.copy().select_lambda(lam_obs - broad_width/2, lam_obs + broad_width/2)
    subcube_data = cube_broad.data

    print(f"[DEBUG] Broad mask wavelength range: {lam_obs - broad_width/2:.2f} to {lam_obs + broad_width/2:.2f}")
    print(f"[DEBUG] Number of spectral channels in broad mask: {np.sum(broad_mask)}")
    print(f"[DEBUG] Broad cube shape: {subcube_data.shape}")


    # Step 2: Collapse over spectral axis and find brightest (y,x)
    collapsed_broad = np.nansum(subcube_data, axis=0)
    max_pos = np.unravel_index(np.nanargmax(collapsed_broad), collapsed_broad.shape)
    y_peak, x_peak = max_pos

    print(f"[DEBUG] Peak position in broad mask: (y, x) = ({y_peak}, {x_peak})")
    print(f"[DEBUG] Peak flux value in broad collapsed cube: {collapsed_broad[y_peak, x_peak]:.3e}")


    # Step 3: Apply narrower spectral mask
    narrow_mask = (lam > lam_obs - narrow_width/2) & (lam < lam_obs + narrow_width/2)
    cube_narrow = cube.copy().select_lambda(lam_obs - narrow_width/2, lam_obs + narrow_width/2)

    print(f"[DEBUG] Narrow mask wavelength range: {lam_obs - narrow_width/2:.2f} to {lam_obs + narrow_width/2:.2f}")
    print(f"[DEBUG] Number of spectral channels in narrow mask: {np.sum(narrow_mask)}")
    print(f"[DEBUG] Narrow cube shape: {cube_narrow.data.shape}")


    # Step 4: Extract spectrum at bright location (optional debug)
    spectrum = cube_narrow[:, y_peak, x_peak]
    print(f"[DEBUG] Spectrum shape at peak position: {spectrum.shape}")
    print(f"[DEBUG] Spectrum at peak (first 5 channels): {spectrum[:5]}")


    # Step 5: Sum over spectral axis to create narrowband map
    flux_map = np.nansum(cube_narrow.data, axis=0)
    print(f"[DEBUG] Flux map shape: {flux_map.shape}")

    return flux_map

def extract_centered_flux_map_with_z(
    cube, line_rest, z_obs, z,
    width_narrow_rest=20, width_broad_rest=40,
    return_metadata=False, mask_flux=False, center=False, cutout_size=40
    ):
    """
    Extracts a flux map by integrating over a redshifted emission line using a
    difference-of-Gaussians approach.
    """

    print(f"\n[DEBUG] Extracting flux map for z = {z:.3f}...")

    line_obs = line_rest * (1 + z)
    width_narrow_obs = width_narrow_rest * (1 + z)
    width_broad_obs = width_broad_rest * (1 + z)

    print(f"[DEBUG] line_obs = {line_obs:.2f} Å")
    print(f"[DEBUG] narrow = {width_narrow_obs:.2f} Å, broad = {width_broad_obs:.2f} Å")

    lam = cube.wave.coord()
    print(f"[DEBUG] cube wavelength range: {lam[0]:.1f} - {lam[-1]:.1f} Å")

    # [MODIFIED] Step 1: Define a broader window ±20 Å to find max flux
    broad_search_mask = np.abs(lam - line_obs) <= 20
    cube_segment = cube.data[broad_search_mask, :, :]

    # [MODIFIED] Step 2: Collapse across spatial dimensions and find λ of max flux
    flux_spectrum = cube_segment.sum(axis=(1, 2))  # sum over x and y
    max_index = np.argmax(flux_spectrum)
    wavelength_max = lam[broad_search_mask][max_index]
    print(f"[DEBUG] wavelength_max (centered on flux peak): {wavelength_max:.2f} Å")

    # [MODIFIED] Step 3: Create new masks centered on wavelength_max
    mask_narrow = np.abs(lam - wavelength_max) <= (width_narrow_obs / 2)
    mask_broad = np.abs(lam - wavelength_max) <= (width_broad_obs / 2)

    print(f"[DEBUG] narrow mask sum: {mask_narrow.sum()} channels")
    print(f"[DEBUG] broad mask sum: {mask_broad.sum()} channels")

    lam_narrow = lam[mask_narrow]
    lam_broad = lam[mask_broad]
    print(lam_narrow)
    print(lam_broad)

    if mask_narrow.sum() == 0 or mask_broad.sum() == 0:
        print("⚠️ Warning: No channels found in mask. Returning empty flux map.")
        if return_metadata:
            return z, np.zeros(cube.shape[1:])
        else:
            return np.zeros(cube.shape[1:])

    flux_narrow = cube.data[mask_narrow, :, :].sum(axis=0)
    flux_broad = cube.data[mask_broad, :, :].sum(axis=0)

    print(f"[DEBUG] flux_narrow stats: min={np.nanmin(flux_narrow):.3g}, max={np.nanmax(flux_narrow):.3g}")
    print(f"[DEBUG] flux_broad stats: min={np.nanmin(flux_broad):.3g}, max={np.nanmax(flux_broad):.3g}")

    flux_map = 2 * flux_narrow - flux_broad

    print(f"[DEBUG] flux_map stats before masking: min={np.nanmin(flux_map):.3g}, max={np.nanmax(flux_map):.3g}")
    
    if center:
        y_peak, x_peak = np.unravel_index(np.nanargmax(flux_map), flux_map.shape)

        y_half = min(cutout_size // 2, y_peak, flux_map.shape[0] - y_peak - 1)
        x_half = min(cutout_size // 2, x_peak, flux_map.shape[1] - x_peak - 1)

        y_min = y_peak - y_half
        y_max = y_peak + y_half + 1
        x_min = x_peak - x_half
        x_max = x_peak + x_half + 1

        flux_map = flux_map[y_min:y_max, x_min:x_max]

    print(f"[DEBUG] Flux map shape after cutout: {flux_map.shape}")

    if mask_flux:
        flux_map[flux_map <= 0] = np.nan

    return (z, flux_map) if return_metadata else flux_map

def plot_flux_map_for_z(ax, flux_map, z, z_ref=0.025, logscale=False,
                        vmin=None, vmax=None, scale_factor=1.0):

    """
    Plot a flux map on a given axis.

    Parameters
    ----------
    ax : matplotlib axis
        Axis to plot on.
    flux_map : np.ndarray or MPDAF image
        2D flux map to plot.
    z : float
        Redshift of the cube (for title only).
    z_ref : float
        Reference redshift to which SB dimming is normalized.
    logscale : bool
        If True, show log10(flux) instead of linear flux.
    vmin, vmax : float
        Intensity range for colormap.
    """


    # --- Fix D: work on a copy to avoid modifying original ---
    flux = flux_map.copy()

    # --- Handle log scaling (early masking) ---
    if logscale:
        flux_map = np.where(flux_map > 1e-12, flux_map, np.nan)  # 1e-12 floor to avoid log(very tiny) = -inf

        with np.errstate(divide='ignore', invalid='ignore'):
            flux = np.log10(flux)
            flux[~np.isfinite(flux)] = np.nan  # sanitize bad values

    # --- Fix A & B: Compute vmin/vmax only on final flux array ---
    if vmin is None or vmax is None:
        finite_flux = flux[np.isfinite(flux)]
        if finite_flux.size == 0:
            vmin, vmax = (0.0, 1.0) if not logscale else (-16, -12)
        else:
            # Optional: Use slightly different stretch for the reference redshift
            if z == z_ref:
                vmin = np.nanpercentile(finite_flux, 5)
                vmax = np.nanpercentile(finite_flux, 99)
            else:
                vmin = np.nanpercentile(finite_flux, 5)
                vmax = np.nanpercentile(finite_flux, 99)

    # --- Fix C: Apply colormap safely ---
    im = ax.imshow(flux, origin="lower", cmap="plasma", vmin=vmin, vmax=vmax)

    # --- Dynamically adjust the colorbar label based on scale_factor ---
    try:
        scale_exponent = int(np.log10(scale_factor))
    except (ValueError, TypeError):
        scale_exponent = 0

    # Unit string adjusts 1e-16 base by scale_factor
    base_exponent = -16 + scale_exponent
    unit_prefix = rf"$10^{{{base_exponent}}}$"
    unit_label = rf"Flux [{unit_prefix} erg s$^{{-1}}$ cm$^{{-2}}$ $\AA^{{-1}}$]"
    log_label = r"log$_{10}$ " + unit_label

    # Add colorbar with adjusted label
    plt.colorbar(im, ax=ax, label=log_label if logscale else unit_label)
    ax.set_title(f"z = {z:.2f}")
    ax.axis('off')
    return im

def plot_rgb_map_for_z(ax, cube, z, z_obs, rest_waves, vmin=None, vmax=None):
    """Extract RGB images and plot on the given axis."""
    rgb_images = extract_rgb_images_from_cube(cube, z, rest_waves, width=20)
    plot_rgb_from_images(ax, rgb_images, vmin=vmin, vmax=vmax, z=z, z_obs=z_obs)
    ax.set_title(f"RGB z={z}")
    return rgb_images

def extract_rgb_images_from_cube(cube, z, rest_waves, width=20, variable_widths=False, widths=None):
    """
    Create MPDAF Images for R, G, B channels by slicing the cube around observed-frame wavelengths.
    """
    print(f"DEBUG: Incoming rest_waves = {rest_waves}")
    print(f"DEBUG: z = {z}")

    obs_waves = [(1 + z) * lam for lam in rest_waves]
    if variable_widths:
        obs_widths = [(1 + z) * w for w in widths]
    else:
        obs_widths = [(1 + z) * width for _ in rest_waves]

    images = []
    for i, (lam, dw) in enumerate(zip(obs_waves, obs_widths)):
        wave_min = cube.wave.coord()[0]
        wave_max = cube.wave.coord()[-1]

        # Print debug info only for blue channel (index 0)
        if i == 0:  # blue channel is the first in rest_waves list
            print(f"Blue channel extraction window for z={z}: {lam - dw/2:.1f} - {lam + dw/2:.1f} Å")
            print(f"Cube wavelength coverage: {wave_min:.1f} - {wave_max:.1f} Å")
            if lam - dw/2 < wave_min or lam + dw/2 > wave_max:
                print("Warning: Blue channel extraction range is outside cube wavelength range!")

        img = cube.get_image((lam - dw / 2, lam + dw / 2))
        images.append(img)
    print(f"1.25x DEBUG: rest_waves in: {rest_waves}")
    print(f"1.25x DEBUG: z={z}, applying obs_waves = rest_waves * (1+z)")

    return images  # [B, G, R]

def plot_rgb(images, vmin=None, vmax=None, ax=None, use_wcs=False, log_scale=False, **kwargs):
    """Generate an RGB image from 3 MPDAF Image objects and plot it."""
    if ax is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

    b, g, r = [np.nan_to_num(im.data, nan=0.0, posinf=0.0, neginf=0.0) for im in images]

    if log_scale:
        b = np.log10(b + 1e-5)
        g = np.log10(g + 1e-5)
        r = np.log10(r + 1e-5)

    def norm(data, vmin, vmax):
        return np.clip((data - vmin) / (vmax - vmin + 1e-10), 0, 1)

    b_img = norm(b, vmin[0], vmax[0])
    g_img = norm(g, vmin[1], vmax[1])
    r_img = norm(r, vmin[2], vmax[2])

    rgb = np.dstack((r_img, g_img, b_img))

    ax.imshow(rgb, origin='lower', **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])

    return ax, rgb

def plot_rgb_from_images(ax, rgb_images, vmin=None, vmax=None, z=None, z_obs=None, **kwargs):
    """
    Plot RGB image using given MPDAF Images (B, G, R order).
    You can specify vmin and vmax directly or let it compute from images.
    If z == z_obs, use custom vmin and vmax based on percentile.
    Otherwise, also use percentile-based scaling to brighten simulated cubes.
    """
    b, g, r = rgb_images

    if z is not None and z_obs is not None and z == z_obs:
        # Original image: percentile-based scaling
        vmin = (0.0, 0.0, 0.0)
        vmax = (
            np.percentile(b.data[~np.isnan(b.data)], 90.0),
            np.percentile(g.data[~np.isnan(g.data)], 90.0),
            np.percentile(r.data[~np.isnan(r.data)], 90.0),
        )
    else:
        # Simulated cubes: also use percentile-based scaling (otherwise sim images are all black)
        vmin = (0.0, 0.0, 0.0)
        vmax = (
            np.percentile(b.data[~np.isnan(b.data)], 99.0),
            np.percentile(g.data[~np.isnan(g.data)], 99.0),
            np.percentile(r.data[~np.isnan(r.data)], 99.0),
        )

    ax, _ = plot_rgb(
        rgb_images,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        use_wcs=True,
        **kwargs
    )

    return ax

def get_global_vmin_vmax(all_rgb_images, lower=0.5, upper=99.5):
    """Compute global vmin/vmax for each RGB channel across all images."""
    stacked = list(zip(*all_rgb_images))  # [(R1, R2, ...), (G1, G2, ...), (B1, B2, ...)]
    vmin = tuple(np.percentile(np.concatenate([im.data.ravel() for im in band]), lower) for band in stacked)
    vmax = tuple(np.percentile(np.concatenate([im.data.ravel() for im in band]), upper) for band in stacked)
    return vmin, vmax

def get_fluxmap_vmin_vmax(all_flux_maps, lower=0.5, upper=99.5):
    flux_data = [flux.ravel() for (_, flux) in all_flux_maps if flux is not None]
    flat_data = np.concatenate(flux_data)
    vmin = np.percentile(flat_data, lower)
    vmax = np.percentile(flat_data, upper)
    return vmin, vmax

def make_big_maps():

    z_obs = 0.025  # Observed redshift of the real galaxy
    all_rgb_images = []  # Will hold tuples: (R, G, B)

    # === Target redshifts for simulation ===
    redshifts = [z_obs, 2.5, 3.0, 4.0]
    telescope = "JWST_NIRSpec"  
    source_telescope=telescope_specs["Keck_KCWI"]
    target_telescope=telescope_specs["JWST_NIRSpec"]
    oiii_rest = 5007
    rest_waves=(4861, 5007, 4500) # Hβ 4861Å, [O III] 5007Å, Stellar Continuum ~4500Å

    # Set up flux map figure
    fig_flux, axes_flux = plt.subplots(2, 2, figsize=(12, 10))
    axes_flux = axes_flux.flatten()
    # Set up RGB map figure
    fig_rgb, axes_rgb = plt.subplots(2, 2, figsize=(12, 10))
    axes_rgb = axes_rgb.flatten()
    # Set up Hbeta/Hgamma ratio figure 
    fig_ratio, axes_ratio = plt.subplots(2, 2, figsize=(12, 10))
    axes_ratio = axes_ratio.flatten()

    # --- First Pass: extract image data only ---
    all_rgb_images = []
    all_flux_maps = []
    filter_info_per_z = {}
    simulated_cubes = {}

    # Store original and simulated flux maps separately for custom vmin/vmax
    original_flux_map = None
    simulated_flux_maps = []


    for z in redshifts:
        print(f"Calling extract_centered_flux_map_with_z with return_metadata=True at z = {z}")

        if z == z_obs:
            cube_z = cube
            trans_filter_name = "original"
            disp_filter_name = "original"

            cube_z_unscaled = cube_z.copy()
            simulated_cubes[z] = cube_z_unscaled  # Save clean, unmodified cube
            cube_z_vis = cube_z_unscaled  # No brightening for original


            cube_z_vis = cube_z_unscaled.copy()
            # cube_z_vis.data *= 1e5

            simulated_cubes[z] = {
                "unscaled": cube_z_unscaled,
                "vis": cube_z_vis
            }

        else:
            try:
                cube_z, trans_filter_name, disp_filter = simulate_observation(
                    cube, telescope, z_obs, z, source_telescope, target_telescope
                )
                disp_filter_name = disp_filter["name"] if isinstance(disp_filter, dict) else str(disp_filter)

                cube_z_unscaled = cube_z.copy()
                simulated_cubes[z] = cube_z_unscaled

                cube_z_vis = cube_z_unscaled.copy()
                # cube_z_vis.data *= 1e7


                simulated_cubes[z] = {
                    "unscaled": cube_z_unscaled,
                    "vis": cube_z_vis
                }

            except Exception as e:
                print(f"Skipping z={z} in first pass due to error: {e}")
                continue

        try:
            rgb_images = extract_rgb_images_from_cube(cube_z_vis, z, rest_waves, width=20)
            print(f"--- RGB Stats for z={z} ---")
            for band, name in zip(rgb_images, ['R', 'G', 'B']):
                print(f"{name} min={np.nanmin(band.data):.3e}, max={np.nanmax(band.data):.3e}")
            all_rgb_images.append(rgb_images)

            # Save filter names for second pass
            filter_info_per_z[z] = (trans_filter_name, disp_filter_name)

            # Use the unscaled (redshifted) version stored earlier
            cube_for_flux = simulated_cubes[z]["unscaled"].copy()
            if z != z_obs:
                cube_for_flux.data *= 1e2
            print(f"[DEBUG] Using cube with wavelength range: {cube_for_flux.wave.coord()[0]:.1f}–{cube_for_flux.wave.coord()[-1]:.1f} Å")

            z_flux, flux_map = extract_centered_flux_map_with_z(
                cube_for_flux, oiii_rest, z_obs, z, return_metadata=True
            )

            all_flux_maps.append((z, flux_map))  # still keep for convenient lookup

            # Save original and simulated separately for independent vmin/vmax
            if z == z_obs:
                original_flux_map = (z, flux_map)
            else:
                simulated_flux_maps.append((z, flux_map))


        except Exception as e:
            print(f"Skipping z={z} due to error: {e}")
            continue


    #  RGB vmin/vmax calculation
    vmin_shared, vmax_shared = get_global_vmin_vmax(all_rgb_images)
    # Flux map vmin/vmax calculation 

    vmin_flux_orig, vmax_flux_orig = get_fluxmap_vmin_vmax([original_flux_map])
    vmin_flux_sim, vmax_flux_sim = get_fluxmap_vmin_vmax(simulated_flux_maps)


    #print(f"[INFO] Global vmin_flux = {vmin_flux:.2e}, vmax_flux = {vmax_flux:.2e}")
    # --- Second Pass: plot maps using saved cubes ---
    for i, z in enumerate(redshifts):
        ax_flux = axes_flux[i]
        ax_rgb = axes_rgb[i]
        ax_ratio = axes_ratio[i]

        try:
            cube_z = simulated_cubes[z]["unscaled"]
            cube_z_vis = simulated_cubes[z]["vis"]
            trans_filter_name, disp_filter_name = filter_info_per_z.get(z, ("unknown", "unknown"))

            # --- Plot Hβ / Hγ ratio map ---
            print("====== [INFO] Hβ / Hγ ratio map ====== ")
            im = plot_hbeta_hgamma_ratio_amp_soft(cube_z, z, ax=ax_ratio, sn_threshold=0)

            # --- Get precomputed flux map ---
            flux_map = dict(all_flux_maps)[z]

            # --- Print diagnostics to understand why map appear yellow ---
            print(f"[{z}] Flux map stats: min={np.nanmin(flux_map):.3e}, max={np.nanmax(flux_map):.3e}")
            #print(f"[{z}] Using vmin_flux={vmin_flux:.3e}, vmax_flux={vmax_flux:.3e}")


            # --- Plot flux map using shared color scale ---
            print(f"[{z}] Flux map shape: {flux_map.shape}, min = {np.nanmin(flux_map)}, max = {np.nanmax(flux_map)}")
            # Choose vmin/vmax depending on whether it's the original or simulated
            if z == z_obs:
                vmin_plot, vmax_plot = vmin_flux_orig, vmax_flux_orig
            else:
                vmin_plot, vmax_plot = vmin_flux_sim, vmax_flux_sim

            print(f"[{z}] Plotting flux map with vmin={vmin_plot:.2e}, vmax={vmax_plot:.2e}")
            plot_flux_map_for_z(
                ax_flux, flux_map, z, logscale=False,
                vmin=vmin_plot, vmax=vmax_plot,
                scale_factor=1e2 if z != z_obs else 1.0
            )

            # --- Plot RGB map using same vmin/vmax across all panels ---

            plot_rgb_map_for_z(
                ax_rgb, cube_z_vis, z, z_obs, rest_waves,
                vmin=vmin_shared, vmax=vmax_shared
            )

        except Exception as e:
            print(f"[{z}] Error plotting: {e}")
            ax_flux.set_visible(False)
            ax_rgb.set_visible(False)
            ax_ratio.set_visible(False)


    # Final display steps

    fig_flux.suptitle("[O III] Flux Maps at Various Redshifts", fontsize=16)

    fig_flux.tight_layout()
    fig_flux.subplots_adjust(top=0.92)

    fig_rgb.suptitle("RGB Maps at Various Redshifts", fontsize=16)

    fig_rgb.tight_layout()
    fig_rgb.subplots_adjust(top=0.92, bottom=0.1)

    legend_elements = [
        Patch(facecolor='red', label='Hβ 4861 (Red)'),
        Patch(facecolor='green', label='[O III] 5007 (Green)'),
        Patch(facecolor='blue', label='Stellar continuum ~4500 Å (Blue)')
    ]

    # Add legend to the overall figure (fig_rgb), not to a specific subplot
    fig_rgb.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=3,
        fontsize=12,
        frameon=False,
        bbox_to_anchor=(0.5, 0.05)
    )
    fig_ratio.suptitle("Hβ / Hγ Line Ratio Maps at Various Redshifts", fontsize=16)
    fig_ratio.tight_layout()
    fig_ratio.subplots_adjust(top=0.92)

    # Optional: Add a shared colorbar
    cbar_ax = fig_ratio.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig_ratio.colorbar(im, cax=cbar_ax, label=r'H$\beta$ / H$\gamma$')


    plt.show()