import numpy as np

emission_line_limits = {
    "Hbeta_left": 4840.0,
    "Hbeta_right": 4880.0,
    "Hgamma_left": 4320.0,
    "Hgamma_right": 4360.0,
}

continuum_band_limits = {
    "Hbeta_left": 4900.0,
    "Hbeta_right": 4950.0,
    "Hgamma_left": 4600.0,
    "Hgamma_right": 4800.0,
}


def plot_hbeta_hgamma_ratio(cube, z=0.0, ax=None, hbeta_width=20, hgamma_width=20, sn_threshold=1.5):
    import numpy as np

    HBETA_REST = 4861.0
    HGAMMA_REST = 4341.0

    hbeta_obs = HBETA_REST * (1 + z)
    hgamma_obs = HGAMMA_REST * (1 + z)

    try:
        # Get flux maps around each line
        hbeta_map = cube.get_image(wave=(hbeta_obs - hbeta_width/2, hbeta_obs + hbeta_width/2))
        hgamma_map = cube.get_image(wave=(hgamma_obs - hgamma_width/2, hgamma_obs + hgamma_width/2))
        
        hgamma_subcube = cube.select_lambda(hgamma_obs - hgamma_width/2, hgamma_obs + hgamma_width/2)
        if hgamma_subcube.data.ndim == 2:
            hgamma_subcube.data = hgamma_subcube.data[np.newaxis, :, :]

        # Define continuum window for S/N estimation
        cont_width = 20
        cont_offset = -50
        cont_start = hgamma_obs - cont_offset - cont_width/2
        cont_end = hgamma_obs - cont_offset + cont_width/2

        wave_min = cube.wave.coord().min()
        wave_max = cube.wave.coord().max()

        # Try adjusting the continuum window if out of bounds
        if cont_start < wave_min or cont_end > wave_max:
            max_shift = 200
            step = 5
            for shift in range(0, max_shift + step, step):
                cont_start_new = cont_start + shift
                cont_end_new = cont_end + shift
                if cont_start_new >= wave_min and cont_end_new <= wave_max:
                    cont_start, cont_end = cont_start_new, cont_end_new
                    print(f"[WARNING] Continuum region adjusted by +{shift}Å to fit within cube.")
                    break
            else:
                print(f"[ERROR] Could not adjust continuum region at z={z}. Skipping.")
                return None

        # Extract continuum region
        continuum_subcube = cube.select_lambda(cont_start, cont_end)
        if continuum_subcube.data.ndim == 2:
            continuum_subcube.data = continuum_subcube.data[np.newaxis, :, :]

        # Mask all-NaN pixels in the continuum
        all_nan_mask = np.all(np.isnan(continuum_subcube.data), axis=0)
        continuum_subcube.data[:, all_nan_mask] = np.nan
        print("Pixels with fully-NaN continuum spectra:", np.sum(all_nan_mask))

        # Compute median & std for continuum S/N
        continuum_median = np.nanmedian(continuum_subcube.data, axis=0)
        continuum_std = np.nanstd(continuum_subcube.data, axis=0)
        bad_cont_mask = np.isnan(continuum_std) | (continuum_std == 0)

        print(" ============== S/N DEBUG ============")
        print("Continuum median shape:", continuum_median.shape)
        print("Num bad continuum pixels:", np.sum(bad_cont_mask))

        # Estimate S/N
        signal = np.nansum(hgamma_subcube.data - continuum_median[np.newaxis, :, :], axis=0)
        n_pix = hgamma_subcube.shape[0]
        snr = signal / (continuum_std * np.sqrt(n_pix))

        # Clean up S/N values
        snr = np.where(snr < 0, 0, snr)
        snr = np.clip(snr, 0, 30)

        print("Signal stats: min =", np.nanmin(signal), " max =", np.nanmax(signal))
        print("SNR stats: min =", np.nanmin(snr), " max =", np.nanmax(snr))
        print("SNR NaNs:", np.isnan(snr).sum())
        print(f"SNR > {sn_threshold} count: {(snr > sn_threshold).sum()} / {snr.size}")

        # -------------------- FLUX AND MASKING --------------------
        # Extract flux and masks
        hgamma_flux = hgamma_map.data
        hbeta_flux = hbeta_map.data

        # Default mask if not present
        if not hasattr(hbeta_map, "mask"):
            hbeta_mask = np.zeros_like(hbeta_flux, dtype=bool)
        else:
            hbeta_mask = hbeta_map.mask

        # Masks based on signal-to-noise and continuum quality
        low_snr_mask = (snr < sn_threshold) | np.isnan(snr) | np.isnan(hgamma_flux) | bad_cont_mask

        # Mask low flux values (soft threshold)
        small_flux_mask = hgamma_flux <= 1e-6  # soften from 1e-20 to 1e-6

        # Example flag to toggle S/N masking for debugging
        use_sn_mask = False  # set False to disable S/N masking temporarily

        # Start with base masks
        combined_mask = (
            hbeta_mask |
            np.isnan(hbeta_amp) | np.isnan(hgamma_amp) |
            (hgamma_amp <= 0) | (hbeta_amp <= 0)
        )

        # Add S/N and flux masks only if toggle is True
        if use_sn_mask:
            combined_mask |= low_snr_mask | small_flux_mask


        # Compute the ratio, applying mask to produce NaNs
        ratio_data = np.where(combined_mask, np.nan, hbeta_flux / hgamma_flux)

        # Optional: clip extreme ratio values to NaN
        ratio_data[(ratio_data < 0) | (ratio_data > 10)] = np.nan

        print(f"Total masked pixels: {np.sum(combined_mask)} / {combined_mask.size}")
        print("Valid ratio pixels:", np.sum(~np.isnan(ratio_data)))

        # Plotting the ratio map
        im = ax.imshow(ratio_data, origin='lower', cmap='plasma', vmin=0, vmax=4)
        ax.set_title(f"Hβ / Hγ at z={z}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        return im


    except Exception as e:
        print(f"Error plotting z={z}: {e}")
        return None



def plot_hbeta_hgamma_ratio_amp(cube, z=0.0, ax=None, sn_threshold=3, cont_subtract=True):
    # Add units and also mention how much it is scaled by!!!
    """
    Plot Hβ / Hγ line ratio using amplitude-based signal and S/N masking.

    Parameters
    ----------
    cube : mpdaf.obj.Cube
        The input MPDAF data cube.
    z : float
        Redshift of the galaxy.
    ax : matplotlib axis
        Axis to plot on.
    sn_threshold : float
        Minimum S/N threshold to include spaxel in the map.
    cont_subtract : bool
        Whether to subtract the continuum before finding line amplitudes.

    Returns
    -------
    im : matplotlib image
        The image returned by ax.imshow().
    """
    try:
        # Extract wavelength and data arrays
        lam = cube.wave.coord()  # 1D wavelength array
        data = cube.data.filled(np.nan)  # Convert to regular ndarray with NaNs

        # (Optional) Smooth/clip the cube here if needed

        # ========== Get amplitudes and S/N mask ==========
        print("λ range:", lam.min(), "-", lam.max())

        result = calc_hbeta_hgamma_amps(lam, data, z, cont_subtract=cont_subtract)
        if len(result) == 4:
            hbeta_amp, hgamma_amp, hbeta_hgamma_ratio, sn_mask = result
        else:
            raise ValueError("Expected four return values from calc_hbeta_hgamma_amps")

        # ========== Build mask ==========
        # Combine masks: low S/N, NaNs, or zero signal
        combined_mask = (
            ~sn_mask |
            np.isnan(hbeta_amp) | np.isnan(hgamma_amp) |
            (hgamma_amp <= 0)
        )

        # Apply combined mask to ratio
        ratio_data = np.where(combined_mask, np.nan, hbeta_amp / hgamma_amp)

        # ========== Plot ==========
        im = ax.imshow(ratio_data, origin='lower', cmap='plasma', vmin=0, vmax=4)
        ax.set_title(f"Hβ / Hγ at z={z}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        # ========== Debug info ==========
        print(f"Total spaxels: {hbeta_amp.size}")
        print(f"Masked spaxels: {np.sum(combined_mask)}")
        print(f"Unmasked spaxels: {np.sum(~combined_mask)}")
        print("Ratio stats: min =", np.nanmin(ratio_data), " max =", np.nanmax(ratio_data))
        print("Median Hβ/Hγ ratio:", np.nanmedian(ratio_data))

        return im

    except Exception as e:
        print(f"[ERROR] Could not compute Hβ / Hγ map: {e}")
        return None


def plot_hbeta_hgamma_ratio_amp_soft(cube, z=0.0, ax=None, sn_threshold=1.5, cont_subtract=True, apply_mask=True):
    """
    Plot Hβ / Hγ line ratio using amplitude-based signal and S/N masking.
    """

    try:
        # Extract wavelength and data arrays
        # just check signal to noise 
        lam = cube.wave.coord()
        data = cube.data.filled(np.nan)

        print("λ range:", lam.min(), "-", lam.max())

        result = calc_hbeta_hgamma_amps(lam, data, z, cont_subtract=cont_subtract)
        if len(result) == 4:
            hbeta_amp, hgamma_amp, hbeta_hgamma_ratio, sn_mask = result
        else:
            raise ValueError("Expected four return values from calc_hbeta_hgamma_amps")

        print("S/N mask: num True =", np.sum(sn_mask), "num False =", np.sum(~sn_mask))

        if apply_mask:
            combined_mask = (
                ~sn_mask 
                #np.isnan(hbeta_amp) | np.isnan(hgamma_amp) |
                #(hgamma_amp <= 1e-6)
            )
            ratio_data = np.where(combined_mask, np.nan, hbeta_amp / hgamma_amp)
        else:
            safe_denominator = hgamma_amp > 1e-6
            ratio_data = np.where(safe_denominator, hbeta_amp / hgamma_amp, np.nan)

        # Clip outliers
        ratio_data[(ratio_data < 0) | (ratio_data > 10)] = np.nan

        # Plot
        im = ax.imshow(ratio_data, origin='lower', cmap='plasma', vmin=0, vmax=4)
        ax.set_title(f"Hβ / Hγ at z={z}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        print(f"Total spaxels: {hbeta_amp.size}")
        if apply_mask:
            print(f"Masked spaxels: {np.sum(combined_mask)}")
            print(f"Unmasked spaxels: {np.sum(~combined_mask)}")
        else:
            print("Masking disabled.")
            print(f"Finite ratio pixels: {np.sum(np.isfinite(ratio_data))}")

        print("Ratio stats: min =", np.nanmin(ratio_data), " max =", np.nanmax(ratio_data))
        print("Median Hβ/Hγ ratio:", np.nanmedian(ratio_data))

        return im

    except Exception as e:
        print(f"[ERROR] Could not compute Hβ / Hγ map: {e}")
        return None



def calc_hbeta_hgamma_amps(lam, data, z, cont_subtract=True):
    """
    Calculate Hβ and Hγ amplitudes and their ratio from a data cube.

    Parameters
    ----------
    lam : ndarray
        1D wavelength array (same length as spectral axis).
    data : ndarray
        3D cube data array with shape (Nλ, Ny, Nx).
    z : float
        Redshift of the galaxy.
    cont_subtract : bool
        Whether to subtract a local median continuum before measuring amplitudes.

    Returns
    -------
    hbeta_amp : ndarray
        2D map of Hβ amplitude.
    hgamma_amp : ndarray
        2D map of Hγ amplitude.
    ratio : ndarray
        2D map of Hβ / Hγ ratio.
    sn_mask : ndarray
        2D boolean mask of spaxels passing the S/N cut.
    """
    # Rest-frame wavelengths
    hbeta_rest = 4861.0
    hgamma_rest = 4340.0

    # Observed wavelengths
    hbeta_obs = hbeta_rest * (1 + z)
    hgamma_obs = hgamma_rest * (1 + z)

    # Calculate pixel scale in wavelength (approx)
    pix_scale = np.median(np.diff(lam))


    # Use a minimum window of 3 pixels or 15 A whichever is larger
    win = max(15, 3 * pix_scale)
        
    # Define slice masks
    hbeta_slice = (lam >= hbeta_obs - win) & (lam <= hbeta_obs + win)
    hgamma_slice = (lam >= hgamma_obs - win) & (lam <= hgamma_obs + win)


    print(f"Hβ λ range: {lam[hbeta_slice].min()} - {lam[hbeta_slice].max()}")
    print(f"Hγ λ range: {lam[hgamma_slice].min()} - {lam[hgamma_slice].max()}")
    print(f"Hβ slice pixels: {np.sum(hbeta_slice)}")
    print(f"Hγ slice pixels: {np.sum(hgamma_slice)}")

    # Slice the cube for each line
    hbeta_spec = data[hbeta_slice, :, :]
    hgamma_spec = data[hgamma_slice, :, :]

    # -- DEBUG: Check if slices contain valid data
    if np.sum(~np.isnan(hbeta_spec)) == 0:
        print("[WARN] Hβ slice is empty or all NaNs!")
    if np.sum(~np.isnan(hgamma_spec)) == 0:
        print("[WARN] Hγ slice is empty or all NaNs!")

    # Subtract local continuum if requested
    if cont_subtract:
        # Only subtract continuum for columns that aren't fully NaN
        valid_cols_hbeta = ~np.all(np.isnan(hbeta_spec), axis=0)
        hbeta_spec[:, valid_cols_hbeta] -= np.nanmedian(hbeta_spec[:, valid_cols_hbeta], axis=0, keepdims=True)

        valid_cols_hgamma = ~np.all(np.isnan(hgamma_spec), axis=0)
        hgamma_spec[:, valid_cols_hgamma] -= np.nanmedian(hgamma_spec[:, valid_cols_hgamma], axis=0, keepdims=True)

    # Use nan-sum instead of max to avoid noise spikes
    hbeta_amp = np.nansum(hbeta_spec, axis=0)
    hgamma_amp = np.nansum(hgamma_spec, axis=0)

    # Clip negative amps to zero to avoid weird ratios
    hbeta_amp = np.where(hbeta_amp < 0, 0, hbeta_amp)
    hgamma_amp = np.where(hgamma_amp < 0, 0, hgamma_amp)

    # Estimate S/N
    sn_hbeta = hbeta_amp / (np.nanstd(hbeta_spec, axis=0) + 1e-10)
    sn_hgamma = hgamma_amp / (np.nanstd(hgamma_spec, axis=0) + 1e-10)

    # Spaxels must pass S/N on both lines
    sn_mask = (sn_hbeta > 3) & (sn_hgamma > 3)

    # Compute ratio safely (with np.where just to avoid division warnings)
    ratio = np.where(hgamma_amp > 0, hbeta_amp / hgamma_amp, np.nan)

    # More debug output
    print("hbeta_amp: min=", np.nanmin(hbeta_amp), " max=", np.nanmax(hbeta_amp), " mean=", np.nanmean(hbeta_amp))
    print("hgamma_amp: min=", np.nanmin(hgamma_amp), " max=", np.nanmax(hgamma_amp), " mean=", np.nanmean(hgamma_amp))
    print("ratio: min=", np.nanmin(ratio), " max=", np.nanmax(ratio), " mean=", np.nanmean(ratio))
    print("Median Hbeta/Hgamma:", np.nanmedian(ratio))
    print("Num hbeta zeros:", np.sum(hbeta_amp == 0))
    print("Num hgamma zeros:", np.sum(hgamma_amp == 0))
    print("Num NaNs in hbeta:", np.sum(np.isnan(hbeta_amp)))
    print("Num NaNs in hgamma:", np.sum(np.isnan(hgamma_amp)))

    return hbeta_amp, hgamma_amp, ratio, sn_mask


def hbeta_extinction_correction(lamdas, data, var, z, sn_cut=0):
    """
    Corrects for the extinction caused by light travelling through the dust and
    gas of the galaxy, as described in Cardelli et al. 1989. Uses Hbeta/Hgamma
    ratio as in Calzetti et al. 2001

    Parameters
    ----------
    lamdas : :obj:'~numpy.ndarray'
        wavelength vector

    data : :obj:'~numpy.ndarray'
        3D cube of data

    var : :obj:'~numpy.ndarray'
        3D cube of variance

    z : float
        redshift

    sn_cut : float
        the signal-to-noise ratio of the Hgamma line above which the extinction
        correction is calculated.  E.g. if sn_cut=3, then the extinction
        is only calculated for spaxels with Hgamma emission with S/N>=3.  For
        all other spaxels, Av = 0.  Default is 0

    Returns
    -------
    data : :obj:'~numpy.ndarray'
        the data corrected for extinction
    """
    #create the S/N array
    hgamma_mask = (lamdas>(4341.68*(1+z)-5)) & (lamdas<(4341.68*(1+z)+5))
    hgamma_cont_mask = (lamdas>(4600*(1+z))) & (lamdas<(4800*(1+z)))

    #hgamma_sn = np.trapz(data[hgamma_mask,:,:], lamdas[hgamma_mask], axis=0)/np.sqrt(np.sum(var[hgamma_mask,:,:]**2, axis=0))
    hgamma_sn = np.trapz(data[hgamma_mask,:,:], lamdas[hgamma_mask], axis=0)/np.nanstd(data[hgamma_cont_mask,:,:], axis=0)

    #use the hbeta/hgamma ratio to calculate EBV
    ebv = calculate_EBV_from_hbeta_hgamma_ratio(lamdas, data, z)

    #define the constant (using MW expected curve)
    Rv = 3.1

    #use that to calculate Av
    Av = ebv * Rv

    #replace the Av with 0 where the S/N is less than the sn_cut
    Av[hgamma_sn < sn_cut] = 0

    #convert lamdas from Angstroms into micrometers
    lamdas = lamdas/10000

    #define the equations from the paper
    y = lamdas**(-1) - 1.82
    a_x = 1.0 + 0.17699*y - 0.50447*(y**2) - 0.02427*(y**3) + 0.72085*(y**4) + 0.01979*(y**5) - 0.77530*(y**6) + 0.32999*(y**7)
    b_x = 1.41338*y + 2.28305*(y**2) + 1.07233*(y**3) - 5.38434*(y**4) - 0.62251*(y**5) + 5.30260*(y**6) - 2.09002*(y**7)

    #tile a_x and b_x so that they're the right array shape
    a_x = np.tile(a_x, [data.shape[2], data.shape[1], 1]).T
    b_x = np.tile(b_x, [data.shape[2], data.shape[1], 1]).T

    #print('median a_x:', np.nanmedian(a_x))
    #print('a_x shape:', a_x.shape)

    #print('median b_x:', np.nanmedian(b_x))
    #print('b_x shape:', b_x.shape)

    #find A(lambda)
    A_lam = (a_x + b_x/Rv)*Av

    #apply to the data
    data = (10**(0.4*A_lam))*data

    return Av, A_lam, data

def subtract_continuum(lamdas, spectrum, left_limit, right_limit):
    """
    Subtract a median value of a section of continuum to the blue of the emission
    line using the bounds given by left_limit and right_limit.

    Parameters
    ----------
    lamdas : :obj:'~numpy.ndarray'
        the wavelength vector, same length as the given spectrum

    spectrum : :obj:'~numpy.ndarray'
        the spectrum or array of spectra.  If in array, needs to be in shape
        [npix, nspec]

    left_limit : float
        The left-hand wavelength limit of the region to integrate over

    right_limit : float
        The right-hand wavelength limit of the region to integrate over

    plot : boolean
        if True, plots the area of the spectrum used for the integral.  Default
        is False.

    Returns
    -------
    spectrum : :obj:'~numpy.ndarray'
        the continuum subtracted continuum

    s_n : :obj:'~numpy.ndarray'
        an array the shape of the input array of spectra giving the signal to
        noise ratio in the continuum for each spectrum
    """
    #use the given limits to define the section of continuum
    cont = spectrum[(lamdas >= left_limit) & (lamdas <= right_limit),]

    #find the median of the continuum
    cont_median = np.nanmedian(cont, axis=0)

    #minus off the continuum value
    spectrum = spectrum - cont_median

    #find the standard deviation of the continuum section
    noise = np.std(cont, axis=0)

    #calculate the signal to noise ratio
    s_n = (cont_median/noise)

    return spectrum, s_n


def calculate_EBV_from_hbeta_hgamma_ratio(lamdas, data, z, cont_subtract=False):
    """
    Uses Hbeta/Hgamma ratio as in Calzetti et al. 2001

    Parameters
    ----------
    lamdas : :obj:'~numpy.ndarray'
        wavelength vector

    data : :obj:'~numpy.ndarray'
        3D cube of data

    z : float
        redshift

    cont_subtract : boolean
        whether to subtract the continuum from the data before calculating the
        Hbeta/Hgamma ratio.  Default is False, which means no continuum subtraction,
        assuming a continuum subtraction has already been applied.

    Returns
    -------
    data : :obj:'~numpy.ndarray'
        the data corrected for extinction
    """
    #calculate the hbeta/hgamma ratio
    ratio_stuff = calc_ext.calc_hbeta_hgamma_amps(lamdas, data, z, cont_subtract=cont_subtract)

    if len(ratio_stuff) > 3:
        hbeta_amp, hgamma_amp, hbeta_hgamma_obs, s_n_mask = ratio_stuff
    else:
        hbeta_amp, hgamma_amp, hbeta_hgamma_obs = ratio_stuff
    #hbeta_flux, hgamma_flux, hbeta_hgamma_obs = calc_ext.calc_hbeta_hgamma_integrals(lamdas, data, z, cont_subtract=False, plot=False)

    #set the expected hbeta/hgamma ratio
    hbeta_hgamma_actual = 2.15

    #set the expected differential extinction [k(hgamma)-k(hbeta)]=0.465
    diff_ext = 0.465

    #create an array for the ebv values
    ebv = np.full_like(hbeta_hgamma_obs, np.nan, dtype=np.double)

    #calculate E(B-V) if hbeta_hgamma_obs >= 2.15
    for (i,j), hbeta_hgamma_obs_value in np.ndenumerate(hbeta_hgamma_obs):
        if hbeta_hgamma_obs_value >= 2.15:
            #calculate ebv
            ebv[i,j] = (2.5*np.log10(hbeta_hgamma_obs_value/hbeta_hgamma_actual)) / diff_ext

        else:
            #set ebv to a small value
            ebv[i,j] = 0.01

    return ebv