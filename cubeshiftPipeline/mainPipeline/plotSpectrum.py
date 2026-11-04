import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpdaf.obj import Cube 
from main import simulate_observation, get_global_vmin_vmax
from matplotlib import gridspec
import matplotlib.patches as mpatches

from applyFilter import NIRSPEC_THROUGHPUT_RANGES
from matchSpectral import NIRSPEC_DISPERSER_RANGES
from main import (
    plot_rgb_from_images,
    plot_rgb_map_for_z,
    plot_rgb,
    extract_rgb_images_from_cube  
)
# Rest-frame line centers (in Angstroms)
# Blue = continuum, Green = OIII, Red = HBeta
CONT_REST = 5500.0
OIII_REST = 5007.0
HBETA_REST = 4861.0
band_names = ['continuum', 'oiii', 'hb']


WIDTH = 20.0  # +/- width for highlighting (in Å or µm depending on redshift)
WIDTHS = {
    "continuum": 50,  # wider continuum band
    "oiii": 20,
    "hb": 20,
}
widths = [WIDTHS[name] for name in band_names]

# Telescope class:

class Telescope:
    def __init__(self, name, spatial_fwhm, pixel_scale_x, pixel_scale_y, spectral_resolution,
                 spectral_sampling=None):
        self.name = name
        self.spatial_fwhm = spatial_fwhm  # arcsec
        self.pixel_scale_x = pixel_scale_x    # arcsec/pixel
        self.pixel_scale_y = pixel_scale_y    # arcsec/pixel
        self.spectral_resolution = spectral_resolution  # R = λ/Δλ
        self.spectral_sampling = spectral_sampling  # Δλ in Å 


telescope_specs = {
    "JWST_NIRCam": Telescope(
        name="JWST NIRCam",
        pixel_scale_x=0.031,  # arcsec/pixel
        pixel_scale_y=0.063,  # arcsec/pixel
        spatial_fwhm=0.07,    # arcsec -- simulating around F200W filter 
        spectral_resolution=1000 # resolving power!
    ),
        "VLT_MUSE": Telescope(
        name="VLT MUSE",
        pixel_scale_x=0.2,     # arcsec/pixel
        pixel_scale_y=0.2,
        spatial_fwhm=0.6,      # arcsec (typical seeing-limited PSF in WFM)
        spectral_resolution=3000,
        spectral_sampling=1.25  # Å (typical for MUSE WFM)
    ),
        "JWST_NIRSpec": Telescope(
        name="JWST NIRSpec",
        pixel_scale_x=0.1,     # arcsec/pixel (microshutter projected size)
        pixel_scale_y=0.1,
        spatial_fwhm=0.07,     # arcsec — diffraction-limited like NIRCam
        spectral_resolution=1000,  # for medium-resolution gratings (R~1000–2700)
        spectral_sampling=2.0  # Å — approximate; varies with configuration
    ),
    "Keck_KCWI": Telescope(
        name="Keck KCWI",
        pixel_scale_x=0.29,     # arcsec/pixel (medium slicer)
        pixel_scale_y=0.29,
        spatial_fwhm=1.0,       # arcsec (seeing-limited, typical value)
        spectral_resolution=4000,
        spectral_sampling=0.5   # Å (depends on grating; ~0.5 Å for medium)
    ),
}
def find_line_center_near(cube, rest_line, z, window=20):
    """
    Find the peak wavelength near a redshifted emission line in the integrated spectrum of a cube.

    Parameters:
        cube: Spectral cube object with .wave and .data attributes.
        rest_line (float): Rest-frame line center in Å (e.g. 5007).
        z (float): Redshift.
        window (float): Half-width of the search window in Å.

    Returns:
        float: Observed wavelength of the peak flux near the line.
    """
    # Step 1: extract wavelength and flux from cube
    wave, flux = extract_integrated_spectrum(cube)
    flux /= np.nanmax(flux)  # or nansum, or a windowed mean
    #if z == z_obs:
        # wave /= 1.25



    # Step 2: calculate expected line position
    obs_line = rest_line * (1 + z)

    # Step 3: build search window
    idx = (wave > obs_line - window) & (wave < obs_line + window)
    if not np.any(idx):
        return np.nan

    # Step 4: find max in window
    flux_window = flux[idx]
    wave_window = wave[idx]
    peak_idx = np.nanargmax(flux_window)

    return wave_window[peak_idx]


def extract_integrated_spectrum(cube):
    """
    Extract the total spectrum by summing over all spatial pixels.
    """
    # Collapse cube spatially to get total spectrum
    collapsed = cube.sum(axis=(1, 2))
    wave = collapsed.wave.coord()
    print(f"DEBUG PT2: Wavelength axis min={wave[0]:.2f} max={wave[-1]:.2f} step={wave[1]-wave[0]:.2f}")
    return collapsed.wave.coord(), collapsed.data  # (wavelengths, fluxes)

def get_safe_continuum_center(z, cube, width=50.0):
    """
    Choose a valid continuum center wavelength (observed frame) fully inside the cube wavelength range.
    Checks a prioritized list of rest-frame wavelengths and returns the first that fits.
    The width parameter is half-bandwidth in rest-frame Ångstroms.

    Parameters:
        z : float
            Redshift
        cube : MPDAF Cube object
            Cube with .wave.coord() providing wavelength array in observed frame Å
        width : float, optional (default=50.0)
            Half-width of the continuum band in rest-frame Ångstroms.

    Returns:
        float
            Observed-frame continuum center wavelength (Å) guaranteed to fit inside the cube's wavelength range.
    """
    print(" Getting safe continuum center ...")
    
    # Cube wavelength range in observed frame
    wave_min = cube.wave.coord()[0]
    wave_max = cube.wave.coord()[-1]

    # Prioritized list of rest-frame candidate continuum centers (Å)
    candidate_rest = [5300, 4600, 6200, 4100]

    # Scale rest-frame width to observed frame width
    obs_width = width * (1 + z)

    for rest_wavelength in candidate_rest:
        obs_center = rest_wavelength * (1 + z)
        band_min = obs_center - obs_width
        band_max = obs_center + obs_width
        # Check if entire band fits inside cube wavelength range
        if band_min >= wave_min and band_max <= wave_max:
            print(f"[z={z:.2f}] Using continuum rest λ = {rest_wavelength} Å → obs λ = {obs_center:.1f} Å")
            print(f"Band window: {band_min:.1f} - {band_max:.1f} Å inside cube {wave_min:.1f} - {wave_max:.1f} Å")
            return obs_center

    # If none fit, fallback to center of cube (observed frame)
    fallback = (wave_min + wave_max) / 2
    print(f"[z={z:.2f}] No valid continuum band found — using fallback λ = {fallback:.1f} Å")
    return fallback



def plot_multiple_spectra_with_bands(cube_list, redshifts, z_obs, filter_info=None, display_filter_bands=False):
    fig = plt.figure(figsize=(12, 10))
    outer_grid = plt.GridSpec(2, 2, figure=fig, wspace=0.3, hspace=0.3)


    # --- Step 1: Precompute all RGB images for consistent scaling ---
    all_rgb_images = []
    for cube, z in zip(cube_list, redshifts):
        obs_width_rest = WIDTHS["continuum"]  # e.g. 50 Å rest-frame half-width
        cont_obs = get_safe_continuum_center(z, cube, width=obs_width_rest) # This returns an observed wavelength
        cont_rest = cont_obs / (1 + z)                  # Convert back to rest-frame
        rest_waves = [cont_rest, OIII_REST, HBETA_REST]  # B, G, R order with safe continuum
        rgb_images = extract_rgb_images_from_cube(cube, z, rest_waves, width=20, variable_widths=True, widths=widths)
        all_rgb_images.append(rgb_images)

    vmin, vmax = get_global_vmin_vmax(all_rgb_images, lower=0.5, upper=99.5)
    spec_axes = []  # store all spectrum axes

    # --- Step 2: Loop through each redshift ---
    for i, (cube, z) in enumerate(zip(cube_list, redshifts)):
        # Inner grid for RGB (top) + spectrum (bottom)
        inner_grid = gridspec.GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec=outer_grid[i],
            height_ratios=[5, 1],
            hspace=0.05
        )

        ax_rgb = fig.add_subplot(inner_grid[0])
        ax_spec = fig.add_subplot(inner_grid[1])

        spec_axes.append(ax_spec)  # save for resizing later


        # --- RGB map ---
        rgb_images = all_rgb_images[i]


        # Make a deep copy of the original HDUs, but ensure .data is writable
        rgb_images_safe = []
        for hdu in all_rgb_images[i]:
            hdu_copy = hdu.copy()
            hdu_copy.data = np.array(hdu.data, copy=True)
            rgb_images_safe.append(hdu_copy)

        # Print basic stats for each channel
        for color, img in zip(['Blue', 'Green', 'Red'], rgb_images_safe):
            data = np.nan_to_num(img.data, nan=0.0, posinf=0.0, neginf=0.0)
            print(f"{color} channel stats: min={np.min(data):.3e}, max={np.max(data):.3e}, mean={np.mean(data):.3e}")

        plot_rgb_from_images(ax_rgb, rgb_images_safe, vmin=vmin, vmax=vmax, z=z, z_obs=z_obs)


        # Prepare title with filters
        trans_name, disp_name = filter_info.get(z, (None, None))
        title_str = f"RGB z={z}"
        if trans_name or disp_name:
            trans_str = f"Trans: {trans_name}" if trans_name else ""
            disp_str = f"Disp: {disp_name}" if disp_name else ""
            filter_str = " | ".join(s for s in [trans_str, disp_str] if s)
            if filter_str:
                title_str += f"\n{filter_str}"


        ax_rgb.set_title(title_str, pad=-15)

        # --- Spectrum ---
        wave, flux = extract_integrated_spectrum(cube)
        flux /= np.nanmax(flux)

        #if z == z_obs:
            # scale wavelengths by 1/1.25 (compress)
            # wave = wave / 1.25
            # wave = wave - delta_lambda  # to translate by delta in Å: 

        if z > 0.5:
            wave_plot = wave / 1e4
            unit = "Wavelength (μm)"
            scale = 1e4
        else:
            wave_plot = wave
            unit = "Wavelength (Å)"
            scale = 1

        ax_spec.plot(wave_plot, flux, color='k', lw=1.2)

        # Highlight lines
        hbeta_center = find_line_center_near(cube, HBETA_REST, z=z)
        oiii_center = find_line_center_near(cube, OIII_REST, z=z)
        cont_center = get_safe_continuum_center(z, cube)

        band_widths_scaled = {k: WIDTHS[k] * (1 + z) / scale for k in WIDTHS}

        ax_spec.axvspan(hbeta_center/scale - band_widths_scaled["hb"], hbeta_center/scale + band_widths_scaled["hb"],
                        color='red', alpha=0.3)
        ax_spec.axvspan(oiii_center/scale - band_widths_scaled["oiii"], oiii_center/scale + band_widths_scaled["oiii"],
                        color='green', alpha=0.3)
        ax_spec.axvspan(cont_center/scale - band_widths_scaled["continuum"], cont_center/scale + band_widths_scaled["continuum"],
                        color='blue', alpha=0.3)

        
        # make the disperser ranges usable
        DISPERSER_DICT = {item['name'].lower(): (item['lambda_min'], item['lambda_max']) for item in NIRSPEC_DISPERSER_RANGES}

        # Optional: filter_info shading
        if filter_info is not None:
            if display_filter_bands: 
                if filter_info and z in filter_info:
                    trans_name, disp_name = filter_info[z]
                    
                    # Plot transmission filter 
                    if trans_name and trans_name.lower() in NIRSPEC_THROUGHPUT_RANGES:
                        t_min, t_max = NIRSPEC_THROUGHPUT_RANGES[trans_name.lower()]
                        ax_spec.axvspan(t_min, t_max, color='cyan', alpha=0.1, label=f"{trans_name.upper()} (trans)")

                    # Plot disperser filter 
                    if disp_name and disp_name.lower() in DISPERSER_DICT:
                        d_min, d_max = DISPERSER_DICT[disp_name.lower()]
                        ax_spec.axvspan(d_min, d_max, color='magenta', alpha=0.1, label=f"{disp_name.upper()} (disp)")

        ax_spec.set_ylabel("Flux")
        ax_spec.grid(True)
        ax_spec.set_xlabel(unit)

    # Define colored patches corresponding to your RGB emission lines
    red_patch = mpatches.Patch(color='red', label='Hβ emission')
    green_patch = mpatches.Patch(color='green', label='[O III] emission')
    blue_patch = mpatches.Patch(color='blue', label='Stellar continuum')

    # Add legend below all subplots, centered
    fig.legend(handles=[red_patch, green_patch, blue_patch], 
            loc='lower center', 
            bbox_to_anchor=(0.5, 0.02), 
            ncol=3, 
            fontsize='medium', 
            frameon=False)
    fig.subplots_adjust(top=0.93)  # smaller value moves whole plot down, bigger moves up


    plt.tight_layout()


    # Now shrink and center all spectra
    for ax in spec_axes:
        pos = ax.get_position()
        new_width = pos.width * 0.7  # adjust % as needed
        new_x0 = pos.x0 + (pos.width - new_width) / 2
        ax.set_position([new_x0, pos.y0, new_width, pos.height])


    plt.show()



if __name__ == "__main__":

    # === Load original KCWI cube ===

    file_0_025 = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/cgcg453_red_mosaic_fresh.fits"
    file_2_5 = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/Output_cubes/z_2.5_f070lp_prism_lsf.fits"
    file_3 = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/Output_cubes/z_3.0_f070lp_g235h_lsf.fits"
    file_4 = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/Output_cubes/z_4.0_f070lp_g235h_lsf.fits"
    cube = Cube(file_0_025) 
    cube_2_5 = Cube(file_2_5)
    cube_3 = Cube(file_3)
    cube_4 = Cube(file_4)

    z_obs = 0.025  # Observed redshift of the real galaxy
    redshifts = [z_obs, 3.0, 4.0, 5.0]
    #cubes = [cube, cube_2_5, cube_3, cube_4]
    telescope = "JWST_NIRSpec"  
    source_telescope=telescope_specs["Keck_KCWI"]
    target_telescope=telescope_specs["JWST_NIRSpec"]


    filter_info = {}
    cubes = []

    for z in redshifts:
        if z == z_obs:
            cube_z = cube
            trans_filter_name = None
            disp_filter_name = None
            
        else:
            cube_z, trans_filter_name, disp_filter = simulate_observation(cube, telescope, z_obs, z, source_telescope, target_telescope)
            disp_filter_name = disp_filter["name"] if isinstance(disp_filter, dict) else str(disp_filter)
        
        cubes.append(cube_z)
        filter_info[z] = (trans_filter_name, disp_filter_name)
        print(f"[FILTER INFO] z={z}: trans={trans_filter_name}, disp={disp_filter_name}")


    plot_multiple_spectra_with_bands(cubes, redshifts, z_obs, filter_info=filter_info)


