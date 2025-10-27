import matplotlib.pyplot as plt
import numpy as np
from mpdaf.obj import Cube
from astropy import units as u
from scipy.ndimage import gaussian_filter1d

# === Load original KCWI cube ===

file_path = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/cgcg453_red_mosaic.fits"
cube = Cube(file_path) 

z_obs = 0.025  # Observed redshift of the real galaxy

# === Target redshifts for simulation ===
redshifts = [z_obs, 2.5, 3.0, 4.0]
telescope = 'jwst_nircam'  # <<< Change if needed

# === [OIII] 5007 in rest-frame ===
oiii_rest = 5007  # Angstrom

# === Image extraction function with redshifted width & peak centering ===
def extract_centered_oiii_image(cube, z, rest_wavelength=5007, rest_width=20):
    # Convert to observed frame
    lam_center = rest_wavelength * (1 + z)
    width = rest_width * (1 + z)

    # Collapse cube spatially to get 1D spectrum
    spectrum = cube.sum(axis=(1, 2))  # Wavelength axis is 0
    wave = cube.wave.coord()  # Wavelength array in Angstrom

    # Create mask around expected line center
    window = 100 * (1 + z)
    mask = (wave > lam_center - window) & (wave < lam_center + window)
    if np.sum(mask) < 5:
        raise ValueError(f"[O III] line likely outside cube for z = {z}")

    # Smooth the spectrum and locate line peak
    smooth_spec = gaussian_filter1d(spectrum.data[mask], sigma=3)
    lam_window = wave[mask]
    peak_idx = np.argmax(smooth_spec)
    lam_peak = lam_window[peak_idx]

    # Extract flux image centered on peak wavelength
    img = cube.get_image((lam_peak, width))
    return img

# === Create 2x2 plot ===
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, z in enumerate(redshifts):
    ax = axes[i]
    
    if z == z_obs:
        cube_z = kcwi_cube
    else:
        cube_z, _ = simulate_observation(kcwi_cube, telescope, z_obs, z)

    try:
        img = extract_centered_oiii_image(cube_z, z)
        im = ax.imshow(img.data, origin='lower', cmap='viridis',
                       vmin=0, vmax=np.nanpercentile(img.data, 99))
        ax.set_title(f'[O III] flux map at z = {z}')
    except Exception as e:
        ax.text(0.5, 0.5, f'Error at z={z}\n{str(e)}',
                ha='center', va='center', fontsize=10)
        im = None
    
    ax.axis('off')

# Add a single colorbar if at least one image worked
if im is not None:
    fig.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.02)

plt.tight_layout()
plt.show()

