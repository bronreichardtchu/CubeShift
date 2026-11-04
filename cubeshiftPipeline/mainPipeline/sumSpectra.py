from mpdaf.obj import Cube
import numpy as np
import matplotlib.pyplot as plt
from astropy.constants import c


# Load IFU cube (MPDAF automatically handles WCS)
cube = Cube("/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/Output_cubes/z_4.0_f070lp_g235h_lsf.fits")
masking_enabled = True  # Toggle galaxy masking on/off

# ===== OPTIONAL MASKING =====
if masking_enabled:
    # Make a narrow-band image to detect galaxy region
    narrow_band = cube.get_image((5000, 5020), unit_wave='angstrom')
    # Mask definition (adjust threshold)
    mask2d = narrow_band.data < 0  # shape: (ny, nx)
    # Expand to match cube shape (nwave, ny, nx)
    mask3d = np.broadcast_to(mask2d, cube.shape)
    # Apply mask
    cube.mask = cube.mask | mask3d  # combine with existing mask
else:
    print("Masking disabled: summing all spaxels...")

# Sum over spatial dimensions
integrated_spec = cube.sum(axis=(1, 2))

# Plot integrated spectrum
plt.figure()
plt.plot(cube.wave.coord(), integrated_spec.data)
plt.xlabel(f"Wavelength [{cube.wave.unit}]")
plt.ylabel(f"Flux [{cube.unit}]")
plt.title("Integrated Spectrum of Galaxy")
plt.show()

save_path = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/Summed_spectra/z4_cgcg453_summed_spectrum.fits"
# Save to FITS
integrated_spec.write(save_path)
print(f" Summed spectrum saved to: {save_path}")

# Now save to .txt file for ETC use

# Extract wavelength (Å) and flux (erg/s/cm²/Å)
wavelength_A = cube.wave.coord()
flux_erg = integrated_spec.data

# Convert Å → microns
wavelength_um = wavelength_A * 1e-4  # 1 Å = 1e-4 μm

# Convert erg/s/cm²/Å → mJy
# Step 1: Å → cm
wavelength_cm = wavelength_A * 1e-8
# Step 2: Fnu in erg/s/cm²/Hz
flux_per_Hz = flux_erg * (wavelength_cm**2) / c.cgs.value
# Step 3: erg/s/cm²/Hz → Jy
flux_Jy = flux_per_Hz / 1e-23
# Step 4: Jy → mJy
flux_mJy = flux_Jy * 1e3

# Stack into two-column array
spec_array = np.column_stack((wavelength_um, flux_mJy))

# ===== PLOT =====
plt.figure(figsize=(8, 5))
plt.plot(wavelength_um, flux_mJy, color="blue", lw=1)
plt.xlabel("Wavelength (μm)")
plt.ylabel("Flux (mJy)")
plt.title("Integrated Spectrum of Galaxy")
plt.grid(True)
plt.show()

# Save as plain text (no headers, no comments)
text_file = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/Summed_spectra/z4_spectrum_cgcg453.txt"
np.savetxt(text_file, spec_array, fmt='%.6e')
print(f" Summed spectrum saved as text file: {text_file}")

