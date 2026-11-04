"""Sum a cube over its spatial axes, plot the integrated spectrum and save it as
FITS and as a two-column (micron, mJy) text file for the JWST ETC.

Example
-------
python scripts/sum_spectra.py Output_cubes/z_4.0_f070lp_g235h_lsf.fits \
    --fits-out z4_cgcg453_summed_spectrum.fits --txt-out z4_spectrum_cgcg453.txt
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from astropy.constants import c
from mpdaf.obj import Cube


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cube", help="Input cube (FITS)")
    parser.add_argument("--fits-out", default=None, help="Path to save the summed spectrum (FITS)")
    parser.add_argument("--txt-out", default=None, help="Path to save the (micron, mJy) text spectrum")
    parser.add_argument("--no-mask", action="store_true", help="Disable galaxy masking (sum all spaxels)")
    parser.add_argument("--mask-wave", type=float, nargs=2, default=[5000.0, 5020.0],
                        metavar=("LMIN", "LMAX"),
                        help="Wavelength range (Å) of the narrow-band image used for masking")
    parser.add_argument("--no-show", action="store_true", help="Do not display plots")
    args = parser.parse_args(argv)

    # Load IFU cube (MPDAF automatically handles WCS)
    cube = Cube(args.cube)
    masking_enabled = not args.no_mask  # Toggle galaxy masking on/off

    # ===== OPTIONAL MASKING =====
    if masking_enabled:
        # Make a narrow-band image to detect galaxy region
        narrow_band = cube.get_image(tuple(args.mask_wave), unit_wave='angstrom')
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
    if not args.no_show:
        plt.show()

    # Save to FITS
    if args.fits_out is not None:
        integrated_spec.write(args.fits_out)
        print(f" Summed spectrum saved to: {args.fits_out}")

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
    if not args.no_show:
        plt.show()

    # Save as plain text (no headers, no comments)
    if args.txt_out is not None:
        np.savetxt(args.txt_out, spec_array, fmt='%.6e')
        print(f" Summed spectrum saved as text file: {args.txt_out}")


if __name__ == "__main__":
    main()
