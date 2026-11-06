# CubeShift roadmap

These are known issues and planned improvements that are not yet done. The numbers refer to the
code review (`cubeshift_review.md`); items 1–7, 9 and 13 have been fixed (PRs #7 and #8).

## Science

### #8: Noise model (new functionality)
The pipeline dims the signal *and* the original noise together, by ~5×10⁻⁶ at z=3 for the KCWI test
case. The output is therefore effectively noiseless at the original S/N or better. The noise added
when padding to the target FOV is drawn from the dimmed cube, not from the target instrument.

**Proposed:**
- Add the target instrument's noise for a given exposure time *after* all convolutions: sky/zodi,
  thermal, read noise, dark current, throughput.
- Options are Pandeia (the JWST/HST ETC engine) or a simple analytic model with per-instrument
  parameters in `Telescope`.
- Set `var` consistently with the noise added.

This needs a decision on which model to use, and the exposure-time inputs.

### #10: Filter throughput and filter choice (NIRSpec)
- **Throughput applied to calibrated flux.** `throughput.apply_nirspec_throughput` multiplies the cube
  by the filter throughput, but JWST cubes are flux-calibrated. Throughput should only enter the noise
  calculation (#8) and set the usable band limits.
- **Wrong filter at z=1–1.5.** The filter is chosen as the narrowest range containing [OIII]5007. At
  z = 1.0 and 1.5 this picks **f110w**, a target-acquisition filter that is not used with the IFU
  gratings.
- **Filter and disperser chosen independently.** This can give invalid grating/filter pairs.
- **Fixed line.** The hard-coded 5007 Å line drives the whole band choice.

**Proposed:**
- Choose valid grating+filter pairs together, from a table of allowed NIRSpec IFU configurations.
- Select by the cube's redshifted wavelength range, not by one line.
- Drop the throughput multiplication from the flux path.

### #11: Disperser choice
- `spectral.select_best_disperser_with_partial` returns the **first** disperser with >50% coverage,
  not the best one.
- The G140M/G140H ranges stop at 1.27 µm, which is only true with F070LP; with F100LP they extend to
  ~1.89 µm.

**Proposed:** rank all valid configurations by coverage, then by resolution; fold this into #10.

### #12: Integer binning → flux-conserving resampling
- Integer bin factors make the effective spaxel deviate from the target pixel by −8% to +13% over
  z = 0.5–7.
- Edge pixels that don't fill a bin are discarded.

**Proposed:** resample onto the exact target grid at z_sim with a flux-conserving drizzle or
`reproject.reproject_exact`. Propagate the variance, and keep the WCS at the target pixel scale.

**Partly covered bins (decision needed).** A bin with some masked inputs holds the *mean* surface
brightness of its unmasked pixels × the whole bin area, which extrapolates into the masked part.
For the NGC 300 ACS `_drc` (7.4% of the frame uncovered, outside the rotated footprint), this adds
1.5–4.8% to the total counts, depending on z_sim. Options:
- add a `min_coverage` threshold that masks bins with less coverage;
- treat uncovered pixels as zero;
- return a coverage map alongside the output.

### Other science items
- **Air vs vacuum.** KCWI wavelengths are in air and NIRSpec's in vacuum, but no conversion is
  applied. `redshift.air_to_vac` exists but isn't used. It also implements Greisen et al. (2006)
  eq. 65, which gives Hα 6564.71 Å against the IAU-standard 6564.61 Å (~5 km/s), so check the formula
  before using it.
- **NIRSpec PSF is idealised.** It is the diffraction limit (1.025 λ/D, `Telescope.psf_fwhm`), which is
  a lower bound; use measured IFU PSF widths when available.
- **No pixel covariance.** Variance propagation ignores the covariance between pixels introduced by
  convolution and resampling.
- **Imaging uses no K-correction.** `simulate_image_observation` assumes the same filter and a flat
  f_ν SED.

## Code
- `io.read_in_data` (FITS branch) builds the wavelength axis with `np.arange(CRVAL3, ...)` and ignores
  CRPIX3. With a Cube input it returns the *primary* header, so cubes whose WCS sits in the data
  extension fail in `redshift_wavelength_axis`.
- `plotting.quicklook.plotSpectrumOfPixel` and `plotSpectrumWithEmissionLines` call
  `cube.get_spectrum`, which mpdaf `Cube` doesn't have.
- `plotting.maps.plot_flux_map_for_z(logscale=True)` masks `flux_map` but takes the log of `flux`.
- `plotting.maps.test_MUSE_FOV` overwrites CDELT on a header that may use a PC/CD matrix.
- `cubeshift.mosaic.MosaicData.make_mosaic` is not implemented yet.
