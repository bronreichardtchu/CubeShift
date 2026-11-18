"""The full simulation: re-observe a cube at another redshift with another instrument."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import astropy.units as u
from astropy.cosmology import Planck18
from mpdaf.obj import Cube

from .cosmology import flux_dimming_factor, scale_cube
from .instruments import TELESCOPES, Telescope
from .redshift import redshift_wavelength_axis
from .resample import get_spaxel_area, resample_cube_to_telescope_and_redshift, shrink_to_redshift
from .spectral import apply_dispersion_pipeline
from .throughput import apply_transmission_filter


@dataclass
class SimulationResult:
    """Output of :func:`simulate_observation`."""

    cube: Cube
    """Final simulated mpdaf Cube."""
    filter_name: str
    """NIRSpec filter applied, e.g. 'f170lp'."""
    disperser: str
    """NIRSpec disperser used, e.g. 'g235h'."""
    bin_factors: tuple
    """Integer spatial binning (x, y)."""
    final_path: str | None = None
    """Where the final cube was written (if ``output_dir`` was given)."""
    intermediate_paths: dict = field(default_factory=dict)


def _as_telescope(t):
    if isinstance(t, Telescope):
        return t
    try:
        return TELESCOPES[t]
    except KeyError:
        raise KeyError(f"Unknown telescope {t!r}; choose from {list(TELESCOPES)} or pass a Telescope") from None


def simulate_observation(cube, z_obs, z_sim, source_telescope, target_telescope, *,
                         data_dir=None, output_dir=None, galaxy_name=None, save_intermediates=False,
                         line_rest_wavelength=5007, noise_scale=1.0, rng=None, cosmo=Planck18,
                         background="auto"):
    """Simulate how a cube would look observed with another telescope at another redshift.

    Steps
    -----
    1. Redshift the wavelength axis.
    2. Dim the flux densities (luminosity distance and (1+z)), and shrink the
       spatial WCS to the angular scale of z_sim (D_A(z_obs)/D_A(z_sim)).
    3. Convert flux to surface brightness (per arcsec^2 at z_sim).
    4. Apply the NIRSpec filter throughput for ``line_rest_wavelength``.
    5. Pad once (PSF margin, multiple of the bin factor, target FOV) with the
       background level and noise; on the fine grid, convolve from the source
       PSF (``source_telescope.spatial_fwhm``, shrunk to z_sim) to the target PSF
       (``target_telescope.psf_fwhm(lambda)``); then bin to the target spaxels.
    6. Convert back to flux per (new) spaxel.
    7. Choose a NIRSpec disperser; on the native wavelength grid convolve from the
       source resolution (``source_telescope.spectral_resolution``) to the
       disperser's R(lambda); rebin (flux-conserving) onto the disperser's grid.

    Parameters
    ----------
    cube : mpdaf.obj.Cube or str
        Input cube (or FITS path), e.g. from Keck/KCWI.
    z_obs, z_sim : float
        Observed and simulated redshift.
    source_telescope, target_telescope : Telescope or str
        Instruments, as objects or keys of :data:`cubeshift.instruments.TELESCOPES`.
    data_dir : path-like, optional
        JWST filter/dispersion files (see :mod:`cubeshift.config`).
    output_dir : path-like, optional
        If given, the final cube is written here (in ``<output_dir>/<galaxy_name>/``
        when ``galaxy_name`` is set).
    galaxy_name : str, optional
        Used for output file names.
    save_intermediates : bool
        Also write the cube after each step (needs ``output_dir``).
    line_rest_wavelength : float
        Rest wavelength (Angstrom) of the line that sets the filter choice.
    noise_scale : float
        Multiplier on the background noise used for the padding.
    background : 'auto', 'zero' or number
        Level and noise of the padding: 'auto' measures them from sky pixels in
        the border of the field, or uses mean 0 if the galaxy fills the field
        (see :func:`cubeshift.padding.estimate_background`).
    rng : numpy.random.Generator, optional
        Random generator for the padding noise.

    Returns
    -------
    SimulationResult
    """
    source_telescope = _as_telescope(source_telescope)
    target_telescope = _as_telescope(target_telescope)
    print(f"Simulating observation as seen by {target_telescope.name} at z = {z_sim}")

    paths = {}

    def _save(c, tag, final=False):
        if output_dir is None or not (final or save_intermediates):
            return None
        out = os.path.join(output_dir, galaxy_name) if galaxy_name else output_dir
        os.makedirs(out, exist_ok=True)
        prefix = f"{galaxy_name}_z_{z_sim}" if galaxy_name else f"z_{z_sim}"
        path = os.path.join(out, f"{prefix}_{tag}.fits")
        c.write(path)
        paths[tag] = path
        return path

    # 1. redshift wavelength axis
    # (a path and a Cube take different branches of io.read_in_data; pass through unchanged)
    redshifted_cube, _ = redshift_wavelength_axis(cube, z_obs, z_sim)
    _save(redshifted_cube, "redshifted")

    # 2. flux dimming;  2b. angular size at z_sim;  3. flux -> surface brightness per z_sim arcsec^2
    scale_cube(redshifted_cube, flux_dimming_factor(z_obs, z_sim, cosmo=cosmo))
    redshifted_cube = shrink_to_redshift(redshifted_cube, z_obs, z_sim, cosmo=cosmo)
    scale_cube(redshifted_cube, 1.0 / get_spaxel_area(redshifted_cube))
    # the values are now per arcsec^2, so say so in the unit (undone in step 6)
    redshifted_cube.unit = redshifted_cube.unit / u.arcsec ** 2
    _save(redshifted_cube, "fluxscaled")

    # 4. filter throughput (in place)
    transmission_cube, filter_name = apply_transmission_filter(
        redshifted_cube, z_sim, line_rest_wavelength, data_dir=data_dir)

    # 5. pad once, match the PSF on the fine grid (source seeing shrunk to z_sim ->
    #    target PSF(lambda)), then bin to the target spaxels
    cube_resampled, bin_factors, _, _, _ = resample_cube_to_telescope_and_redshift(
        transmission_cube, target_telescope, z_obs, z_sim, noise_scale=noise_scale, rng=rng, cosmo=cosmo,
        fwhm_source_arcsec=source_telescope.spatial_fwhm, fwhm_target_arcsec=target_telescope.psf_fwhm,
        background=background)

    # 6. surface brightness -> flux per binned spaxel (masked spaxels stay masked)
    scale_cube(cube_resampled, get_spaxel_area(cube_resampled))
    # back to the input's flux unit, now per (larger) spaxel
    cube_resampled.unit = cube_resampled.unit * u.arcsec ** 2
    _save(cube_resampled, "rebinned_fluxscaled")

    # 7. disperser: LSF from the source R to the grating's R(lambda) on the fine grid,
    #    then flux-conserving rebin onto the grating's wavelength grid
    final_cube, disperser = apply_dispersion_pipeline(
        cube_resampled, source_resolution=source_telescope.spectral_resolution, data_dir=data_dir)
    final_path = _save(final_cube, f"{filter_name}_{disperser}_lsf", final=True)

    return SimulationResult(cube=final_cube, filter_name=filter_name, disperser=disperser,
                            bin_factors=bin_factors, final_path=final_path, intermediate_paths=paths)
