"""CubeShift: artificially redshift IFU data cubes and re-observe them with another instrument.

Typical use::

    import cubeshift
    result = cubeshift.simulate_observation("cube.fits", z_obs=0.025, z_sim=3,
                                            source_telescope="Keck_KCWI",
                                            target_telescope="JWST_NIRSpec",
                                            data_dir="path/to/jwst_files")
    result.cube.write("simulated.fits")

Modules
-------
pipeline      simulate_observation (the full chain)
redshift      wavelength-axis redshifting, air_to_vac
cosmology     flux dimming and angular scales
binning       integer spatial binning (bin_data)
resample      binning to a target telescope + FOV padding
psf           spatial PSF matching
spectral      NIRSpec disperser choice, spectral resampling and LSF matching
throughput    NIRSpec filter throughput
extinction    Milky Way extinction (CCM89)
cropping      spatial cropping helpers
io            reading FITS data
instruments   Telescope specs and NIRSpec mode tables
config        locating JWST instrument files (CUBESHIFT_DATA_DIR)
testing       synthetic cubes and mock instrument files
analysis      line fluxes, Balmer decrement, SFR
plotting      maps, spectra, RGB images
"""
from .instruments import TELESCOPES, Telescope
from .io import read_cube
from .pipeline import SimulationResult, simulate_observation

__version__ = "0.1.0"

__all__ = ["simulate_observation", "SimulationResult", "Telescope", "TELESCOPES", "read_cube"]
