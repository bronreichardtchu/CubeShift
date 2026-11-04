"""Instrument definitions: telescope specs and JWST/NIRSpec mode tables."""
from __future__ import annotations


class Telescope:
    """Basic description of an IFU instrument.

    Parameters
    ----------
    name : str
    spatial_fwhm : float
        PSF FWHM in arcsec.
    pixel_scale_x, pixel_scale_y : float
        Spaxel size in arcsec/pixel.
    spectral_resolution : float
        Resolving power R = lambda / delta-lambda.
    fov_x, fov_y : float, optional
        Field of view in arcsec (needed when resampling to this instrument).
    spectral_sampling : float, optional
        Wavelength step in Angstrom.
    mirror_diameter : float, optional
        Primary mirror diameter in metres.  If given, the instrument is treated
        as diffraction limited: :meth:`psf_fwhm` = 1.025 lambda / D (an
        idealised lower bound; the delivered IFU PSF is broader).
    """

    def __init__(self, name, spatial_fwhm, pixel_scale_x, pixel_scale_y, spectral_resolution,
                 fov_x=None, fov_y=None, spectral_sampling=None, mirror_diameter=None):
        self.name = name
        self.spatial_fwhm = spatial_fwhm
        self.pixel_scale_x = pixel_scale_x
        self.pixel_scale_y = pixel_scale_y
        self.spectral_resolution = spectral_resolution
        self.fov_x = fov_x
        self.fov_y = fov_y
        self.spectral_sampling = spectral_sampling
        self.mirror_diameter = mirror_diameter

    def psf_fwhm(self, wavelength_angstrom):
        """PSF FWHM in arcsec at the given wavelength(s) (Angstrom).

        Diffraction limit 1.025 lambda/D if ``mirror_diameter`` is set, otherwise
        the constant ``spatial_fwhm`` (e.g. seeing).
        """
        import numpy as np

        lam = np.asarray(wavelength_angstrom, dtype=float)
        if self.mirror_diameter is None:
            return np.full_like(lam, self.spatial_fwhm)
        return np.degrees(1.025 * lam * 1e-10 / self.mirror_diameter) * 3600.0

    def __repr__(self):
        return (f"Telescope({self.name!r}, pixel={self.pixel_scale_x}x{self.pixel_scale_y}\", "
                f"fwhm={self.spatial_fwhm}\", R={self.spectral_resolution}, "
                f"fov={self.fov_x}x{self.fov_y}\")")


TELESCOPES = {
    # FOV is only defined for JWST NIRSpec; add fov_x/fov_y to use others as a target.
    "JWST_NIRCam": Telescope(
        name="JWST NIRCam",
        pixel_scale_x=0.031,
        pixel_scale_y=0.063,
        spatial_fwhm=0.07,          # around F200W
        spectral_resolution=1000,
        mirror_diameter=6.5,
    ),
    "VLT_MUSE": Telescope(
        name="VLT MUSE",
        pixel_scale_x=0.2,
        pixel_scale_y=0.2,
        spatial_fwhm=0.6,           # seeing-limited WFM
        spectral_resolution=3000,
        spectral_sampling=1.25,
    ),
    "JWST_NIRSpec": Telescope(
        name="JWST NIRSpec",
        pixel_scale_x=0.1,          # native IFU spaxel
        pixel_scale_y=0.1,
        spatial_fwhm=0.07,          # nominal; psf_fwhm() uses the diffraction limit
        spectral_resolution=1000,   # medium-resolution gratings (R~1000-2700)
        fov_x=3.0,
        fov_y=3.0,
        spectral_sampling=2.0,
        mirror_diameter=6.5,
    ),
    "Keck_KCWI": Telescope(
        name="Keck KCWI",
        pixel_scale_x=0.29,         # medium slicer
        pixel_scale_y=0.29,
        spatial_fwhm=1.0,           # typical seeing
        spectral_resolution=4000,
        spectral_sampling=0.5,
    ),
}

#: Backwards-compatible name used throughout the original scripts.
telescope_specs = TELESCOPES

#: NIRSpec filter bandpasses in microns (JDox).
NIRSPEC_THROUGHPUT_RANGES = {
    "f140x": (0.8, 2.0),      # target acquisition
    "f110w": (1.0, 1.3),      # bright target acquisition
    "f070lp": (0.7, 1.3),     # with G140M
    "f100lp": (1.0, 1.9),     # with G140M
    "f170lp": (1.7, 3.2),     # with G235M
    "f290lp": (2.9, 5.3),     # with G395M
    "clear": (0.6, 5.3),      # with PRISM or TA
}

#: NIRSpec dispersers; wavelengths in microns.
NIRSPEC_DISPERSER_RANGES = [
    {"name": "g140m", "R": 1000, "lambda_min": 0.70, "lambda_max": 1.27},
    {"name": "g235m", "R": 1000, "lambda_min": 1.66, "lambda_max": 3.07},
    {"name": "g395m", "R": 1000, "lambda_min": 2.87, "lambda_max": 5.10},
    {"name": "g140h", "R": 2700, "lambda_min": 0.81, "lambda_max": 1.27},
    {"name": "g235h", "R": 2700, "lambda_min": 1.66, "lambda_max": 3.05},
    {"name": "g395h", "R": 2700, "lambda_min": 2.87, "lambda_max": 5.14},
    {"name": "prism", "R": 100, "lambda_min": 0.60, "lambda_max": 5.30},
]
