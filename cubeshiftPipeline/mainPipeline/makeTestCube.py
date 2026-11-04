import numpy as np
import matplotlib.pyplot as plt
from astropy.io.fits import Header
from mpdaf.obj import Cube, WCS as MPDAF_WCS, WaveCoord
from astropy.modeling.models import Sersic2D

def make_emission_line_cube(nx=100, ny=100, nw=50,
                            lam_min=5000.0, lam_max=5100.0,
                            fwhm_spectral=2.0,
                            pixscale_arcsec=0.03,
                            sersic_n=1.0, r_eff=10.0, amp=1.0):
    """Make a synthetic cube with identical Gaussian line in each spaxel,
       then modulate flux spatially by a Sérsic profile."""
    
    # --- spectral axis: Gaussian line profile ---
    cdelt = (lam_max - lam_min) / (nw - 1)
    wave = WaveCoord(crval=lam_min, cdelt=cdelt, crpix=1,
                     ctype='WAVE', cunit='Angstrom')

    spec_pix = np.arange(nw)
    spec_center = (nw - 1)/2.0
    sigma = fwhm_spectral / 2.355
    spectral_profile = np.exp(-0.5 * ((spec_pix - spec_center)/sigma)**2)

    # --- build flat cube: same spectrum everywhere ---
    data = np.tile(spectral_profile[:, None, None], (1, ny, nx))

    # --- Sérsic profile for spatial modulation ---
    y, x = np.mgrid[0:ny, 0:nx]
    x0 = (nx - 1)/2.0
    y0 = (ny - 1)/2.0
    sersic = Sersic2D(amplitude=amp, r_eff=r_eff, n=sersic_n,
                      x_0=x0, y_0=y0)(x, y)
    
    # normalize Sérsic so center = 1, then scale cube
    sersic /= sersic.max()
    for i in range(nw):
        data[i] *= sersic

    # --- FITS WCS for spatial axes ---
    h = Header()
    h['NAXIS']  = 2
    h['NAXIS1'] = nx
    h['NAXIS2'] = ny
    h['CTYPE1'] = 'RA---TAN'
    h['CTYPE2'] = 'DEC--TAN'
    h['CRVAL1'] = 0.0
    h['CRVAL2'] = 0.0
    h['CRPIX1'] = (nx + 1)/2.0
    h['CRPIX2'] = (ny + 1)/2.0
    scale_deg = pixscale_arcsec / 3600.0
    h['CDELT1'] = scale_deg
    h['CDELT2'] = scale_deg
    h['CUNIT1'] = 'deg'
    h['CUNIT2'] = 'deg'

    wcs_spatial = MPDAF_WCS(h)

    # --- make the cube ---
    cube = Cube(data=data, wave=wave, wcs=wcs_spatial)

    # add spectral axis keywords for consistency
    hdr = cube.data_header
    hdr['CRVAL3'] = lam_min
    hdr['CDELT3'] = cdelt
    hdr['CRPIX3'] = 1
    hdr['CTYPE3'] = 'WAVE'
    hdr['CUNIT3'] = 'Angstrom'

    # --- diagnostic plots ---
    summed = data.sum(axis=(1, 2))
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(lam_min + cdelt * np.arange(nw), summed)
    plt.xlabel('Wavelength [Å]')
    plt.ylabel('Summed flux')
    plt.title('Summed spectrum')

    plt.subplot(1,2,2)
    plt.imshow(sersic, origin='lower', cmap='magma')
    plt.colorbar(label='Relative flux')
    plt.title('Sérsic profile (spatial)')
    plt.tight_layout()
    plt.show()

    return cube
