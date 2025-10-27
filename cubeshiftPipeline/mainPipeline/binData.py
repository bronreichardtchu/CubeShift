import numpy as np 
from mpdaf.obj import Cube, Image
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from reprojectBinData import reproject_cube_preserve_wcs
from astropy.wcs import WCS as AstropyWCS 
from cropCube import auto_crop_cube, trim_empty_edges, crop_cube_to_size
from ioTools import read_in_data

class Telescope:
    def __init__(self, name, spatial_fwhm, pixel_scale_x, pixel_scale_y, spectral_resolution, fov_x=None, fov_y=None,
                 spectral_sampling=None):
        self.name = name
        self.spatial_fwhm = spatial_fwhm  # arcsec
        self.pixel_scale_x = pixel_scale_x    # arcsec/pixel
        self.pixel_scale_y = pixel_scale_y    # arcsec/pixel
        self.spectral_resolution = spectral_resolution  # R = λ/Δλ
        self.fov_x = fov_x
        self.fov_y = fov_y
        self.spectral_sampling = spectral_sampling  # Δλ in Å 

# Telescope dict
# Holds telescope OBJECTS (!)
telescope_specs = {
    # Need to add FOV to the rest of the telescopes if you want to use them (JWST NIRSpec is the only one that currently has it)
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
        fov_x=3.0,           # arcseconds
        fov_y=3.0,            # arcseconds
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

def calc_proper_dist(z):
    """Return angular diameter distance in parsecs at redshift z"""
    return cosmo.angular_diameter_distance(z).to(u.pc)


def get_spaxel_area(cube):
    """
    Return spaxel area in arcsec^2 from MPDAF cube.
    Uses MPDAF's WCS helper instead of raw astropy.wcs.
    """
    dx_arcsec, dy_arcsec = cube.wcs.get_step(unit='arcsec')
    return abs(dx_arcsec * dy_arcsec)




def _reduction_factor_to_array(x_factor, y_factor, data_cube):
    """Turns the reduction factors (x and y) into an array.  We keep the 
    z-direction as 1 since we don't want to bin in the wavelength direction.

    Parameters
    ----------
    x_factor : int
        The integer reduction factor along the image x axis
    y_factor : int
        The integer reduction factor along the image y axis
    data_cube : `mpdaf.obj.Cube`
        mpdaf cube object of the data

    Returns
    -------
    `~numpy.ndarray`
        An array of the reduction factors dependent on the shape of the data array
    """
    
    # putting the first reduction factor as 1 since we don't want to bin 
    # in the wavelength direction
    if len(data_cube.shape)==3:
        factor = np.asarray([1, y_factor, x_factor])
    elif len(data_cube.shape)==2:
        factor = np.asarray([y_factor, x_factor])
    elif len(data_cube.shape)==1:
        factor = np.asarray([x_factor])

    return factor 



def bin_cube(x_factor, y_factor, data_cube, margin='center', method='sum', inplace=False, surface_brightness=True):
    """Combine the neighbouring pixels to reduce the spatial size of the array 
    by integer factors along the x and y axes.  Each output pixel is the sum of 
    n pixels, where n is the product of the reduction factors x_factor and 
    y_factor.  Adapted from mpdaf.DataArray._rebin(), changed to allow the choice 
    to sum the pixels, take the median or take the mean while binning.

    Parameters
    ----------
    x_factor : int
        The integer reduction factor along the image x axis.
    y_factor : int
        The integer reduction factor along the image y axis.
    data_cube : `mpdaf.obj.Cube`
        mpdaf cube object of the data
    margin : str, optional
        When the dimensions of the input array are not integer multiples of the 
        reduction factor, the array is truncated to remove just enough pixels
        that its dimensions are multiples of the reduction factor.  This subarray 
        is then rebinned in place of the original array.  The margin parameter 
        determines which pixels of the input array are truncated, and which remain.
        By default 'center'.

        The options are:
            'origin' or 'left':
                The starts of the axes of the output array are coincident with 
                the starts of the axes of the input array.
            'center':
                The centre of the output array is aligned with the centre of the 
                input array, within one pixel along each axis.
            'right':
                The ends of the axes of the output array are coincident with the 
                ends of the axes of the input array.
    method : str, optional
        The method used to combine pixels when binning.  By default 'sum'.
        
        The options are:
            'sum':
                Takes the sum of the included pixels
            'median':
                Takes the median of the included pixels
            'mean':
                Takes the mean of the included pixels 
    inplace : bool, optional
        If False, return a rebinned copy of the data array (the default).
        If True, rebin the original data array in-place, and return that.

    Returns
    -------
    `mpdaf.obj.Cube`
        The rebinned cube

    Raises
    ------
    ValueError
        'Unsupported margin parameter' - the margin mode is not one of 'center',
        'origin', 'left', or 'right'.
    ValueError
        'The reduction factors must be from 1 to shape' - the shape of the array 
        is not larger than the reduction factors, so the array can't be divided
        into bins of that size.
    """
    # If inplace is false, create a copy of the cube to change
    data_cube = data_cube if inplace else data_cube.copy()
    
    # Reject unsupported margin modes.
    if margin not in ('center', 'origin', 'left', 'right'):
        raise ValueError('Unsupported margin parameter: %s' % margin)

    # turn the reduction factors into an array
    factor = _reduction_factor_to_array(x_factor, y_factor, data_cube)
    
    # check that the reduction factors are in the range 1 to shape-1
    if np.any(factor < 1) or np.any(factor>=data_cube.shape):
        raise ValueError('The reduction factors must be from 1 to shape')

    #compute the number of pixels by which each axis dimension is more than
    #an integer multiple of the reduction factor
    n = np.mod(data_cube.shape, factor).astype(int)

    #if necessary, compute the slices needed to shorten the dimensions to be 
    #integer multiples of the axis reduction
    if np.any(n != 0):

        # Add a slice for each axis to a list of slices.
        slices = []
        for k in range(data_cube.ndim):
            # Compute the slice of axis k needed to truncate this axis.
            if margin == 'origin' or margin == 'left':
                nstart = 0
            elif margin == 'center':
                nstart = n[k] // 2
            elif margin == 'right':
                nstart = n[k]
            slices.append(slice(nstart, data_cube.shape[k] - n[k] + nstart))

        slices = tuple(slices)

        # Get a sliced copy of the input object.
        tmp = data_cube[slices]

        # Copy the sliced data back into data_cube, so that inplace=True works.
        data_cube._data = tmp._data
        data_cube._var = tmp._var
        data_cube._mask = tmp._mask
        data_cube.wcs = tmp.wcs
        data_cube.wave = tmp.wave

    # Now the dimensions should be integer multiples of the reduction factors.
    # Need to figure out the shape of the output image 
    newshape = data_cube.shape // factor 

    # create a list of array dimensions that are made up of each of the final 
    # dimensions of the array, followed by the corresponding axis reduction
    # factor.  Reshaping with these dimensions places all of the pixels from 
    # each axis that are to be summed on their own axis.
    preshape = np.column_stack((newshape, factor)).ravel()

    # compute the number of unmasked pixels of the data cube that will contribute
    # to each summed pixel in the output array 
    # Count how many unmasked pixels are in each binned region
    unmasked_mask = ~data_cube.data.mask  # True = unmasked
    unmasked = unmasked_mask.reshape(preshape).astype(int)

    if data_cube.ndim == 3:
        unmasked = unmasked.sum(axis=(1, 3, 5))
    elif data_cube.ndim == 2:
        unmasked = unmasked.sum(axis=(1, 3))


    # Reshape the data array to prepare for binning
    newdata = data_cube.data.reshape(preshape)

    # Determine which axes to reduce (these are the axes of the binning dimensions)
    if data_cube.ndim == 3:
        bin_axes = (1, 3, 5)
    elif data_cube.ndim == 2:
        bin_axes = (1, 3)
    else:
        raise ValueError("Unsupported data cube dimensionality")

    # Apply the chosen reduction method across the binning axes
    if method == 'sum':
        newdata = np.nansum(newdata, axis=bin_axes)
        # if surface brightness is requested, divide by the number of pixels
        # summed to create each output pixel
        if surface_brightness:
            newdata /= unmasked
        data_cube._data = newdata.data 
    elif method == 'mean':
        newdata = np.nanmean(newdata, axis=bin_axes)
    elif method == 'median':
        newdata = np.nanmedian(newdata, axis=bin_axes)
    else:
        raise ValueError(f"Unknown binning method: {method}")

    # Store the result
    data_cube._data = newdata


    # the treatment of the variance array is complicated by the possibility 
    # of masked pixels in the data array. 
    if data_cube._var is not None:
        newvar = data_cube.var.reshape(preshape)
        # When calculating the sum: 
        # A sum of N data pixels p[i] with variance v[i] has a variance of 
        # sum(v[i]) 
        if method == 'sum':
            for k in range(1, data_cube.ndim+1):
                newvar = newvar.sum(k)
        # When calculating the mean: 
        # A sum of N data pixels p[i] with variance v[i] has a variance of 
        # sum(v[i]/N^2) 
        # where N^2 is the number of unmasked pixels in that particular sum.
        elif method == 'mean':
            for k in range(1, data_cube.ndim+1):
                newvar = newvar.sum(k)
            newvar /= unmasked**2
        # When calculating the median: 
        # A sum of N data pixels p[i] has an estimated variance of 
        # (1.253 * stdev(p[i]))^2 / N 
        # where N is the number of unmasked pixels in that particular sum.
        elif method == 'median':
            for k in range(1, data_cube.ndim+1):
                newvar = (1.253 * np.nanstd(newdata, axis=k))**2
            newvar /= unmasked
        # add whichever one it was to the data_cube
        data_cube._var = newvar.data

    # Any pixels in the output array that come from zero unmasked pixels of the 
    # input array should be masked
    data_cube._mask = unmasked < 1

    # update spatial world coordinates
    if data_cube._has_wcs and data_cube.wcs is not None and data_cube.ndim > 1:
        data_cube.wcs = data_cube.wcs.rebin([factor[-2], factor[-1]])
    
    # Preserve wavelength axis
    if hasattr(data_cube, "wave") and data_cube.wave is not None:
        data_cube.wave = data_cube.wave.copy()

    print(f" Wave of binned cube: {data_cube.wave}")
    return data_cube



def resample_cube_to_telescope_and_redshift(
    cube,
    target_telescope,
    z_source,
    z_target,
    trim=True,
    check_numbers=False,
    scaling="Giavalisco",   # "DA" or "Giavalisco"
    name=None
):
    """
    Resample an input cube by binning it to match the spatial resolution
    of a target telescope at a higher redshift, then pad or crop to match
    the target telescope FOV.

    Returns
    -------
    If check_numbers is False:
        cube_resampled : mpdaf.obj.Cube
        bin_factors    : tuple(int, int)  -> (bin_x, bin_y)
        wave_step      : float or None
        dx_arcsec_new  : float  -> pixel scale along x after binning (arcsec/pix)
        dy_arcsec_new  : float  -> pixel scale along y after binning (arcsec/pix)
    If check_numbers is True:
        cube_resampled, bin_factors, wave_step, dx_arcsec_new, dy_arcsec_new, extra1, s_x, s_y
        where extra1 is f_ang (DA) or (L_source, L_target) (Giavalisco)
    """
    import numpy as np
    import astropy.units as u
    from astropy.cosmology import Planck18 as cosmo

    # --- Step 0: extract original cube info from header ---
    try:
        _, _, hdr = read_in_data(cube)
    except Exception:
        _, _, _, hdr = read_in_data(cube)

    print(f"[DEBUG] cube type: {type(cube)}")
    print(f"[DEBUG] cube WCS: {getattr(cube, 'wcs', None)}")

    # Spatial pixel scale in arcsec/pixel (robust for MPDAF)
    # MPDAF returns (dx, dy) in the requested unit.
    dx_arcsec_src, dy_arcsec_src = cube.wcs.get_step(unit='arcsec')
    dx_arcsec_src = float(abs(dx_arcsec_src))
    dy_arcsec_src = float(abs(dy_arcsec_src))

    # Wavelength step (if needed)
    wave_step = hdr.get('CDELT3', None)

    print("[DEBUG] Source pixel scale (x,y):", dx_arcsec_src, dy_arcsec_src)
    print("[DEBUG] Target telescope pixel scale (x,y):", target_telescope.pixel_scale_x, target_telescope.pixel_scale_y)
    print("[DEBUG] z_obs =", z_source, " z_sim =", z_target)

    # --- Step 1: compute scaling ---
    scaling = scaling.lower()
    if scaling == "da":
        f_ang = (cosmo.angular_diameter_distance(z_source) /
                 cosmo.angular_diameter_distance(z_target)).value
        s_x = f_ang * (dx_arcsec_src / target_telescope.pixel_scale_x)
        s_y = f_ang * (dy_arcsec_src / target_telescope.pixel_scale_y)
        method_used = "Angular Diameter Distance"
    elif scaling == "giavalisco":
        L_source = cosmo.luminosity_distance(z_source).to(u.Mpc).value
        L_target = cosmo.luminosity_distance(z_target).to(u.Mpc).value
        # Note: these s_x, s_y are your existing definitions
        s_x = ((1+z_source)**2 / (1+z_target)**2) * (L_target / L_source) * (target_telescope.pixel_scale_x / dx_arcsec_src)
        s_y = ((1+z_source)**2 / (1+z_target)**2) * (L_target / L_source) * (target_telescope.pixel_scale_y / dy_arcsec_src)
        method_used = "Giavalisco scaling"
    else:
        raise ValueError(f"Unknown scaling method: {scaling}")

    print(f"Scaling method: {method_used}")
    print(f"Raw resampling factors: s_x = {s_x:.3f}, s_y = {s_y:.3f}")

    # --- Step 2: Round to integer for binning ---
    # Remember: mpdaf cube shape is (nw, ny, nx)
    bin_factor_x = max(1, int(round(s_x)))
    bin_factor_y = max(1, int(round(s_y)))
    print(f"Integer bin factors: {bin_factor_x}, {bin_factor_y}")

    # --- Step 3: Bin the cube (order is bin_y, bin_x) ---
    cube_binned = bin_cube(bin_factor_y, bin_factor_x, cube)

    if trim:
        cube_binned = trim_empty_edges(cube_binned)

    ny_binned, nx_binned = cube_binned.shape[1], cube_binned.shape[2]
    print(f"[DEBUG] After binning: cube shape = {ny_binned} × {nx_binned}")

    # Compute new pixel scales (arcsec/pix). Prefer reading from the binned WCS.
    try:
        dx_arcsec_new, dy_arcsec_new = cube_binned.wcs.get_step(unit='arcsec')
        dx_arcsec_new = float(abs(dx_arcsec_new))
        dy_arcsec_new = float(abs(dy_arcsec_new))
    except Exception:
        # Fallback: scale original by the bin factors
        dx_arcsec_new = float(abs(dx_arcsec_src * bin_factor_x))
        dy_arcsec_new = float(abs(dy_arcsec_src * bin_factor_y))

    # --- Step 4: Determine target FOV in pixels from telescope ---
    target_nx = int(np.round(target_telescope.fov_x / target_telescope.pixel_scale_x))
    target_ny = int(np.round(target_telescope.fov_y / target_telescope.pixel_scale_y))
    print(f"[DEBUG] Target telescope FOV in pixels: {target_ny} × {target_nx}")

    # --- Step 5: Pad or crop cube to match telescope FOV ---
    needs_pad_x = nx_binned < target_nx
    needs_pad_y = ny_binned < target_ny

    if needs_pad_x or needs_pad_y:
        print("[DEBUG] Padding cube in one or both directions")
        cube_resampled = pad_cube_with_background(
            cube_binned,
            target_nx if needs_pad_x else nx_binned,
            target_ny if needs_pad_y else ny_binned,
            name=name
        )
    else:
        cube_resampled = cube_binned

    # After padding/cropping, pixel scale is unchanged.
    print(f"[DEBUG] Final pixel scale after binning (x,y) = ({dx_arcsec_new}, {dy_arcsec_new}) arcsec/pix")

    # --- Return values ---
    if check_numbers:
        if scaling == "giavalisco":
            # Keep s_x, s_y for your debug; also return (L_source, L_target)
            return (cube_resampled,
                    (bin_factor_x, bin_factor_y),
                    wave_step,
                    dx_arcsec_new, dy_arcsec_new,
                    (L_source, L_target),
                    s_x, s_y)
        else:
            return (cube_resampled,
                    (bin_factor_x, bin_factor_y),
                    wave_step,
                    dx_arcsec_new, dy_arcsec_new,
                    float(f_ang),
                    s_x, s_y)

    return cube_resampled, (bin_factor_x, bin_factor_y), wave_step, dx_arcsec_new, dy_arcsec_new


def pad_cube_with_background(cube, target_nx, target_ny, name=None):
    """
    Pad an MPDAF Cube to a larger spatial size with background values
    estimated from each wavelength slice, while preserving WCS,
    variance, and mask if they exist. Noise is added to the padding
    based on the background level and its standard deviation.

    Parameters
    ----------
    cube : mpdaf.obj.Cube
        Input cube to pad.
    target_nx : int
        Desired number of pixels along x axis.
    target_ny : int
        Desired number of pixels along y axis.

    Returns
    -------
    cube_padded : mpdaf.obj.Cube
        Cube padded with noisy background values to target size.
    """

    # padding should be determined by the bin factor -- a certain amount to make up the field of view
    # does it need to be cropped? if yes, crop, if no ask:
    # does it fill the FOV
    # if yes, leave it, if no, pad to fit target FOV
    nz, ny, nx = cube.shape
    print(f"[DEBUG] Original cube shape: {nz} x {ny} x {nx}")
    print(f"[DEBUG] Target cube shape: {nz} x {target_ny} x {target_nx}")

    # Compute offsets to center original cube in padded cube
    x0 = (target_nx - nx) // 2
    y0 = (target_ny - ny) // 2
    print(f"[DEBUG] Offsets for centering: x0={x0}, y0={y0}")
    print(f"Cube shape after padding: {nz} x {target_ny} x {target_nx}, memory ~ {nz*target_ny*target_nx*8/1e9:.2f} GB")


    # Initialize padded arrays
    padded_data = np.zeros((nz, target_ny, target_nx), dtype=np.float32)
    padded_var  = None
    padded_mask = None

    if cube.var is not None:
        padded_var = np.zeros((nz, target_ny, target_nx), dtype=cube.var.dtype)
    if cube.mask is not None:
        padded_mask = np.ones((nz, target_ny, target_nx), dtype=bool)

    # Estimate background for each slice and add Gaussian noise
    bkg_list = []
    for k in range(cube.shape[0]):  # loop over wavelength slices
        bkg, bkg_std = cube[k].background()  # mean and std from MPDAF
        bkg_list.append(bkg)
        

        # fill slice with noisy background
        if name == "CGCG453":
            bkg_std *= 0.5 # smaller → less noisy, darker background; larger → more noisy

        padded_data[k, :, :] = np.random.normal(
            loc=bkg, scale=bkg_std, size=(target_ny, target_nx)
        )

        # insert the real data back into the center
        padded_data[k,
                    y0:y0 + cube.shape[1],
                    x0:x0 + cube.shape[2]] = cube[k].data

        if padded_var is not None:
            padded_var[k, y0:y0+ny, x0:x0+nx] = cube.var[k, :, :]
        if padded_mask is not None:
            padded_mask[k, y0:y0+ny, x0:x0+nx] = cube.mask[k, :, :]
    bkg_list = np.array(bkg_list)

    print(f"bkg mean: {np.mean(bkg_list)}")
    if name == "CGCG453":
        print(" bkg std scaled by 0.5 ")


    # Carry over WCS from the original cube and preserve wave + mask
    # Ensure types match what Cube expects
    # Convert padded arrays to masked array where appropriate
    if padded_mask is not None:
        padded_mask = np.asarray(padded_mask, dtype=bool)
        padded_data = np.ma.array(padded_data, mask=~padded_mask)
    else:
        # keep as plain ndarray if there's no mask
        padded_data = np.asarray(padded_data)

    # Create Cube while explicitly passing wave and mask
    cube_padded = Cube(
        data=padded_data,
        var=padded_var,
        wcs=cube.wcs,
        wave=cube.wave,     # <-- preserve the spectral axis
        mask=padded_mask,   # <-- preserve mask if available
        copy=False
    )

    # Safety: if constructor did not set wave for some reason, restore it
    if getattr(cube_padded, "wave", None) is None and getattr(cube, "wave", None) is not None:
        cube_padded.wave = cube.wave.copy()

    print(f"[DEBUG] Padded cube shape: {cube_padded.shape}")
    print(f"[DEBUG] Padded cube wave: {cube_padded.wave}")
    return cube_padded



if __name__ == "__main__":




    # === Load data ===
    print("\n=== Loading Cube ===")
    # name = "TEST"
    # file_path = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/Output_cubes/test_z_0.5_redshifted.fits"
    # cube = Cube(file_path)
    # z_obs = 0.01  # original galaxy redshift
    # z_sim = 0.5    # simulated redshift

    # name = "CGCG453"
    # file_path = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/Output_cubes/CGCG453/CGCG453_redshifted_cube_3.fits"
    # cube = Cube(file_path)
    # z_obs = 0.025  # original galaxy redshift
    # z_sim = 0.3    # simulated redshift

    # name = "UGC10099"
    # file_path = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/UGC10099/J155636_red_mosaic.fits"
    # cube = Cube(file_path)
    # z_obs = 0.035 # from NED: 0.034713 (heliocentric)
    # z_sim = 3

    name = "IRAS08"
    file_path = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/IRAS08/IRAS08_combined_final_metacube.fits"
    cube = Cube(file_path, ext=1)
    z_obs = 0.019 
    z_sim = 3

    # === Define source and target telescopes ===
    source_telescope = telescope_specs["Keck_KCWI"]
    target_telescope = telescope_specs["JWST_NIRSpec"]

    print(f"\nSimulating observation:")
    print(f"  From → {source_telescope.name}")
    print(f"  To   → {target_telescope.name}")


    cropped_resampled_cube, bin_factors, wave_step = resample_cube_to_telescope_and_redshift(
        cube=cube,
        target_telescope=telescope_specs["JWST_NIRSpec"],
        z_source=z_obs,
        z_target=z_sim,
        name=name

    )

    print(f"Final bin factors applied: {bin_factors}")

    output_path = f"/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/Output_cubes/{name}/binData_cube_{name}.fits"
    cropped_resampled_cube.write(output_path)
    print(f"✔ Resampled cube saved to: {output_path}")


    # === Pixel scale & aspect ratio check ===
    # --- Safe way to get pixel scales from WCS ---
    try:
        # If cube is an MPDAF cube, this works directly
        cdelt = cropped_resampled_cube.wcs.get_step(unit='arcsec')  # (dy, dx)
        cdelt1, cdelt2 = cdelt[1], cdelt[0]  # RA, Dec
    except AttributeError:
        # Fall back to Astropy WCS
        cdelt = cropped_resampled_cube.wcs.pixel_scale_matrix
        # Convert degrees → arcsec
        cdelt1 = abs(cdelt[0, 0]) * 3600
        cdelt2 = abs(cdelt[1, 1]) * 3600

    aspect = abs(cdelt1 / cdelt2)

    print("Pixel scales (arcsec):", cdelt1, cdelt2)
    print("Aspect ratio:", aspect)


    print("Pixel scales from header (arcsec):", cdelt1, cdelt2)
    print("Aspect ratio:", aspect)

    # === Visualize original vs resampled ===
    print("\n=== Displaying First Slice of Cube ===")
    plt.figure(figsize=(12, 5))

    # --- Original cube ---
    pixscale_orig = cube.wcs.get_step(unit='arcsec')  # (dy, dx)
    ny, nx = cube.shape[1:]
    extent_orig = [0, nx * pixscale_orig[1], 0, ny * pixscale_orig[0]]

    plt.subplot(1, 2, 1)
    plt.title(f"Original Cube ({source_telescope.name})")
    plt.imshow(cube.data[0], origin="lower", cmap="viridis",
            extent=extent_orig, aspect='equal')
    plt.colorbar(label='Flux')
    plt.xlabel("X [arcsec]")
    plt.ylabel("Y [arcsec]")

    # --- Resampled cube ---
    pixscale_new = cropped_resampled_cube.wcs.get_step(unit='arcsec')  # (dy, dx)
    ny2, nx2 = cropped_resampled_cube.shape[1:]
    extent_new = [0, nx2 * pixscale_new[1], 0, ny2 * pixscale_new[0]]

    plt.subplot(1, 2, 2)
    plt.title(f"Simulated at z={z_sim} ({target_telescope.name})")
    plt.imshow(cropped_resampled_cube.data[0], origin="lower", cmap="viridis",
            extent=extent_new, aspect=aspect)
    plt.colorbar(label='Flux')
    plt.xlabel("X [arcsec]")
    plt.ylabel("Y [arcsec]")

    plt.tight_layout()
    plt.show()
