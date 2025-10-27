from astropy.io import fits 
from astropy.wcs import WCS 

from mpdaf import Cube, Image

def read_in_datacube(filename, ext=(1,2)):
    """Reads in the data from a filename to an mpdaf Cube

    Parameters
    ----------
    filename : str
        the location of the file to read in, should be a fits cube
    ext : int or (int, int) or str or (str, str)
        the optional number/name of the data extension or the numbers/names of 
        the data and variance extensions

    Returns
    -------
    `mpdaf.obj.Cube` obj
        A data cube
    """
    datacube = Cube(filename, ext=ext)

    return datacube 

def read_in_dataim(filename, ext=(1,2)):
    """Reads in the data from a filename, into an mpdaf Image

    Parameters
    ----------
    filename : str
        the location of the file to read in, should be a fits cube
    ext : int or (int, int) or str or (str, str)
        the optional number/name of the data extension or the numbers/names of 
        the data and variance extensions

    Returns
    -------
    `mpdaf.obj.Image` obj
        A data image
    """
    dataim = Image(filename, ext=ext)

    return dataim