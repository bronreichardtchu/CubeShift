"""Mosaicking several cubes (work in progress).

Adapted from Anna McLeod's script for mosaicking MUSE cubes.  Needs the
optional ``montage-wrapper`` dependency (``pip install cubeshift[mosaic]``).
"""
from __future__ import annotations

import glob
from pathlib import Path


class MosaicData:
    """Collection of cube files to be mosaicked.

    Parameters
    ----------
    files : list of str or str
        List of FITS files, or a directory containing them.
    output_dir : str
        Where outputs will be written.
    """

    def __init__(self, files: str | list[str], output_dir: str) -> None:
        if isinstance(files, list):
            for file in files:
                if not Path(file).is_file():
                    raise FileNotFoundError(f"File does not exist: {file}")
        elif isinstance(files, str):
            if not Path(files).is_dir():
                raise FileNotFoundError(f"Input directory does not exist, or is not a directory: {files}")
            files = sorted(glob.glob(str(Path(files) / "*.fits")))
            if len(files) < 1:
                raise FileNotFoundError("No .fits files in the input directory")
        else:
            raise TypeError(f"files must be a list of paths or a directory, not {type(files)}")

        self.list_of_files = files
        self.output_dir = output_dir

    def make_mosaic(self, lam_min, lam_max, mosaic_name):
        """Mosaic the cubes over [lam_min, lam_max].  Not implemented yet.

        Plan: limit the wavelength range of all cubes (consider memmap); save
        them; extract matching slices from each cube; mosaic each slice with
        montage; reassemble the cube.
        """
        raise NotImplementedError("MosaicData.make_mosaic is not implemented yet")
