"""Generate golden (regression) outputs from the *legacy* code.

This script imports the original flat modules in ``cubeshiftPipeline/mainPipeline``
and ``cubeshift/`` exactly as they were at commit a58af38 (the merge of main into
dev_binning) and records the output of every pipeline step for a synthetic cube.

The refactored package must reproduce these numbers (see tests/test_golden.py)
until a deliberate science fix changes them.  The legacy modules no longer exist
after the refactor, so to regenerate, check out a58af38 first:

    git worktree add /tmp/cs_legacy a58af38
    cp tests/golden/make_golden.py /tmp/cs_legacy/
    cd /tmp/cs_legacy && python make_golden.py --out <repo>/tests/golden
"""
import argparse
import contextlib
import io
import os
import sys
import tempfile
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")

import astropy.units as u
from astropy.io import fits
from mpdaf.obj import WCS as MpdafWCS
from mpdaf.obj import Cube, WaveCoord

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Synthetic inputs (kept self-contained so this file runs against legacy code)
# ---------------------------------------------------------------------------
Z_OBS, Z_SIM = 0.025, 3.0
SEED = 12345


def make_cube(nx=30, ny=30, pix=0.29, z=Z_OBS, lam0=5080.0, dlam=0.5, nw=200,
              line_sig=1.0, sig_pix=3.0, wcs_in_primary=True, nan_cols=0):
    """Gaussian blob with an [OIII]5007 line on a flat continuum (KCWI-like)."""
    lam = lam0 + dlam * np.arange(nw)
    yy, xx = np.mgrid[0:ny, 0:nx]
    spat = np.exp(-((xx - (nx - 1) / 2) ** 2 + (yy - (ny - 1) / 2) ** 2) / (2 * sig_pix ** 2))
    spec = 1e-17 * (0.05 + np.exp(-0.5 * ((lam - 5006.84 * (1 + z)) / line_sig) ** 2))
    data = spec[:, None, None] * spat[None]
    # smooth, non-uniform variance (distinguishes sum vs mean propagation, compresses well)
    var = (0.01 * data.max()) ** 2 * (1 + xx / nx + 2 * yy / ny)[None] * np.ones((nw, 1, 1))
    if nan_cols:
        data[:, :, :nan_cols] = np.nan
    c = Cube(data=data, var=var,
             wcs=MpdafWCS(crval=(10.0, 10.0), cdelt=(pix / 3600, -pix / 3600),
                          crpix=(ny / 2, nx / 2), shape=(ny, nx), deg=True),
             wave=WaveCoord(crval=lam0, cdelt=dlam, crpix=1, cunit=u.AA))
    if wcs_in_primary:  # KCWI-like: spatial WCS in primary header
        for k, v in c.wcs.to_header().items():
            c.primary_header[k] = v
    return c


def write_mock_instrument_files(directory):
    """Stand-in JWST filter-throughput and dispersion tables."""
    os.makedirs(directory, exist_ok=True)
    for mode, (lo, hi) in {"f170lp": (1.7, 3.2), "f290lp": (2.9, 5.3),
                           "f100lp": (1.0, 1.9), "f070lp": (0.7, 1.3)}.items():
        wl = np.linspace(lo - 0.1, hi + 0.1, 400)
        thr = 0.6 * np.exp(-0.5 * ((wl - (lo + hi) / 2) / (hi - lo)) ** 2)
        thr[(wl < lo) | (wl > hi)] = 0.0
        fits.BinTableHDU.from_columns([
            fits.Column(name="WAVELENGTH", format="D", array=wl),
            fits.Column(name="THROUGHPUT", format="D", array=thr),
        ]).writeto(os.path.join(directory, f"jwst_nirspec_{mode}_trans.fits"), overwrite=True)
    for name, lo, hi, R0 in [("g235h", 1.66, 3.05, 2700), ("g235m", 1.66, 3.07, 1000)]:
        wl = np.arange(lo, hi, 3.96e-4)
        fits.BinTableHDU.from_columns([
            fits.Column(name="WAVELENGTH", format="D", array=wl),
            fits.Column(name="R", format="D", array=R0 * wl / 2.36),
        ]).writeto(os.path.join(directory, f"jwst_nirspec_{name}_disp.fits"), overwrite=True)


def snap(cube, prefix):
    """Flatten an mpdaf Cube into arrays for np.savez."""
    out = {}
    d = cube.data
    out[prefix + "data"] = np.ma.getdata(d).astype(float)
    out[prefix + "mask"] = np.ma.getmaskarray(d)
    out[prefix + "var"] = (np.ma.getdata(cube.var).astype(float)
                           if cube.var is not None else np.zeros(0))
    out[prefix + "wave"] = cube.wave.coord()
    out[prefix + "wcs_step"] = np.asarray(cube.wcs.get_step(unit=u.arcsec))
    out[prefix + "wcs_crpix"] = np.asarray([cube.wcs.get_crpix1(), cube.wcs.get_crpix2()])
    out[prefix + "wcs_crval"] = np.asarray([cube.wcs.get_crval1(), cube.wcs.get_crval2()])
    return out


quiet = lambda: contextlib.redirect_stdout(io.StringIO())


# ---------------------------------------------------------------------------
def main(out_dir, repo):
    sys.path.insert(0, os.path.join(repo, "cubeshiftPipeline", "mainPipeline"))
    with quiet():
        import binData
        import matchSpectral
        import simulateObs
        import zWavelengths
        from applyFilter import apply_nirspec_throughput

    inst_dir = tempfile.mkdtemp(prefix="cubeshift_mock_")
    write_mock_instrument_files(inst_dir)
    _orig_load = matchSpectral.load_dispersion_file
    matchSpectral.load_dispersion_file = lambda n, d: _orig_load(n, inst_dir)

    nirspec = binData.telescope_specs["JWST_NIRSpec"]
    res = {}

    # ---- full simulate_observation chain (main.py steps, without file writes)
    cube = make_cube()
    with quiet():
        rc, lam_new = zWavelengths.redshift_wavelength_axis(cube, Z_OBS, Z_SIM)
    res.update(snap(rc, "s1_redshift_"))

    rc.data = simulateObs.rescale_flux(rc.data, Z_OBS, Z_SIM)
    rc.data = rc.data / binData.get_spaxel_area(rc)
    res.update(snap(rc, "s3_sb_"))

    with quiet():
        tc, _, _, _, filt = apply_nirspec_throughput(rc, Z_SIM, 5007, inst_dir)
    res.update(snap(tc, "s4_throughput_"))
    res["s4_filter"] = np.array(filt)

    np.random.seed(SEED)
    with quiet():
        cr, bf, wstep, dx, dy = binData.resample_cube_to_telescope_and_redshift(
            tc, nirspec, Z_OBS, Z_SIM)
    res.update(snap(cr, "s5_resampled_"))
    res["s5_bin_factors"] = np.asarray(bf)
    res["s5_dxdy"] = np.asarray([dx, dy])

    cr.data = cr.data * binData.get_spaxel_area(cr)
    res.update(snap(cr, "s6_flux_"))

    with quiet():
        cp = simulateObs.convolve_to_match_psf(cr, dx, dy, nirspec.spatial_fwhm, Z_OBS, Z_SIM)
    res.update(snap(cp, "s7_psf_"))

    with quiet():
        bl, disp = matchSpectral.apply_dispersion_pipeline(cp, Z_OBS, Z_SIM)
    res.update(snap(bl, "s8_disp_"))
    res["s8_disperser"] = np.array(disp)

    R = nirspec.spectral_resolution
    w = bl.wave.coord()
    bw = np.median(w) / R
    with quiet():
        fin = matchSpectral.resample_spectral_axis(bl, np.arange(w[0], w[-1] + bw, bw))
    res.update(snap(fin, "s9_final_"))

    np.savez_compressed(os.path.join(out_dir, "pipeline_chain.npz"), **res)

    # ---- individual functions with parameters that exercise their branches
    unit = {}
    small = make_cube(nx=23, ny=17, lam0=4950.0, nw=40, nan_cols=2)
    for method in ("sum", "mean"):
        for sb in (True, False):
            with quiet():
                b = binData.bin_cube(3, 2, small, method=method, surface_brightness=sb)
            unit.update(snap(b, f"bin_cube_{method}_{int(sb)}_"))
    with quiet():
        pc = simulateObs.convolve_to_match_psf(small, 0.29, 0.29, 0.5, 0.0, 0.0)
    unit.update(snap(pc, "psf_conv_"))
    with quiet():
        rs = matchSpectral.resample_spectral_axis(small, np.arange(4955.0, 4965.0, 0.3))
    unit.update(snap(rs, "resample_spec_"))
    w = small.wave.coord()
    with quiet():
        mk = matchSpectral.precomputed_match_spectral_resolution_variable_kernel(
            small[:, :4, :4], np.full(w.size, 4000.0), 1500.0)
    unit.update(snap(mk, "lsf_kernel_"))
    with quiet():
        cr2 = matchSpectral.crop_cube_to_wavelength_range(small, 4960.0, 4965.0)
    unit.update(snap(cr2, "crop_wave_"))
    unit["rescale_flux"] = simulateObs.rescale_flux(np.ones(3), 0.025, 3.0)
    unit["redshift_lam"] = lam_new

    import extinctionCorrection as ec  # needs dustmaps/bs4/pandas installed
    lam = np.linspace(3500, 9000, 50)
    unit["ccm_A_lambda"] = ec.compute_A_lambda(lam, Av=0.3)
    with quiet():
        dcorr, _ = ec.milky_way_extinction_correction(lam, np.ones((50, 2, 2)), Av=0.3)
        dundo, _ = ec.milky_way_extinction_correction(lam, np.ones((50, 2, 2)), Av=0.3, undo=True)
    unit["mw_corr"] = dcorr
    unit["mw_undo"] = dundo

    # Bron's helper-based binning (cubeshift/binData.py), sum/mean on cube with var
    sys.path.insert(0, repo)
    import importlib.util
    spec = importlib.util.spec_from_file_location("cs_binData", os.path.join(repo, "cubeshift", "binData.py"))
    cs_bin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cs_bin)
    for method in ("sum", "mean"):
        b = cs_bin.bin_data(3, 2, small, method=method)
        unit.update(snap(b, f"bin_data_{method}_"))

    np.savez_compressed(os.path.join(out_dir, "units.npz"), **unit)
    print("wrote", out_dir, "| filter", filt, "| disperser", disp, "| bin", bf)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--repo", default=os.path.abspath(os.path.join(HERE, "..", "..")))
    a = ap.parse_args()
    main(a.out, a.repo)
