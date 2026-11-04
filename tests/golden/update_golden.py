"""Regenerate the golden files from the *current* package.

Run this after a deliberate science fix changes pipeline outputs, then commit the
new .npz files together with the fix (say in the commit message which records
changed and why).  Uses the same inputs and steps as make_golden.py, which
recorded the original (pre-refactor) behaviour.

    python tests/golden/update_golden.py            # rewrite both files
    python tests/golden/update_golden.py --check    # only report what would change
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
import matplotlib  # noqa: E402

matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from helpers import snap as _snap  # noqa: E402

from cubeshift import binning, cosmology, extinction, psf, resample, spectral, throughput  # noqa: E402
from cubeshift.instruments import TELESCOPES  # noqa: E402
from cubeshift.redshift import redshift_wavelength_axis  # noqa: E402
from cubeshift.testing import make_test_cube, write_mock_instrument_files  # noqa: E402

Z_OBS, Z_SIM, SEED = 0.025, 3.0, 12345
quiet = lambda: contextlib.redirect_stdout(io.StringIO())  # noqa: E731


def snap(cube, prefix):
    return {prefix + k: v for k, v in _snap(cube).items()}


def chain(data_dir):
    nirspec = TELESCOPES["JWST_NIRSpec"]
    res = {}
    with quiet():
        rc, _ = redshift_wavelength_axis(make_test_cube(), Z_OBS, Z_SIM)
    res.update(snap(rc, "s1_redshift_"))
    rc.data = cosmology.rescale_flux(rc.data, Z_OBS, Z_SIM)
    rc.data = rc.data / resample.get_spaxel_area(rc)
    res.update(snap(rc, "s3_sb_"))
    with quiet():
        tc, filt = throughput.apply_transmission_filter(rc, Z_SIM, 5007, data_dir=data_dir)
    res.update(snap(tc, "s4_throughput_"))
    res["s4_filter"] = np.array(filt)
    np.random.seed(SEED)
    with quiet():
        cr, bf, _, dx, dy = resample.resample_cube_to_telescope_and_redshift(tc, nirspec, Z_OBS, Z_SIM)
    res.update(snap(cr, "s5_resampled_"))
    res["s5_bin_factors"] = np.asarray(bf)
    res["s5_dxdy"] = np.asarray([dx, dy])
    cr.data = cr.data * resample.get_spaxel_area(cr)
    res.update(snap(cr, "s6_flux_"))
    with quiet():
        cp = psf.convolve_to_match_psf(cr, dx, dy, nirspec.spatial_fwhm, Z_OBS, Z_SIM)
    res.update(snap(cp, "s7_psf_"))
    with quiet():
        bl, disp = spectral.apply_dispersion_pipeline(cp, data_dir=data_dir)
    res.update(snap(bl, "s8_disp_"))
    res["s8_disperser"] = np.array(disp)
    w = bl.wave.coord()
    bw = np.median(w) / nirspec.spectral_resolution
    with quiet():
        fin = spectral.resample_spectral_axis(bl, np.arange(w[0], w[-1] + bw, bw))
    res.update(snap(fin, "s9_final_"))
    return res


def units():
    unit = {}
    small = make_test_cube(nx=23, ny=17, lam0=4950.0, nw=40, nan_cols=2)
    for method in ("sum", "mean"):
        for sb in (True, False):
            with quiet():
                b = binning.bin_data(3, 2, small, method=method, surface_brightness=sb)
            unit.update(snap(b, f"bin_cube_{method}_{int(sb)}_"))
    with quiet():
        unit.update(snap(psf.convolve_to_match_psf(small, 0.29, 0.29, 0.5, 0.0, 0.0), "psf_conv_"))
        unit.update(snap(spectral.resample_spectral_axis(small, np.arange(4955.0, 4965.0, 0.3)), "resample_spec_"))
        w = small.wave.coord()
        unit.update(snap(spectral.precomputed_match_spectral_resolution_variable_kernel(
            small[:, :4, :4], np.full(w.size, 4000.0), 1500.0), "lsf_kernel_"))
        unit.update(snap(spectral.crop_cube_to_wavelength_range(small, 4960.0, 4965.0), "crop_wave_"))
    unit["rescale_flux"] = cosmology.rescale_flux(np.ones(3), 0.025, 3.0)
    with quiet():
        _, lam_new = redshift_wavelength_axis(make_test_cube(), Z_OBS, Z_SIM)
    unit["redshift_lam"] = lam_new
    lam = np.linspace(3500, 9000, 50)
    unit["ccm_A_lambda"] = extinction.cardelli_A_lambda(lam, Av=0.3)
    unit["mw_corr"], _ = extinction.milky_way_extinction_correction(lam, np.ones((50, 2, 2)), Av=0.3)
    unit["mw_undo"], _ = extinction.milky_way_extinction_correction(lam, np.ones((50, 2, 2)), Av=0.3, undo=True)
    for method in ("sum", "mean"):
        unit.update(snap(binning.bin_data(3, 2, small, method=method), f"bin_data_{method}_"))
    return unit


def _diff(name, new, path):
    old = np.load(path)
    changed = []
    for k in sorted(set(old.files) | set(new)):
        if k not in old.files or k not in new:
            changed.append(f"{k} (added/removed)")
            continue
        a, b = np.asarray(old[k]), np.asarray(new[k])
        if a.shape != b.shape or a.dtype.kind != b.dtype.kind:
            changed.append(f"{k} (shape {a.shape}->{b.shape})")
        elif a.dtype.kind in "fc" and not np.allclose(a, b, rtol=1e-6, atol=0, equal_nan=True):
            changed.append(k)
        elif a.dtype.kind not in "fc" and not np.array_equal(a, b):
            changed.append(k)
    print(f"{name}: {len(changed)} changed" + ("".join(f"\n  {c}" for c in changed)))
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report differences, don't write")
    a = ap.parse_args()
    data_dir = write_mock_instrument_files(tempfile.mkdtemp(prefix="cubeshift_mock_"))
    for name, new in (("pipeline_chain.npz", chain(data_dir)), ("units.npz", units())):
        path = os.path.join(HERE, name)
        _diff(name, new, path)
        if not a.check:
            np.savez_compressed(path, **new)
            print("  wrote", path)


if __name__ == "__main__":
    main()
