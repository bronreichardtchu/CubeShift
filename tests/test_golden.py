"""The refactored package must reproduce the legacy outputs (tests/golden/make_golden.py).

When a deliberate science fix changes an output, regenerate the affected golden
file from the fixed code and say why in the commit message.
"""
import numpy as np
import pytest

from cubeshift import binning, cosmology, extinction, psf, resample, spectral, throughput
from cubeshift.instruments import TELESCOPES
from cubeshift.redshift import redshift_wavelength_axis
from cubeshift.testing import make_test_cube

from helpers import assert_matches

Z_OBS, Z_SIM, SEED = 0.025, 3.0, 12345


def _small():
    return make_test_cube(nx=23, ny=17, lam0=4950.0, nw=40, nan_cols=2)


@pytest.mark.parametrize("method", ["sum", "mean"])
@pytest.mark.parametrize("sb", [True, False])
def test_bin_data_matches_pipeline_bin_cube(golden_units, quiet, method, sb):
    with quiet():
        b = binning.bin_data(3, 2, _small(), method=method, surface_brightness=sb)
    assert_matches(b, golden_units, f"bin_cube_{method}_{int(sb)}_")


@pytest.mark.parametrize("method", ["sum", "mean"])
def test_bin_data_matches_cubeshift_bin_data(golden_units, method):
    b = binning.bin_data(3, 2, _small(), method=method)
    assert_matches(b, golden_units, f"bin_data_{method}_")


def test_units(golden_units, quiet):
    small = _small()
    with quiet():
        assert_matches(psf.convolve_to_match_psf(small, 0.3, 1.0, 0.025, 0.025), golden_units, "psf_conv_")
        assert_matches(spectral.resample_spectral_axis(small, np.arange(4955.0, 4965.0, 0.3)),
                       golden_units, "resample_spec_")
        assert_matches(spectral.match_spectral_resolution(small[:, :4, :4], 1500.0, 4000.0),
                       golden_units, "lsf_kernel_")
        assert_matches(spectral.crop_cube_to_wavelength_range(small, 4960.0, 4965.0), golden_units, "crop_wave_")
    np.testing.assert_allclose(cosmology.rescale_flux(np.ones(3), 0.025, 3.0), golden_units["rescale_flux"],
                               rtol=1e-12)
    lam = np.linspace(3500, 9000, 50)
    np.testing.assert_allclose(extinction.cardelli_A_lambda(lam, Av=0.3), golden_units["ccm_A_lambda"], rtol=1e-12)
    d, _ = extinction.milky_way_extinction_correction(lam, np.ones((50, 2, 2)), Av=0.3)
    np.testing.assert_allclose(d, golden_units["mw_corr"], rtol=1e-12)
    d, _ = extinction.milky_way_extinction_correction(lam, np.ones((50, 2, 2)), Av=0.3, undo=True)
    np.testing.assert_allclose(d, golden_units["mw_undo"], rtol=1e-12)


@pytest.mark.slow
def test_pipeline_chain(golden_chain, mock_data_dir, quiet):
    """Every step of main.simulate_observation, step by step."""
    g = golden_chain
    nirspec = TELESCOPES["JWST_NIRSpec"]
    with quiet():
        rc, lam_new = redshift_wavelength_axis(make_test_cube(), Z_OBS, Z_SIM)
    assert_matches(rc, g, "s1_redshift_")

    cosmology.scale_cube(rc, cosmology.flux_dimming_factor(Z_OBS, Z_SIM))
    rc = resample.shrink_to_redshift(rc, Z_OBS, Z_SIM)
    cosmology.scale_cube(rc, 1.0 / resample.get_spaxel_area(rc))
    assert_matches(rc, g, "s3_sb_")

    with quiet():
        tc, filt = throughput.apply_transmission_filter(rc, Z_SIM, 5007, data_dir=mock_data_dir)
    assert_matches(tc, g, "s4_throughput_")
    assert filt == str(g["s4_filter"])

    np.random.seed(SEED)
    with quiet():
        cr, bf, _, dx, dy = resample.resample_cube_to_telescope_and_redshift(tc, nirspec, Z_OBS, Z_SIM)
    # padding noise is float32 in the original code
    assert_matches(cr, g, "s5_resampled_", rtol=1e-6)
    np.testing.assert_array_equal(bf, g["s5_bin_factors"])
    np.testing.assert_allclose([dx, dy], g["s5_dxdy"], rtol=1e-12)

    cosmology.scale_cube(cr, resample.get_spaxel_area(cr))
    assert_matches(cr, g, "s6_flux_", rtol=1e-6)

    with quiet():
        cp = psf.convolve_to_match_psf(cr, TELESCOPES["Keck_KCWI"].spatial_fwhm, nirspec.psf_fwhm, Z_OBS, Z_SIM)
    assert_matches(cp, g, "s7_psf_", rtol=1e-6)

    with quiet():
        bl, disp = spectral.apply_dispersion_pipeline(
            cp, source_resolution=TELESCOPES["Keck_KCWI"].spectral_resolution, data_dir=mock_data_dir)
    assert_matches(bl, g, "s8_disp_", rtol=1e-6)
    assert disp == str(g["s8_disperser"])


@pytest.mark.slow
def test_simulate_observation_matches_chain(golden_chain, mock_data_dir, quiet, tmp_path):
    """The public entry point gives the same final cube as the legacy steps."""
    from cubeshift import simulate_observation

    np.random.seed(SEED)
    with quiet():
        res = simulate_observation(make_test_cube(), Z_OBS, Z_SIM, "Keck_KCWI", "JWST_NIRSpec",
                                   data_dir=mock_data_dir, output_dir=str(tmp_path), galaxy_name="TEST")
    assert_matches(res.cube, golden_chain, "s8_disp_", rtol=1e-6)
    assert res.filter_name == "f170lp" and res.disperser == "g235h"
    assert res.final_path.endswith("TEST_z_3.0_f170lp_g235h_lsf.fits")
