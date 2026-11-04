import contextlib
import io
import os

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")


@pytest.fixture(scope="session")
def mock_data_dir(tmp_path_factory):
    from cubeshift.testing import write_mock_instrument_files

    return write_mock_instrument_files(str(tmp_path_factory.mktemp("jwst_mock")))


@pytest.fixture(scope="session")
def golden_chain():
    return np.load(os.path.join(GOLDEN_DIR, "pipeline_chain.npz"))


@pytest.fixture(scope="session")
def golden_units():
    return np.load(os.path.join(GOLDEN_DIR, "units.npz"))


@pytest.fixture
def quiet():
    return lambda: contextlib.redirect_stdout(io.StringIO())
