"""Tests for XrDebye class"""

from pathlib import Path

import numpy as np
import pytest

from ase.cluster.cubic import FaceCenteredCubic
from ase.utils.xrdebye import XrDebye, wavelengths

tolerance = 1E-5


@pytest.fixture()
def xrd():
    # test system -- cluster of 587 silver atoms
    atoms = FaceCenteredCubic('Ag', [(1, 0, 0), (1, 1, 0), (1, 1, 1)],
                              [6, 8, 8], 4.09)
    return XrDebye(atoms=atoms, wavelength=wavelengths['CuKa1'], damping=0.04,
                   method='Iwasa', alpha=1.01, warn=True)


def test_get(xrd):
    expected = 116850.37344
    obtained = xrd.get(s=0.09)
    assert np.abs((obtained - expected) / expected) < tolerance


def test_xrd(testdir, xrd):
    expected = np.array([18549.274677, 52303.116995, 38502.372027])
    obtained = xrd.calc_pattern(x=np.array([15, 30, 50]), mode='XRD')
    assert np.allclose(obtained, expected, rtol=tolerance)
    xrd.write_pattern('tmp.txt')
    assert Path('tmp.txt').exists()


def test_saxs_and_files(testdir, figure, xrd):
    expected = np.array([372650934.006398, 280252013.563702,
                         488123.103628])
    obtained = xrd.calc_pattern(x=np.array([0.021, 0.09, 0.53]),
                                mode='SAXS')
    assert np.allclose(obtained, expected, rtol=tolerance)

    # (Admittedly these tests are a little bit toothless)
    xrd.write_pattern('tmp.txt')
    assert Path('tmp.txt').exists()
    ax = figure.add_subplot(111)
    xrd.plot_pattern(ax=ax, filename='pattern.png')
    assert Path('pattern.png').exists()
