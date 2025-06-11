"""Tests for `read_bands`."""
import io

import numpy as np

from ase.io.castep import read_bands, units_CODATA2002

BUF = """\
Number of k-points     4
Number of spin components 2
Number of electrons  5.500     2.500
Number of eigenvalues     10    10
Fermi energies (in atomic units)     0.182885    0.182885
Unit cell vectors
   -2.704198    2.704198    2.704198
    2.704198   -2.704198    2.704198
    2.704198    2.704198   -2.704198
K-point     1 -0.25000000 -0.25000000 -0.25000000  0.25000000
Spin component 1
    0.03797397
    0.03809342
    0.03809342
    0.13390429
    0.13390430
    0.46846377
    0.46852514
    0.46852514
    0.53647671
    1.55837771
Spin component 2
    0.12773024
    0.12785189
    0.12785190
    0.24697150
    0.24697150
    0.53413872
    0.53422298
    0.53422298
    0.59174515
    1.58602595
K-point     2 -0.25000000 -0.25000000  0.25000000  0.25000000
Spin component 1
   -0.00308414
    0.06298593
    0.08730381
    0.08740240
    0.11207930
    0.17128862
    0.76788026
    0.76791965
    0.79890351
    0.87628010
Spin component 2
    0.04322315
    0.15531185
    0.19576154
    0.19587468
    0.22857350
    0.27180878
    0.82703085
    0.82706954
    0.84689967
    0.91245541
K-point     3 -0.25000000  0.25000000  0.25000000  0.25000000
Spin component 1
   -0.00308413
    0.06298593
    0.08730382
    0.08740240
    0.11207930
    0.17128862
    0.76788026
    0.76791964
    0.79890351
    0.87628010
Spin component 2
    0.04322315
    0.15531186
    0.19576154
    0.19587468
    0.22857351
    0.27180877
    0.82703086
    0.82706953
    0.84689968
    0.91245539
K-point     4  0.25000000 -0.25000000  0.25000000  0.25000000
Spin component 1
   -0.00308413
    0.06298593
    0.08730381
    0.08740240
    0.11207930
    0.17128862
    0.76788026
    0.76791965
    0.79890351
    0.87628010
Spin component 2
    0.04322315
    0.15531185
    0.19576154
    0.19587468
    0.22857351
    0.27180878
    0.82703085
    0.82706954
    0.84689968
    0.91245540
"""


def test_read_bands():
    """Test `read_bands`."""
    Hartree = units_CODATA2002['Eh']
    kpts, weights, eigenvalues, efermi = read_bands(io.StringIO(BUF))
    kpts_ref = [
        [-0.25000000, -0.25000000, -0.25000000],
        [-0.25000000, -0.25000000, +0.25000000],
        [-0.25000000, +0.25000000, +0.25000000],
        [+0.25000000, -0.25000000, +0.25000000],
    ]
    weights_ref = [0.25, 0.25, 0.25, 0.25]
    eigenvalues_ref0 = [
        +0.03797397,
        +0.03809342,
        +0.03809342,
        +0.13390429,
        +0.13390430,
        +0.46846377,
        +0.46852514,
        +0.46852514,
        +0.53647671,
        +1.55837771,
    ]
    eigenvalues_ref1 = [
        +0.12773024,
        +0.12785189,
        +0.12785190,
        +0.24697150,
        +0.24697150,
        +0.53413872,
        +0.53422298,
        +0.53422298,
        +0.59174515,
        +1.58602595,
    ]
    efermi_ref = 0.182885
    np.testing.assert_allclose(kpts, kpts_ref)
    np.testing.assert_allclose(weights, weights_ref)
    assert eigenvalues.shape == (2, 4, 10)
    np.testing.assert_allclose(eigenvalues[0, 0] / Hartree, eigenvalues_ref0)
    np.testing.assert_allclose(eigenvalues[1, 0] / Hartree, eigenvalues_ref1)
    np.testing.assert_allclose(efermi / Hartree, efermi_ref)
