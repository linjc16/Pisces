import numpy as np
import pytest

from ase.build import molecule
from ase.calculators.siesta.siesta_lrtddft import SiestaLRTDDFT


def test_siesta_lrtddft(siesta_factory):

    pynao = pytest.importorskip('pynao')
    print("pynao version: ", pynao.__version__)

    # Define the systems
    ch4 = molecule('CH4')

    lrtddft = SiestaLRTDDFT(label="siesta", xc_code='LDA,PZ')

    # run siesta
    lrtddft.get_ground_state(ch4)

    freq = np.arange(0.0, 25.0, 0.5)
    pmat = lrtddft.get_polarizability(freq)
    assert pmat.size == 3 * 3 * freq.size
