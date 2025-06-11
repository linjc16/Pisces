# type: ignore
import os.path

import numpy as np
import pytest

from ase import Atoms
from ase.calculators.turbomole import Turbomole


@pytest.fixture(scope="function")
def atoms():
    return Atoms('H2', positions=[(0, 0, 0), (0, 0, 1.1)])


def test_turbomole_H2_rhf_singlet(atoms, turbomole_factory):
    # Write all commands for the define command in a string
    define_str = '\n\na coord\n*\nno\nb all sto-3g hondo\n*\neht\n\n\n\n*'

    atoms.calc = Turbomole(define_str=define_str)

    # Run turbomole
    assert np.isclose(atoms.get_potential_energy(), -28.205659, atol=1e-5)


def test_turbomole_H2_uhf_singlet(atoms, turbomole_factory):
    atoms.calc = Turbomole(**{
        "multiplicity": 1, "uhf": True, "use dft": True
    })

    # Run turbomole
    assert np.isclose(atoms.get_potential_energy(), -30.828865, atol=1e-5)

    # check that it performed a DFT calculation (i.e. basic inputs were most
    # likely understood, cf. issue #735)
    dft_in_output = False
    with open("ASE.TM.dscf.out") as fd:
        for line in fd:
            if "density functional" in line:
                dft_in_output = True
    assert dft_in_output

    # also check that UHF was understood (alpha and beta files present)
    assert os.path.exists("alpha")
    assert os.path.exists("beta")
    assert not os.path.exists("mos")
