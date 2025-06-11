"""
Run some VASP tests to ensure that the VASP calculator works. This
is conditional on the existence of the VASP_COMMAND or VASP_SCRIPT
environment variables

"""

import numpy as np
import pytest

from ase.io import read

calc = pytest.mark.calculator


@calc('vasp')
def test_main(factory, atoms_co):

    # simple test calculation of CO molecule
    co = atoms_co  # aliasing

    calc = factory.calc(xc='LDA',
                        prec='Low',
                        algo='Fast',
                        lorbit=11,
                        ismear=0,
                        sigma=1.,
                        nbands=12,
                        istart=0,
                        nelm=3,
                        lwave=False,
                        lcharg=False,
                        ldipol=True)

    co.calc = calc
    energy = co.get_potential_energy()
    forces = co.get_forces()
    dipole_moment = co.get_dipole_moment()

    # check that parsing of vasprun.xml file works
    conf = read('vasprun.xml')
    assert conf.calc.parameters['kpoints_generation']
    assert conf.calc.parameters['sigma'] == 1.0
    assert conf.calc.parameters['ialgo'] == 68
    assert energy - conf.get_potential_energy() == 0.0

    # Check some arrays
    assert np.allclose(conf.get_forces(), forces)
    assert np.allclose(conf.get_dipole_moment(), dipole_moment, atol=1e-6)

    # Check k-point-dependent properties
    assert len(conf.calc.get_eigenvalues(spin=0)) >= 12
    assert conf.calc.get_occupation_numbers()[2] == 2
    assert conf.calc.get_eigenvalues(spin=1) is None
    kpt = conf.calc.get_kpt(0)
    assert kpt.weight == 1.

    # Perform a spin-polarised calculation
    co.calc.set(ispin=2, ibrion=-1)
    co.get_potential_energy()
    conf = read('vasprun.xml')
    assert len(conf.calc.get_eigenvalues(spin=1)) >= 12
    assert conf.calc.get_occupation_numbers(spin=1)[0] == 1.

    # Cleanup
    calc.clean()
