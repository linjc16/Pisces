import numpy as np
import pytest

from ase.constraints import FixAtoms

calc = pytest.mark.calculator


@pytest.fixture()
def calc_settings():
    """Some simple fast calculation settings"""
    return dict(xc='lda',
                prec='Low',
                algo='Fast',
                setups='minimal',
                ismear=0,
                nelm=1,
                sigma=0.1,
                istart=0,
                ibrion=5,
                nfree=2,
                potim=0.05,
                gamma=True,
                txt="-",
                lwave=False,
                lcharg=False)


@calc('vasp')
def test_vasp_freq(factory, atoms_nh3, calc_settings):
    """
    Run some VASP tests to ensure that the frequency aspects of the
    calculator work. This is conditional on the existence of the
    VASP_COMMAND or VASP_SCRIPT environment variables

    Tests read_vib_freq and get_vibrations against each other.
    """
    calc = factory.calc(**calc_settings)
    mol = atoms_nh3
    # one constraint
    c = FixAtoms(indices=[atom.index for atom in mol if atom.symbol == 'N'])
    mol.set_constraint(c)
    mol.calc = calc
    en = mol.get_potential_energy()
    assert isinstance(en, float)

    n_free = 3 * (len(mol) - 1)  # one constraint

    e, i_e = calc.read_vib_freq()
    i_e = [complex(0, x) for x in i_e]
    assert len(e) + len(i_e) == n_free
    assert i_e
    outcar_data = i_e[-1::-1] + e[-1::-1]

    vib_obj = calc.get_vibrations()
    vib_data = vib_obj.get_energies() * 1000  # to meV
    np.testing.assert_allclose(vib_data, outcar_data, rtol=1e-6)
    # Cleanup
    calc.clean()
