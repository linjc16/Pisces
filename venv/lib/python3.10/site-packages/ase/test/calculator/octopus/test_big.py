import pytest

from ase.build import bulk
from ase.collections import g2


def calculate(factory, system, **kwargs):
    calc = factory.calc(**kwargs)
    system.calc = calc
    system.get_potential_energy()
    calc.get_eigenvalues()
    return calc


calc = pytest.mark.calculator


@calc('octopus', Spacing='0.2 * angstrom')
def test_o2(factory):
    atoms = g2['O2']
    atoms.center(vacuum=2.5)
    calculate(factory,
              atoms,
              BoxShape='parallelepiped',
              SpinComponents='spin_polarized',
              ExtraStates=2)


@calc('octopus')
def test_si(factory):
    calc = calculate(factory,
                     bulk('Si'),  # , orthorhombic=True),
                     KPointsGrid=[[4, 4, 4]],
                     KPointsUseSymmetries=True,
                     SmearingFunction='fermi_dirac',
                     ExtraStates=2,
                     Smearing='0.1 * eV',
                     ExperimentalFeatures=True,
                     Spacing='0.45 * Angstrom')
    eF = calc.get_fermi_level()
    print('eF', eF)
