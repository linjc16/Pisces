import pytest

from ase.build import molecule
from ase.calculators.siesta.siesta_lrtddft import RamanCalculatorInterface
from ase.vibrations.placzek import PlaczekStatic
from ase.vibrations.raman import StaticRamanCalculator


def test_N2(siesta_factory):

    pynao = pytest.importorskip('pynao')
    print("pynao version: ", pynao.__version__)

    n2 = molecule('N2')

    # enter siesta input
    n2.calc = siesta_factory.calc(
        basis_set='DZP',
        fdf_arguments={
            'COOP.Write': True,
            'WriteDenchar': True,
            'XML.Write': True})

    name = 'n2'
    pynao_args = dict(label="siesta", jcutoff=7, iter_broadening=0.15,
                      xc_code='LDA,PZ', tol_loc=1e-6, tol_biloc=1e-7)
    rm = StaticRamanCalculator(n2, RamanCalculatorInterface, name=name,
                               delta=0.011, exkwargs=pynao_args)
    rm.run()

    Pz = PlaczekStatic(n2, name=name)
    e_vib = Pz.get_energies()
    assert len(e_vib) == 6
    Pz.summary()
