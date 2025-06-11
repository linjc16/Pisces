from ase.atoms import Atoms
from ase.calculators.singlepoint import (
    SinglePointDFTCalculator,
    SinglePointKPoint,
)
from ase.dft.dos import DOS


def test_dos():

    atoms = Atoms('H')
    eFermi = [0, 1]
    kpts = [SinglePointKPoint(1, 0, 0), SinglePointKPoint(1, 1, 0)]
    kpts[0].eps_n = [-2, -1, 1]
    kpts[0].f_n = [1, 0, 0]
    kpts[1].eps_n = [-2.5, -1.5, 0.5]
    kpts[1].f_n = [1, 0, 0]

    calc = SinglePointDFTCalculator(atoms, efermi=eFermi)
    calc.kpts = kpts

    DOS(calc)
    # Maybe actually assert something?
