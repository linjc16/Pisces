import numpy as np
import pytest

from ase import io
from ase.build import bulk
from ase.filters import StrainFilter
from ase.optimize import BFGS

calc = pytest.mark.calculator


@calc('vasp')
def test_vasp_Al_volrelax(factory):
    """
    Run VASP tests to ensure that relaxation with the VASP calculator works.
    This is conditional on the existence of the VASP_COMMAND or VASP_SCRIPT
    environment variables.

    """

    # -- Perform Volume relaxation within Vasp
    def vasp_vol_relax():
        Al = bulk('Al', 'fcc', a=4.5, cubic=True)
        calc = factory.calc(xc='LDA',
                            isif=7,
                            nsw=5,
                            ibrion=1,
                            ediffg=-1e-3,
                            lwave=False,
                            lcharg=False)
        Al.calc = calc
        Al.get_potential_energy()  # Execute

        # Explicitly parse atomic position output file from Vasp
        CONTCAR_Al = io.read('CONTCAR', format='vasp')

        print('Stress after relaxation:\n', calc.read_stress())

        print('Al cell post relaxation from calc:\n',
              calc.get_atoms().get_cell())
        print('Al cell post relaxation from atoms:\n', Al.get_cell())
        print('Al cell post relaxation from CONTCAR:\n', CONTCAR_Al.get_cell())

        # All the cells should be the same.
        assert (calc.get_atoms().get_cell() == CONTCAR_Al.get_cell()).all()
        assert (Al.get_cell() == CONTCAR_Al.get_cell()).all()

        return Al

    # -- Volume relaxation using ASE with Vasp as force/stress calculator
    def ase_vol_relax():
        Al = bulk('Al', 'fcc', a=4.5, cubic=True)
        calc = factory.calc(xc='LDA')
        Al.calc = calc

        sf = StrainFilter(Al)
        with BFGS(sf, logfile='relaxation.log') as qn:
            qn.run(fmax=0.1, steps=5)

        print('Stress:\n', calc.read_stress())
        print('Al post ASE volume relaxation\n', calc.get_atoms().get_cell())

        return Al

    # Test function for comparing two cells
    def cells_almost_equal(cellA, cellB, tol=0.01):
        return (np.abs(cellA - cellB) < tol).all()

    # Correct LDA relaxed cell
    a_rel = 4.18
    LDA_cell = np.diag([a_rel, a_rel, a_rel])

    Al_vasp = vasp_vol_relax()
    Al_ase = ase_vol_relax()

    assert cells_almost_equal(LDA_cell, Al_vasp.get_cell())
    assert cells_almost_equal(LDA_cell, Al_ase.get_cell())

    # Cleanup
    Al_ase.calc.clean()
