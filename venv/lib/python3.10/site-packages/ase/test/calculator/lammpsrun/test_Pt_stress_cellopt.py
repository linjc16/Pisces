import numpy as np
import pytest
from numpy.testing import assert_allclose

from ase.build import bulk
from ase.calculators.fd import calculate_numerical_stress
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS


@pytest.mark.calculator_lite()
@pytest.mark.calculator('lammpsrun')
def test_Pt_stress_cellopt(factory):
    params = {}
    params['pair_style'] = 'eam'
    params['pair_coeff'] = ['1 1 Pt_u3.eam']
    files = [f'{factory.factory.potentials_path}/Pt_u3.eam']
    # XXX Should it accept Path objects?  Yes definitely for files.
    with factory.calc(specorder=['Pt'], files=files, **params) as calc:
        rng = np.random.RandomState(17)

        atoms = bulk('Pt') * (2, 2, 2)
        atoms.rattle(stdev=0.1)
        atoms.cell += 2 * rng.random((3, 3))
        atoms.calc = calc

        assert_allclose(atoms.get_stress(),
                        calculate_numerical_stress(atoms),
                        atol=1e-4, rtol=1e-4)

        with BFGS(FrechetCellFilter(atoms)) as opt:
            for i, _ in enumerate(opt.irun(fmax=0.001)):
                pass

        cell1_ref = np.array([
            [0.178351, 3.885347, 3.942046],
            [4.19978, 0.591071, 5.062568],
            [4.449044, 3.264038, 0.471548],
        ])

        assert_allclose(np.asarray(atoms.cell), cell1_ref,
                        atol=3e-4, rtol=3e-4)
        assert_allclose(atoms.get_stress(),
                        calculate_numerical_stress(atoms),
                        atol=1e-4, rtol=1e-4)
