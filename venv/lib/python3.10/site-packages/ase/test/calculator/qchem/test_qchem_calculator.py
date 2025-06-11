import numpy as np

from ase.build import molecule
from ase.calculators.qchem import QChem
from ase.optimize import LBFGS


def test_qchem_calculator(qchem_factory):
    mol = molecule('C2H6')
    calc = QChem(label='calc/ethane',
                 method='B3LYP',
                 basis='6-31+G*')

    mol.calc = calc
    # Check energy and forces
    np.testing.assert_allclose(mol.get_potential_energy(), -2172.379183703419,
                               atol=10.)
    np.testing.assert_allclose(
        mol.get_forces(),
        np.array([[0., 0.00240141, 0.04992568],
                  [-0., -0.00240141, -0.04992568],
                  [-0., 0.11626015, 0.07267481],
                  [-0.10132204, -0.05804009, 0.07538475],
                  [0.10132204, -0.05804009, 0.07538475],
                  [-0., -0.11626015, -0.07267481],
                  [-0.10132204, 0.05804009, -0.07538475],
                  [0.10132204, 0.05804009, -0.07538475]]),
        atol=0.05)

    with LBFGS(mol) as opt:
        opt.run()
    assert opt.converged()
