import numpy as np

from ase import Atoms
from ase.calculators.demonnano import DemonNano
from ase.optimize import BFGS
from ase.units import Bohr, Hartree


def test_h2o(demonnano_factory):
    d = 0.9775
    t = np.pi / 180 * 110.51
    atoms = Atoms('H2O',
                  positions=[(d, 0, 0),
                             (d * np.cos(t), d * np.sin(t), 0),
                             (0, 0, 0)])

    input_arguments = {'DFTB': 'SCC',
                       'CHARGE': '0.0',
                       'PARAM': 'PTYPE=MAT'}

    calc = DemonNano(input_arguments=input_arguments)

    atoms.calc = calc

    # energy
    energy = atoms.get_potential_energy()
    ref = -4.08209 * Hartree

    print('energy')
    print(energy)

    error = np.sqrt(np.sum((energy - ref)**2))
    print('diff from reference:')
    print(error)

    tol = 1.0e-6
    assert error < tol

    # analytical forces
    forces_an = atoms.get_forces()
    ref = np.array([[0.11381E-01, -0.16761E-01, 0.0],
                    [-0.19688E-01, 0.47899E-02, 0.0],
                    [0.83062E-02, 0.11971E-01, 0.0]])

    ref *= -Hartree / Bohr

    error = np.sqrt(np.sum((forces_an - ref)**2))
    print('forces_an')
    print(forces_an)
    print('diff from reference:')
    print(error)

    tol = 1.0e-3
    assert error < tol

    # optimize geometry
    with BFGS(atoms) as dyn:
        dyn.run(fmax=0.01)

    positions = atoms.get_positions()

    ref = np.array([[0.943765, 0.046188, 0.0],
                    [-0.287409, 0.900126, 0.0],
                    [-0.021346, -0.030774, 0.0]])

    error = np.sqrt(np.sum((positions - ref)**2))
    print('positions')
    print(positions)
    print('diff from reference:')
    print(error)

    tol = 1.0e-3
    assert error < tol

    print('tests passed')
