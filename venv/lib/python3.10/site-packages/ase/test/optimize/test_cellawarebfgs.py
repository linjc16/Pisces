import numpy as np
import pytest

from ase.build import bulk, fcc110
from ase.calculators.emt import EMT
from ase.filters import FrechetCellFilter, UnitCellFilter
from ase.optimize import BFGS
from ase.optimize.cellawarebfgs import CellAwareBFGS
from ase.stress import get_elasticity_tensor
from ase.units import GPa


def test_rattle_supercell_old():
    """The default len(atoms) to exp_cell_factor acts as a preconditioner
    and therefore makes the repeat unit cell of rattled atoms to converge
    in different number of steps.
    """
    def relax(atoms):
        atoms.calc = EMT()
        relax = BFGS(FrechetCellFilter(atoms), alpha=70)
        relax.run(fmax=0.05)
        return relax.nsteps

    atoms = bulk('Au')
    atoms *= (2, 1, 1)
    atoms.rattle(0.05)
    nsteps = relax(atoms.copy())
    atoms *= (2, 1, 1)
    nsteps2 = relax(atoms.copy())
    assert nsteps != nsteps2


def relax(atoms):
    atoms.calc = EMT()
    relax = CellAwareBFGS(FrechetCellFilter(atoms, exp_cell_factor=1.0),
                          alpha=70, long_output=True)
    relax.run(fmax=0.005, smax=0.00005)
    return relax.nsteps


def test_rattle_supercell():
    """Make sure that relaxing a rattled cell converges in the same number
    of iterations than a corresponding supercell with CellAwareBFGS.
    """
    atoms = bulk('Au')
    atoms *= (2, 1, 1)
    atoms.rattle(0.05)
    nsteps = relax(atoms.copy())
    atoms *= (2, 1, 1)
    nsteps2 = relax(atoms.copy())
    assert nsteps == nsteps2


def test_two_stage_relaxation():
    """Make sure that we can split relaxation in two stages and relax the
    structure in the same number of steps.
    """
    atoms = bulk('Au')
    atoms *= (2, 1, 1)
    atoms.rattle(0.05)
    # Perform full_relaxation
    nsteps = relax(atoms.copy())
    # Perform relaxation in steps
    atoms.calc = EMT()
    optimizer = CellAwareBFGS(FrechetCellFilter(atoms, exp_cell_factor=1.0),
                              alpha=70, long_output=True)
    optimizer.run(fmax=0.005, smax=0.00005, steps=5)
    assert optimizer.nsteps == 5
    optimizer.run(fmax=0.005, smax=0.00005)
    assert nsteps == optimizer.nsteps


@pytest.mark.parametrize('filt', [FrechetCellFilter, UnitCellFilter])
def test_cellaware_bfgs_2d(filt):
    """Make sure that the mask works with CellAwareBFGS
    by requiring that cell vectors on suppressed col and row remain
    unchanged.
    """
    atoms = fcc110('Au', size=(2, 2, 3), vacuum=4)
    orig_cell = atoms.cell.copy()
    atoms.cell = atoms.cell @ np.array([[1.0, 0.05, 0],
                                        [0.0, 1.0, 0.0],
                                        [0.0, 0.0, 1.0]])
    atoms.calc = EMT()
    if filt == FrechetCellFilter:
        dct = dict(exp_cell_factor=1.0)
    else:
        dct = dict(cell_factor=1.0)
    relax = CellAwareBFGS(filt(atoms, mask=[1, 1, 0, 0, 0, 1], **dct),
                          alpha=70, long_output=True)
    relax.run(fmax=0.05)
    assert np.allclose(atoms.cell[2, :], orig_cell[2, :])
    assert np.allclose(atoms.cell[:, 2], orig_cell[:, 2])


def test_cellaware_bfgs():
    """Make sure that a supercell relaxes in same number of steps as the
    unit cell with CellAwareBFGS.
    """
    steps = []
    for scale in [1, 2]:
        atoms = bulk('Au')
        atoms *= scale
        atoms.calc = EMT()
        relax = CellAwareBFGS(FrechetCellFilter(atoms, exp_cell_factor=1.0),
                              alpha=70, long_output=True)
        relax.run()
        steps.append(relax.nsteps)
    assert steps[0] == steps[1]


def test_elasticity_tensor():
    """Calculate the exact elasticity tensor. Create an optimizer with
    that exact hessian, and deform it slightly and verify that within
    the quadratic reqion, it only takes one step to get back.

    Also verify, that we really set rotation_supression eigenvalues
    to alpha, and that CellAwareBFGS can approximatily build that exact
    Hessian within 10% tolerance.
    """
    atoms = bulk('Au')
    atoms *= 2
    atoms.calc = EMT()
    relax(atoms)
    C_ijkl = get_elasticity_tensor(atoms, verbose=True)

    # d = 0.01
    # deformation = np.eye(3) + d * (rng.random((3, 3)) - 0.5)
    deformation = np.array([[9.99163386e-01, -8.49034327e-04, -3.1448271e-03],
                            [3.25727960e-03, 9.98723923e-01, 2.76098324e-03],
                            [9.85751768e-04, 4.61517003e-03, 9.95895994e-01]])
    atoms.set_cell(atoms.get_cell() @ deformation, scale_atoms=True)

    def ExactHessianBFGS(atoms, C_ijkl, alpha=70):
        atoms_and_cell = FrechetCellFilter(atoms, exp_cell_factor=1.0)
        relax = CellAwareBFGS(atoms_and_cell, alpha=70, long_output=True)
        C_ijkl = C_ijkl.copy()
        # Supplement the tensor with suppression of pure rotations
        # which are right now 0 eigenvalues. Loop over all basis
        # vectors of skew symmetric real matrix.
        for i, j in ((0, 1), (0, 2), (1, 2)):
            Q = np.zeros((3, 3))
            Q[i, j], Q[j, i] = 1, -1
            C_ijkl += np.einsum('ij,kl->ijkl', Q, Q) * alpha / 2
        relax.H0[-9:, -9:] = C_ijkl.reshape((9, 9)) * atoms.cell.volume
        return relax

    rlx = ExactHessianBFGS(atoms, C_ijkl)
    rlx.run(fmax=0.05, smax=0.005)
    assert rlx.nsteps == 1

    # Make sure we can approximate the elasticity tensor within 10%
    # using the CellAwareBFGS
    tmp = CellAwareBFGS(FrechetCellFilter(atoms, exp_cell_factor=1.0),
                        bulk_modulus=175 * GPa, poisson_ratio=0.46)
    for a, b in zip(rlx.H0[-9:, -9:].ravel(), tmp.H0[-9:, -9:].ravel()):
        if abs(a) > 0.001:
            print(a, b)
            assert np.abs((a - b) / a) < 0.1

    # Make sure we know how to add exactly alpha to
    # the rotation supression eigenvalues (multiplied by volume).
    eigs, _ = np.linalg.eigh(tmp.H0[-9:, -9:])
    assert np.sum(np.abs(eigs - 70 * atoms.cell.volume) < 1e-3) == 3
