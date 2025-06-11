# fmt: off

import time
from typing import IO, Optional, Union

import numpy as np

from ase import Atoms
from ase.geometry import cell_to_cellpar
from ase.optimize import BFGS
from ase.optimize.optimize import Dynamics
from ase.units import GPa


def calculate_isotropic_elasticity_tensor(bulk_modulus, poisson_ratio,
                                          suppress_rotation=0):
    """
    Parameters:
        bulk_modulus Bulk Modulus of the isotropic system used to set up the
                     Hessian (in ASE units (eV/Å^3)).

        poisson_ratio Poisson ratio of the isotropic system used to set up the
                      initial Hessian (unitless, between -1 and 0.5).

        suppress_rotation The rank-2 matrix C_ijkl.reshape((9,9)) has by
                          default 6 non-zero eigenvalues, because energy is
                          invariant to orthonormal rotations of the cell
                          vector. This serves as a bad initial Hessian due to 3
                          zero eigenvalues. Suppress rotation sets a value for
                          those zero eigenvalues.

           Returns C_ijkl
    """

    # https://scienceworld.wolfram.com/physics/LameConstants.html
    _lambda = 3 * bulk_modulus * poisson_ratio / (1 + 1 * poisson_ratio)
    _mu = _lambda * (1 - 2 * poisson_ratio) / (2 * poisson_ratio)

    # https://en.wikipedia.org/wiki/Elasticity_tensor
    g_ij = np.eye(3)

    # Construct 4th rank Elasticity tensor for isotropic systems
    C_ijkl = _lambda * np.einsum('ij,kl->ijkl', g_ij, g_ij)
    C_ijkl += _mu * (np.einsum('ik,jl->ijkl', g_ij, g_ij) +
                     np.einsum('il,kj->ijkl', g_ij, g_ij))

    # Supplement the tensor with suppression of pure rotations that are right
    # now 0 eigenvalues.
    # Loop over all basis vectors of skew symmetric real matrix
    for i, j in ((0, 1), (0, 2), (1, 2)):
        Q = np.zeros((3, 3))
        Q[i, j], Q[j, i] = 1, -1
        C_ijkl += (np.einsum('ij,kl->ijkl', Q, Q)
                   * suppress_rotation / 2)

    return C_ijkl


class CellAwareBFGS(BFGS):
    def __init__(
        self,
        atoms: Atoms,
        restart: Optional[str] = None,
        logfile: Union[IO, str] = '-',
        trajectory: Optional[str] = None,
        append_trajectory: bool = False,
        maxstep: Optional[float] = None,
        bulk_modulus: Optional[float] = 145 * GPa,
        poisson_ratio: Optional[float] = 0.3,
        alpha: Optional[float] = None,
        long_output: Optional[bool] = False,
        **kwargs,
    ):
        self.bulk_modulus = bulk_modulus
        self.poisson_ratio = poisson_ratio
        self.long_output = long_output
        BFGS.__init__(self, atoms=atoms, restart=restart, logfile=logfile,
                      trajectory=trajectory, maxstep=maxstep,
                      alpha=alpha, append_trajectory=append_trajectory,
                      **kwargs)
        assert not isinstance(atoms, Atoms)
        if hasattr(atoms, 'exp_cell_factor'):
            assert atoms.exp_cell_factor == 1.0

    def initialize(self):
        BFGS.initialize(self)
        C_ijkl = calculate_isotropic_elasticity_tensor(
            self.bulk_modulus,
            self.poisson_ratio,
            suppress_rotation=self.alpha)
        cell_H = self.H0[-9:, -9:]
        ind = np.where(self.atoms.mask.ravel() != 0)[0]
        cell_H[np.ix_(ind, ind)] = C_ijkl.reshape((9, 9))[
            np.ix_(ind, ind)] * self.atoms.atoms.cell.volume

    def converged(self, forces=None):
        if forces is None:
            forces = self.atoms.atoms.get_forces()
        stress = self.atoms.atoms.get_stress(voigt=False) * self.atoms.mask
        return np.max(np.sum(forces**2, axis=1))**0.5 < self.fmax and \
            np.max(np.abs(stress)) < self.smax

    def run(self, fmax=0.05, smax=0.005, steps=None):
        """ call Dynamics.run and keep track of fmax"""
        self.fmax = fmax
        self.smax = smax
        if steps is not None:
            return Dynamics.run(self, steps=steps)
        return Dynamics.run(self)

    def log(self, forces=None):
        if forces is None:
            forces = self.atoms.atoms.get_forces()
        fmax = (forces ** 2).sum(axis=1).max() ** 0.5
        e = self.optimizable.get_potential_energy()
        T = time.localtime()
        smax = abs(self.atoms.atoms.get_stress(voigt=False) *
                   self.atoms.mask).max()
        volume = self.atoms.atoms.cell.volume
        if self.logfile is not None:
            name = self.__class__.__name__
            if self.nsteps == 0:
                args = (" " * len(name),
                        "Step", "Time", "Energy", "fmax", "smax", "volume")
                msg = "\n%s  %4s %8s %15s  %15s %15s %15s" % args
                if self.long_output:
                    msg += ("%8s %8s %8s %8s %8s %8s" %
                            ('A', 'B', 'C', 'α', 'β', 'γ'))
                msg += '\n'
                self.logfile.write(msg)

            ast = ''
            args = (name, self.nsteps, T[3], T[4], T[5], e, ast, fmax, smax,
                    volume)
            msg = ("%s:  %3d %02d:%02d:%02d %15.6f%1s %15.6f %15.6f %15.6f" %
                   args)
            if self.long_output:
                msg += ("%8.3f %8.3f %8.3f %8.3f %8.3f %8.3f" %
                        tuple(cell_to_cellpar(self.atoms.atoms.cell)))
            msg += '\n'
            self.logfile.write(msg)

            self.logfile.flush()
