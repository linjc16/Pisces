# fmt: off

import os
from typing import Any, Union

import numpy as np

from ase import Atoms
from ase.io.jsonio import read_json, write_json
from ase.parallel import parprint, world

DFTCalculator = Any


def ensemble(energy: float,
             contributions: np.ndarray,
             xc: str,
             verbose: bool = False) -> np.ndarray:
    """Returns an array of ensemble total energies."""
    ensemble = BEEFEnsemble(None, energy, contributions, xc, verbose)
    return ensemble.get_ensemble_energies()


class BEEFEnsemble:
    """BEEF type ensemble error estimation."""

    def __init__(self,
                 atoms: Union[Atoms, DFTCalculator] = None,
                 e: float = None,
                 contribs: np.ndarray = None,
                 xc: str = None,
                 verbose: bool = True):
        if (atoms is not None or contribs is not None or xc is not None):
            if atoms is None:
                assert e is not None
                assert contribs is not None
                assert xc is not None
            else:
                if isinstance(atoms, Atoms):
                    calc = atoms.calc
                    self.atoms = atoms
                else:
                    calc = atoms
                    self.atoms = calc.atoms
                self.calc = calc
                xc = self.calc.get_xc_functional()
            self.e = e
            self.contribs = contribs
            self.xc = xc
            self.verbose = verbose
            self.done = False
            if self.xc in ['BEEF-vdW', 'BEEF', 'PBE']:
                self.beef_type = 'beefvdw'
            elif self.xc == 'mBEEF':
                self.beef_type = 'mbeef'
            elif self.xc == 'mBEEF-vdW':
                self.beef_type = 'mbeefvdw'
            else:
                raise NotImplementedError(f'No ensemble for xc = {self.xc}')

    def get_ensemble_energies(self,
                              size: int = 2000,
                              seed: int = 0) -> np.ndarray:
        """Returns an array of ensemble total energies"""
        self.seed = seed
        if self.verbose:
            parprint(self.beef_type, 'ensemble started')

        if self.contribs is None:
            self.contribs = self.calc.get_nonselfconsistent_energies(
                self.beef_type)
            self.e = self.calc.get_potential_energy(self.atoms)
        if self.beef_type == 'beefvdw':
            assert len(self.contribs) == 32
            coefs = self.get_beefvdw_ensemble_coefs(size, seed)
        elif self.beef_type == 'mbeef':
            assert len(self.contribs) == 64
            coefs = self.get_mbeef_ensemble_coefs(size, seed)
        elif self.beef_type == 'mbeefvdw':
            assert len(self.contribs) == 28
            coefs = self.get_mbeefvdw_ensemble_coefs(size, seed)
        self.de = np.dot(coefs, self.contribs)
        self.done = True

        if self.verbose:
            parprint(self.beef_type, 'ensemble finished')

        return self.e + self.de

    def get_beefvdw_ensemble_coefs(self, size=2000, seed=0):
        """Perturbation coefficients of the BEEF-vdW ensemble"""
        from ase.dft.pars_beefvdw import uiOmega as omega
        assert np.shape(omega) == (31, 31)

        W, V, generator = self.eigendecomposition(omega, seed)
        RandV = generator.randn(31, size)

        for j in range(size):
            v = RandV[:, j]
            coefs_i = (np.dot(np.dot(V, np.diag(np.sqrt(W))), v)[:])
            if j == 0:
                ensemble_coefs = coefs_i
            else:
                ensemble_coefs = np.vstack((ensemble_coefs, coefs_i))
        PBEc_ens = -ensemble_coefs[:, 30]
        return (np.vstack((ensemble_coefs.T, PBEc_ens))).T

    def get_mbeef_ensemble_coefs(self, size=2000, seed=0):
        """Perturbation coefficients of the mBEEF ensemble"""
        from ase.dft.pars_mbeef import uiOmega as omega
        assert np.shape(omega) == (64, 64)

        W, V, generator = self.eigendecomposition(omega, seed)
        mu, sigma = 0.0, 1.0
        rand = np.array(generator.normal(mu, sigma, (len(W), size)))
        return (np.sqrt(2) * np.dot(np.dot(V, np.diag(np.sqrt(W))),
                                    rand)[:]).T

    def get_mbeefvdw_ensemble_coefs(self, size=2000, seed=0):
        """Perturbation coefficients of the mBEEF-vdW ensemble"""
        from ase.dft.pars_mbeefvdw import uiOmega as omega
        assert np.shape(omega) == (28, 28)

        W, V, generator = self.eigendecomposition(omega, seed)
        mu, sigma = 0.0, 1.0
        rand = np.array(generator.normal(mu, sigma, (len(W), size)))
        return (np.sqrt(2) * np.dot(np.dot(V, np.diag(np.sqrt(W))), rand)[:]).T

    def eigendecomposition(self, omega, seed=0):
        _u, s, v = np.linalg.svd(omega)  # unsafe: W, V = np.linalg.eig(omega)
        generator = np.random.RandomState(seed)
        return s, v.T, generator

    def write(self, fname):
        """Write ensemble data file"""
        if not fname.endswith('.bee'):
            fname += '.bee'
        assert self.done
        if world.rank == 0:
            if os.path.isfile(fname):
                os.rename(fname, fname + '.old')
            obj = [self.e, self.de, self.contribs, self.seed, self.xc]
            with open(fname, 'w') as fd:
                write_json(fd, obj)


def readbee(fname: str, all: bool = False):
    if not fname.endswith('.bee'):
        fname += '.bee'
    with open(fname) as fd:
        e, de, contribs, seed, xc = read_json(fd, always_array=False)
    if all:
        return e, de, contribs, seed, xc
    else:
        return e + de


def BEEF_Ensemble(*args, **kwargs):
    import warnings
    warnings.warn('Please use BEEFEnsemble instead of BEEF_Ensemble.')
    return BEEFEnsemble(*args, **kwargs)
