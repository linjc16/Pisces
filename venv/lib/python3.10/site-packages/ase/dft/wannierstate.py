# fmt: off

import numpy as np
from scipy.linalg import qr


def random_orthogonal_matrix(dim, rng=np.random, real=False):
    """Generate uniformly distributed random orthogonal matrices"""
    if real:
        from scipy.stats import special_ortho_group
        ortho_m = special_ortho_group.rvs(dim=dim, random_state=rng)
    else:
        # The best method but not supported on old systems
        # from scipy.stats import unitary_group
        # ortho_m = unitary_group.rvs(dim=dim, random_state=rng)

        # Alternative method from https://stackoverflow.com/questions/38426349
        H = rng.random((dim, dim))
        Q, R = qr(H)
        ortho_m = Q @ np.diag(np.sign(np.diag(R)))

    return ortho_m


def _empty():
    return np.empty(0, complex)


class WannierSpec:
    def __init__(self, Nk, Nw, Nb, fixedstates_k):
        self.Nk = Nk
        self.Nw = Nw
        self.Nb = Nb
        self.fixedstates_k = fixedstates_k

    def _zeros(self):
        return np.zeros((self.Nk, self.Nw, self.Nw), complex)

    def bloch(self, edf_k):
        U_kww = self._zeros()
        C_kul = []
        for U, M, L in zip(U_kww, self.fixedstates_k, edf_k):
            U[:] = np.identity(self.Nw, complex)
            if L > 0:
                C_kul.append(np.identity(self.Nb - M, complex)[:, :L])
            else:
                C_kul.append(_empty())
        return WannierState(C_kul, U_kww)

    def random(self, rng, edf_k):
        # Set U and C to random (orthogonal) matrices
        U_kww = self._zeros()
        C_kul = []
        for U, M, L in zip(U_kww, self.fixedstates_k, edf_k):
            U[:] = random_orthogonal_matrix(self.Nw, rng, real=False)
            if L > 0:
                C_kul.append(random_orthogonal_matrix(
                    self.Nb - M, rng=rng, real=False)[:, :L])
            else:
                C_kul.append(_empty())
        return WannierState(C_kul, U_kww)

    def initial_orbitals(self, calc, orbitals, kptgrid, edf_k, spin):
        C_kul, U_kww = calc.initial_wannier(
            orbitals, kptgrid, self.fixedstates_k, edf_k, spin, self.Nb)
        return WannierState(C_kul, U_kww)

    def initial_wannier(self, calc, method, kptgrid, edf_k, spin):
        C_kul, U_kww = calc.initial_wannier(
            method, kptgrid, self.fixedstates_k,
            edf_k, spin, self.Nb)
        return WannierState(C_kul, U_kww)

    def scdm(self, calc, kpt_kc, spin):
        from ase.dft.wannier import scdm

        # get the size of the grid and check if there are Nw bands:
        ps = calc.get_pseudo_wave_function(band=self.Nw,
                                           kpt=0, spin=0)
        Ng = ps.size
        pseudo_nkG = np.zeros((self.Nb, self.Nk, Ng),
                              dtype=np.complex128)
        for k in range(self.Nk):
            for n in range(self.Nb):
                pseudo_nkG[n, k] = \
                    calc.get_pseudo_wave_function(
                        band=n, kpt=k, spin=spin).ravel()

        # Use initial guess to determine U and C
        C_kul, U_kww = scdm(pseudo_nkG,
                            kpts=kpt_kc,
                            fixed_k=self.fixedstates_k,
                            Nw=self.Nw)
        return WannierState(C_kul, U_kww)


class WannierState:
    def __init__(self, C_kul, U_kww):
        # Number of u is not always the same, so C_kul is ragged
        self.C_kul = [C_ul.astype(complex) for C_ul in C_kul]
        self.U_kww = U_kww
