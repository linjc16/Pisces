# fmt: off

import warnings
from dataclasses import dataclass

import numpy as np

spin_error = (
    'The spin keyword is no longer supported.  Please call the function '
    'with the energies corresponding to the desired spins.')
_deprecated = object()


def get_band_gap(calc, direct=False, spin=_deprecated):
    warnings.warn('Please use ase.dft.bandgap.bandgap() instead!')
    gap, (s1, k1, _n1), (s2, k2, _n2) = bandgap(calc, direct, spin=spin)
    ns = calc.get_number_of_spins()
    if ns == 2:
        return gap, (s1, k1), (s2, k2)
    return gap, k1, k2


@dataclass
class GapInfo:
    eigenvalues: np.ndarray

    def __post_init__(self):
        self._gapinfo = _bandgap(self.eigenvalues, direct=False)
        self._direct_gapinfo = _bandgap(self.eigenvalues, direct=True)

    @classmethod
    def fromcalc(cls, calc):
        kpts = calc.get_ibz_k_points()
        nk = len(kpts)
        ns = calc.get_number_of_spins()
        eigenvalues = np.array([[calc.get_eigenvalues(kpt=k, spin=s)
                                 for k in range(nk)]
                                for s in range(ns)])

        efermi = calc.get_fermi_level()
        return cls(eigenvalues - efermi)

    def gap(self):
        return self._gapinfo

    def direct_gap(self):
        return self._direct_gapinfo

    @property
    def is_metallic(self) -> bool:
        return self._gapinfo[0] == 0.0

    @property
    def gap_is_direct(self) -> bool:
        """Whether the direct and indirect gaps are the same transition."""
        return self._gapinfo[1:] == self._direct_gapinfo[1:]

    def description(self, *, ibz_kpoints=None) -> str:
        """Return human-friendly description of direct/indirect gap.

        If ibz_k_points are given, coordinates are printed as well."""
        from typing import List

        lines: List[str] = []
        add = lines.append

        def skn(skn):
            """Convert k-point indices (s, k, n) to string."""
            description = 's={}, k={}, n={}'.format(*skn)
            if ibz_kpoints is not None:
                coordtxt = '[{:.2f}, {:.2f}, {:.2f}]'.format(
                    *ibz_kpoints[skn[1]])
                description = f'{description}, {coordtxt}'
            return f'({description})'

        gap, skn1, skn2 = self.gap()
        direct_gap, skn_direct1, skn_direct2 = self.direct_gap()

        if self.is_metallic:
            add('No gap')
        else:
            add(f'Gap: {gap:.3f} eV')
            add('Transition (v -> c):')
            add(f'  {skn(skn1)} -> {skn(skn2)}')

        if self.gap_is_direct:
            add('No difference between direct/indirect transitions')
        else:
            add('Direct/indirect transitions are different')
            add(f'Direct gap: {direct_gap:.3f} eV')
            if skn_direct1[0] == skn_direct2[0]:
                add(f'Transition at: {skn(skn_direct1)}')
            else:
                transition = skn((f'{skn_direct1[0]}->{skn_direct2[0]}',
                                  *skn_direct1[1:]))
                add(f'Transition at: {transition}')

        return '\n'.join(lines)


def bandgap(calc=None, direct=False, spin=_deprecated,
            eigenvalues=None, efermi=None, output=None, kpts=None):
    """Calculates the band-gap.

    Parameters:

    calc: Calculator object
        Electronic structure calculator object.
    direct: bool
        Calculate direct band-gap.
    eigenvalues: ndarray of shape (nspin, nkpt, nband) or (nkpt, nband)
        Eigenvalues.
    efermi: float
        Fermi level (defaults to 0.0).

    Returns a (gap, p1, p2) tuple where p1 and p2 are tuples of indices of the
    valence and conduction points (s, k, n).

    Example:

    >>> gap, p1, p2 = bandgap(silicon.calc)
    >>> print(gap, p1, p2)
    1.2 (0, 0, 3), (0, 5, 4)
    >>> gap, p1, p2 = bandgap(silicon.calc, direct=True)
    >>> print(gap, p1, p2)
    3.4 (0, 0, 3), (0, 0, 4)
    """

    if spin is not _deprecated:
        raise RuntimeError(spin_error)

    if calc:
        kpts = calc.get_ibz_k_points()
        nk = len(kpts)
        ns = calc.get_number_of_spins()
        eigenvalues = np.array([[calc.get_eigenvalues(kpt=k, spin=s)
                                 for k in range(nk)]
                                for s in range(ns)])
        if efermi is None:
            efermi = calc.get_fermi_level()

    efermi = efermi or 0.0

    gapinfo = GapInfo(eigenvalues - efermi)

    e_skn = gapinfo.eigenvalues
    if eigenvalues.ndim == 2:
        e_skn = e_skn[np.newaxis]  # spinors

    if not np.isfinite(e_skn).all():
        raise ValueError('Bad eigenvalues!')

    gap, (s1, k1, n1), (s2, k2, n2) = _bandgap(e_skn, direct)

    if eigenvalues.ndim != 3:
        p1 = (k1, n1)
        p2 = (k2, n2)
    else:
        p1 = (s1, k1, n1)
        p2 = (s2, k2, n2)

    return gap, p1, p2


def _bandgap(e_skn, direct):
    """Helper function."""
    ns, nk, nb = e_skn.shape
    s1 = s2 = k1 = k2 = n1 = n2 = None

    N_sk = (e_skn < 0.0).sum(2)  # number of occupied bands

    # Check for bands crossing the fermi-level
    if ns == 1:
        if np.ptp(N_sk[0]) > 0:
            return 0.0, (None, None, None), (None, None, None)
    else:
        if (np.ptp(N_sk, axis=1) > 0).any():
            return 0.0, (None, None, None), (None, None, None)

    if (N_sk == 0).any() or (N_sk == nb).any():
        raise ValueError('Too few bands!')

    e_skn = np.array([[e_skn[s, k, N_sk[s, k] - 1:N_sk[s, k] + 1]
                       for k in range(nk)]
                      for s in range(ns)])
    ev_sk = e_skn[:, :, 0]  # valence band
    ec_sk = e_skn[:, :, 1]  # conduction band

    if ns == 1:
        s1 = 0
        s2 = 0
        gap, k1, k2 = find_gap(ev_sk[0], ec_sk[0], direct)
        n1 = N_sk[0, 0] - 1
        n2 = n1 + 1
        return gap, (0, k1, n1), (0, k2, n2)

    gap, k1, k2 = find_gap(ev_sk.ravel(), ec_sk.ravel(), direct)
    if direct:
        # Check also spin flips:
        for s in [0, 1]:
            g, k, _ = find_gap(ev_sk[s], ec_sk[1 - s], direct)
            if g < gap:
                gap = g
                k1 = k + nk * s
                k2 = k + nk * (1 - s)

    if gap > 0.0:
        s1, k1 = divmod(k1, nk)
        s2, k2 = divmod(k2, nk)
        n1 = N_sk[s1, k1] - 1
        n2 = N_sk[s2, k2]
        return gap, (s1, k1, n1), (s2, k2, n2)
    return 0.0, (None, None, None), (None, None, None)


def find_gap(ev_k, ec_k, direct):
    """Helper function."""
    if direct:
        gap_k = ec_k - ev_k
        k = gap_k.argmin()
        return gap_k[k], k, k
    kv = ev_k.argmax()
    kc = ec_k.argmin()
    return ec_k[kc] - ev_k[kv], kv, kc
