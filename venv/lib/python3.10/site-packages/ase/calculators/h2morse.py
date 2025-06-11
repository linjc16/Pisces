# fmt: off

from itertools import count

import numpy as np

from ase import Atoms
from ase.calculators.calculator import all_changes
from ase.calculators.excitation_list import Excitation, ExcitationList
from ase.calculators.morse import MorsePotential
from ase.data import atomic_masses
from ase.units import Ha, invcm

"""The H2 molecule represented by Morse-Potentials for
gound and first 3 excited singlet states B + C(doubly degenerate)"""

npa = np.array
# data from:
# https://webbook.nist.gov/cgi/cbook.cgi?ID=C1333740&Mask=1000#Diatomic
#         X        B       C       C
Re = npa([0.74144, 1.2928, 1.0327, 1.0327])  # eq. bond length
ome = npa([4401.21, 1358.09, 2443.77, 2443.77])  # vibrational frequency
# electronic transition energy
Etrans = npa([0, 91700.0, 100089.9, 100089.9]) * invcm

# dissociation energy
# GS: https://aip.scitation.org/doi/10.1063/1.3120443
De = np.ones(4) * 36118.069 * invcm
# B, C separated energy E(1s) - E(2p)
De[1:] += Ha / 2 - Ha / 8
De -= Etrans

# Morse parameter
m = atomic_masses[1] * 0.5  # reduced mass
# XXX find scaling factor
rho0 = Re * ome * invcm * np.sqrt(m / 2 / De) * 4401.21 / 284.55677429605862


def H2Morse(state=0):
    """Return H2 as a Morse-Potential with calculator attached."""
    atoms = Atoms('H2', positions=np.zeros((2, 3)))
    atoms[1].position[2] = Re[state]
    atoms.calc = H2MorseCalculator(state=state)
    atoms.get_potential_energy()
    return atoms


class H2MorseCalculator(MorsePotential):
    """H2 ground or excited state as Morse potential"""
    _count = count(0)

    def __init__(self, restart=None, state=0, rng=np.random):
        self.rng = rng
        MorsePotential.__init__(self,
                                restart=restart,
                                epsilon=De[state],
                                r0=Re[state], rho0=rho0[state])

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        if atoms is not None:
            assert len(atoms) == 2
        MorsePotential.calculate(self, atoms, properties, system_changes)

        # determine 'wave functions' including
        # Berry phase (arbitrary sign) and
        # random orientation of wave functions perpendicular
        # to the molecular axis

        # molecular axis
        vr = atoms[1].position - atoms[0].position
        r = np.linalg.norm(vr)
        hr = vr / r
        # perpendicular axes
        vrand = self.rng.random(3)
        hx = np.cross(hr, vrand)
        hx /= np.linalg.norm(hx)
        hy = np.cross(hr, hx)
        hy /= np.linalg.norm(hy)
        wfs = [1, hr, hx, hy]
        # Berry phase
        berry = (-1)**self.rng.randint(0, 2, 4)
        self.wfs = [wf * b for wf, b in zip(wfs, berry)]

    def read(self, filename):
        ms = self
        with open(filename) as fd:
            ms.wfs = [int(fd.readline().split()[0])]
            for _ in range(1, 4):
                ms.wfs.append(
                    np.array([float(x)
                              for x in fd.readline().split()[:4]]))
        ms.filename = filename
        return ms

    def write(self, filename, option=None):
        """write calculated state to a file"""
        with open(filename, 'w') as fd:
            fd.write(f'{self.wfs[0]}\n')
            for wf in self.wfs[1:]:
                fd.write('{:g} {:g} {:g}\n'.format(*wf))

    def overlap(self, other):
        ov = np.zeros((4, 4))
        ov[0, 0] = self.wfs[0] * other.wfs[0]
        wfs = np.array(self.wfs[1:])
        owfs = np.array(other.wfs[1:])
        ov[1:, 1:] = np.dot(wfs, owfs.T)
        return ov


class H2MorseExcitedStatesCalculator():
    """First singlet excited states of H2 from Morse potentials"""

    def __init__(self, nstates=3):
        """
        Parameters
        ----------
        nstates: int
          Numer of states to calculate 0 < nstates < 4, default 3
        """
        assert nstates > 0 and nstates < 4
        self.nstates = nstates

    def calculate(self, atoms):
        """Calculate excitation spectrum

        Parameters
        ----------
        atoms: Ase atoms object
        """
        # central me value and rise, unit Bohr
        # from DOI: 10.1021/acs.jctc.9b00584
        mc = [0, 0.8, 0.7, 0.7]
        mr = [0, 1.0, 0.5, 0.5]

        cgs = atoms.calc
        r = atoms.get_distance(0, 1)
        E0 = cgs.get_potential_energy(atoms)

        exl = H2MorseExcitedStates()
        for i in range(1, self.nstates + 1):
            hvec = cgs.wfs[0] * cgs.wfs[i]
            energy = Ha * (0.5 - 1. / 8) - E0
            calc = H2MorseCalculator(state=i)
            calc.calculate(atoms)
            energy += calc.get_potential_energy()

            mur = hvec * (mc[i] + (r - Re[0]) * mr[i])
            muv = mur

            exl.append(H2Excitation(energy, i, mur, muv))
        return exl


class H2MorseExcitedStates(ExcitationList):
    """First singlet excited states of H2"""

    def __init__(self, nstates=3):
        """
        Parameters
        ----------
        nstates: int, 1 <= nstates <= 3
          Number of excited states to consider, default 3
        """
        self.nstates = nstates
        super().__init__()

    def overlap(self, ov_nn, other):
        return (ov_nn[1:len(self) + 1, 1:len(self) + 1] *
                ov_nn[0, 0])

    @classmethod
    def read(cls, filename, nstates=3):
        """Read myself from a file"""
        exl = cls(nstates)
        with open(filename) as fd:
            exl.filename = filename
            n = int(fd.readline().split()[0])
            for _ in range(min(n, exl.nstates)):
                exl.append(H2Excitation.fromstring(fd.readline()))
        return exl

    def write(self, fname):
        with open(fname, 'w') as fd:
            fd.write(f'{len(self)}\n')
            for ex in self:
                fd.write(ex.outstring())


class H2Excitation(Excitation):
    def __eq__(self, other):
        """Considered to be equal when their indices are equal."""
        return self.index == other.index

    def __hash__(self):
        """Hash similar to __eq__"""
        if not hasattr(self, 'hash'):
            self.hash = hash(self.index)
        return self.hash


class H2MorseExcitedStatesAndCalculator(
        H2MorseExcitedStatesCalculator, H2MorseExcitedStates):
    """Traditional joined object for backward compatibility only"""

    def __init__(self, calculator, nstates=3):
        if isinstance(calculator, str):
            exlist = H2MorseExcitedStates.read(calculator, nstates)
        else:
            atoms = calculator.atoms
            atoms.calc = calculator
            excalc = H2MorseExcitedStatesCalculator(nstates)
            exlist = excalc.calculate(atoms)
        H2MorseExcitedStates.__init__(self, nstates=nstates)
        for ex in exlist:
            self.append(ex)
