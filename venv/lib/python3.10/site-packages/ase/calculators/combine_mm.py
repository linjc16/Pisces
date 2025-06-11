# fmt: off

import copy

import numpy as np

from ase import units
from ase.calculators.calculator import Calculator
from ase.calculators.qmmm import combine_lj_lorenz_berthelot

k_c = units.Hartree * units.Bohr


class CombineMM(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, idx, apm1, apm2, calc1, calc2,
                 sig1, eps1, sig2, eps2, rc=7.0, width=1.0):
        """A calculator that combines two MM calculators
        (TIPnP, Counterions, ...)

        parameters:

        idx: List of indices of atoms belonging to calculator 1
        apm1,2: atoms pr molecule of each subsystem (NB: apm for TIP4P is 3!)
        calc1,2: calculator objects for each subsystem
        sig1,2, eps1,2: LJ parameters for each subsystem. Should be a numpy
                        array of length = apm
        rc = long range cutoff
        width = width of cutoff region.

        Currently the interactions are limited to being:
        - Nonbonded
        - Hardcoded to two terms:
            - Coulomb electrostatics
            - Lennard-Jones

        It could of course benefit from being more like the EIQMMM class
        where the interactions are switchable. But this is in princple
        just meant for adding counter ions to a qmmm simulation to neutralize
        the charge of the total systemn

        Maybe it can combine n MM calculators in the future?
        """

        self.idx = idx
        self.apm1 = apm1  # atoms per mol for LJ calculator
        self.apm2 = apm2

        self.rc = rc
        self.width = width

        self.atoms1 = None
        self.atoms2 = None
        self.mask = None

        self.calc1 = calc1
        self.calc2 = calc2

        self.sig1 = sig1
        self.eps1 = eps1
        self.sig2 = sig2
        self.eps2 = eps2

        Calculator.__init__(self)

    def initialize(self, atoms):
        self.mask = np.zeros(len(atoms), bool)
        self.mask[self.idx] = True

        constraints = atoms.constraints
        atoms.constraints = []
        self.atoms1 = atoms[self.mask]
        self.atoms2 = atoms[~self.mask]

        atoms.constraints = constraints

        self.atoms1.calc = self.calc1
        self.atoms2.calc = self.calc2

        self.cell = atoms.cell
        self.pbc = atoms.pbc

        self.sigma, self.epsilon =\
            combine_lj_lorenz_berthelot(self.sig1, self.sig2,
                                        self.eps1, self.eps2)

        self.make_virtual_mask()

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        if self.atoms1 is None:
            self.initialize(atoms)

        pos1 = atoms.positions[self.mask]
        pos2 = atoms.positions[~self.mask]
        self.atoms1.set_positions(pos1)
        self.atoms2.set_positions(pos2)

        # positions and charges for the coupling term, which should
        # include virtual charges and sites:
        spm1 = self.atoms1.calc.sites_per_mol
        spm2 = self.atoms2.calc.sites_per_mol
        xpos1 = self.atoms1.calc.add_virtual_sites(pos1)
        xpos2 = self.atoms2.calc.add_virtual_sites(pos2)

        xc1 = self.atoms1.calc.get_virtual_charges(self.atoms1)
        xc2 = self.atoms2.calc.get_virtual_charges(self.atoms2)

        xpos1 = xpos1.reshape((-1, spm1, 3))
        xpos2 = xpos2.reshape((-1, spm2, 3))

        e_c, f_c = self.coulomb(xpos1, xpos2, xc1, xc2, spm1, spm2)

        e_lj, f1, f2 = self.lennard_jones(self.atoms1, self.atoms2)

        f_lj = np.zeros((len(atoms), 3))
        f_lj[self.mask] += f1
        f_lj[~self.mask] += f2

        # internal energy, forces of each subsystem:
        f12 = np.zeros((len(atoms), 3))
        e1 = self.atoms1.get_potential_energy()
        fi1 = self.atoms1.get_forces()

        e2 = self.atoms2.get_potential_energy()
        fi2 = self.atoms2.get_forces()

        f12[self.mask] += fi1
        f12[~self.mask] += fi2

        self.results['energy'] = e_c + e_lj + e1 + e2
        self.results['forces'] = f_c + f_lj + f12

    def get_virtual_charges(self, atoms):
        if self.atoms1 is None:
            self.initialize(atoms)

        vc1 = self.atoms1.calc.get_virtual_charges(atoms[self.mask])
        vc2 = self.atoms2.calc.get_virtual_charges(atoms[~self.mask])
        # Need to expand mask with possible new virtual sites.
        # Virtual sites should ALWAYS be put AFTER actual atoms, like in
        # TIP4P: OHHX, OHHX, ...

        vc = np.zeros(len(vc1) + len(vc2))
        vc[self.virtual_mask] = vc1
        vc[~self.virtual_mask] = vc2

        return vc

    def add_virtual_sites(self, positions):
        vs1 = self.atoms1.calc.add_virtual_sites(positions[self.mask])
        vs2 = self.atoms2.calc.add_virtual_sites(positions[~self.mask])
        vs = np.zeros((len(vs1) + len(vs2), 3))

        vs[self.virtual_mask] = vs1
        vs[~self.virtual_mask] = vs2

        return vs

    def make_virtual_mask(self):
        virtual_mask = []
        ct1 = 0
        ct2 = 0
        for i in range(len(self.mask)):
            virtual_mask.append(self.mask[i])
            if self.mask[i]:
                ct1 += 1
            if not self.mask[i]:
                ct2 += 1
            if ((ct2 == self.apm2) &
                    (self.apm2 != self.atoms2.calc.sites_per_mol)):
                virtual_mask.append(False)
                ct2 = 0
            if ((ct1 == self.apm1) &
                    (self.apm1 != self.atoms1.calc.sites_per_mol)):
                virtual_mask.append(True)
                ct1 = 0

        self.virtual_mask = np.array(virtual_mask)

    def coulomb(self, xpos1, xpos2, xc1, xc2, spm1, spm2):
        energy = 0.0
        forces = np.zeros((len(xc1) + len(xc2), 3))

        self.xpos1 = xpos1
        self.xpos2 = xpos2

        R1 = xpos1
        R2 = xpos2
        F1 = np.zeros_like(R1)
        F2 = np.zeros_like(R2)
        C1 = xc1.reshape((-1, np.shape(xpos1)[1]))
        C2 = xc2.reshape((-1, np.shape(xpos2)[1]))
        # Vectorized evaluation is not as trivial when spm1 != spm2.
        # This is pretty inefficient, but for ~1-5 counter ions as region 1
        # it should not matter much ..
        # There is definitely room for improvements here.
        cell = self.cell.diagonal()
        for m1, (r1, c1) in enumerate(zip(R1, C1)):
            for m2, (r2, c2) in enumerate(zip(R2, C2)):
                r00 = r2[0] - r1[0]
                shift = np.zeros(3)
                for i, periodic in enumerate(self.pbc):
                    if periodic:
                        L = cell[i]
                        shift[i] = (r00[i] + L / 2.) % L - L / 2. - r00[i]
                r00 += shift

                d00 = (r00**2).sum()**0.5
                t = 1
                dtdd = 0
                if d00 > self.rc:
                    continue
                elif d00 > self.rc - self.width:
                    y = (d00 - self.rc + self.width) / self.width
                    t -= y**2 * (3.0 - 2.0 * y)
                    dtdd = r00 * 6 * y * (1.0 - y) / (self.width * d00)

                for a1 in range(spm1):
                    for a2 in range(spm2):
                        r = r2[a2] - r1[a1] + shift
                        d2 = (r**2).sum()
                        d = d2**0.5
                        e = k_c * c1[a1] * c2[a2] / d
                        energy += t * e

                        F1[m1, a1] -= t * (e / d2) * r
                        F2[m2, a2] += t * (e / d2) * r

                        F1[m1, 0] -= dtdd * e
                        F2[m2, 0] += dtdd * e

        F1 = F1.reshape((-1, 3))
        F2 = F2.reshape((-1, 3))

        # Redist forces but dont save forces in org calculators
        atoms1 = self.atoms1.copy()
        atoms1.calc = copy.copy(self.calc1)
        atoms1.calc.atoms = atoms1
        F1 = atoms1.calc.redistribute_forces(F1)
        atoms2 = self.atoms2.copy()
        atoms2.calc = copy.copy(self.calc2)
        atoms2.calc.atoms = atoms2
        F2 = atoms2.calc.redistribute_forces(F2)

        forces = np.zeros((len(self.atoms), 3))
        forces[self.mask] = F1
        forces[~self.mask] = F2
        return energy, forces

    def lennard_jones(self, atoms1, atoms2):
        pos1 = atoms1.get_positions().reshape((-1, self.apm1, 3))
        pos2 = atoms2.get_positions().reshape((-1, self.apm2, 3))

        f1 = np.zeros_like(atoms1.positions)
        f2 = np.zeros_like(atoms2.positions)
        energy = 0.0

        cell = self.cell.diagonal()
        for q, p1 in enumerate(pos1):  # molwise loop
            eps = self.epsilon
            sig = self.sigma

            R00 = pos2[:, 0] - p1[0, :]

            # cutoff from first atom of each mol
            shift = np.zeros_like(R00)
            for i, periodic in enumerate(self.pbc):
                if periodic:
                    L = cell[i]
                    shift[:, i] = (R00[:, i] + L / 2) % L - L / 2 - R00[:, i]
            R00 += shift

            d002 = (R00**2).sum(1)
            d00 = d002**0.5
            x1 = d00 > self.rc - self.width
            x2 = d00 < self.rc
            x12 = np.logical_and(x1, x2)
            y = (d00[x12] - self.rc + self.width) / self.width
            t = np.zeros(len(d00))
            t[x2] = 1.0
            t[x12] -= y**2 * (3.0 - 2.0 * y)
            dt = np.zeros(len(d00))
            dt[x12] -= 6.0 / self.width * y * (1.0 - y)
            for qa in range(len(p1)):
                if ~np.any(eps[qa, :]):
                    continue
                R = pos2 - p1[qa, :] + shift[:, None]
                d2 = (R**2).sum(2)
                c6 = (sig[qa, :]**2 / d2)**3
                c12 = c6**2
                e = 4 * eps[qa, :] * (c12 - c6)
                energy += np.dot(e.sum(1), t)
                f = t[:, None, None] * (24 * eps[qa, :] *
                                        (2 * c12 - c6) / d2)[:, :, None] * R
                f00 = - (e.sum(1) * dt / d00)[:, None] * R00
                f2 += f.reshape((-1, 3))
                f1[q * self.apm1 + qa, :] -= f.sum(0).sum(0)
                f1[q * self.apm1, :] -= f00.sum(0)
                f2[::self.apm2, :] += f00

        return energy, f1, f2

    def redistribute_forces(self, forces):
        f1 = self.calc1.redistribute_forces(forces[self.virtual_mask])
        f2 = self.calc2.redistribute_forces(forces[~self.virtual_mask])
        # and then they are back on the real atom centers so
        f = np.zeros((len(self.atoms), 3))
        f[self.mask] = f1
        f[~self.mask] = f2
        return f
