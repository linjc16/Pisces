# fmt: off

"""Soft-mutation operator and associated tools"""
import inspect
import json

import numpy as np
from scipy.spatial.distance import cdist

from ase.data import covalent_radii
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.utilities import atoms_too_close, gather_atoms_by_tag
from ase.neighborlist import NeighborList


class TagFilter:
    """Filter which constrains same-tag atoms to behave
    like internally rigid moieties.
    """

    def __init__(self, atoms):
        self.atoms = atoms
        gather_atoms_by_tag(self.atoms)
        self.tags = self.atoms.get_tags()
        self.unique_tags = np.unique(self.tags)
        self.n = len(self.unique_tags)

    def get_positions(self):
        all_pos = self.atoms.get_positions()
        cop_pos = np.zeros((self.n, 3))
        for i in range(self.n):
            indices = np.where(self.tags == self.unique_tags[i])
            cop_pos[i] = np.average(all_pos[indices], axis=0)
        return cop_pos

    def set_positions(self, positions, **kwargs):
        cop_pos = self.get_positions()
        all_pos = self.atoms.get_positions()
        assert np.all(np.shape(positions) == np.shape(cop_pos))
        for i in range(self.n):
            indices = np.where(self.tags == self.unique_tags[i])
            shift = positions[i] - cop_pos[i]
            all_pos[indices] += shift
        self.atoms.set_positions(all_pos, **kwargs)

    def get_forces(self, *args, **kwargs):
        f = self.atoms.get_forces()
        forces = np.zeros((self.n, 3))
        for i in range(self.n):
            indices = np.where(self.tags == self.unique_tags[i])
            forces[i] = np.sum(f[indices], axis=0)
        return forces

    def get_masses(self):
        m = self.atoms.get_masses()
        masses = np.zeros(self.n)
        for i in range(self.n):
            indices = np.where(self.tags == self.unique_tags[i])
            masses[i] = np.sum(m[indices])
        return masses

    def __len__(self):
        return self.n


class PairwiseHarmonicPotential:
    """Parent class for interatomic potentials of the type
    E(r_ij) = 0.5 * k_ij * (r_ij - r0_ij) ** 2
    """

    def __init__(self, atoms, rcut=10.):
        self.atoms = atoms
        self.pos0 = atoms.get_positions()
        self.rcut = rcut

        # build neighborlist
        nat = len(self.atoms)
        self.nl = NeighborList([self.rcut / 2.] * nat, skin=0., bothways=True,
                               self_interaction=False)
        self.nl.update(self.atoms)

        self.calculate_force_constants()

    def calculate_force_constants(self):
        msg = 'Child class needs to define a calculate_force_constants() ' \
              'method which computes the force constants and stores them ' \
              'in self.force_constants (as a list which contains, for every ' \
              'atom, a list of the atom\'s force constants with its neighbors.'
        raise NotImplementedError(msg)

    def get_forces(self, atoms):
        pos = atoms.get_positions()
        cell = atoms.get_cell()
        forces = np.zeros_like(pos)

        for i, p in enumerate(pos):
            indices, offsets = self.nl.get_neighbors(i)
            p = pos[indices] + np.dot(offsets, cell)
            r = cdist(p, [pos[i]])
            v = (p - pos[i]) / r
            p0 = self.pos0[indices] + np.dot(offsets, cell)
            r0 = cdist(p0, [self.pos0[i]])
            dr = r - r0
            forces[i] = np.dot(self.force_constants[i].T, dr * v)

        return forces


def get_number_of_valence_electrons(Z):
    """Return the number of valence electrons for the element with
    atomic number Z, simply based on its periodic table group.
    """
    groups = [[], [1, 3, 11, 19, 37, 55, 87], [2, 4, 12, 20, 38, 56, 88],
              [21, 39, 57, 89]]

    for i in range(9):
        groups.append(i + np.array([22, 40, 72, 104]))

    for i in range(6):
        groups.append(i + np.array([5, 13, 31, 49, 81, 113]))

    for i, group in enumerate(groups):
        if Z in group:
            nval = i if i < 13 else i - 10
            break
    else:
        raise ValueError('Z=%d not included in this dataset.' % Z)

    return nval


class BondElectroNegativityModel(PairwiseHarmonicPotential):
    """Pairwise harmonic potential where the force constants are
    determined using the "bond electronegativity" model, see:

    * `Lyakhov, Oganov, Valle, Comp. Phys. Comm. 181 (2010) 1623-1632`__

      __ https://dx.doi.org/10.1016/j.cpc.2010.06.007

    * `Lyakhov, Oganov, Phys. Rev. B 84 (2011) 092103`__

      __ https://dx.doi.org/10.1103/PhysRevB.84.092103
    """

    def calculate_force_constants(self):
        cell = self.atoms.get_cell()
        pos = self.atoms.get_positions()
        num = self.atoms.get_atomic_numbers()
        nat = len(self.atoms)
        nl = self.nl

        # computing the force constants
        s_norms = []
        valence_states = []
        r_cov = []
        for i in range(nat):
            indices, offsets = nl.get_neighbors(i)
            p = pos[indices] + np.dot(offsets, cell)
            r = cdist(p, [pos[i]])
            r_ci = covalent_radii[num[i]]
            s = 0.
            for j, index in enumerate(indices):
                d = r[j] - r_ci - covalent_radii[num[index]]
                s += np.exp(-d / 0.37)
            s_norms.append(s)
            valence_states.append(get_number_of_valence_electrons(num[i]))
            r_cov.append(r_ci)

        self.force_constants = []
        for i in range(nat):
            indices, offsets = nl.get_neighbors(i)
            p = pos[indices] + np.dot(offsets, cell)
            r = cdist(p, [pos[i]])[:, 0]
            fc = []
            for j, ii in enumerate(indices):
                d = r[j] - r_cov[i] - r_cov[ii]
                chi_ik = 0.481 * valence_states[i] / (r_cov[i] + 0.5 * d)
                chi_jk = 0.481 * valence_states[ii] / (r_cov[ii] + 0.5 * d)
                cn_ik = s_norms[i] / np.exp(-d / 0.37)
                cn_jk = s_norms[ii] / np.exp(-d / 0.37)
                fc.append(np.sqrt(chi_ik * chi_jk / (cn_ik * cn_jk)))
            self.force_constants.append(np.array(fc))


class SoftMutation(OffspringCreator):
    """Mutates the structure by displacing it along the lowest
    (nonzero) frequency modes found by vibrational analysis, as in:

    `Lyakhov, Oganov, Valle, Comp. Phys. Comm. 181 (2010) 1623-1632`__

    __ https://dx.doi.org/10.1016/j.cpc.2010.06.007

    As in the reference above, the next-lowest mode is used if the
    structure has already been softmutated along the current-lowest
    mode. This mutation hence acts in a deterministic way, in contrast
    to most other genetic operators.

    If you find this implementation useful in your work,
    please consider citing:

    `Van den Bossche, Gronbeck, Hammer, J. Chem. Theory Comput. 14 (2018)`__

    __ https://dx.doi.org/10.1021/acs.jctc.8b00039

    in addition to the paper mentioned above.

    Parameters:

    blmin: dict
        The closest allowed interatomic distances on the form:
        {(Z, Z*): dist, ...}, where Z and Z* are atomic numbers.

    bounds: list
        Lower and upper limits (in Angstrom) for the largest
        atomic displacement in the structure. For a given mode,
        the algorithm starts at zero amplitude and increases
        it until either blmin is violated or the largest
        displacement exceeds the provided upper bound).
        If the largest displacement in the resulting structure
        is lower than the provided lower bound, the mutant is
        considered too similar to the parent and None is
        returned.

    calculator: ASE calculator object
        The calculator to be used in the vibrational
        analysis. The default is to use a calculator
        based on pairwise harmonic potentials with force
        constants from the "bond electronegativity"
        model described in the reference above.
        Any calculator with a working :func:`get_forces()`
        method will work.

    rcut: float
        Cutoff radius in Angstrom for the pairwise harmonic
        potentials.

    used_modes_file: str or None
        Name of json dump file where previously used
        modes will be stored (and read). If None,
        no such file will be used. Default is to use
        the filename 'used_modes.json'.

    use_tags: boolean
        Whether to use the atomic tags to preserve molecular identity.
    """

    def __init__(self, blmin, bounds=[0.5, 2.0],
                 calculator=BondElectroNegativityModel, rcut=10.,
                 used_modes_file='used_modes.json', use_tags=False,
                 verbose=False):
        OffspringCreator.__init__(self, verbose)
        self.blmin = blmin
        self.bounds = bounds
        self.calc = calculator
        self.rcut = rcut
        self.used_modes_file = used_modes_file
        self.use_tags = use_tags
        self.descriptor = 'SoftMutation'

        self.used_modes = {}
        if self.used_modes_file is not None:
            try:
                self.read_used_modes(self.used_modes_file)
            except OSError:
                # file doesn't exist (yet)
                pass

    def _get_hessian(self, atoms, dx):
        """Returns the Hessian matrix d2E/dxi/dxj using a first-order
        central difference scheme with displacements dx.
        """
        N = len(atoms)
        pos = atoms.get_positions()
        hessian = np.zeros((3 * N, 3 * N))

        for i in range(3 * N):
            row = np.zeros(3 * N)
            for direction in [-1, 1]:
                disp = np.zeros(3)
                disp[i % 3] = direction * dx
                pos_disp = np.copy(pos)
                pos_disp[i // 3] += disp
                atoms.set_positions(pos_disp)
                f = atoms.get_forces()
                row += -1 * direction * f.flatten()

            row /= (2. * dx)
            hessian[i] = row

        hessian += np.copy(hessian).T
        hessian *= 0.5
        atoms.set_positions(pos)

        return hessian

    def _calculate_normal_modes(self, atoms, dx=0.02, massweighing=False):
        """Performs the vibrational analysis."""
        hessian = self._get_hessian(atoms, dx)
        if massweighing:
            m = np.array([np.repeat(atoms.get_masses()**-0.5, 3)])
            hessian *= (m * m.T)

        eigvals, eigvecs = np.linalg.eigh(hessian)
        modes = {eigval: eigvecs[:, i] for i, eigval in enumerate(eigvals)}
        return modes

    def animate_mode(self, atoms, mode, nim=30, amplitude=1.0):
        """Returns an Atoms object showing an animation of the mode."""
        pos = atoms.get_positions()
        mode = mode.reshape(np.shape(pos))
        animation = []
        for i in range(nim):
            newpos = pos + amplitude * mode * np.sin(i * 2 * np.pi / nim)
            image = atoms.copy()
            image.positions = newpos
            animation.append(image)
        return animation

    def read_used_modes(self, filename):
        """Read used modes from json file."""
        with open(filename) as fd:
            modes = json.load(fd)
            self.used_modes = {int(k): modes[k] for k in modes}
        return

    def write_used_modes(self, filename):
        """Dump used modes to json file."""
        with open(filename, 'w') as fd:
            json.dump(self.used_modes, fd)
        return

    def get_new_individual(self, parents):
        f = parents[0]

        indi = self.mutate(f)
        if indi is None:
            return indi, 'mutation: soft'

        indi = self.initialize_individual(f, indi)
        indi.info['data']['parents'] = [f.info['confid']]

        return self.finalize_individual(indi), 'mutation: soft'

    def mutate(self, atoms):
        """Does the actual mutation."""
        a = atoms.copy()

        if inspect.isclass(self.calc):
            assert issubclass(self.calc, PairwiseHarmonicPotential)
            calc = self.calc(atoms, rcut=self.rcut)
        else:
            calc = self.calc
        a.calc = calc

        if self.use_tags:
            a = TagFilter(a)

        pos = a.get_positions()
        modes = self._calculate_normal_modes(a)

        # Select the mode along which we want to move the atoms;
        # The first 3 translational modes as well as previously
        # applied modes are discarded.

        keys = np.array(sorted(modes))
        index = 3
        confid = atoms.info['confid']
        if confid in self.used_modes:
            while index in self.used_modes[confid]:
                index += 1
            self.used_modes[confid].append(index)
        else:
            self.used_modes[confid] = [index]

        if self.used_modes_file is not None:
            self.write_used_modes(self.used_modes_file)

        key = keys[index]
        mode = modes[key].reshape(np.shape(pos))

        # Find a suitable amplitude for translation along the mode;
        # at every trial amplitude both positive and negative
        # directions are tried.

        mutant = atoms.copy()
        amplitude = 0.
        increment = 0.1
        direction = 1
        largest_norm = np.max(np.apply_along_axis(np.linalg.norm, 1, mode))

        def expand(atoms, positions):
            if isinstance(atoms, TagFilter):
                a.set_positions(positions)
                return a.atoms.get_positions()
            else:
                return positions

        while amplitude * largest_norm < self.bounds[1]:
            pos_new = pos + direction * amplitude * mode
            pos_new = expand(a, pos_new)
            mutant.set_positions(pos_new)
            mutant.wrap()
            too_close = atoms_too_close(mutant, self.blmin,
                                        use_tags=self.use_tags)
            if too_close:
                amplitude -= increment
                pos_new = pos + direction * amplitude * mode
                pos_new = expand(a, pos_new)
                mutant.set_positions(pos_new)
                mutant.wrap()
                break

            if direction == 1:
                direction = -1
            else:
                direction = 1
                amplitude += increment

        if amplitude * largest_norm < self.bounds[0]:
            mutant = None

        return mutant
