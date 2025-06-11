# fmt: off

"""Constraints"""
from typing import Sequence
from warnings import warn

import numpy as np

from ase import Atoms
from ase.filters import ExpCellFilter as ExpCellFilterOld
from ase.filters import Filter as FilterOld
from ase.filters import StrainFilter as StrainFilterOld
from ase.filters import UnitCellFilter as UnitCellFilterOld
from ase.geometry import (
    conditional_find_mic,
    find_mic,
    get_angles,
    get_angles_derivatives,
    get_dihedrals,
    get_dihedrals_derivatives,
    get_distances_derivatives,
    wrap_positions,
)
from ase.spacegroup.symmetrize import (
    prep_symmetry,
    refine_symmetry,
    symmetrize_rank1,
    symmetrize_rank2,
)
from ase.stress import full_3x3_to_voigt_6_stress, voigt_6_to_full_3x3_stress
from ase.utils import deprecated
from ase.utils.parsemath import eval_expression

__all__ = [
    'FixCartesian', 'FixBondLength', 'FixedMode',
    'FixAtoms', 'FixScaled', 'FixCom', 'FixSubsetCom', 'FixedPlane',
    'FixConstraint', 'FixedLine', 'FixBondLengths', 'FixLinearTriatomic',
    'FixInternals', 'Hookean', 'ExternalForce', 'MirrorForce', 'MirrorTorque',
    'FixScaledParametricRelations', 'FixCartesianParametricRelations',
    'FixSymmetry']


def dict2constraint(dct):
    if dct['name'] not in __all__:
        raise ValueError
    return globals()[dct['name']](**dct['kwargs'])


def slice2enlist(s, n):
    """Convert a slice object into a list of (new, old) tuples."""
    if isinstance(s, slice):
        return enumerate(range(*s.indices(n)))
    return enumerate(s)


def constrained_indices(atoms, only_include=None):
    """Returns a list of indices for the atoms that are constrained
    by a constraint that is applied.  By setting only_include to a
    specific type of constraint you can make it only look for that
    given constraint.
    """
    indices = []
    for constraint in atoms.constraints:
        if only_include is not None:
            if not isinstance(constraint, only_include):
                continue
        indices.extend(np.array(constraint.get_indices()))
    return np.array(np.unique(indices))


class FixConstraint:
    """Base class for classes that fix one or more atoms in some way."""

    def index_shuffle(self, atoms: Atoms, ind):
        """Change the indices.

        When the ordering of the atoms in the Atoms object changes,
        this method can be called to shuffle the indices of the
        constraints.

        ind -- List or tuple of indices.

        """
        raise NotImplementedError

    def repeat(self, m: int, n: int):
        """ basic method to multiply by m, needs to know the length
        of the underlying atoms object for the assignment of
        multiplied constraints to work.
        """
        msg = ("Repeat is not compatible with your atoms' constraints."
               ' Use atoms.set_constraint() before calling repeat to '
               'remove your constraints.')
        raise NotImplementedError(msg)

    def get_removed_dof(self, atoms: Atoms):
        """Get number of removed degrees of freedom due to constraint."""
        raise NotImplementedError

    def adjust_positions(self, atoms: Atoms, new):
        """Adjust positions."""

    def adjust_momenta(self, atoms: Atoms, momenta):
        """Adjust momenta."""
        # The default is in identical manner to forces.
        # TODO: The default is however not always reasonable.
        self.adjust_forces(atoms, momenta)

    def adjust_forces(self, atoms: Atoms, forces):
        """Adjust forces."""

    def copy(self):
        """Copy constraint."""
        return dict2constraint(self.todict().copy())

    def todict(self):
        """Convert constraint to dictionary."""


class IndexedConstraint(FixConstraint):
    def __init__(self, indices=None, mask=None):
        """Constrain chosen atoms.

        Parameters
        ----------
        indices : sequence of int
           Indices for those atoms that should be constrained.
        mask : sequence of bool
           One boolean per atom indicating if the atom should be
           constrained or not.
        """

        if mask is not None:
            if indices is not None:
                raise ValueError('Use only one of "indices" and "mask".')
            indices = mask
        indices = np.atleast_1d(indices)
        if np.ndim(indices) > 1:
            raise ValueError('indices has wrong amount of dimensions. '
                             f'Got {np.ndim(indices)}, expected ndim <= 1')

        if indices.dtype == bool:
            indices = np.arange(len(indices))[indices]
        elif len(indices) == 0:
            indices = np.empty(0, dtype=int)
        elif not np.issubdtype(indices.dtype, np.integer):
            raise ValueError('Indices must be integers or boolean mask, '
                             f'not dtype={indices.dtype}')

        if len(set(indices)) < len(indices):
            raise ValueError(
                'The indices array contains duplicates. '
                'Perhaps you want to specify a mask instead, but '
                'forgot the mask= keyword.')

        self.index = indices

    def index_shuffle(self, atoms, ind):
        # See docstring of superclass
        index = []

        # Resolve negative indices:
        actual_indices = set(np.arange(len(atoms))[self.index])

        for new, old in slice2enlist(ind, len(atoms)):
            if old in actual_indices:
                index.append(new)
        if len(index) == 0:
            raise IndexError('All indices in FixAtoms not part of slice')
        self.index = np.asarray(index, int)
        # XXX make immutable

    def get_indices(self):
        return self.index.copy()

    def repeat(self, m, n):
        i0 = 0
        natoms = 0
        if isinstance(m, int):
            m = (m, m, m)
        index_new = []
        for _ in range(m[2]):
            for _ in range(m[1]):
                for _ in range(m[0]):
                    i1 = i0 + n
                    index_new += [i + natoms for i in self.index]
                    i0 = i1
                    natoms += n
        self.index = np.asarray(index_new, int)
        # XXX make immutable
        return self

    def delete_atoms(self, indices, natoms):
        """Removes atoms from the index array, if present.

        Required for removing atoms with existing constraint.
        """

        i = np.zeros(natoms, int) - 1
        new = np.delete(np.arange(natoms), indices)
        i[new] = np.arange(len(new))
        index = i[self.index]
        self.index = index[index >= 0]
        # XXX make immutable
        if len(self.index) == 0:
            return None
        return self


class FixAtoms(IndexedConstraint):
    """Fix chosen atoms.

    Examples
    --------
    Fix all Copper atoms:

    >>> from ase.build import bulk

    >>> atoms = bulk('Cu', 'fcc', a=3.6)
    >>> mask = (atoms.symbols == 'Cu')
    >>> c = FixAtoms(mask=mask)
    >>> atoms.set_constraint(c)

    Fix all atoms with z-coordinate less than 1.0 Angstrom:

    >>> c = FixAtoms(mask=atoms.positions[:, 2] < 1.0)
    >>> atoms.set_constraint(c)
    """

    def get_removed_dof(self, atoms):
        return 3 * len(self.index)

    def adjust_positions(self, atoms, new):
        new[self.index] = atoms.positions[self.index]

    def adjust_forces(self, atoms, forces):
        forces[self.index] = 0.0

    def __repr__(self):
        clsname = type(self).__name__
        indices = ints2string(self.index)
        return f'{clsname}(indices={indices})'

    def todict(self):
        return {'name': 'FixAtoms',
                'kwargs': {'indices': self.index.tolist()}}


class FixCom(FixConstraint):
    """Constraint class for fixing the center of mass."""

    index = slice(None)  # all atoms

    def get_removed_dof(self, atoms):
        return 3

    def adjust_positions(self, atoms, new):
        masses = atoms.get_masses()[self.index]
        old_cm = atoms.get_center_of_mass(indices=self.index)
        new_cm = masses @ new[self.index] / masses.sum()
        diff = old_cm - new_cm
        new += diff

    def adjust_momenta(self, atoms, momenta):
        """Adjust momenta so that the center-of-mass velocity is zero."""
        masses = atoms.get_masses()[self.index]
        velocity_com = momenta[self.index].sum(axis=0) / masses.sum()
        momenta[self.index] -= masses[:, None] * velocity_com

    def adjust_forces(self, atoms, forces):
        # Eqs. (3) and (7) in https://doi.org/10.1021/jp9722824
        masses = atoms.get_masses()[self.index]
        lmd = masses @ forces[self.index] / sum(masses**2)
        forces[self.index] -= masses[:, None] * lmd

    def todict(self):
        return {'name': 'FixCom',
                'kwargs': {}}


class FixSubsetCom(FixCom, IndexedConstraint):
    """Constraint class for fixing the center of mass of a subset of atoms."""

    def __init__(self, indices):
        super().__init__(indices=indices)

    def todict(self):
        return {'name': self.__class__.__name__,
                'kwargs': {'indices': self.index.tolist()}}


def ints2string(x, threshold=None):
    """Convert ndarray of ints to string."""
    if threshold is None or len(x) <= threshold:
        return str(x.tolist())
    return str(x[:threshold].tolist())[:-1] + ', ...]'


class FixBondLengths(FixConstraint):
    maxiter = 500

    def __init__(self, pairs, tolerance=1e-13,
                 bondlengths=None, iterations=None):
        """iterations:
                Ignored"""
        self.pairs = np.asarray(pairs)
        self.tolerance = tolerance
        self.bondlengths = bondlengths

    def get_removed_dof(self, atoms):
        return len(self.pairs)

    def adjust_positions(self, atoms, new):
        old = atoms.positions
        masses = atoms.get_masses()

        if self.bondlengths is None:
            self.bondlengths = self.initialize_bond_lengths(atoms)

        for i in range(self.maxiter):
            converged = True
            for j, ab in enumerate(self.pairs):
                a = ab[0]
                b = ab[1]
                cd = self.bondlengths[j]
                r0 = old[a] - old[b]
                d0, _ = find_mic(r0, atoms.cell, atoms.pbc)
                d1 = new[a] - new[b] - r0 + d0
                m = 1 / (1 / masses[a] + 1 / masses[b])
                x = 0.5 * (cd**2 - np.dot(d1, d1)) / np.dot(d0, d1)
                if abs(x) > self.tolerance:
                    new[a] += x * m / masses[a] * d0
                    new[b] -= x * m / masses[b] * d0
                    converged = False
            if converged:
                break
        else:
            raise RuntimeError('Did not converge')

    def adjust_momenta(self, atoms, p):
        old = atoms.positions
        masses = atoms.get_masses()

        if self.bondlengths is None:
            self.bondlengths = self.initialize_bond_lengths(atoms)

        for i in range(self.maxiter):
            converged = True
            for j, ab in enumerate(self.pairs):
                a = ab[0]
                b = ab[1]
                cd = self.bondlengths[j]
                d = old[a] - old[b]
                d, _ = find_mic(d, atoms.cell, atoms.pbc)
                dv = p[a] / masses[a] - p[b] / masses[b]
                m = 1 / (1 / masses[a] + 1 / masses[b])
                x = -np.dot(dv, d) / cd**2
                if abs(x) > self.tolerance:
                    p[a] += x * m * d
                    p[b] -= x * m * d
                    converged = False
            if converged:
                break
        else:
            raise RuntimeError('Did not converge')

    def adjust_forces(self, atoms, forces):
        self.constraint_forces = -forces
        self.adjust_momenta(atoms, forces)
        self.constraint_forces += forces

    def initialize_bond_lengths(self, atoms):
        bondlengths = np.zeros(len(self.pairs))

        for i, ab in enumerate(self.pairs):
            bondlengths[i] = atoms.get_distance(ab[0], ab[1], mic=True)

        return bondlengths

    def get_indices(self):
        return np.unique(self.pairs.ravel())

    def todict(self):
        return {'name': 'FixBondLengths',
                'kwargs': {'pairs': self.pairs.tolist(),
                           'tolerance': self.tolerance}}

    def index_shuffle(self, atoms, ind):
        """Shuffle the indices of the two atoms in this constraint"""
        map = np.zeros(len(atoms), int)
        map[ind] = 1
        n = map.sum()
        map[:] = -1
        map[ind] = range(n)
        pairs = map[self.pairs]
        self.pairs = pairs[(pairs != -1).all(1)]
        if len(self.pairs) == 0:
            raise IndexError('Constraint not part of slice')


def FixBondLength(a1, a2):
    """Fix distance between atoms with indices a1 and a2."""
    return FixBondLengths([(a1, a2)])


class FixLinearTriatomic(FixConstraint):
    """Holonomic constraints for rigid linear triatomic molecules."""

    def __init__(self, triples):
        """Apply RATTLE-type bond constraints between outer atoms n and m
           and linear vectorial constraints to the position of central
           atoms o to fix the geometry of linear triatomic molecules of the
           type:

           n--o--m

           Parameters:

           triples: list
               Indices of the atoms forming the linear molecules to constrain
               as triples. Sequence should be (n, o, m) or (m, o, n).

           When using these constraints in molecular dynamics or structure
           optimizations, atomic forces need to be redistributed within a
           triple. The function redistribute_forces_optimization implements
           the redistribution of forces for structure optimization, while
           the function redistribute_forces_md implements the redistribution
           for molecular dynamics.

           References:

           Ciccotti et al. Molecular Physics 47 (1982)
           :doi:`10.1080/00268978200100942`
        """
        self.triples = np.asarray(triples)
        if self.triples.shape[1] != 3:
            raise ValueError('"triples" has wrong size')
        self.bondlengths = None

    def get_removed_dof(self, atoms):
        return 4 * len(self.triples)

    @property
    def n_ind(self):
        return self.triples[:, 0]

    @property
    def m_ind(self):
        return self.triples[:, 2]

    @property
    def o_ind(self):
        return self.triples[:, 1]

    def initialize(self, atoms):
        masses = atoms.get_masses()
        self.mass_n, self.mass_m, self.mass_o = self.get_slices(masses)

        self.bondlengths = self.initialize_bond_lengths(atoms)
        self.bondlengths_nm = self.bondlengths.sum(axis=1)

        C1 = self.bondlengths[:, ::-1] / self.bondlengths_nm[:, None]
        C2 = (C1[:, 0] ** 2 * self.mass_o * self.mass_m +
              C1[:, 1] ** 2 * self.mass_n * self.mass_o +
              self.mass_n * self.mass_m)
        C2 = C1 / C2[:, None]
        C3 = self.mass_n * C1[:, 1] - self.mass_m * C1[:, 0]
        C3 = C2 * self.mass_o[:, None] * C3[:, None]
        C3[:, 1] *= -1
        C3 = (C3 + 1) / np.vstack((self.mass_n, self.mass_m)).T
        C4 = (C1[:, 0]**2 + C1[:, 1]**2 + 1)
        C4 = C1 / C4[:, None]

        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.C4 = C4

    def adjust_positions(self, atoms, new):
        old = atoms.positions
        new_n, new_m, new_o = self.get_slices(new)

        if self.bondlengths is None:
            self.initialize(atoms)

        r0 = old[self.n_ind] - old[self.m_ind]
        d0, _ = find_mic(r0, atoms.cell, atoms.pbc)
        d1 = new_n - new_m - r0 + d0
        a = np.einsum('ij,ij->i', d0, d0)
        b = np.einsum('ij,ij->i', d1, d0)
        c = np.einsum('ij,ij->i', d1, d1) - self.bondlengths_nm ** 2
        g = (b - (b**2 - a * c)**0.5) / (a * self.C3.sum(axis=1))
        g = g[:, None] * self.C3
        new_n -= g[:, 0, None] * d0
        new_m += g[:, 1, None] * d0
        if np.allclose(d0, r0):
            new_o = (self.C1[:, 0, None] * new_n
                     + self.C1[:, 1, None] * new_m)
        else:
            v1, _ = find_mic(new_n, atoms.cell, atoms.pbc)
            v2, _ = find_mic(new_m, atoms.cell, atoms.pbc)
            rb = self.C1[:, 0, None] * v1 + self.C1[:, 1, None] * v2
            new_o = wrap_positions(rb, atoms.cell, atoms.pbc)

        self.set_slices(new_n, new_m, new_o, new)

    def adjust_momenta(self, atoms, p):
        old = atoms.positions
        p_n, p_m, p_o = self.get_slices(p)

        if self.bondlengths is None:
            self.initialize(atoms)

        mass_nn = self.mass_n[:, None]
        mass_mm = self.mass_m[:, None]
        mass_oo = self.mass_o[:, None]

        d = old[self.n_ind] - old[self.m_ind]
        d, _ = find_mic(d, atoms.cell, atoms.pbc)
        dv = p_n / mass_nn - p_m / mass_mm
        k = np.einsum('ij,ij->i', dv, d) / self.bondlengths_nm ** 2
        k = self.C3 / (self.C3.sum(axis=1)[:, None]) * k[:, None]
        p_n -= k[:, 0, None] * mass_nn * d
        p_m += k[:, 1, None] * mass_mm * d
        p_o = (mass_oo * (self.C1[:, 0, None] * p_n / mass_nn +
                          self.C1[:, 1, None] * p_m / mass_mm))

        self.set_slices(p_n, p_m, p_o, p)

    def adjust_forces(self, atoms, forces):

        if self.bondlengths is None:
            self.initialize(atoms)

        A = self.C4 * np.diff(self.C1)
        A[:, 0] *= -1
        A -= 1
        B = np.diff(self.C4) / (A.sum(axis=1))[:, None]
        A /= (A.sum(axis=1))[:, None]

        self.constraint_forces = -forces
        old = atoms.positions

        fr_n, fr_m, fr_o = self.redistribute_forces_optimization(forces)

        d = old[self.n_ind] - old[self.m_ind]
        d, _ = find_mic(d, atoms.cell, atoms.pbc)
        df = fr_n - fr_m
        k = -np.einsum('ij,ij->i', df, d) / self.bondlengths_nm ** 2
        forces[self.n_ind] = fr_n + k[:, None] * d * A[:, 0, None]
        forces[self.m_ind] = fr_m - k[:, None] * d * A[:, 1, None]
        forces[self.o_ind] = fr_o + k[:, None] * d * B

        self.constraint_forces += forces

    def redistribute_forces_optimization(self, forces):
        """Redistribute forces within a triple when performing structure
        optimizations.

        The redistributed forces needs to be further adjusted using the
        appropriate Lagrange multipliers as implemented in adjust_forces."""
        forces_n, forces_m, forces_o = self.get_slices(forces)
        C1_1 = self.C1[:, 0, None]
        C1_2 = self.C1[:, 1, None]
        C4_1 = self.C4[:, 0, None]
        C4_2 = self.C4[:, 1, None]

        fr_n = ((1 - C4_1 * C1_1) * forces_n -
                C4_1 * (C1_2 * forces_m - forces_o))
        fr_m = ((1 - C4_2 * C1_2) * forces_m -
                C4_2 * (C1_1 * forces_n - forces_o))
        fr_o = ((1 - 1 / (C1_1**2 + C1_2**2 + 1)) * forces_o +
                C4_1 * forces_n + C4_2 * forces_m)

        return fr_n, fr_m, fr_o

    def redistribute_forces_md(self, atoms, forces, rand=False):
        """Redistribute forces within a triple when performing molecular
        dynamics.

        When rand=True, use the equations for random force terms, as
        used e.g. by Langevin dynamics, otherwise apply the standard
        equations for deterministic forces (see Ciccotti et al. Molecular
        Physics 47 (1982))."""
        if self.bondlengths is None:
            self.initialize(atoms)
        forces_n, forces_m, forces_o = self.get_slices(forces)
        C1_1 = self.C1[:, 0, None]
        C1_2 = self.C1[:, 1, None]
        C2_1 = self.C2[:, 0, None]
        C2_2 = self.C2[:, 1, None]
        mass_nn = self.mass_n[:, None]
        mass_mm = self.mass_m[:, None]
        mass_oo = self.mass_o[:, None]
        if rand:
            mr1 = (mass_mm / mass_nn) ** 0.5
            mr2 = (mass_oo / mass_nn) ** 0.5
            mr3 = (mass_nn / mass_mm) ** 0.5
            mr4 = (mass_oo / mass_mm) ** 0.5
        else:
            mr1 = 1.0
            mr2 = 1.0
            mr3 = 1.0
            mr4 = 1.0

        fr_n = ((1 - C1_1 * C2_1 * mass_oo * mass_mm) * forces_n -
                C2_1 * (C1_2 * mr1 * mass_oo * mass_nn * forces_m -
                        mr2 * mass_mm * mass_nn * forces_o))

        fr_m = ((1 - C1_2 * C2_2 * mass_oo * mass_nn) * forces_m -
                C2_2 * (C1_1 * mr3 * mass_oo * mass_mm * forces_n -
                        mr4 * mass_mm * mass_nn * forces_o))

        self.set_slices(fr_n, fr_m, 0.0, forces)

    def get_slices(self, a):
        a_n = a[self.n_ind]
        a_m = a[self.m_ind]
        a_o = a[self.o_ind]

        return a_n, a_m, a_o

    def set_slices(self, a_n, a_m, a_o, a):
        a[self.n_ind] = a_n
        a[self.m_ind] = a_m
        a[self.o_ind] = a_o

    def initialize_bond_lengths(self, atoms):
        bondlengths = np.zeros((len(self.triples), 2))

        for i in range(len(self.triples)):
            bondlengths[i, 0] = atoms.get_distance(self.n_ind[i],
                                                   self.o_ind[i], mic=True)
            bondlengths[i, 1] = atoms.get_distance(self.o_ind[i],
                                                   self.m_ind[i], mic=True)

        return bondlengths

    def get_indices(self):
        return np.unique(self.triples.ravel())

    def todict(self):
        return {'name': 'FixLinearTriatomic',
                'kwargs': {'triples': self.triples.tolist()}}

    def index_shuffle(self, atoms, ind):
        """Shuffle the indices of the three atoms in this constraint"""
        map = np.zeros(len(atoms), int)
        map[ind] = 1
        n = map.sum()
        map[:] = -1
        map[ind] = range(n)
        triples = map[self.triples]
        self.triples = triples[(triples != -1).all(1)]
        if len(self.triples) == 0:
            raise IndexError('Constraint not part of slice')


class FixedMode(FixConstraint):
    """Constrain atoms to move along directions orthogonal to
    a given mode only. Initialize with a mode, such as one produced by
    ase.vibrations.Vibrations.get_mode()."""

    def __init__(self, mode):
        mode = np.asarray(mode)
        self.mode = (mode / np.sqrt((mode**2).sum())).reshape(-1)

    def get_removed_dof(self, atoms):
        return len(atoms)

    def adjust_positions(self, atoms, newpositions):
        newpositions = newpositions.ravel()
        oldpositions = atoms.positions.ravel()
        step = newpositions - oldpositions
        newpositions -= self.mode * np.dot(step, self.mode)

    def adjust_forces(self, atoms, forces):
        forces = forces.ravel()
        forces -= self.mode * np.dot(forces, self.mode)

    def index_shuffle(self, atoms, ind):
        eps = 1e-12
        mode = self.mode.reshape(-1, 3)
        excluded = np.ones(len(mode), dtype=bool)
        excluded[ind] = False
        if (abs(mode[excluded]) > eps).any():
            raise IndexError('All nonzero parts of mode not in slice')
        self.mode = mode[ind].ravel()

    def get_indices(self):
        # This function will never properly work because it works on all
        # atoms and it has no idea how to tell how many atoms it is
        # attached to.  If it is being used, surely the user knows
        # everything is being constrained.
        return []

    def todict(self):
        return {'name': 'FixedMode',
                'kwargs': {'mode': self.mode.tolist()}}

    def __repr__(self):
        return f'FixedMode({self.mode.tolist()})'


def _normalize(direction):
    if np.shape(direction) != (3,):
        raise ValueError("len(direction) is {len(direction)}. Has to be 3")

    direction = np.asarray(direction) / np.linalg.norm(direction)
    return direction


class FixedPlane(IndexedConstraint):
    """
    Constraint object for fixing chosen atoms to only move in a plane.

    The plane is defined by its normal vector *direction*
    """

    def __init__(self, indices, direction):
        """Constrain chosen atoms.

        Parameters
        ----------
        indices : int or list of int
            Index or indices for atoms that should be constrained
        direction : list of 3 int
            Direction of the normal vector

        Examples
        --------
        Fix all Copper atoms to only move in the yz-plane:

        >>> from ase.build import bulk
        >>> from ase.constraints import FixedPlane

        >>> atoms = bulk('Cu', 'fcc', a=3.6)
        >>> c = FixedPlane(
        ...     indices=[atom.index for atom in atoms if atom.symbol == 'Cu'],
        ...     direction=[1, 0, 0],
        ... )
        >>> atoms.set_constraint(c)

        or constrain a single atom with the index 0 to move in the xy-plane:

        >>> c = FixedPlane(indices=0, direction=[0, 0, 1])
        >>> atoms.set_constraint(c)
        """
        super().__init__(indices=indices)
        self.dir = _normalize(direction)

    def adjust_positions(self, atoms, newpositions):
        step = newpositions[self.index] - atoms.positions[self.index]
        newpositions[self.index] -= _projection(step, self.dir)

    def adjust_forces(self, atoms, forces):
        forces[self.index] -= _projection(forces[self.index], self.dir)

    def get_removed_dof(self, atoms):
        return len(self.index)

    def todict(self):
        return {
            'name': 'FixedPlane',
            'kwargs': {'indices': self.index.tolist(),
                       'direction': self.dir.tolist()}
        }

    def __repr__(self):
        return f'FixedPlane(indices={self.index}, {self.dir.tolist()})'


def _projection(vectors, direction):
    dotprods = vectors @ direction
    projection = direction[None, :] * dotprods[:, None]
    return projection


class FixedLine(IndexedConstraint):
    """
    Constrain an atom index or a list of atom indices to move on a line only.

    The line is defined by its vector *direction*
    """

    def __init__(self, indices, direction):
        """Constrain chosen atoms.

        Parameters
        ----------
        indices : int or list of int
            Index or indices for atoms that should be constrained
        direction : list of 3 int
            Direction of the vector defining the line

        Examples
        --------
        Fix all Copper atoms to only move in the x-direction:

        >>> from ase.constraints import FixedLine
        >>> c = FixedLine(
        ...     indices=[atom.index for atom in atoms if atom.symbol == 'Cu'],
        ...     direction=[1, 0, 0],
        ... )
        >>> atoms.set_constraint(c)

        or constrain a single atom with the index 0 to move in the z-direction:

        >>> c = FixedLine(indices=0, direction=[0, 0, 1])
        >>> atoms.set_constraint(c)
        """
        super().__init__(indices)
        self.dir = _normalize(direction)

    def adjust_positions(self, atoms, newpositions):
        step = newpositions[self.index] - atoms.positions[self.index]
        projection = _projection(step, self.dir)
        newpositions[self.index] = atoms.positions[self.index] + projection

    def adjust_forces(self, atoms, forces):
        forces[self.index] = _projection(forces[self.index], self.dir)

    def get_removed_dof(self, atoms):
        return 2 * len(self.index)

    def __repr__(self):
        return f'FixedLine(indices={self.index}, {self.dir.tolist()})'

    def todict(self):
        return {
            'name': 'FixedLine',
            'kwargs': {'indices': self.index.tolist(),
                       'direction': self.dir.tolist()}
        }


class FixCartesian(IndexedConstraint):
    """Fix atoms in the directions of the cartesian coordinates.

    Parameters
    ----------
    a : Sequence[int]
        Indices of atoms to be fixed.
    mask : tuple[bool, bool, bool], default: (True, True, True)
        Cartesian directions to be fixed. (False: unfixed, True: fixed)
    """

    def __init__(self, a, mask=(True, True, True)):
        super().__init__(indices=a)
        self.mask = np.asarray(mask, bool)

    def get_removed_dof(self, atoms: Atoms):
        return self.mask.sum() * len(self.index)

    def adjust_positions(self, atoms: Atoms, new):
        new[self.index] = np.where(
            self.mask[None, :],
            atoms.positions[self.index],
            new[self.index],
        )

    def adjust_forces(self, atoms: Atoms, forces):
        forces[self.index] *= ~self.mask[None, :]

    def todict(self):
        return {'name': 'FixCartesian',
                'kwargs': {'a': self.index.tolist(),
                           'mask': self.mask.tolist()}}

    def __repr__(self):
        name = type(self).__name__
        return f'{name}(indices={self.index.tolist()}, {self.mask.tolist()})'


class FixScaled(IndexedConstraint):
    """Fix atoms in the directions of the unit vectors.

    Parameters
    ----------
    a : Sequence[int]
        Indices of atoms to be fixed.
    mask : tuple[bool, bool, bool], default: (True, True, True)
        Cell directions to be fixed. (False: unfixed, True: fixed)
    """

    def __init__(self, a, mask=(True, True, True), cell=None):
        # XXX The unused cell keyword is there for compatibility
        # with old trajectory files.
        super().__init__(indices=a)
        self.mask = np.asarray(mask, bool)

    def get_removed_dof(self, atoms: Atoms):
        return self.mask.sum() * len(self.index)

    def adjust_positions(self, atoms: Atoms, new):
        cell = atoms.cell
        scaled_old = cell.scaled_positions(atoms.positions[self.index])
        scaled_new = cell.scaled_positions(new[self.index])
        scaled_new[:, self.mask] = scaled_old[:, self.mask]
        new[self.index] = cell.cartesian_positions(scaled_new)

    def adjust_forces(self, atoms: Atoms, forces):
        # Forces are covariant to the coordinate transformation,
        # use the inverse transformations
        cell = atoms.cell
        scaled_forces = cell.cartesian_positions(forces[self.index])
        scaled_forces *= -(self.mask - 1)
        forces[self.index] = cell.scaled_positions(scaled_forces)

    def todict(self):
        return {'name': 'FixScaled',
                'kwargs': {'a': self.index.tolist(),
                           'mask': self.mask.tolist()}}

    def __repr__(self):
        name = type(self).__name__
        return f'{name}(indices={self.index.tolist()}, {self.mask.tolist()})'


# TODO: Better interface might be to use dictionaries in place of very
# nested lists/tuples
class FixInternals(FixConstraint):
    """Constraint object for fixing multiple internal coordinates.

    Allows fixing bonds, angles, dihedrals as well as linear combinations
    of bonds (bondcombos).

    Please provide angular units in degrees using `angles_deg` and
    `dihedrals_deg`.
    Fixing planar angles is not supported at the moment.
    """

    def __init__(self, bonds=None, angles=None, dihedrals=None,
                 angles_deg=None, dihedrals_deg=None,
                 bondcombos=None,
                 mic=False, epsilon=1.e-7):
        """
        A constrained internal coordinate is defined as a nested list:
        '[value, [atom indices]]'. The constraint is initialized with a list of
        constrained internal coordinates, i.e. '[[value, [atom indices]], ...]'.
        If 'value' is None, the current value of the coordinate is constrained.

        Parameters
        ----------
        bonds: nested python list, optional
            List with targetvalue and atom indices defining the fixed bonds,
            i.e. [[targetvalue, [index0, index1]], ...]

        angles_deg: nested python list, optional
            List with targetvalue and atom indices defining the fixedangles,
            i.e. [[targetvalue, [index0, index1, index3]], ...]

        dihedrals_deg: nested python list, optional
            List with targetvalue and atom indices defining the fixed dihedrals,
            i.e. [[targetvalue, [index0, index1, index3]], ...]

        bondcombos: nested python list, optional
            List with targetvalue, atom indices and linear coefficient defining
            the fixed linear combination of bonds,
            i.e. [[targetvalue, [[index0, index1, coefficient_for_bond],
            [index1, index2, coefficient_for_bond]]], ...]

        mic: bool, optional, default: False
            Minimum image convention.

        epsilon: float, optional, default: 1e-7
            Convergence criterion.
        """
        warn_msg = 'Please specify {} in degrees using the {} argument.'
        if angles:
            warn(warn_msg.format('angles', 'angle_deg'), FutureWarning)
            angles = np.asarray(angles)
            angles[:, 0] = angles[:, 0] / np.pi * 180
            angles = angles.tolist()
        else:
            angles = angles_deg
        if dihedrals:
            warn(warn_msg.format('dihedrals', 'dihedrals_deg'), FutureWarning)
            dihedrals = np.asarray(dihedrals)
            dihedrals[:, 0] = dihedrals[:, 0] / np.pi * 180
            dihedrals = dihedrals.tolist()
        else:
            dihedrals = dihedrals_deg

        self.bonds = bonds or []
        self.angles = angles or []
        self.dihedrals = dihedrals or []
        self.bondcombos = bondcombos or []
        self.mic = mic
        self.epsilon = epsilon

        self.n = (len(self.bonds) + len(self.angles) + len(self.dihedrals)
                  + len(self.bondcombos))

        # Initialize these at run-time:
        self.constraints = []
        self.initialized = False

    def get_removed_dof(self, atoms):
        return self.n

    def initialize(self, atoms):
        if self.initialized:
            return
        masses = np.repeat(atoms.get_masses(), 3)
        cell = None
        pbc = None
        if self.mic:
            cell = atoms.cell
            pbc = atoms.pbc
        self.constraints = []
        for data, ConstrClass in [(self.bonds, self.FixBondLengthAlt),
                                  (self.angles, self.FixAngle),
                                  (self.dihedrals, self.FixDihedral),
                                  (self.bondcombos, self.FixBondCombo)]:
            for datum in data:
                targetvalue = datum[0]
                if targetvalue is None:  # set to current value
                    targetvalue = ConstrClass.get_value(atoms, datum[1],
                                                        self.mic)
                constr = ConstrClass(targetvalue, datum[1], masses, cell, pbc)
                self.constraints.append(constr)
        self.initialized = True

    @staticmethod
    def get_bondcombo(atoms, indices, mic=False):
        """Convenience function to return the value of the bondcombo coordinate
        (linear combination of bond lengths) for the given Atoms object 'atoms'.
        Example: Get the current value of the linear combination of two bond
        lengths defined as `bondcombo = [[0, 1, 1.0], [2, 3, -1.0]]`."""
        c = sum(df[2] * atoms.get_distance(*df[:2], mic=mic) for df in indices)
        return c

    def get_subconstraint(self, atoms, definition):
        """Get pointer to a specific subconstraint.
        Identification by its definition via indices (and coefficients)."""
        self.initialize(atoms)
        for subconstr in self.constraints:
            if isinstance(definition[0], Sequence):  # Combo constraint
                defin = [d + [c] for d, c in zip(subconstr.indices,
                                                 subconstr.coefs)]
                if defin == definition:
                    return subconstr
            else:  # identify primitive constraints by their indices
                if subconstr.indices == [definition]:
                    return subconstr
        raise ValueError('Given `definition` not found on Atoms object.')

    def shuffle_definitions(self, shuffle_dic, internal_type):
        dfns = []  # definitions
        for dfn in internal_type:  # e.g. for bond in self.bonds
            append = True
            new_dfn = [dfn[0], list(dfn[1])]
            for old in dfn[1]:
                if old in shuffle_dic:
                    new_dfn[1][dfn[1].index(old)] = shuffle_dic[old]
                else:
                    append = False
                    break
            if append:
                dfns.append(new_dfn)
        return dfns

    def shuffle_combos(self, shuffle_dic, internal_type):
        dfns = []  # definitions
        for dfn in internal_type:  # i.e. for bondcombo in self.bondcombos
            append = True
            all_indices = [idx[0:-1] for idx in dfn[1]]
            new_dfn = [dfn[0], list(dfn[1])]
            for i, indices in enumerate(all_indices):
                for old in indices:
                    if old in shuffle_dic:
                        new_dfn[1][i][indices.index(old)] = shuffle_dic[old]
                    else:
                        append = False
                        break
                if not append:
                    break
            if append:
                dfns.append(new_dfn)
        return dfns

    def index_shuffle(self, atoms, ind):
        # See docstring of superclass
        self.initialize(atoms)
        shuffle_dic = dict(slice2enlist(ind, len(atoms)))
        shuffle_dic = {old: new for new, old in shuffle_dic.items()}
        self.bonds = self.shuffle_definitions(shuffle_dic, self.bonds)
        self.angles = self.shuffle_definitions(shuffle_dic, self.angles)
        self.dihedrals = self.shuffle_definitions(shuffle_dic, self.dihedrals)
        self.bondcombos = self.shuffle_combos(shuffle_dic, self.bondcombos)
        self.initialized = False
        self.initialize(atoms)
        if len(self.constraints) == 0:
            raise IndexError('Constraint not part of slice')

    def get_indices(self):
        cons = []
        for dfn in self.bonds + self.dihedrals + self.angles:
            cons.extend(dfn[1])
        for dfn in self.bondcombos:
            for partial_dfn in dfn[1]:
                cons.extend(partial_dfn[0:-1])  # last index is the coefficient
        return list(set(cons))

    def todict(self):
        return {'name': 'FixInternals',
                'kwargs': {'bonds': self.bonds,
                           'angles_deg': self.angles,
                           'dihedrals_deg': self.dihedrals,
                           'bondcombos': self.bondcombos,
                           'mic': self.mic,
                           'epsilon': self.epsilon}}

    def adjust_positions(self, atoms, newpos):
        self.initialize(atoms)
        for constraint in self.constraints:
            constraint.setup_jacobian(atoms.positions)
        for _ in range(50):
            maxerr = 0.0
            for constraint in self.constraints:
                constraint.adjust_positions(atoms.positions, newpos)
                maxerr = max(abs(constraint.sigma), maxerr)
            if maxerr < self.epsilon:
                return
        msg = 'FixInternals.adjust_positions did not converge.'
        if any(constr.targetvalue > 175. or constr.targetvalue < 5. for constr
                in self.constraints if isinstance(constr, self.FixAngle)):
            msg += (' This may be caused by an almost planar angle.'
                    ' Support for planar angles would require the'
                    ' implementation of ghost, i.e. dummy, atoms.'
                    ' See issue #868.')
        raise ValueError(msg)

    def adjust_forces(self, atoms, forces):
        """Project out translations and rotations and all other constraints"""
        self.initialize(atoms)
        positions = atoms.positions
        N = len(forces)
        list2_constraints = list(np.zeros((6, N, 3)))
        tx, ty, tz, rx, ry, rz = list2_constraints

        list_constraints = [r.ravel() for r in list2_constraints]

        tx[:, 0] = 1.0
        ty[:, 1] = 1.0
        tz[:, 2] = 1.0
        ff = forces.ravel()

        # Calculate the center of mass
        center = positions.sum(axis=0) / N

        rx[:, 1] = -(positions[:, 2] - center[2])
        rx[:, 2] = positions[:, 1] - center[1]
        ry[:, 0] = positions[:, 2] - center[2]
        ry[:, 2] = -(positions[:, 0] - center[0])
        rz[:, 0] = -(positions[:, 1] - center[1])
        rz[:, 1] = positions[:, 0] - center[0]

        # Normalizing transl., rotat. constraints
        for r in list2_constraints:
            r /= np.linalg.norm(r.ravel())

        # Add all angle, etc. constraint vectors
        for constraint in self.constraints:
            constraint.setup_jacobian(positions)
            constraint.adjust_forces(positions, forces)
            list_constraints.insert(0, constraint.jacobian)
        # QR DECOMPOSITION - GRAM SCHMIDT

        list_constraints = [r.ravel() for r in list_constraints]
        aa = np.column_stack(list_constraints)
        (aa, _bb) = np.linalg.qr(aa)
        # Projection
        hh = []
        for i, constraint in enumerate(self.constraints):
            hh.append(aa[:, i] * np.vstack(aa[:, i]))

        txx = aa[:, self.n] * np.vstack(aa[:, self.n])
        tyy = aa[:, self.n + 1] * np.vstack(aa[:, self.n + 1])
        tzz = aa[:, self.n + 2] * np.vstack(aa[:, self.n + 2])
        rxx = aa[:, self.n + 3] * np.vstack(aa[:, self.n + 3])
        ryy = aa[:, self.n + 4] * np.vstack(aa[:, self.n + 4])
        rzz = aa[:, self.n + 5] * np.vstack(aa[:, self.n + 5])
        T = txx + tyy + tzz + rxx + ryy + rzz
        for vec in hh:
            T += vec
        ff = np.dot(T, np.vstack(ff))
        forces[:, :] -= np.dot(T, np.vstack(ff)).reshape(-1, 3)

    def __repr__(self):
        constraints = [repr(constr) for constr in self.constraints]
        return f'FixInternals(_copy_init={constraints}, epsilon={self.epsilon})'

    # Classes for internal use in FixInternals
    class FixInternalsBase:
        """Base class for subclasses of FixInternals."""

        def __init__(self, targetvalue, indices, masses, cell, pbc):
            self.targetvalue = targetvalue  # constant target value
            self.indices = [defin[0:-1] for defin in indices]  # indices, defs
            self.coefs = np.asarray([defin[-1] for defin in indices])
            self.masses = masses
            self.jacobian = []  # geometric Jacobian matrix, Wilson B-matrix
            self.sigma = 1.  # difference between current and target value
            self.projected_force = None  # helps optimizers scan along constr.
            self.cell = cell
            self.pbc = pbc

        def finalize_jacobian(self, pos, n_internals, n, derivs):
            """Populate jacobian with derivatives for `n_internals` defined
            internals. n = 2 (bonds), 3 (angles), 4 (dihedrals)."""
            jacobian = np.zeros((n_internals, *pos.shape))
            for i, idx in enumerate(self.indices):
                for j in range(n):
                    jacobian[i, idx[j]] = derivs[i, j]
            jacobian = jacobian.reshape((n_internals, 3 * len(pos)))
            return self.coefs @ jacobian

        def finalize_positions(self, newpos):
            jacobian = self.jacobian / self.masses
            lamda = -self.sigma / (jacobian @ self.get_jacobian(newpos))
            dnewpos = lamda * jacobian
            newpos += dnewpos.reshape(newpos.shape)

        def adjust_forces(self, positions, forces):
            self.projected_forces = ((self.jacobian @ forces.ravel())
                                     * self.jacobian)
            self.jacobian /= np.linalg.norm(self.jacobian)

    class FixBondCombo(FixInternalsBase):
        """Constraint subobject for fixing linear combination of bond lengths
        within FixInternals.

        sum_i( coef_i * bond_length_i ) = constant
        """

        def get_jacobian(self, pos):
            bondvectors = [pos[k] - pos[h] for h, k in self.indices]
            derivs = get_distances_derivatives(bondvectors, cell=self.cell,
                                               pbc=self.pbc)
            return self.finalize_jacobian(pos, len(bondvectors), 2, derivs)

        def setup_jacobian(self, pos):
            self.jacobian = self.get_jacobian(pos)

        def adjust_positions(self, oldpos, newpos):
            bondvectors = [newpos[k] - newpos[h] for h, k in self.indices]
            (_, ), (dists, ) = conditional_find_mic([bondvectors],
                                                    cell=self.cell,
                                                    pbc=self.pbc)
            value = self.coefs @ dists
            self.sigma = value - self.targetvalue
            self.finalize_positions(newpos)

        @staticmethod
        def get_value(atoms, indices, mic):
            return FixInternals.get_bondcombo(atoms, indices, mic)

        def __repr__(self):
            return (f'FixBondCombo({self.targetvalue}, {self.indices}, '
                    '{self.coefs})')

    class FixBondLengthAlt(FixBondCombo):
        """Constraint subobject for fixing bond length within FixInternals.
        Fix distance between atoms with indices a1, a2."""

        def __init__(self, targetvalue, indices, masses, cell, pbc):
            if targetvalue <= 0.:
                raise ZeroDivisionError('Invalid targetvalue for fixed bond')
            indices = [list(indices) + [1.]]  # bond definition with coef 1.
            super().__init__(targetvalue, indices, masses, cell=cell, pbc=pbc)

        @staticmethod
        def get_value(atoms, indices, mic):
            return atoms.get_distance(*indices, mic=mic)

        def __repr__(self):
            return f'FixBondLengthAlt({self.targetvalue}, {self.indices})'

    class FixAngle(FixInternalsBase):
        """Constraint subobject for fixing an angle within FixInternals.

        Convergence is potentially problematic for angles very close to
        0 or 180 degrees as there is a singularity in the Cartesian derivative.
        Fixing planar angles is therefore not supported at the moment.
        """

        def __init__(self, targetvalue, indices, masses, cell, pbc):
            """Fix atom movement to construct a constant angle."""
            if targetvalue <= 0. or targetvalue >= 180.:
                raise ZeroDivisionError('Invalid targetvalue for fixed angle')
            indices = [list(indices) + [1.]]  # angle definition with coef 1.
            super().__init__(targetvalue, indices, masses, cell=cell, pbc=pbc)

        def gather_vectors(self, pos):
            v0 = [pos[h] - pos[k] for h, k, l in self.indices]
            v1 = [pos[l] - pos[k] for h, k, l in self.indices]
            return v0, v1

        def get_jacobian(self, pos):
            v0, v1 = self.gather_vectors(pos)
            derivs = get_angles_derivatives(v0, v1, cell=self.cell,
                                            pbc=self.pbc)
            return self.finalize_jacobian(pos, len(v0), 3, derivs)

        def setup_jacobian(self, pos):
            self.jacobian = self.get_jacobian(pos)

        def adjust_positions(self, oldpos, newpos):
            v0, v1 = self.gather_vectors(newpos)
            value = get_angles(v0, v1, cell=self.cell, pbc=self.pbc)
            self.sigma = value - self.targetvalue
            self.finalize_positions(newpos)

        @staticmethod
        def get_value(atoms, indices, mic):
            return atoms.get_angle(*indices, mic=mic)

        def __repr__(self):
            return f'FixAngle({self.targetvalue}, {self.indices})'

    class FixDihedral(FixInternalsBase):
        """Constraint subobject for fixing a dihedral angle within FixInternals.

        A dihedral becomes undefined when at least one of the inner two angles
        becomes planar. Make sure to avoid this situation.
        """

        def __init__(self, targetvalue, indices, masses, cell, pbc):
            indices = [list(indices) + [1.]]  # dihedral def. with coef 1.
            super().__init__(targetvalue, indices, masses, cell=cell, pbc=pbc)

        def gather_vectors(self, pos):
            v0 = [pos[k] - pos[h] for h, k, l, m in self.indices]
            v1 = [pos[l] - pos[k] for h, k, l, m in self.indices]
            v2 = [pos[m] - pos[l] for h, k, l, m in self.indices]
            return v0, v1, v2

        def get_jacobian(self, pos):
            v0, v1, v2 = self.gather_vectors(pos)
            derivs = get_dihedrals_derivatives(v0, v1, v2, cell=self.cell,
                                               pbc=self.pbc)
            return self.finalize_jacobian(pos, len(v0), 4, derivs)

        def setup_jacobian(self, pos):
            self.jacobian = self.get_jacobian(pos)

        def adjust_positions(self, oldpos, newpos):
            v0, v1, v2 = self.gather_vectors(newpos)
            value = get_dihedrals(v0, v1, v2, cell=self.cell, pbc=self.pbc)
            # apply minimum dihedral difference 'convention': (diff <= 180)
            self.sigma = (value - self.targetvalue + 180) % 360 - 180
            self.finalize_positions(newpos)

        @staticmethod
        def get_value(atoms, indices, mic):
            return atoms.get_dihedral(*indices, mic=mic)

        def __repr__(self):
            return f'FixDihedral({self.targetvalue}, {self.indices})'


class FixParametricRelations(FixConstraint):

    def __init__(
        self,
        indices,
        Jacobian,
        const_shift,
        params=None,
        eps=1e-12,
        use_cell=False,
    ):
        """Constrains the degrees of freedom to act in a reduced parameter
        space defined by the Jacobian

        These constraints are based off the work in:
        https://arxiv.org/abs/1908.01610

        The constraints linearly maps the full 3N degrees of freedom,
        where N is number of active lattice vectors/atoms onto a
        reduced subset of M free parameters, where M <= 3*N. The
        Jacobian matrix and constant shift vector map the full set of
        degrees of freedom onto the reduced parameter space.

        Currently the constraint is set up to handle either atomic
        positions or lattice vectors at one time, but not both. To do
        both simply add a two constraints for each set. This is done
        to keep the mathematics behind the operations separate.

        It would be possible to extend these constraints to allow
        non-linear transformations if functionality to update the
        Jacobian at each position update was included. This would
        require passing an update function evaluate it every time
        adjust_positions is callled.  This is currently NOT supported,
        and there are no plans to implement it in the future.

        Args:
            indices (list of int): indices of the constrained atoms
                (if not None or empty then cell_indices must be None or Empty)
            Jacobian (np.ndarray(shape=(3*len(indices), len(params)))):
                The Jacobian describing
                the parameter space transformation
            const_shift (np.ndarray(shape=(3*len(indices)))):
                A vector describing the constant term
                in the transformation not accounted for in the Jacobian
            params (list of str):
                parameters used in the parametric representation
                if None a list is generated based on the shape of the Jacobian
            eps (float): a small number to compare the similarity of
                numbers and set the precision used
                to generate the constraint expressions
            use_cell (bool): if True then act on the cell object

        """
        self.indices = np.array(indices)
        self.Jacobian = np.array(Jacobian)
        self.const_shift = np.array(const_shift)

        assert self.const_shift.shape[0] == 3 * len(self.indices)
        assert self.Jacobian.shape[0] == 3 * len(self.indices)

        self.eps = eps
        self.use_cell = use_cell

        if params is None:
            params = []
            if self.Jacobian.shape[1] > 0:
                int_fmt_str = "{:0" + \
                    str(int(np.ceil(np.log10(self.Jacobian.shape[1])))) + "d}"
                for param_ind in range(self.Jacobian.shape[1]):
                    params.append("param_" + int_fmt_str.format(param_ind))
        else:
            assert len(params) == self.Jacobian.shape[-1]

        self.params = params

        self.Jacobian_inv = np.linalg.inv(
            self.Jacobian.T @ self.Jacobian) @ self.Jacobian.T

    @classmethod
    def from_expressions(cls, indices, params, expressions,
                         eps=1e-12, use_cell=False):
        """Converts the expressions into a Jacobian Matrix/const_shift
        vector and constructs a FixParametricRelations constraint

        The expressions must be a list like object of size 3*N and
        elements must be ordered as:
        [n_0,i; n_0,j; n_0,k; n_1,i; n_1,j; .... ; n_N-1,i; n_N-1,j; n_N-1,k],
        where i, j, and k are the first, second and third
        component of the atomic position/lattice
        vector. Currently only linear operations are allowed to be
        included in the expressions so
        only terms like:
            - const * param_0
            - sqrt[const] * param_1
            - const * param_0 +/- const * param_1 +/- ... +/- const * param_M
        where const is any real number and param_0, param_1, ..., param_M are
        the parameters passed in
        params, are allowed.

        For example, fractional atomic position constraints for wurtzite are:
        params = ["z1", "z2"]
        expressions = [
            "1.0/3.0", "2.0/3.0", "z1",
            "2.0/3.0", "1.0/3.0", "0.5 + z1",
            "1.0/3.0", "2.0/3.0", "z2",
            "2.0/3.0", "1.0/3.0", "0.5 + z2",
        ]

        For diamond are:
        params = []
        expressions = [
            "0.0", "0.0", "0.0",
            "0.25", "0.25", "0.25",
        ],

        and for stannite are
        params=["x4", "z4"]
        expressions = [
            "0.0", "0.0", "0.0",
            "0.0", "0.5", "0.5",
            "0.75", "0.25", "0.5",
            "0.25", "0.75", "0.5",
            "x4 + z4", "x4 + z4", "2*x4",
            "x4 - z4", "x4 - z4", "-2*x4",
             "0.0", "-1.0 * (x4 + z4)", "x4 - z4",
             "0.0", "x4 - z4", "-1.0 * (x4 + z4)",
        ]

        Args:
            indices (list of int): indices of the constrained atoms
                (if not None or empty then cell_indices must be None or Empty)
            params (list of str): parameters used in the
            parametric representation
            expressions (list of str): expressions used to convert from the
            parametric to the real space representation
            eps (float): a small number to compare the similarity of
                numbers and set the precision used
                to generate the constraint expressions
            use_cell (bool): if True then act on the cell object

        Returns:
            cls(
                indices,
                Jacobian generated from expressions,
                const_shift generated from expressions,
                params,
                eps-12,
                use_cell,
            )
        """
        Jacobian = np.zeros((3 * len(indices), len(params)))
        const_shift = np.zeros(3 * len(indices))

        for expr_ind, expression in enumerate(expressions):
            expression = expression.strip()

            # Convert subtraction to addition
            expression = expression.replace("-", "+(-1.0)*")
            if expression[0] == "+":
                expression = expression[1:]
            elif expression[:2] == "(+":
                expression = "(" + expression[2:]

            # Explicitly add leading zeros so when replacing param_1 with 0.0
            # param_11 does not become 0.01
            int_fmt_str = "{:0" + \
                str(int(np.ceil(np.log10(len(params) + 1)))) + "d}"

            param_dct = {}
            param_map = {}

            # Construct a standardized param template for A/B filling
            for param_ind, param in enumerate(params):
                param_str = "param_" + int_fmt_str.format(param_ind)
                param_map[param] = param_str
                param_dct[param_str] = 0.0

            # Replace the parameters according to the map
            # Sort by string length (long to short) to prevent cases like x11
            # becoming f"{param_map["x1"]}1"
            for param in sorted(params, key=lambda s: -1.0 * len(s)):
                expression = expression.replace(param, param_map[param])

            # Partial linearity check
            for express_sec in expression.split("+"):
                in_sec = [param in express_sec for param in param_dct]
                n_params_in_sec = len(np.where(np.array(in_sec))[0])
                if n_params_in_sec > 1:
                    raise ValueError(
                        "FixParametricRelations expressions must be linear.")

            const_shift[expr_ind] = float(
                eval_expression(expression, param_dct))

            for param_ind in range(len(params)):
                param_str = "param_" + int_fmt_str.format(param_ind)
                if param_str not in expression:
                    Jacobian[expr_ind, param_ind] = 0.0
                    continue
                param_dct[param_str] = 1.0
                test_1 = float(eval_expression(expression, param_dct))
                test_1 -= const_shift[expr_ind]
                Jacobian[expr_ind, param_ind] = test_1

                param_dct[param_str] = 2.0
                test_2 = float(eval_expression(expression, param_dct))
                test_2 -= const_shift[expr_ind]
                if abs(test_2 / test_1 - 2.0) > eps:
                    raise ValueError(
                        "FixParametricRelations expressions must be linear.")
                param_dct[param_str] = 0.0

        args = [
            indices,
            Jacobian,
            const_shift,
            params,
            eps,
            use_cell,
        ]
        if cls is FixScaledParametricRelations:
            args = args[:-1]
        return cls(*args)

    @property
    def expressions(self):
        """Generate the expressions represented by the current self.Jacobian
        and self.const_shift objects"""
        expressions = []
        per = int(round(-1 * np.log10(self.eps)))
        fmt_str = "{:." + str(per + 1) + "g}"
        for index, shift_val in enumerate(self.const_shift):
            exp = ""
            if np.all(np.abs(self.Jacobian[index]) < self.eps) or np.abs(
                    shift_val) > self.eps:
                exp += fmt_str.format(shift_val)

            param_exp = ""
            for param_index, jacob_val in enumerate(self.Jacobian[index]):
                abs_jacob_val = np.round(np.abs(jacob_val), per + 1)
                if abs_jacob_val < self.eps:
                    continue

                param = self.params[param_index]
                if param_exp or exp:
                    if jacob_val > -1.0 * self.eps:
                        param_exp += " + "
                    else:
                        param_exp += " - "
                elif (not exp) and (not param_exp) and (
                        jacob_val < -1.0 * self.eps):
                    param_exp += "-"

                if np.abs(abs_jacob_val - 1.0) <= self.eps:
                    param_exp += f"{param:s}"
                else:
                    param_exp += (fmt_str +
                                  "*{:s}").format(abs_jacob_val, param)

            exp += param_exp

            expressions.append(exp)
        return np.array(expressions).reshape((-1, 3))

    def todict(self):
        """Create a dictionary representation of the constraint"""
        return {
            "name": type(self).__name__,
            "kwargs": {
                "indices": self.indices,
                "params": self.params,
                "Jacobian": self.Jacobian,
                "const_shift": self.const_shift,
                "eps": self.eps,
                "use_cell": self.use_cell,
            }
        }

    def __repr__(self):
        """The str representation of the constraint"""
        if len(self.indices) > 1:
            indices_str = "[{:d}, ..., {:d}]".format(
                self.indices[0], self.indices[-1])
        else:
            indices_str = f"[{self.indices[0]:d}]"

        if len(self.params) > 1:
            params_str = "[{:s}, ..., {:s}]".format(
                self.params[0], self.params[-1])
        elif len(self.params) == 1:
            params_str = f"[{self.params[0]:s}]"
        else:
            params_str = "[]"

        return '{:s}({:s}, {:s}, ..., {:e})'.format(
            type(self).__name__,
            indices_str,
            params_str,
            self.eps
        )


class FixScaledParametricRelations(FixParametricRelations):

    def __init__(
        self,
        indices,
        Jacobian,
        const_shift,
        params=None,
        eps=1e-12,
    ):
        """The fractional coordinate version of FixParametricRelations

        All arguments are the same, but since this is for fractional
        coordinates use_cell is false"""
        super().__init__(
            indices,
            Jacobian,
            const_shift,
            params,
            eps,
            False,
        )

    def adjust_contravariant(self, cell, vecs, B):
        """Adjust the values of a set of vectors that are contravariant
        with the unit transformation"""
        scaled = cell.scaled_positions(vecs).flatten()
        scaled = self.Jacobian_inv @ (scaled - B)
        scaled = ((self.Jacobian @ scaled) + B).reshape((-1, 3))

        return cell.cartesian_positions(scaled)

    def adjust_positions(self, atoms, positions):
        """Adjust positions of the atoms to match the constraints"""
        positions[self.indices] = self.adjust_contravariant(
            atoms.cell,
            positions[self.indices],
            self.const_shift,
        )
        positions[self.indices] = self.adjust_B(
            atoms.cell, positions[self.indices])

    def adjust_B(self, cell, positions):
        """Wraps the positions back to the unit cell and adjust B to
        keep track of this change"""
        fractional = cell.scaled_positions(positions)
        wrapped_fractional = (fractional % 1.0) % 1.0
        self.const_shift += np.round(wrapped_fractional - fractional).flatten()
        return cell.cartesian_positions(wrapped_fractional)

    def adjust_momenta(self, atoms, momenta):
        """Adjust momenta of the atoms to match the constraints"""
        momenta[self.indices] = self.adjust_contravariant(
            atoms.cell,
            momenta[self.indices],
            np.zeros(self.const_shift.shape),
        )

    def adjust_forces(self, atoms, forces):
        """Adjust forces of the atoms to match the constraints"""
        # Forces are coavarient to the coordinate transformation, use the
        # inverse transformations
        cart2frac_jacob = np.zeros(2 * (3 * len(atoms),))
        for i_atom in range(len(atoms)):
            cart2frac_jacob[3 * i_atom:3 * (i_atom + 1),
                            3 * i_atom:3 * (i_atom + 1)] = atoms.cell.T

        jacobian = cart2frac_jacob @ self.Jacobian
        jacobian_inv = np.linalg.inv(jacobian.T @ jacobian) @ jacobian.T

        reduced_forces = jacobian.T @ forces.flatten()
        forces[self.indices] = (jacobian_inv.T @ reduced_forces).reshape(-1, 3)

    def todict(self):
        """Create a dictionary representation of the constraint"""
        dct = super().todict()
        del dct["kwargs"]["use_cell"]
        return dct


class FixCartesianParametricRelations(FixParametricRelations):

    def __init__(
        self,
        indices,
        Jacobian,
        const_shift,
        params=None,
        eps=1e-12,
        use_cell=False,
    ):
        """The Cartesian coordinate version of FixParametricRelations"""
        super().__init__(
            indices,
            Jacobian,
            const_shift,
            params,
            eps,
            use_cell,
        )

    def adjust_contravariant(self, vecs, B):
        """Adjust the values of a set of vectors that are contravariant with
        the unit transformation"""
        vecs = self.Jacobian_inv @ (vecs.flatten() - B)
        vecs = ((self.Jacobian @ vecs) + B).reshape((-1, 3))
        return vecs

    def adjust_positions(self, atoms, positions):
        """Adjust positions of the atoms to match the constraints"""
        if self.use_cell:
            return
        positions[self.indices] = self.adjust_contravariant(
            positions[self.indices],
            self.const_shift,
        )

    def adjust_momenta(self, atoms, momenta):
        """Adjust momenta of the atoms to match the constraints"""
        if self.use_cell:
            return
        momenta[self.indices] = self.adjust_contravariant(
            momenta[self.indices],
            np.zeros(self.const_shift.shape),
        )

    def adjust_forces(self, atoms, forces):
        """Adjust forces of the atoms to match the constraints"""
        if self.use_cell:
            return

        forces_reduced = self.Jacobian.T @ forces[self.indices].flatten()
        forces[self.indices] = (self.Jacobian_inv.T @
                                forces_reduced).reshape(-1, 3)

    def adjust_cell(self, atoms, cell):
        """Adjust the cell of the atoms to match the constraints"""
        if not self.use_cell:
            return
        cell[self.indices] = self.adjust_contravariant(
            cell[self.indices],
            np.zeros(self.const_shift.shape),
        )

    def adjust_stress(self, atoms, stress):
        """Adjust the stress of the atoms to match the constraints"""
        if not self.use_cell:
            return

        stress_3x3 = voigt_6_to_full_3x3_stress(stress)
        stress_reduced = self.Jacobian.T @ stress_3x3[self.indices].flatten()
        stress_3x3[self.indices] = (
            self.Jacobian_inv.T @ stress_reduced).reshape(-1, 3)

        stress[:] = full_3x3_to_voigt_6_stress(stress_3x3)


class Hookean(FixConstraint):
    """Applies a Hookean restorative force between a pair of atoms, an atom
    and a point, or an atom and a plane."""

    def __init__(self, a1, a2, k, rt=None):
        """Forces two atoms to stay close together by applying no force if
        they are below a threshold length, rt, and applying a Hookean
        restorative force when the distance between them exceeds rt. Can
        also be used to tether an atom to a fixed point in space or to a
        distance above a plane.

        a1 : int
           Index of atom 1
        a2 : one of three options
           1) index of atom 2
           2) a fixed point in cartesian space to which to tether a1
           3) a plane given as (A, B, C, D) in A x + B y + C z + D = 0.
        k : float
           Hooke's law (spring) constant to apply when distance
           exceeds threshold_length. Units of eV A^-2.
        rt : float
           The threshold length below which there is no force. The
           length is 1) between two atoms, 2) between atom and point.
           This argument is not supplied in case 3. Units of A.

        If a plane is specified, the Hooke's law force is applied if the atom
        is on the normal side of the plane. For instance, the plane with
        (A, B, C, D) = (0, 0, 1, -7) defines a plane in the xy plane with a z
        intercept of +7 and a normal vector pointing in the +z direction.
        If the atom has z > 7, then a downward force would be applied of
        k * (atom.z - 7). The same plane with the normal vector pointing in
        the -z direction would be given by (A, B, C, D) = (0, 0, -1, 7).

        References:

           Andrew A. Peterson,  Topics in Catalysis volume 57, pages40–53 (2014)
           https://link.springer.com/article/10.1007%2Fs11244-013-0161-8
        """

        if isinstance(a2, int):
            self._type = 'two atoms'
            self.indices = [a1, a2]
        elif len(a2) == 3:
            self._type = 'point'
            self.index = a1
            self.origin = np.array(a2)
        elif len(a2) == 4:
            self._type = 'plane'
            self.index = a1
            self.plane = a2
        else:
            raise RuntimeError('Unknown type for a2')
        self.threshold = rt
        self.spring = k

    def get_removed_dof(self, atoms):
        return 0

    def todict(self):
        dct = {'name': 'Hookean'}
        dct['kwargs'] = {'rt': self.threshold,
                         'k': self.spring}
        if self._type == 'two atoms':
            dct['kwargs']['a1'] = self.indices[0]
            dct['kwargs']['a2'] = self.indices[1]
        elif self._type == 'point':
            dct['kwargs']['a1'] = self.index
            dct['kwargs']['a2'] = self.origin
        elif self._type == 'plane':
            dct['kwargs']['a1'] = self.index
            dct['kwargs']['a2'] = self.plane
        else:
            raise NotImplementedError(f'Bad type: {self._type}')
        return dct

    def adjust_positions(self, atoms, newpositions):
        pass

    def adjust_momenta(self, atoms, momenta):
        pass

    def adjust_forces(self, atoms, forces):
        positions = atoms.positions
        if self._type == 'plane':
            A, B, C, D = self.plane
            x, y, z = positions[self.index]
            d = ((A * x + B * y + C * z + D) /
                 np.sqrt(A**2 + B**2 + C**2))
            if d < 0:
                return
            magnitude = self.spring * d
            direction = - np.array((A, B, C)) / np.linalg.norm((A, B, C))
            forces[self.index] += direction * magnitude
            return
        if self._type == 'two atoms':
            p1, p2 = positions[self.indices]
        elif self._type == 'point':
            p1 = positions[self.index]
            p2 = self.origin
        displace, _ = find_mic(p2 - p1, atoms.cell, atoms.pbc)
        bondlength = np.linalg.norm(displace)
        if bondlength > self.threshold:
            magnitude = self.spring * (bondlength - self.threshold)
            direction = displace / np.linalg.norm(displace)
            if self._type == 'two atoms':
                forces[self.indices[0]] += direction * magnitude
                forces[self.indices[1]] -= direction * magnitude
            else:
                forces[self.index] += direction * magnitude

    def adjust_potential_energy(self, atoms):
        """Returns the difference to the potential energy due to an active
        constraint. (That is, the quantity returned is to be added to the
        potential energy.)"""
        positions = atoms.positions
        if self._type == 'plane':
            A, B, C, D = self.plane
            x, y, z = positions[self.index]
            d = ((A * x + B * y + C * z + D) /
                 np.sqrt(A**2 + B**2 + C**2))
            if d > 0:
                return 0.5 * self.spring * d**2
            else:
                return 0.
        if self._type == 'two atoms':
            p1, p2 = positions[self.indices]
        elif self._type == 'point':
            p1 = positions[self.index]
            p2 = self.origin
        displace, _ = find_mic(p2 - p1, atoms.cell, atoms.pbc)
        bondlength = np.linalg.norm(displace)
        if bondlength > self.threshold:
            return 0.5 * self.spring * (bondlength - self.threshold)**2
        else:
            return 0.

    def get_indices(self):
        if self._type == 'two atoms':
            return self.indices
        elif self._type == 'point':
            return self.index
        elif self._type == 'plane':
            return self.index

    def index_shuffle(self, atoms, ind):
        # See docstring of superclass
        if self._type == 'two atoms':
            newa = [-1, -1]  # Signal error
            for new, old in slice2enlist(ind, len(atoms)):
                for i, a in enumerate(self.indices):
                    if old == a:
                        newa[i] = new
            if newa[0] == -1 or newa[1] == -1:
                raise IndexError('Constraint not part of slice')
            self.indices = newa
        elif (self._type == 'point') or (self._type == 'plane'):
            newa = -1   # Signal error
            for new, old in slice2enlist(ind, len(atoms)):
                if old == self.index:
                    newa = new
                    break
            if newa == -1:
                raise IndexError('Constraint not part of slice')
            self.index = newa

    def __repr__(self):
        if self._type == 'two atoms':
            return 'Hookean(%d, %d)' % tuple(self.indices)
        elif self._type == 'point':
            return 'Hookean(%d) to cartesian' % self.index
        else:
            return 'Hookean(%d) to plane' % self.index


class ExternalForce(FixConstraint):
    """Constraint object for pulling two atoms apart by an external force.

    You can combine this constraint for example with FixBondLength but make
    sure that *ExternalForce* comes first in the list if there are overlaps
    between atom1-2 and atom3-4:

    >>> from ase.build import bulk

    >>> atoms = bulk('Cu', 'fcc', a=3.6)
    >>> atom1, atom2, atom3, atom4 = atoms[:4]
    >>> fext = 1.0
    >>> con1 = ExternalForce(atom1, atom2, f_ext)
    >>> con2 = FixBondLength(atom3, atom4)
    >>> atoms.set_constraint([con1, con2])

    see ase/test/external_force.py"""

    def __init__(self, a1, a2, f_ext):
        self.indices = [a1, a2]
        self.external_force = f_ext

    def get_removed_dof(self, atoms):
        return 0

    def adjust_positions(self, atoms, new):
        pass

    def adjust_forces(self, atoms, forces):
        dist = np.subtract.reduce(atoms.positions[self.indices])
        force = self.external_force * dist / np.linalg.norm(dist)
        forces[self.indices] += (force, -force)

    def adjust_potential_energy(self, atoms):
        dist = np.subtract.reduce(atoms.positions[self.indices])
        return -np.linalg.norm(dist) * self.external_force

    def index_shuffle(self, atoms, ind):
        """Shuffle the indices of the two atoms in this constraint"""
        newa = [-1, -1]  # Signal error
        for new, old in slice2enlist(ind, len(atoms)):
            for i, a in enumerate(self.indices):
                if old == a:
                    newa[i] = new
        if newa[0] == -1 or newa[1] == -1:
            raise IndexError('Constraint not part of slice')
        self.indices = newa

    def __repr__(self):
        return 'ExternalForce(%d, %d, %f)' % (self.indices[0],
                                              self.indices[1],
                                              self.external_force)

    def todict(self):
        return {'name': 'ExternalForce',
                'kwargs': {'a1': self.indices[0], 'a2': self.indices[1],
                           'f_ext': self.external_force}}


class MirrorForce(FixConstraint):
    """Constraint object for mirroring the force between two atoms.

    This class is designed to find a transition state with the help of a
    single optimization. It can be used if the transition state belongs to a
    bond breaking reaction. First the given bond length will be fixed until
    all other degrees of freedom are optimized, then the forces of the two
    atoms will be mirrored to find the transition state. The mirror plane is
    perpendicular to the connecting line of the atoms. Transition states in
    dependence of the force can be obtained by stretching the molecule and
    fixing its total length with *FixBondLength* or by using *ExternalForce*
    during the optimization with *MirrorForce*.

    Parameters
    ----------
    a1: int
        First atom index.
    a2: int
        Second atom index.
    max_dist: float
        Upper limit of the bond length interval where the transition state
        can be found.
    min_dist: float
        Lower limit of the bond length interval where the transition state
        can be found.
    fmax: float
        Maximum force used for the optimization.

    Notes
    -----
    You can combine this constraint for example with FixBondLength but make
    sure that *MirrorForce* comes first in the list if there are overlaps
    between atom1-2 and atom3-4:

    >>> from ase.build import bulk

    >>> atoms = bulk('Cu', 'fcc', a=3.6)
    >>> atom1, atom2, atom3, atom4 = atoms[:4]
    >>> con1 = MirrorForce(atom1, atom2)
    >>> con2 = FixBondLength(atom3, atom4)
    >>> atoms.set_constraint([con1, con2])

    """

    def __init__(self, a1, a2, max_dist=2.5, min_dist=1., fmax=0.1):
        self.indices = [a1, a2]
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.fmax = fmax

    def adjust_positions(self, atoms, new):
        pass

    def adjust_forces(self, atoms, forces):
        dist = np.subtract.reduce(atoms.positions[self.indices])
        d = np.linalg.norm(dist)
        if (d < self.min_dist) or (d > self.max_dist):
            # Stop structure optimization
            forces[:] *= 0
            return
        dist /= d
        df = np.subtract.reduce(forces[self.indices])
        f = df.dot(dist)
        con_saved = atoms.constraints
        try:
            con = [con for con in con_saved
                   if not isinstance(con, MirrorForce)]
            atoms.set_constraint(con)
            forces_copy = atoms.get_forces()
        finally:
            atoms.set_constraint(con_saved)
        df1 = -1 / 2. * f * dist
        forces_copy[self.indices] += (df1, -df1)
        # Check if forces would be converged if the bond with mirrored forces
        # would also be fixed
        if (forces_copy**2).sum(axis=1).max() < self.fmax**2:
            factor = 1.
        else:
            factor = 0.
        df1 = -(1 + factor) / 2. * f * dist
        forces[self.indices] += (df1, -df1)

    def index_shuffle(self, atoms, ind):
        """Shuffle the indices of the two atoms in this constraint

        """
        newa = [-1, -1]  # Signal error
        for new, old in slice2enlist(ind, len(atoms)):
            for i, a in enumerate(self.indices):
                if old == a:
                    newa[i] = new
        if newa[0] == -1 or newa[1] == -1:
            raise IndexError('Constraint not part of slice')
        self.indices = newa

    def __repr__(self):
        return 'MirrorForce(%d, %d, %f, %f, %f)' % (
            self.indices[0], self.indices[1], self.max_dist, self.min_dist,
            self.fmax)

    def todict(self):
        return {'name': 'MirrorForce',
                'kwargs': {'a1': self.indices[0], 'a2': self.indices[1],
                           'max_dist': self.max_dist,
                           'min_dist': self.min_dist, 'fmax': self.fmax}}


class MirrorTorque(FixConstraint):
    """Constraint object for mirroring the torque acting on a dihedral
    angle defined by four atoms.

    This class is designed to find a transition state with the help of a
    single optimization. It can be used if the transition state belongs to a
    cis-trans-isomerization with a change of dihedral angle. First the given
    dihedral angle will be fixed until all other degrees of freedom are
    optimized, then the torque acting on the dihedral angle will be mirrored
    to find the transition state. Transition states in
    dependence of the force can be obtained by stretching the molecule and
    fixing its total length with *FixBondLength* or by using *ExternalForce*
    during the optimization with *MirrorTorque*.

    This constraint can be used to find
    transition states of cis-trans-isomerization.

    a1    a4
    |      |
    a2 __ a3

    Parameters
    ----------
    a1: int
        First atom index.
    a2: int
        Second atom index.
    a3: int
        Third atom index.
    a4: int
        Fourth atom index.
    max_angle: float
        Upper limit of the dihedral angle interval where the transition state
        can be found.
    min_angle: float
        Lower limit of the dihedral angle interval where the transition state
        can be found.
    fmax: float
        Maximum force used for the optimization.

    Notes
    -----
    You can combine this constraint for example with FixBondLength but make
    sure that *MirrorTorque* comes first in the list if there are overlaps
    between atom1-4 and atom5-6:

    >>> from ase.build import bulk

    >>> atoms = bulk('Cu', 'fcc', a=3.6)
    >>> atom1, atom2, atom3, atom4, atom5, atom6 = atoms[:6]
    >>> con1 = MirrorTorque(atom1, atom2, atom3, atom4)
    >>> con2 = FixBondLength(atom5, atom6)
    >>> atoms.set_constraint([con1, con2])

    """

    def __init__(self, a1, a2, a3, a4, max_angle=2 * np.pi, min_angle=0.,
                 fmax=0.1):
        self.indices = [a1, a2, a3, a4]
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.fmax = fmax

    def adjust_positions(self, atoms, new):
        pass

    def adjust_forces(self, atoms, forces):
        angle = atoms.get_dihedral(self.indices[0], self.indices[1],
                                   self.indices[2], self.indices[3])
        angle *= np.pi / 180.
        if (angle < self.min_angle) or (angle > self.max_angle):
            # Stop structure optimization
            forces[:] *= 0
            return
        p = atoms.positions[self.indices]
        f = forces[self.indices]

        f0 = (f[1] + f[2]) / 2.
        ff = f - f0
        p0 = (p[2] + p[1]) / 2.
        m0 = np.cross(p[1] - p0, ff[1]) / (p[1] - p0).dot(p[1] - p0)
        fff = ff - np.cross(m0, p - p0)
        d1 = np.cross(np.cross(p[1] - p0, p[0] - p[1]), p[1] - p0) / \
            (p[1] - p0).dot(p[1] - p0)
        d2 = np.cross(np.cross(p[2] - p0, p[3] - p[2]), p[2] - p0) / \
            (p[2] - p0).dot(p[2] - p0)
        omegap1 = (np.cross(d1, fff[0]) / d1.dot(d1)).dot(p[1] - p0) / \
            np.linalg.norm(p[1] - p0)
        omegap2 = (np.cross(d2, fff[3]) / d2.dot(d2)).dot(p[2] - p0) / \
            np.linalg.norm(p[2] - p0)
        omegap = omegap1 + omegap2
        con_saved = atoms.constraints
        try:
            con = [con for con in con_saved
                   if not isinstance(con, MirrorTorque)]
            atoms.set_constraint(con)
            forces_copy = atoms.get_forces()
        finally:
            atoms.set_constraint(con_saved)
        df1 = -1 / 2. * omegap * np.cross(p[1] - p0, d1) / \
            np.linalg.norm(p[1] - p0)
        df2 = -1 / 2. * omegap * np.cross(p[2] - p0, d2) / \
            np.linalg.norm(p[2] - p0)
        forces_copy[self.indices] += (df1, [0., 0., 0.], [0., 0., 0.], df2)
        # Check if forces would be converged if the dihedral angle with
        # mirrored torque would also be fixed
        if (forces_copy**2).sum(axis=1).max() < self.fmax**2:
            factor = 1.
        else:
            factor = 0.
        df1 = -(1 + factor) / 2. * omegap * np.cross(p[1] - p0, d1) / \
            np.linalg.norm(p[1] - p0)
        df2 = -(1 + factor) / 2. * omegap * np.cross(p[2] - p0, d2) / \
            np.linalg.norm(p[2] - p0)
        forces[self.indices] += (df1, [0., 0., 0.], [0., 0., 0.], df2)

    def index_shuffle(self, atoms, ind):
        # See docstring of superclass
        indices = []
        for new, old in slice2enlist(ind, len(atoms)):
            if old in self.indices:
                indices.append(new)
        if len(indices) == 0:
            raise IndexError('All indices in MirrorTorque not part of slice')
        self.indices = np.asarray(indices, int)

    def __repr__(self):
        return 'MirrorTorque(%d, %d, %d, %d, %f, %f, %f)' % (
            self.indices[0], self.indices[1], self.indices[2],
            self.indices[3], self.max_angle, self.min_angle, self.fmax)

    def todict(self):
        return {'name': 'MirrorTorque',
                'kwargs': {'a1': self.indices[0], 'a2': self.indices[1],
                           'a3': self.indices[2], 'a4': self.indices[3],
                           'max_angle': self.max_angle,
                           'min_angle': self.min_angle, 'fmax': self.fmax}}


class FixSymmetry(FixConstraint):
    """
    Constraint to preserve spacegroup symmetry during optimisation.

    Requires spglib package to be available.
    """

    def __init__(self, atoms, symprec=0.01, adjust_positions=True,
                 adjust_cell=True, verbose=False):
        self.atoms = atoms.copy()
        self.symprec = symprec
        self.verbose = verbose
        refine_symmetry(atoms, symprec, self.verbose)  # refine initial symmetry
        sym = prep_symmetry(atoms, symprec, self.verbose)
        self.rotations, self.translations, self.symm_map = sym
        self.do_adjust_positions = adjust_positions
        self.do_adjust_cell = adjust_cell

    def adjust_cell(self, atoms, cell):
        if not self.do_adjust_cell:
            return
        # stress should definitely be symmetrized as a rank 2 tensor
        # UnitCellFilter uses deformation gradient as cell DOF with steps
        # dF = stress.F^-T quantity that should be symmetrized is therefore dF .
        # F^T assume prev F = I, so just symmetrize dF
        cur_cell = atoms.get_cell()
        cur_cell_inv = atoms.cell.reciprocal().T

        # F defined such that cell = cur_cell . F^T
        # assume prev F = I, so dF = F - I
        delta_deform_grad = np.dot(cur_cell_inv, cell).T - np.eye(3)

        # symmetrization doesn't work properly with large steps, since
        # it depends on current cell, and cell is being changed by deformation
        # gradient
        max_delta_deform_grad = np.max(np.abs(delta_deform_grad))
        if max_delta_deform_grad > 0.25:
            raise RuntimeError('FixSymmetry adjust_cell does not work properly'
                               ' with large deformation gradient step {} > 0.25'
                               .format(max_delta_deform_grad))
        elif max_delta_deform_grad > 0.15:
            warn('FixSymmetry adjust_cell may be ill behaved with large '
                 'deformation gradient step {}'.format(max_delta_deform_grad))

        symmetrized_delta_deform_grad = symmetrize_rank2(cur_cell, cur_cell_inv,
                                                         delta_deform_grad,
                                                         self.rotations)
        cell[:] = np.dot(cur_cell,
                         (symmetrized_delta_deform_grad + np.eye(3)).T)

    def adjust_positions(self, atoms, new):
        if not self.do_adjust_positions:
            return
        # symmetrize changes in position as rank 1 tensors
        step = new - atoms.positions
        symmetrized_step = symmetrize_rank1(atoms.get_cell(),
                                            atoms.cell.reciprocal().T, step,
                                            self.rotations, self.translations,
                                            self.symm_map)
        new[:] = atoms.positions + symmetrized_step

    def adjust_forces(self, atoms, forces):
        # symmetrize forces as rank 1 tensors
        # print('adjusting forces')
        forces[:] = symmetrize_rank1(atoms.get_cell(),
                                     atoms.cell.reciprocal().T, forces,
                                     self.rotations, self.translations,
                                     self.symm_map)

    def adjust_stress(self, atoms, stress):
        # symmetrize stress as rank 2 tensor
        raw_stress = voigt_6_to_full_3x3_stress(stress)
        symmetrized_stress = symmetrize_rank2(atoms.get_cell(),
                                              atoms.cell.reciprocal().T,
                                              raw_stress, self.rotations)
        stress[:] = full_3x3_to_voigt_6_stress(symmetrized_stress)

    def index_shuffle(self, atoms, ind):
        if len(atoms) != len(ind) or len(set(ind)) != len(ind):
            raise RuntimeError("FixSymmetry can only accomodate atom"
                               " permutions, and len(Atoms) == {} "
                               "!= len(ind) == {} or ind has duplicates"
                               .format(len(atoms), len(ind)))

        ind_reversed = np.zeros((len(ind)), dtype=int)
        ind_reversed[ind] = range(len(ind))
        new_symm_map = []
        for sm in self.symm_map:
            new_sm = np.array([-1] * len(atoms))
            for at_i in range(len(ind)):
                new_sm[ind_reversed[at_i]] = ind_reversed[sm[at_i]]
            new_symm_map.append(new_sm)

        self.symm_map = new_symm_map

    def todict(self):
        return {
            'name': 'FixSymmetry',
            'kwargs': {
                'atoms': self.atoms,
                'symprec': self.symprec,
                'adjust_positions': self.do_adjust_positions,
                'adjust_cell': self.do_adjust_cell,
                'verbose': self.verbose,
            },
        }


class Filter(FilterOld):
    @deprecated('Import Filter from ase.filters')
    def __init__(self, *args, **kwargs):
        """
        .. deprecated:: 3.23.0
            Import ``Filter`` from :mod:`ase.filters`
        """
        super().__init__(*args, **kwargs)


class StrainFilter(StrainFilterOld):
    @deprecated('Import StrainFilter from ase.filters')
    def __init__(self, *args, **kwargs):
        """
        .. deprecated:: 3.23.0
            Import ``StrainFilter`` from :mod:`ase.filters`
        """
        super().__init__(*args, **kwargs)


class UnitCellFilter(UnitCellFilterOld):
    @deprecated('Import UnitCellFilter from ase.filters')
    def __init__(self, *args, **kwargs):
        """
        .. deprecated:: 3.23.0
            Import ``UnitCellFilter`` from :mod:`ase.filters`
        """
        super().__init__(*args, **kwargs)


class ExpCellFilter(ExpCellFilterOld):
    @deprecated('Import ExpCellFilter from ase.filters')
    def __init__(self, *args, **kwargs):
        """
        .. deprecated:: 3.23.0
            Import ``ExpCellFilter`` from :mod:`ase.filters`
            or use :class:`~ase.filters.FrechetCellFilter` for better
            convergence w.r.t. cell variables
        """
        super().__init__(*args, **kwargs)
