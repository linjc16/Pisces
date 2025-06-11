# fmt: off

"""A collection of mutations that can be used."""
from math import cos, pi, sin

import numpy as np

from ase import Atoms
from ase.calculators.lammps.coordinatetransform import calc_rotated_cell
from ase.cell import Cell
from ase.ga.offspring_creator import CombinationMutation, OffspringCreator
from ase.ga.utilities import (
    atoms_too_close,
    atoms_too_close_two_sets,
    gather_atoms_by_tag,
    get_rotation_matrix,
)


class RattleMutation(OffspringCreator):
    """An implementation of the rattle mutation as described in:

    R.L. Johnston Dalton Transactions, Vol. 22,
    No. 22. (2003), pp. 4193-4207

    Parameters:

    blmin: Dictionary defining the minimum distance between atoms
        after the rattle.

    n_top: Number of atoms optimized by the GA.

    rattle_strength: Strength with which the atoms are moved.

    rattle_prop: The probability with which each atom is rattled.

    test_dist_to_slab: whether to also make sure that the distances
        between the atoms and the slab satisfy the blmin.

    use_tags: if True, the atomic tags will be used to preserve
        molecular identity. Same-tag atoms will then be
        displaced collectively, so that the internal
        geometry is preserved.

    rng: Random number generator
        By default numpy.random.
    """

    def __init__(self, blmin, n_top, rattle_strength=0.8,
                 rattle_prop=0.4, test_dist_to_slab=True, use_tags=False,
                 verbose=False, rng=np.random):
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.blmin = blmin
        self.n_top = n_top
        self.rattle_strength = rattle_strength
        self.rattle_prop = rattle_prop
        self.test_dist_to_slab = test_dist_to_slab
        self.use_tags = use_tags

        self.descriptor = 'RattleMutation'
        self.min_inputs = 1

    def get_new_individual(self, parents):
        f = parents[0]

        indi = self.mutate(f)
        if indi is None:
            return indi, 'mutation: rattle'

        indi = self.initialize_individual(f, indi)
        indi.info['data']['parents'] = [f.info['confid']]

        return self.finalize_individual(indi), 'mutation: rattle'

    def mutate(self, atoms):
        """Does the actual mutation."""
        N = len(atoms) if self.n_top is None else self.n_top
        slab = atoms[:len(atoms) - N]
        atoms = atoms[-N:]
        tags = atoms.get_tags() if self.use_tags else np.arange(N)
        pos_ref = atoms.get_positions()
        num = atoms.get_atomic_numbers()
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()
        st = 2. * self.rattle_strength

        count = 0
        maxcount = 1000
        too_close = True
        while too_close and count < maxcount:
            count += 1
            pos = pos_ref.copy()
            ok = False
            for tag in np.unique(tags):
                select = np.where(tags == tag)
                if self.rng.random() < self.rattle_prop:
                    ok = True
                    r = self.rng.random(3)
                    pos[select] += st * (r - 0.5)

            if not ok:
                # Nothing got rattled
                continue

            top = Atoms(num, positions=pos, cell=cell, pbc=pbc, tags=tags)
            too_close = atoms_too_close(
                top, self.blmin, use_tags=self.use_tags)
            if not too_close and self.test_dist_to_slab:
                too_close = atoms_too_close_two_sets(top, slab, self.blmin)

        if count == maxcount:
            return None

        mutant = slab + top
        return mutant


class PermutationMutation(OffspringCreator):
    """Mutation that permutes a percentage of the atom types in the cluster.

    Parameters:

    n_top: Number of atoms optimized by the GA.

    probability: The probability with which an atom is permuted.

    test_dist_to_slab: whether to also make sure that the distances
        between the atoms and the slab satisfy the blmin.

    use_tags: if True, the atomic tags will be used to preserve
        molecular identity. Permutations will then happen
        at the molecular level, i.e. swapping the center-of-
        positions of two moieties while preserving their
        internal geometries.

    blmin: Dictionary defining the minimum distance between atoms
        after the permutation. If equal to None (the default),
        no such check is performed.

    rng: Random number generator
        By default numpy.random.
    """

    def __init__(self, n_top, probability=0.33, test_dist_to_slab=True,
                 use_tags=False, blmin=None, rng=np.random, verbose=False):
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.n_top = n_top
        self.probability = probability
        self.test_dist_to_slab = test_dist_to_slab
        self.use_tags = use_tags
        self.blmin = blmin

        self.descriptor = 'PermutationMutation'
        self.min_inputs = 1

    def get_new_individual(self, parents):
        f = parents[0]

        indi = self.mutate(f)
        if indi is None:
            return indi, 'mutation: permutation'

        indi = self.initialize_individual(f, indi)
        indi.info['data']['parents'] = [f.info['confid']]

        return self.finalize_individual(indi), 'mutation: permutation'

    def mutate(self, atoms):
        """Does the actual mutation."""
        N = len(atoms) if self.n_top is None else self.n_top
        slab = atoms[:len(atoms) - N]
        atoms = atoms[-N:]
        if self.use_tags:
            gather_atoms_by_tag(atoms)
        tags = atoms.get_tags() if self.use_tags else np.arange(N)
        pos_ref = atoms.get_positions()
        num = atoms.get_atomic_numbers()
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()
        symbols = atoms.get_chemical_symbols()

        unique_tags = np.unique(tags)
        n = len(unique_tags)
        swaps = int(np.ceil(n * self.probability / 2.))

        sym = []
        for tag in unique_tags:
            indices = np.where(tags == tag)[0]
            s = ''.join([symbols[j] for j in indices])
            sym.append(s)

        assert len(np.unique(sym)) > 1, \
            'Permutations with one atom (or molecule) type is not valid'

        count = 0
        maxcount = 1000
        too_close = True
        while too_close and count < maxcount:
            count += 1
            pos = pos_ref.copy()
            for _ in range(swaps):
                i = j = 0
                while sym[i] == sym[j]:
                    i = self.rng.randint(0, high=n)
                    j = self.rng.randint(0, high=n)
                ind1 = np.where(tags == i)
                ind2 = np.where(tags == j)
                cop1 = np.mean(pos[ind1], axis=0)
                cop2 = np.mean(pos[ind2], axis=0)
                pos[ind1] += cop2 - cop1
                pos[ind2] += cop1 - cop2

            top = Atoms(num, positions=pos, cell=cell, pbc=pbc, tags=tags)
            if self.blmin is None:
                too_close = False
            else:
                too_close = atoms_too_close(
                    top, self.blmin, use_tags=self.use_tags)
                if not too_close and self.test_dist_to_slab:
                    too_close = atoms_too_close_two_sets(top, slab, self.blmin)

        if count == maxcount:
            return None

        mutant = slab + top
        return mutant


class MirrorMutation(OffspringCreator):
    """A mirror mutation, as described in
    TO BE PUBLISHED.

    This mutation mirrors half of the cluster in a
    randomly oriented cutting plane discarding the other half.

    Parameters:

    blmin: Dictionary defining the minimum allowed
        distance between atoms.

    n_top: Number of atoms the GA optimizes.

    reflect: Defines if the mirrored half is also reflected
        perpendicular to the mirroring plane.

    rng: Random number generator
        By default numpy.random.
    """

    def __init__(self, blmin, n_top, reflect=False, rng=np.random,
                 verbose=False):
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.blmin = blmin
        self.n_top = n_top
        self.reflect = reflect

        self.descriptor = 'MirrorMutation'
        self.min_inputs = 1

    def get_new_individual(self, parents):
        f = parents[0]

        indi = self.mutate(f)
        if indi is None:
            return indi, 'mutation: mirror'

        indi = self.initialize_individual(f, indi)
        indi.info['data']['parents'] = [f.info['confid']]

        return self.finalize_individual(indi), 'mutation: mirror'

    def mutate(self, atoms):
        """ Do the mutation of the atoms input. """
        reflect = self.reflect
        tc = True
        slab = atoms[0:len(atoms) - self.n_top]
        top = atoms[len(atoms) - self.n_top: len(atoms)]
        num = top.numbers
        unique_types = list(set(num))
        nu = {u: sum(num == u) for u in unique_types}
        n_tries = 1000
        counter = 0
        changed = False

        while tc and counter < n_tries:
            counter += 1
            cand = top.copy()
            pos = cand.get_positions()

            cm = np.average(top.get_positions(), axis=0)

            # first select a randomly oriented cutting plane
            theta = pi * self.rng.random()
            phi = 2. * pi * self.rng.random()
            n = (cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta))
            n = np.array(n)

            # Calculate all atoms signed distance to the cutting plane
            D = []
            for (i, p) in enumerate(pos):
                d = np.dot(p - cm, n)
                D.append((i, d))

            # Sort the atoms by their signed distance
            D.sort(key=lambda x: x[1])
            nu_taken = {}

            # Select half of the atoms needed for a full cluster
            p_use = []
            n_use = []
            for (i, d) in D:
                if num[i] not in nu_taken.keys():
                    nu_taken[num[i]] = 0
                if nu_taken[num[i]] < nu[num[i]] / 2.:
                    p_use.append(pos[i])
                    n_use.append(num[i])
                    nu_taken[num[i]] += 1

            # calculate the mirrored position and add these.
            pn = []
            for p in p_use:
                pt = p - 2. * np.dot(p - cm, n) * n
                if reflect:
                    pt = -pt + 2 * cm + 2 * n * np.dot(pt - cm, n)
                pn.append(pt)

            n_use.extend(n_use)
            p_use.extend(pn)

            # In the case of an uneven number of
            # atoms we need to add one extra
            for n in nu:
                if nu[n] % 2 == 0:
                    continue
                while sum(n_use == n) > nu[n]:
                    for i in range(int(len(n_use) / 2), len(n_use)):
                        if n_use[i] == n:
                            del p_use[i]
                            del n_use[i]
                            break
                assert sum(n_use == n) == nu[n]

            # Make sure we have the correct number of atoms
            # and rearrange the atoms so they are in the right order
            for i in range(len(n_use)):
                if num[i] == n_use[i]:
                    continue
                for j in range(i + 1, len(n_use)):
                    if n_use[j] == num[i]:
                        tn = n_use[i]
                        tp = p_use[i]
                        n_use[i] = n_use[j]
                        p_use[i] = p_use[j]
                        p_use[j] = tp
                        n_use[j] = tn

            # Finally we check that nothing is too close in the end product.
            cand = Atoms(num, p_use, cell=slab.get_cell(), pbc=slab.get_pbc())

            tc = atoms_too_close(cand, self.blmin)
            if tc:
                continue
            tc = atoms_too_close_two_sets(slab, cand, self.blmin)

            if not changed and counter > n_tries // 2:
                reflect = not reflect
                changed = True

            tot = slab + cand

        if counter == n_tries:
            return None
        return tot


class StrainMutation(OffspringCreator):
    """ Mutates a candidate by applying a randomly generated strain.

    For more information, see also:

      * :doi:`Glass, Oganov, Hansen, Comp. Phys. Comm. 175 (2006) 713-720
        <10.1016/j.cpc.2006.07.020>`

      * :doi:`Lonie, Zurek, Comp. Phys. Comm. 182 (2011) 372-387
        <10.1016/j.cpc.2010.07.048>`

    After initialization of the mutation, a scaling volume
    (to which each mutated structure is scaled before checking the
    constraints) is typically generated from the population,
    which is then also occasionally updated in the course of the
    GA run.

    Parameters:

    blmin: dict
        The closest allowed interatomic distances on the form:
        {(Z, Z*): dist, ...}, where Z and Z* are atomic numbers.

    cellbounds: ase.ga.utilities.CellBounds instance
        Describes limits on the cell shape, see
        :class:`~ase.ga.utilities.CellBounds`.

    stddev: float
        Standard deviation used in the generation of the
        strain matrix elements.

    number_of_variable_cell_vectors: int (default 3)
        The number of variable cell vectors (1, 2 or 3).
        To keep things simple, it is the 'first' vectors which
        will be treated as variable, i.e. the 'a' vector in the
        univariate case, the 'a' and 'b' vectors in the bivariate
        case, etc.

    use_tags: boolean
        Whether to use the atomic tags to preserve molecular identity.

    rng: Random number generator
        By default numpy.random.
    """

    def __init__(self, blmin, cellbounds=None, stddev=0.7,
                 number_of_variable_cell_vectors=3, use_tags=False,
                 rng=np.random, verbose=False):
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.blmin = blmin
        self.cellbounds = cellbounds
        self.stddev = stddev
        self.number_of_variable_cell_vectors = number_of_variable_cell_vectors
        self.use_tags = use_tags

        self.scaling_volume = None
        self.descriptor = 'StrainMutation'
        self.min_inputs = 1

    def update_scaling_volume(self, population, w_adapt=0.5, n_adapt=0):
        """Function to initialize or update the scaling volume in a GA run.

        w_adapt: weight of the new vs the old scaling volume

        n_adapt: number of best candidates in the population that
                 are used to calculate the new scaling volume
        """
        if not n_adapt:
            # if not set, take best 20% of the population
            n_adapt = int(np.ceil(0.2 * len(population)))
        v_new = np.mean([a.get_volume() for a in population[:n_adapt]])

        if not self.scaling_volume:
            self.scaling_volume = v_new
        else:
            volumes = [self.scaling_volume, v_new]
            weights = [1 - w_adapt, w_adapt]
            self.scaling_volume = np.average(volumes, weights=weights)

    def get_new_individual(self, parents):
        f = parents[0]

        indi = self.mutate(f)
        if indi is None:
            return indi, 'mutation: strain'

        indi = self.initialize_individual(f, indi)
        indi.info['data']['parents'] = [f.info['confid']]

        return self.finalize_individual(indi), 'mutation: strain'

    def mutate(self, atoms):
        """ Does the actual mutation. """
        cell_ref = atoms.get_cell()
        pos_ref = atoms.get_positions()

        if self.scaling_volume is None:
            # The scaling_volume has not been set (yet),
            # so we give it the same volume as the parent
            vol_ref = atoms.get_volume()
        else:
            vol_ref = self.scaling_volume

        if self.use_tags:
            tags = atoms.get_tags()
            gather_atoms_by_tag(atoms)
            pos = atoms.get_positions()

        mutant = atoms.copy()

        count = 0
        too_close = True
        maxcount = 1000
        while too_close and count < maxcount:
            count += 1

            # generating the strain matrix:
            strain = np.identity(3)
            for i in range(self.number_of_variable_cell_vectors):
                for j in range(i + 1):
                    r = self.rng.normal(loc=0., scale=self.stddev)
                    if i == j:
                        strain[i, j] += r
                    else:
                        epsilon = 0.5 * r
                        strain[i, j] += epsilon
                        strain[j, i] += epsilon

            # applying the strain:
            cell_new = np.dot(strain, cell_ref)

            # convert the submatrix with the variable cell vectors
            # to a lower triangular form
            cell_new = calc_rotated_cell(cell_new)
            for i in range(self.number_of_variable_cell_vectors, 3):
                cell_new[i] = cell_ref[i]

            cell_new = Cell(cell_new)

            # volume scaling:
            if self.number_of_variable_cell_vectors > 0:
                scaling = vol_ref / cell_new.volume
                scaling **= 1. / self.number_of_variable_cell_vectors
                cell_new[:self.number_of_variable_cell_vectors] *= scaling

            # check cell dimensions:
            if not self.cellbounds.is_within_bounds(cell_new):
                continue

            # ensure non-variable cell vectors are indeed unchanged
            for i in range(self.number_of_variable_cell_vectors, 3):
                assert np.allclose(cell_new[i], cell_ref[i])

            # check that the volume is correct
            assert np.isclose(vol_ref, cell_new.volume)

            # apply the new unit cell and scale
            # the atomic positions accordingly
            mutant.set_cell(cell_ref, scale_atoms=False)

            if self.use_tags:
                transfo = np.linalg.solve(cell_ref, cell_new)
                for tag in np.unique(tags):
                    select = np.where(tags == tag)
                    cop = np.mean(pos[select], axis=0)
                    disp = np.dot(cop, transfo) - cop
                    mutant.positions[select] += disp
            else:
                mutant.set_positions(pos_ref)

            mutant.set_cell(cell_new, scale_atoms=not self.use_tags)
            mutant.wrap()

            # check the interatomic distances
            too_close = atoms_too_close(mutant, self.blmin,
                                        use_tags=self.use_tags)

        if count == maxcount:
            mutant = None

        return mutant


class PermuStrainMutation(CombinationMutation):
    """Combination of PermutationMutation and StrainMutation.

    For more information, see also:

      * :doi:`Lonie, Zurek, Comp. Phys. Comm. 182 (2011) 372-387
        <10.1016/j.cpc.2010.07.048>`

    Parameters:

    permutationmutation: OffspringCreator instance
        A mutation that permutes atom types.

    strainmutation: OffspringCreator instance
        A mutation that mutates by straining.
    """

    def __init__(self, permutationmutation, strainmutation, verbose=False):
        super().__init__(permutationmutation,
                         strainmutation,
                         verbose=verbose)
        self.descriptor = 'permustrain'


class RotationalMutation(OffspringCreator):
    """Mutates a candidate by applying random rotations
    to multi-atom moieties in the structure (atoms with
    the same tag are considered part of one such moiety).

    Only performs whole-molecule rotations, no internal
    rotations.

    For more information, see also:

      * `Zhu Q., Oganov A.R., Glass C.W., Stokes H.T,
        Acta Cryst. (2012), B68, 215-226.`__

        __ https://dx.doi.org/10.1107/S0108768112017466

    Parameters:

    blmin: dict
        The closest allowed interatomic distances on the form:
        {(Z, Z*): dist, ...}, where Z and Z* are atomic numbers.

    n_top: int or None
        The number of atoms to optimize (None = include all).

    fraction: float
        Fraction of the moieties to be rotated.

    tags: None or list of integers
        Specifies, respectively, whether all moieties or only those
        with matching tags are eligible for rotation.

    min_angle: float
        Minimal angle (in radians) for each rotation;
        should lie in the interval [0, pi].

    test_dist_to_slab: boolean
        Whether also the distances to the slab
        should be checked to satisfy the blmin.

    rng: Random number generator
        By default numpy.random.
    """

    def __init__(self, blmin, n_top=None, fraction=0.33, tags=None,
                 min_angle=1.57, test_dist_to_slab=True, rng=np.random,
                 verbose=False):
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.blmin = blmin
        self.n_top = n_top
        self.fraction = fraction
        self.tags = tags
        self.min_angle = min_angle
        self.test_dist_to_slab = test_dist_to_slab
        self.descriptor = 'RotationalMutation'
        self.min_inputs = 1

    def get_new_individual(self, parents):
        f = parents[0]

        indi = self.mutate(f)
        if indi is None:
            return indi, 'mutation: rotational'

        indi = self.initialize_individual(f, indi)
        indi.info['data']['parents'] = [f.info['confid']]

        return self.finalize_individual(indi), 'mutation: rotational'

    def mutate(self, atoms):
        """Does the actual mutation."""
        N = len(atoms) if self.n_top is None else self.n_top
        slab = atoms[:len(atoms) - N]
        atoms = atoms[-N:]

        mutant = atoms.copy()
        gather_atoms_by_tag(mutant)
        pos = mutant.get_positions()
        tags = mutant.get_tags()
        eligible_tags = tags if self.tags is None else self.tags

        indices = {}
        for tag in np.unique(tags):
            hits = np.where(tags == tag)[0]
            if len(hits) > 1 and tag in eligible_tags:
                indices[tag] = hits

        n_rot = int(np.ceil(len(indices) * self.fraction))
        chosen_tags = self.rng.choice(list(indices.keys()), size=n_rot,
                                      replace=False)

        too_close = True
        count = 0
        maxcount = 10000
        while too_close and count < maxcount:
            newpos = np.copy(pos)
            for tag in chosen_tags:
                p = np.copy(newpos[indices[tag]])
                cop = np.mean(p, axis=0)

                if len(p) == 2:
                    line = (p[1] - p[0]) / np.linalg.norm(p[1] - p[0])
                    while True:
                        axis = self.rng.random(3)
                        axis /= np.linalg.norm(axis)
                        a = np.arccos(np.dot(axis, line))
                        if np.pi / 4 < a < np.pi * 3 / 4:
                            break
                else:
                    axis = self.rng.random(3)
                    axis /= np.linalg.norm(axis)

                angle = self.min_angle
                angle += 2 * (np.pi - self.min_angle) * self.rng.random()

                m = get_rotation_matrix(axis, angle)
                newpos[indices[tag]] = np.dot(m, (p - cop).T).T + cop

            mutant.set_positions(newpos)
            mutant.wrap()
            too_close = atoms_too_close(mutant, self.blmin, use_tags=True)
            count += 1

            if not too_close and self.test_dist_to_slab:
                too_close = atoms_too_close_two_sets(slab, mutant, self.blmin)

        if count == maxcount:
            mutant = None
        else:
            mutant = slab + mutant

        return mutant


class RattleRotationalMutation(CombinationMutation):
    """Combination of RattleMutation and RotationalMutation.

    Parameters:

    rattlemutation: OffspringCreator instance
        A mutation that rattles atoms.

    rotationalmutation: OffspringCreator instance
        A mutation that rotates moieties.
    """

    def __init__(self, rattlemutation, rotationalmutation, verbose=False):
        super().__init__(rattlemutation,
                         rotationalmutation,
                         verbose=verbose)
        self.descriptor = 'rattlerotational'
