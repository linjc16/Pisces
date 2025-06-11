# fmt: off

"""Tools for generating new random starting candidates."""
import numpy as np

from ase import Atoms
from ase.build import molecule
from ase.data import atomic_numbers
from ase.ga.utilities import (
    atoms_too_close,
    atoms_too_close_two_sets,
    closest_distances_generator,
)


class StartGenerator:
    """Class for generating random starting candidates.

    Its basic task consists of randomly placing atoms or
    molecules within a predescribed box, while respecting
    certain minimal interatomic distances.

    Depending on the problem at hand, certain box vectors
    may not be known or chosen beforehand, and hence also
    need to be generated at random. Common cases include
    bulk crystals, films and chains, with respectively
    3, 2 and 1 unknown cell vectors.

    Parameters:

    slab: Atoms object
        Specifies the cell vectors and periodic boundary conditions
        to be applied to the randomly generated structures.
        Any included atoms (e.g. representing an underlying slab)
        are copied to these new structures.
        Variable cell vectors (see number_of_variable_cell_vectors)
        will be ignored because these will be generated at random.

    blocks: list
        List of building units for the structure. Each item can be:

        * an integer: representing a single atom by its atomic number,
        * a string: for a single atom (a chemical symbol) or a
          molecule (name recognized by ase.build.molecule),
        * an Atoms object,
        * an (A, B) tuple or list where A is any of the above
          and B is the number of A units to include.

        A few examples:

        >>> blocks = ['Ti'] * 4 + ['O'] * 8
        >>> blocks = [('Ti', 4), ('O', 8)]
        >>> blocks = [('CO2', 3)]  # 3 CO2 molecules
        >>> co = Atoms('CO', positions=[[0, 0, 0], [1.4, 0, 0]])
        >>> blocks = [(co, 3)]

        Each individual block (single atom or molecule) in the
        randomly generated candidates is given a unique integer
        tag. These can be used to preserve the molecular identity
        of these subunits.

    blmin: dict or float
        Dictionary with minimal interatomic distances.
        If a number is provided instead, the dictionary will
        be generated with this ratio of covalent bond radii.
        Note: when preserving molecular identity (see use_tags),
        the blmin dict will (naturally) only be applied
        to intermolecular distances (not the intramolecular
        ones).

    number_of_variable_cell_vectors: int (default 0)
        The number of variable cell vectors (0, 1, 2 or 3).
        To keep things simple, it is the 'first' vectors which
        will be treated as variable, i.e. the 'a' vector in the
        univariate case, the 'a' and 'b' vectors in the bivariate
        case, etc.

    box_to_place_in: [list, list of lists] (default None)
        The box in which the atoms can be placed.
        The default (None) means the box is equal to the
        entire unit cell of the 'slab' object.
        In many cases, however, smaller boxes are desired
        (e.g. for adsorbates on a slab surface or for isolated
        clusters). Then, box_to_place_in can be set as
        [p0, [v1, v2, v3]] with positions being generated as
        p0 + r1 * v1 + r2 * v2 + r3 + v3.
        In case of one or more variable cell vectors,
        the corresponding items in p0/v1/v2/v3 will be ignored.

    box_volume: int or float or None (default)
        Initial guess for the box volume in cubic Angstrom
        (used in generating the variable cell vectors).
        Typical values in the solid state are 8-12 A^3 per atom.
        If there are no variable cell vectors, the default None
        is required (box volume equal to the box_to_place_in
        volume).

    splits: dict or None
        Splitting scheme for increasing the translational symmetry
        in the random candidates, based on:

        * `Lyakhov, Oganov, Valle, Comp. Phys. Comm. 181 (2010) 1623-32`__

        __ http://dx.doi.org/10.1016/j.cpc.2010.06.007

        This should be a dict specifying the relative probabilities
        for each split, written as tuples. For example,

        >>> splits = {(2,): 3, (1,): 1}

        This means that, for each structure, either a splitting
        factor of 2 is applied to one randomly chosen axis,
        or a splitting factor of 1 is applied (i.e., no splitting).
        The probability ratio of the two scenararios will be 3:1,
        i.e. 75% chance for the former and 25% chance for the latter
        splitting scheme. Only the directions in which the 'slab'
        object is periodic are eligible for splitting.

        To e.g. always apply splitting factors of 2 and 3 along two
        randomly chosen axes:

        >>> splits = {(2, 3): 1}

        By default, no splitting is applied (splits = None = {(1,): 1}).

    cellbounds: ase.ga.utilities.CellBounds instance
        Describing limits on the cell shape, see
        :class:`~ase.ga.utilities.CellBounds`.
        Note that it only make sense to impose conditions
        regarding cell vectors which have been marked as
        variable (see number_of_variable_cell_vectors).

    test_dist_to_slab: bool (default True)
        Whether to make sure that the distances between
        the atoms and the slab satisfy the blmin.

    test_too_far: bool (default True)
        Whether to also make sure that there are no isolated
        atoms or molecules with nearest-neighbour bond lengths
        larger than 2x the value in the blmin dict.

    rng: Random number generator
        By default numpy.random.
    """

    def __init__(self, slab, blocks, blmin, number_of_variable_cell_vectors=0,
                 box_to_place_in=None, box_volume=None, splits=None,
                 cellbounds=None, test_dist_to_slab=True, test_too_far=True,
                 rng=np.random):

        self.slab = slab

        self.blocks = []
        for item in blocks:
            if isinstance(item, (tuple, list)):
                assert len(item) == 2, 'Item length %d != 2' % len(item)
                block, count = item
            else:
                block, count = item, 1

            # Convert block into Atoms object
            if isinstance(block, Atoms):
                pass
            elif block in atomic_numbers:
                block = Atoms(block)
            elif isinstance(block, str):
                block = molecule(block)
            elif block in atomic_numbers.values():
                block = Atoms(numbers=[block])
            else:
                raise ValueError('Cannot parse this block:', block)

            # Add to self.blocks, taking into account that
            # we want to group the same blocks together.
            # This is important for the cell splitting.
            for i, (b, c) in enumerate(self.blocks):
                if block == b:
                    self.blocks[i][1] += count
                    break
            else:
                self.blocks.append([block, count])

        if isinstance(blmin, dict):
            self.blmin = blmin
        else:
            numbers = np.unique([b.get_atomic_numbers() for b in self.blocks])
            self.blmin = closest_distances_generator(
                numbers,
                ratio_of_covalent_radii=blmin)

        self.number_of_variable_cell_vectors = number_of_variable_cell_vectors
        assert self.number_of_variable_cell_vectors in range(4)
        if len(self.slab) > 0:
            msg = 'Including atoms in the slab only makes sense'
            msg += ' if there are no variable unit cell vectors'
            assert self.number_of_variable_cell_vectors == 0, msg
        for i in range(self.number_of_variable_cell_vectors):
            msg = f'Unit cell {("abc"[i])}-vector is marked as variable '
            msg += 'and slab must then also be periodic in this direction'
            assert self.slab.pbc[i], msg

        if box_to_place_in is None:
            p0 = np.array([0., 0., 0.])
            cell = self.slab.get_cell()
            self.box_to_place_in = [p0, [cell[0, :], cell[1, :], cell[2, :]]]
        else:
            self.box_to_place_in = box_to_place_in

        if box_volume is None:
            assert self.number_of_variable_cell_vectors == 0
            box_volume = abs(np.linalg.det(self.box_to_place_in[1]))
        else:
            assert self.number_of_variable_cell_vectors > 0
        self.box_volume = box_volume
        assert self.box_volume > 0

        if splits is None:
            splits = {(1,): 1}
        tot = sum(v for v in splits.values())
        self.splits = {k: v * 1. / tot for k, v in splits.items()}

        self.cellbounds = cellbounds
        self.test_too_far = test_too_far
        self.test_dist_to_slab = test_dist_to_slab
        self.rng = rng

    def get_new_candidate(self, maxiter=None):
        """Returns a new candidate.

        maxiter: upper bound on the total number of times
             the random position generator is called
             when generating the new candidate.

             By default (maxiter=None) no such bound
             is imposed. If the generator takes too
             long time to create a new candidate, it
             may be suitable to specify a finite value.
             When the bound is exceeded, None is returned.
        """
        pbc = self.slab.get_pbc()

        # Choose cell splitting
        r = self.rng.random()
        cumprob = 0
        for split, prob in self.splits.items():
            cumprob += prob
            if cumprob > r:
                break

        # Choose direction(s) along which to split
        # and by how much
        directions = [i for i in range(3) if pbc[i]]
        repeat = [1, 1, 1]
        if len(directions) > 0:
            for number in split:
                d = self.rng.choice(directions)
                repeat[d] = number
        repeat = tuple(repeat)

        # Generate the 'full' unit cell
        # for the eventual candidates
        cell = self.generate_unit_cell(repeat)
        if self.number_of_variable_cell_vectors == 0:
            assert np.allclose(cell, self.slab.get_cell())

        # Make the smaller 'box' in which we are
        # allowed to place the atoms and which will
        # then be repeated to fill the 'full' unit cell
        box = np.copy(cell)
        for i in range(self.number_of_variable_cell_vectors, 3):
            box[i] = np.array(self.box_to_place_in[1][i])
        box /= np.array([repeat]).T

        # Here we gather the (reduced) number of blocks
        # to put in the smaller box, and the 'surplus'
        # occurring when the block count is not divisible
        # by the number of repetitions.
        # E.g. if we have a ('Ti', 4) block and do a
        # [2, 3, 1] repetition, we employ a ('Ti', 1)
        # block in the smaller box and delete 2 out 6
        # Ti atoms afterwards
        nrep = int(np.prod(repeat))
        blocks, ids, surplus = [], [], []
        for i, (block, count) in enumerate(self.blocks):
            count_part = int(np.ceil(count * 1. / nrep))
            blocks.extend([block] * count_part)
            surplus.append(nrep * count_part - count)
            ids.extend([i] * count_part)

        N_blocks = len(blocks)

        # Shuffle the ordering so different blocks
        # are added in random order
        order = np.arange(N_blocks)
        self.rng.shuffle(order)
        blocks = [blocks[i] for i in order]
        ids = np.array(ids)[order]

        # Add blocks one by one until we have found
        # a valid candidate
        blmin = self.blmin
        blmin_too_far = {key: 2 * val for key, val in blmin.items()}

        niter = 0
        while maxiter is None or niter < maxiter:
            cand = Atoms('', cell=box, pbc=pbc)

            for i in range(N_blocks):
                atoms = blocks[i].copy()
                atoms.set_tags(i)
                atoms.set_pbc(pbc)
                atoms.set_cell(box, scale_atoms=False)

                while maxiter is None or niter < maxiter:
                    niter += 1
                    cop = atoms.get_positions().mean(axis=0)
                    pos = np.dot(self.rng.random((1, 3)), box)
                    atoms.translate(pos - cop)

                    if len(atoms) > 1:
                        # Apply a random rotation to multi-atom blocks
                        phi, theta, psi = 360 * self.rng.random(3)
                        atoms.euler_rotate(phi=phi, theta=0.5 * theta, psi=psi,
                                           center=pos)

                    if not atoms_too_close_two_sets(cand, atoms, blmin):
                        cand += atoms
                        break
                else:
                    # Reached maximum iteration number
                    # Break out of the for loop above
                    cand = None
                    break

            if cand is None:
                # Exit the main while loop
                break

            # Rebuild the candidate after repeating,
            # randomly deleting surplus blocks and
            # sorting back to the original order
            cand_full = cand.repeat(repeat)

            tags_full = cand_full.get_tags()
            for i in range(nrep):
                tags_full[len(cand) * i:len(cand) * (i + 1)] += i * N_blocks
            cand_full.set_tags(tags_full)

            cand = Atoms('', cell=cell, pbc=pbc)
            ids_full = np.tile(ids, nrep)

            tag_counter = 0
            if len(self.slab) > 0:
                tag_counter = int(max(self.slab.get_tags())) + 1

            for i, (block, count) in enumerate(self.blocks):
                tags = np.where(ids_full == i)[0]
                bad = self.rng.choice(tags, size=surplus[i], replace=False)
                for tag in tags:
                    if tag not in bad:
                        select = [a.index for a in cand_full if a.tag == tag]
                        atoms = cand_full[select]  # is indeed a copy!
                        atoms.set_tags(tag_counter)
                        assert len(atoms) == len(block)
                        cand += atoms
                        tag_counter += 1

            for i in range(self.number_of_variable_cell_vectors, 3):
                cand.positions[:, i] += self.box_to_place_in[0][i]

            # By construction, the minimal interatomic distances
            # within the structure should already be respected
            assert not atoms_too_close(cand, blmin, use_tags=True), \
                'This is not supposed to happen; please report this bug'

            if self.test_dist_to_slab and len(self.slab) > 0:
                if atoms_too_close_two_sets(self.slab, cand, blmin):
                    continue

            if self.test_too_far:
                tags = cand.get_tags()

                for tag in np.unique(tags):
                    too_far = True
                    indices_i = np.where(tags == tag)[0]
                    indices_j = np.where(tags != tag)[0]
                    too_far = not atoms_too_close_two_sets(cand[indices_i],
                                                           cand[indices_j],
                                                           blmin_too_far)

                    if too_far and len(self.slab) > 0:
                        # the block is too far from the rest
                        # but might still be sufficiently
                        # close to the slab
                        too_far = not atoms_too_close_two_sets(cand[indices_i],
                                                               self.slab,
                                                               blmin_too_far)
                    if too_far:
                        break
                else:
                    too_far = False

                if too_far:
                    continue

            # Passed all the tests
            cand = self.slab + cand
            cand.set_cell(cell, scale_atoms=False)
            break
        else:
            # Reached max iteration count in the while loop
            return None

        return cand

    def generate_unit_cell(self, repeat):
        """Generates a random unit cell.

        For this, we use the vectors in self.slab.cell
        in the fixed directions and randomly generate
        the variable ones. For such a cell to be valid,
        it has to satisfy the self.cellbounds constraints.

        The cell will also be such that the volume of the
        box in which the atoms can be placed (box limits
        described by self.box_to_place_in) is equal to
        self.box_volume.

        Parameters:

        repeat: tuple of 3 integers
            Indicates by how much each cell vector
            will later be reduced by cell splitting.

            This is used to ensure that the original
            cell is large enough so that the cell lengths
            of the smaller cell exceed the largest
            (X,X)-minimal-interatomic-distance in self.blmin.
        """
        # Find the minimal cell length 'Lmin'
        # that we need in order to ensure that
        # an added atom or molecule will never
        # be 'too close to itself'
        Lmin = 0.
        for atoms, count in self.blocks:
            dist = atoms.get_all_distances(mic=False, vector=False)
            num = atoms.get_atomic_numbers()

            for i in range(len(atoms)):
                dist[i, i] += self.blmin[(num[i], num[i])]
                for j in range(i):
                    bl = self.blmin[(num[i], num[j])]
                    dist[i, j] += bl
                    dist[j, i] += bl

            L = np.max(dist)
            if L > Lmin:
                Lmin = L

        # Generate a suitable unit cell
        valid = False
        while not valid:
            cell = np.zeros((3, 3))

            for i in range(self.number_of_variable_cell_vectors):
                # on-diagonal values
                cell[i, i] = self.rng.random() * np.cbrt(self.box_volume)
                cell[i, i] *= repeat[i]
                for j in range(i):
                    # off-diagonal values
                    cell[i, j] = (self.rng.random() - 0.5) * cell[i - 1, i - 1]

            # volume scaling
            for i in range(self.number_of_variable_cell_vectors, 3):
                cell[i] = self.box_to_place_in[1][i]

            if self.number_of_variable_cell_vectors > 0:
                volume = abs(np.linalg.det(cell))
                scaling = self.box_volume / volume
                scaling **= 1. / self.number_of_variable_cell_vectors
                cell[:self.number_of_variable_cell_vectors] *= scaling

            for i in range(self.number_of_variable_cell_vectors, 3):
                cell[i] = self.slab.get_cell()[i]

            # bounds checking
            valid = True

            if self.cellbounds is not None:
                if not self.cellbounds.is_within_bounds(cell):
                    valid = False

            if valid:
                for i in range(3):
                    if np.linalg.norm(cell[i]) < repeat[i] * Lmin:
                        assert self.number_of_variable_cell_vectors > 0
                        valid = False

        return cell
