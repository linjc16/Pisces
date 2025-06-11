import numpy as np

from ase import Atoms
from ase.build import molecule
from ase.ga.cutandsplicepairing import CutAndSplicePairing
from ase.ga.standardmutations import StrainMutation
from ase.ga.startgenerator import StartGenerator
from ase.ga.utilities import (
    CellBounds,
    atoms_too_close,
    closest_distances_generator,
)


def test_film_operators(seed):

    # set up the random number generator
    rng = np.random.RandomState(seed)

    slab = Atoms('', cell=(0, 0, 15), pbc=[True, True, False])

    cation, anion = 'Mg', molecule('OH')
    d_oh = anion.get_distance(0, 1)
    blocks = [(cation, 4), (anion, 8)]
    n_top = 4 + 8 * len(anion)

    use_tags = True
    num_vcv = 2
    box_volume = 8. * n_top

    blmin = closest_distances_generator(atom_numbers=[1, 8, 12],
                                        ratio_of_covalent_radii=0.6)

    cellbounds = CellBounds(bounds={'phi': [0.1 * 180., 0.9 * 180.],
                                    'chi': [0.1 * 180., 0.9 * 180.],
                                    'psi': [0.1 * 180., 0.9 * 180.],
                                    'a': [2, 8], 'b': [2, 8]})

    box_to_place_in = [[None, None, 3.], [None, None, [0., 0., 5.]]]

    sg = StartGenerator(slab, blocks, blmin, box_volume=box_volume,
                        splits={(2, 1): 1}, box_to_place_in=box_to_place_in,
                        number_of_variable_cell_vectors=num_vcv,
                        cellbounds=cellbounds, test_too_far=True,
                        test_dist_to_slab=False, rng=rng)

    parents = []
    for i in range(2):
        a = None
        while a is None:
            a = sg.get_new_candidate()

        a.info['confid'] = i
        parents.append(a)

        assert len(a) == n_top
        assert len(np.unique(a.get_tags())) == 4 + 8
        assert np.allclose(a.get_pbc(), slab.get_pbc())

        p = a.get_positions()
        assert np.min(p[:, 2]) > 3. - 0.5 * d_oh
        assert np.max(p[:, 2]) < 3. + 5. + 0.5 * d_oh
        assert not atoms_too_close(a, blmin, use_tags=use_tags)

        c = a.get_cell()
        assert np.allclose(c[2], slab.get_cell()[2])
        assert cellbounds.is_within_bounds(c)

        v = a.get_volume() * 5. / 15.
        assert abs(v - box_volume) < 1e-5

    # Test cut-and-splice pairing and strain mutation
    pairing = CutAndSplicePairing(slab, n_top, blmin,
                                  number_of_variable_cell_vectors=num_vcv,
                                  p1=1., p2=0., minfrac=0.15,
                                  cellbounds=cellbounds, use_tags=use_tags,
                                  rng=rng)

    strainmut = StrainMutation(blmin, cellbounds=cellbounds,
                               number_of_variable_cell_vectors=num_vcv,
                               use_tags=use_tags, rng=rng)
    strainmut.update_scaling_volume(parents)

    for operator in [pairing, strainmut]:
        child = None
        while child is None:
            child, _desc = operator.get_new_individual(parents)

        assert not atoms_too_close(child, blmin, use_tags=use_tags)
        cell = child.get_cell()
        assert cellbounds.is_within_bounds(cell)
        assert np.allclose(cell[2], slab.get_cell()[2])
