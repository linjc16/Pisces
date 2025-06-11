import numpy as np
import pytest

from ase import Atoms
from ase.build import bulk
from ase.ga.cutandsplicepairing import CutAndSplicePairing
from ase.ga.offspring_creator import CombinationMutation
from ase.ga.ofp_comparator import OFPComparator
from ase.ga.soft_mutation import SoftMutation
from ase.ga.standardmutations import (
    PermutationMutation,
    RattleMutation,
    RattleRotationalMutation,
    RotationalMutation,
    StrainMutation,
)
from ase.ga.startgenerator import StartGenerator
from ase.ga.utilities import (
    CellBounds,
    atoms_too_close,
    closest_distances_generator,
)


@pytest.mark.slow()
def test_bulk_operators(seed, tmp_path):
    # set up the random number generator
    rng = np.random.RandomState(seed)

    h2 = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.75]])
    blocks = [('H', 4), ('H2O', 3), (h2, 2)]  # the building blocks
    volume = 40. * sum(x[1] for x in blocks)
    splits = {(2,): 1, (1,): 1}  # cell splitting scheme

    stoichiometry = []
    for block, count in blocks:
        if isinstance(block, str):
            stoichiometry += list(Atoms(block).numbers) * count
        else:
            stoichiometry += list(block.numbers) * count

    atom_numbers = list(set(stoichiometry))
    blmin = closest_distances_generator(atom_numbers=atom_numbers,
                                        ratio_of_covalent_radii=1.3)

    cellbounds = CellBounds(bounds={'phi': [30, 150], 'chi': [30, 150],
                                    'psi': [30, 150], 'a': [3, 50],
                                    'b': [3, 50], 'c': [3, 50]})

    slab = Atoms('', pbc=True)
    sg = StartGenerator(slab, blocks, blmin, box_volume=volume,
                        number_of_variable_cell_vectors=3,
                        cellbounds=cellbounds, splits=splits, rng=rng)

    # Generate 2 candidates
    a1 = sg.get_new_candidate()
    a1.info['confid'] = 1
    a2 = sg.get_new_candidate()
    a2.info['confid'] = 2

    # Define and test genetic operators
    n_top = len(a1)
    pairing = CutAndSplicePairing(slab, n_top, blmin, p1=1., p2=0.,
                                  minfrac=0.15,
                                  number_of_variable_cell_vectors=3,
                                  cellbounds=cellbounds, use_tags=True,
                                  rng=rng)

    a3, _desc = pairing.get_new_individual([a1, a2])
    cell = a3.get_cell()
    assert cellbounds.is_within_bounds(cell)
    assert not atoms_too_close(a3, blmin, use_tags=True)

    n_top = len(a1)
    strainmut = StrainMutation(blmin, stddev=0.7, cellbounds=cellbounds,
                               number_of_variable_cell_vectors=3,
                               use_tags=True, rng=rng)
    softmut = SoftMutation(blmin, bounds=[2., 5.], used_modes_file=None,
                           use_tags=True)  # no rng
    rotmut = RotationalMutation(blmin, fraction=0.3, min_angle=0.5 * np.pi,
                                rng=rng)
    rattlemut = RattleMutation(blmin, n_top, rattle_prop=0.3,
                               rattle_strength=0.5,
                               use_tags=True, test_dist_to_slab=False, rng=rng)
    rattlerotmut = RattleRotationalMutation(rattlemut, rotmut)  # no rng
    permut = PermutationMutation(n_top, probability=0.33,
                                 test_dist_to_slab=False,
                                 use_tags=True, blmin=blmin, rng=rng)
    combmut = CombinationMutation(rattlemut, rotmut, verbose=True)  # no rng
    mutations = [strainmut, softmut, rotmut,
                 rattlemut, rattlerotmut, permut, combmut]

    for i, mut in enumerate(mutations):
        a = [a1, a2][i % 2]
        a3 = None
        while a3 is None:
            a3, _desc = mut.get_new_individual([a])

        cell = a3.get_cell()
        assert cellbounds.is_within_bounds(cell)
        assert np.all(a3.numbers == a.numbers)
        assert not atoms_too_close(a3, blmin, use_tags=True)

    modes_file = tmp_path / 'modes.txt'
    softmut_with = SoftMutation(blmin, bounds=[2., 5.], use_tags=True,
                                used_modes_file=modes_file)  # no rng
    no_muts = 3
    for _ in range(no_muts):
        softmut_with.get_new_individual([a1])
    softmut_with.read_used_modes(modes_file)
    assert len(list(softmut_with.used_modes.values())[0]) == no_muts

    comparator = OFPComparator(recalculate=True)
    gold = bulk('Au') * (2, 2, 2)
    assert comparator.looks_like(gold, gold)

    # This move should not exceed the default threshold
    gc = gold.copy()
    gc[0].x += .1
    assert comparator.looks_like(gold, gc)

    # An additional step will exceed the threshold
    gc[0].x += .2
    assert not comparator.looks_like(gold, gc)
