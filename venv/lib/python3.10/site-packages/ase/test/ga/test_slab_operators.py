import numpy as np
import pytest

from ase.build import fcc111
from ase.ga.slab_operators import (
    CutSpliceSlabCrossover,
    NeighborhoodElementMutation,
    RandomCompositionMutation,
    RandomElementMutation,
    RandomSlabPermutation,
)


@pytest.fixture()
def cu_slab():
    a = 1
    size = (2, 4, 3)
    p1 = fcc111('Cu', size, orthogonal=True, a=a)
    p1.info['confid'] = 1
    return p1


def test_cut_splice(seed, cu_slab):
    # set up the random number generator
    rng = np.random.RandomState(seed)

    ratio = .4
    op = CutSpliceSlabCrossover(min_ratio=ratio, rng=rng)
    p1 = cu_slab
    natoms = len(p1)

    p2 = cu_slab.copy()
    p2.symbols = ['Au'] * natoms

    p2.info['confid'] = 2
    child, desc = op.get_new_individual([p1, p2])
    assert desc == 'CutSpliceSlabCrossover: Parents 1 2'

    # Check the ratio of elements
    syms = child.get_chemical_symbols()
    new_ratio = syms.count('Au') / natoms
    assert new_ratio > ratio and new_ratio < 1 - ratio

    op = CutSpliceSlabCrossover(element_pools=['Cu', 'Au'],
                                allowed_compositions=[(12, 12)], rng=rng)
    child = op.operate(p1, p2)
    assert child.get_chemical_symbols().count('Au') == 12


def test_random_composition_mutation(seed, cu_slab):
    # set up the random number generator
    rng = np.random.RandomState(seed)

    p1 = cu_slab
    p1.symbols[3] = 'Au'
    op = RandomCompositionMutation(element_pools=['Cu', 'Au'],
                                   allowed_compositions=[(12, 12), (18, 6)],
                                   rng=rng)
    child, _ = op.get_new_individual([p1])
    no_Au = (child.symbols == 'Au').sum()
    assert no_Au in [6, 12]

    op = RandomCompositionMutation(element_pools=['Cu', 'Au'], rng=rng)
    child2 = op.operate(child)
    # Make sure we have gotten a new stoichiometry
    assert (child2.symbols == 'Au').sum() != no_Au


def test_random_element_mutation(seed, cu_slab):
    # set up the random number generator
    rng = np.random.RandomState(seed)

    op = RandomElementMutation(element_pools=[['Cu', 'Au']], rng=rng)

    child, _desc = op.get_new_individual([cu_slab.copy()])

    assert (child.symbols == 'Au').sum() == 24


def test_neighborhood_element_mutation(seed, cu_slab):
    # set up the random number generator
    rng = np.random.RandomState(seed)

    op = NeighborhoodElementMutation(element_pools=[['Cu', 'Ni', 'Au']],
                                     rng=rng)

    child, _desc = op.get_new_individual([cu_slab])

    assert (child.symbols == 'Ni').sum() == 24


def test_random_permutation(seed, cu_slab):
    # set up the random number generator
    rng = np.random.RandomState(seed)

    p1 = cu_slab
    p1.symbols[:8] = 'Au'

    op = RandomSlabPermutation(rng=rng)
    child, _desc = op.get_new_individual([p1])

    assert (child.symbols == 'Au').sum() == 8
    assert sum(p1.numbers == child.numbers) == 22
