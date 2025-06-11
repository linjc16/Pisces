import numpy as np

from ase import Atoms
from ase.ga import get_raw_score, set_raw_score
from ase.ga.element_crossovers import OnePointElementCrossover
from ase.ga.element_mutations import (
    MoveDownMutation,
    MoveLeftMutation,
    MoveRightMutation,
    MoveUpMutation,
    RandomElementMutation,
)


def test_element_operators(seed):

    # set up the random number generator
    rng = np.random.RandomState(seed)

    a1 = Atoms('SrSrSrBaClClClClBrBrBrBr')
    a1.info['confid'] = 1
    a2 = Atoms('CaCaMgBaFFFFFFFF')
    a2.info['confid'] = 2

    cations = ['Sr', 'Ba', 'Ca', 'Mg']
    anions = ['Cl', 'F', 'Br']
    op = OnePointElementCrossover([cations, anions], [3, 2], [.25, .5],
                                  rng=rng)

    a3, _desc = op.get_new_individual([a1, a2])

    syms = a3.get_chemical_symbols()
    assert len({i for i in syms if i in cations}) < 4
    assert len({i for i in syms if i in anions}) < 3

    op = RandomElementMutation([cations, anions], [3, 2], [.25, .5], rng=rng)
    a4, _desc = op.get_new_individual([a1])
    syms = a4.get_chemical_symbols()

    assert len({i for i in syms if i in cations}) < 4
    assert len({i for i in syms if i in anions}) < 3

    op = RandomElementMutation(anions, 2, .5, rng=rng)
    a4, _desc = op.get_new_individual([a2])
    syms = a4.get_chemical_symbols()

    assert len({i for i in syms if i in anions}) == 2

    a1 = Atoms('SrSrClClClCl')
    a1.info['confid'] = 1
    op = MoveDownMutation(cations, 2, .5, rng=rng)
    a2, _desc = op.get_new_individual([a1])
    a2.info['confid'] = 2

    syms = a2.get_chemical_symbols()
    assert 'Ba' in syms
    assert len(set(syms)) == 3

    op = MoveUpMutation(cations, 1, 1., rng=rng)
    a3, _desc = op.get_new_individual([a2])
    syms = a3.get_chemical_symbols()
    assert 'Ba' not in syms
    assert len(set(syms)) == 2

    cations = ['Co', 'Ni', 'Cu']
    a1 = Atoms('NiNiBrBr')
    a1.info['confid'] = 1
    op = MoveRightMutation(cations, 1, 1., rng=rng)
    a2, _desc = op.get_new_individual([a1])
    a2.info['confid'] = 2
    syms = a2.get_chemical_symbols()

    assert len(set(syms)) == 2
    assert len([i for i in syms if i == 'Cu']) == 2

    op = MoveLeftMutation(cations, 2, .5, rng=rng)
    a3, _desc = op.get_new_individual([a2])
    syms = a3.get_chemical_symbols()

    assert len(set(syms)) == 3
    set_raw_score(a3, 5.0)
    assert get_raw_score(a3) == 5.0
