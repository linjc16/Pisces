import numpy as np

from ase import Atoms


def test_species_index():

    a = Atoms(['H', 'H', 'C', 'C', 'H'])

    spind = a.symbols.species_indices()

    assert (np.array(spind) == [0, 1, 0, 1, 2]).all()

    # It should work as the inverse to this
    allind = a.symbols.indices()

    for i, s in enumerate(a.symbols):
        assert (list(allind[s]).index(i) == spind[i])
