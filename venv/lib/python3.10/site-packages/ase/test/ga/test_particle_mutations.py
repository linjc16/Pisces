import numpy as np

from ase.build import fcc111
from ase.ga.particle_mutations import Poor2richPermutation, Rich2poorPermutation


def test_rich2poor_permutations():
    rng = np.random.RandomState(seed=1234)

    slab = fcc111("Pd", size=(10, 10, 1), vacuum=10.0)
    for i in range(0, len(slab), 2):
        slab.symbols[i] = "Ag"
    slab.info["confid"] = ""
    elements = ["Pd", "Ag"]

    parents = [slab]
    creator = Rich2poorPermutation(elements=elements, rng=rng, num_muts=1)
    offspring, _ = creator.get_new_individual(parents)
    assert offspring.symbols[17] == "Ag"
    assert offspring.symbols[92] == "Pd"

    creator = Poor2richPermutation(elements=elements, rng=rng, num_muts=1)
    offspring, _ = creator.get_new_individual(parents)
    assert offspring.symbols[7] == "Ag"
    assert offspring.symbols[38] == "Pd"
