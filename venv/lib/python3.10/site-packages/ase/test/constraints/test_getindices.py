from ase.build import fcc111
from ase.constraints import (
    FixAtoms,
    FixBondLengths,
    FixInternals,
    FixLinearTriatomic,
    Hookean,
    constrained_indices,
)


def test_getindices():
    slab = fcc111('Pt', (4, 4, 4))

    C1 = FixAtoms([0, 2, 4])
    C2 = FixBondLengths([[0, 1], [0, 2]])
    C3 = FixInternals(bonds=[[1, [7, 8]], [1, [8, 9]]])
    C4 = Hookean(a1=30, a2=40, rt=1.79, k=5.)
    C5 = FixLinearTriatomic(triples=[(0, 1, 2), (3, 4, 5)])

    slab.set_constraint([C1, C2, C3, C4, C5])
    assert all(constrained_indices(slab, (FixAtoms, FixBondLengths)) ==
               [0, 1, 2, 4])
    assert all(constrained_indices(
        slab, (FixBondLengths, FixLinearTriatomic)) == [0, 1, 2, 3, 4, 5])
    assert all(constrained_indices(slab) == [0, 1, 2, 3, 4, 5, 7, 8, 9, 30, 40])
