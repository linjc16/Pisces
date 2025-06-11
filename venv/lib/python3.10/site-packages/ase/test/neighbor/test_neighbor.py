"""Tests for NeighborList"""
import numpy as np
import pytest

from ase import Atoms
from ase.build import bulk
from ase.neighborlist import (
    NeighborList,
    NewPrimitiveNeighborList,
    PrimitiveNeighborList,
)


@pytest.mark.parametrize(
    'primitive',
    [PrimitiveNeighborList, NewPrimitiveNeighborList],
)
@pytest.mark.parametrize('bothways', [False, True])
@pytest.mark.parametrize('self_interaction', [False, True])
@pytest.mark.parametrize('sorted', [False, True])
def test_unique(
    sorted: bool,
    self_interaction: bool,
    bothways: bool,
    primitive,
):
    """Test if there are no duplicates in the neighbor lists"""
    atoms = Atoms('H2', positions=[(0, 0, 0), (0, 0, 1)])
    nl = NeighborList(
        [0.5, 0.5],
        skin=0.1,
        sorted=sorted,
        self_interaction=self_interaction,
        bothways=bothways,
        primitive=primitive,
    )
    nl.update(atoms)
    tmp = []
    for i in range(len(atoms)):
        neighbors, offsets = nl.get_neighbors(i)
        tmp += [(i, n, *o) for n, o in zip(neighbors, offsets)]
    assert len(set(tmp)) == len(tmp)


def count(nl: NeighborList, atoms: Atoms):
    """Count the numbers of neighboring atoms for all the atoms

    Returns
    -------
    d : float
        Sum of distances over all nearest-neighbor pairs
    c : npt.NDArray[np.int_]
        Numbers of neighboring atoms for all the atoms
    """
    c = np.zeros(len(atoms), int)
    R = atoms.get_positions()
    cell = atoms.get_cell()
    d = 0.0
    for a in range(len(atoms)):
        i, offsets = nl.get_neighbors(a)
        for j in i:
            c[j] += 1
        c[a] += len(i)
        d += (((R[i] + np.dot(offsets, cell) - R[a])**2).sum(1)**0.5).sum()
    return d, c


# scipy sparse uses matrix subclass internally
@pytest.mark.filterwarnings('ignore:the matrix subclass')
@pytest.mark.slow()
@pytest.mark.parametrize('sorted', [False, True])
def test_supercell(sorted):
    """Test if NeighborList works for a supercell as expected"""
    atoms = Atoms(numbers=range(10),
                  cell=[(0.2, 1.2, 1.4),
                        (1.4, 0.1, 1.6),
                        (1.3, 2.0, -0.1)])
    rng = np.random.RandomState(42)
    atoms.set_scaled_positions(3 * rng.random((10, 3)) - 1)

    for p1 in range(2):
        for p2 in range(2):
            for p3 in range(2):
                # print(p1, p2, p3)
                atoms.set_pbc((p1, p2, p3))
                cutoffs = atoms.numbers * 0.2 + 0.5
                nl = NeighborList(cutoffs, skin=0.0, sorted=sorted)
                nl.update(atoms)
                d, c = count(nl, atoms)

                atoms2 = atoms.repeat((p1 + 1, p2 + 1, p3 + 1))
                cutoffs2 = atoms2.numbers * 0.2 + 0.5
                nl2 = NeighborList(cutoffs2, skin=0.0, sorted=sorted)
                nl2.update(atoms2)
                d2, c2 = count(nl2, atoms2)

                c2.shape = (-1, 10)  # row: images, column: atoms

                # if the sum of nearest-neighbor distances gets larger
                # according to the supercell size
                dd = d * (p1 + 1) * (p2 + 1) * (p3 + 1) - d2
                assert abs(dd) < 1e-10

                # if each repeated image has the same numbers of neighbors
                assert not (c2 - c).any()


def test_H2():
    h2 = Atoms('H2', positions=[(0, 0, 0), (0, 0, 1)])
    nl = NeighborList([0.5, 0.5], skin=0.1, sorted=True, self_interaction=False)
    nl2 = NeighborList([0.5, 0.5], skin=0.1, sorted=True,
                       self_interaction=False,
                       primitive=NewPrimitiveNeighborList)
    assert nl2.update(h2)
    assert nl.update(h2)
    assert not nl.update(h2)
    assert (nl.get_neighbors(0)[0] == [1]).all()
    m = np.zeros((2, 2))
    m[0, 1] = 1
    assert np.array_equal(nl.get_connectivity_matrix(sparse=False), m)
    assert np.array_equal(nl.get_connectivity_matrix(sparse=True).todense(), m)
    assert np.array_equal(nl.get_connectivity_matrix().todense(),
                          nl2.get_connectivity_matrix().todense())

    h2[1].z += 0.09
    assert not nl.update(h2)
    assert (nl.get_neighbors(0)[0] == [1]).all()

    h2[1].z += 0.09
    assert nl.update(h2)
    assert (nl.get_neighbors(0)[0] == []).all()
    assert nl.nupdates == 2


def test_H2_shape_and_type():
    h2 = Atoms('H2', positions=[(0, 0, 0), (0, 0, 1)])
    nl = NeighborList([0.1, 0.1], skin=0.1, bothways=True,
                      self_interaction=False)
    assert nl.update(h2)
    assert nl.get_neighbors(0)[1].shape == (0, 3)
    assert nl.get_neighbors(0)[1].dtype == int


def test_fcc():
    x = bulk('X', 'fcc', a=2**0.5)

    nl = NeighborList([0.5], skin=0.01, bothways=True, self_interaction=False)
    nl.update(x)
    assert len(nl.get_neighbors(0)[0]) == 12

    nl = NeighborList([0.5] * 27, skin=0.01, bothways=True,
                      self_interaction=False)
    nl.update(x * (3, 3, 3))
    for a in range(27):
        assert len(nl.get_neighbors(a)[0]) == 12
    assert not np.any(nl.get_neighbors(13)[1])


def test_use_scaled_positions():
    c = 0.0058
    for NeighborListClass in [PrimitiveNeighborList, NewPrimitiveNeighborList]:
        nl = NeighborListClass([c, c],
                               skin=0.0,
                               sorted=True,
                               self_interaction=False,
                               use_scaled_positions=True)
        nl.update([True, True, True],
                  np.eye(3) * 7.56,
                  np.array([[0, 0, 0],
                            [0, 0, 0.99875]]))
        n0, d0 = nl.get_neighbors(0)
        n1, d1 = nl.get_neighbors(1)
        # != is xor
        assert (np.all(n0 == [0]) and np.all(d0 == [0, 0, 1])) != \
            (np.all(n1 == [1]) and np.all(d1 == [0, 0, -1]))


def test_empty_neighbor_list():
    # Test empty neighbor list
    nl = PrimitiveNeighborList([])
    nl.update([True, True, True],
              np.eye(3) * 7.56,
              np.zeros((0, 3)))


@pytest.mark.parametrize('bothways', [False, True])
@pytest.mark.parametrize('self_interaction', [False, True])
@pytest.mark.parametrize('sort', [False, True])
def test_equivalence_of_primitive_classes(sort, self_interaction, bothways):
    """Test if two primitive neighbor-list classes make the same naighbors"""
    # diamond structure in the primitive cell
    pbc_c = np.array([True, True, True])
    cell_cv = np.array([[0., 3.37316113, 3.37316113],
                        [3.37316113, 0., 3.37316113],
                        [3.37316113, 3.37316113, 0.]])
    spos_ac = np.array([[0., 0., 0.],
                        [0.25, 0.25, 0.25]])
    natoms = len(spos_ac)

    cutoff_a = np.array([8.0, 8.0])

    info = [[] for _ in range(2)]  # neighbor info collector for each primitive
    primitives = [PrimitiveNeighborList, NewPrimitiveNeighborList]
    for ip, primitive in enumerate(primitives):
        nl = primitive(
            cutoff_a,
            skin=0.0,
            sorted=sort,
            self_interaction=self_interaction,
            bothways=bothways,
            use_scaled_positions=True,
        )
        nl.update(pbc_c, cell_cv, spos_ac)

        # collect neighbor info into a list of tuples
        # each tuple has the form (i1, i2, o1, o2, o3)
        # i1: 1st atom
        # i2: 2nd atom
        # o1: offset along 1st cell vector
        # o2: offset along 2nd cell vector
        # o3: offset along 3rd cell vector
        for i in range(natoms):
            info[ip].extend([(i, n, *o) for n, o in zip(*nl.get_neighbors(i))])

    def reverse(t: tuple):
        return t[1], t[0], -t[2], -t[3], -t[4]

    for ip in range(2):
        # (i1, i2, +o1, +o2, +o3) and (i2, i1, -o1, -o2, -o3) is the same pair
        # the following guarantees i0 <= i1
        info[ip] = [t if t[0] <= t[1] else reverse(t) for t in info[ip]]
        info[ip] = sorted(info[ip])  # sort by i1, i2, o1, o2, o3

    # check if the both primitive classes provide the same neighbors
    assert np.all(info[0] == info[1])


def test_small_cell_and_large_cutoff():
    # See: https://gitlab.com/ase/ase/-/issues/441
    cutoff = 50

    atoms = bulk('Cu', 'fcc', a=3.6)
    atoms *= (2, 2, 2)
    atoms.set_pbc(False)
    radii = cutoff * np.ones(len(atoms.get_atomic_numbers()))

    neighborhood_new = NeighborList(
        radii, skin=0.0, self_interaction=False, bothways=True,
        primitive=NewPrimitiveNeighborList
    )
    neighborhood_old = NeighborList(
        radii, skin=0.0, self_interaction=False, bothways=True,
        primitive=PrimitiveNeighborList
    )

    neighborhood_new.update(atoms)
    neighborhood_old.update(atoms)

    n0, d0 = neighborhood_new.get_neighbors(0)
    n1, d1 = neighborhood_old.get_neighbors(0)

    assert np.all(n0 == n1)
    assert np.all(d0 == d1)
