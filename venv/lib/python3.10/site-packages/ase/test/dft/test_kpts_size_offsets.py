import numpy as np

from ase import Atoms
from ase.calculators.calculator import kpts2sizeandoffsets as k2so

a = 6.0


def test_gamma_pt():
    size, offsets = map(tuple, k2so())
    assert (size, offsets) == ((1, 1, 1), (0, 0, 0))


def test_222():
    size, offsets = map(tuple, k2so(even=True))
    assert (size, offsets) == ((2, 2, 2), (0, 0, 0))


def test_shape_from_density():
    kd = 25 / (2 * np.pi)
    size, offsets = map(tuple, k2so(density=kd,
                                    atoms=Atoms(cell=(a, a, a), pbc=True)))
    assert (size, offsets) == ((5, 5, 5), (0, 0, 0))


def test_shape_from_size():
    size, offsets = map(tuple, k2so(size=(3, 4, 5),))
    assert (size, offsets) == ((3, 4, 5), (0, 0, 0))


def test_gamma_centering_from_density():
    kd = 24 / (2 * np.pi)
    size, offsets = map(tuple, k2so(density=kd,
                                    gamma=True,
                                    atoms=Atoms(cell=(a, a, a), pbc=True)))
    assert (size, offsets) == ((4, 4, 4), (0.125, 0.125, 0.125))


def test_gamma_centering_from_size():
    size, offsets = map(tuple, k2so(size=(3, 4, 5),
                                    gamma=True))
    assert (size, offsets) == ((3, 4, 5), (0., 0.125, 0.))


def test_antigamma_centering_from_default_111():
    size, offsets = map(tuple, k2so(gamma=False,
                                    atoms=Atoms(cell=(a, a, a), pbc=True)))
    assert (size, offsets) == ((1, 1, 1), (0.5, 0.5, 0.5))


def test_density_with_irregular_shape():
    cell = [[2, 1, 0], [1, 2, 2], [-1, 0, 2]]
    kd = 3
    size, offsets = map(tuple, k2so(density=kd,
                                    atoms=Atoms(cell=cell, pbc=True)))
    assert (size, offsets) == ((29, 22, 26), (0, 0, 0))

    # Set even numbers with density
    size, offsets = map(tuple, k2so(density=kd,
                                    even=True,
                                    atoms=Atoms(cell=cell, pbc=True)))
    assert (size, offsets) == ((30, 22, 26), (0, 0, 0))

    # Set even numbers and Gamma centre with density
    size, offsets = map(tuple, k2so(density=kd,
                                    even=True,
                                    gamma=True,
                                    atoms=Atoms(cell=cell, pbc=True)))
    assert (size, offsets) == ((30, 22, 26), (1 / 60, 1 / 44, 1 / 52))

    # Set odd with density
    size, offsets = map(tuple, k2so(density=kd,
                                    even=False,
                                    atoms=Atoms(cell=cell, pbc=True)))
    assert (size, offsets) == ((29, 23, 27), (0, 0, 0))

    # Set even with size
    size, offsets = map(tuple, k2so(size=(3, 4, 5), even=True))
    assert (size, offsets) == ((4, 4, 6), (0, 0, 0))

    # Set odd with size
    size, offsets = map(tuple, k2so(size=(3, 4, 5), even=False))
    assert (size, offsets) == ((3, 5, 5), (0, 0, 0))

    # Interaction with PBC: don't shift non-periodic directions away from gamma
    size, offsets = map(tuple, k2so(size=(5, 5, 1),
                                    gamma=False,
                                    atoms=Atoms(cell=(a, a, a),
                                                pbc=[True, True, False])))
    assert (size, offsets) == ((5, 5, 1), (0.1, 0.1, 0.))
