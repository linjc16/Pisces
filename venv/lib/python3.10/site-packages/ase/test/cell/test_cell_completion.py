import numpy as np

from ase.geometry.cell import complete_cell


def test_cell_completion():

    eps = 1E-10
    rng = np.random.RandomState(0)

    def random_unit_vector():
        while True:
            v = rng.uniform(-1, 1, 3)
            norm = np.linalg.norm(v)
            if norm > eps:
                return v / norm

    for it in range(100):

        cell = np.zeros((3, 3))
        index = rng.randint(0, 3)
        cell[index] = random_unit_vector()
        complete = complete_cell(cell)

        assert abs(np.linalg.norm(complete[index]) - 1) < eps
        assert np.linalg.det(complete) > 0
        assert np.linalg.matrix_rank(complete) == 3
