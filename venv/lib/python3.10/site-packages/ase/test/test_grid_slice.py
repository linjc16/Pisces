import numpy as np

from ase.utils.cube import grid_2d_slice


def test_slice():
    x = np.arange(0, 1, 0.1)
    y = np.arange(0, 1, 0.1)
    z = np.arange(0, 1, 0.1)

    X, Y, Z = np.meshgrid(x, y, z)

    spacings = np.eye(3) * 0.1

    grid = np.cos(X) + np.sin(Y) + np.cos(Z)

    #  We test the basic xy plane
    _, _, D = grid_2d_slice(
        spacings, grid, (1, 0, 0), (0, 1, 0), (0, 0, 0), step=0.1, size_u=(
            0, 1), size_v=(
            0, 1))

    assert np.allclose(grid[:, :, 0], D) or \
        np.allclose(grid[:, :, 0], D.T)

    # We test the offset
    _, _, D = grid_2d_slice(
        spacings, grid, (1, 0, 0), (0, 1, 0), (0, 0, 0.5), step=0.1, size_u=(
            0, 1), size_v=(
            0, 1))

    assert np.allclose(grid[:, :, 5], D) or \
        np.allclose(grid[:, :, 5], D.T)

    grid = np.sin(X) - np.sin(Y)

    # We test a more "complex" plane, the x - y = 0 plane should be zeros
    # everywhere...
    _, _, D = grid_2d_slice(
        spacings, grid, (1, 1, 0), (0, 0, 1), (0, 0, 0.0), step=0.1, size_u=(
            0, 1), size_v=(
            0, 1))

    assert np.allclose(D, np.zeros_like(D))
