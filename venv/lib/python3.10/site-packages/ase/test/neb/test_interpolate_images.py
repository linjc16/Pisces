import numpy as np
import pytest

from ase import Atoms
from ase.constraints import FixAtoms
from ase.mep import interpolate


@pytest.fixture()
def initial():
    return Atoms('H', positions=[(1, 0.1, 0.1)], cell=[
        [1, 0, 0], [0, 1, 0], [0, 0, 1]], pbc=True)


@pytest.fixture()
def final():
    return Atoms('H', positions=[(2, 0.2, 0.1)], cell=[
        [2, 0, 0], [0, 2, 0], [0, 0, 2]], pbc=True)


@pytest.fixture()
def average_pos(initial, final):
    return np.average([initial.positions, final.positions], axis=0)


@pytest.fixture()
def images(initial, final):
    images = [initial.copy()]
    images += [initial.copy()]
    images += [final.copy()]
    return images


def assert_interpolated(values):
    step = (values[-1] - values[0]) / (len(values) - 1)
    for v1, v2 in zip(*[values[i:i + 1] for i in range(len(values) - 1)]):
        assert v2 - v1 == pytest.approx(step)


def test_interpolate_images_default(images, initial, average_pos):
    interpolate(images)
    assert images[1].positions == pytest.approx(average_pos)
    assert_interpolated([image.positions for image in images])
    assert np.allclose(images[1].cell, initial.cell)


def test_interpolate_images_fixed(images, initial, average_pos):

    for image in images:
        image.set_constraint(FixAtoms([0]))

    # test raising a RuntimeError here
    with pytest.raises(RuntimeError, match=r"Constraints in image "):
        interpolate(images)

    interpolate(images, apply_constraint=True)
    assert images[1].positions == pytest.approx(images[0].positions)
    assert np.allclose(images[1].cell, initial.cell)

    interpolate(images, apply_constraint=False)
    assert images[1].positions == pytest.approx(average_pos)
    assert_interpolated([image.positions for image in images])
    assert np.allclose(images[1].cell, initial.cell)


def test_interpolate_images_scaled_coord(images, initial):
    interpolate(images, use_scaled_coord=True)
    assert_interpolated([image.get_scaled_positions() for image in images])
    assert np.allclose(images[1].cell, initial.cell)


def test_interpolate_images_cell(images, initial, average_pos):
    interpolate(images, interpolate_cell=True)
    assert images[1].positions == pytest.approx(average_pos)
    assert_interpolated([image.positions for image in images])
    assert_interpolated([image.cell for image in images])


def test_interpolate_images_cell_default_interpolate_cell_scaled_coord(
        images,
        initial):
    interpolate(images, interpolate_cell=True, use_scaled_coord=True)
    assert_interpolated([image.get_scaled_positions() for image in images])
    assert_interpolated([image.cell for image in images])
