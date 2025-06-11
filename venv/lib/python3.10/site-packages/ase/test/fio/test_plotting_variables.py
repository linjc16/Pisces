

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ase import Atoms
from ase.io.utils import PlottingVariables


@pytest.fixture
def atoms_in_cell():
    atoms = Atoms('H3',
                positions=[
                    [0.0, 0, 0],
                    [0.3, 0, 0],
                    [0.8, 0, 0]],
                cell=[1, 1, 1],
                pbc=True)

    return atoms


@pytest.fixture
def random_rotation():
    myrng = np.random.default_rng(seed=453)
    random_rotation = Rotation.random(random_state=myrng)
    random_rotation_matrix = random_rotation.as_matrix()

    return random_rotation_matrix


def test_set_bbox(atoms_in_cell):

    rotation = '0x, 0y, 0z'

    generic_projection_settings = {
        'rotation': rotation,
        'bbox': (0, 0, 1, 1),
        'show_unit_cell': 2}

    pl = PlottingVariables(atoms=atoms_in_cell, **generic_projection_settings)

    camera_location = pl.get_image_plane_center()
    assert np.allclose(camera_location, [0.5, 0.5, 1.0])

    bbox2 = [0, 0, 0.5, 0.5]
    pl.update_image_plane_offset_and_size_from_structure(
        bbox=bbox2)

    camera_location = pl.get_image_plane_center()

    assert np.allclose(camera_location, [0.25, 0.25, 1.0])
    assert np.allclose(pl.get_bbox(), bbox2)


def test_camera_directions(atoms_in_cell):

    rotation = '0x, 45y, 0z'

    generic_projection_settings = {'rotation': rotation}

    pl = PlottingVariables(atoms=atoms_in_cell, **generic_projection_settings)

    camdir = pl.get_camera_direction()
    up = pl.get_camera_up()
    right = pl.get_camera_right()

    assert np.allclose(camdir.T @ up, 0)
    assert np.allclose(camdir.T @ right, 0)
    assert np.allclose(right.T @ up, 0)

    r22 = np.sqrt(2) / 2
    assert np.allclose(camdir, [r22, 0, -r22])
    assert np.allclose(up, [0, 1, 0])
    assert np.allclose(right, [r22, 0, r22])


def test_set_rotation_from_camera_directions(atoms_in_cell):
    '''Looks down the <111> direction'''
    generic_projection_settings = {
        'show_unit_cell': 2}

    pl = PlottingVariables(atoms=atoms_in_cell, **generic_projection_settings)

    pl.set_rotation_from_camera_directions(
        look=[-1, -1, -1], up=None, right=[-1, 1, 0],
        scaled_position=True)

    camdir = pl.get_camera_direction()
    up = pl.get_camera_up()
    right = pl.get_camera_right()

    invrt3 = 1 / np.sqrt(3)
    invrt2 = 1 / np.sqrt(2)
    assert np.allclose(right, [-invrt2, invrt2, 0])
    assert np.allclose(camdir, [-invrt3, -invrt3, -invrt3])
    assert np.allclose(up, [-1 / np.sqrt(6), -1 / np.sqrt(6), np.sqrt(2 / 3)])


def test_center_camera_on_position(atoms_in_cell):
    '''look at the upper left corner, camera should be above that point'''

    generic_projection_settings = {'show_unit_cell': 2}
    pl = PlottingVariables(atoms=atoms_in_cell, **generic_projection_settings)
    pl.center_camera_on_position([1, 1, 0])
    camera_location = pl.get_image_plane_center()

    assert np.allclose(camera_location, [1, 1, 1])


def test_camera_string_with_random_rotation(atoms_in_cell, random_rotation):
    '''Checks that a randome rotation matrix can be converted to a Euler
    rotation string and back'''

    random_rotation_matrix = random_rotation
    generic_projection_settings = {'show_unit_cell': 2,
                                   'rotation': random_rotation_matrix}
    pl = PlottingVariables(atoms=atoms_in_cell, **generic_projection_settings)

    rotation_string = pl.get_rotation_angles_string()
    pl.set_rotation(rotation_string)
    # higher atol since the default digits are 5
    assert np.allclose(random_rotation_matrix, pl.rotation, atol=1e-07)
