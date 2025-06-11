from subprocess import DEVNULL, check_call

import numpy as np
import pytest

from ase import Atoms
from ase.build import molecule
from ase.cell import Cell
from ase.io import write
from ase.io.pov import (
    POVRAYIsosurface,
    get_bondpairs,
    set_high_bondorder_pairs,
    write_pov,
)


def test_povray_io(testdir, povray_executable):
    H2 = molecule('H2')
    write_pov('H2.pov', H2)
    check_call([povray_executable, 'H2.pov'], stderr=DEVNULL)


def test_povray_highorder(testdir, povray_executable):
    atoms = molecule('CH4')
    radii = [0.2] * len(atoms)
    bondpairs = get_bondpairs(atoms, radius=1.0)
    assert len(bondpairs) == 4

    high_bondorder_pairs = {}

    def setbond(target, order):
        high_bondorder_pairs[(0, target)] = ((0, 0, 0), order, (0.1, -0.2, 0))

    setbond(2, 2)
    setbond(3, 3)
    bondpairs = set_high_bondorder_pairs(bondpairs, high_bondorder_pairs)

    renderer = write_pov(
        'atoms.pov', atoms,
        povray_settings=dict(canvas_width=50, bondatoms=bondpairs),
        radii=radii,
    )

    # XXX Not sure how to test that the bondpairs data processing is correct.
    pngfile = renderer.render()
    assert pngfile.is_file()
    print(pngfile.absolute())


def test_deprecated(testdir):
    with pytest.warns(FutureWarning):
        write_pov('tmp.pov', molecule('H2'), run_povray=True)


@pytest.fixture()
def skimage():
    return pytest.importorskip('skimage')


@pytest.fixture()
def isosurface_things(skimage):
    rng = np.random.RandomState(42)
    cell = Cell(rng.random((3, 3)))

    values = np.zeros((3, 5, 7))
    values[1, 2, 3] = 1
    center_cell_position = cell.cartesian_positions([1 / 3, 2 / 5, 3 / 7])

    # This is the step which requires scikit-image:
    surface = POVRAYIsosurface(
        density_grid=values,
        cut_off=0.12345,
        cell=cell,
        cell_origin=(0, 0, 0))

    return cell, center_cell_position, surface


def test_compute_isosurface(isosurface_things):
    cell, center_cell_position, isosurf = isosurface_things

    vert_centroid = np.mean(isosurf.verts @ cell, axis=0)
    assert np.allclose(vert_centroid, center_cell_position)


def test_render_isosurface(testdir, isosurface_things, povray_executable):
    cell, _center_cell_position, isosurf = isosurface_things

    atoms = Atoms(
        'H3',
        scaled_positions=[[0, 0, 0], [1 / 3, 0, 0], [2 / 3, 0, 0]],
        cell=cell)

    renderer = write('tmp.pov', atoms, isosurface_data=[isosurf])
    png_path = renderer.render(povray_executable=povray_executable)
    # does the diamond appear over the second hydrogen atom?
    assert png_path.is_file()
