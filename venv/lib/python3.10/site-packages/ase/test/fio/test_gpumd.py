"""GPUMD input file parser.

Implemented:
* Input file (xyz.in)

"""
import numpy as np
import pytest

from ase import io
from ase.build import bulk
from ase.io.gpumd import load_xyz_input_gpumd


@pytest.fixture()
def gpumd_input_text():
    """This file is parsed correctly by GPUMD, since it is included
    among the examples distributed with the package, i.e.
    GPUMD/examples/ex5/xyz.in"""
    return """16 4 1.1 0 1 2
    1 1 1 6.18408 6.18408 6.18408
    0 24.6747 -9.37798 21.2733 28.085 0.0250834 -0.00953331 0.0216256 0 0
    1 29.7632 16.4709 -20.349 12.011 0.0291449 0.0156324 -0.0217974 1 1
    0 -11.3942 20.7898 -16.1723 28.085 -0.0138055 0.0189116 -0.0164401 2 0
    1 14.1056 -16.6803 15.5969 12.011 0.0110053 -0.0202905 0.0147439 3 1
    0 -6.83647 -7.06634 33.159 28.085 -0.00917232 -0.00718339 0.0314857 4 0
    1 44.7294 -12.1643 19.0095 12.011 0.0421363 -0.0134771 0.0159904 5 1
    0 -20.8261 1.14004 -38.0268 28.085 -0.0256162 -0.00106368 -0.0408793 6 0
    1 -4.53023 -50.3483 27.9666 12.011 -0.0101618 -0.0545161 0.0250959 7 1
    0 -24.0175 -12.38 -22.0283 28.085 -0.0244153 -0.0148076 -0.0246158 8 0
    1 -19.4088 19.7935 -10.8377 12.011 -0.0208416 0.0167874 -0.0143511 9 1
    0 7.39496 22.8278 17.9552 28.085 0.00529484 0.0187607 0.01603 10 0
    1 -1.24082 -7.64731 15.2911 12.011 -0.00459528 -0.0133305 0.0122104 11 1
    0 8.12018 25.4121 -10.6444 28.085 0.00603208 0.0236104 -0.0152659 12 0
    1 38.2451 -24.2186 31.0212 12.011 0.0355446 -0.0279537 0.0259785 13 1
    0 4.30332 19.9209 -9.21584 28.085 -7.06147e-05 0.0158056 -0.0138137 14 0
    1 8.91989 -1.32745 44.8546 12.011 0.00351112 -0.00690595 0.040041 15 1 """


@pytest.fixture()
def rocksalt():
    return bulk('NiO', 'rocksalt', 4.813, cubic=True)


def check_against_example_xyz(atoms):
    """Check atoms object against example structure
    (defined in gpumd_input_text)"""
    assert len(atoms) == 16
    assert set(atoms.symbols) == {'Si', 'C'}
    assert all(atoms.get_pbc())
    assert len(atoms.info) == len(atoms)
    assert len(atoms.get_velocities()) == len(atoms)

    # Check groups
    groupings = [[[i] for i in range(len(atoms))],
                 [[i for i, s in
                   enumerate(atoms.get_chemical_symbols()) if s == 'Si'],
                  [i for i, s in
                   enumerate(atoms.get_chemical_symbols()) if s == 'C']]]
    groups = [[[j for j, group in enumerate(grouping) if i in group][0]
               for grouping in groupings] for i in range(len(atoms))]
    assert all(np.array_equal(
        atoms.info[i]['groups'], np.array(groups[i])) for i in
        range(len(atoms)))


def test_read_gpumd(gpumd_input_text):
    """Test GPUMD default read."""
    with open('xyz.in', 'w') as fd:
        fd.write(gpumd_input_text)
    atoms = io.read('xyz.in', format='gpumd')
    check_against_example_xyz(atoms)


def test_read_gpumd_with_specified_species(gpumd_input_text):
    """Test GPUMD read when species are specified."""
    with open('xyz.in', 'w') as fd:
        fd.write(gpumd_input_text)

    species = ['Si', 'C']
    atoms = io.read('xyz.in', format='gpumd', species=species)
    check_against_example_xyz(atoms)


def test_read_gpumd_with_specified_masses(gpumd_input_text):
    """Test GPUMD read when isotope masses are specified."""
    with open('xyz.in', 'w') as fd:
        fd.write(gpumd_input_text)

    isotope_masses = {'Si': [28.085], 'C': [12.011]}
    atoms = io.read('xyz.in', format='gpumd',
                    isotope_masses=isotope_masses)
    check_against_example_xyz(atoms)


def test_load_gpumd_input(gpumd_input_text):
    """Load all information from a GPUMD input file."""
    with open('xyz.in', 'w') as fd:
        fd.write(gpumd_input_text)

    species_ref = ['Si', 'C']
    with open('xyz.in') as fd:
        atoms, input_parameters, species =\
            load_xyz_input_gpumd(fd, species=species_ref)
    input_parameters_ref = {'N': 16, 'M': 4, 'cutoff': 1.1,
                            'triclinic': 0, 'has_velocity': 1,
                            'num_of_groups': 2}
    assert input_parameters == input_parameters_ref
    assert species == species_ref
    check_against_example_xyz(atoms)


def test_gpumd_write_cubic(rocksalt):
    """Test write and load with triclinic cell."""
    rocksalt.write('xyz.in')

    # Ensure that we get the same structure back when reading
    readback = io.read('xyz.in')
    assert np.allclose(rocksalt.positions, readback.positions)
    assert np.allclose(rocksalt.cell, readback.cell)
    assert np.array_equal(rocksalt.numbers, readback.numbers)


def test_gpumd_write_triclinic(rocksalt):
    """Test write and load with triclinic cell."""
    rocksalt.write('xyz.in', use_triclinic=True)
    with open('xyz.in') as fd:
        readback, input_parameters, _ = load_xyz_input_gpumd(fd)
    assert input_parameters['triclinic'] == 1
    assert np.allclose(rocksalt.positions, readback.positions)
    assert np.allclose(rocksalt.cell, readback.cell)
    assert np.array_equal(rocksalt.numbers, readback.numbers)


def test_gpumd_write_with_groupings(rocksalt):
    """Test write and load with groupings."""
    groupings = [[[i for i, s in
                   enumerate(rocksalt.get_chemical_symbols()) if s == 'Ni'],
                  [i for i, s in
                   enumerate(rocksalt.get_chemical_symbols()) if s == 'O']],
                 [[i] for i in range(len(rocksalt))]]
    groups = [[[j for j, group in enumerate(grouping) if i in group][0]
               for grouping in groupings] for i in range(len(rocksalt))]
    rocksalt.write('xyz.in', groupings=groupings)
    with open('xyz.in') as fd:
        readback, input_parameters, _ = load_xyz_input_gpumd(fd)
    assert input_parameters['num_of_groups'] == 2
    assert len(readback.info) == len(rocksalt)
    assert all(np.array_equal(
        readback.info[i]['groups'], np.array(groups[i])) for i in
        range(len(rocksalt)))


def test_gpumd_write_with_velocities(rocksalt):
    """Test write and load with velocities."""
    velocities = np.array([[-0.3, 2.3, 0.7], [0.0, 0.3, 0.8],
                           [-0.6, 0.9, 0.1], [-1.7, -0.1, -0.5],
                           [-0.5, 0.0, 0.6], [-0.2, 0.1, 0.5],
                           [-0.1, 1.4, -1.9], [-1.0, -0.5, -1.2]])
    rocksalt.set_velocities(velocities)
    rocksalt.write('xyz.in')
    with open('xyz.in') as fd:
        readback, input_parameters, _ = load_xyz_input_gpumd(fd)
    assert input_parameters['has_velocity'] == 1
    assert np.allclose(readback.get_velocities(), rocksalt.get_velocities())


def test_gpumd_write_with_custom_species(rocksalt):
    """Test write and read with custom species definitions."""
    rocksalt.write('xyz.in', species=['O', 'Ni'])
    readback = io.read('xyz.in', species=['O', 'Ni'])
    assert np.allclose(rocksalt.positions, readback.positions)
    assert np.allclose(rocksalt.cell, readback.cell)
    assert np.array_equal(rocksalt.numbers, readback.numbers)


def test_gpumd_write_raises_exception_on_species_mismatch(rocksalt):
    """Test that writing raises an exception when there is a
    mismatch between species and atoms object."""
    with pytest.raises(ValueError):
        rocksalt.write('xyz.in', species=['O', 'H'])
