import warnings

import numpy as np
import pytest

from ase import Atom, Atoms
from ase.io import NetCDFTrajectory, read


@pytest.fixture(scope='module')
def netCDF4():
    return pytest.importorskip('netCDF4')


@pytest.fixture(autouse=True)
def catch_netcdf4_warning():
    with warnings.catch_warnings():
        # XXX Ignore deprecation warning from numpy over how netCDF4
        # uses numpy.  We can't really do anything about that.
        warnings.simplefilter('ignore', DeprecationWarning)
        yield


@pytest.fixture()
def co(netCDF4):
    return Atoms([Atom('C', (0, 0, 0)),
                  Atom('O', (0, 0, 1.2))],
                 cell=[3, 3, 3],
                 pbc=True)


def test_netcdftrajectory(co):
    rng = np.random.RandomState(17)
    traj = NetCDFTrajectory('1.nc', 'w', co)
    for _ in range(5):
        co.positions[:, 2] += 0.1
        traj.write()
    del traj
    traj = NetCDFTrajectory('1.nc', 'a')
    co = traj[-1]
    print(co.positions)
    co.positions[:] += 1
    traj.write(co)
    del traj
    t = NetCDFTrajectory('1.nc', 'a')

    print(t[-1].positions)
    print('.--------')
    for i, a in enumerate(t):
        if i < 4:
            print(1, a.positions[-1, 2], 1.3 + i * 0.1)
            assert abs(a.positions[-1, 2] - 1.3 - i * 0.1) < 1e-6
        else:
            print(1, a.positions[-1, 2], 1.7 + i - 4)
            assert abs(a.positions[-1, 2] - 1.7 - i + 4) < 1e-6
        assert a.pbc.all()
    co.positions[:] += 1
    t.write(co)
    for i, a in enumerate(t):
        if i < 4:
            print(2, a.positions[-1, 2], 1.3 + i * 0.1)
            assert abs(a.positions[-1, 2] - 1.3 - i * 0.1) < 1e-6
        else:
            print(2, a.positions[-1, 2], 1.7 + i - 4)
            assert abs(a.positions[-1, 2] - 1.7 - i + 4) < 1e-6
    assert len(t) == 7

    # Change atom type and append
    co[0].number = 1
    t.write(co)
    t2 = NetCDFTrajectory('1.nc', 'r')
    co2 = t2[-1]
    assert (co2.numbers == co.numbers).all()
    del t2

    co[0].number = 6
    t.write(co)

    co.pbc = False
    o = co.pop(1)
    try:
        t.write(co)
    except ValueError:
        pass
    else:
        assert False

    co.append(o)
    co.pbc = True
    t.write(co)
    del t

    # append to a nonexisting file
    fname = '2.nc'
    t = NetCDFTrajectory(fname, 'a', co)
    del t

    fname = '3.nc'
    t = NetCDFTrajectory(fname, 'w', co)
    # File is not created before first write
    co.set_pbc([True, False, False])
    d = co.get_distance(0, 1)

    with pytest.warns(UserWarning, match='Atoms have nonperiodic'):
        t.write(co)

    del t
    # Check pbc
    for c in [1, 1000]:
        t = NetCDFTrajectory(fname, chunk_size=c)
        a = t[-1]
        assert a.pbc[0] and not a.pbc[1] and not a.pbc[2]
        assert abs(a.get_distance(0, 1) - d) < 1e-6
        del t
    # Append something in Voigt notation
    t = NetCDFTrajectory(fname, 'a')
    for frame, a in enumerate(t):
        test = rng.random((len(a), 6))
        a.set_array('test', test)
        t.write_arrays(a, frame, ['test'])
    del t

    # Check cell origin
    co.set_pbc(True)
    co.set_celldisp([1, 2, 3])
    traj = NetCDFTrajectory('4.nc', 'w', co)
    traj.write(co)
    traj.close()

    traj = NetCDFTrajectory('4.nc', 'r')
    a = traj[0]
    assert np.all(abs(a.get_celldisp() - np.array([1, 2, 3])) < 1e-12)
    traj.close()

    # Add 'id' field and check if it is read correctly
    co.set_array('id', np.array([2, 1]))
    traj = NetCDFTrajectory('5.nc', 'w', co)
    traj.write(co, arrays=['id'])
    traj.close()

    traj = NetCDFTrajectory('5.nc', 'r')
    assert np.all(traj[0].numbers == [8, 6])
    assert np.all(np.abs(traj[0].positions - np.array([[2, 2, 3.7],
                                                       [2., 2., 2.5]])) < 1e-6)
    traj.close()

    a = read('5.nc')
    assert len(a) == 2


def test_netcdf_with_variable_atomic_numbers(netCDF4):
    # Create a NetCDF file with a per-file definition of atomic numbers. ASE
    # NetCDFTrajectory can read but not write these types of files.
    nc = netCDF4.Dataset('6.nc', 'w')
    nc.createDimension('frame', None)
    nc.createDimension('atom', 2)
    nc.createDimension('spatial', 3)
    nc.createDimension('cell_spatial', 3)
    nc.createDimension('cell_angular', 3)

    nc.createVariable('atom_types', 'i', ('atom',))
    nc.createVariable('coordinates', 'f4', ('frame', 'atom', 'spatial',))
    nc.createVariable('cell_lengths', 'f4', ('frame', 'cell_spatial',))
    nc.createVariable('cell_angles', 'f4', ('frame', 'cell_angular',))

    r0 = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    r1 = 2 * r0

    nc.variables['atom_types'][:] = [1, 2]
    nc.variables['coordinates'][0] = r0
    nc.variables['coordinates'][1] = r1
    nc.variables['cell_lengths'][:] = 0
    nc.variables['cell_angles'][:] = 90

    nc.close()

    traj = NetCDFTrajectory('6.nc', 'r')
    assert np.allclose(traj[0].positions, r0)
    assert np.allclose(traj[1].positions, r1)
    traj.close()


def test_netcdf_with_nonconsecutive_index(netCDF4):
    nc = netCDF4.Dataset('7.nc', 'w')
    nc.createDimension('frame', None)
    nc.createDimension('atom', 3)
    nc.createDimension('spatial', 3)
    nc.createDimension('cell_spatial', 3)
    nc.createDimension('cell_angular', 3)

    nc.createVariable('atom_types', 'i', ('atom',))
    nc.createVariable('coordinates', 'f4', ('frame', 'atom', 'spatial',))
    nc.createVariable('cell_lengths', 'f4', ('frame', 'cell_spatial',))
    nc.createVariable('cell_angles', 'f4', ('frame', 'cell_angular',))
    nc.createVariable('id', 'i', ('frame', 'atom',))

    r0 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    r1 = 2 * r0

    nc.variables['atom_types'][:] = [1, 2, 3]
    nc.variables['coordinates'][0] = r0
    nc.variables['coordinates'][1] = r1
    nc.variables['cell_lengths'][:] = 0
    nc.variables['cell_angles'][:] = 90
    nc.variables['id'][0] = [13, 3, 5]
    nc.variables['id'][1] = [-1, 0, -5]

    nc.close()

    traj = NetCDFTrajectory('7.nc', 'r')
    assert (traj[0].numbers == [2, 3, 1]).all()
    assert (traj[1].numbers == [3, 1, 2]).all()
    traj.close()


def test_types_to_numbers_argument(co):
    traj = NetCDFTrajectory('8.nc', 'w', co)
    traj.write()
    traj.close()
    d = {6: 15, 8: 15}
    traj = NetCDFTrajectory('8.nc', mode="r", types_to_numbers=d)
    assert np.allclose(traj[-1].get_masses(), 30.974)
    assert (traj[-1].numbers == [15, 15]).all()
    d = {3: 14}
    traj = NetCDFTrajectory('8.nc', mode="r", types_to_numbers=d)
    assert (traj[-1].numbers == [6, 8]).all()
    traj = NetCDFTrajectory('8.nc', 'r',
                            types_to_numbers=[0, 0, 0, 0, 0, 0, 15])
    assert (traj[-1].numbers == [15, 8]).all()

    traj.close()
