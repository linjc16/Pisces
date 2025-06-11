# fmt: off

import numpy as np

import ase.units as units


def write_amber_coordinates(atoms, filename):
    from scipy.io import netcdf_file
    with netcdf_file(filename, 'w', mmap=False) as fout:
        _write_amber_coordinates(atoms, fout)


def read_amber_coordinates(filename):
    from scipy.io import netcdf_file
    with netcdf_file(filename, 'r', mmap=False) as fin:
        return _read_amber_coordinates(fin)


def _write_amber_coordinates(atoms, fout):
    fout.Conventions = 'AMBERRESTART'
    fout.ConventionVersion = "1.0"
    fout.title = 'Ase-generated-amber-restart-file'
    fout.application = "AMBER"
    fout.program = "ASE"
    fout.programVersion = "1.0"
    fout.createDimension('cell_spatial', 3)
    fout.createDimension('label', 5)
    fout.createDimension('cell_angular', 3)
    fout.createDimension('time', 1)
    time = fout.createVariable('time', 'd', ('time',))
    time.units = 'picosecond'
    time[0] = 0
    fout.createDimension('spatial', 3)
    spatial = fout.createVariable('spatial', 'c', ('spatial',))
    spatial[:] = np.asarray(list('xyz'))

    natom = len(atoms)
    fout.createDimension('atom', natom)
    coordinates = fout.createVariable('coordinates', 'd',
                                      ('atom', 'spatial'))
    coordinates.units = 'angstrom'
    coordinates[:] = atoms.get_positions()

    velocities = fout.createVariable('velocities', 'd',
                                     ('atom', 'spatial'))
    velocities.units = 'angstrom/picosecond'
    # Amber's units of time are 1/20.455 ps
    # Any other units are ignored in restart files, so these
    # are the only ones it is safe to print
    # See: http://ambermd.org/Questions/units.html
    # Apply conversion factor from ps:
    velocities.scale_factor = 20.455
    # get_velocities call returns velocities with units sqrt(eV/u)
    # so convert to Ang/ps
    factor = units.fs * 1000 / velocities.scale_factor
    velocities[:] = atoms.get_velocities() * factor

    fout.createVariable('cell_angular', 'c', ('cell_angular', 'label'))

    cell_spatial = fout.createVariable('cell_spatial', 'c', ('cell_spatial',))
    cell_spatial[:] = ['a', 'b', 'c']

    cell_lengths = fout.createVariable('cell_lengths', 'd', ('cell_spatial',))
    cell_lengths.units = 'angstrom'
    cell_lengths[:] = atoms.cell.lengths()

    if not atoms.cell.orthorhombic:
        raise ValueError('Non-orthorhombic cell not supported with amber')

    cell_angles = fout.createVariable('cell_angles', 'd', ('cell_angular',))
    cell_angles[:3] = 90.0
    cell_angles.units = 'degree'


def _read_amber_coordinates(fin):
    from ase import Atoms

    all_coordinates = fin.variables['coordinates'][:]
    get_last_frame = False
    if hasattr(all_coordinates, 'ndim'):
        if all_coordinates.ndim == 3:
            get_last_frame = True
    elif hasattr(all_coordinates, 'shape'):
        if len(all_coordinates.shape) == 3:
            get_last_frame = True
    if get_last_frame:
        all_coordinates = all_coordinates[-1]

    atoms = Atoms(positions=all_coordinates)

    if 'velocities' in fin.variables:
        all_velocities = fin.variables['velocities']
        if hasattr(all_velocities, 'units'):
            if all_velocities.units != b'angstrom/picosecond':
                raise Exception(
                    f'Unrecognised units {all_velocities.units}')
        if hasattr(all_velocities, 'scale_factor'):
            scale_factor = all_velocities.scale_factor
        else:
            scale_factor = 1.0
        all_velocities = all_velocities[:] * scale_factor
        all_velocities = all_velocities / (1000 * units.fs)
        if get_last_frame:
            all_velocities = all_velocities[-1]
        atoms.set_velocities(all_velocities)
    if 'cell_lengths' in fin.variables:
        all_abc = fin.variables['cell_lengths']
        if get_last_frame:
            all_abc = all_abc[-1]
        a, b, c = all_abc
        all_angles = fin.variables['cell_angles']
        if get_last_frame:
            all_angles = all_angles[-1]
        alpha, beta, gamma = all_angles

        if (all(angle > 89.99 for angle in [alpha, beta, gamma]) and
                all(angle < 90.01 for angle in [alpha, beta, gamma])):
            atoms.set_cell(
                np.array([[a, 0, 0],
                          [0, b, 0],
                          [0, 0, c]]))
            atoms.set_pbc(True)
        else:
            raise NotImplementedError('only rectangular cells are'
                                      ' implemented in ASE-AMBER')

    else:
        atoms.set_pbc(False)

    return atoms
