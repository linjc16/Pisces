import re
import warnings
from collections.abc import Iterable, Iterator
from io import StringIO
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError
from ase.calculators.singlepoint import SinglePointDFTCalculator
from ase.io import ParseError, read
from ase.units import Bohr, Hartree
from ase.utils import deprecated, reader, writer


@reader
def read_geom_orcainp(fd):
    """Method to read geometry from an ORCA input file."""
    lines = fd.readlines()

    # Find geometry region of input file.
    stopline = 0
    for index, line in enumerate(lines):
        if line[1:].startswith('xyz '):
            startline = index + 1
            stopline = -1
        elif line.startswith('end') and stopline == -1:
            stopline = index
        elif line.startswith('*') and stopline == -1:
            stopline = index
    # Format and send to read_xyz.
    xyz_text = '%i\n' % (stopline - startline)
    xyz_text += ' geometry\n'
    for line in lines[startline:stopline]:
        xyz_text += line
    atoms = read(StringIO(xyz_text), format='xyz')
    atoms.set_cell((0.0, 0.0, 0.0))  # no unit cell defined

    return atoms


@writer
def write_orca(fd, atoms, params):
    # conventional filename: '<name>.inp'
    fd.write(f'! {params["orcasimpleinput"]} \n')
    fd.write(f'{params["orcablocks"]} \n')

    if 'coords' not in params['orcablocks']:
        fd.write('*xyz')
        fd.write(' %d' % params['charge'])
        fd.write(' %d \n' % params['mult'])
        for atom in atoms:
            if atom.tag == 71:  # 71 is ascii G (Ghost)
                symbol = atom.symbol + ' : '
            else:
                symbol = atom.symbol + '   '
            fd.write(
                symbol
                + str(atom.position[0])
                + ' '
                + str(atom.position[1])
                + ' '
                + str(atom.position[2])
                + '\n'
            )
        fd.write('*\n')


def read_charge(lines: List[str]) -> Optional[float]:
    """Read sum of atomic charges."""
    charge = None
    for line in lines:
        if 'Sum of atomic charges' in line:
            charge = float(line.split()[-1])
    return charge


def read_energy(lines: List[str]) -> Optional[float]:
    """Read energy."""
    energy = None
    for line in lines:
        if 'FINAL SINGLE POINT ENERGY' in line:
            if 'Wavefunction not fully converged' in line:
                energy = float('nan')
            else:
                energy = float(line.split()[-1])
    if energy is not None:
        return energy * Hartree
    return energy


def read_center_of_mass(lines: List[str]) -> Optional[np.ndarray]:
    """Scan through text for the center of mass"""
    # Example:
    # 'The origin for moment calculation is the CENTER OF MASS  =
    # ( 0.002150, -0.296255  0.086315)'
    # Note the missing comma in the output
    com = None
    for line in lines:
        if 'The origin for moment calculation is the CENTER OF MASS' in line:
            line = re.sub(r'[(),]', '', line)
            com = np.array([float(_) for _ in line.split()[-3:]])
    if com is not None:
        return com * Bohr  # return the last match
    return com


def read_dipole(lines: List[str]) -> Optional[np.ndarray]:
    """Read dipole moment.

    Note that the read dipole moment is for the COM frame of reference.
    """
    dipole = None
    for line in lines:
        if 'Total Dipole Moment' in line:
            dipole = np.array([float(x) for x in line.split()[-3:]]) * Bohr
    return dipole  # Return the last match


def _read_atoms(lines: Sequence[str]) -> Atoms:
    """Read atomic positions and symbols. Create Atoms object."""
    line_start = -1
    natoms = 0

    for ll, line in enumerate(lines):
        if 'Number of atoms' in line:
            natoms = int(line.split()[4])
        elif 'CARTESIAN COORDINATES (ANGSTROEM)' in line:
            line_start = ll + 2

    # Check if atoms present and if their number is given.
    if line_start == -1:
        raise ParseError(
            'No information about the structure in the ORCA output file.'
        )
    elif natoms == 0:
        raise ParseError(
            'No information about number of atoms in the ORCA output file.'
        )

    positions = np.zeros((natoms, 3))
    symbols = [''] * natoms

    for ll, line in enumerate(lines[line_start : (line_start + natoms)]):
        inp = line.split()
        positions[ll, :] = [float(pos) for pos in inp[1:4]]
        symbols[ll] = inp[0]

    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.set_pbc([False, False, False])

    return atoms


def read_forces(lines: List[str]) -> Optional[np.ndarray]:
    """Read forces from output file if available. Else return None.

    Taking the forces from the output files (instead of the engrad-file) to
    be more general. The forces can be present in general output even if
    the engrad file is not there.

    Note: If more than one geometry relaxation step is available,
          forces do not always exist for the first step. In this case, for
          the first step an array of None will be returned. The following
          relaxation steps will then have forces available.
    """
    line_start = -1
    natoms = 0
    record_gradient = True

    for ll, line in enumerate(lines):
        if 'Number of atoms' in line:
            natoms = int(line.split()[4])
        # Read in only first set of forces for each chunk
        # (Excited state calculations can have several sets of
        # forces per chunk)
        elif 'CARTESIAN GRADIENT' in line and record_gradient:
            line_start = ll + 3
            record_gradient = False

    # Check if number of atoms is available.
    if natoms == 0:
        raise ParseError(
            'No information about number of atoms in the ORCA output file.'
        )

    # Forces are not always available. If not available, return None.
    if line_start == -1:
        return None

    forces = np.zeros((natoms, 3))

    for ll, line in enumerate(lines[line_start : (line_start + natoms)]):
        inp = line.split()
        forces[ll, :] = [float(pos) for pos in inp[3:6]]

    forces *= -Hartree / Bohr
    return forces


def get_chunks(lines: Iterable[str]) -> Iterator[list[str]]:
    """Separate out the chunks for each geometry relaxation step."""
    finished = False
    relaxation_finished = False
    relaxation = False

    chunk_endings = [
        'ORCA TERMINATED NORMALLY',
        'ORCA GEOMETRY RELAXATION STEP',
    ]
    chunk_lines = []
    for line in lines:
        # Assemble chunks
        if any([ending in line for ending in chunk_endings]):
            chunk_lines.append(line)
            yield chunk_lines
            chunk_lines = []
        else:
            chunk_lines.append(line)

        if 'ORCA TERMINATED NORMALLY' in line:
            finished = True

        if 'THE OPTIMIZATION HAS CONVERGED' in line:
            relaxation_finished = True

        # Check if calculation is an optimization.
        if 'ORCA SCF GRADIENT CALCULATION' in line:
            relaxation = True

    # Give error if calculation not finished for single-point calculations.
    if not finished and not relaxation:
        raise ParseError('Error: Calculation did not finish!')
    # Give warning if calculation not finished for geometry optimizations.
    elif not finished and relaxation:
        warnings.warn('Calculation did not finish!')
    # Calculation may have finished, but relaxation may have not.
    elif not relaxation_finished and relaxation:
        warnings.warn('Geometry optimization did not converge!')


@reader
def read_orca_output(fd, index=slice(None)):
    """From the ORCA output file: Read Energy, positions, forces
       and dipole moment.

    Create separated atoms object for each geometry frame through
    parsing the output file in chunks.
    """
    images = []

    # Iterate over chunks and create a separate atoms object for each
    for chunk in get_chunks(fd):
        energy = read_energy(chunk)
        atoms = _read_atoms(chunk)
        forces = read_forces(chunk)
        dipole = read_dipole(chunk)
        charge = read_charge(chunk)
        com = read_center_of_mass(chunk)

        # Correct dipole moment for centre-of-mass
        if com is not None and dipole is not None:
            dipole = dipole + com * charge

        atoms.calc = SinglePointDFTCalculator(
            atoms,
            energy=energy,
            free_energy=energy,
            forces=forces,
            # stress=self.stress,
            # stresses=self.stresses,
            # magmom=self.magmom,
            dipole=dipole,
            # dielectric_tensor=self.dielectric_tensor,
            # polarization=self.polarization,
        )
        # collect images
        images.append(atoms)

    return images[index]


@reader
def read_orca_engrad(fd):
    """Read Forces from ORCA .engrad file."""
    getgrad = False
    gradients = []
    tempgrad = []
    for _, line in enumerate(fd):
        if line.find('# The current gradient') >= 0:
            getgrad = True
            gradients = []
            tempgrad = []
            continue
        if getgrad and '#' not in line:
            grad = line.split()[-1]
            tempgrad.append(float(grad))
            if len(tempgrad) == 3:
                gradients.append(tempgrad)
                tempgrad = []
        if '# The at' in line:
            getgrad = False

    forces = -np.array(gradients) * Hartree / Bohr
    return forces


@deprecated(
    'Please use ase.io.read instead of read_orca_outputs, e.g.,\n'
    'from ase.io import read \n'
    'atoms = read("orca.out")',
    DeprecationWarning,
)
def read_orca_outputs(directory, stdout_path):
    """Reproduces old functionality of reading energy, forces etc
       directly from output without creation of atoms object.
       This is kept to ensure backwards compatability
    .. deprecated:: 3.24.0
       Use of read_orca_outputs is deprected, please
       process ORCA output by using ase.io.read
       e.g., read('orca.out')"
    """
    stdout_path = Path(stdout_path)
    atoms = read_orca_output(stdout_path, index=-1)
    results = {}
    results['energy'] = atoms.get_total_energy()
    results['free_energy'] = atoms.get_total_energy()

    try:
        results['dipole'] = atoms.get_dipole_moment()
    except PropertyNotImplementedError:
        pass

    # Does engrad always exist? - No!
    # Will there be other files -No -> We should just take engrad
    # as a direct argument.  Or maybe this function does not even need to
    # exist.
    engrad_path = stdout_path.with_suffix('.engrad')
    if engrad_path.is_file():
        results['forces'] = read_orca_engrad(engrad_path)
        print("""Warning: If you are reading in an engrad file from a
              geometry optimization, check very carefully.
              ORCA does not by default supply the forces for the
              converged geometry!""")
    return results
