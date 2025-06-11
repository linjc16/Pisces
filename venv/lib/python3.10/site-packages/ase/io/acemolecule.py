# fmt: off

import numpy as np

import ase.units
from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.data import chemical_symbols
from ase.io import read


def parse_geometry(filename):
    '''Read atoms geometry from ACE-Molecule log file and put it to self.data.
    Parameters
    ==========
    filename: ACE-Molecule log file.

    Returns
    =======
    Dictionary of parsed geometry data.
    retval["Atomic_numbers"]: list of atomic numbers
    retval["Positions"]: list of [x, y, z] coordinates for each atoms.
    '''
    with open(filename) as fd:
        lines = fd.readlines()
        start_line = 0
        end_line = 0
        for i, line in enumerate(lines):
            if line == '====================  Atoms  =====================\n':
                start_line = i
            if start_line != 0 and len(line.split('=')) > 3:
                end_line = i
                if start_line != end_line:
                    break
        atoms = []
        positions = []
        for i in range(start_line + 1, end_line):
            atomic_number = lines[i].split()[0]
            atoms.append(str(chemical_symbols[int(atomic_number)]))
            xyz = [float(n) for n in lines[i].split()[1:4]]
            positions.append(xyz)

        return {"Atomic_numbers": atoms, "Positions": positions}


def read_acemolecule_out(filename):
    '''Interface to ACEMoleculeReader, return values for corresponding quantity

    Parameters
    ==========
    filename: ACE-Molecule log file.
    quantity: One of atoms, energy, forces, excitation-energy.

    Returns
    =======
     - quantity = 'excitation-energy':
       returns None. This is placeholder function to run TDDFT calculations
       without IndexError.
     - quantity = 'energy':
       returns energy as float value.
     - quantity = 'forces':
       returns force of each atoms as numpy array of shape (natoms, 3).
     - quantity = 'atoms':
       returns ASE atoms object.
    '''
    data = parse_geometry(filename)
    atom_symbol = np.array(data["Atomic_numbers"])
    positions = np.array(data["Positions"])
    atoms = Atoms(atom_symbol, positions=positions)
    energy = None
    forces = None
    excitation_energy = None
#    results = {}
#    if len(results)<1:
    with open(filename) as fd:
        lines = fd.readlines()

    for i in range(len(lines) - 1, 1, -1):
        line = lines[i].split()
        if len(line) > 2:
            if line[0] == 'Total' and line[1] == 'energy':
                energy = float(line[3])
                break
    # energy must be modified, hartree to eV
    energy *= ase.units.Hartree

    forces = []
    for i in range(len(lines) - 1, 1, -1):
        if '!============================' in lines[i]:
            endline_num = i
        if '! Force:: List of total force in atomic unit' in lines[i]:
            startline_num = i + 2
            for j in range(startline_num, endline_num):
                forces.append(lines[j].split()[3:6])
            convert = ase.units.Hartree / ase.units.Bohr
            forces = np.array(forces, dtype=float) * convert
            break
    if len(forces) <= 0:
        forces = None

    # Set calculator to
    calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
    atoms.calc = calc

    results = {}
    results['energy'] = energy
    results['atoms'] = atoms
    results['forces'] = forces
    results['excitation-energy'] = excitation_energy
    return results


def read_acemolecule_input(filename):
    '''Reads a ACE-Molecule input file
    Parameters
    ==========
    filename: ACE-Molecule input file name

    Returns
    =======
    ASE atoms object containing geometry only.
    '''
    with open(filename) as fd:
        for line in fd:
            if len(line.split('GeometryFilename')) > 1:
                geometryfile = line.split()[1]
                break
    atoms = read(geometryfile, format='xyz')
    return atoms
