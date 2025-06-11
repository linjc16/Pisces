# fmt: off

"""Reads Quantum ESPRESSO files.

Read multiple structures and results from pw.x output files. Read
structures from pw.x input files.

Built for PWSCF v.5.3.0 but should work with earlier and later versions.
Can deal with most major functionality, with the notable exception of ibrav,
for which we only support ibrav == 0 and force CELL_PARAMETERS to be provided
explicitly.

Units are converted using CODATA 2006, as used internally by Quantum
ESPRESSO.
"""

import operator as op
import re
import warnings
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np

from ase.atoms import Atoms
from ase.calculators.calculator import kpts2ndarray, kpts2sizeandoffsets
from ase.calculators.singlepoint import (
    SinglePointDFTCalculator,
    SinglePointKPoint,
)
from ase.constraints import FixAtoms, FixCartesian
from ase.data import chemical_symbols
from ase.dft.kpoints import kpoint_convert
from ase.io.espresso_namelist.keys import pw_keys
from ase.io.espresso_namelist.namelist import Namelist
from ase.units import create_units
from ase.utils import deprecated, reader, writer

# Quantum ESPRESSO uses CODATA 2006 internally
units = create_units('2006')

# Section identifiers
_PW_START = 'Program PWSCF'
_PW_END = 'End of self-consistent calculation'
_PW_CELL = 'CELL_PARAMETERS'
_PW_POS = 'ATOMIC_POSITIONS'
_PW_MAGMOM = 'Magnetic moment per site'
_PW_FORCE = 'Forces acting on atoms'
_PW_TOTEN = '!    total energy'
_PW_STRESS = 'total   stress'
_PW_FERMI = 'the Fermi energy is'
_PW_HIGHEST_OCCUPIED = 'highest occupied level'
_PW_HIGHEST_OCCUPIED_LOWEST_FREE = 'highest occupied, lowest unoccupied level'
_PW_KPTS = 'number of k points='
_PW_BANDS = _PW_END
_PW_BANDSTRUCTURE = 'End of band structure calculation'
_PW_DIPOLE = "Debye"
_PW_DIPOLE_DIRECTION = "Computed dipole along edir"

# ibrav error message
ibrav_error_message = (
    'ASE does not support ibrav != 0. Note that with ibrav '
    '== 0, Quantum ESPRESSO will still detect the symmetries '
    'of your system because the CELL_PARAMETERS are defined '
    'to a high level of precision.')


@reader
def read_espresso_out(fileobj, index=slice(None), results_required=True):
    """Reads Quantum ESPRESSO output files.

    The atomistic configurations as well as results (energy, force, stress,
    magnetic moments) of the calculation are read for all configurations
    within the output file.

    Will probably raise errors for broken or incomplete files.

    Parameters
    ----------
    fileobj : file|str
        A file like object or filename
    index : slice
        The index of configurations to extract.
    results_required : bool
        If True, atomistic configurations that do not have any
        associated results will not be included. This prevents double
        printed configurations and incomplete calculations from being
        returned as the final configuration with no results data.

    Yields
    ------
    structure : Atoms
        The next structure from the index slice. The Atoms has a
        SinglePointCalculator attached with any results parsed from
        the file.


    """
    # work with a copy in memory for faster random access
    pwo_lines = fileobj.readlines()

    # TODO: index -1 special case?
    # Index all the interesting points
    indexes = {
        _PW_START: [],
        _PW_END: [],
        _PW_CELL: [],
        _PW_POS: [],
        _PW_MAGMOM: [],
        _PW_FORCE: [],
        _PW_TOTEN: [],
        _PW_STRESS: [],
        _PW_FERMI: [],
        _PW_HIGHEST_OCCUPIED: [],
        _PW_HIGHEST_OCCUPIED_LOWEST_FREE: [],
        _PW_KPTS: [],
        _PW_BANDS: [],
        _PW_BANDSTRUCTURE: [],
        _PW_DIPOLE: [],
        _PW_DIPOLE_DIRECTION: [],
    }

    for idx, line in enumerate(pwo_lines):
        for identifier in indexes:
            if identifier in line:
                indexes[identifier].append(idx)

    # Configurations are either at the start, or defined in ATOMIC_POSITIONS
    # in a subsequent step. Can deal with concatenated output files.
    all_config_indexes = sorted(indexes[_PW_START] +
                                indexes[_PW_POS])

    # Slice only requested indexes
    # setting results_required argument stops configuration-only
    # structures from being returned. This ensures the [-1] structure
    # is one that has results. Two cases:
    # - SCF of last configuration is not converged, job terminated
    #   abnormally.
    # - 'relax' and 'vc-relax' re-prints the final configuration but
    #   only 'vc-relax' recalculates.
    if results_required:
        results_indexes = sorted(indexes[_PW_TOTEN] + indexes[_PW_FORCE] +
                                 indexes[_PW_STRESS] + indexes[_PW_MAGMOM] +
                                 indexes[_PW_BANDS] +
                                 indexes[_PW_BANDSTRUCTURE])

        # Prune to only configurations with results data before the next
        # configuration
        results_config_indexes = []
        for config_index, config_index_next in zip(
                all_config_indexes,
                all_config_indexes[1:] + [len(pwo_lines)]):
            if any(config_index < results_index < config_index_next
                    for results_index in results_indexes):
                results_config_indexes.append(config_index)

        # slice from the subset
        image_indexes = results_config_indexes[index]
    else:
        image_indexes = all_config_indexes[index]

    # Extract initialisation information each time PWSCF starts
    # to add to subsequent configurations. Use None so slices know
    # when to fill in the blanks.
    pwscf_start_info = {idx: None for idx in indexes[_PW_START]}

    for image_index in image_indexes:
        # Find the nearest calculation start to parse info. Needed in,
        # for example, relaxation where cell is only printed at the
        # start.
        if image_index in indexes[_PW_START]:
            prev_start_index = image_index
        else:
            # The greatest start index before this structure
            prev_start_index = [idx for idx in indexes[_PW_START]
                                if idx < image_index][-1]

        # add structure to reference if not there
        if pwscf_start_info[prev_start_index] is None:
            pwscf_start_info[prev_start_index] = parse_pwo_start(
                pwo_lines, prev_start_index)

        # Get the bounds for information for this structure. Any associated
        # values will be between the image_index and the following one,
        # EXCEPT for cell, which will be 4 lines before if it exists.
        for next_index in all_config_indexes:
            if next_index > image_index:
                break
        else:
            # right to the end of the file
            next_index = len(pwo_lines)

        # Get the structure
        # Use this for any missing data
        prev_structure = pwscf_start_info[prev_start_index]['atoms']
        cell_alat = pwscf_start_info[prev_start_index]['alat']
        if image_index in indexes[_PW_START]:
            structure = prev_structure.copy()  # parsed from start info
        else:
            if _PW_CELL in pwo_lines[image_index - 5]:
                # CELL_PARAMETERS would be just before positions if present
                cell, _ = get_cell_parameters(
                    pwo_lines[image_index - 5:image_index])
            else:
                cell = prev_structure.cell
                cell_alat = pwscf_start_info[prev_start_index]['alat']

            # give at least enough lines to parse the positions
            # should be same format as input card
            n_atoms = len(prev_structure)
            positions_card = get_atomic_positions(
                pwo_lines[image_index:image_index + n_atoms + 1],
                n_atoms=n_atoms, cell=cell, alat=cell_alat)

            # convert to Atoms object
            symbols = [label_to_symbol(position[0]) for position in
                       positions_card]
            positions = [position[1] for position in positions_card]
            structure = Atoms(symbols=symbols, positions=positions, cell=cell,
                              pbc=True)

        # Extract calculation results
        # Energy
        energy = None
        for energy_index in indexes[_PW_TOTEN]:
            if image_index < energy_index < next_index:
                energy = float(
                    pwo_lines[energy_index].split()[-2]) * units['Ry']

        # Forces
        forces = None
        for force_index in indexes[_PW_FORCE]:
            if image_index < force_index < next_index:
                # Before QE 5.3 'negative rho' added 2 lines before forces
                # Use exact lines to stop before 'non-local' forces
                # in high verbosity
                if not pwo_lines[force_index + 2].strip():
                    force_index += 4
                else:
                    force_index += 2
                # assume contiguous
                forces = [
                    [float(x) for x in force_line.split()[-3:]] for force_line
                    in pwo_lines[force_index:force_index + len(structure)]]
                forces = np.array(forces) * units['Ry'] / units['Bohr']

        # Stress
        stress = None
        for stress_index in indexes[_PW_STRESS]:
            if image_index < stress_index < next_index:
                sxx, sxy, sxz = pwo_lines[stress_index + 1].split()[:3]
                _, syy, syz = pwo_lines[stress_index + 2].split()[:3]
                _, _, szz = pwo_lines[stress_index + 3].split()[:3]
                stress = np.array([sxx, syy, szz, syz, sxz, sxy], dtype=float)
                # sign convention is opposite of ase
                stress *= -1 * units['Ry'] / (units['Bohr'] ** 3)

        # Magmoms
        magmoms = None
        for magmoms_index in indexes[_PW_MAGMOM]:
            if image_index < magmoms_index < next_index:
                magmoms = [
                    float(mag_line.split()[-1]) for mag_line
                    in pwo_lines[magmoms_index + 1:
                                 magmoms_index + 1 + len(structure)]]

        # Dipole moment
        dipole = None
        if indexes[_PW_DIPOLE]:
            for dipole_index in indexes[_PW_DIPOLE]:
                if image_index < dipole_index < next_index:
                    _dipole = float(pwo_lines[dipole_index].split()[-2])

            for dipole_index in indexes[_PW_DIPOLE_DIRECTION]:
                if image_index < dipole_index < next_index:
                    _direction = pwo_lines[dipole_index].strip()
                    prefix = 'Computed dipole along edir('
                    _direction = _direction[len(prefix):]
                    _direction = int(_direction[0])

            dipole = np.eye(3)[_direction - 1] * _dipole * units['Debye']

        # Fermi level / highest occupied level
        efermi = None
        for fermi_index in indexes[_PW_FERMI]:
            if image_index < fermi_index < next_index:
                efermi = float(pwo_lines[fermi_index].split()[-2])

        if efermi is None:
            for ho_index in indexes[_PW_HIGHEST_OCCUPIED]:
                if image_index < ho_index < next_index:
                    efermi = float(pwo_lines[ho_index].split()[-1])

        if efermi is None:
            for holf_index in indexes[_PW_HIGHEST_OCCUPIED_LOWEST_FREE]:
                if image_index < holf_index < next_index:
                    efermi = float(pwo_lines[holf_index].split()[-2])

        # K-points
        ibzkpts = None
        weights = None
        kpoints_warning = "Number of k-points >= 100: " + \
                          "set verbosity='high' to print them."

        for kpts_index in indexes[_PW_KPTS]:
            nkpts = int(re.findall(r'\b\d+\b', pwo_lines[kpts_index])[0])
            kpts_index += 2

            if pwo_lines[kpts_index].strip() == kpoints_warning:
                continue

            # QE prints the k-points in units of 2*pi/alat
            cell = structure.get_cell()
            ibzkpts = []
            weights = []
            for i in range(nkpts):
                L = pwo_lines[kpts_index + i].split()
                weights.append(float(L[-1]))
                coord = np.array([L[-6], L[-5], L[-4].strip('),')],
                                 dtype=float)
                coord *= 2 * np.pi / cell_alat
                coord = kpoint_convert(cell, ckpts_kv=coord)
                ibzkpts.append(coord)
            ibzkpts = np.array(ibzkpts)
            weights = np.array(weights)

        # Bands
        kpts = None
        kpoints_warning = "Number of k-points >= 100: " + \
                          "set verbosity='high' to print the bands."

        for bands_index in indexes[_PW_BANDS] + indexes[_PW_BANDSTRUCTURE]:
            if image_index < bands_index < next_index:
                bands_index += 1
                # skip over the lines with DFT+U occupation matrices
                if 'enter write_ns' in pwo_lines[bands_index]:
                    while 'exit write_ns' not in pwo_lines[bands_index]:
                        bands_index += 1
                bands_index += 1

                if pwo_lines[bands_index].strip() == kpoints_warning:
                    continue

                assert ibzkpts is not None
                spin, bands, eigenvalues = 0, [], [[], []]

                while True:
                    L = pwo_lines[bands_index].replace('-', ' -').split()
                    if len(L) == 0:
                        if len(bands) > 0:
                            eigenvalues[spin].append(bands)
                            bands = []
                    elif L == ['occupation', 'numbers']:
                        # Skip the lines with the occupation numbers
                        bands_index += len(eigenvalues[spin][0]) // 8 + 1
                    elif L[0] == 'k' and L[1].startswith('='):
                        pass
                    elif 'SPIN' in L:
                        if 'DOWN' in L:
                            spin += 1
                    else:
                        try:
                            bands.extend(map(float, L))
                        except ValueError:
                            break
                    bands_index += 1

                if spin == 1:
                    assert len(eigenvalues[0]) == len(eigenvalues[1])
                assert len(eigenvalues[0]) == len(ibzkpts), \
                    (np.shape(eigenvalues), len(ibzkpts))

                kpts = []
                for s in range(spin + 1):
                    for w, k, e in zip(weights, ibzkpts, eigenvalues[s]):
                        kpt = SinglePointKPoint(w, s, k, eps_n=e)
                        kpts.append(kpt)

        # Put everything together
        #
        # In PW the forces are consistent with the "total energy"; that's why
        # its value must be assigned to free_energy.
        # PW doesn't compute the extrapolation of the energy to 0K smearing
        # the closer thing to this is again the total energy that contains
        # the correct (i.e. variational) form of the band energy is
        #   Eband = \int e N(e) de   for e<Ef , where N(e) is the DOS
        # This differs by the term (-TS)  from the sum of KS eigenvalues:
        #    Eks = \sum wg(n,k) et(n,k)
        # which is non variational. When a Fermi-Dirac function is used
        # for a given T, the variational energy is REALLY the free energy F,
        # and F = E - TS , with E = non variational energy.
        #
        calc = SinglePointDFTCalculator(structure, energy=energy,
                                        free_energy=energy,
                                        forces=forces, stress=stress,
                                        magmoms=magmoms, efermi=efermi,
                                        ibzkpts=ibzkpts, dipole=dipole)
        calc.kpts = kpts
        structure.calc = calc

        yield structure


def parse_pwo_start(lines, index=0):
    """Parse Quantum ESPRESSO calculation info from lines,
    starting from index. Return a dictionary containing extracted
    information.

    - `celldm(1)`: lattice parameters (alat)
    - `cell`: unit cell in Angstrom
    - `symbols`: element symbols for the structure
    - `positions`: cartesian coordinates of atoms in Angstrom
    - `atoms`: an `ase.Atoms` object constructed from the extracted data

    Parameters
    ----------
    lines : list[str]
        Contents of PWSCF output file.
    index : int
        Line number to begin parsing. Only first calculation will
        be read.

    Returns
    -------
    info : dict
        Dictionary of calculation parameters, including `celldm(1)`, `cell`,
        `symbols`, `positions`, `atoms`.

    Raises
    ------
    KeyError
        If interdependent values cannot be found (especially celldm(1))
        an error will be raised as other quantities cannot then be
        calculated (e.g. cell and positions).
    """
    # TODO: extend with extra DFT info?

    info = {}

    for idx, line in enumerate(lines[index:], start=index):
        if 'celldm(1)' in line:
            # celldm(1) has more digits than alat!!
            info['celldm(1)'] = float(line.split()[1]) * units['Bohr']
            info['alat'] = info['celldm(1)']
        elif 'number of atoms/cell' in line:
            info['nat'] = int(line.split()[-1])
        elif 'number of atomic types' in line:
            info['ntyp'] = int(line.split()[-1])
        elif 'crystal axes:' in line:
            info['cell'] = info['celldm(1)'] * np.array([
                [float(x) for x in lines[idx + 1].split()[3:6]],
                [float(x) for x in lines[idx + 2].split()[3:6]],
                [float(x) for x in lines[idx + 3].split()[3:6]]])
        elif 'positions (alat units)' in line:
            info['symbols'], info['positions'] = [], []

            for at_line in lines[idx + 1:idx + 1 + info['nat']]:
                sym, x, y, z = parse_position_line(at_line)
                info['symbols'].append(label_to_symbol(sym))
                info['positions'].append([x * info['celldm(1)'],
                                          y * info['celldm(1)'],
                                          z * info['celldm(1)']])
            # This should be the end of interesting info.
            # Break here to avoid dealing with large lists of kpoints.
            # Will need to be extended for DFTCalculator info.
            break

    # Make atoms for convenience
    info['atoms'] = Atoms(symbols=info['symbols'],
                          positions=info['positions'],
                          cell=info['cell'], pbc=True)

    return info


def parse_position_line(line):
    """Parse a single line from a pw.x output file.

    The line must contain information about the atomic symbol and the position,
    e.g.

    995           Sb  tau( 995) = (   1.4212023   0.7037863   0.1242640  )

    Parameters
    ----------
    line : str
        Line to be parsed.

    Returns
    -------
    sym : str
        Atomic symbol.
    x : float
        x-position.
    y : float
        y-position.
    z : float
        z-position.
    """
    pat = re.compile(r'\s*\d+\s*(\S+)\s*tau\(\s*\d+\)\s*='
                     r'\s*\(\s*(\S+)\s+(\S+)\s+(\S+)\s*\)')
    match = pat.match(line)
    assert match is not None
    sym, x, y, z = match.group(1, 2, 3, 4)
    return sym, float(x), float(y), float(z)


@reader
def read_espresso_in(fileobj):
    """Parse a Quantum ESPRESSO input files, '.in', '.pwi'.

    ESPRESSO inputs are generally a fortran-namelist format with custom
    blocks of data. The namelist is parsed as a dict and an atoms object
    is constructed from the included information.

    Parameters
    ----------
    fileobj : file | str
        A file-like object that supports line iteration with the contents
        of the input file, or a filename.

    Returns
    -------
    atoms : Atoms
        Structure defined in the input file.

    Raises
    ------
    KeyError
        Raised for missing keys that are required to process the file
    """
    # parse namelist section and extract remaining lines
    data, card_lines = read_fortran_namelist(fileobj)

    # get the cell if ibrav=0
    if 'system' not in data:
        raise KeyError('Required section &SYSTEM not found.')
    elif 'ibrav' not in data['system']:
        raise KeyError('ibrav is required in &SYSTEM')
    elif data['system']['ibrav'] == 0:
        # celldm(1) is in Bohr, A is in angstrom. celldm(1) will be
        # used even if A is also specified.
        if 'celldm(1)' in data['system']:
            alat = data['system']['celldm(1)'] * units['Bohr']
        elif 'A' in data['system']:
            alat = data['system']['A']
        else:
            alat = None
        cell, _ = get_cell_parameters(card_lines, alat=alat)
    else:
        raise ValueError(ibrav_error_message)

    # species_info holds some info for each element
    species_card = get_atomic_species(
        card_lines, n_species=data['system']['ntyp'])
    species_info = {}
    for ispec, (label, weight, pseudo) in enumerate(species_card):
        symbol = label_to_symbol(label)

        # starting_magnetization is in fractions of valence electrons
        magnet_key = f"starting_magnetization({ispec + 1})"
        magmom = data["system"].get(magnet_key, 0.0)
        species_info[symbol] = {"weight": weight, "pseudo": pseudo,
                                "magmom": magmom}

    positions_card = get_atomic_positions(
        card_lines, n_atoms=data['system']['nat'], cell=cell, alat=alat)

    symbols = [label_to_symbol(position[0]) for position in positions_card]
    positions = [position[1] for position in positions_card]
    constraint_flags = [position[2] for position in positions_card]
    magmoms = [species_info[symbol]["magmom"] for symbol in symbols]

    # TODO: put more info into the atoms object
    # e.g magmom, forces.
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True,
                  magmoms=magmoms)
    atoms.set_constraint(convert_constraint_flags(constraint_flags))

    return atoms


def get_atomic_positions(lines, n_atoms, cell=None, alat=None):
    """Parse atom positions from ATOMIC_POSITIONS card.

    Parameters
    ----------
    lines : list[str]
        A list of lines containing the ATOMIC_POSITIONS card.
    n_atoms : int
        Expected number of atoms. Only this many lines will be parsed.
    cell : np.array
        Unit cell of the crystal. Only used with crystal coordinates.
    alat : float
        Lattice parameter for atomic coordinates. Only used for alat case.

    Returns
    -------
    positions : list[(str, (float, float, float), (int, int, int))]
        A list of the ordered atomic positions in the format:
        label, (x, y, z), (if_x, if_y, if_z)
        Force multipliers are set to None if not present.

    Raises
    ------
    ValueError
        Any problems parsing the data result in ValueError

    """

    positions = None
    # no blanks or comment lines, can the consume n_atoms lines for positions
    trimmed_lines = (line for line in lines if line.strip() and line[0] != '#')

    for line in trimmed_lines:
        if line.strip().startswith('ATOMIC_POSITIONS'):
            if positions is not None:
                raise ValueError('Multiple ATOMIC_POSITIONS specified')
            # Priority and behaviour tested with QE 5.3
            if 'crystal_sg' in line.lower():
                raise NotImplementedError('CRYSTAL_SG not implemented')
            elif 'crystal' in line.lower():
                cell = cell
            elif 'bohr' in line.lower():
                cell = np.identity(3) * units['Bohr']
            elif 'angstrom' in line.lower():
                cell = np.identity(3)
            # elif 'alat' in line.lower():
            #     cell = np.identity(3) * alat
            else:
                if alat is None:
                    raise ValueError('Set lattice parameter in &SYSTEM for '
                                     'alat coordinates')
                # Always the default, will be DEPRECATED as mandatory
                # in future
                cell = np.identity(3) * alat

            positions = []
            for _ in range(n_atoms):
                split_line = next(trimmed_lines).split()
                # These can be fractions and other expressions
                position = np.dot((infix_float(split_line[1]),
                                   infix_float(split_line[2]),
                                   infix_float(split_line[3])), cell)
                if len(split_line) > 4:
                    force_mult = tuple(int(split_line[i]) for i in (4, 5, 6))
                else:
                    force_mult = None

                positions.append((split_line[0], position, force_mult))

    return positions


def get_atomic_species(lines, n_species):
    """Parse atomic species from ATOMIC_SPECIES card.

    Parameters
    ----------
    lines : list[str]
        A list of lines containing the ATOMIC_POSITIONS card.
    n_species : int
        Expected number of atom types. Only this many lines will be parsed.

    Returns
    -------
    species : list[(str, float, str)]

    Raises
    ------
    ValueError
        Any problems parsing the data result in ValueError
    """

    species = None
    # no blanks or comment lines, can the consume n_atoms lines for positions
    trimmed_lines = (line.strip() for line in lines
                     if line.strip() and not line.startswith('#'))

    for line in trimmed_lines:
        if line.startswith('ATOMIC_SPECIES'):
            if species is not None:
                raise ValueError('Multiple ATOMIC_SPECIES specified')

            species = []
            for _dummy in range(n_species):
                label_weight_pseudo = next(trimmed_lines).split()
                species.append((label_weight_pseudo[0],
                                float(label_weight_pseudo[1]),
                                label_weight_pseudo[2]))

    return species


def get_cell_parameters(lines, alat=None):
    """Parse unit cell from CELL_PARAMETERS card.

    Parameters
    ----------
    lines : list[str]
        A list with lines containing the CELL_PARAMETERS card.
    alat : float | None
        Unit of lattice vectors in Angstrom. Only used if the card is
        given in units of alat. alat must be None if CELL_PARAMETERS card
        is in Bohr or Angstrom. For output files, alat will be parsed from
        the card header and used in preference to this value.

    Returns
    -------
    cell : np.array | None
        Cell parameters as a 3x3 array in Angstrom. If no cell is found
        None will be returned instead.
    cell_alat : float | None
        If a value for alat is given in the card header, this is also
        returned, otherwise this will be None.

    Raises
    ------
    ValueError
        If CELL_PARAMETERS are given in units of bohr or angstrom
        and alat is not
    """

    cell = None
    cell_alat = None
    # no blanks or comment lines, can take three lines for cell
    trimmed_lines = (line for line in lines if line.strip() and line[0] != '#')

    for line in trimmed_lines:
        if line.strip().startswith('CELL_PARAMETERS'):
            if cell is not None:
                # multiple definitions
                raise ValueError('CELL_PARAMETERS specified multiple times')
            # Priority and behaviour tested with QE 5.3
            if 'bohr' in line.lower():
                if alat is not None:
                    raise ValueError('Lattice parameters given in '
                                     '&SYSTEM celldm/A and CELL_PARAMETERS '
                                     'bohr')
                cell_units = units['Bohr']
            elif 'angstrom' in line.lower():
                if alat is not None:
                    raise ValueError('Lattice parameters given in '
                                     '&SYSTEM celldm/A and CELL_PARAMETERS '
                                     'angstrom')
                cell_units = 1.0
            elif 'alat' in line.lower():
                # Output file has (alat = value) (in Bohrs)
                if '=' in line:
                    alat = float(line.strip(') \n').split()[-1]) * units['Bohr']
                    cell_alat = alat
                elif alat is None:
                    raise ValueError('Lattice parameters must be set in '
                                     '&SYSTEM for alat units')
                cell_units = alat
            elif alat is None:
                # may be DEPRECATED in future
                cell_units = units['Bohr']
            else:
                # may be DEPRECATED in future
                cell_units = alat
            # Grab the parameters; blank lines have been removed
            cell = [[ffloat(x) for x in next(trimmed_lines).split()[:3]],
                    [ffloat(x) for x in next(trimmed_lines).split()[:3]],
                    [ffloat(x) for x in next(trimmed_lines).split()[:3]]]
            cell = np.array(cell) * cell_units

    return cell, cell_alat


def convert_constraint_flags(constraint_flags):
    """Convert Quantum ESPRESSO constraint flags to ASE Constraint objects.

    Parameters
    ----------
    constraint_flags : list[tuple[int, int, int]]
        List of constraint flags (0: fixed, 1: moved) for all the atoms.
        If the flag is None, there are no constraints on the atom.

    Returns
    -------
    constraints : list[FixAtoms | FixCartesian]
        List of ASE Constraint objects.
    """
    constraints = []
    for i, constraint in enumerate(constraint_flags):
        if constraint is None:
            continue
        # mask: False (0): moved, True (1): fixed
        mask = ~np.asarray(constraint, bool)
        constraints.append(FixCartesian(i, mask))
    return canonicalize_constraints(constraints)


def canonicalize_constraints(constraints):
    """Canonicalize ASE FixCartesian constraints.

    If the given FixCartesian constraints share the same `mask`, they can be
    merged into one. Further, if `mask == (True, True, True)`, they can be
    converted as `FixAtoms`. This method "canonicalizes" FixCartesian objects
    in such a way.

    Parameters
    ----------
    constraints : List[FixCartesian]
        List of ASE FixCartesian constraints.

    Returns
    -------
    constrants_canonicalized : List[FixAtoms | FixCartesian]
        List of ASE Constraint objects.
    """
    # https://docs.python.org/3/library/collections.html#defaultdict-examples
    indices_for_masks = defaultdict(list)
    for constraint in constraints:
        key = tuple((constraint.mask).tolist())
        indices_for_masks[key].extend(constraint.index.tolist())

    constraints_canonicalized = []
    for mask, indices in indices_for_masks.items():
        if mask == (False, False, False):  # no directions are fixed
            continue
        if mask == (True, True, True):  # all three directions are fixed
            constraints_canonicalized.append(FixAtoms(indices))
        else:
            constraints_canonicalized.append(FixCartesian(indices, mask))

    return constraints_canonicalized


def str_to_value(string):
    """Attempt to convert string into int, float (including fortran double),
    or bool, in that order, otherwise return the string.
    Valid (case-insensitive) bool values are: '.true.', '.t.', 'true'
    and 't' (or false equivalents).

    Parameters
    ----------
    string : str
        Test to parse for a datatype

    Returns
    -------
    value : any
        Parsed string as the most appropriate datatype of int, float,
        bool or string.
    """

    # Just an integer
    try:
        return int(string)
    except ValueError:
        pass
    # Standard float
    try:
        return float(string)
    except ValueError:
        pass
    # Fortran double
    try:
        return ffloat(string)
    except ValueError:
        pass

    # possible bool, else just the raw string
    if string.lower() in ('.true.', '.t.', 'true', 't'):
        return True
    elif string.lower() in ('.false.', '.f.', 'false', 'f'):
        return False
    else:
        return string.strip("'")


def read_fortran_namelist(fileobj):
    """Takes a fortran-namelist formatted file and returns nested
    dictionaries of sections and key-value data, followed by a list
    of lines of text that do not fit the specifications.
    Behaviour is taken from Quantum ESPRESSO 5.3. Parses fairly
    convoluted files the same way that QE should, but may not get
    all the MANDATORY rules and edge cases for very non-standard files
    Ignores anything after '!' in a namelist, split pairs on ','
    to include multiple key=values on a line, read values on section
    start and end lines, section terminating character, '/', can appear
    anywhere on a line. All of these are ignored if the value is in 'quotes'.

    Parameters
    ----------
    fileobj : file
        An open file-like object.

    Returns
    -------
    data : dict[str, dict]
        Dictionary for each section in the namelist with
        key = value pairs of data.
    additional_cards : list[str]
        Any lines not used to create the data,
        assumed to belong to 'cards' in the input file.
    """

    data = {}
    card_lines = []
    in_namelist = False
    section = 'none'  # can't be in a section without changing this

    for line in fileobj:
        # leading and trailing whitespace never needed
        line = line.strip()
        if line.startswith('&'):
            # inside a namelist
            section = line.split()[0][1:].lower()  # case insensitive
            if section in data:
                # Repeated sections are completely ignored.
                # (Note that repeated keys overwrite within a section)
                section = "_ignored"
            data[section] = {}
            in_namelist = True
        if not in_namelist and line:
            # Stripped line is Truthy, so safe to index first character
            if line[0] not in ('!', '#'):
                card_lines.append(line)
        if in_namelist:
            # parse k, v from line:
            key = []
            value = None
            in_quotes = False
            for character in line:
                if character == ',' and value is not None and not in_quotes:
                    # finished value:
                    data[section][''.join(key).strip()] = str_to_value(
                        ''.join(value).strip())
                    key = []
                    value = None
                elif character == '=' and value is None and not in_quotes:
                    # start writing value
                    value = []
                elif character == "'":
                    # only found in value anyway
                    in_quotes = not in_quotes
                    value.append("'")
                elif character == '!' and not in_quotes:
                    break
                elif character == '/' and not in_quotes:
                    in_namelist = False
                    break
                elif value is not None:
                    value.append(character)
                else:
                    key.append(character)
            if value is not None:
                data[section][''.join(key).strip()] = str_to_value(
                    ''.join(value).strip())

    return Namelist(data), card_lines


def ffloat(string):
    """Parse float from fortran compatible float definitions.

    In fortran exponents can be defined with 'd' or 'q' to symbolise
    double or quad precision numbers. Double precision numbers are
    converted to python floats and quad precision values are interpreted
    as numpy longdouble values (platform specific precision).

    Parameters
    ----------
    string : str
        A string containing a number in fortran real format

    Returns
    -------
    value : float | np.longdouble
        Parsed value of the string.

    Raises
    ------
    ValueError
        Unable to parse a float value.

    """

    if 'q' in string.lower():
        return np.longdouble(string.lower().replace('q', 'e'))
    else:
        return float(string.lower().replace('d', 'e'))


def label_to_symbol(label):
    """Convert a valid espresso ATOMIC_SPECIES label to a
    chemical symbol.

    Parameters
    ----------
    label : str
        chemical symbol X (1 or 2 characters, case-insensitive)
        or chemical symbol plus a number or a letter, as in
        "Xn" (e.g. Fe1) or "X_*" or "X-*" (e.g. C1, C_h;
        max total length cannot exceed 3 characters).

    Returns
    -------
    symbol : str
        The best matching species from ase.utils.chemical_symbols

    Raises
    ------
    KeyError
        Couldn't find an appropriate species.

    Notes
    -----
        It's impossible to tell whether e.g. He is helium
        or hydrogen labelled 'e'.
    """

    # possibly a two character species
    # ase Atoms need proper case of chemical symbols.
    if len(label) >= 2:
        test_symbol = label[0].upper() + label[1].lower()
        if test_symbol in chemical_symbols:
            return test_symbol
    # finally try with one character
    test_symbol = label[0].upper()
    if test_symbol in chemical_symbols:
        return test_symbol
    else:
        raise KeyError('Could not parse species from label {}.'
                       ''.format(label))


def infix_float(text):
    """Parse simple infix maths into a float for compatibility with
    Quantum ESPRESSO ATOMIC_POSITIONS cards. Note: this works with the
    example, and most simple expressions, but the capabilities of
    the two parsers are not identical. Will also parse a normal float
    value properly, but slowly.

    >>> infix_float('1/2*3^(-1/2)')
    0.28867513459481287

    Parameters
    ----------
    text : str
        An arithmetic expression using +, -, *, / and ^, including brackets.

    Returns
    -------
    value : float
        Result of the mathematical expression.

    """

    def middle_brackets(full_text):
        """Extract text from innermost brackets."""
        start, end = 0, len(full_text)
        for (idx, char) in enumerate(full_text):
            if char == '(':
                start = idx
            if char == ')':
                end = idx + 1
                break
        return full_text[start:end]

    def eval_no_bracket_expr(full_text):
        """Calculate value of a mathematical expression, no brackets."""
        exprs = [('+', op.add), ('*', op.mul),
                 ('/', op.truediv), ('^', op.pow)]
        full_text = full_text.lstrip('(').rstrip(')')
        try:
            return float(full_text)
        except ValueError:
            for symbol, func in exprs:
                if symbol in full_text:
                    left, right = full_text.split(symbol, 1)  # single split
                    return func(eval_no_bracket_expr(left),
                                eval_no_bracket_expr(right))

    while '(' in text:
        middle = middle_brackets(text)
        text = text.replace(middle, f'{eval_no_bracket_expr(middle)}')

    return float(eval_no_bracket_expr(text))


# Number of valence electrons in the pseudopotentials recommended by
# http://materialscloud.org/sssp/. These are just used as a fallback for
# calculating initial magetization values which are given as a fraction
# of valence electrons.
SSSP_VALENCE = [
    0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
    18.0, 19.0, 20.0, 13.0, 14.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 12.0, 13.0, 14.0, 15.0, 6.0,
    7.0, 18.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
    19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 36.0, 27.0, 14.0, 15.0, 30.0,
    15.0, 32.0, 19.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0]


def kspacing_to_grid(atoms, spacing, calculated_spacing=None):
    """
    Calculate the kpoint mesh that is equivalent to the given spacing
    in reciprocal space (units Angstrom^-1). The number of kpoints is each
    dimension is rounded up (compatible with CASTEP).

    Parameters
    ----------
    atoms: ase.Atoms
        A structure that can have get_reciprocal_cell called on it.
    spacing: float
        Minimum K-Point spacing in $A^{-1}$.
    calculated_spacing : list
        If a three item list (or similar mutable sequence) is given the
        members will be replaced with the actual calculated spacing in
        $A^{-1}$.

    Returns
    -------
    kpoint_grid : [int, int, int]
        MP grid specification to give the required spacing.

    """
    # No factor of 2pi in ase, everything in A^-1
    # reciprocal dimensions
    r_x, r_y, r_z = np.linalg.norm(atoms.cell.reciprocal(), axis=1)

    kpoint_grid = [int(r_x / spacing) + 1,
                   int(r_y / spacing) + 1,
                   int(r_z / spacing) + 1]

    for i, _ in enumerate(kpoint_grid):
        if not atoms.pbc[i]:
            kpoint_grid[i] = 1

    if calculated_spacing is not None:
        calculated_spacing[:] = [r_x / kpoint_grid[0],
                                 r_y / kpoint_grid[1],
                                 r_z / kpoint_grid[2]]

    return kpoint_grid


def format_atom_position(atom, crystal_coordinates, mask='', tidx=None):
    """Format one line of atomic positions in
    Quantum ESPRESSO ATOMIC_POSITIONS card.

    >>> for atom in make_supercell(bulk('Li', 'bcc'), np.ones(3)-np.eye(3)):
    >>>     format_atom_position(atom, True)
    Li 0.0000000000 0.0000000000 0.0000000000
    Li 0.5000000000 0.5000000000 0.5000000000

    Parameters
    ----------
    atom : Atom
        A structure that has symbol and [position | (a, b, c)].
    crystal_coordinates: bool
        Whether the atomic positions should be written to the QE input file in
        absolute (False, default) or relative (crystal) coordinates (True).
    mask, optional : str
        String of ndim=3 0 or 1 for constraining atomic positions.
    tidx, optional : int
        Magnetic type index.

    Returns
    -------
    atom_line : str
        Input line for atom position
    """
    if crystal_coordinates:
        coords = [atom.a, atom.b, atom.c]
    else:
        coords = atom.position
    line_fmt = '{atom.symbol}'
    inps = dict(atom=atom)
    if tidx is not None:
        line_fmt += '{tidx}'
        inps["tidx"] = tidx
    line_fmt += ' {coords[0]:.10f} {coords[1]:.10f} {coords[2]:.10f} '
    inps["coords"] = coords
    line_fmt += ' ' + mask + '\n'
    astr = line_fmt.format(**inps)
    return astr


@writer
def write_espresso_in(fd, atoms, input_data=None, pseudopotentials=None,
                      kspacing=None, kpts=None, koffset=(0, 0, 0),
                      crystal_coordinates=False, additional_cards=None,
                      **kwargs):
    """
    Create an input file for pw.x.

    Use set_initial_magnetic_moments to turn on spin, if nspin is set to 2
    with no magnetic moments, they will all be set to 0.0. Magnetic moments
    will be converted to the QE units (fraction of valence electrons) using
    any pseudopotential files found, or a best guess for the number of
    valence electrons.

    Units are not converted for any other input data, so use Quantum ESPRESSO
    units (Usually Ry or atomic units).

    Keys with a dimension (e.g. Hubbard_U(1)) will be incorporated as-is
    so the `i` should be made to match the output.

    Implemented features:

    - Conversion of :class:`ase.constraints.FixAtoms` and
      :class:`ase.constraints.FixCartesian`.
    - ``starting_magnetization`` derived from the ``magmoms`` and
      pseudopotentials (searches default paths for pseudo files.)
    - Automatic assignment of options to their correct sections.

    Not implemented:

    - Non-zero values of ibrav
    - Lists of k-points
    - Other constraints
    - Hubbard parameters
    - Validation of the argument types for input
    - Validation of required options

    Parameters
    ----------
    fd: file | str
        A file to which the input is written.
    atoms: Atoms
        A single atomistic configuration to write to ``fd``.
    input_data: dict
        A flat or nested dictionary with input parameters for pw.x
    pseudopotentials: dict
        A filename for each atomic species, e.g.
        {'O': 'O.pbe-rrkjus.UPF', 'H': 'H.pbe-rrkjus.UPF'}.
        A dummy name will be used if none are given.
    kspacing: float
        Generate a grid of k-points with this as the minimum distance,
        in A^-1 between them in reciprocal space. If set to None, kpts
        will be used instead.
    kpts: (int, int, int), dict or np.ndarray
        If ``kpts`` is a tuple (or list) of 3 integers, it is interpreted
        as the dimensions of a Monkhorst-Pack grid.
        If ``kpts`` is set to ``None``, only the Γ-point will be included
        and QE will use routines optimized for Γ-point-only calculations.
        Compared to Γ-point-only calculations without this optimization
        (i.e. with ``kpts=(1, 1, 1)``), the memory and CPU requirements
        are typically reduced by half.
        If kpts is a dict, it will either be interpreted as a path
        in the Brillouin zone (*) if it contains the 'path' keyword,
        otherwise it is converted to a Monkhorst-Pack grid (**).
        If ``kpts`` is a NumPy array, the raw k-points will be passed to
        Quantum Espresso as given in the array (in crystal coordinates).
        Must be of shape (n_kpts, 4). The fourth column contains the
        k-point weights.
        (*) see ase.dft.kpoints.bandpath
        (**) see ase.calculators.calculator.kpts2sizeandoffsets
    koffset: (int, int, int)
        Offset of kpoints in each direction. Must be 0 (no offset) or
        1 (half grid offset). Setting to True is equivalent to (1, 1, 1).
    crystal_coordinates: bool
        Whether the atomic positions should be written to the QE input file in
        absolute (False, default) or relative (crystal) coordinates (True).

    """

    # Convert to a namelist to make working with parameters much easier
    # Note that the name ``input_data`` is chosen to prevent clash with
    # ``parameters`` in Calculator objects
    input_parameters = Namelist(input_data)
    input_parameters.to_nested('pw', **kwargs)

    # Convert ase constraints to QE constraints
    # Nx3 array of force multipliers matches what QE uses
    # Do this early so it is available when constructing the atoms card
    moved = np.ones((len(atoms), 3), dtype=bool)
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            moved[constraint.index] = False
        elif isinstance(constraint, FixCartesian):
            moved[constraint.index] = ~constraint.mask
        else:
            warnings.warn(f'Ignored unknown constraint {constraint}')
    masks = []
    for atom in atoms:
        # only inclued mask if something is fixed
        if not all(moved[atom.index]):
            mask = ' {:d} {:d} {:d}'.format(*moved[atom.index])
        else:
            mask = ''
        masks.append(mask)

    # Species info holds the information on the pseudopotential and
    # associated for each element
    if pseudopotentials is None:
        pseudopotentials = {}
    species_info = {}
    for species in set(atoms.get_chemical_symbols()):
        # Look in all possible locations for the pseudos and try to figure
        # out the number of valence electrons
        pseudo = pseudopotentials[species]
        species_info[species] = {'pseudo': pseudo}

    # Convert atoms into species.
    # Each different magnetic moment needs to be a separate type even with
    # the same pseudopotential (e.g. an up and a down for AFM).
    # if any magmom are > 0 or nspin == 2 then use species labels.
    # Rememeber: magnetisation uses 1 based indexes
    atomic_species = {}
    atomic_species_str = []
    atomic_positions_str = []

    nspin = input_parameters['system'].get('nspin', 1)  # 1 is the default
    noncolin = input_parameters['system'].get('noncolin', False)
    rescale_magmom_fac = kwargs.get('rescale_magmom_fac', 1.0)
    if any(atoms.get_initial_magnetic_moments()):
        if nspin == 1 and not noncolin:
            # Force spin on
            input_parameters['system']['nspin'] = 2
            nspin = 2

    if nspin == 2 or noncolin:
        # Magnetic calculation on
        for atom, mask, magmom in zip(
                atoms, masks, atoms.get_initial_magnetic_moments()):
            if (atom.symbol, magmom) not in atomic_species:
                # for qe version 7.2 or older magmon must be rescale by
                # about a factor 10 to assume sensible values
                # since qe-v7.3 magmom values will be provided unscaled
                fspin = float(magmom) / rescale_magmom_fac
                # Index in the atomic species list
                sidx = len(atomic_species) + 1
                # Index for that atom type; no index for first one
                tidx = sum(atom.symbol == x[0] for x in atomic_species) or ' '
                atomic_species[(atom.symbol, magmom)] = (sidx, tidx)
                # Add magnetization to the input file
                mag_str = f"starting_magnetization({sidx})"
                input_parameters['system'][mag_str] = fspin
                species_pseudo = species_info[atom.symbol]['pseudo']
                atomic_species_str.append(
                    f"{atom.symbol}{tidx} {atom.mass} {species_pseudo}\n")
            # lookup tidx to append to name
            sidx, tidx = atomic_species[(atom.symbol, magmom)]
            # construct line for atomic positions
            atomic_positions_str.append(
                format_atom_position(
                    atom, crystal_coordinates, mask=mask, tidx=tidx)
            )
    else:
        # Do nothing about magnetisation
        for atom, mask in zip(atoms, masks):
            if atom.symbol not in atomic_species:
                atomic_species[atom.symbol] = True  # just a placeholder
                species_pseudo = species_info[atom.symbol]['pseudo']
                atomic_species_str.append(
                    f"{atom.symbol} {atom.mass} {species_pseudo}\n")
            # construct line for atomic positions
            atomic_positions_str.append(
                format_atom_position(atom, crystal_coordinates, mask=mask)
            )

    # Add computed parameters
    # different magnetisms means different types
    input_parameters['system']['ntyp'] = len(atomic_species)
    input_parameters['system']['nat'] = len(atoms)

    # Use cell as given or fit to a specific ibrav
    if 'ibrav' in input_parameters['system']:
        ibrav = input_parameters['system']['ibrav']
        if ibrav != 0:
            raise ValueError(ibrav_error_message)
    else:
        # Just use standard cell block
        input_parameters['system']['ibrav'] = 0

    # Construct input file into this
    pwi = input_parameters.to_string(list_form=True)

    # Pseudopotentials
    pwi.append('ATOMIC_SPECIES\n')
    pwi.extend(atomic_species_str)
    pwi.append('\n')

    # KPOINTS - add a MP grid as required
    if kspacing is not None:
        kgrid = kspacing_to_grid(atoms, kspacing)
    elif kpts is not None:
        if isinstance(kpts, dict) and 'path' not in kpts:
            kgrid, shift = kpts2sizeandoffsets(atoms=atoms, **kpts)
            koffset = []
            for i, x in enumerate(shift):
                assert x == 0 or abs(x * kgrid[i] - 0.5) < 1e-14
                koffset.append(0 if x == 0 else 1)
        else:
            kgrid = kpts
    else:
        kgrid = "gamma"

    # True and False work here and will get converted by ':d' format
    if isinstance(koffset, int):
        koffset = (koffset, ) * 3

    # BandPath object or bandpath-as-dictionary:
    if isinstance(kgrid, dict) or hasattr(kgrid, 'kpts'):
        pwi.append('K_POINTS crystal_b\n')
        assert hasattr(kgrid, 'path') or 'path' in kgrid
        kgrid = kpts2ndarray(kgrid, atoms=atoms)
        pwi.append(f'{len(kgrid)}\n')
        for k in kgrid:
            pwi.append(f"{k[0]:.14f} {k[1]:.14f} {k[2]:.14f} 0\n")
        pwi.append('\n')
    elif isinstance(kgrid, str) and (kgrid == "gamma"):
        pwi.append('K_POINTS gamma\n')
        pwi.append('\n')
    elif isinstance(kgrid, np.ndarray):
        if np.shape(kgrid)[1] != 4:
            raise ValueError('Only Nx4 kgrids are supported right now.')
        pwi.append('K_POINTS crystal\n')
        pwi.append(f'{len(kgrid)}\n')
        for k in kgrid:
            pwi.append(f"{k[0]:.14f} {k[1]:.14f} {k[2]:.14f} {k[3]:.14f}\n")
        pwi.append('\n')
    else:
        pwi.append('K_POINTS automatic\n')
        pwi.append(f"{kgrid[0]} {kgrid[1]} {kgrid[2]} "
                   f" {koffset[0]:d} {koffset[1]:d} {koffset[2]:d}\n")
        pwi.append('\n')

    # CELL block, if required
    if input_parameters['SYSTEM']['ibrav'] == 0:
        pwi.append('CELL_PARAMETERS angstrom\n')
        pwi.append('{cell[0][0]:.14f} {cell[0][1]:.14f} {cell[0][2]:.14f}\n'
                   '{cell[1][0]:.14f} {cell[1][1]:.14f} {cell[1][2]:.14f}\n'
                   '{cell[2][0]:.14f} {cell[2][1]:.14f} {cell[2][2]:.14f}\n'
                   ''.format(cell=atoms.cell))
        pwi.append('\n')

    # Positions - already constructed, but must appear after namelist
    if crystal_coordinates:
        pwi.append('ATOMIC_POSITIONS crystal\n')
    else:
        pwi.append('ATOMIC_POSITIONS angstrom\n')
    pwi.extend(atomic_positions_str)
    pwi.append('\n')

    # DONE!
    fd.write(''.join(pwi))

    if additional_cards:
        if isinstance(additional_cards, list):
            additional_cards = "\n".join(additional_cards)
            additional_cards += "\n"

        fd.write(additional_cards)


def write_espresso_ph(
        fd,
        input_data=None,
        qpts=None,
        nat_todo_indices=None,
        **kwargs) -> None:
    """
    Function that write the input file for a ph.x calculation. Normal namelist
    cards are passed in the input_data dictionary. Which can be either nested
    or flat, ASE style. The q-points are passed in the qpts list. If qplot is
    set to True then qpts is expected to be a list of list|tuple of length 4.
    Where the first three elements are the coordinates of the q-point in units
    of 2pi/alat and the last element is the weight of the q-point. if qplot is
    set to False then qpts is expected to be a simple list of length 4 (single
    q-point). Finally if ldisp is set to True, the above is discarded and the
    q-points are read from the nq1, nq2, nq3 cards in the input_data dictionary.

    Additionally, a nat_todo_indices kwargs (list[int]) can be specified in the
    kwargs. It will be used if nat_todo is set to True in the input_data
    dictionary.

    Globally, this function follows the convention set in the ph.x documentation
    (https://www.quantum-espresso.org/Doc/INPUT_PH.html)

    Parameters
    ----------
    fd
        The file descriptor of the input file.

    kwargs
        kwargs dictionary possibly containing the following keys:

        - input_data: dict
        - qpts: list[list[float]] | list[tuple[float]] | list[float]
        - nat_todo_indices: list[int]

    Returns
    -------
    None
    """

    input_data = Namelist(input_data)
    input_data.to_nested('ph', **kwargs)

    input_ph = input_data["inputph"]

    inp_nat_todo = input_ph.get("nat_todo", 0)
    qpts = qpts or (0, 0, 0)

    pwi = input_data.to_string()

    fd.write(pwi)

    qplot = input_ph.get("qplot", False)
    ldisp = input_ph.get("ldisp", False)

    if qplot:
        fd.write(f"{len(qpts)}\n")
        for qpt in qpts:
            fd.write(
                f"{qpt[0]:0.8f} {qpt[1]:0.8f} {qpt[2]:0.8f} {qpt[3]:1d}\n"
            )
    elif not (qplot or ldisp):
        fd.write(f"{qpts[0]:0.8f} {qpts[1]:0.8f} {qpts[2]:0.8f}\n")
    if inp_nat_todo:
        tmp = [str(i) for i in nat_todo_indices]
        fd.write(" ".join(tmp))
        fd.write("\n")


def read_espresso_ph(fileobj):
    """
    Function that reads the output of a ph.x calculation.
    It returns a dictionary where each q-point number is a key and
    the value is a dictionary with the following keys if available:

    - qpoints: The q-point in cartesian coordinates.
    - kpoints: The k-points in cartesian coordinates.
    - dieltensor: The dielectric tensor.
    - borneffcharge: The effective Born charges.
    - borneffcharge_dfpt: The effective Born charges from DFPT.
    - polarizability: The polarizability tensor.
    - modes: The phonon modes.
    - eqpoints: The symmetrically equivalent q-points.
    - freqs: The phonon frequencies.
    - mode_symmetries: The symmetries of the modes.
    - atoms: The atoms object.

    Some notes:

        - For some reason, the cell is not defined to high level of
          precision in ph.x outputs. Be careful when using the atoms object
          retrieved from this function.
        - This function can be called on incomplete calculations i.e.
          if the calculation couldn't diagonalize the dynamical matrix
          for some q-points, the results for the other q-points will
          still be returned.

    Parameters
    ----------
    fileobj
        The file descriptor of the output file.

    Returns
    -------
    dict
        The results dictionnary as described above.
    """
    freg = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+\-]?\d+)?")

    QPOINTS = r"(?i)^\s*Calculation\s*of\s*q"
    NKPTS = r"(?i)^\s*number\s*of\s*k\s*points\s*"
    DIEL = r"(?i)^\s*Dielectric\s*constant\s*in\s*cartesian\s*axis\s*$"
    BORN = r"(?i)^\s*Effective\s*charges\s*\(d\s*Force\s*/\s*dE\)"
    POLA = r"(?i)^\s*Polarizability\s*(a.u.)\^3"
    REPR = r"(?i)^\s*There\s*are\s*\d+\s*irreducible\s*representations\s*$"
    EQPOINTS = r"(?i)^\s*Number\s*of\s*q\s*in\s*the\s*star\s*=\s*"
    DIAG = r"(?i)^\s*Diagonalizing\s*the\s*dynamical\s*matrix\s*$"
    MODE_SYM = r"(?i)^\s*Mode\s*symmetry,\s*"
    BORN_DFPT = r"(?i)^\s*Effective\s*charges\s*\(d\s*P\s*/\s*du\)"
    POSITIONS = r"(?i)^\s*site\s*n\..*\(alat\s*units\)"
    ALAT = r"(?i)^\s*celldm\(1\)="
    CELL = (
        r"^\s*crystal\s*axes:\s*\(cart.\s*coord.\s*in\s*units\s*of\s*alat\)"
    )
    ELECTRON_PHONON = r"(?i)^\s*electron-phonon\s*interaction\s*...\s*$"

    output = {
        QPOINTS: [],
        NKPTS: [],
        DIEL: [],
        BORN: [],
        BORN_DFPT: [],
        POLA: [],
        REPR: [],
        EQPOINTS: [],
        DIAG: [],
        MODE_SYM: [],
        POSITIONS: [],
        ALAT: [],
        CELL: [],
        ELECTRON_PHONON: [],
    }

    names = {
        QPOINTS: "qpoints",
        NKPTS: "kpoints",
        DIEL: "dieltensor",
        BORN: "borneffcharge",
        BORN_DFPT: "borneffcharge_dfpt",
        POLA: "polarizability",
        REPR: "representations",
        EQPOINTS: "eqpoints",
        DIAG: "freqs",
        MODE_SYM: "mode_symmetries",
        POSITIONS: "positions",
        ALAT: "alat",
        CELL: "cell",
        ELECTRON_PHONON: "ep_data",
    }

    unique = {
        QPOINTS: True,
        NKPTS: False,
        DIEL: True,
        BORN: True,
        BORN_DFPT: True,
        POLA: True,
        REPR: True,
        EQPOINTS: True,
        DIAG: True,
        MODE_SYM: True,
        POSITIONS: True,
        ALAT: True,
        CELL: True,
        ELECTRON_PHONON: True,
    }

    results = {}
    fdo_lines = [i for i in fileobj.read().splitlines() if i]
    n_lines = len(fdo_lines)

    for idx, line in enumerate(fdo_lines):
        for key in output:
            if bool(re.match(key, line)):
                output[key].append(idx)

    output = {key: np.array(value) for key, value in output.items()}

    def _read_qpoints(idx):
        match = re.findall(freg, fdo_lines[idx])
        return tuple(round(float(x), 7) for x in match)

    def _read_kpoints(idx):
        n_kpts = int(re.findall(freg, fdo_lines[idx])[0])
        kpts = []
        for line in fdo_lines[idx + 2: idx + 2 + n_kpts]:
            if bool(re.search(r"^\s*k\(.*wk", line)):
                kpts.append([round(float(x), 7)
                            for x in re.findall(freg, line)[1:]])
        return np.array(kpts)

    def _read_repr(idx):
        n_repr, curr, n = int(re.findall(freg, fdo_lines[idx])[0]), 0, 0
        representations = {}
        while idx + n < n_lines:
            if re.search(r"^\s*Representation.*modes", fdo_lines[idx + n]):
                curr = int(re.findall(freg, fdo_lines[idx + n])[0])
                representations[curr] = {"done": False, "modes": []}
            if re.search(r"Calculated\s*using\s*symmetry", fdo_lines[idx + n]) \
                    or re.search(r"-\s*Done\s*$", fdo_lines[idx + n]):
                representations[curr]["done"] = True
            if re.search(r"(?i)^\s*(mode\s*#\s*\d\s*)+", fdo_lines[idx + n]):
                representations[curr]["modes"] = _read_modes(idx + n)
                if curr == n_repr:
                    break
            n += 1
        return representations

    def _read_modes(idx):
        n = 1
        n_modes = len(re.findall(r"mode", fdo_lines[idx]))
        modes = []
        while not modes or bool(re.match(r"^\s*\(", fdo_lines[idx + n])):
            tmp = re.findall(freg, fdo_lines[idx + n])
            modes.append([round(float(x), 5) for x in tmp])
            n += 1
        return np.hsplit(np.array(modes), n_modes)

    def _read_eqpoints(idx):
        n_star = int(re.findall(freg, fdo_lines[idx])[0])
        return np.loadtxt(
            fdo_lines[idx + 2: idx + 2 + n_star], usecols=(1, 2, 3)
        ).reshape(-1, 3)

    def _read_freqs(idx):
        n = 0
        freqs = []
        stop = 0
        while not freqs or stop < 2:
            if bool(re.search(r"^\s*freq", fdo_lines[idx + n])):
                tmp = re.findall(freg, fdo_lines[idx + n])[1]
                freqs.append(float(tmp))
            if bool(re.search(r"\*{5,}", fdo_lines[idx + n])):
                stop += 1
            n += 1
        return np.array(freqs)

    def _read_sym(idx):
        n = 1
        sym = {}
        while bool(re.match(r"^\s*freq", fdo_lines[idx + n])):
            r = re.findall("\\d+", fdo_lines[idx + n])
            r = tuple(range(int(r[0]), int(r[1]) + 1))
            sym[r] = fdo_lines[idx + n].split("-->")[1].strip()
            sym[r] = re.sub(r"\s+", " ", sym[r])
            n += 1
        return sym

    def _read_epsil(idx):
        epsil = np.zeros((3, 3))
        for n in range(1, 4):
            tmp = re.findall(freg, fdo_lines[idx + n])
            epsil[n - 1] = [round(float(x), 9) for x in tmp]
        return epsil

    def _read_born(idx):
        n = 1
        born = []
        while idx + n < n_lines:
            if re.search(r"^\s*atom\s*\d\s*\S", fdo_lines[idx + n]):
                pass
            elif re.search(r"^\s*E\*?(x|y|z)\s*\(", fdo_lines[idx + n]):
                tmp = re.findall(freg, fdo_lines[idx + n])
                born.append([round(float(x), 5) for x in tmp])
            else:
                break
            n += 1
        born = np.array(born)
        return np.vsplit(born, len(born) // 3)

    def _read_born_dfpt(idx):
        n = 1
        born = []
        while idx + n < n_lines:
            if re.search(r"^\s*atom\s*\d\s*\S", fdo_lines[idx + n]):
                pass
            elif re.search(r"^\s*P(x|y|z)\s*\(", fdo_lines[idx + n]):
                tmp = re.findall(freg, fdo_lines[idx + n])
                born.append([round(float(x), 5) for x in tmp])
            else:
                break
            n += 1
        born = np.array(born)
        return np.vsplit(born, len(born) // 3)

    def _read_pola(idx):
        pola = np.zeros((3, 3))
        for n in range(1, 4):
            tmp = re.findall(freg, fdo_lines[idx + n])[:3]
            pola[n - 1] = [round(float(x), 2) for x in tmp]
        return pola

    def _read_positions(idx):
        positions = []
        symbols = []
        n = 1
        while re.findall(r"^\s*\d+", fdo_lines[idx + n]):
            symbols.append(fdo_lines[idx + n].split()[1])
            positions.append(
                [round(float(x), 5)
                 for x in re.findall(freg, fdo_lines[idx + n])[-3:]]
            )
            n += 1
        atoms = Atoms(positions=positions, symbols=symbols)
        atoms.pbc = True
        return atoms

    def _read_alat(idx):
        return round(float(re.findall(freg, fdo_lines[idx])[1]), 5)

    def _read_cell(idx):
        cell = []
        n = 1
        while re.findall(r"^\s*a\(\d\)", fdo_lines[idx + n]):
            cell.append([round(float(x), 4)
                         for x in re.findall(freg, fdo_lines[idx + n])[-3:]])
            n += 1
        return np.array(cell)

    def _read_electron_phonon(idx):
        results = {}

        broad_re = (
            r"^\s*Gaussian\s*Broadening:\s+([\d.]+)\s+Ry, ngauss=\s+\d+"
        )
        dos_re = (
            r"^\s*DOS\s*=\s*([\d.]+)\s*states/"
            r"spin/Ry/Unit\s*Cell\s*at\s*Ef=\s+([\d.]+)\s+eV"
        )
        lg_re = (
            r"^\s*lambda\(\s+(\d+)\)=\s+([\d.]+)\s+gamma=\s+([\d.]+)\s+GHz"
        )
        end_re = r"^\s*Number\s*of\s*q\s*in\s*the\s*star\s*=\s+(\d+)$"

        lambdas = []
        gammas = []

        current = None

        n = 1
        while idx + n < n_lines:
            line = fdo_lines[idx + n]

            broad_match = re.match(broad_re, line)
            dos_match = re.match(dos_re, line)
            lg_match = re.match(lg_re, line)
            end_match = re.match(end_re, line)

            if broad_match:
                if lambdas:
                    results[current]["lambdas"] = lambdas
                    results[current]["gammas"] = gammas
                    lambdas = []
                    gammas = []
                current = float(broad_match[1])
                results[current] = {}
            elif dos_match:
                results[current]["dos"] = float(dos_match[1])
                results[current]["fermi"] = float(dos_match[2])
            elif lg_match:
                lambdas.append(float(lg_match[2]))
                gammas.append(float(lg_match[3]))

            if end_match:
                results[current]["lambdas"] = lambdas
                results[current]["gammas"] = gammas
                break

            n += 1

        return results

    properties = {
        NKPTS: _read_kpoints,
        DIEL: _read_epsil,
        BORN: _read_born,
        BORN_DFPT: _read_born_dfpt,
        POLA: _read_pola,
        REPR: _read_repr,
        EQPOINTS: _read_eqpoints,
        DIAG: _read_freqs,
        MODE_SYM: _read_sym,
        POSITIONS: _read_positions,
        ALAT: _read_alat,
        CELL: _read_cell,
        ELECTRON_PHONON: _read_electron_phonon,
    }

    iblocks = np.append(output[QPOINTS], n_lines)

    for qnum, (past, future) in enumerate(zip(iblocks[:-1], iblocks[1:])):
        qpoint = _read_qpoints(past)
        results[qnum + 1] = curr_result = {"qpoint": qpoint}
        for prop in properties:
            p = (past < output[prop]) & (output[prop] < future)
            selected = output[prop][p]
            if len(selected) == 0:
                continue
            if unique[prop]:
                idx = output[prop][p][-1]
                curr_result[names[prop]] = properties[prop](idx)
            else:
                tmp = {k + 1: 0 for k in range(len(selected))}
                for k, idx in enumerate(selected):
                    tmp[k + 1] = properties[prop](idx)
                curr_result[names[prop]] = tmp
        alat = curr_result.pop("alat", 1.0)
        atoms = curr_result.pop("positions", None)
        cell = curr_result.pop("cell", np.eye(3))
        if atoms:
            atoms.positions *= alat * units["Bohr"]
            atoms.cell = cell * alat * units["Bohr"]
            atoms.wrap()
            curr_result["atoms"] = atoms

    return results


def write_fortran_namelist(
        fd,
        input_data=None,
        binary=None,
        additional_cards=None,
        **kwargs) -> None:
    """
    Function which writes input for simple espresso binaries.
    List of supported binaries are in the espresso_keys.py file.
    Non-exhaustive list (to complete)

    Note: "EOF" is appended at the end of the file.
    (https://lists.quantum-espresso.org/pipermail/users/2020-November/046269.html)

    Additional fields are written 'as is' in the input file. It is expected
    to be a string or a list of strings.

    Parameters
    ----------
    fd
        The file descriptor of the input file.
    input_data: dict
        A flat or nested dictionary with input parameters for the binary.x
    binary: str
        Name of the binary
    additional_cards: str | list[str]
        Additional fields to be written at the end of the input file, after
        the namelist. It is expected to be a string or a list of strings.

    Returns
    -------
    None
    """
    input_data = Namelist(input_data)

    if binary:
        input_data.to_nested(binary, **kwargs)

    pwi = input_data.to_string()

    fd.write(pwi)

    if additional_cards:
        if isinstance(additional_cards, list):
            additional_cards = "\n".join(additional_cards)
            additional_cards += "\n"

        fd.write(additional_cards)

    fd.write("EOF")


@deprecated('Please use the ase.io.espresso.Namelist class',
            DeprecationWarning)
def construct_namelist(parameters=None, keys=None, warn=False, **kwargs):
    """
    Construct an ordered Namelist containing all the parameters given (as
    a dictionary or kwargs). Keys will be inserted into their appropriate
    section in the namelist and the dictionary may contain flat and nested
    structures. Any kwargs that match input keys will be incorporated into
    their correct section. All matches are case-insensitive, and returned
    Namelist object is a case-insensitive dict.

    If a key is not known to ase, but in a section within `parameters`,
    it will be assumed that it was put there on purpose and included
    in the output namelist. Anything not in a section will be ignored (set
    `warn` to True to see ignored keys).

    Keys with a dimension (e.g. Hubbard_U(1)) will be incorporated as-is
    so the `i` should be made to match the output.

    The priority of the keys is:
        kwargs[key] > parameters[key] > parameters[section][key]
    Only the highest priority item will be included.

    .. deprecated:: 3.23.0
        Please use :class:`ase.io.espresso.Namelist` instead.

    Parameters
    ----------
    parameters: dict
        Flat or nested set of input parameters.
    keys: Namelist | dict
        Namelist to use as a template for the output.
    warn: bool
        Enable warnings for unused keys.

    Returns
    -------
    input_namelist: Namelist
        pw.x compatible namelist of input parameters.

    """

    if keys is None:
        keys = deepcopy(pw_keys)
    # Convert everything to Namelist early to make case-insensitive
    if parameters is None:
        parameters = Namelist()
    else:
        # Maximum one level of nested dict
        # Don't modify in place
        parameters_namelist = Namelist()
        for key, value in parameters.items():
            if isinstance(value, dict):
                parameters_namelist[key] = Namelist(value)
            else:
                parameters_namelist[key] = value
        parameters = parameters_namelist

    # Just a dict
    kwargs = Namelist(kwargs)

    # Final parameter set
    input_namelist = Namelist()

    # Collect
    for section in keys:
        sec_list = Namelist()
        for key in keys[section]:
            # Check all three separately and pop them all so that
            # we can check for missing values later
            value = None

            if key in parameters.get(section, {}):
                value = parameters[section].pop(key)
            if key in parameters:
                value = parameters.pop(key)
            if key in kwargs:
                value = kwargs.pop(key)

            if value is not None:
                sec_list[key] = value

            # Check if there is a key(i) version (no extra parsing)
            for arg_key in list(parameters.get(section, {})):
                if arg_key.split('(')[0].strip().lower() == key.lower():
                    sec_list[arg_key] = parameters[section].pop(arg_key)
            cp_parameters = parameters.copy()
            for arg_key in cp_parameters:
                if arg_key.split('(')[0].strip().lower() == key.lower():
                    sec_list[arg_key] = parameters.pop(arg_key)
            cp_kwargs = kwargs.copy()
            for arg_key in cp_kwargs:
                if arg_key.split('(')[0].strip().lower() == key.lower():
                    sec_list[arg_key] = kwargs.pop(arg_key)

        # Add to output
        input_namelist[section] = sec_list

    unused_keys = list(kwargs)
    # pass anything else already in a section
    for key, value in parameters.items():
        if key in keys and isinstance(value, dict):
            input_namelist[key].update(value)
        elif isinstance(value, dict):
            unused_keys.extend(list(value))
        else:
            unused_keys.append(key)

    if warn and unused_keys:
        warnings.warn('Unused keys: {}'.format(', '.join(unused_keys)))

    return input_namelist


@deprecated('Please use the .to_string() method of Namelist instead.',
            DeprecationWarning)
def namelist_to_string(input_parameters):
    """Format a Namelist object as a string for writing to a file.
    Assume sections are ordered (taken care of in namelist construction)
    and that repr converts to a QE readable representation (except bools)

    .. deprecated:: 3.23.0
        Please use the :meth:`ase.io.espresso.Namelist.to_string` method
        instead.

    Parameters
    ----------
    input_parameters : Namelist | dict
        Expecting a nested dictionary of sections and key-value data.

    Returns
    -------
    pwi : List[str]
        Input line for the namelist
    """
    pwi = []
    for section in input_parameters:
        pwi.append(f'&{section.upper()}\n')
        for key, value in input_parameters[section].items():
            if value is True:
                pwi.append(f'   {key:16} = .true.\n')
            elif value is False:
                pwi.append(f'   {key:16} = .false.\n')
            elif isinstance(value, Path):
                pwi.append(f'   {key:16} = "{value}"\n')
            else:
                # repr format to get quotes around strings
                pwi.append(f'   {key:16} = {value!r}\n')
        pwi.append('/\n')  # terminate section
    pwi.append('\n')
    return pwi
