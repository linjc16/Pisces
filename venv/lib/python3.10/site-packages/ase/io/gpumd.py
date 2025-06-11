# fmt: off

import numpy as np

from ase import Atoms
from ase.data import atomic_masses, chemical_symbols
from ase.neighborlist import NeighborList


def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def find_nearest_value(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def write_gpumd(fd, atoms, maximum_neighbors=None, cutoff=None,
                groupings=None, use_triclinic=False, species=None):
    """
    Writes atoms into GPUMD input format.

    Parameters
    ----------
    fd : file
        File like object to which the atoms object should be written
    atoms : Atoms
        Input structure
    maximum_neighbors: int
        Maximum number of neighbors any atom can ever have (not relevant when
        using force constant potentials)
    cutoff: float
        Initial cutoff distance used for building the neighbor list (not
        relevant when using force constant potentials)
    groupings : list[list[list[int]]]
        Groups into which the individual atoms should be divided in the form of
        a list of list of lists. Specifically, the outer list corresponds to
        the grouping methods, of which there can be three at the most, which
        contains a list of groups in the form of lists of site indices. The
        sum of the lengths of the latter must be the same as the total number
        of atoms.
    use_triclinic: bool
        Use format for triclinic cells
    species : List[str]
        GPUMD uses integers to define atom types. This list allows customized
        such definitions (e.g, ['Pd', 'H'] means Pd is type 0 and H type 1).
        If None, this list is built by assigning each distinct species to
        an integer in the order of appearance in `atoms`.

    Raises
    ------
    ValueError
        Raised if parameters are incompatible
    """

    # Check velocties parameter
    if atoms.get_velocities() is None:
        has_velocity = 0
    else:
        has_velocity = 1
        velocities = atoms.get_velocities()

    # Check groupings parameter
    if groupings is None:
        number_of_grouping_methods = 0
    else:
        number_of_grouping_methods = len(groupings)
        if number_of_grouping_methods > 3:
            raise ValueError('There can be no more than 3 grouping methods!')
        for g, grouping in enumerate(groupings):
            all_indices = [i for group in grouping for i in group]
            if len(all_indices) != len(atoms) or\
                    set(all_indices) != set(range(len(atoms))):
                raise ValueError('The indices listed in grouping method {} are'
                                 ' not compatible with the input'
                                 ' structure!'.format(g))

    # If not specified, estimate the maximum_neighbors
    if maximum_neighbors is None:
        if cutoff is None:
            cutoff = 0.1
            maximum_neighbors = 1
        else:
            nl = NeighborList([cutoff / 2] * len(atoms), skin=2, bothways=True)
            nl.update(atoms)
            maximum_neighbors = 0
            for atom in atoms:
                maximum_neighbors = max(maximum_neighbors,
                                        len(nl.get_neighbors(atom.index)[0]))
            maximum_neighbors *= 2
            maximum_neighbors = min(maximum_neighbors, 1024)

    # Add header and cell parameters
    lines = []
    if atoms.cell.orthorhombic and not use_triclinic:
        triclinic = 0
    else:
        triclinic = 1
    lines.append('{} {} {} {} {} {}'.format(len(atoms), maximum_neighbors,
                                            cutoff, triclinic, has_velocity,
                                            number_of_grouping_methods))
    if triclinic:
        lines.append((' {}' * 12)[1:].format(*atoms.pbc.astype(int),
                                             *atoms.cell[:].flatten()))
    else:
        lines.append((' {}' * 6)[1:].format(*atoms.pbc.astype(int),
                                            *atoms.cell.lengths()))

    # Create symbols-to-type map, i.e. integers starting at 0
    if not species:
        symbol_type_map = {}
        for symbol in atoms.get_chemical_symbols():
            if symbol not in symbol_type_map:
                symbol_type_map[symbol] = len(symbol_type_map)
    else:
        if any(sym not in species
               for sym in set(atoms.get_chemical_symbols())):
            raise ValueError('The species list does not contain all chemical '
                             'species that are present in the atoms object.')
        else:
            symbol_type_map = {symbol: i for i, symbol in enumerate(species)}

    # Add lines for all atoms
    for a, atm in enumerate(atoms):
        t = symbol_type_map[atm.symbol]
        line = (' {}' * 5)[1:].format(t, *atm.position, atm.mass)
        if has_velocity:
            line += (' {}' * 3).format(*velocities[a])
        if groupings is not None:
            for grouping in groupings:
                for i, group in enumerate(grouping):
                    if a in group:
                        line += f' {i}'
                        break
        lines.append(line)

    # Write file
    fd.write('\n'.join(lines))


def load_xyz_input_gpumd(fd, species=None, isotope_masses=None):
    """
    Read the structure input file for GPUMD and return an ase Atoms object
    togehter with a dictionary with parameters and a types-to-symbols map

    Parameters
    ----------
    fd : file | str
        File object or name of file from which to read the Atoms object
    species : List[str]
        List with the chemical symbols that correspond to each type, will take
        precedence over isotope_masses
    isotope_masses: Dict[str, List[float]]
        Dictionary with chemical symbols and lists of the associated atomic
        masses, which is used to identify the chemical symbols that correspond
        to the types not found in species_types. The default is to find the
        closest match :data:`ase.data.atomic_masses`.

    Returns
    -------
    atoms : Atoms
        Atoms object
    input_parameters : Dict[str, int]
        Dictionary with parameters from the first row of the input file, namely
        'N', 'M', 'cutoff', 'triclinic', 'has_velocity' and 'num_of_groups'
    species : List[str]
        List with the chemical symbols that correspond to each type

    Raises
    ------
    ValueError
        Raised if the list of species is incompatible with the input file
    """
    # Parse first line
    first_line = next(fd)
    input_parameters = {}
    keys = ['N', 'M', 'cutoff', 'triclinic', 'has_velocity',
            'num_of_groups']
    types = [float if key == 'cutoff' else int for key in keys]
    for k, (key, typ) in enumerate(zip(keys, types)):
        input_parameters[key] = typ(first_line.split()[k])

    # Parse second line
    second_line = next(fd)
    second_arr = np.array(second_line.split())
    pbc = second_arr[:3].astype(bool)
    if input_parameters['triclinic']:
        cell = second_arr[3:].astype(float).reshape((3, 3))
    else:
        cell = np.diag(second_arr[3:].astype(float))

    # Parse all remaining rows
    n_rows = input_parameters['N']
    n_columns = 5 + input_parameters['has_velocity'] * 3 +\
        input_parameters['num_of_groups']
    rest_lines = [next(fd) for _ in range(n_rows)]
    rest_arr = np.array([line.split() for line in rest_lines])
    assert rest_arr.shape == (n_rows, n_columns)

    # Extract atom types, positions and masses
    atom_types = rest_arr[:, 0].astype(int)
    positions = rest_arr[:, 1:4].astype(float)
    masses = rest_arr[:, 4].astype(float)

    # Determine the atomic species
    if species is None:
        type_symbol_map = {}
    if isotope_masses is not None:
        mass_symbols = {mass: symbol for symbol, masses in
                        isotope_masses.items() for mass in masses}
    symbols = []
    for atom_type, mass in zip(atom_types, masses):
        if species is None:
            if atom_type not in type_symbol_map:
                if isotope_masses is not None:
                    nearest_value = find_nearest_value(
                        list(mass_symbols.keys()), mass)
                    symbol = mass_symbols[nearest_value]
                else:
                    symbol = chemical_symbols[
                        find_nearest_index(atomic_masses, mass)]
                type_symbol_map[atom_type] = symbol
            else:
                symbol = type_symbol_map[atom_type]
        else:
            if atom_type > len(species):
                raise Exception('There is no entry for atom type {} in the '
                                'species list!'.format(atom_type))
            symbol = species[atom_type]
        symbols.append(symbol)

    if species is None:
        species = [type_symbol_map[i] for i in sorted(type_symbol_map.keys())]

    # Create the Atoms object
    atoms = Atoms(symbols=symbols, positions=positions, masses=masses, pbc=pbc,
                  cell=cell)
    if input_parameters['has_velocity']:
        velocities = rest_arr[:, 5:8].astype(float)
        atoms.set_velocities(velocities)
    if input_parameters['num_of_groups']:
        start_col = 5 + 3 * input_parameters['has_velocity']
        groups = rest_arr[:, start_col:].astype(int)
        atoms.info = {i: {'groups': groups[i, :]} for i in range(n_rows)}

    return atoms, input_parameters, species


def read_gpumd(fd, species=None, isotope_masses=None):
    """
    Read Atoms object from a GPUMD structure input file

    Parameters
    ----------
    fd : file | str
        File object or name of file from which to read the Atoms object
    species : List[str]
        List with the chemical symbols that correspond to each type, will take
        precedence over isotope_masses
    isotope_masses: Dict[str, List[float]]
        Dictionary with chemical symbols and lists of the associated atomic
        masses, which is used to identify the chemical symbols that correspond
        to the types not found in species_types. The default is to find the
        closest match :data:`ase.data.atomic_masses`.

    Returns
    -------
    atoms : Atoms
        Atoms object

    Raises
    ------
    ValueError
        Raised if the list of species is incompatible with the input file
    """

    return load_xyz_input_gpumd(fd, species, isotope_masses)[0]
