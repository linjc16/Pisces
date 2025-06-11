# fmt: off

"""This module defines I/O routines with CASTEP files.
The key idea is that all function accept or return  atoms objects.
CASTEP specific parameters will be returned through the <atoms>.calc
attribute.
"""
import os
import re
import warnings
from copy import deepcopy
from typing import List, Tuple

import numpy as np

import ase

# independent unit management included here:
# When high accuracy is required, this allows to easily pin down
# unit conversion factors from different "unit definition systems"
# (CODATA1986 for ase-3.6.0.2515 vs CODATA2002 for CASTEP 5.01).
#
# ase.units in in ase-3.6.0.2515 is based on CODATA1986
import ase.units
from ase.constraints import FixAtoms, FixCartesian, FixedLine, FixedPlane
from ase.geometry.cell import cellpar_to_cell
from ase.io.castep.castep_reader import read_castep_castep
from ase.parallel import paropen
from ase.spacegroup import Spacegroup
from ase.utils import atoms_to_spglib_cell, reader, writer

from .geom_md_ts import (
    read_castep_geom,
    read_castep_md,
    write_castep_geom,
    write_castep_md,
)

units_ase = {
    'hbar': ase.units._hbar * ase.units.J,
    'Eh': ase.units.Hartree,
    'kB': ase.units.kB,
    'a0': ase.units.Bohr,
    't0': ase.units._hbar * ase.units.J / ase.units.Hartree,
    'c': ase.units._c,
    'me': ase.units._me / ase.units._amu,
    'Pascal': 1.0 / ase.units.Pascal}

# CODATA1986 (included herein for the sake of completeness)
# taken from
#    http://physics.nist.gov/cuu/Archive/1986RMP.pdf
units_CODATA1986 = {
    'hbar': 6.5821220E-16,      # eVs
    'Eh': 27.2113961,           # eV
    'kB': 8.617385E-5,          # eV/K
    'a0': 0.529177249,          # A
    'c': 299792458,             # m/s
    'e': 1.60217733E-19,        # C
    'me': 5.485799110E-4}       # u

# CODATA2002: default in CASTEP 5.01
# (-> check in more recent CASTEP in case of numerical discrepancies?!)
# taken from
#    http://physics.nist.gov/cuu/Document/all_2002.pdf
units_CODATA2002 = {
    'hbar': 6.58211915E-16,     # eVs
    'Eh': 27.2113845,           # eV
    'kB': 8.617343E-5,          # eV/K
    'a0': 0.5291772108,         # A
    'c': 299792458,             # m/s
    'e': 1.60217653E-19,        # C
    'me': 5.4857990945E-4}      # u

# (common) derived entries
for d in (units_CODATA1986, units_CODATA2002):
    d['t0'] = d['hbar'] / d['Eh']     # s
    d['Pascal'] = d['e'] * 1E30       # Pa


__all__ = [
    # routines for the generic io function
    'read_castep_castep',
    'read_castep_cell',
    'read_castep_geom',
    'read_castep_md',
    'read_phonon',
    'read_castep_phonon',
    # additional reads that still need to be wrapped
    'read_param',
    'read_seed',
    # write that is already wrapped
    'write_castep_cell',
    'write_castep_geom',
    'write_castep_md',
    # param write - in principle only necessary in junction with the calculator
    'write_param']


def write_freeform(fd, outputobj):
    """
    Prints out to a given file a CastepInputFile or derived class, such as
    CastepCell or CastepParam.
    """

    options = outputobj._options

    # Some keywords, if present, are printed in this order
    preferred_order = ['lattice_cart', 'lattice_abc',
                       'positions_frac', 'positions_abs',
                       'species_pot', 'symmetry_ops',   # CELL file
                       'task', 'cut_off_energy'         # PARAM file
                       ]

    keys = outputobj.get_attr_dict().keys()
    # This sorts only the ones in preferred_order and leaves the rest
    # untouched
    keys = sorted(keys, key=lambda x: preferred_order.index(x)
                  if x in preferred_order
                  else len(preferred_order))

    for kw in keys:
        opt = options[kw]
        if opt.type.lower() == 'block':
            fd.write('%BLOCK {0}\n{1}\n%ENDBLOCK {0}\n\n'.format(
                     kw.upper(),
                     opt.value.strip('\n')))
        else:
            fd.write(f'{kw.upper()}: {opt.value}\n')


@writer
def write_castep_cell(fd, atoms, positions_frac=False, force_write=False,
                      precision=6, magnetic_moments=None,
                      castep_cell=None):
    """
    This CASTEP export function write minimal information to
    a .cell file. If the atoms object is a trajectory, it will
    take the last image.

    Note that function has been altered in order to require a filedescriptor
    rather than a filename. This allows to use the more generic write()
    function from formats.py

    Note that the "force_write" keywords has no effect currently.

    Arguments:

        positions_frac: boolean. If true, positions are printed as fractional
                        rather than absolute. Default is false.
        castep_cell: if provided, overrides the existing CastepCell object in
                     the Atoms calculator
        precision: number of digits to which lattice and positions are printed
        magnetic_moments: if None, no SPIN values are initialised.
                          If 'initial', the values from
                          get_initial_magnetic_moments() are used.
                          If 'calculated', the values from
                          get_magnetic_moments() are used.
                          If an array of the same length as the atoms object,
                          its contents will be used as magnetic moments.
    """

    if isinstance(atoms, list):
        if len(atoms) > 1:
            atoms = atoms[-1]

    # Header
    fd.write('# written by ASE\n\n')

    # To write this we simply use the existing Castep calculator, or create
    # one
    from ase.calculators.castep import Castep, CastepCell

    try:
        has_cell = isinstance(atoms.calc.cell, CastepCell)
    except AttributeError:
        has_cell = False

    if has_cell:
        cell = deepcopy(atoms.calc.cell)
    else:
        cell = Castep(keyword_tolerance=2).cell

    # Write lattice
    fformat = f'%{precision + 3}.{precision}f'
    cell_block_format = ' '.join([fformat] * 3)
    cell.lattice_cart = [cell_block_format % tuple(line)
                         for line in atoms.get_cell()]

    if positions_frac:
        pos_keyword = 'positions_frac'
        positions = atoms.get_scaled_positions()
    else:
        pos_keyword = 'positions_abs'
        positions = atoms.get_positions()

    if atoms.has('castep_custom_species'):
        elems = atoms.get_array('castep_custom_species')
    else:
        elems = atoms.get_chemical_symbols()
    if atoms.has('masses'):

        from ase.data import atomic_masses
        masses = atoms.get_array('masses')
        custom_masses = {}

        for i, species in enumerate(elems):
            custom_mass = masses[i]

            # build record of different masses for each species
            if species not in custom_masses:

                # build dictionary of positions of all species with
                # same name and mass value ideally there should only
                # be one mass per species
                custom_masses[species] = {custom_mass: [i]}

            # if multiple masses found for a species
            elif custom_mass not in custom_masses[species].keys():

                # if custom species were already manually defined raise an error
                if atoms.has('castep_custom_species'):
                    raise ValueError(
                        "Could not write custom mass block for {0}. \n"
                        "Custom mass was set ({1}), but an inconsistent set of "
                        "castep_custom_species already defines "
                        "({2}) for {0}. \n"
                        "If using both features, ensure that "
                        "each species type in "
                        "atoms.arrays['castep_custom_species'] "
                        "has consistent mass values and that each atom "
                        "with non-standard "
                        "mass belongs to a custom species type."
                        "".format(
                            species, custom_mass, list(
                                custom_masses[species].keys())[0]))

                # append mass to create custom species later
                else:
                    custom_masses[species][custom_mass] = [i]
            else:
                custom_masses[species][custom_mass].append(i)

        # create species_mass block
        mass_block = []

        for el, mass_dict in custom_masses.items():

            # ignore mass record that match defaults
            default = mass_dict.pop(atomic_masses[atoms.get_array(
                'numbers')[list(elems).index(el)]], None)
            if mass_dict:
                # no custom species need to be created
                if len(mass_dict) == 1 and not default:
                    mass_block.append('{} {}'.format(
                        el, list(mass_dict.keys())[0]))
                # for each custom mass, create new species and change names to
                # match in 'elems' list
                else:
                    warnings.warn(
                        'Custom mass specified for '
                        'standard species {}, creating custom species'
                        .format(el))

                    for i, vals in enumerate(mass_dict.items()):
                        mass_val, idxs = vals
                        custom_species_name = f"{el}:{i}"
                        warnings.warn(
                            'Creating custom species {} with mass {}'.format(
                                custom_species_name, str(mass_dict)))
                        for idx in idxs:
                            elems[idx] = custom_species_name
                        mass_block.append('{} {}'.format(
                            custom_species_name, mass_val))

        cell.species_mass = mass_block

    if atoms.has('castep_labels'):
        labels = atoms.get_array('castep_labels')
    else:
        labels = ['NULL'] * len(elems)

    if str(magnetic_moments).lower() == 'initial':
        magmoms = atoms.get_initial_magnetic_moments()
    elif str(magnetic_moments).lower() == 'calculated':
        magmoms = atoms.get_magnetic_moments()
    elif np.array(magnetic_moments).shape == (len(elems),):
        magmoms = np.array(magnetic_moments)
    else:
        magmoms = [0] * len(elems)

    pos_block = []
    pos_block_format = '%s ' + cell_block_format

    for i, el in enumerate(elems):
        xyz = positions[i]
        line = pos_block_format % tuple([el] + list(xyz))
        # ADD other keywords if necessary
        if magmoms[i] != 0:
            line += f' SPIN={magmoms[i]} '
        if labels[i].strip() not in ('NULL', ''):
            line += f' LABEL={labels[i]} '
        pos_block.append(line)

    setattr(cell, pos_keyword, pos_block)

    constr_block = _make_block_ionic_constraints(atoms)
    if constr_block:
        cell.ionic_constraints = constr_block

    write_freeform(fd, cell)


def _make_block_ionic_constraints(atoms: ase.Atoms) -> List[str]:
    constr_block: List[str] = []
    species_indices = atoms.symbols.species_indices()
    for constr in atoms.constraints:
        if not _is_constraint_valid(constr, len(atoms)):
            continue
        for i in constr.index:
            symbol = atoms.get_chemical_symbols()[i]
            nis = species_indices[i] + 1
            if isinstance(constr, FixAtoms):
                for j in range(3):  # constraint for all three directions
                    ic = len(constr_block) + 1
                    line = f'{ic:6d} {symbol:3s} {nis:3d}   '
                    line += ['1 0 0', '0 1 0', '0 0 1'][j]
                    constr_block.append(line)
            elif isinstance(constr, FixCartesian):
                for j, m in enumerate(constr.mask):
                    if m == 0:  # not constrained
                        continue
                    ic = len(constr_block) + 1
                    line = f'{ic:6d} {symbol:3s} {nis:3d}   '
                    line += ['1 0 0', '0 1 0', '0 0 1'][j]
                    constr_block.append(line)
            elif isinstance(constr, FixedPlane):
                ic = len(constr_block) + 1
                line = f'{ic:6d} {symbol:3s} {nis:3d}   '
                line += ' '.join([str(d) for d in constr.dir])
                constr_block.append(line)
            elif isinstance(constr, FixedLine):
                for direction in _calc_normal_vectors(constr):
                    ic = len(constr_block) + 1
                    line = f'{ic:6d} {symbol:3s} {nis:3d}   '
                    line += ' '.join(str(_) for _ in direction)
                    constr_block.append(line)
    return constr_block


def _is_constraint_valid(constraint, natoms: int) -> bool:
    supported_constraints = (FixAtoms, FixedPlane, FixedLine, FixCartesian)
    if not isinstance(constraint, supported_constraints):
        warnings.warn(f'{constraint} is not supported by ASE CASTEP, skipped')
        return False
    if any(i < 0 or i >= natoms for i in constraint.index):
        warnings.warn(f'{constraint} contains invalid indices, skipped')
        return False
    return True


def _calc_normal_vectors(constr: FixedLine) -> Tuple[np.ndarray, np.ndarray]:
    direction = constr.dir

    i2, i1 = np.argsort(np.abs(direction))[1:]
    v1 = direction[i1]
    v2 = direction[i2]
    n1 = np.zeros(3)
    n1[i2] = v1
    n1[i1] = -v2
    n1 = n1 / np.linalg.norm(n1)

    n2 = np.cross(direction, n1)
    n2 = n2 / np.linalg.norm(n2)

    return n1, n2


def read_freeform(fd):
    """
    Read a CASTEP freeform file (the basic format of .cell and .param files)
    and return keyword-value pairs as a dict (values are strings for single
    keywords and lists of strings for blocks).
    """

    from ase.io.castep.castep_input_file import CastepInputFile

    inputobj = CastepInputFile(keyword_tolerance=2)

    filelines = fd.readlines()

    keyw = None
    read_block = False
    block_lines = None

    for i, l in enumerate(filelines):

        # Strip all comments, aka anything after a hash
        L = re.split(r'[#!;]', l, maxsplit=1)[0].strip()

        if L == '':
            # Empty line... skip
            continue

        lsplit = re.split(r'\s*[:=]*\s+', L, maxsplit=1)

        if read_block:
            if lsplit[0].lower() == '%endblock':
                if len(lsplit) == 1 or lsplit[1].lower() != keyw:
                    raise ValueError('Out of place end of block at '
                                     'line %i in freeform file' % i + 1)
                else:
                    read_block = False
                    inputobj.__setattr__(keyw, block_lines)
            else:
                block_lines += [L]
        else:
            # Check the first word

            # Is it a block?
            read_block = (lsplit[0].lower() == '%block')
            if read_block:
                if len(lsplit) == 1:
                    raise ValueError(('Unrecognizable block at line %i '
                                      'in io freeform file') % i + 1)
                else:
                    keyw = lsplit[1].lower()
            else:
                keyw = lsplit[0].lower()

            # Now save the value
            if read_block:
                block_lines = []
            else:
                inputobj.__setattr__(keyw, ' '.join(lsplit[1:]))

    return inputobj.get_attr_dict(types=True)


@reader
def read_castep_cell(fd, index=None, calculator_args={}, find_spg=False,
                     units=units_CODATA2002):
    """Read a .cell file and return an atoms object.
    Any value found that does not fit the atoms API
    will be stored in the atoms.calc attribute.

    By default, the Castep calculator will be tolerant and in the absence of a
    castep_keywords.json file it will just accept all keywords that aren't
    automatically parsed.
    """

    from ase.calculators.castep import Castep

    cell_units = {  # Units specifiers for CASTEP
        'bohr': units_CODATA2002['a0'],
        'ang': 1.0,
        'm': 1e10,
        'cm': 1e8,
        'nm': 10,
        'pm': 1e-2
    }

    calc = Castep(**calculator_args)

    if calc.cell.castep_version == 0 and calc._kw_tol < 3:
        # No valid castep_keywords.json was found
        warnings.warn(
            'read_cell: Warning - Was not able to validate CASTEP input. '
            'This may be due to a non-existing '
            '"castep_keywords.json" '
            'file or a non-existing CASTEP installation. '
            'Parsing will go on but keywords will not be '
            'validated and may cause problems if incorrect during a CASTEP '
            'run.')

    celldict = read_freeform(fd)

    def parse_blockunit(line_tokens, blockname):
        u = 1.0
        if len(line_tokens[0]) == 1:
            usymb = line_tokens[0][0].lower()
            u = cell_units.get(usymb, 1)
            if usymb not in cell_units:
                warnings.warn('read_cell: Warning - ignoring invalid '
                              'unit specifier in %BLOCK {} '
                              '(assuming Angstrom instead)'.format(blockname))
            line_tokens = line_tokens[1:]
        return u, line_tokens

    # Arguments to pass to the Atoms object at the end
    aargs = {
        'pbc': True
    }

    # Start by looking for the lattice
    lat_keywords = [w in celldict for w in ('lattice_cart', 'lattice_abc')]
    if all(lat_keywords):
        warnings.warn('read_cell: Warning - two lattice blocks present in the'
                      ' same file. LATTICE_ABC will be ignored')
    elif not any(lat_keywords):
        raise ValueError('Cell file must contain at least one between '
                         'LATTICE_ABC and LATTICE_CART')

    if 'lattice_abc' in celldict:

        lines = celldict.pop('lattice_abc')[0].split('\n')
        line_tokens = [line.split() for line in lines]

        u, line_tokens = parse_blockunit(line_tokens, 'lattice_abc')

        if len(line_tokens) != 2:
            warnings.warn('read_cell: Warning - ignoring additional '
                          'lines in invalid %BLOCK LATTICE_ABC')

        abc = [float(p) * u for p in line_tokens[0][:3]]
        angles = [float(phi) for phi in line_tokens[1][:3]]

        aargs['cell'] = cellpar_to_cell(abc + angles)

    if 'lattice_cart' in celldict:

        lines = celldict.pop('lattice_cart')[0].split('\n')
        line_tokens = [line.split() for line in lines]

        u, line_tokens = parse_blockunit(line_tokens, 'lattice_cart')

        if len(line_tokens) != 3:
            warnings.warn('read_cell: Warning - ignoring more than '
                          'three lattice vectors in invalid %BLOCK '
                          'LATTICE_CART')

        aargs['cell'] = [[float(x) * u for x in lt[:3]] for lt in line_tokens]

    # Now move on to the positions
    pos_keywords = [w in celldict
                    for w in ('positions_abs', 'positions_frac')]

    if all(pos_keywords):
        warnings.warn('read_cell: Warning - two lattice blocks present in the'
                      ' same file. POSITIONS_FRAC will be ignored')
        del celldict['positions_frac']
    elif not any(pos_keywords):
        raise ValueError('Cell file must contain at least one between '
                         'POSITIONS_FRAC and POSITIONS_ABS')

    aargs['symbols'] = []
    pos_type = 'positions'
    pos_block = celldict.pop('positions_abs', [None])[0]
    if pos_block is None:
        pos_type = 'scaled_positions'
        pos_block = celldict.pop('positions_frac', [None])[0]
    aargs[pos_type] = []

    lines = pos_block.split('\n')
    line_tokens = [line.split() for line in lines]

    if 'scaled' not in pos_type:
        u, line_tokens = parse_blockunit(line_tokens, 'positions_abs')
    else:
        u = 1.0

    # Here we extract all the possible additional info
    # These are marked by their type

    add_info = {
        'SPIN': (float, 0.0),   # (type, default)
        'MAGMOM': (float, 0.0),
        'LABEL': (str, 'NULL')
    }
    add_info_arrays = {k: [] for k in add_info}

    def parse_info(raw_info):

        re_keys = (r'({0})\s*[=:\s]{{1}}\s'
                   r'*([^\s]*)').format('|'.join(add_info.keys()))
        # Capture all info groups
        info = re.findall(re_keys, raw_info)
        info = {g[0]: add_info[g[0]][0](g[1]) for g in info}
        return info

    # Array for custom species (a CASTEP special thing)
    # Usually left unused
    custom_species = None

    for tokens in line_tokens:
        # Now, process the whole 'species' thing
        spec_custom = tokens[0].split(':', 1)
        elem = spec_custom[0]
        if len(spec_custom) > 1 and custom_species is None:
            # Add it to the custom info!
            custom_species = list(aargs['symbols'])
        if custom_species is not None:
            custom_species.append(tokens[0])
        aargs['symbols'].append(elem)
        aargs[pos_type].append([float(p) * u for p in tokens[1:4]])
        # Now for the additional information
        info = ' '.join(tokens[4:])
        info = parse_info(info)
        for k in add_info:
            add_info_arrays[k] += [info.get(k, add_info[k][1])]

    # read in custom species mass
    if 'species_mass' in celldict:
        spec_list = custom_species if custom_species else aargs['symbols']
        aargs['masses'] = [None for _ in spec_list]
        lines = celldict.pop('species_mass')[0].split('\n')
        line_tokens = [line.split() for line in lines]

        if len(line_tokens[0]) == 1:
            if line_tokens[0][0].lower() not in ('amu', 'u'):
                raise ValueError(
                    "unit specifier '{}' in %BLOCK SPECIES_MASS "
                    "not recognised".format(
                        line_tokens[0][0].lower()))
            line_tokens = line_tokens[1:]

        for tokens in line_tokens:
            token_pos_list = [i for i, x in enumerate(
                spec_list) if x == tokens[0]]
            if len(token_pos_list) == 0:
                warnings.warn(
                    'read_cell: Warning - ignoring unused '
                    'species mass {} in %BLOCK SPECIES_MASS'.format(
                        tokens[0]))
            for idx in token_pos_list:
                aargs['masses'][idx] = tokens[1]

    # Now on to the species potentials...
    if 'species_pot' in celldict:
        lines = celldict.pop('species_pot')[0].split('\n')
        line_tokens = [line.split() for line in lines]

        for tokens in line_tokens:
            if len(tokens) == 1:
                # It's a library
                all_spec = (set(custom_species) if custom_species is not None
                            else set(aargs['symbols']))
                for s in all_spec:
                    calc.cell.species_pot = (s, tokens[0])
            else:
                calc.cell.species_pot = tuple(tokens[:2])

    # Ionic constraints
    raw_constraints = {}

    if 'ionic_constraints' in celldict:
        lines = celldict.pop('ionic_constraints')[0].split('\n')
        line_tokens = [line.split() for line in lines]

        for tokens in line_tokens:
            if len(tokens) != 6:
                continue
            _, species, nic, x, y, z = tokens
            # convert xyz to floats
            x = float(x)
            y = float(y)
            z = float(z)

            nic = int(nic)
            if (species, nic) not in raw_constraints:
                raw_constraints[(species, nic)] = []
            raw_constraints[(species, nic)].append(np.array(
                                                   [x, y, z]))

    # Symmetry operations
    if 'symmetry_ops' in celldict:
        lines = celldict.pop('symmetry_ops')[0].split('\n')
        line_tokens = [line.split() for line in lines]

        # Read them in blocks of four
        blocks = np.array(line_tokens).astype(float)
        if (len(blocks.shape) != 2 or blocks.shape[1] != 3
                or blocks.shape[0] % 4 != 0):
            warnings.warn('Warning: could not parse SYMMETRY_OPS'
                          ' block properly, skipping')
        else:
            blocks = blocks.reshape((-1, 4, 3))
            rotations = blocks[:, :3]
            translations = blocks[:, 3]

            # Regardless of whether we recognize them, store these
            calc.cell.symmetry_ops = (rotations, translations)

    # Anything else that remains, just add it to the cell object:
    for k, (val, otype) in celldict.items():
        try:
            if otype == 'block':
                val = val.split('\n')  # Avoids a bug for one-line blocks
            calc.cell.__setattr__(k, val)
        except Exception as e:
            raise RuntimeError(
                f'Problem setting calc.cell.{k} = {val}: {e}')

    # Get the relevant additional info
    aargs['magmoms'] = np.array(add_info_arrays['SPIN'])
    # SPIN or MAGMOM are alternative keywords
    aargs['magmoms'] = np.where(aargs['magmoms'] != 0,
                                aargs['magmoms'],
                                add_info_arrays['MAGMOM'])
    labels = np.array(add_info_arrays['LABEL'])

    aargs['calculator'] = calc

    atoms = ase.Atoms(**aargs)

    # Spacegroup...
    if find_spg:
        # Try importing spglib
        try:
            import spglib
        except ImportError:
            warnings.warn('spglib not found installed on this system - '
                          'automatic spacegroup detection is not possible')
            spglib = None

        if spglib is not None:
            symmd = spglib.get_symmetry_dataset(atoms_to_spglib_cell(atoms))
            atoms_spg = Spacegroup(int(symmd['number']))
            atoms.info['spacegroup'] = atoms_spg

    atoms.new_array('castep_labels', labels)
    if custom_species is not None:
        atoms.new_array('castep_custom_species', np.array(custom_species))

    fixed_atoms = []
    constraints = []
    index_dict = atoms.symbols.indices()
    for (species, nic), value in raw_constraints.items():

        absolute_nr = index_dict[species][nic - 1]
        if len(value) == 3:
            # Check if they are linearly independent
            if np.linalg.det(value) == 0:
                warnings.warn(
                    'Error: Found linearly dependent constraints attached '
                    'to atoms %s' %
                    (absolute_nr))
                continue
            fixed_atoms.append(absolute_nr)
        elif len(value) == 2:
            direction = np.cross(value[0], value[1])
            # Check if they are linearly independent
            if np.linalg.norm(direction) == 0:
                warnings.warn(
                    'Error: Found linearly dependent constraints attached '
                    'to atoms %s' %
                    (absolute_nr))
                continue
            constraint = FixedLine(indices=absolute_nr, direction=direction)
            constraints.append(constraint)
        elif len(value) == 1:
            direction = np.array(value[0], dtype=float)
            constraint = FixedPlane(indices=absolute_nr, direction=direction)
            constraints.append(constraint)
        else:
            warnings.warn(
                f'Error: Found {len(value)} statements attached to atoms '
                f'{absolute_nr}'
            )

    # we need to sort the fixed atoms list in order not to raise an assertion
    # error in FixAtoms
    if fixed_atoms:
        constraints.append(FixAtoms(indices=sorted(fixed_atoms)))
    if constraints:
        atoms.set_constraint(constraints)

    atoms.calc.atoms = atoms
    atoms.calc.push_oldstate()

    return atoms


def read_phonon(filename, index=None, read_vib_data=False,
                gamma_only=True, frequency_factor=None,
                units=units_CODATA2002):
    """
    Wrapper function for the more generic read() functionality.

    Note that this is function is intended to maintain backwards-compatibility
    only. For documentation see read_castep_phonon().
    """
    from ase.io import read

    if read_vib_data:
        full_output = True
    else:
        full_output = False

    return read(filename, index=index, format='castep-phonon',
                full_output=full_output, read_vib_data=read_vib_data,
                gamma_only=gamma_only, frequency_factor=frequency_factor,
                units=units)


def read_castep_phonon(fd, index=None, read_vib_data=False,
                       gamma_only=True, frequency_factor=None,
                       units=units_CODATA2002):
    """
    Reads a .phonon file written by a CASTEP Phonon task and returns an atoms
    object, as well as the calculated vibrational data if requested.

    Note that the index argument has no effect as of now.
    """

    # fd is closed by embracing read() routine
    lines = fd.readlines()

    atoms = None
    cell = []
    N = Nb = Nq = 0
    scaled_positions = []
    symbols = []
    masses = []

    # header
    L = 0
    while L < len(lines):

        line = lines[L]

        if 'Number of ions' in line:
            N = int(line.split()[3])
        elif 'Number of branches' in line:
            Nb = int(line.split()[3])
        elif 'Number of wavevectors' in line:
            Nq = int(line.split()[3])
        elif 'Unit cell vectors (A)' in line:
            for _ in range(3):
                L += 1
                fields = lines[L].split()
                cell.append([float(x) for x in fields[0:3]])
        elif 'Fractional Co-ordinates' in line:
            for _ in range(N):
                L += 1
                fields = lines[L].split()
                scaled_positions.append([float(x) for x in fields[1:4]])
                symbols.append(fields[4])
                masses.append(float(fields[5]))
        elif 'END header' in line:
            L += 1
            atoms = ase.Atoms(symbols=symbols,
                              scaled_positions=scaled_positions,
                              cell=cell)
            break

        L += 1

    # Eigenmodes and -vectors
    if frequency_factor is None:
        Kayser_to_eV = 1E2 * 2 * np.pi * units['hbar'] * units['c']
    # N.B. "fixed default" unit for frequencies in .phonon files is "cm-1"
    # (i.e. the latter is unaffected by the internal unit conversion system of
    # CASTEP!) set conversion factor to convert therefrom to eV by default for
    # now
    frequency_factor = Kayser_to_eV
    qpoints = []
    weights = []
    frequencies = []
    displacements = []
    for _ in range(Nq):
        fields = lines[L].split()
        qpoints.append([float(x) for x in fields[2:5]])
        weights.append(float(fields[5]))
    freqs = []
    for _ in range(Nb):
        L += 1
        fields = lines[L].split()
        freqs.append(frequency_factor * float(fields[1]))
    frequencies.append(np.array(freqs))

    # skip the two Phonon Eigenvectors header lines
    L += 2

    # generate a list of displacements with a structure that is identical to
    # what is stored internally in the Vibrations class (see in
    # ase.vibrations.Vibrations.modes):
    #      np.array(displacements).shape == (Nb,3*N)

    disps = []
    for _ in range(Nb):
        disp_coords = []
        for _ in range(N):
            L += 1
            fields = lines[L].split()
            disp_x = float(fields[2]) + float(fields[3]) * 1.0j
            disp_y = float(fields[4]) + float(fields[5]) * 1.0j
            disp_z = float(fields[6]) + float(fields[7]) * 1.0j
            disp_coords.extend([disp_x, disp_y, disp_z])
        disps.append(np.array(disp_coords))
    displacements.append(np.array(disps))

    if read_vib_data:
        if gamma_only:
            vibdata = [frequencies[0], displacements[0]]
        else:
            vibdata = [qpoints, weights, frequencies, displacements]
        return vibdata, atoms
    else:
        return atoms


# Routines that only the calculator requires

def read_param(filename='', calc=None, fd=None, get_interface_options=False):
    if fd is None:
        if filename == '':
            raise ValueError('One between filename and fd must be provided')
        fd = open(filename)
    elif filename:
        warnings.warn('Filestream used to read param, file name will be '
                      'ignored')

    # If necessary, get the interface options
    if get_interface_options:
        int_opts = {}
        optre = re.compile(r'# ASE_INTERFACE ([^\s]+) : ([^\s]+)')

        lines = fd.readlines()
        fd.seek(0)

        for line in lines:
            m = optre.search(line)
            if m:
                int_opts[m.groups()[0]] = m.groups()[1]

    data = read_freeform(fd)

    if calc is None:
        from ase.calculators.castep import Castep
        calc = Castep(check_castep_version=False, keyword_tolerance=2)

    for kw, (val, otype) in data.items():
        if otype == 'block':
            val = val.split('\n')  # Avoids a bug for one-line blocks
        calc.param.__setattr__(kw, val)

    if not get_interface_options:
        return calc
    else:
        return calc, int_opts


def write_param(filename, param, check_checkfile=False,
                force_write=False,
                interface_options=None):
    """Writes a CastepParam object to a CASTEP .param file

    Parameters:
        filename: the location of the file to write to. If it
        exists it will be overwritten without warning. If it
        doesn't it will be created.
        param: a CastepParam instance
        check_checkfile : if set to True, write_param will
        only write continuation or reuse statement
        if a restart file exists in the same directory
    """
    if os.path.isfile(filename) and not force_write:
        warnings.warn('ase.io.castep.write_param: Set optional argument '
                      'force_write=True to overwrite %s.' % filename)
        return False

    out = paropen(filename, 'w')
    out.write('#######################################################\n')
    out.write(f'#CASTEP param file: {filename}\n')
    out.write('#Created using the Atomic Simulation Environment (ASE)#\n')
    if interface_options is not None:
        out.write('# Internal settings of the calculator\n')
        out.write('# This can be switched off by settings\n')
        out.write('# calc._export_settings = False\n')
        out.write('# If stated, this will be automatically processed\n')
        out.write('# by ase.io.castep.read_seed()\n')
        for option, value in sorted(interface_options.items()):
            out.write(f'# ASE_INTERFACE {option} : {value}\n')
    out.write('#######################################################\n\n')

    if check_checkfile:
        param = deepcopy(param)  # To avoid modifying the parent one
        for checktype in ['continuation', 'reuse']:
            opt = getattr(param, checktype)
            if opt and opt.value:
                fname = opt.value
                if fname == 'default':
                    fname = os.path.splitext(filename)[0] + '.check'
                if not (os.path.exists(fname) or
                        # CASTEP also understands relative path names, hence
                        # also check relative to the param file directory
                        os.path.exists(
                    os.path.join(os.path.dirname(filename),
                                 opt.value))):
                    opt.clear()

    write_freeform(out, param)

    out.close()


def read_seed(seed, new_seed=None, ignore_internal_keys=False):
    """A wrapper around the CASTEP Calculator in conjunction with
    read_cell and read_param. Basically this can be used to reuse
    a previous calculation which results in a triple of
    cell/param/castep file. The label of the calculation if pre-
    fixed with `copy_of_` and everything else will be recycled as
    much as possible from the addressed calculation.

    Please note that this routine will return an atoms ordering as specified
    in the cell file! It will thus undo the potential reordering internally
    done by castep.
    """

    directory = os.path.abspath(os.path.dirname(seed))
    seed = os.path.basename(seed)

    paramfile = os.path.join(directory, f'{seed}.param')
    cellfile = os.path.join(directory, f'{seed}.cell')
    castepfile = os.path.join(directory, f'{seed}.castep')
    checkfile = os.path.join(directory, f'{seed}.check')

    atoms = read_castep_cell(cellfile)
    atoms.calc._directory = directory
    atoms.calc._rename_existing_dir = False
    atoms.calc._castep_pp_path = directory
    atoms.calc.merge_param(paramfile,
                           ignore_internal_keys=ignore_internal_keys)
    if new_seed is None:
        atoms.calc._label = f'copy_of_{seed}'
    else:
        atoms.calc._label = str(new_seed)
    if os.path.isfile(castepfile):
        # _set_atoms needs to be True here
        # but we set it right back to False
        # atoms.calc._set_atoms = False
        # BUGFIX: I do not see a reason to do that!
        atoms.calc.read(castepfile)
        # atoms.calc._set_atoms = False

        # if here is a check file, we also want to re-use this information
        if os.path.isfile(checkfile):
            atoms.calc._check_file = os.path.basename(checkfile)

        # sync the top-level object with the
        # one attached to the calculator
        atoms = atoms.calc.atoms
    else:
        # There are cases where we only want to restore a calculator/atoms
        # setting without a castep file...
        # No print statement required in these cases
        warnings.warn(
            'Corresponding *.castep file not found. '
            'Atoms object will be restored from *.cell and *.param only.')
    atoms.calc.push_oldstate()

    return atoms


@reader
def read_bands(fd, units=units_CODATA2002):
    """Read Castep.bands file to kpoints, weights and eigenvalues.

    Parameters
    ----------
    fd : str | io.TextIOBase
        Path to the `.bands` file or file descriptor for open `.bands` file.
    units : dict
        Conversion factors for atomic units.

    Returns
    -------
    kpts : np.ndarray
        1d NumPy array for k-point coordinates.
    weights : np.ndarray
        1d NumPy array for k-point weights.
    eigenvalues : np.ndarray
        NumPy array for eigenvalues with shape (spin, kpts, nbands).
    efermi : float
        Fermi energy.

    """
    Hartree = units['Eh']

    nkpts = int(fd.readline().split()[-1])
    nspin = int(fd.readline().split()[-1])
    _ = float(fd.readline().split()[-1])
    nbands = int(fd.readline().split()[-1])
    efermi = float(fd.readline().split()[-1])

    kpts = np.zeros((nkpts, 3))
    weights = np.zeros(nkpts)
    eigenvalues = np.zeros((nspin, nkpts, nbands))

    # Skip unit cell
    for _ in range(4):
        fd.readline()

    def _kptline_to_i_k_wt(line: str) -> Tuple[int, List[float], float]:
        split_line = line.split()
        i_kpt = int(split_line[1]) - 1
        kpt = list(map(float, split_line[2:5]))
        wt = float(split_line[5])
        return i_kpt, kpt, wt

    # CASTEP often writes these out-of-order, so check index and write directly
    # to the correct row
    for _ in range(nkpts):
        i_kpt, kpt, wt = _kptline_to_i_k_wt(fd.readline())
        kpts[i_kpt, :], weights[i_kpt] = kpt, wt
        for spin in range(nspin):
            fd.readline()  # Skip 'Spin component N' line
            eigenvalues[spin, i_kpt, :] = [float(fd.readline())
                                           for _ in range(nbands)]

    return (kpts, weights, eigenvalues * Hartree, efermi * Hartree)
