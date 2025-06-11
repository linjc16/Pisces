# fmt: off

"""
This module contains functionality for reading and writing an ASE
Atoms object in VASP POSCAR format.

"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, TextIO, Tuple

import numpy as np

from ase import Atoms
from ase.constraints import FixAtoms, FixedLine, FixedPlane, FixScaled
from ase.io import ParseError
from ase.io.formats import string2index
from ase.io.utils import ImageIterator
from ase.symbols import Symbols
from ase.units import Ang, fs
from ase.utils import reader, writer

from .vasp_parsers import vasp_outcar_parsers as vop

__all__ = [
    'read_vasp', 'read_vasp_out', 'iread_vasp_out', 'read_vasp_xdatcar',
    'read_vasp_xml', 'write_vasp', 'write_vasp_xdatcar'
]


def parse_poscar_scaling_factor(line: str) -> np.ndarray:
    """Parse scaling factor(s) in the second line in POSCAR/CONTCAR.

    This can also be one negative number or three positive numbers.

    https://www.vasp.at/wiki/index.php/POSCAR#Full_format_specification

    """
    scale = []
    for _ in line.split()[:3]:
        try:
            scale.append(float(_))
        except ValueError:
            break
    if len(scale) not in {1, 3}:
        raise RuntimeError('The number of scaling factors must be 1 or 3.')
    if len(scale) == 3 and any(_ < 0.0 for _ in scale):
        raise RuntimeError('All three scaling factors must be positive.')
    return np.array(scale)


def get_atomtypes(fname):
    """Given a file name, get the atomic symbols.

    The function can get this information from OUTCAR and POTCAR
    format files.  The files can also be compressed with gzip or
    bzip2.

    """
    fpath = Path(fname)

    atomtypes = []
    atomtypes_alt = []
    if fpath.suffix == '.gz':
        import gzip
        opener = gzip.open
    elif fpath.suffix == '.bz2':
        import bz2
        opener = bz2.BZ2File
    else:
        opener = open
    with opener(fpath) as fd:
        for line in fd:
            if 'TITEL' in line:
                atomtypes.append(line.split()[3].split('_')[0].split('.')[0])
            elif 'POTCAR:' in line:
                atomtypes_alt.append(
                    line.split()[2].split('_')[0].split('.')[0])

    if len(atomtypes) == 0 and len(atomtypes_alt) > 0:
        # old VASP doesn't echo TITEL, but all versions print out species
        # lines preceded by "POTCAR:", twice
        if len(atomtypes_alt) % 2 != 0:
            raise ParseError(
                f'Tried to get atom types from {len(atomtypes_alt)}'
                '"POTCAR": lines in OUTCAR, but expected an even number'
            )
        atomtypes = atomtypes_alt[0:len(atomtypes_alt) // 2]

    return atomtypes


def atomtypes_outpot(posfname, numsyms):
    """Try to retrieve chemical symbols from OUTCAR or POTCAR

    If getting atomtypes from the first line in POSCAR/CONTCAR fails, it might
    be possible to find the data in OUTCAR or POTCAR, if these files exist.

    posfname -- The filename of the POSCAR/CONTCAR file we're trying to read

    numsyms -- The number of symbols we must find

    """
    posfpath = Path(posfname)

    # Check files with exactly same path except POTCAR/OUTCAR instead
    # of POSCAR/CONTCAR.
    fnames = [posfpath.with_name('POTCAR'),
              posfpath.with_name('OUTCAR')]
    # Try the same but with compressed files
    fsc = []
    for fnpath in fnames:
        fsc.append(fnpath.parent / (fnpath.name + '.gz'))
        fsc.append(fnpath.parent / (fnpath.name + '.bz2'))
    for f in fsc:
        fnames.append(f)
    # Code used to try anything with POTCAR or OUTCAR in the name
    # but this is no longer supported

    tried = []
    for fn in fnames:
        if fn in posfpath.parent.iterdir():
            tried.append(fn)
            at = get_atomtypes(fn)
            if len(at) == numsyms:
                return at

    raise ParseError('Could not determine chemical symbols. Tried files ' +
                     str(tried))


def get_atomtypes_from_formula(formula):
    """Return atom types from chemical formula (optionally prepended
    with and underscore).
    """
    from ase.symbols import string2symbols
    symbols = string2symbols(formula.split('_')[0])
    atomtypes = [symbols[0]]
    for s in symbols[1:]:
        if s != atomtypes[-1]:
            atomtypes.append(s)
    return atomtypes


@reader
def read_vasp(fd):
    """Import POSCAR/CONTCAR type file.

    Reads unitcell, atom positions and constraints from the POSCAR/CONTCAR
    file and tries to read atom types from POSCAR/CONTCAR header, if this
    fails the atom types are read from OUTCAR or POTCAR file.
    """
    atoms = read_vasp_configuration(fd)
    velocities = read_velocities_if_present(fd, len(atoms))
    if velocities is not None:
        atoms.set_velocities(velocities)
    return atoms


def read_vasp_configuration(fd):
    """Read common POSCAR/CONTCAR/CHGCAR/CHG quantities and return Atoms."""
    from ase.data import chemical_symbols

    # The first line is in principle a comment line, however in VASP
    # 4.x a common convention is to have it contain the atom symbols,
    # eg. "Ag Ge" in the same order as later in the file (and POTCAR
    # for the full vasp run). In the VASP 5.x format this information
    # is found on the fifth line. Thus we save the first line and use
    # it in case we later detect that we're reading a VASP 4.x format
    # file.
    line1 = fd.readline()

    scale = parse_poscar_scaling_factor(fd.readline())

    # Now the lattice vectors
    cell = np.array([fd.readline().split()[:3] for _ in range(3)], dtype=float)
    # Negative scaling factor corresponds to the cell volume.
    if scale[0] < 0.0:
        scale = np.cbrt(-1.0 * scale / np.linalg.det(cell))
    cell *= scale  # This works for both one and three scaling factors.

    # Number of atoms. Again this must be in the same order as
    # in the first line
    # or in the POTCAR or OUTCAR file
    atom_symbols = []
    numofatoms = fd.readline().split()
    # Check whether we have a VASP 4.x or 5.x format file. If the
    # format is 5.x, use the fifth line to provide information about
    # the atomic symbols.
    vasp5 = False
    try:
        int(numofatoms[0])
    except ValueError:
        vasp5 = True
        atomtypes = numofatoms
        numofatoms = fd.readline().split()

    # check for comments in numofatoms line and get rid of them if necessary
    commentcheck = np.array(['!' in s for s in numofatoms])
    if commentcheck.any():
        # only keep the elements up to the first including a '!':
        numofatoms = numofatoms[:np.arange(len(numofatoms))[commentcheck][0]]

    if not vasp5:
        # Split the comment line (first in the file) into words and
        # try to compose a list of chemical symbols
        from ase.formula import Formula
        atomtypes = []
        for word in line1.split():
            word_without_delims = re.sub(r"-|_|,|\.|=|[0-9]|^", "", word)
            if len(word_without_delims) < 1:
                continue
            try:
                atomtypes.extend(list(Formula(word_without_delims)))
            except ValueError:
                # print(atomtype, e, 'is comment')
                pass
        # Now the list of chemical symbols atomtypes must be formed.
        # For example: atomtypes = ['Pd', 'C', 'O']

        numsyms = len(numofatoms)
        if len(atomtypes) < numsyms:
            # First line in POSCAR/CONTCAR didn't contain enough symbols.

            # Sometimes the first line in POSCAR/CONTCAR is of the form
            # "CoP3_In-3.pos". Check for this case and extract atom types
            if len(atomtypes) == 1 and '_' in atomtypes[0]:
                atomtypes = get_atomtypes_from_formula(atomtypes[0])
            else:
                atomtypes = atomtypes_outpot(fd.name, numsyms)
        else:
            try:
                for atype in atomtypes[:numsyms]:
                    if atype not in chemical_symbols:
                        raise KeyError
            except KeyError:
                atomtypes = atomtypes_outpot(fd.name, numsyms)

    for i, num in enumerate(numofatoms):
        numofatoms[i] = int(num)
        atom_symbols.extend(numofatoms[i] * [atomtypes[i]])

    # Check if Selective dynamics is switched on
    sdyn = fd.readline()
    selective_dynamics = sdyn[0].lower() == 's'

    # Check if atom coordinates are cartesian or direct
    if selective_dynamics:
        ac_type = fd.readline()
    else:
        ac_type = sdyn
    cartesian = ac_type[0].lower() in ['c', 'k']
    tot_natoms = sum(numofatoms)
    atoms_pos = np.empty((tot_natoms, 3))
    if selective_dynamics:
        selective_flags = np.empty((tot_natoms, 3), dtype=bool)
    for atom in range(tot_natoms):
        ac = fd.readline().split()
        atoms_pos[atom] = [float(_) for _ in ac[0:3]]
        if selective_dynamics:
            selective_flags[atom] = [_ == 'F' for _ in ac[3:6]]

    atoms = Atoms(symbols=atom_symbols, cell=cell, pbc=True)
    if cartesian:
        atoms_pos *= scale
        atoms.set_positions(atoms_pos)
    else:
        atoms.set_scaled_positions(atoms_pos)
    if selective_dynamics:
        set_constraints(atoms, selective_flags)

    return atoms


def read_velocities_if_present(fd, natoms) -> np.ndarray | None:
    """Read velocities from POSCAR/CONTCAR if present, return in ASE units."""
    ac_type = fd.readline()

    # Check if velocities are present
    if not ac_type:
        return None

    atoms_vel = np.empty((natoms, 3))
    for atom in range(natoms):
        words = fd.readline().split()
        assert len(words) == 3
        atoms_vel[atom] = (float(words[0]), float(words[1]), float(words[2]))

    # unit conversion from Angstrom/fs to ASE units
    return atoms_vel * (Ang / fs)


def set_constraints(atoms: Atoms, selective_flags: np.ndarray):
    """Set constraints based on selective_flags"""
    from ase.constraints import FixAtoms, FixConstraint, FixScaled

    constraints: List[FixConstraint] = []
    indices = []
    for ind, sflags in enumerate(selective_flags):
        if sflags.any() and not sflags.all():
            constraints.append(FixScaled(ind, sflags, atoms.get_cell()))
        elif sflags.all():
            indices.append(ind)
    if indices:
        constraints.append(FixAtoms(indices))
    if constraints:
        atoms.set_constraint(constraints)


def iread_vasp_out(filename, index=-1):
    """Import OUTCAR type file, as a generator."""
    it = ImageIterator(vop.outcarchunks)
    return it(filename, index=index)


@reader
def read_vasp_out(filename='OUTCAR', index=-1):
    """Import OUTCAR type file.

    Reads unitcell, atom positions, energies, and forces from the OUTCAR file
    and attempts to read constraints (if any) from CONTCAR/POSCAR, if present.
    """
    # "filename" is actually a file-descriptor thanks to @reader
    g = iread_vasp_out(filename, index=index)
    # Code borrowed from formats.py:read
    if isinstance(index, (slice, str)):
        # Return list of atoms
        return list(g)
    else:
        # Return single atoms object
        return next(g)


@reader
def read_vasp_xdatcar(filename='XDATCAR', index=-1):
    """Import XDATCAR file.

    Parameters
    ----------
    index : int or slice or str
        Which frame(s) to read. The default is -1 (last frame).
        See :func:`ase.io.read` for details.

    Notes
    -----
    Constraints ARE NOT stored in the XDATCAR, and as such, Atoms objects
    retrieved from the XDATCAR will not have constraints.
    """
    fd = filename  # @reader decorator ensures this is a file descriptor
    images = []

    cell = np.eye(3)
    atomic_formula = ''

    while True:
        comment_line = fd.readline()
        if "Direct configuration=" not in comment_line:
            try:
                lattice_constant = float(fd.readline())
            except Exception:
                # XXX: When would this happen?
                break

            xx = [float(x) for x in fd.readline().split()]
            yy = [float(y) for y in fd.readline().split()]
            zz = [float(z) for z in fd.readline().split()]
            cell = np.array([xx, yy, zz]) * lattice_constant

            symbols = fd.readline().split()
            numbers = [int(n) for n in fd.readline().split()]
            total = sum(numbers)

            atomic_formula = ''.join(f'{sym:s}{numbers[n]:d}'
                                     for n, sym in enumerate(symbols))

            fd.readline()

        coords = [np.array(fd.readline().split(), float) for _ in range(total)]

        image = Atoms(atomic_formula, cell=cell, pbc=True)
        image.set_scaled_positions(np.array(coords))
        images.append(image)

    if index is None:
        index = -1

    if isinstance(index, str):
        index = string2index(index)

    return images[index]


def __get_xml_parameter(par):
    """An auxiliary function that enables convenient extraction of
    parameter values from a vasprun.xml file with proper type
    handling.

    """
    def to_bool(b):
        if b == 'T':
            return True
        else:
            return False

    to_type = {'int': int, 'logical': to_bool, 'string': str, 'float': float}

    text = par.text
    if text is None:
        text = ''

    # Float parameters do not have a 'type' attrib
    var_type = to_type[par.attrib.get('type', 'float')]

    try:
        if par.tag == 'v':
            return list(map(var_type, text.split()))
        else:
            return var_type(text.strip())
    except ValueError:
        # Vasp can sometimes write "*****" due to overflow
        return None


def read_vasp_xml(filename='vasprun.xml', index=-1):
    """Parse vasprun.xml file.

    Reads unit cell, atom positions, energies, forces, and constraints
    from vasprun.xml file

    Examples:
        >>> import ase.io
        >>> ase.io.write("out.traj", ase.io.read("vasprun.xml", index=":"))
    """

    import xml.etree.ElementTree as ET
    from collections import OrderedDict

    from ase.calculators.singlepoint import (
        SinglePointDFTCalculator,
        SinglePointKPoint,
    )
    from ase.units import GPa

    tree = ET.iterparse(filename, events=['start', 'end'])

    atoms_init = None
    calculation = []
    ibz_kpts = None
    kpt_weights = None
    parameters = OrderedDict()

    try:
        for event, elem in tree:

            if event == 'end':
                if elem.tag == 'kpoints':
                    for subelem in elem.iter(tag='generation'):
                        kpts_params = OrderedDict()
                        parameters['kpoints_generation'] = kpts_params
                        for par in subelem.iter():
                            if par.tag in ['v', 'i']:
                                parname = par.attrib['name'].lower()
                                kpts_params[parname] = __get_xml_parameter(par)

                    kpts = elem.findall("varray[@name='kpointlist']/v")
                    ibz_kpts = np.zeros((len(kpts), 3))

                    for i, kpt in enumerate(kpts):
                        ibz_kpts[i] = [float(val) for val in kpt.text.split()]

                    kpt_weights = elem.findall('varray[@name="weights"]/v')
                    kpt_weights = [float(val.text) for val in kpt_weights]

                elif elem.tag == 'parameters':
                    for par in elem.iter():
                        if par.tag in ['v', 'i']:
                            parname = par.attrib['name'].lower()
                            parameters[parname] = __get_xml_parameter(par)

                elif elem.tag == 'atominfo':
                    species = []

                    for entry in elem.find("array[@name='atoms']/set"):
                        species.append(entry[0].text.strip())

                    natoms = len(species)

                elif (elem.tag == 'structure'
                      and elem.attrib.get('name') == 'initialpos'):
                    cell_init = np.zeros((3, 3), dtype=float)

                    for i, v in enumerate(
                            elem.find("crystal/varray[@name='basis']")):
                        cell_init[i] = np.array(
                            [float(val) for val in v.text.split()])

                    scpos_init = np.zeros((natoms, 3), dtype=float)

                    for i, v in enumerate(
                            elem.find("varray[@name='positions']")):
                        scpos_init[i] = np.array(
                            [float(val) for val in v.text.split()])

                    constraints = []
                    fixed_indices = []

                    for i, entry in enumerate(
                            elem.findall("varray[@name='selective']/v")):
                        flags = (np.array(
                            entry.text.split() == np.array(['F', 'F', 'F'])))
                        if flags.all():
                            fixed_indices.append(i)
                        elif flags.any():
                            constraints.append(FixScaled(i, flags, cell_init))

                    if fixed_indices:
                        constraints.append(FixAtoms(fixed_indices))

                    atoms_init = Atoms(species,
                                       cell=cell_init,
                                       scaled_positions=scpos_init,
                                       constraint=constraints,
                                       pbc=True)

                elif elem.tag == 'dipole':
                    dblock = elem.find('v[@name="dipole"]')
                    if dblock is not None:
                        dipole = np.array(
                            [float(val) for val in dblock.text.split()])

            elif event == 'start' and elem.tag == 'calculation':
                calculation.append(elem)

    except ET.ParseError as parse_error:
        if atoms_init is None:
            raise parse_error
        if calculation and calculation[-1].find("energy") is None:
            calculation = calculation[:-1]
        if not calculation:
            yield atoms_init

    if calculation:
        if isinstance(index, int):
            steps = [calculation[index]]
        else:
            steps = calculation[index]
    else:
        steps = []

    for step in steps:
        # Workaround for VASP bug, e_0_energy contains the wrong value
        # in calculation/energy, but calculation/scstep/energy does not
        # include classical VDW corrections. So, first calculate
        # e_0_energy - e_fr_energy from calculation/scstep/energy, then
        # apply that correction to e_fr_energy from calculation/energy.
        lastscf = step.findall('scstep/energy')[-1]
        dipoles = step.findall('scstep/dipole')
        if dipoles:
            lastdipole = dipoles[-1]
        else:
            lastdipole = None

        de = (float(lastscf.find('i[@name="e_0_energy"]').text) -
              float(lastscf.find('i[@name="e_fr_energy"]').text))

        free_energy = float(step.find('energy/i[@name="e_fr_energy"]').text)
        energy = free_energy + de

        cell = np.zeros((3, 3), dtype=float)
        for i, vector in enumerate(
                step.find('structure/crystal/varray[@name="basis"]')):
            cell[i] = np.array([float(val) for val in vector.text.split()])

        scpos = np.zeros((natoms, 3), dtype=float)
        for i, vector in enumerate(
                step.find('structure/varray[@name="positions"]')):
            scpos[i] = np.array([float(val) for val in vector.text.split()])

        forces = None
        fblocks = step.find('varray[@name="forces"]')
        if fblocks is not None:
            forces = np.zeros((natoms, 3), dtype=float)
            for i, vector in enumerate(fblocks):
                forces[i] = np.array(
                    [float(val) for val in vector.text.split()])

        stress = None
        sblocks = step.find('varray[@name="stress"]')
        if sblocks is not None:
            stress = np.zeros((3, 3), dtype=float)
            for i, vector in enumerate(sblocks):
                stress[i] = np.array(
                    [float(val) for val in vector.text.split()])
            stress *= -0.1 * GPa
            stress = stress.reshape(9)[[0, 4, 8, 5, 2, 1]]

        dipole = None
        if lastdipole is not None:
            dblock = lastdipole.find('v[@name="dipole"]')
            if dblock is not None:
                dipole = np.zeros((1, 3), dtype=float)
                dipole = np.array([float(val) for val in dblock.text.split()])

        dblock = step.find('dipole/v[@name="dipole"]')
        if dblock is not None:
            dipole = np.zeros((1, 3), dtype=float)
            dipole = np.array([float(val) for val in dblock.text.split()])

        efermi = step.find('dos/i[@name="efermi"]')
        if efermi is not None:
            efermi = float(efermi.text)

        kpoints = []
        for ikpt in range(1, len(ibz_kpts) + 1):
            kblocks = step.findall(
                'eigenvalues/array/set/set/set[@comment="kpoint %d"]' % ikpt)
            if kblocks is not None:
                for spin, kpoint in enumerate(kblocks):
                    eigenvals = kpoint.findall('r')
                    eps_n = np.zeros(len(eigenvals))
                    f_n = np.zeros(len(eigenvals))
                    for j, val in enumerate(eigenvals):
                        val = val.text.split()
                        eps_n[j] = float(val[0])
                        f_n[j] = float(val[1])
                    if len(kblocks) == 1:
                        f_n *= 2
                    kpoints.append(
                        SinglePointKPoint(kpt_weights[ikpt - 1], spin, ikpt,
                                          eps_n, f_n))
        if len(kpoints) == 0:
            kpoints = None

        # DFPT properties
        # dielectric tensor
        dielectric_tensor = None
        sblocks = step.find('varray[@name="dielectric_dft"]')
        if sblocks is not None:
            dielectric_tensor = np.zeros((3, 3), dtype=float)
            for ii, vector in enumerate(sblocks):
                dielectric_tensor[ii] = np.fromstring(vector.text, sep=' ')

        # Born effective charges
        born_charges = None
        fblocks = step.find('array[@name="born_charges"]')
        if fblocks is not None:
            born_charges = np.zeros((natoms, 3, 3), dtype=float)
            for ii, block in enumerate(fblocks[1:]):  # 1. element = dimension
                for jj, vector in enumerate(block):
                    born_charges[ii, jj] = np.fromstring(vector.text, sep=' ')

        atoms = atoms_init.copy()
        atoms.set_cell(cell)
        atoms.set_scaled_positions(scpos)
        atoms.calc = SinglePointDFTCalculator(
            atoms,
            energy=energy,
            forces=forces,
            stress=stress,
            free_energy=free_energy,
            ibzkpts=ibz_kpts,
            efermi=efermi,
            dipole=dipole,
            dielectric_tensor=dielectric_tensor,
            born_effective_charges=born_charges
        )
        atoms.calc.name = 'vasp'
        atoms.calc.kpts = kpoints
        atoms.calc.parameters = parameters
        yield atoms


@writer
def write_vasp_xdatcar(fd, images, label=None):
    """Write VASP MD trajectory (XDATCAR) file

    Only Vasp 5 format is supported (for consistency with read_vasp_xdatcar)

    Args:
        fd (str, fp): Output file
        images (iterable of Atoms): Atoms images to write. These must have
            consistent atom order and lattice vectors - this will not be
            checked.
        label (str): Text for first line of file. If empty, default to list
            of elements.

    """

    images = iter(images)
    image = next(images)

    if not isinstance(image, Atoms):
        raise TypeError("images should be a sequence of Atoms objects.")

    symbol_count = _symbol_count_from_symbols(image.get_chemical_symbols())

    if label is None:
        label = ' '.join([s for s, _ in symbol_count])
    fd.write(label + '\n')

    # Not using lattice constants, set it to 1
    fd.write('           1\n')

    # Lattice vectors; use first image
    float_string = '{:11.6f}'
    for row_i in range(3):
        fd.write('  ')
        fd.write(' '.join(float_string.format(x) for x in image.cell[row_i]))
        fd.write('\n')

    fd.write(_symbol_count_string(symbol_count, vasp5=True))
    _write_xdatcar_config(fd, image, index=1)
    for i, image in enumerate(images):
        # Index is off by 2: 1-indexed file vs 0-indexed Python;
        # and we already wrote the first block.
        _write_xdatcar_config(fd, image, i + 2)


def _write_xdatcar_config(fd, atoms, index):
    """Write a block of positions for XDATCAR file

    Args:
        fd (fd): writeable Python file descriptor
        atoms (ase.Atoms): Atoms to write
        index (int): configuration number written to block header

    """
    fd.write(f"Direct configuration={index:6d}\n")
    float_string = '{:11.8f}'
    scaled_positions = atoms.get_scaled_positions()
    for row in scaled_positions:
        fd.write(' ')
        fd.write(' '.join([float_string.format(x) for x in row]))
        fd.write('\n')


def _symbol_count_from_symbols(symbols: Symbols) -> List[Tuple[str, int]]:
    """Reduce list of chemical symbols into compact VASP notation

    Args:
        symbols (iterable of str)

    Returns:
        list of pairs [(el1, c1), (el2, c2), ...]

    Example:
    >>> s = Atoms('Ar3NeHe2ArNe').symbols
    >>> _symbols_count_from_symbols(s)
    [('Ar', 3), ('Ne', 1), ('He', 2), ('Ar', 1), ('Ne', 1)]
    """
    sc = []
    psym = str(symbols[0])  # we cast to str to appease mypy
    count = 0
    for sym in symbols:
        if sym != psym:
            sc.append((psym, count))
            psym = sym
            count = 1
        else:
            count += 1

    sc.append((psym, count))
    return sc


@writer
def write_vasp(
    fd: TextIO,
    atoms: Atoms,
    direct: bool = False,
    sort: bool = False,
    symbol_count: Optional[List[Tuple[str, int]]] = None,
    vasp5: bool = True,
    vasp6: bool = False,
    ignore_constraints: bool = False,
    potential_mapping: Optional[dict] = None
) -> None:
    """Method to write VASP position (POSCAR/CONTCAR) files.

    Writes label, scalefactor, unitcell, # of various kinds of atoms,
    positions in cartesian or scaled coordinates (Direct), and constraints
    to file. Cartesian coordinates is default and default label is the
    atomic species, e.g. 'C N H Cu'.

    Args:
        fd (TextIO): writeable Python file descriptor
        atoms (ase.Atoms): Atoms to write
        direct (bool): Write scaled coordinates instead of cartesian
        sort (bool): Sort the atomic indices alphabetically by element
        symbol_count (list of tuples of str and int, optional): Use the given
            combination of symbols and counts instead of automatically compute
            them
        vasp5 (bool): Write to the VASP 5+ format, where the symbols are
            written to file
        vasp6 (bool): Write symbols in VASP 6 format (which allows for
            potential type and hash)
        ignore_constraints (bool): Ignore all constraints on `atoms`
        potential_mapping (dict, optional): Map of symbols to potential file
            (and hash). Only works if `vasp6=True`. See `_symbol_string_count`

    Raises:
        RuntimeError: raised if any of these are true:

            1. `atoms` is not a single `ase.Atoms` object.
            2. The cell dimensionality is lower than 3 (0D-2D)
            3. One FixedPlane normal is not parallel to a unit cell vector
            4. One FixedLine direction is not parallel to a unit cell vector
    """
    if isinstance(atoms, (list, tuple)):
        if len(atoms) > 1:
            raise RuntimeError(
                'Only one atomic structure can be saved to VASP '
                'POSCAR/CONTCAR. Several were given.'
            )
        else:
            atoms = atoms[0]

    # Check lattice vectors are finite
    if atoms.cell.rank < 3:
        raise RuntimeError(
            'Lattice vectors must be finite and non-parallel. At least '
            'one lattice length or angle is zero.'
        )

    # Write atomic positions in scaled or cartesian coordinates
    if direct:
        coord = atoms.get_scaled_positions(wrap=False)
    else:
        coord = atoms.positions

    # Convert ASE constraints to VASP POSCAR constraints
    constraints_present = atoms.constraints and not ignore_constraints
    if constraints_present:
        sflags = _handle_ase_constraints(atoms)

    # Conditionally sort ordering of `atoms` alphabetically by symbols
    if sort:
        ind = np.argsort(atoms.symbols)
        symbols = atoms.symbols[ind]
        coord = coord[ind]
        if constraints_present:
            sflags = sflags[ind]
    else:
        symbols = atoms.symbols

    # Set or create a list of (symbol, count) pairs
    sc = symbol_count or _symbol_count_from_symbols(symbols)

    # Write header as atomic species in `symbol_count` order
    label = ' '.join(f'{sym:2s}' for sym, _ in sc)
    fd.write(label + '\n')

    # For simplicity, we write the unitcell in real coordinates, so the
    # scaling factor is always set to 1.0.
    fd.write(f'{1.0:19.16f}\n')

    for vec in atoms.cell:
        fd.write('  ' + ' '.join([f'{el:21.16f}' for el in vec]) + '\n')

    # Write version-dependent species-and-count section
    sc_str = _symbol_count_string(sc, vasp5, vasp6, potential_mapping)
    fd.write(sc_str)

    # Write POSCAR switches
    if constraints_present:
        fd.write('Selective dynamics\n')

    fd.write('Direct\n' if direct else 'Cartesian\n')

    # Write atomic positions and, if any, the cartesian constraints
    for iatom, atom in enumerate(coord):
        for dcoord in atom:
            fd.write(f' {dcoord:19.16f}')
        if constraints_present:
            flags = ['F' if flag else 'T' for flag in sflags[iatom]]
            fd.write(''.join([f'{f:>4s}' for f in flags]))
        fd.write('\n')

    # if velocities in atoms object write velocities
    if atoms.has('momenta'):
        cform = 3 * ' {:19.16f}' + '\n'
        fd.write('Cartesian\n')
        # unit conversion to Angstrom / fs
        vel = atoms.get_velocities() / (Ang / fs)
        for vatom in vel:
            fd.write(cform.format(*vatom))


def _handle_ase_constraints(atoms: Atoms) -> np.ndarray:
    """Convert the ASE constraints on `atoms` to VASP constraints

    Returns a boolean array with dimensions Nx3, where N is the number of
    atoms. A value of `True` indicates that movement along the given lattice
    vector is disallowed for that atom.

    Args:
        atoms (Atoms)

    Returns:
        boolean numpy array with dimensions Nx3

    Raises:
        RuntimeError: If there is a FixedPlane or FixedLine constraint, that
                      is not parallel to a cell vector.
    """
    sflags = np.zeros((len(atoms), 3), dtype=bool)
    for constr in atoms.constraints:
        if isinstance(constr, FixScaled):
            sflags[constr.index] = constr.mask
        elif isinstance(constr, FixAtoms):
            sflags[constr.index] = 3 * [True]
        elif isinstance(constr, FixedPlane):
            # Calculate if the plane normal is parallel to a cell vector
            mask = np.all(
                np.abs(np.cross(constr.dir, atoms.cell)) < 1e-5, axis=1
            )
            if mask.sum() != 1:
                raise RuntimeError(
                    'VASP requires that the direction of FixedPlane '
                    'constraints is parallel with one of the cell axis'
                )
            sflags[constr.index] = mask
        elif isinstance(constr, FixedLine):
            # Calculate if line is parallel to a cell vector
            mask = np.all(
                np.abs(np.cross(constr.dir, atoms.cell)) < 1e-5, axis=1
            )
            if mask.sum() != 1:
                raise RuntimeError(
                    'VASP requires that the direction of FixedLine '
                    'constraints is parallel with one of the cell axis'
                )
            sflags[constr.index] = ~mask

    return sflags


def _symbol_count_string(
    symbol_count: List[Tuple[str, int]], vasp5: bool = True,
    vasp6: bool = True, symbol_mapping: Optional[dict] = None
) -> str:
    """Create the symbols-and-counts block for POSCAR or XDATCAR

    Args:
        symbol_count (list of 2-tuple): list of paired elements and counts
        vasp5 (bool): if False, omit symbols and only write counts
        vasp6 (bool): if True, write symbols in VASP 6 format (allows for
                      potential type and hash)
        symbol_mapping (dict): mapping of symbols to VASP 6 symbols

    e.g. if sc is [(Sn, 4), (S, 6)] then write for vasp 5:
      Sn   S
       4   6

    and for vasp 6 with mapping {'Sn': 'Sn_d_GW', 'S': 'S_GW/357d'}:
        Sn_d_GW S_GW/357d
                4        6
    """
    symbol_mapping = symbol_mapping or {}
    out_str = ' '

    # Allow for VASP 6 format, i.e., specifying the pseudopotential used
    if vasp6:
        out_str += ' '.join([
            f"{symbol_mapping.get(s, s)[:14]:16s}" for s, _ in symbol_count
        ]) + "\n "
        out_str += ' '.join([f"{c:16d}" for _, c in symbol_count]) + '\n'
        return out_str

    # Write the species for VASP 5+
    if vasp5:
        out_str += ' '.join([f"{s:3s}" for s, _ in symbol_count]) + "\n "

    # Write counts line
    out_str += ' '.join([f"{c:3d}" for _, c in symbol_count]) + '\n'

    return out_str
