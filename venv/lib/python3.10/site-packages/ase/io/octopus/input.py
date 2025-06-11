# fmt: off

import os
import re

import numpy as np

from ase import Atoms
from ase.calculators.calculator import kpts2ndarray
from ase.data import atomic_numbers
from ase.io import read
from ase.units import Bohr, Hartree
from ase.utils import reader

special_ase_keywords = {'kpts'}


def process_special_kwargs(atoms, kwargs):
    kwargs = kwargs.copy()
    kpts = kwargs.pop('kpts', None)
    if kpts is not None:
        for kw in ['kpoints', 'reducedkpoints', 'kpointsgrid']:
            if kw in kwargs:
                raise ValueError('k-points specified multiple times')

        kptsarray = kpts2ndarray(kpts, atoms)
        nkpts = len(kptsarray)
        fullarray = np.empty((nkpts, 4))
        fullarray[:, 0] = 1.0 / nkpts  # weights
        fullarray[:, 1:4] = kptsarray
        kwargs['kpointsreduced'] = fullarray.tolist()

    # TODO xc=LDA/PBE etc.

    # The idea is to get rid of the special keywords, since the rest
    # will be passed to Octopus
    # XXX do a better check of this
    for kw in special_ase_keywords:
        assert kw not in kwargs, kw
    return kwargs


class OctopusParseError(Exception):
    pass  # Cannot parse input file


# Parse value as written in input file *or* something that one would be
# passing to the ASE interface, i.e., this might already be a boolean
def octbool2bool(value):
    value = value.lower()
    if isinstance(value, int):
        return bool(value)
    if value in ['true', 't', 'yes', '1']:
        return True
    elif value in ['no', 'f', 'false', '0']:
        return False
    else:
        raise ValueError(f'Failed to interpret "{value}" as a boolean.')


def list2block(name, rows):
    """Construct 'block' of Octopus input.

    convert a list of rows to a string with the format x | x | ....
    for the octopus input file"""
    lines = []
    lines.append('%' + name)
    for row in rows:
        lines.append(' ' + ' | '.join(str(obj) for obj in row))
    lines.append('%')
    return lines


def normalize_keywords(kwargs):
    """Reduce keywords to unambiguous form (lowercase)."""
    newkwargs = {}
    for arg, value in kwargs.items():
        lkey = arg.lower()
        newkwargs[lkey] = value
    return newkwargs


def input_line_iter(lines):
    """Convenient iterator for parsing input files 'cleanly'.

    Discards comments etc."""
    for line in lines:
        line = line.split('#')[0].strip()
        if not line or line.isspace():
            continue
        line = line.strip()
        yield line


def block2list(namespace, lines, header=None):
    """Parse lines of block and return list of lists of strings."""
    lines = iter(lines)
    block = []
    if header is None:
        header = next(lines)
    assert header.startswith('%'), header
    name = header[1:].strip().lower()
    for line in lines:
        if line.startswith('%'):  # Could also say line == '%' most likely.
            break
        tokens = [namespace.evaluate(token)
                  for token in line.strip().split('|')]
        # XXX will fail for string literals containing '|'
        block.append(tokens)
    return name, block


class OctNamespace:
    def __init__(self):
        self.names = {}
        self.consts = {'pi': np.pi,
                       'angstrom': 1. / Bohr,
                       'ev': 1. / Hartree,
                       'yes': True,
                       'no': False,
                       't': True,
                       'f': False,
                       'i': 1j,  # This will probably cause trouble
                       'true': True,
                       'false': False}

    def evaluate(self, value):
        value = value.strip()

        for char in '"', "'":  # String literal
            if value.startswith(char):
                assert value.endswith(char)
                return value

        value = value.lower()

        if value in self.consts:  # boolean or other constant
            return self.consts[value]

        if value in self.names:  # existing variable
            return self.names[value]

        try:  # literal integer
            v = int(value)
        except ValueError:
            pass
        else:
            if v == float(v):
                return v

        try:  # literal float
            return float(value)
        except ValueError:
            pass

        if ('*' in value or '/' in value
                and not any(char in value for char in '()+')):
            floatvalue = 1.0
            op = '*'
            for token in re.split(r'([\*/])', value):
                if token in '*/':
                    op = token
                    continue

                v = self.evaluate(token)

                try:
                    v = float(v)
                except TypeError:
                    try:
                        v = complex(v)
                    except ValueError:
                        break
                except ValueError:
                    break  # Cannot evaluate expression
                else:
                    if op == '*':
                        floatvalue *= v
                    else:
                        assert op == '/', op
                        floatvalue /= v
            else:  # Loop completed successfully
                return floatvalue
        return value  # unknown name, or complex arithmetic expression

    def add(self, name, value):
        value = self.evaluate(value)
        self.names[name.lower().strip()] = value


def parse_input_file(fd):
    namespace = OctNamespace()
    lines = input_line_iter(fd)
    blocks = {}
    while True:
        try:
            line = next(lines)
        except StopIteration:
            break
        else:
            if line.startswith('%'):
                name, value = block2list(namespace, lines, header=line)
                blocks[name] = value
            else:
                tokens = line.split('=', 1)
                assert len(tokens) == 2, tokens
                name, value = tokens
                namespace.add(name, value)

    namespace.names.update(blocks)
    return namespace.names


def kwargs2cell(kwargs):
    # kwargs -> cell + remaining kwargs
    # cell will be None if not ASE-compatible.
    #
    # Returns numbers verbatim; caller must convert units.
    kwargs = normalize_keywords(kwargs)

    if boxshape_is_ase_compatible(kwargs):
        kwargs.pop('boxshape', None)
        if 'lsize' in kwargs:
            Lsize = kwargs.pop('lsize')
            if not isinstance(Lsize, list):
                Lsize = [[Lsize] * 3]
            assert len(Lsize) == 1
            cell = np.array([2 * float(x) for x in Lsize[0]])
        elif 'latticeparameters' in kwargs:
            # Eval latparam and latvec
            latparam = np.array(kwargs.pop('latticeparameters'), float).T
            cell = np.array(kwargs.pop('latticevectors', np.eye(3)), float)
            for a, vec in zip(latparam, cell):
                vec *= a
            assert cell.shape == (3, 3)
    else:
        cell = None
    return cell, kwargs


def boxshape_is_ase_compatible(kwargs):
    pdims = int(kwargs.get('periodicdimensions', 0))
    default_boxshape = 'parallelepiped' if pdims > 0 else 'minimum'
    boxshape = kwargs.get('boxshape', default_boxshape).lower()
    # XXX add support for experimental keyword 'latticevectors'
    return boxshape == 'parallelepiped'


def kwargs2atoms(kwargs, directory=None):
    """Extract atoms object from keywords and return remaining keywords.

    Some keyword arguments may refer to files.  The directory keyword
    may be necessary to resolve the paths correctly, and is used for
    example when running 'ase gui somedir/inp'."""
    kwargs = normalize_keywords(kwargs)

    # Only input units accepted nowadays are 'atomic'.
    # But if we are loading an old file, and it specifies something else,
    # we can be sure that the user wanted that back then.

    coord_keywords = ['coordinates',
                      'xyzcoordinates',
                      'pdbcoordinates',
                      'reducedcoordinates',
                      'xsfcoordinates',
                      'xsfcoordinatesanimstep']

    nkeywords = 0
    for keyword in coord_keywords:
        if keyword in kwargs:
            nkeywords += 1
    if nkeywords == 0:
        raise OctopusParseError('No coordinates')
    elif nkeywords > 1:
        raise OctopusParseError('Multiple coordinate specifications present.  '
                                'This may be okay in Octopus, but we do not '
                                'implement it.')

    def get_positions_from_block(keyword):
        # %Coordinates or %ReducedCoordinates -> atomic numbers, positions.
        block = kwargs.pop(keyword)
        positions = []
        numbers = []
        tags = []
        types = {}
        for row in block:
            assert len(row) in [ndims + 1, ndims + 2]
            row = row[:ndims + 1]
            sym = row[0]
            assert sym.startswith('"') or sym.startswith("'")
            assert sym[0] == sym[-1]
            sym = sym[1:-1]
            pos0 = np.zeros(3)
            ndim = int(kwargs.get('dimensions', 3))
            pos0[:ndim] = [float(element) for element in row[1:]]
            number = atomic_numbers.get(sym)  # Use 0 ~ 'X' for unknown?
            tag = 0
            if number is None:
                if sym not in types:
                    tag = len(types) + 1
                    types[sym] = tag
                number = 0
                tag = types[sym]
            tags.append(tag)
            numbers.append(number)
            positions.append(pos0)
        positions = np.array(positions)
        tags = np.array(tags, int)
        if types:
            ase_types = {}
            for sym, tag in types.items():
                ase_types[('X', tag)] = sym
            info = {'types': ase_types}  # 'info' dict for Atoms object
        else:
            tags = None
            info = None
        return numbers, positions, tags, info

    def read_atoms_from_file(fname, fmt):
        assert fname.startswith('"') or fname.startswith("'")
        assert fname[0] == fname[-1]
        fname = fname[1:-1]
        if directory is not None:
            fname = os.path.join(directory, fname)
        # XXX test xyz, pbd and xsf
        if fmt == 'xsf' and 'xsfcoordinatesanimstep' in kwargs:
            anim_step = kwargs.pop('xsfcoordinatesanimstep')
            theslice = slice(anim_step, anim_step + 1, 1)
            # XXX test animstep
        else:
            theslice = slice(None, None, 1)
        images = read(fname, theslice, fmt)
        if len(images) != 1:
            raise OctopusParseError('Expected only one image.  Don\'t know '
                                    'what to do with %d images.' % len(images))
        return images[0]

    # We will attempt to extract cell and pbc from kwargs if 'lacking'.
    # But they might have been left unspecified on purpose.
    #
    # We need to keep track of these two variables "externally"
    # because the Atoms object assigns values when they are not given.
    cell = None
    pbc = None
    adjust_positions_by_half_cell = False

    atoms = None
    xsfcoords = kwargs.pop('xsfcoordinates', None)
    if xsfcoords is not None:
        atoms = read_atoms_from_file(xsfcoords, 'xsf')
        atoms.positions *= Bohr
        atoms.cell *= Bohr
        # As it turns out, non-periodic xsf is not supported by octopus.
        # Also, it only supports fully periodic or fully non-periodic....
        # So the only thing that we can test here is 3D fully periodic.
        if sum(atoms.pbc) != 3:
            raise NotImplementedError('XSF not fully periodic with Octopus')
        cell = atoms.cell
        pbc = atoms.pbc
        # Position adjustment doesn't actually matter but this should work
        # most 'nicely':
        adjust_positions_by_half_cell = False
    xyzcoords = kwargs.pop('xyzcoordinates', None)
    if xyzcoords is not None:
        atoms = read_atoms_from_file(xyzcoords, 'xyz')
        atoms.positions *= Bohr
        adjust_positions_by_half_cell = True
    pdbcoords = kwargs.pop('pdbcoordinates', None)
    if pdbcoords is not None:
        atoms = read_atoms_from_file(pdbcoords, 'pdb')
        pbc = atoms.pbc
        adjust_positions_by_half_cell = True
        # Due to an error in ASE pdb, we can only test the nonperiodic case.
        # atoms.cell *= Bohr # XXX cell?  Not in nonperiodic case...
        atoms.positions *= Bohr
        if sum(atoms.pbc) != 0:
            raise NotImplementedError('Periodic pdb not supported by ASE.')

    if cell is None:
        # cell could not be established from the file, so we set it on the
        # Atoms now if possible:
        cell, kwargs = kwargs2cell(kwargs)
        if cell is not None:
            cell *= Bohr
        if cell is not None and atoms is not None:
            atoms.cell = cell
        # In case of boxshape = sphere and similar, we still do not have
        # a cell.

    ndims = int(kwargs.get('dimensions', 3))
    if ndims != 3:
        raise NotImplementedError('Only 3D calculations supported.')

    coords = kwargs.get('coordinates')
    if coords is not None:
        numbers, pos, tags, info = get_positions_from_block('coordinates')
        pos *= Bohr
        adjust_positions_by_half_cell = True
        atoms = Atoms(cell=cell, numbers=numbers, positions=pos,
                      tags=tags, info=info)
    rcoords = kwargs.get('reducedcoordinates')
    if rcoords is not None:
        numbers, spos, tags, info = get_positions_from_block(
            'reducedcoordinates')
        if cell is None:
            raise ValueError('Cannot figure out what the cell is, '
                             'and thus cannot interpret reduced coordinates.')
        atoms = Atoms(cell=cell, numbers=numbers, scaled_positions=spos,
                      tags=tags, info=info)
    if atoms is None:
        raise OctopusParseError('Apparently there are no atoms.')

    # Either we have non-periodic BCs or the atoms object already
    # got its BCs from reading the file.  In the latter case
    # we shall override only if PeriodicDimensions was given specifically:

    if pbc is None:
        pdims = int(kwargs.pop('periodicdimensions', 0))
        pbc = np.zeros(3, dtype=bool)
        pbc[:pdims] = True
        atoms.pbc = pbc

    if (cell is not None and cell.shape == (3,)
            and adjust_positions_by_half_cell):
        nonpbc = (atoms.pbc == 0)
        atoms.positions[:, nonpbc] += np.array(cell)[None, nonpbc] / 2.0

    return atoms, kwargs


def generate_input(atoms, kwargs):
    """Convert atoms and keyword arguments to Octopus input file."""
    _lines = []

    kwargs = {key.lower(): value for key, value in kwargs.items()}

    def append(line):
        _lines.append(line)

    def extend(lines):
        _lines.extend(lines)
        append('')

    def setvar(key, var):
        append(f'{key} = {var}')

    for kw in ['lsize', 'latticevectors', 'latticeparameters']:
        assert kw not in kwargs

    defaultboxshape = 'parallelepiped' if atoms.cell.rank > 0 else 'minimum'
    boxshape = kwargs.pop('boxshape', defaultboxshape).lower()
    use_ase_cell = (boxshape == 'parallelepiped')
    atomskwargs = atoms2kwargs(atoms, use_ase_cell)

    setvar('boxshape', boxshape)

    if use_ase_cell:
        if 'reducedcoordinates' in atomskwargs:
            extend(list2block('LatticeParameters', [[1., 1., 1.]]))
            block = list2block('LatticeVectors', atomskwargs['latticevectors'])
        else:
            assert 'lsize' in atomskwargs
            block = list2block('LSize', atomskwargs['lsize'])

        extend(block)

    # Allow override or issue errors?
    pdim = 'periodicdimensions'
    if pdim in kwargs:
        if int(kwargs[pdim]) != int(atomskwargs[pdim]):
            raise ValueError('Cannot reconcile periodicity in input '
                             'with that of Atoms object')
    setvar('periodicdimensions', atomskwargs[pdim])

    # We should say that we want the forces if the user requests forces.
    # However the Output keyword has changed between Octopus 10.1 and 11.4.
    # We'll leave it to the user to manually set keywords correctly.
    # The most complicated part is that the user probably needs to tell
    # Octopus to save the forces in xcrysden format in order for us to
    # parse them.

    for key, val in kwargs.items():
        # Most datatypes are straightforward but blocks require some attention.
        if isinstance(val, list):
            append('')
            dict_data = list2block(key, val)
            extend(dict_data)
        else:
            setvar(key, str(val))
    append('')

    if 'reducedcoordinates' in atomskwargs:
        coord_block = list2block('ReducedCoordinates',
                                 atomskwargs['reducedcoordinates'])
    else:
        coord_block = list2block('Coordinates', atomskwargs['coordinates'])
    extend(coord_block)
    return '\n'.join(_lines)


def atoms2kwargs(atoms, use_ase_cell):
    kwargs = {}

    if any(atoms.pbc):
        coordtype = 'reducedcoordinates'
        coords = atoms.get_scaled_positions(wrap=False)
    else:
        coordtype = 'coordinates'
        coords = atoms.positions / Bohr

    if use_ase_cell:
        cell = atoms.cell / Bohr
        if coordtype == 'coordinates':
            cell_offset = 0.5 * cell.sum(axis=0)
            coords -= cell_offset
        else:
            assert coordtype == 'reducedcoordinates'

        if atoms.cell.orthorhombic:
            Lsize = 0.5 * np.diag(cell)
            kwargs['lsize'] = [[str(size) for size in Lsize]]
            # ASE uses (0...cell) while Octopus uses -L/2...L/2.
            # Lsize is really cell / 2, and we have to adjust our
            # positions by subtracting Lsize (see construction of the coords
            # block) in non-periodic directions.
        else:
            kwargs['latticevectors'] = cell.tolist()

    types = atoms.info.get('types', {})

    coord_block = []
    for sym, pos, tag in zip(atoms.symbols, coords, atoms.get_tags()):
        if sym == 'X':
            sym = types.get((sym, tag))
            if sym is None:
                raise ValueError('Cannot represent atom X without tags and '
                                 'species info in atoms.info')
        coord_block.append([repr(sym)] + [str(x) for x in pos])

    kwargs[coordtype] = coord_block
    npbc = sum(atoms.pbc)
    for c in range(npbc):
        if not atoms.pbc[c]:
            msg = ('Boundary conditions of Atoms object inconsistent '
                   'with requirements of Octopus.  pbc must be either '
                   '000, 100, 110, or 111.')
            raise ValueError(msg)
    kwargs['periodicdimensions'] = npbc

    # TODO InitialSpins
    #
    # TODO can use maximumiterations + output/outputformat to extract
    # things from restart file into output files without trouble.
    #
    # Velocities etc.?
    return kwargs


@reader
def read_octopus_in(fd, get_kwargs=False):
    kwargs = parse_input_file(fd)

    # input files may contain internal references to other files such
    # as xyz or xsf.  We need to know the directory where the file
    # resides in order to locate those.  If fd is a real file
    # object, it contains the path and we can use it.  Else assume
    # pwd.
    #
    # Maybe this is ugly; maybe it can lead to strange bugs if someone
    # wants a non-standard file-like type.  But it's probably better than
    # failing 'ase gui somedir/inp'
    try:
        fname = fd.name
    except AttributeError:
        directory = None
    else:
        directory = os.path.split(fname)[0]

    atoms, remaining_kwargs = kwargs2atoms(kwargs, directory=directory)
    if get_kwargs:
        return atoms, remaining_kwargs
    else:
        return atoms
