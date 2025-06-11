# fmt: off

"""Helper functions for read_fdf."""
from pathlib import Path
from re import compile

import numpy as np

from ase import Atoms
from ase.units import Bohr
from ase.utils import reader

_label_strip_re = compile(r'[\s._-]')


def _labelize(raw_label):
    # Labels are case insensitive and -_. should be ignored, lower and strip it
    return _label_strip_re.sub('', raw_label).lower()


def _is_block(val):
    # Tell whether value is a block-value or an ordinary value.
    # A block is represented as a list of lists of strings,
    # and a ordinary value is represented as a list of strings
    if isinstance(val, list) and \
       len(val) > 0 and \
       isinstance(val[0], list):
        return True
    return False


def _get_stripped_lines(fd):
    # Remove comments, leading blanks, and empty lines
    return [_f for _f in [L.split('#')[0].strip() for L in fd] if _f]


@reader
def _read_fdf_lines(file):
    # Read lines and resolve includes
    lbz = _labelize

    lines = []
    for L in _get_stripped_lines(file):
        w0 = lbz(L.split(None, 1)[0])

        if w0 == '%include':
            # Include the contents of fname
            fname = L.split(None, 1)[1].strip()
            parent_fname = getattr(file, 'name', None)
            if isinstance(parent_fname, str):
                fname = Path(parent_fname).parent / fname
            lines += _read_fdf_lines(fname)

        elif '<' in L:
            L, fname = L.split('<', 1)
            w = L.split()
            fname = fname.strip()

            if w0 == '%block':
                # "%block label < filename" means that the block contents
                # should be read from filename
                if len(w) != 2:
                    raise OSError('Bad %%block-statement "%s < %s"' %
                                  (L, fname))
                label = lbz(w[1])
                lines.append('%%block %s' % label)
                lines += _get_stripped_lines(open(fname))
                lines.append('%%endblock %s' % label)
            else:
                # "label < filename.fdf" means that the label
                # (_only_ that label) is to be resolved from filename.fdf
                label = lbz(w[0])
                fdf = read_fdf(fname)
                if label in fdf:
                    if _is_block(fdf[label]):
                        lines.append('%%block %s' % label)
                        lines += [' '.join(x) for x in fdf[label]]
                        lines.append('%%endblock %s' % label)
                    else:
                        lines.append('{} {}'.format(
                            label, ' '.join(fdf[label])))
                # else:
                #    label unresolved!
                #    One should possibly issue a warning about this!
        else:
            # Simple include line L
            lines.append(L)
    return lines


def read_fdf(fname):
    """Read a siesta style fdf-file.

    The data is returned as a dictionary
    ( label:value ).

    All labels are converted to lower case characters and
    are stripped of any '-', '_', or '.'.

    Ordinary values are stored as a list of strings (splitted on WS),
    and block values are stored as list of lists of strings
    (splitted per line, and on WS).
    If a label occurres more than once, the first occurrence
    takes precedence.

    The implementation applies no intelligence, and does not
    "understand" the data or the concept of units etc.
    Values are never parsed in any way, just stored as
    split strings.

    The implementation tries to comply with the fdf-format
    specification as presented in the siesta 2.0.2 manual.

    An fdf-dictionary could e.g. look like this::

        {'atomiccoordinatesandatomicspecies': [
              ['4.9999998', '5.7632392', '5.6095972', '1'],
              ['5.0000000', '6.5518100', '4.9929091', '2'],
              ['5.0000000', '4.9746683', '4.9929095', '2']],
         'atomiccoordinatesformat': ['Ang'],
         'chemicalspecieslabel': [['1', '8', 'O'],
                                  ['2', '1', 'H']],
         'dmmixingweight': ['0.1'],
         'dmnumberpulay': ['5'],
         'dmusesavedm': ['True'],
         'latticeconstant': ['1.000000', 'Ang'],
         'latticevectors': [
              ['10.00000000', '0.00000000', '0.00000000'],
              ['0.00000000', '11.52647800', '0.00000000'],
              ['0.00000000', '0.00000000', '10.59630900']],
         'maxscfiterations': ['120'],
         'meshcutoff': ['2721.139566', 'eV'],
         'numberofatoms': ['3'],
         'numberofspecies': ['2'],
         'paobasissize': ['dz'],
         'solutionmethod': ['diagon'],
         'systemlabel': ['H2O'],
         'wavefunckpoints': [['0.0', '0.0', '0.0']],
         'writedenchar': ['T'],
         'xcauthors': ['PBE'],
         'xcfunctional': ['GGA']}

    """
    fdf = {}
    lbz = _labelize
    lines = _read_fdf_lines(fname)
    while lines:
        w = lines.pop(0).split(None, 1)
        if lbz(w[0]) == '%block':
            # Block value
            if len(w) == 2:
                label = lbz(w[1])
                content = []
                while True:
                    if len(lines) == 0:
                        raise OSError('Unexpected EOF reached in %s, '
                                      'un-ended block %s' % (fname, label))
                    w = lines.pop(0).split()
                    if lbz(w[0]) == '%endblock':
                        break
                    content.append(w)

                if label not in fdf:
                    # Only first appearance of label is to be used
                    fdf[label] = content
            else:
                raise OSError('%%block statement without label')
        else:
            # Ordinary value
            label = lbz(w[0])
            if len(w) == 1:
                # Siesta interpret blanks as True for logical variables
                fdf[label] = []
            else:
                fdf[label] = w[1].split()
    return fdf


def read_struct_out(fd):
    """Read a siesta struct file"""

    cell = []
    for _ in range(3):
        line = next(fd)
        v = np.array(line.split(), float)
        cell.append(v)

    natoms = int(next(fd))

    numbers = np.empty(natoms, int)
    scaled_positions = np.empty((natoms, 3))
    for i, line in enumerate(fd):
        tokens = line.split()
        numbers[i] = int(tokens[1])
        scaled_positions[i] = np.array(tokens[2:5], float)

    return Atoms(numbers,
                 cell=cell,
                 pbc=True,
                 scaled_positions=scaled_positions)


def read_siesta_xv(fd):
    vectors = []
    for _ in range(3):
        data = next(fd).split()
        vectors.append([float(data[j]) * Bohr for j in range(3)])

    # Read number of atoms (line 4)
    natoms = int(next(fd).split()[0])

    # Read remaining lines
    speciesnumber, atomnumbers, xyz, V = [], [], [], []
    for line in fd:
        if len(line) > 5:  # Ignore blank lines
            data = line.split()
            speciesnumber.append(int(data[0]))
            atomnumbers.append(int(data[1]))
            xyz.append([float(data[2 + j]) * Bohr for j in range(3)])
            V.append([float(data[5 + j]) * Bohr for j in range(3)])

    vectors = np.array(vectors)
    atomnumbers = np.array(atomnumbers)
    xyz = np.array(xyz)
    atoms = Atoms(numbers=atomnumbers, positions=xyz, cell=vectors,
                  pbc=True)
    assert natoms == len(atoms)
    return atoms
