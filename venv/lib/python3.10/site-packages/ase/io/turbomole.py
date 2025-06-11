# fmt: off

from ase.units import Bohr


def read_turbomole(fd):
    """Method to read turbomole coord file

    coords in bohr, atom types in lowercase, format:
    $coord
    x y z atomtype
    x y z atomtype f
    $end
    Above 'f' means a fixed atom.
    """
    from ase import Atoms
    from ase.constraints import FixAtoms

    lines = fd.readlines()
    atoms_pos = []
    atom_symbols = []
    myconstraints = []

    # find $coord section;
    # does not necessarily have to be the first $<something> in file...
    for i, l in enumerate(lines):
        if l.strip().startswith('$coord'):
            start = i
            break
    for line in lines[start + 1:]:
        if line.startswith('$'):  # start of new section
            break
        else:
            x, y, z, symbolraw = line.split()[:4]
            symbolshort = symbolraw.strip()
            symbol = symbolshort[0].upper() + symbolshort[1:].lower()
            # print(symbol)
            atom_symbols.append(symbol)
            atoms_pos.append(
                [float(x) * Bohr, float(y) * Bohr, float(z) * Bohr]
            )
            cols = line.split()
            if (len(cols) == 5):
                fixedstr = line.split()[4].strip()
                if (fixedstr == "f"):
                    myconstraints.append(True)
                else:
                    myconstraints.append(False)
            else:
                myconstraints.append(False)

    # convert Turbomole ghost atom Q to X
    atom_symbols = [element if element !=
                    'Q' else 'X' for element in atom_symbols]
    atoms = Atoms(positions=atoms_pos, symbols=atom_symbols, pbc=False)
    c = FixAtoms(mask=myconstraints)
    atoms.set_constraint(c)
    return atoms


class TurbomoleFormatError(ValueError):
    default_message = ('Data format in file does not correspond to known '
                       'Turbomole gradient format')

    def __init__(self, *args, **kwargs):
        if args or kwargs:
            ValueError.__init__(self, *args, **kwargs)
        else:
            ValueError.__init__(self, self.default_message)


def read_turbomole_gradient(fd, index=-1):
    """ Method to read turbomole gradient file """

    # read entire file
    lines = [x.strip() for x in fd.readlines()]

    # find $grad section
    start = end = -1
    for i, line in enumerate(lines):
        if not line.startswith('$'):
            continue
        if line.split()[0] == '$grad':
            start = i
        elif start >= 0:
            end = i
            break

    if end <= start:
        raise RuntimeError('File does not contain a valid \'$grad\' section')

    # trim lines to $grad
    del lines[:start + 1]
    del lines[end - 1 - start:]

    # Interpret $grad section
    from ase import Atom, Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.units import Bohr, Hartree
    images = []
    while lines:  # loop over optimization cycles
        # header line
        # cycle =      1    SCF energy =     -267.6666811409   |dE/dxyz| =  0.157112  # noqa: E501
        fields = lines[0].split('=')
        try:
            # cycle = int(fields[1].split()[0])
            energy = float(fields[2].split()[0]) * Hartree
            # gradient = float(fields[3].split()[0])
        except (IndexError, ValueError) as e:
            raise TurbomoleFormatError from e

        # coordinates/gradient
        atoms = Atoms()
        forces = []
        for line in lines[1:]:
            fields = line.split()
            if len(fields) == 4:  # coordinates
                # 0.00000000000000      0.00000000000000      0.00000000000000      c  # noqa: E501
                try:
                    symbol = fields[3].lower().capitalize()
                    # if dummy atom specified, substitute 'Q' with 'X'
                    if symbol == 'Q':
                        symbol = 'X'
                    position = tuple(Bohr * float(x) for x in fields[0:3])
                except ValueError as e:
                    raise TurbomoleFormatError from e
                atoms.append(Atom(symbol, position))
            elif len(fields) == 3:  # gradients
                #  -.51654903354681D-07  -.51654903206651D-07  0.51654903169644D-07  # noqa: E501
                grad = []
                for val in fields[:3]:
                    try:
                        grad.append(
                            -float(val.replace('D', 'E')) * Hartree / Bohr
                        )
                    except ValueError as e:
                        raise TurbomoleFormatError from e
                forces.append(grad)
            else:  # next cycle
                break

        # calculator
        calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
        atoms.calc = calc

        # save frame
        images.append(atoms)

        # delete this frame from data to be handled
        del lines[:2 * len(atoms) + 1]

    return images[index]


def write_turbomole(fd, atoms):
    """ Method to write turbomole coord file
    """
    from ase.constraints import FixAtoms

    coord = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()

    # convert X to Q for Turbomole ghost atoms
    symbols = [element if element != 'X' else 'Q' for element in symbols]

    fix_indices = set()
    if atoms.constraints:
        for constr in atoms.constraints:
            if isinstance(constr, FixAtoms):
                fix_indices.update(constr.get_indices())

    fix_str = []
    for i in range(len(atoms)):
        if i in fix_indices:
            fix_str.append('f')
        else:
            fix_str.append('')

    fd.write('$coord\n')
    for (x, y, z), s, fix in zip(coord, symbols, fix_str):
        fd.write('%20.14f  %20.14f  %20.14f      %2s  %2s \n'
                 % (x / Bohr, y / Bohr, z / Bohr, s.lower(), fix))

    fd.write('$end\n')
