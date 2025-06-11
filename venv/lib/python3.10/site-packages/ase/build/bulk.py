# fmt: off

"""Build crystalline systems"""
from math import sqrt
from typing import Any

from ase.atoms import Atoms
from ase.data import atomic_numbers, chemical_symbols, reference_states
from ase.symbols import string2symbols
from ase.utils import plural


def incompatible_cell(*, want, have):
    return RuntimeError(f'Cannot create {want} cell for {have} structure')


def bulk(
    name: str,
    crystalstructure: str = None,
    a: float = None,
    b: float = None,
    c: float = None,
    *,
    alpha: float = None,
    covera: float = None,
    u: float = None,
    orthorhombic: bool = False,
    cubic: bool = False,
    basis=None,
) -> Atoms:
    """Creating bulk systems.

    Crystal structure and lattice constant(s) will be guessed if not
    provided.

    name: str
        Chemical symbol or symbols as in 'MgO' or 'NaCl'.
    crystalstructure: str
        Must be one of sc, fcc, bcc, tetragonal, bct, hcp, rhombohedral,
        orthorhombic, mcl, diamond, zincblende, rocksalt, cesiumchloride,
        fluorite or wurtzite.
    a: float
        Lattice constant.
    b: float
        Lattice constant.  If only a and b is given, b will be interpreted
        as c instead.
    c: float
        Lattice constant.
    alpha: float
        Angle in degrees for rhombohedral lattice.
    covera: float
        c/a ratio used for hcp.  Default is ideal ratio: sqrt(8/3).
    u: float
        Internal coordinate for Wurtzite structure.
    orthorhombic: bool
        Construct orthorhombic unit cell instead of primitive cell
        which is the default.
    cubic: bool
        Construct cubic unit cell if possible.
    """

    if c is None and b is not None:
        # If user passes (a, b) positionally, we want it as (a, c) instead:
        c, b = b, c

    if covera is not None and c is not None:
        raise ValueError("Don't specify both c and c/a!")

    xref = ''
    ref: Any = {}

    if name in chemical_symbols:  # single element
        atomic_number = atomic_numbers[name]
        ref = reference_states[atomic_number]
        if ref is None:
            ref = {}  # easier to 'get' things from empty dictionary than None
        else:
            xref = ref['symmetry']

        if crystalstructure is None:
            # `ref` requires `basis` but not given and not pre-defined
            if basis is None and 'basis' in ref and ref['basis'] is None:
                raise ValueError('This structure requires an atomic basis')
            if xref == 'cubic':
                # P and Mn are listed as 'cubic' but the lattice constants
                # are 7 and 9.  They must be something other than simple cubic
                # then. We used to just return the cubic one but that must
                # have been wrong somehow.  --askhl
                raise ValueError(
                    f'The reference structure of {name} is not implemented')

    # Mapping of name to number of atoms in primitive cell.
    structures = {'sc': 1, 'fcc': 1, 'bcc': 1,
                  'tetragonal': 1,
                  'bct': 1,
                  'hcp': 1,
                  'rhombohedral': 1,
                  'orthorhombic': 1,
                  'mcl': 1,
                  'diamond': 1,
                  'zincblende': 2, 'rocksalt': 2, 'cesiumchloride': 2,
                  'fluorite': 3, 'wurtzite': 2}

    if crystalstructure is None:
        crystalstructure = xref
        if crystalstructure not in structures:
            raise ValueError(f'No suitable reference data for bulk {name}.'
                             f'  Reference data: {ref}')

    magmom_per_atom = None
    if crystalstructure == xref:
        magmom_per_atom = ref.get('magmom_per_atom')

    if crystalstructure not in structures:
        raise ValueError(f'Unknown structure: {crystalstructure}.')

    # Check name:
    natoms = len(string2symbols(name))
    natoms0 = structures[crystalstructure]
    if natoms != natoms0:
        raise ValueError('Please specify {} for {} and not {}'
                         .format(plural(natoms0, 'atom'),
                                 crystalstructure, natoms))

    if alpha is None:
        alpha = ref.get('alpha')

    if a is None:
        if xref != crystalstructure:
            raise ValueError('You need to specify the lattice constant.')
        if 'a' in ref:
            a = ref['a']
        else:
            raise KeyError(f'No reference lattice parameter "a" for "{name}"')

    if b is None:
        bovera = ref.get('b/a')
        if bovera is not None and a is not None:
            b = bovera * a

    if crystalstructure in ['hcp', 'wurtzite']:
        if c is not None:
            covera = c / a
        elif covera is None:
            if xref == crystalstructure:
                covera = ref['c/a']
            else:
                covera = sqrt(8 / 3)

    if covera is None:
        covera = ref.get('c/a')
        if c is None and covera is not None:
            c = covera * a

    if crystalstructure == 'bct':
        from ase.lattice import BCT
        if basis is None:
            basis = ref.get('basis')
        if basis is not None:
            natoms = len(basis)
        lat = BCT(a=a, c=c)
        atoms = Atoms([name] * natoms, cell=lat.tocell(), pbc=True,
                      scaled_positions=basis)
    elif crystalstructure == 'rhombohedral':
        atoms = _build_rhl(name, a, alpha, basis)
    elif crystalstructure == 'orthorhombic':
        atoms = Atoms(name, cell=[a, b, c], pbc=True)
    elif orthorhombic:
        atoms = _orthorhombic_bulk(name, crystalstructure, a, covera, u)
    elif cubic:
        atoms = _cubic_bulk(name, crystalstructure, a)
    else:
        atoms = _primitive_bulk(name, crystalstructure, a, covera, u)

    if magmom_per_atom is not None:
        magmoms = [magmom_per_atom] * len(atoms)
        atoms.set_initial_magnetic_moments(magmoms)

    if cubic or orthorhombic:
        assert atoms.cell.orthorhombic

    return atoms


def _build_rhl(name, a, alpha, basis):
    from ase.lattice import RHL
    lat = RHL(a, alpha)
    cell = lat.tocell()
    if basis is None:
        # RHL: Given by A&M as scaled coordinates "x" of cell.sum(0):
        basis_x = reference_states[atomic_numbers[name]]['basis_x']
        basis = basis_x[:, None].repeat(3, axis=1)
    natoms = len(basis)
    return Atoms([name] * natoms, cell=cell, scaled_positions=basis, pbc=True)


def _orthorhombic_bulk(name, crystalstructure, a, covera=None, u=None):
    if crystalstructure in ('sc', 'bcc', 'cesiumchloride'):
        atoms = _cubic_bulk(name, crystalstructure, a)
    elif crystalstructure == 'fcc':
        b = a / sqrt(2)
        cell = (b, b, a)
        scaled_positions = ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5))
        atoms = Atoms(2 * name, cell=cell, scaled_positions=scaled_positions)
    elif crystalstructure == 'hcp':
        cell = (a, a * sqrt(3), covera * a)
        scaled_positions = [
            (0.0, 0 / 6, 0.0),
            (0.5, 3 / 6, 0.0),
            (0.5, 1 / 6, 0.5),
            (0.0, 4 / 6, 0.5),
        ]
        atoms = Atoms(4 * name, cell=cell, scaled_positions=scaled_positions)
    elif crystalstructure == 'diamond':
        b = a / sqrt(2)
        cell = (b, b, a)
        scaled_positions = [
            (0.0, 0.0, 0.0), (0.5, 0.0, 0.25),
            (0.5, 0.5, 0.5), (0.0, 0.5, 0.75),
        ]
        atoms = Atoms(4 * name, cell=cell, scaled_positions=scaled_positions)
    elif crystalstructure == 'rocksalt':
        b = a / sqrt(2)
        cell = (b, b, a)
        scaled_positions = [
            (0.0, 0.0, 0.0), (0.5, 0.5, 0.0),
            (0.5, 0.5, 0.5), (0.0, 0.0, 0.5),
        ]
        atoms = Atoms(2 * name, cell=cell, scaled_positions=scaled_positions)
    elif crystalstructure == 'zincblende':
        symbol0, symbol1 = string2symbols(name)
        atoms = _orthorhombic_bulk(symbol0, 'diamond', a)
        atoms.symbols[[1, 3]] = symbol1
    elif crystalstructure == 'wurtzite':
        cell = (a, a * sqrt(3), covera * a)
        u = u or 0.25 + 1 / 3 / covera**2
        scaled_positions = [
            (0.0, 0 / 6, 0.0), (0.0, 2 / 6, 0.5 - u),
            (0.0, 2 / 6, 0.5), (0.0, 0 / 6, 1.0 - u),
            (0.5, 3 / 6, 0.0), (0.5, 5 / 6, 0.5 - u),
            (0.5, 5 / 6, 0.5), (0.5, 3 / 6, 1.0 - u),
        ]
        atoms = Atoms(4 * name, cell=cell, scaled_positions=scaled_positions)
    else:
        raise incompatible_cell(want='orthorhombic', have=crystalstructure)

    atoms.pbc = True

    return atoms


def _cubic_bulk(name: str, crystalstructure: str, a: float) -> Atoms:
    cell = (a, a, a)
    if crystalstructure == 'sc':
        atoms = Atoms(name, cell=cell)
    elif crystalstructure == 'fcc':
        scaled_positions = [
            (0.0, 0.0, 0.0),
            (0.0, 0.5, 0.5),
            (0.5, 0.0, 0.5),
            (0.5, 0.5, 0.0),
        ]
        atoms = Atoms(4 * name, cell=cell, scaled_positions=scaled_positions)
    elif crystalstructure == 'bcc':
        scaled_positions = [
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.5),
        ]
        atoms = Atoms(2 * name, cell=cell, scaled_positions=scaled_positions)
    elif crystalstructure == 'diamond':
        scaled_positions = [
            (0.0, 0.0, 0.0), (0.25, 0.25, 0.25),
            (0.0, 0.5, 0.5), (0.25, 0.75, 0.75),
            (0.5, 0.0, 0.5), (0.75, 0.25, 0.75),
            (0.5, 0.5, 0.0), (0.75, 0.75, 0.25),
        ]
        atoms = Atoms(8 * name, cell=cell, scaled_positions=scaled_positions)
    elif crystalstructure == 'cesiumchloride':
        symbol0, symbol1 = string2symbols(name)
        atoms = _cubic_bulk(symbol0, 'bcc', a)
        atoms.symbols[[1]] = symbol1
    elif crystalstructure == 'zincblende':
        symbol0, symbol1 = string2symbols(name)
        atoms = _cubic_bulk(symbol0, 'diamond', a)
        atoms.symbols[[1, 3, 5, 7]] = symbol1
    elif crystalstructure == 'rocksalt':
        scaled_positions = [
            (0.0, 0.0, 0.0), (0.5, 0.0, 0.0),
            (0.0, 0.5, 0.5), (0.5, 0.5, 0.5),
            (0.5, 0.0, 0.5), (0.0, 0.0, 0.5),
            (0.5, 0.5, 0.0), (0.0, 0.5, 0.0),
        ]
        atoms = Atoms(4 * name, cell=cell, scaled_positions=scaled_positions)
    elif crystalstructure == 'fluorite':
        scaled_positions = [
            (0.00, 0.00, 0.00), (0.25, 0.25, 0.25), (0.75, 0.75, 0.75),
            (0.00, 0.50, 0.50), (0.25, 0.75, 0.75), (0.75, 0.25, 0.25),
            (0.50, 0.00, 0.50), (0.75, 0.25, 0.75), (0.25, 0.75, 0.25),
            (0.50, 0.50, 0.00), (0.75, 0.75, 0.25), (0.25, 0.25, 0.75),
        ]
        atoms = Atoms(4 * name, cell=cell, scaled_positions=scaled_positions)
    else:
        raise incompatible_cell(want='cubic', have=crystalstructure)

    atoms.pbc = True

    return atoms


def _primitive_bulk(name, crystalstructure, a, covera=None, u=None):
    if crystalstructure == 'sc':
        atoms = Atoms(name, cell=(a, a, a))
    elif crystalstructure == 'fcc':
        b = 0.5 * a
        cell = ((0, b, b), (b, 0, b), (b, b, 0))
        atoms = Atoms(name, cell=cell)
    elif crystalstructure == 'bcc':
        b = 0.5 * a
        cell = ((-b, b, b), (b, -b, b), (b, b, -b))
        atoms = Atoms(name, cell=cell)
    elif crystalstructure == 'hcp':
        c = covera * a
        cell = ((a, 0, 0), (-0.5 * a, 0.5 * sqrt(3) * a, 0), (0, 0, c))
        scaled_positions = [
            (0 / 3, 0 / 3, 0.0),
            (1 / 3, 2 / 3, 0.5),
        ]
        atoms = Atoms(2 * name, cell=cell, scaled_positions=scaled_positions)
    elif crystalstructure == 'diamond':
        atoms = \
            _primitive_bulk(name, 'fcc', a) + \
            _primitive_bulk(name, 'fcc', a)
        atoms.positions[1, :] += 0.25 * a
    elif crystalstructure == 'rocksalt':
        symbol0, symbol1 = string2symbols(name)
        atoms = \
            _primitive_bulk(symbol0, 'fcc', a) + \
            _primitive_bulk(symbol1, 'fcc', a)
        atoms.positions[1, 0] += 0.5 * a
    elif crystalstructure == 'cesiumchloride':
        symbol0, symbol1 = string2symbols(name)
        atoms = \
            _primitive_bulk(symbol0, 'sc', a) + \
            _primitive_bulk(symbol1, 'sc', a)
        atoms.positions[1, :] += 0.5 * a
    elif crystalstructure == 'zincblende':
        symbol0, symbol1 = string2symbols(name)
        atoms = \
            _primitive_bulk(symbol0, 'fcc', a) + \
            _primitive_bulk(symbol1, 'fcc', a)
        atoms.positions[1, :] += 0.25 * a
    elif crystalstructure == 'fluorite':
        symbol0, symbol1, symbol2 = string2symbols(name)
        atoms = \
            _primitive_bulk(symbol0, 'fcc', a) + \
            _primitive_bulk(symbol1, 'fcc', a) + \
            _primitive_bulk(symbol2, 'fcc', a)
        atoms.positions[1, :] += 0.25 * a
        atoms.positions[2, :] += 0.75 * a
    elif crystalstructure == 'wurtzite':
        c = covera * a
        cell = ((a, 0, 0), (-0.5 * a, 0.5 * sqrt(3) * a, 0), (0, 0, c))
        u = u or 0.25 + 1 / 3 / covera**2
        scaled_positions = [
            (0 / 3, 0 / 3, 0.0), (1 / 3, 2 / 3, 0.5 - u),
            (1 / 3, 2 / 3, 0.5), (0 / 3, 0 / 3, 1.0 - u),
        ]
        atoms = Atoms(2 * name, cell=cell, scaled_positions=scaled_positions)
    else:
        raise incompatible_cell(want='primitive', have=crystalstructure)

    atoms.pbc = True

    return atoms
