# fmt: off

import uuid
from typing import Any, Dict

import numpy as np

from ase import Atoms
from ase.calculators.calculator import (
    PropertyNotImplementedError,
    all_properties,
    kptdensity2monkhorstpack,
)
from ase.calculators.singlepoint import SinglePointCalculator
from ase.data import atomic_masses, chemical_symbols
from ase.formula import Formula
from ase.geometry import cell_to_cellpar
from ase.io.jsonio import decode


class FancyDict(dict):
    """Dictionary with keys available as attributes also."""

    def __getattr__(self, key):
        if key not in self:
            return dict.__getattribute__(self, key)
        value = self[key]
        if isinstance(value, dict):
            return FancyDict(value)
        return value

    def __dir__(self):
        return self.keys()  # for tab-completion


def atoms2dict(atoms):
    dct = {
        'numbers': atoms.numbers,
        'positions': atoms.positions,
        'unique_id': uuid.uuid4().hex}
    if atoms.pbc.any():
        dct['pbc'] = atoms.pbc
    if atoms.cell.any():
        dct['cell'] = atoms.cell
    if atoms.has('initial_magmoms'):
        dct['initial_magmoms'] = atoms.get_initial_magnetic_moments()
    if atoms.has('initial_charges'):
        dct['initial_charges'] = atoms.get_initial_charges()
    if atoms.has('masses'):
        dct['masses'] = atoms.get_masses()
    if atoms.has('tags'):
        dct['tags'] = atoms.get_tags()
    if atoms.has('momenta'):
        dct['momenta'] = atoms.get_momenta()
    if atoms.constraints:
        dct['constraints'] = [c.todict() for c in atoms.constraints]
    if atoms.calc is not None:
        dct['calculator'] = atoms.calc.name.lower()
        dct['calculator_parameters'] = atoms.calc.todict()
        if len(atoms.calc.check_state(atoms)) == 0:
            for prop in all_properties:
                try:
                    x = atoms.calc.get_property(prop, atoms, False)
                except PropertyNotImplementedError:
                    pass
                else:
                    if x is not None:
                        dct[prop] = x
    return dct


class AtomsRow:
    mtime: float
    positions: np.ndarray
    id: int

    def __init__(self, dct):
        if isinstance(dct, dict):
            dct = dct.copy()
            if 'calculator_parameters' in dct:
                # Earlier version of ASE would encode the calculator
                # parameter dict again and again and again ...
                while isinstance(dct['calculator_parameters'], str):
                    dct['calculator_parameters'] = decode(
                        dct['calculator_parameters'])
        else:
            dct = atoms2dict(dct)
        assert 'numbers' in dct
        self._constraints = dct.pop('constraints', [])
        self._constrained_forces = None
        self._data = dct.pop('data', {})
        kvp = dct.pop('key_value_pairs', {})
        self._keys = list(kvp.keys())
        self.__dict__.update(kvp)
        self.__dict__.update(dct)
        if 'cell' not in dct:
            self.cell = np.zeros((3, 3))
        if 'pbc' not in dct:
            self.pbc = np.zeros(3, bool)

    def __contains__(self, key):
        return key in self.__dict__

    def __iter__(self):
        return (key for key in self.__dict__ if key[0] != '_')

    def get(self, key, default=None):
        """Return value of key if present or default if not."""
        return getattr(self, key, default)

    @property
    def key_value_pairs(self):
        """Return dict of key-value pairs."""
        return {key: self.get(key) for key in self._keys}

    def count_atoms(self):
        """Count atoms.

        Return dict mapping chemical symbol strings to number of atoms.
        """
        count = {}
        for symbol in self.symbols:
            count[symbol] = count.get(symbol, 0) + 1
        return count

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __str__(self):
        return '<AtomsRow: formula={}, keys={}>'.format(
            self.formula, ','.join(self._keys))

    @property
    def constraints(self):
        """List of constraints."""
        from ase.constraints import dict2constraint
        if not isinstance(self._constraints, list):
            # Lazy decoding:
            cs = decode(self._constraints)
            self._constraints = []
            for c in cs:
                # Convert to new format:
                name = c.pop('__name__', None)
                if name:
                    c = {'name': name, 'kwargs': c}
                if c['name'].startswith('ase'):
                    c['name'] = c['name'].rsplit('.', 1)[1]
                self._constraints.append(c)
        return [dict2constraint(d) for d in self._constraints]

    @property
    def data(self):
        """Data dict."""
        if isinstance(self._data, str):
            self._data = decode(self._data)  # lazy decoding
        elif isinstance(self._data, bytes):
            from ase.db.core import bytes_to_object
            self._data = bytes_to_object(self._data)  # lazy decoding
        return FancyDict(self._data)

    @property
    def natoms(self):
        """Number of atoms."""
        return len(self.numbers)

    @property
    def formula(self):
        """Chemical formula string."""
        return Formula('', _tree=[(self.symbols, 1)]).format('metal')

    @property
    def symbols(self):
        """List of chemical symbols."""
        return [chemical_symbols[Z] for Z in self.numbers]

    @property
    def fmax(self):
        """Maximum atomic force."""
        forces = self.constrained_forces
        return (forces**2).sum(1).max()**0.5

    @property
    def constrained_forces(self):
        """Forces after applying constraints."""
        if self._constrained_forces is not None:
            return self._constrained_forces
        forces = self.forces
        constraints = self.constraints
        if constraints:
            forces = forces.copy()
            atoms = self.toatoms()
            for constraint in constraints:
                constraint.adjust_forces(atoms, forces)

        self._constrained_forces = forces
        return forces

    @property
    def smax(self):
        """Maximum stress tensor component."""
        return (self.stress**2).max()**0.5

    @property
    def mass(self):
        """Total mass."""
        if 'masses' in self:
            return self.masses.sum()
        return atomic_masses[self.numbers].sum()

    @property
    def volume(self):
        """Volume of unit cell."""
        if self.cell is None:
            return None
        vol = abs(np.linalg.det(self.cell))
        if vol == 0.0:
            raise AttributeError
        return vol

    @property
    def charge(self):
        """Total charge."""
        charges = self.get('initial_charges')
        if charges is None:
            return 0.0
        return charges.sum()

    def toatoms(self,
                add_additional_information=False):
        """Create Atoms object."""
        atoms = Atoms(self.numbers,
                      self.positions,
                      cell=self.cell,
                      pbc=self.pbc,
                      magmoms=self.get('initial_magmoms'),
                      charges=self.get('initial_charges'),
                      tags=self.get('tags'),
                      masses=self.get('masses'),
                      momenta=self.get('momenta'),
                      constraint=self.constraints)

        results = {prop: self[prop] for prop in all_properties if prop in self}
        if results:
            atoms.calc = SinglePointCalculator(atoms, **results)
            atoms.calc.name = self.get('calculator', 'unknown')

        if add_additional_information:
            atoms.info = {}
            atoms.info['unique_id'] = self.unique_id
            if self._keys:
                atoms.info['key_value_pairs'] = self.key_value_pairs
            data = self.get('data')
            if data:
                atoms.info['data'] = data

        return atoms


def row2dct(row, key_descriptions) -> Dict[str, Any]:
    """Convert row to dict of things for printing or a web-page."""

    from ase.db.core import float_to_time_string, now

    dct = {}

    atoms = Atoms(cell=row.cell, pbc=row.pbc)
    dct['size'] = kptdensity2monkhorstpack(atoms,
                                           kptdensity=1.8,
                                           even=False)

    dct['cell'] = [[f'{a:.3f}' for a in axis] for axis in row.cell]
    par = [f'{x:.3f}' for x in cell_to_cellpar(row.cell)]
    dct['lengths'] = par[:3]
    dct['angles'] = par[3:]

    stress = row.get('stress')
    if stress is not None:
        dct['stress'] = ', '.join(f'{s:.3f}' for s in stress)

    dct['formula'] = Formula(row.formula).format('abc')

    dipole = row.get('dipole')
    if dipole is not None:
        dct['dipole'] = ', '.join(f'{d:.3f}' for d in dipole)

    data = row.get('data')
    if data:
        dct['data'] = ', '.join(data.keys())

    constraints = row.get('constraints')
    if constraints:
        dct['constraints'] = ', '.join(c.__class__.__name__
                                       for c in constraints)

    keys = ({'id', 'energy', 'fmax', 'smax', 'mass', 'age'} |
            set(key_descriptions) |
            set(row.key_value_pairs))
    dct['table'] = []

    from ase.db.project import KeyDescription
    for key in keys:
        if key == 'age':
            age = float_to_time_string(now() - row.ctime, True)
            dct['table'].append(('ctime', 'Age', age))
            continue
        value = row.get(key)
        if value is not None:
            if isinstance(value, float):
                value = f'{value:.3f}'
            elif not isinstance(value, str):
                value = str(value)

            nokeydesc = KeyDescription(key, '', '', '')
            keydesc = key_descriptions.get(key, nokeydesc)
            unit = keydesc.unit
            if unit:
                value += ' ' + unit
            dct['table'].append((key, keydesc.longdesc, value))

    return dct
