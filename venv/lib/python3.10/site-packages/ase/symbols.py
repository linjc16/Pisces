# fmt: off

import collections.abc
import numbers
import warnings
from typing import Dict, Iterator, List, Sequence, Set, Union

import numpy as np

from ase.data import atomic_numbers, chemical_symbols
from ase.formula import Formula

Integers = Union[Sequence[int], np.ndarray]


def string2symbols(s: str) -> List[str]:
    """Convert string to list of chemical symbols."""
    return list(Formula(s))


def symbols2numbers(symbols) -> List[int]:
    if isinstance(symbols, str):
        symbols = string2symbols(symbols)
    numbers = []
    for s in symbols:
        if isinstance(s, str):
            numbers.append(atomic_numbers[s])
        else:
            numbers.append(int(s))
    return numbers


class Symbols(collections.abc.Sequence):
    """A sequence of chemical symbols.

    ``atoms.symbols`` is a :class:`ase.symbols.Symbols` object.  This
    object works like an editable view of ``atoms.numbers``, except
    its elements are manipulated as strings.

    Examples:

    >>> from ase.build import molecule
    >>> atoms = molecule('CH3CH2OH')
    >>> atoms.symbols
    Symbols('C2OH6')
    >>> atoms.symbols[:3]
    Symbols('C2O')
    >>> atoms.symbols == 'H'  # doctest: +ELLIPSIS
    array([False, False, False,  True,  True,  True,  True,  True,  True]...)
    >>> atoms.symbols[-3:] = 'Pu'
    >>> atoms.symbols
    Symbols('C2OH3Pu3')
    >>> atoms.symbols[3:6] = 'Mo2U'
    >>> atoms.symbols
    Symbols('C2OMo2UPu3')
    >>> atoms.symbols.formula
    Formula('C2OMo2UPu3')

    The :class:`ase.formula.Formula` object is useful for extended
    formatting options and analysis.

    """

    def __init__(self, numbers) -> None:
        self.numbers = np.asarray(numbers, int)

    @classmethod
    def fromsymbols(cls, symbols) -> 'Symbols':
        numbers = symbols2numbers(symbols)
        return cls(np.array(numbers))

    @property
    def formula(self) -> Formula:
        """Formula object."""
        string = Formula.from_list(self).format('reduce')
        return Formula(string)

    def __getitem__(self, key) -> Union['Symbols', str]:
        num = self.numbers[key]
        if isinstance(num, numbers.Integral):
            return chemical_symbols[num]
        return Symbols(num)

    def __iter__(self) -> Iterator[str]:
        for num in self.numbers:
            yield chemical_symbols[num]

    def __setitem__(self, key, value) -> None:
        numbers = symbols2numbers(value)
        if len(numbers) == 1:
            self.numbers[key] = numbers[0]
        else:
            self.numbers[key] = numbers

    def __len__(self) -> int:
        return len(self.numbers)

    def __str__(self) -> str:
        return self.get_chemical_formula('reduce')

    def __repr__(self) -> str:
        return f'Symbols(\'{self}\')'

    def __eq__(self, obj) -> bool:
        if not hasattr(obj, '__len__'):
            return False

        try:
            symbols = Symbols.fromsymbols(obj)
        except Exception:
            # Typically this would happen if obj cannot be converged to
            # atomic numbers.
            return False
        return self.numbers == symbols.numbers

    def get_chemical_formula(
            self,
            mode: str = 'hill',
            empirical: bool = False,
    ) -> str:
        """Get chemical formula.

        See documentation of ase.atoms.Atoms.get_chemical_formula()."""
        # XXX Delegate the work to the Formula object!
        if mode in ('reduce', 'all') and empirical:
            warnings.warn("Empirical chemical formula not available "
                          "for mode '{}'".format(mode))

        if len(self) == 0:
            return ''

        numbers = self.numbers

        if mode == 'reduce':
            n = len(numbers)
            changes = np.concatenate(([0], np.arange(1, n)[numbers[1:] !=
                                                           numbers[:-1]]))
            symbols = [chemical_symbols[e] for e in numbers[changes]]
            counts = np.append(changes[1:], n) - changes

            tokens = []
            for s, c in zip(symbols, counts):
                tokens.append(s)
                if c > 1:
                    tokens.append(str(c))
            formula = ''.join(tokens)
        elif mode == 'all':
            formula = ''.join([chemical_symbols[n] for n in numbers])
        else:
            symbols = [chemical_symbols[Z] for Z in numbers]
            f = Formula('', _tree=[(symbols, 1)])
            if empirical:
                f, _ = f.reduce()
            if mode in {'hill', 'metal'}:
                formula = f.format(mode)
            else:
                raise ValueError(
                    "Use mode = 'all', 'reduce', 'hill' or 'metal'.")

        return formula

    def search(self, symbols) -> Integers:
        """Return the indices of elements with given symbol or symbols."""
        numbers = set(symbols2numbers(symbols))
        indices = [i for i, number in enumerate(self.numbers)
                   if number in numbers]
        return np.array(indices, int)

    def species(self) -> Set[str]:
        """Return unique symbols as a set."""
        return set(self)

    def indices(self) -> Dict[str, Integers]:
        """Return dictionary mapping each unique symbol to indices.

        >>> from ase.build import molecule
        >>> atoms = molecule('CH3CH2OH')
        >>> atoms.symbols.indices()
        {'C': array([0, 1]), 'O': array([2]), 'H': array([3, 4, 5, 6, 7, 8])}

        """
        dct: Dict[str, List[int]] = {}
        for i, symbol in enumerate(self):
            dct.setdefault(symbol, []).append(i)
        return {key: np.array(value, int) for key, value in dct.items()}

    def species_indices(self) -> Sequence[int]:
        """Return the indices of each atom within their individual species.

        >>> from ase import Atoms
        >>> atoms = Atoms('CH3CH2OH')
        >>> atoms.symbols.species_indices()
        [0, 0, 1, 2, 1, 3, 4, 0, 5]

         ^  ^  ^  ^  ^  ^  ^  ^  ^
         C  H  H  H  C  H  H  O  H

        """

        counts: Dict[str, int] = {}
        result = []
        for i, n in enumerate(self.numbers):
            counts[n] = counts.get(n, -1) + 1
            result.append(counts[n])

        return result
