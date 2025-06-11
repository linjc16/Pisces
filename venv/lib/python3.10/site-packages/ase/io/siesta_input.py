# fmt: off

"""SiestaInput"""
import warnings

import numpy as np

from ase import Atoms
from ase.constraints import FixAtoms, FixCartesian, FixedLine, FixedPlane


class SiestaInput:
    """SiestaInput"""
    @classmethod
    def is_along_cartesian(cls, norm_dir: np.ndarray) -> bool:
        """Return whether `norm_dir` is along a Cartesian coordidate."""
        directions = [
            [+1, 0, 0],
            [-1, 0, 0],
            [0, +1, 0],
            [0, -1, 0],
            [0, 0, +1],
            [0, 0, -1],
        ]
        for direction in directions:
            if np.allclose(norm_dir, direction, rtol=0.0, atol=1e-6):
                return True
        return False

    @classmethod
    def generate_kpts(cls, kpts):
        """Write kpts.

        Parameters:
            - f : Open filename.
        """
        yield '\n'
        yield '#KPoint grid\n'
        yield '%block kgrid_Monkhorst_Pack\n'

        for i in range(3):
            s = ''
            if i < len(kpts):
                number = kpts[i]
                displace = 0.0
            else:
                number = 1
                displace = 0
            for j in range(3):
                if j == i:
                    write_this = number
                else:
                    write_this = 0
                s += f'     {write_this:d}  '
            s += f'{displace:1.1f}\n'
            yield s
        yield '%endblock kgrid_Monkhorst_Pack\n'
        yield '\n'

    @classmethod
    def get_species(cls, atoms, species, basis_set):
        from ase.calculators.siesta.parameters import Species

        # For each element use default species from the species input, or set
        # up a default species  from the general default parameters.
        tags = atoms.get_tags()
        default_species = [
            s for s in species
            if (s['tag'] is None) and s['symbol'] in atoms.symbols]
        default_symbols = [s['symbol'] for s in default_species]
        for symbol in atoms.symbols:
            if symbol not in default_symbols:
                spec = Species(symbol=symbol,
                               basis_set=basis_set,
                               tag=None)
                default_species.append(spec)
                default_symbols.append(symbol)
        assert len(default_species) == len(set(atoms.symbols))

        # Set default species as the first species.
        species_numbers = np.zeros(len(atoms), int)
        i = 1
        for spec in default_species:
            mask = atoms.symbols == spec['symbol']
            species_numbers[mask] = i
            i += 1

        # Set up the non-default species.
        non_default_species = [s for s in species if s['tag'] is not None]
        for spec in non_default_species:
            mask1 = tags == spec['tag']
            mask2 = atoms.symbols == spec['symbol']
            mask = np.logical_and(mask1, mask2)
            if sum(mask) > 0:
                species_numbers[mask] = i
            i += 1
        all_species = default_species + non_default_species

        return all_species, species_numbers

    @classmethod
    def make_xyz_constraints(cls, atoms: Atoms):
        """ Create coordinate-resolved list of constraints [natoms, 0:3]
        The elements of the list must be integers 0 or 1
          1 -- means that the coordinate will be updated during relaxation
          0 -- mains that the coordinate will be fixed during relaxation
        """
        moved = np.ones((len(atoms), 3), dtype=int)  # (0: fixed, 1: updated)
        for const in atoms.constraints:
            if isinstance(const, FixAtoms):
                moved[const.get_indices()] = 0
            elif isinstance(const, FixedLine):
                norm_dir = const.dir / np.linalg.norm(const.dir)
                if not cls.is_along_cartesian(norm_dir):
                    raise RuntimeError(
                        f'norm_dir {norm_dir} is not one of the Cartesian axes'
                    )
                norm_dir = norm_dir.round().astype(int)
                moved[const.get_indices()] = norm_dir
            elif isinstance(const, FixedPlane):
                norm_dir = const.dir / np.linalg.norm(const.dir)
                if not cls.is_along_cartesian(norm_dir):
                    raise RuntimeError(
                        f'norm_dir {norm_dir} is not one of the Cartesian axes'
                    )
                norm_dir = norm_dir.round().astype(int)
                moved[const.get_indices()] = abs(1 - norm_dir)
            elif isinstance(const, FixCartesian):
                moved[const.get_indices()] = 1 - const.mask.astype(int)
            else:
                warnings.warn(f'Constraint {const!s} is ignored')
        return moved
