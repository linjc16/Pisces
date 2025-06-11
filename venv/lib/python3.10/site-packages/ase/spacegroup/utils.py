# fmt: off

from typing import List

import numpy as np

from ase import Atoms

from .spacegroup import _SPACEGROUP, Spacegroup

__all__ = ('get_basis', )


def _has_spglib() -> bool:
    """Check if spglib is available"""
    try:
        import spglib
        assert spglib  # silence flakes
    except ImportError:
        return False
    return True


def _get_basis_ase(atoms: Atoms,
                   spacegroup: _SPACEGROUP,
                   tol: float = 1e-5) -> np.ndarray:
    """Recursively get a reduced basis, by removing equivalent sites.
    Uses the first index as a basis, then removes all equivalent sites,
    uses the next index which hasn't been placed into a basis, etc.

    :param atoms: Atoms object to get basis from.
    :param spacegroup: ``int``, ``str``, or
        :class:`ase.spacegroup.Spacegroup` object.
    :param tol: ``float``, numeric tolerance for positional comparisons
        Default: ``1e-5``
    """
    scaled_positions = atoms.get_scaled_positions()
    spacegroup = Spacegroup(spacegroup)

    def scaled_in_sites(scaled_pos: np.ndarray, sites: np.ndarray):
        """Check if a scaled position is in a site"""
        for site in sites:
            if np.allclose(site, scaled_pos, atol=tol):
                return True
        return False

    def _get_basis(scaled_positions: np.ndarray,
                   spacegroup: Spacegroup,
                   all_basis=None) -> np.ndarray:
        """Main recursive function to be executed"""
        if all_basis is None:
            # Initialization, first iteration
            all_basis = []
        if len(scaled_positions) == 0:
            # End termination
            return np.array(all_basis)

        basis = scaled_positions[0]
        all_basis.append(basis.tolist())  # Add the site as a basis

        # Get equivalent sites
        sites, _ = spacegroup.equivalent_sites(basis)

        # Remove equivalent
        new_scaled = np.array(
            [sc for sc in scaled_positions if not scaled_in_sites(sc, sites)])
        # We should always have at least popped off the site itself
        assert len(new_scaled) < len(scaled_positions)

        return _get_basis(new_scaled, spacegroup, all_basis=all_basis)

    return _get_basis(scaled_positions, spacegroup)


def _get_basis_spglib(atoms: Atoms, tol: float = 1e-5) -> np.ndarray:
    """Get a reduced basis using spglib. This requires having the
    spglib package installed.

    :param atoms: Atoms, atoms object to get basis from
    :param tol: ``float``, numeric tolerance for positional comparisons
        Default: ``1e-5``
    """
    if not _has_spglib():
        # Give a reasonable alternative solution to this function.
        raise ImportError(
            'This function requires spglib. Use "get_basis" and specify '
            'the spacegroup instead, or install spglib.')

    scaled_positions = atoms.get_scaled_positions()
    reduced_indices = _get_reduced_indices(atoms, tol=tol)
    return scaled_positions[reduced_indices]


def _can_use_spglib(spacegroup: _SPACEGROUP = None) -> bool:
    """Helper dispatch function, for deciding if the spglib implementation
    can be used"""
    if not _has_spglib():
        # Spglib not installed
        return False
    if spacegroup is not None:
        # Currently, passing an explicit space group is not supported
        # in spglib implementation
        return False
    return True


# Dispatcher function for chosing get_basis implementation.
def get_basis(atoms: Atoms,
              spacegroup: _SPACEGROUP = None,
              method: str = 'auto',
              tol: float = 1e-5) -> np.ndarray:
    """Function for determining a reduced basis of an atoms object.
    Can use either an ASE native algorithm or an spglib based one.
    The native ASE version requires specifying a space group,
    while the (current) spglib version cannot.
    The default behavior is to automatically determine which implementation
    to use, based on the the ``spacegroup`` parameter,
    and whether spglib is installed.

    :param atoms: ase Atoms object to get basis from
    :param spacegroup: Optional, ``int``, ``str``
        or :class:`ase.spacegroup.Spacegroup` object.
        If unspecified, the spacegroup can be inferred using spglib,
        if spglib is installed, and ``method`` is set to either
        ``'spglib'`` or ``'auto'``.
        Inferring the spacegroup requires spglib.
    :param method: ``str``, one of: ``'auto'`` | ``'ase'`` | ``'spglib'``.
        Selection of which implementation to use.
        It is recommended to use ``'auto'``, which is also the default.
    :param tol: ``float``, numeric tolerance for positional comparisons
        Default: ``1e-5``
    """
    ALLOWED_METHODS = ('auto', 'ase', 'spglib')

    if method not in ALLOWED_METHODS:
        raise ValueError('Expected one of {} methods, got {}'.format(
            ALLOWED_METHODS, method))

    if method == 'auto':
        # Figure out which implementation we want to use automatically
        # Essentially figure out if we can use the spglib version or not
        use_spglib = _can_use_spglib(spacegroup=spacegroup)
    else:
        # User told us which implementation they wanted
        use_spglib = method == 'spglib'

    if use_spglib:
        # Use the spglib implementation
        # Note, we do not pass the spacegroup, as the function cannot handle
        # an explicit space group right now. This may change in the future.
        return _get_basis_spglib(atoms, tol=tol)
    else:
        # Use the ASE native non-spglib version, since a specific
        # space group is requested
        if spacegroup is None:
            # We have reached this point either because spglib is not installed,
            # or ASE was explicitly required
            raise ValueError(
                'A space group must be specified for the native ASE '
                'implementation. Try using the spglib version instead, '
                'or explicitly specifying a space group.')
        return _get_basis_ase(atoms, spacegroup, tol=tol)


def _get_reduced_indices(atoms: Atoms, tol: float = 1e-5) -> List[int]:
    """Get a list of the reduced atomic indices using spglib.
    Note: Does no checks to see if spglib is installed.

    :param atoms: ase Atoms object to reduce
    :param tol: ``float``, numeric tolerance for positional comparisons
    """
    from ase.spacegroup.symmetrize import spglib_get_symmetry_dataset

    # Create input for spglib
    spglib_cell = (atoms.get_cell(), atoms.get_scaled_positions(),
                   atoms.numbers)
    symmetry_data = spglib_get_symmetry_dataset(spglib_cell,
                                                symprec=tol)
    return list(set(symmetry_data.equivalent_atoms))
