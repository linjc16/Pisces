# Copyright 2008, 2009 CAMd
# (see accompanying license files for details).

"""Atomic Simulation Environment."""

# import ase.parallel early to avoid circular import problems when
# ase.parallel does "from gpaw.mpi import world":
import ase.parallel  # noqa
from ase.atom import Atom
from ase.atoms import Atoms

__all__ = ['Atoms', 'Atom']
__version__ = '3.25.0'

ase.parallel  # silence pyflakes
