"""Temporary file while we deprecate this locaation."""

from ase.mep import DimerControl as RealDimerControl
from ase.mep import MinModeAtoms as RealMinModeAtoms
from ase.mep import MinModeTranslate as RealMinModeTranslate
from ase.utils import deprecated


class DimerControl(RealDimerControl):
    @deprecated('Please import DimerControl from ase.mep, not ase.dimer.')
    def __init__(self, *args, **kwargs):
        """
        .. deprecated:: 3.23.0
            Please import ``DimerControl`` from :mod:`ase.mep`
        """
        super().__init__(*args, **kwargs)


class MinModeAtoms(RealMinModeAtoms):
    @deprecated('Please import MinModeAtoms from ase.mep, not ase.dimer.')
    def __init__(self, *args, **kwargs):
        """
        .. deprecated:: 3.23.0
            Please import ``MinModeAtoms`` from :mod:`ase.mep`
        """
        super().__init__(*args, **kwargs)


class MinModeTranslate(RealMinModeTranslate):
    @deprecated('Please import MinModeTranslate from ase.mep, not ase.dimer.')
    def __init__(self, *args, **kwargs):
        """
        .. deprecated:: 3.23.0
            Please import ``MinModeTranslate`` from :mod:`ase.mep`
        """
        super().__init__(*args, **kwargs)
