"""Temporary file while we deprecate this location."""

from ase.mep import DyNEB as RealDyNEB
from ase.utils import deprecated


class DyNEB(RealDyNEB):
    @deprecated('Please import DyNEB from ase.mep, not ase.dyneb.')
    def __init__(self, *args, **kwargs):
        """
        .. deprecated:: 3.23.0
            Please import ``DyNEB`` from :mod:`ase.mep`
        """
        super().__init__(*args, **kwargs)
