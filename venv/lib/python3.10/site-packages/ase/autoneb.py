"""Temporary file while we deprecate this locaation."""

from ase.mep import AutoNEB as RealAutoNEB
from ase.utils import deprecated


class AutoNEB(RealAutoNEB):
    @deprecated('Please import AutoNEB from ase.mep, not ase.autoneb.')
    def __init__(self, *args, **kwargs):
        """
        .. deprecated:: 3.23.0
            Please import ``AutoNEB`` from :mod:`ase.mep`
        """
        super().__init__(*args, **kwargs)
