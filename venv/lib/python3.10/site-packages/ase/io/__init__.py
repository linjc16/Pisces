# fmt: off

from ase.io.bundletrajectory import BundleTrajectory
from ase.io.formats import iread, read, string2index, write
from ase.io.netcdftrajectory import NetCDFTrajectory
from ase.io.trajectory import PickleTrajectory, Trajectory


class ParseError(Exception):
    """Parse error during reading of a file"""


__all__ = [
    'Trajectory', 'PickleTrajectory', 'BundleTrajectory', 'NetCDFTrajectory',
    'read', 'iread', 'write', 'string2index'
]
