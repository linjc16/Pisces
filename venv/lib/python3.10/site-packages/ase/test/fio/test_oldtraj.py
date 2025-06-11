"""Test that we can read old trajectory files."""
from base64 import b64decode, b64encode
from pathlib import Path

from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read
from ase.io.trajectory import Trajectory


def write():
    """Run this with an old version of ASE.

    Did it with 3.18.1.
    """

    a1 = Atoms('H')
    a1.constraints = FixAtoms(indices=[0])

    a2 = Atoms('HLi', cell=[1, 2, 3, 90, 80, 70], pbc=True)

    with Trajectory('old.traj', 'w') as traj:
        traj.write(a1)
        traj.write(a2)

    b = Path('old.traj').read_bytes()
    data = b64encode(b)
    print(f'data = {data!r}  # noqa')


def test_oldtraj():
    Path('old.traj').write_bytes(b64decode(data))
    a1, a2 = read('old.traj@:')
    assert len(a1.constraints) == 1
    assert len(a2.constraints) == 0
    assert not a1.pbc.any()
    assert a2.pbc.all()


# base64 encoded old traj file with 2 images:
data = b'LSBvZiBVbG1BU0UtVHJhamVjdG9yeSAgAwAAAAAAAAACAAAAAAAAAOACAAAAAAAAWAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADABAAAAAAAAeyJ2ZXJzaW9uIjogMSwgImFzZV92ZXJzaW9uIjogIjMuMTguMCIsICJwYmMiOiBbZmFsc2UsIGZhbHNlLCBmYWxzZV0sICJudW1iZXJzLiI6IHsibmRhcnJheSI6IFtbMV0sICJpbnQ2NCIsIDU2XX0sICJjb25zdHJhaW50cyI6ICJbe1wibmFtZVwiOiBcIkZpeEF0b21zXCIsIFwia3dhcmdzXCI6IHtcImluZGljZXNcIjogWzBdfX1dIiwgInBvc2l0aW9ucy4iOiB7Im5kYXJyYXkiOiBbWzEsIDNdLCAiZmxvYXQ2NCIsIDY0XX0sICJjZWxsIjogW1swLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMF1dfQEAAAAAAAAAAwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGAQAAAAAAAHsicGJjIjogW3RydWUsIHRydWUsIHRydWVdLCAibnVtYmVycy4iOiB7Im5kYXJyYXkiOiBbWzJdLCAiaW50NjQiLCA0MDBdfSwgInBvc2l0aW9ucy4iOiB7Im5kYXJyYXkiOiBbWzIsIDNdLCAiZmxvYXQ2NCIsIDQxNl19LCAiY2VsbCI6IFtbMS4wLCAwLjAsIDAuMF0sIFswLjY4NDA0MDI4NjY1MTMzNzYsIDEuODc5Mzg1MjQxNTcxODE2NiwgMC4wXSwgWzAuNTIwOTQ0NTMzMDAwNzkxMiwgLTAuMTg5NjA4MzAzNzE1OTk1NDYsIDIuOTQ4MzMyNjYxODEwNDldXX0jI1gAAAAAAAAA0AEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=='  # noqa
