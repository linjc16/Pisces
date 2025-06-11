from ase.atoms import Atoms
from ase.constraints import FixScaled


def test_negativeindex():
    a1 = Atoms(symbols='X2',
               positions=[[0., 0., 0.], [2., 0., 0.]],
               cell=[[4., 0., 0.], [0., 4., 0.], [0., 0., 4.]])

    fs1 = FixScaled(-1, mask=(True, False, False))
    fs2 = FixScaled(1, mask=(False, True, False))

    a1.set_constraint([fs1, fs2])

    # reassigning using atoms.__getitem__
    a2 = a1[0:2]

    assert len(a1.constraints) == len(a2.constraints)
