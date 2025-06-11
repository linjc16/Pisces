import numpy as np

from ase.build import bulk
from ase.calculators.emt import EMT
from ase.constraints import FixScaled


def test_fixscaled():
    a = bulk("Ni", cubic=True)
    a.calc = EMT()

    pos = a.get_positions()

    a.set_constraint(FixScaled(0))
    a.set_positions(pos * 1.01)

    assert np.sum(np.abs(a.get_forces()[0])) < 1e-12
    assert np.sum(np.abs(a.get_positions() - pos)[0]) < 1e-12
    assert np.sum(np.abs(a.get_positions() - pos * 1.01)[1:].flatten()) < 1e-12


def test_fixscaled_misc():
    indices = [2, 3, 4]
    mask = (0, 1, 1)
    constraint = FixScaled(indices, mask=mask)
    # XXX not providing Atoms to dof
    assert constraint.get_removed_dof(None) == sum(mask) * len(indices)
    dct = constraint.todict()
    assert dct['kwargs']['a'] == indices
    assert '2, 3, 4' in str(constraint)
