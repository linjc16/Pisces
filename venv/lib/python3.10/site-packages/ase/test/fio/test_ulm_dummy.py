"""Test ase.io.ulm.DummyWriter."""
import numpy as np

from ase.io.ulm import DummyWriter


def test_dummy_write():
    # This writer does not write anything but needs
    # to be able to do the dummy actions
    with DummyWriter() as w:
        w.write(a=1, b=2)
        w.add_array('psi', (1, 2))
        w.fill(np.ones((1, 2)))
        w.sync()
        with w.child('child') as w2:
            w2.write(c=3)
        assert len(w) == 0
