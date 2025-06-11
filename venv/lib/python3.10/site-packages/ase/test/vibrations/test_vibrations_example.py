import io

from ase import Atoms
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.vibrations import Vibrations

expected_summary = """---------------------
  #    meV     cm^-1
---------------------
  0    0.0       0.0
  1    0.0       0.0
  2    0.0       0.0
  3    1.4      11.5
  4    1.4      11.5
  5  152.7    1231.3
---------------------
Zero-point energy: 0.078 eV
"""


def test_vibrations_example(testdir):
    """Test the example from the Vibrations.__init__() docstring"""
    n2 = Atoms('N2', [(0, 0, 0), (0, 0, 1.1)],
               calculator=EMT())
    BFGS(n2).run(fmax=0.01)

    vib = Vibrations(n2)
    vib.run()

    with io.StringIO() as fd:
        vib.summary(log=fd)
        fd.seek(0)

        summary = fd.read()
        assert len(summary.split()) == len(expected_summary.split())
