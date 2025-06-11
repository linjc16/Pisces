import pytest

from ase.build import bulk
from ase.calculators.emt import EMT
from ase.optimize import BFGS


@pytest.fixture()
def opt():
    atoms = bulk('Au', cubic=True)
    atoms.rattle(stdev=0.12345, seed=42)
    atoms.calc = EMT()
    with BFGS(atoms) as opt:
        yield opt


@pytest.mark.parametrize('steps', [0, 1, 4])
def test_nsteps(opt, steps):
    """Test if the number of iterations is as expected.

    For opt.irun(steps=n), the number of iterations should be n + 1,
    including 0 and n.
    """
    irun = opt.irun(fmax=0, steps=steps)

    for _ in range(steps + 1):
        next(irun)

    with pytest.raises(StopIteration):
        next(irun)
