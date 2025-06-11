import pytest

from ase import Atoms
from ase.optimize import BFGS


@pytest.mark.xfail(reason='needs update for new dftd+ version')
@pytest.mark.calculator_lite()
@pytest.mark.calculator('dftb')
def test_dftb_relax_dimer(factory):
    calc = factory.calc(
        label='dftb',
        Hamiltonian_SCC='No',
        Hamiltonian_PolynomialRepulsive='SetForAll {Yes}',
    )

    atoms = Atoms('Si2', positions=[[5., 5., 5.], [7., 5., 5.]],
                  cell=[12.] * 3, pbc=False)
    atoms.calc = calc

    with BFGS(atoms, logfile='-') as dyn:
        dyn.run(fmax=0.1)

    e = atoms.get_potential_energy()
    assert abs(e - -64.830901) < 1., e
