from ase.build import bulk
from ase.calculators.calculator import FileIOCalculator
from ase.calculators.dftb import Dftb
from ase.filters import FrechetCellFilter
from ase.optimize import QuasiNewton


def test_init():
    assert isinstance(Dftb(), FileIOCalculator)


def test_dftb_relax_bulk(dftb_factory):
    calc = dftb_factory.calc(
        label='dftb',
        kpts=(3, 3, 3),
        Hamiltonian_SCC='Yes'
    )

    atoms = bulk('Si')
    atoms.calc = calc

    ecf = FrechetCellFilter(atoms)
    with QuasiNewton(ecf) as dyn:
        dyn.run(fmax=0.01)

    e = atoms.get_potential_energy()
    assert abs(e - -73.150819) < 1., e
