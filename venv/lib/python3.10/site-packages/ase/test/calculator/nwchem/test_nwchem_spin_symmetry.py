"""Check if spin-symmetry is conserved"""
import pytest

from ase import Atoms


@pytest.mark.calculator('nwchem')
def test_main(factory):
    """Check is independence of alignment is conserved"""
    energies = []
    cr_atom = Atoms('Cr', positions=[(0, 0, 0)], pbc=False)
    for orientation in range(2):
        imm = 6 * (-1) ** orientation
        cr_atom.set_initial_magnetic_moments([imm])
        calculator = factory.calc(
            task='energy',
            dft=dict(convergence=dict(energy=1e-3,
                                      density=1e-2,
                                      gradient=5e-2)),
            basis='m6-31g*',
            basispar='"ao basis" spherical',
            charge=0
        )
        cr_atom.calc = calculator
        energies.append(cr_atom.get_potential_energy())
    assert abs(energies[0] - energies[1]) < 1e-9
