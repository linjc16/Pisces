import numpy as np
import pytest

from ase.build import molecule


def run(calc):
    atoms = molecule('H2', vacuum=3.0)
    atoms.center(vacuum=4.0, axis=2)
    atoms.calc = calc

    d0 = atoms.get_distance(0, 1)

    energies = []
    distances = []

    # Coarse binding curve:
    for factor in [0.95, 1.0, 1.05]:
        dist = d0 * factor
        atoms.positions[:, 2] = [-0.5 * dist, 0.5 * dist]
        atoms.center()

        energy = atoms.get_potential_energy()
        energies.append(energy)
        distances.append(dist)

    # We use bad/inconsistent parameters, but the minimum will
    # still be somewhere around 0.7
    poly = np.polyfit(distances, energies, deg=2)
    dmin = -0.5 * poly[1] / poly[0]

    assert dmin == pytest.approx(0.766, abs=0.03)  # bond length

    # 1.5 enough without siesta
    # 2.5 with siesta
    assert poly[0] == pytest.approx(20.0, abs=2.5)  # bond stiffness


calc = pytest.mark.calculator

# marks=[pytest.mark.filterwarnings('ignore::DeprecationWarning')])


@pytest.mark.filterwarnings('ignore:Subprocess exited')
@pytest.mark.calculator_lite()
@calc('abinit')
@calc('espresso', input_data={"system": {"ecutwfc": 30}})
@calc('nwchem')
@calc('aims')
@calc('siesta')
# @pytest.mark.calculator('dftb', Hamiltonian_MaxAngularMomentum_H='"s"')
def test_socketio_h2(factory):
    """SocketIO integration test; fit coarse binding curve of H2 molecule."""
    with factory.socketio(unixsocket=f'ase_test_h2_{factory.name}') as calc:
        run(calc)
