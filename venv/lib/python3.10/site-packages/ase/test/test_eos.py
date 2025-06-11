import numpy as np

from ase.build import bulk
from ase.calculators.emt import EMT
from ase.eos import EquationOfState as EOS
from ase.eos import eos_names


def test_eos():
    b = bulk('Al', 'fcc', a=4.0, orthorhombic=True)
    b.calc = EMT()
    cell = b.get_cell()

    volumes = []
    energies = []
    for x in np.linspace(0.98, 1.01, 5):
        b.set_cell(cell * x, scale_atoms=True)
        volumes.append(b.get_volume())
        energies.append(b.get_potential_energy())

    results = []
    for name in eos_names:
        if name == 'antonschmidt':
            # Someone should fix this!
            continue
        eos = EOS(volumes, energies, name)
        v, e, b = eos.fit()
        print(f'{name:20} {v:.8f} {e:.8f} {b:.8f} ')
        assert abs(v - 3.18658700e+01) < 4e-4
        assert abs(e - -9.76187802e-03) < 5e-7
        assert abs(b - 2.46812688e-01) < 2e-4
        results.append((v, e, b))

    print(np.ptp(results, 0))
    print(np.mean(results, 0))
