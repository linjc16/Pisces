import numpy as np
import pytest

import ase.units as u
from ase.build import bulk

# For symmetry reasons, stress[1] and [2] should be equal as well as
# stress[4] and [5].
# These references values are based on "consensus" between codes i.e.
# whatever it takes to get them to pass this test.
ref_stress = np.array([-0.22, -0.17, -0.17,
                       -0.17, 0.18, 0.18])

calc = pytest.mark.calculator


@pytest.mark.calculator_lite()
@calc('aims', compute_analytical_stress=True)
@calc('gpaw', mode={'name': 'pw', 'ecut': 350}, txt=None)
@calc('abinit', chksymtnons=0, ecut=350)
@calc('espresso',
      input_data={"control": {"tprnfor": True,
                              "tstress": True},
                  "system": {"ecutwfc": 350 / u.Ry}})
@calc('siesta')
def test_si_stress(factory):
    atoms = bulk('Si')
    atoms.calc = factory.calc(kpts=[4, 4, 4])
    atoms.cell[0] *= 0.85
    stress = atoms.get_stress()
    print(stress)
    assert stress == pytest.approx(ref_stress, rel=0.15)
    assert stress[1] == pytest.approx(stress[2], rel=0.01)
    assert stress[4] == pytest.approx(stress[5], rel=0.01)
