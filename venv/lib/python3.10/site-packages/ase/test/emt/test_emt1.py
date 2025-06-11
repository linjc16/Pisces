from ase import Atoms
from ase.calculators.emt import EMT
from ase.constraints import FixBondLength
from ase.io import Trajectory
from ase.optimize import BFGS


def test_emt1(testdir):
    a = 3.6
    b = a / 2
    cu = Atoms('Cu2Ag',
               positions=[(0, 0, 0),
                          (b, b, 0),
                          (a, a, b)],
               calculator=EMT())
    e0 = cu.get_potential_energy()
    print(e0)

    d0 = cu.get_distance(0, 1)
    cu.set_constraint(FixBondLength(0, 1))

    def f():
        print(cu.get_distance(0, 1))

    qn = BFGS(cu)
    with Trajectory('cu2ag.traj', 'w', cu) as t:
        qn.attach(t.write)

        qn.attach(f)
        qn.run(fmax=0.001)

    assert abs(cu.get_distance(0, 1) - d0) < 1e-14
