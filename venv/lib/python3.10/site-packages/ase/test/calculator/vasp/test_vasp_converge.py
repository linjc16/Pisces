from pathlib import Path

from ase.calculators.vasp import Vasp

parent = Path(__file__).parents[2]


def test_vasp_converge():
    calc = Vasp()

    # apparently converged, and message says converged

    # with VASP6 message
    with open(parent / "testdata/vasp/convergence_OUTCAR_y_y") as fin:
        assert calc.read_convergence(fin.readlines())
    # without VASP6 message
    with open(parent / "testdata/vasp/convergence_OUTCAR_y_y") as fin:
        assert calc.read_convergence(fin.readlines()[:-1])

    # apparently converged, but message says unconverged

    # with VASP6 message
    with open(parent / "testdata/vasp/convergence_OUTCAR_y_n") as fin:
        assert not calc.read_convergence(fin.readlines())
    # without VASP6 message
    with open(parent / "testdata/vasp/convergence_OUTCAR_y_n") as fin:
        assert calc.read_convergence(fin.readlines()[:-1])

    # apparently unconverged, but message says converged

    # with VASP6 message
    with open(parent / "testdata/vasp/convergence_OUTCAR_n_y") as fin:
        assert calc.read_convergence(fin.readlines())
    # without VASP6 message
    with open(parent / "testdata/vasp/convergence_OUTCAR_n_y") as fin:
        assert not calc.read_convergence(fin.readlines()[:-1])
