from ase.build import bulk
from ase.calculators.emt import EMT
from ase.calculators.loggingcalc import LoggingCalculator


def test_loggingcalc(tmp_path, figure):
    atoms = bulk('Au', orthorhombic=True)
    calc = LoggingCalculator(EMT())
    atoms.calc = calc

    for _ in range(4):
        atoms.get_forces()
        atoms.get_potential_energy()
        atoms.positions[0, 0] += 0.1

    # This is kind of improper testing.
    # Should split plotting away from processing so we can
    # test the processing.
    calc.tabulate()
    calc.plot()
