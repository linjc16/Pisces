from ase import io
from ase.calculators.singlepoint import SinglePointDFTCalculator


def test_read_gpaw_out(datadir):
    """Test reading of gpaw text output"""

    # read input

    output_file_name = datadir / 'gpaw_expected_text_output'
    atoms = io.read(output_file_name)

    # test calculator

    calc = atoms.calc
    assert isinstance(calc, SinglePointDFTCalculator)
    assert calc.name == 'vdwtkatchenko09prl'
    assert calc.parameters['calculator'] == 'gpaw'

    for contribution in [
            'kinetic', 'potential', 'external', 'xc',
            'entropy (-st)', 'local']:
        assert contribution in calc.energy_contributions
