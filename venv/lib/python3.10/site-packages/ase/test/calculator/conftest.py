import pytest

from ase.calculators.genericfileio import BaseProfile, CalculatorTemplate


@pytest.fixture(autouse=True)
def _calculator_tests_always_use_testdir(testdir):
    pass


class DummyProfile(BaseProfile):
    def get_calculator_command(self, inputfile):
        if not inputfile:
            return []
        return [inputfile]

    def version(self):
        return "0.0.0"


class DummyTemplate(CalculatorTemplate):

    def __init__(self):
        super().__init__(
            name="dummy",
            implemented_properties=()
        )

    def write_input(self, directory, atoms, parameters, properties):
        pass

    def load_profile(self, cfg, **kwargs):
        return DummyProfile.from_config(cfg, self.name, **kwargs)

    def execute(self, directory, profile):
        pass

    def read_results(self, directory):
        pass


@pytest.fixture()
def dummy_template():
    return DummyTemplate()
