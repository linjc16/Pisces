import pytest


@pytest.mark.calculator_lite()
@pytest.mark.calculator('nwchem')
def test_version(factory):
    version = factory.factory.version()
    assert version[0].isdigit()
