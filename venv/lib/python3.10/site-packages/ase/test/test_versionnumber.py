from importlib.metadata import PackageNotFoundError, version

import pytest

import ase


def test_versionnumber():
    # Check that the version number reported by importlib matches
    # what is hardcoded in ase/__init__.py
    try:
        version_seen_by_python = version('ase')
    except PackageNotFoundError:
        pytest.skip('ASE not visible to importlib.metadata')
    else:
        assert ase.__version__ == version_seen_by_python, (
            f'ASE is version {ase.__version__} but python thinks it is '
            f'{version_seen_by_python} - perhaps rerun "pip install -e"')
