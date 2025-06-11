import pytest


@pytest.fixture(autouse=True)
def _force_in_tempdir(testdir):
    pass
