import os

import pytest


def installed():
    for env in ['VASP_COMMAND', 'VASP_SCRIPT']:
        if os.getenv(env):
            break
    else:
        pytest.skip('Neither VASP_COMMAND nor VASP_SCRIPT defined')
    return True


def installed2():
    # Check if env variables exist for Vasp2

    for env in ['VASP_COMMAND', 'VASP_SCRIPT', 'ASE_VASP_COMMAND']:
        if os.getenv(env):
            break
    else:
        pytest.skip('Neither ASE_VASP_COMMAND, VASP_COMMAND nor '
                    'VASP_SCRIPT defined')

    return True
