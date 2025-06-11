import numpy as np
import pytest

from ase import Atom, Atoms
from ase.io.nwchem import write_nwchem_in


@pytest.fixture()
def atomic_configuration():
    molecule = Atoms(pbc=False)
    molecule.append(Atom('C', [0, 0, 0]))
    molecule.append(Atom('O', [1.6, 0, 0]))
    return molecule


@pytest.fixture()
def calculator_parameters():
    params = dict(
        memory='1024 mb',
        dft=dict(xc='b3lyp', mult=1, maxiter=300),
        basis='6-311G*',
    )
    return params


def test_echo(atomic_configuration, calculator_parameters, tmp_path):
    with (tmp_path / 'nwchem.in').open('w') as fd:
        write_nwchem_in(
            fd, atomic_configuration, echo=False, **calculator_parameters
        )
    content = (tmp_path / 'nwchem.in').read_text().splitlines()
    assert 'echo' not in content

    with (tmp_path / 'nwchem.in').open('w') as fd:
        write_nwchem_in(
            fd, atomic_configuration, echo=True, **calculator_parameters
        )
    content = (tmp_path / 'nwchem.in').read_text().splitlines()
    assert 'echo' in content


def test_params(atomic_configuration, calculator_parameters, tmp_path):
    with (tmp_path / 'nwchem.in').open('w') as fd:
        write_nwchem_in(fd, atomic_configuration, **calculator_parameters)
    content = (tmp_path / 'nwchem.in').read_text().splitlines()

    for key, value in calculator_parameters.items():
        for line in content:
            flds = line.split()
            if len(flds) == 0:
                continue
            if flds[0] == key:
                break
        else:
            assert False
        if key == 'basis':  # special case
            pass
        elif isinstance(value, str):
            assert len(value.split()) == len(flds[1:])
            assert all(v == f for v, f in zip(value.split(), flds[1:]))
        elif isinstance(value, (int, float)):
            assert len(flds[1:]) == 1
            assert np.isclose(value, float(flds[1]))


def test_write_nwchem_in_set_params(
    atomic_configuration, calculator_parameters, tmp_path
):
    """
    Tests writing NWChem input file with a dictionary
    in the 'set' parameter, ensuring correct section order.
    Closes #1578
    """
    cparams = calculator_parameters
    cparams['set'] = {'geom:dont_verify': True}
    with (tmp_path / 'nwchem.in').open('w') as fd:
        write_nwchem_in(fd, atomic_configuration, echo=False, **cparams)
    content = (tmp_path / 'nwchem.in').read_text().splitlines()
    # 'set geom:dont_verify .true.' must appear before 'geometry'
    set_line_index = next(
        (
            i
            for i, line in enumerate(content)
            if line.strip() == 'set geom:dont_verify .true.'
        ),
        None,
    )
    geometry_line_index = next(
        (
            i
            for i, line in enumerate(content)
            if line.strip().startswith('geometry')
        ),
        None,
    )

    assert set_line_index is not None and geometry_line_index is not None
    assert set_line_index < geometry_line_index
