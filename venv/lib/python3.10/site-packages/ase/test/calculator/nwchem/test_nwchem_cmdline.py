import pytest
from numpy.testing import assert_allclose

from ase.db import connect
from ase.io import read
from ase.io.jsonio import read_json


@pytest.mark.skip('CLI fixture lacks information about configured executables')
def test_cmdline(cli):
    cli.shell("""\
ase build O O.traj &&
ase run nwchem O.traj -o nwchem_cmdline.json &&
ase build O2 O2.traj &&
ase run nwchem O2.traj -o nwchem_cmdline.json""",
              'nwchem')
    c = connect('nwchem_cmdline.json')
    dct = read_json('nwchem_cmdline.json')
    for name in ['O2', 'O']:
        d = c.get([('formula', '=', name)])
        id = d.id
        e1 = d.energy
        e2 = c.get_atoms(id).get_potential_energy()
        e3 = read(f'{name}.nwo').get_potential_energy()
        e4 = dct[id]['energy']
        assert e1 == e2 == e3 == e4
        print(e1)
    ae = 2 * c.get('formula=O').energy - c.get('formula=O2').energy
    assert_allclose(ae, 6.599194233179787, atol=1e-4, rtol=1e-4)
