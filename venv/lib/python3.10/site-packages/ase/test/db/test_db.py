import pytest

from ase.db import connect

cmd = """
ase -T build H | ase -T run emt -o testase.json &&
ase -T build H2O | ase -T run emt -o testase.json &&
ase -T build O2 | ase -T run emt -o testase.json &&
ase -T build H2 | ase -T run emt -f 0.02 -o testase.json &&
ase -T build O2 | ase -T run emt -f 0.02 -o testase.json &&
ase -T build -x fcc Cu | ase -T run emt -E 5,1 -o testase.json &&
ase -T db -v testase.json natoms=1,Cu=1 --delete --yes &&
ase -T db -v testase.json "H>0" -k hydro=1,abc=42,foo=bar &&
ase -T db -v testase.json "H>0" --delete-keys foo"""

dbtypes = ['json', 'db']


@pytest.mark.slow()
@pytest.mark.parametrize('dbtype', dbtypes)
def test_db(dbtype, cli, testdir):
    def count(n, *args, **kwargs):
        m = len(list(con.select(columns=['id'], *args, **kwargs)))
        assert m == n, (m, n)

    name = f'testase.{dbtype}'

    cli.shell(cmd.replace('testase.json', name))

    with connect(name) as con:
        assert con.get_atoms(H=1)[0].magmom == 1
        count(5)
        count(3, 'hydro')
        count(0, 'foo')
        count(3, abc=42)
        count(3, 'abc')
        count(0, 'abc,foo')
        count(3, 'abc,hydro')
        count(0, foo='bar')
        count(1, formula='H2')
        count(1, formula='H2O')
        count(3, 'fmax<0.1')
        count(1, '0.5<mass<1.5')
        count(5, 'energy')
        id = con.reserve(abc=7)
        assert con[id].abc == 7

        for key in ['calculator', 'energy', 'abc', 'name', 'fmax']:
            count(6, sort=key)
            count(6, sort='-' + key)

        con.delete([id])
    cli.shell(f'ase -T gui --terminal -n 3 {name}')
