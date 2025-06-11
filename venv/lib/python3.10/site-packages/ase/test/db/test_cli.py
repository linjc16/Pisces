"""Testing of "ase db" command-line interface."""
from pathlib import Path

import pytest

from ase import Atoms
from ase.build import bulk, molecule
from ase.db import connect


@pytest.fixture(scope='module')
def dbfile(tmp_path_factory) -> str:
    """Create a database file (x.db) with two rows."""
    path = tmp_path_factory.mktemp('db') / 'x.db'

    with connect(path) as db:
        db.write(Atoms())
        db.write(molecule('H2O'), key_value_pairs={'carrots': 3})
        db.write(bulk('Ti'), key_value_pairs={'oranges': 42, 'carrots': 4})

    return str(path)


def test_insert_into(cli, dbfile):
    """Test --insert-into."""
    out = Path(dbfile).with_name('x1.db')
    # Insert 1 row:
    cli.ase(
        *f'db {dbfile} --limit 1 --insert-into {out} --progress-bar'.split())
    # Count:
    txt = cli.ase(*f'db {out} --count'.split())
    num = int(txt.split()[0])
    assert num == 1


def test_analyse(cli, dbfile):
    txt = cli.ase('db', dbfile, '--show-keys')
    print(txt)
    assert 'carrots: 2' in txt
    assert 'oranges: 1' in txt


def test_show_values(cli, dbfile):
    txt = cli.ase('db', dbfile, '--show-values', 'oranges,carrots')
    print(txt)
    assert 'carrots: [3..4]' in txt


def check_tokens(tokens):
    # Order of headers is not reproducible so we just check
    # that certain headers are included:
    assert {'id', 'age', 'formula'} < set(tokens)
    assert 'H2O' in tokens
    assert 'Ti2' in tokens


def test_table(cli, dbfile):
    txt = cli.ase('db', dbfile)
    print(txt)
    tokens = [token.strip() for token in txt.split('|')]
    check_tokens(tokens)


def test_table_csv(cli, dbfile):
    txt = cli.ase('db', dbfile, '--csv')
    print(txt)
    tokens = txt.split(', ')
    check_tokens(tokens)


def test_long(cli, dbfile):
    txt = cli.ase('db', dbfile, 'formula=Ti2', '--long')
    print(txt)
    assert 'length' in txt  # about cell vector lengths
    assert 'oranges' in txt
