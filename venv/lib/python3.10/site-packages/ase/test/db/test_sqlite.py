import os

import pytest

from ase.db import connect

pytestmark = pytest.mark.usefixtures('testdir')
db_name = "test_vacuum.db"


def write_entries_to_db(db, n_entries_db=30):
    for i in range(n_entries_db):
        db.reserve(mykey=f'test_{i}')


def update_keys_in_db(db):
    new_keys = {}
    for i in range(50):
        new_keys.update({f'mynewkey_{i}': 'test'})
    for row in db.select():
        db.update(row.id, **new_keys)


def check_delete_function(db):
    db_size_full = os.path.getsize(db_name)
    db.delete([row.id for row in db.select()])
    db_size_empty = os.path.getsize(db_name)
    assert db_size_full > db_size_empty


def check_update_function(db):
    db_size_update = os.path.getsize(db_name)
    db.vacuum()  # call vacuum explicitly
    db_size_update_vacuum = os.path.getsize(db_name)
    assert db_size_update > db_size_update_vacuum


def test_delete_vacuum():
    # test to call from within the class using the delete function
    db = connect(db_name)
    write_entries_to_db(db)
    check_delete_function(db)


def test_delete_vacuum_context():
    # try within context
    with connect(db_name) as db:
        write_entries_to_db(db)
    with connect(db_name) as db:
        check_delete_function(db)


def test_update_vacuum():
    # test to call vacuum explicitly
    db = connect(db_name)
    write_entries_to_db(db)
    update_keys_in_db(db)
    check_update_function(db)


def test_update_vacuum_context():
    # within context manager
    with connect(db_name) as db:
        write_entries_to_db(db)
    with connect(db_name) as db:
        update_keys_in_db(db)
    with connect(db_name) as db:
        check_update_function(db)
