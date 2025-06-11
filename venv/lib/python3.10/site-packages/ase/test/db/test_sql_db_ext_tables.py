import numpy as np
import pytest

from ase import Atoms
from ase.db import connect
from ase.db.sqlite import all_tables


def test_create_and_delete_ext_tab(testdir):
    ext_tab = ["tab1", "tab2", "tab3"]
    atoms = Atoms()

    db = connect('test.db')
    db.write(atoms)

    for tab in ext_tab:
        db._create_table_if_not_exists(tab, "INTEGER")
    current_ext_tables = db._get_external_table_names()
    for tab in ext_tab:
        assert tab in current_ext_tables

    db.delete_external_table("tab1")
    assert "tab1" not in db._get_external_table_names()


def test_insert_in_external_tables(testdir):
    atoms = Atoms()

    db = connect('test.db')

    # Now a table called insert_tab with schema datatype REAL should
    # be created
    uid = db.write(
        atoms,
        external_tables={
            "insert_tab": {
                "rate": 1.0,
                "rate1": -
                2.0}})

    db.delete([uid])

    # Hack: retrieve the connection
    con = db._connect()
    cur = con.cursor()

    sql = "SELECT * FROM insert_tab WHERE ID=?"
    cur.execute(sql, (uid,))

    entries = [x for x in cur.fetchall()]
    if db.connection is None:
        con.close()
    assert not entries

    # Make sure that there are now entries in the
    # external table with current uid

    # Try to insert something that should not pass
    # i.e. string value into the same table
    with pytest.raises(ValueError):
        db.write(atoms, external_tables={
            "insert_tab": {"rate": "something"}})

    # Try to insert Numpy floats
    db.write(atoms, external_tables={
        "insert_tab": {"rate": np.float32(1.0)}})
    db.write(atoms, external_tables={
        "insert_tab": {"rate": np.float64(1.0)}})

    # Make sure that we cannot insert a Numpy integer types into
    # a float array
    with pytest.raises(ValueError):
        db.write(
            atoms, external_tables={
                "insert_tab": {
                    "rate": np.int32(1.0)}})

    with pytest.raises(ValueError):
        db.write(
            atoms, external_tables={
                "insert_tab": {
                    "rate": np.int64(1.0)}})

    # Create a new table should have INTEGER types
    db.write(atoms, external_tables={"integer_tab": {"rate": 1}})

    # Make sure we can insert Numpy integers
    db.write(atoms, external_tables={"integer_tab": {"rate": np.int32(1)}})
    db.write(atoms, external_tables={"integer_tab": {"rate": np.int64(1)}})

    # Make sure that we cannot insert float
    with pytest.raises(ValueError):
        db.write(
            atoms, external_tables={
                "integer_tab": {
                    "rate": np.float32(1)}})

    with pytest.raises(ValueError):
        db.write(
            atoms, external_tables={
                "integer_tab": {
                    "rate": np.float64(1)}})

    # Make sure that ValueError is raised with mixed datatypes
    with pytest.raises(ValueError):
        db.write(
            atoms,
            external_tables={
                "integer_tab": {
                    "rate": 1,
                    "rate2": 2.0}})

    # Test that we cannot insert anything into a reserved table name
    for tab_name in all_tables:
        with pytest.raises(ValueError):
            db.write(atoms, external_tables={tab_name: {"value": 1}})


def test_extract_from_table(testdir):
    atoms = Atoms()

    db = connect('test.db')
    uid = db.write(
        atoms,
        external_tables={
            "insert_tab": {
                "rate": 12.0,
                "rate1": -
                10.0}})

    row = db.get(id=uid)
    assert abs(row["insert_tab"]["rate"] - 12.0) < 1E-8
    assert abs(row["insert_tab"]["rate1"] + 10.0) < 1E-8


def test_write_atoms_row(testdir):
    atoms = Atoms()

    db = connect('test.db')
    uid = db.write(
        atoms, external_tables={
            "insert_tab": {"rate": 12.0, "rate1": -10.0},
            "another_tab": {"somevalue": 1.0}})
    row = db.get(id=uid)

    # Hack: Just change the unique ID
    row["unique_id"] = "uniqueIDTest"
    db.write(row)


def test_external_table_upon_update(testdir):
    db = connect('test.db')
    no_features = 500
    ext_table = {i: i for i in range(no_features)}
    atoms = Atoms('Pb', positions=[[0, 0, 0]])
    uid = db.write(atoms)
    db.update(uid, external_tables={'sys': ext_table})


def test_external_table_upon_update_with_float(testdir):
    db = connect('test.db')
    ext_table = {'value1': 1.0, 'value2': 2.0}
    atoms = Atoms('Pb', positions=[[0, 0, 0]])
    uid = db.write(atoms)
    db.update(uid, external_tables={'float_table': ext_table})
