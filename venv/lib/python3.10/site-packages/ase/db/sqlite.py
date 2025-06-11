# fmt: off

"""SQLite3 backend.

Versions:

1) Added 3 more columns.
2) Changed "user" to "username".
3) Now adding keys to keyword table and added an "information" table containing
   a version number.
4) Got rid of keywords.
5) Add fmax, smax, mass, volume, charge
6) Use REAL for magmom and drop possibility for non-collinear spin
7) Volume can be None
8) Added name='metadata' row to "information" table
9) Row data is now stored in binary format.
"""

import json
import numbers
import os
import sqlite3
import sys
from contextlib import contextmanager

import numpy as np

import ase.io.jsonio
from ase.calculators.calculator import all_properties
from ase.data import atomic_numbers
from ase.db.core import (
    Database,
    bytes_to_object,
    invop,
    lock,
    now,
    object_to_bytes,
    ops,
    parse_selection,
)
from ase.db.row import AtomsRow
from ase.parallel import parallel_function

VERSION = 9

init_statements = [
    """CREATE TABLE systems (
    id INTEGER PRIMARY KEY AUTOINCREMENT,  -- ID's, timestamps and user name
    unique_id TEXT UNIQUE,
    ctime REAL,
    mtime REAL,
    username TEXT,
    numbers BLOB,  -- stuff that defines an Atoms object
    positions BLOB,
    cell BLOB,
    pbc INTEGER,
    initial_magmoms BLOB,
    initial_charges BLOB,
    masses BLOB,
    tags BLOB,
    momenta BLOB,
    constraints TEXT,  -- constraints and calculator
    calculator TEXT,
    calculator_parameters TEXT,
    energy REAL,  -- calculated properties
    free_energy REAL,
    forces BLOB,
    stress BLOB,
    dipole BLOB,
    magmoms BLOB,
    magmom REAL,
    charges BLOB,
    key_value_pairs TEXT,  -- key-value pairs and data as json
    data BLOB,
    natoms INTEGER,  -- stuff for making queries faster
    fmax REAL,
    smax REAL,
    volume REAL,
    mass REAL,
    charge REAL)""",

    """CREATE TABLE species (
    Z INTEGER,
    n INTEGER,
    id INTEGER,
    FOREIGN KEY (id) REFERENCES systems(id))""",

    """CREATE TABLE keys (
    key TEXT,
    id INTEGER,
    FOREIGN KEY (id) REFERENCES systems(id))""",

    """CREATE TABLE text_key_values (
    key TEXT,
    value TEXT,
    id INTEGER,
    FOREIGN KEY (id) REFERENCES systems(id))""",

    """CREATE TABLE number_key_values (
    key TEXT,
    value REAL,
    id INTEGER,
    FOREIGN KEY (id) REFERENCES systems(id))""",

    """CREATE TABLE information (
    name TEXT,
    value TEXT)""",

    f"INSERT INTO information VALUES ('version', '{VERSION}')"]

index_statements = [
    'CREATE INDEX unique_id_index ON systems(unique_id)',
    'CREATE INDEX ctime_index ON systems(ctime)',
    'CREATE INDEX username_index ON systems(username)',
    'CREATE INDEX calculator_index ON systems(calculator)',
    'CREATE INDEX species_index ON species(Z)',
    'CREATE INDEX key_index ON keys(key)',
    'CREATE INDEX text_index ON text_key_values(key)',
    'CREATE INDEX number_index ON number_key_values(key)']

all_tables = ['systems', 'species', 'keys',
              'text_key_values', 'number_key_values']


def float_if_not_none(x):
    """Convert numpy.float64 to float - old db-interfaces need that."""
    if x is not None:
        return float(x)


class SQLite3Database(Database):
    type = 'db'
    initialized = False
    _allow_reading_old_format = False
    default = 'NULL'  # used for autoincrement id
    connection = None
    version = None
    columnnames = [line.split()[0].lstrip()
                   for line in init_statements[0].splitlines()[1:]]

    def encode(self, obj, binary=False):
        if binary:
            return object_to_bytes(obj)
        return ase.io.jsonio.encode(obj)

    def decode(self, txt, lazy=False):
        if lazy:
            return txt
        if isinstance(txt, str):
            return ase.io.jsonio.decode(txt)
        return bytes_to_object(txt)

    def blob(self, array):
        """Convert array to blob/buffer object."""

        if array is None:
            return None
        if len(array) == 0:
            array = np.zeros(0)
        if array.dtype == np.int64:
            array = array.astype(np.int32)
        if not np.little_endian:
            array = array.byteswap()
        return memoryview(np.ascontiguousarray(array))

    def deblob(self, buf, dtype=float, shape=None):
        """Convert blob/buffer object to ndarray of correct dtype and shape.

        (without creating an extra view)."""
        if buf is None:
            return None
        if len(buf) == 0:
            array = np.zeros(0, dtype)
        else:
            array = np.frombuffer(buf, dtype)
            if not np.little_endian:
                array = array.byteswap()
        if shape is not None:
            array.shape = shape
        return array

    def _connect(self):
        return sqlite3.connect(self.filename, timeout=20)

    def __enter__(self):
        assert self.connection is None
        self.change_count = 0
        self.connection = self._connect()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is None:
            self.connection.commit()
        else:
            self.connection.rollback()
        self.connection.close()
        self.connection = None

    @contextmanager
    def managed_connection(self, commit_frequency=5000):
        try:
            con = self.connection or self._connect()
            self._initialize(con)
            yield con
        except ValueError as exc:
            if self.connection is None:
                con.close()
            raise exc
        else:
            if self.connection is None:
                con.commit()
                con.close()
            else:
                self.change_count += 1
                if self.change_count % commit_frequency == 0:
                    con.commit()

    def _initialize(self, con):
        if self.initialized:
            return

        self._metadata = {}

        cur = con.execute(
            'SELECT COUNT(*) FROM sqlite_master WHERE name="systems"')

        if cur.fetchone()[0] == 0:
            for statement in init_statements:
                con.execute(statement)
            if self.create_indices:
                for statement in index_statements:
                    con.execute(statement)
            con.commit()
            self.version = VERSION
        else:
            cur = con.execute(
                'SELECT COUNT(*) FROM sqlite_master WHERE name="user_index"')
            if cur.fetchone()[0] == 1:
                # Old version with "user" instead of "username" column
                self.version = 1
            else:
                try:
                    cur = con.execute(
                        'SELECT value FROM information WHERE name="version"')
                except sqlite3.OperationalError:
                    self.version = 2
                else:
                    self.version = int(cur.fetchone()[0])

                cur = con.execute(
                    'SELECT value FROM information WHERE name="metadata"')
                results = cur.fetchall()
                if results:
                    self._metadata = json.loads(results[0][0])

        if self.version > VERSION:
            raise OSError('Can not read new ase.db format '
                          '(version {}).  Please update to latest ASE.'
                          .format(self.version))
        if self.version < 5 and not self._allow_reading_old_format:
            raise OSError('Please convert to new format. ' +
                          'Use: python -m ase.db.convert ' + self.filename)

        self.initialized = True

    def _write(self, atoms, key_value_pairs, data, id):
        ext_tables = key_value_pairs.pop("external_tables", {})
        Database._write(self, atoms, key_value_pairs, data)

        mtime = now()

        encode = self.encode
        blob = self.blob

        if not isinstance(atoms, AtomsRow):
            row = AtomsRow(atoms)
            row.ctime = mtime
            row.user = os.getenv('USER')
        else:
            row = atoms
            # Extract the external tables from AtomsRow
            names = self._get_external_table_names()
            for name in names:
                new_table = row.get(name, {})
                if new_table:
                    ext_tables[name] = new_table

        if not id and not key_value_pairs and not ext_tables:
            key_value_pairs = row.key_value_pairs

        for k, v in ext_tables.items():
            dtype = self._guess_type(v)
            self._create_table_if_not_exists(k, dtype)

        constraints = row._constraints
        if constraints:
            if isinstance(constraints, list):
                constraints = encode(constraints)
        else:
            constraints = None

        values = (row.unique_id,
                  row.ctime,
                  mtime,
                  row.user,
                  blob(row.numbers),
                  blob(row.positions),
                  blob(row.cell),
                  int(np.dot(row.pbc, [1, 2, 4])),
                  blob(row.get('initial_magmoms')),
                  blob(row.get('initial_charges')),
                  blob(row.get('masses')),
                  blob(row.get('tags')),
                  blob(row.get('momenta')),
                  constraints)

        if 'calculator' in row:
            values += (row.calculator, encode(row.calculator_parameters))
        else:
            values += (None, None)

        if not data:
            data = row._data

        with self.managed_connection() as con:
            if not isinstance(data, (str, bytes)):
                data = encode(data, binary=self.version >= 9)

            values += (float_if_not_none(row.get('energy')),
                       float_if_not_none(row.get('free_energy')),
                       blob(row.get('forces')),
                       blob(row.get('stress')),
                       blob(row.get('dipole')),
                       blob(row.get('magmoms')),
                       row.get('magmom'),
                       blob(row.get('charges')),
                       encode(key_value_pairs),
                       data,
                       len(row.numbers),
                       float_if_not_none(row.get('fmax')),
                       float_if_not_none(row.get('smax')),
                       float_if_not_none(row.get('volume')),
                       float(row.mass),
                       float(row.charge))

            cur = con.cursor()
            if id is None:
                q = self.default + ', ' + ', '.join('?' * len(values))
                cur.execute(f'INSERT INTO systems VALUES ({q})',
                            values)
                id = self.get_last_id(cur)
            else:
                self._delete(cur, [id], ['keys', 'text_key_values',
                                         'number_key_values', 'species'])
                q = ', '.join(name + '=?' for name in self.columnnames[1:])
                cur.execute(f'UPDATE systems SET {q} WHERE id=?',
                            values + (id,))

            count = row.count_atoms()
            if count:
                species = [(atomic_numbers[symbol], n, id)
                           for symbol, n in count.items()]
                cur.executemany('INSERT INTO species VALUES (?, ?, ?)',
                                species)

            text_key_values = []
            number_key_values = []
            for key, value in key_value_pairs.items():
                if isinstance(value, (numbers.Real, np.bool_)):
                    number_key_values.append([key, float(value), id])
                else:
                    assert isinstance(value, str)
                    text_key_values.append([key, value, id])

            cur.executemany('INSERT INTO text_key_values VALUES (?, ?, ?)',
                            text_key_values)
            cur.executemany('INSERT INTO number_key_values VALUES (?, ?, ?)',
                            number_key_values)
            cur.executemany('INSERT INTO keys VALUES (?, ?)',
                            [(key, id) for key in key_value_pairs])

            # Insert entries in the valid tables
            for tabname in ext_tables.keys():
                entries = ext_tables[tabname]
                entries['id'] = id
                self._insert_in_external_table(
                    cur, name=tabname, entries=ext_tables[tabname])

        return id

    def _update(self, id, key_value_pairs, data=None):
        """Update key_value_pairs and data for a single row """
        encode = self.encode
        ext_tables = key_value_pairs.pop('external_tables', {})

        for k, v in ext_tables.items():
            dtype = self._guess_type(v)
            self._create_table_if_not_exists(k, dtype)

        mtime = now()
        with self.managed_connection() as con:
            cur = con.cursor()
            cur.execute(
                'UPDATE systems SET mtime=?, key_value_pairs=? WHERE id=?',
                (mtime, encode(key_value_pairs), id))
            if data:
                if not isinstance(data, (str, bytes)):
                    data = encode(data, binary=self.version >= 9)
                cur.execute('UPDATE systems set data=? where id=?', (data, id))

            self._delete(cur, [id], ['keys', 'text_key_values',
                                     'number_key_values'])

            text_key_values = []
            number_key_values = []
            for key, value in key_value_pairs.items():
                if isinstance(value, (numbers.Real, np.bool_)):
                    number_key_values.append([key, float(value), id])
                else:
                    assert isinstance(value, str)
                    text_key_values.append([key, value, id])

            cur.executemany('INSERT INTO text_key_values VALUES (?, ?, ?)',
                            text_key_values)
            cur.executemany('INSERT INTO number_key_values VALUES (?, ?, ?)',
                            number_key_values)
            cur.executemany('INSERT INTO keys VALUES (?, ?)',
                            [(key, id) for key in key_value_pairs])

            # Insert entries in the valid tables
            for tabname in ext_tables.keys():
                entries = ext_tables[tabname]
                entries['id'] = id
                self._insert_in_external_table(
                    cur, name=tabname, entries=ext_tables[tabname])

        return id

    def get_last_id(self, cur):
        cur.execute('SELECT seq FROM sqlite_sequence WHERE name="systems"')
        result = cur.fetchone()
        if result is not None:
            id = result[0]
            return id
        else:
            return 0

    def _get_row(self, id):
        with self.managed_connection() as con:
            cur = con.cursor()
            if id is None:
                cur.execute('SELECT COUNT(*) FROM systems')
                assert cur.fetchone()[0] == 1
                cur.execute('SELECT * FROM systems')
            else:
                cur.execute('SELECT * FROM systems WHERE id=?', (id,))
            values = cur.fetchone()

        return self._convert_tuple_to_row(values)

    def _convert_tuple_to_row(self, values):
        deblob = self.deblob
        decode = self.decode

        values = self._old2new(values)
        dct = {'id': values[0],
               'unique_id': values[1],
               'ctime': values[2],
               'mtime': values[3],
               'user': values[4],
               'numbers': deblob(values[5], np.int32),
               'positions': deblob(values[6], shape=(-1, 3)),
               'cell': deblob(values[7], shape=(3, 3))}

        if values[8] is not None:
            dct['pbc'] = (values[8] & np.array([1, 2, 4])).astype(bool)
        if values[9] is not None:
            dct['initial_magmoms'] = deblob(values[9])
        if values[10] is not None:
            dct['initial_charges'] = deblob(values[10])
        if values[11] is not None:
            dct['masses'] = deblob(values[11])
        if values[12] is not None:
            dct['tags'] = deblob(values[12], np.int32)
        if values[13] is not None:
            dct['momenta'] = deblob(values[13], shape=(-1, 3))
        if values[14] is not None:
            dct['constraints'] = values[14]
        if values[15] is not None:
            dct['calculator'] = values[15]
        if values[16] is not None:
            dct['calculator_parameters'] = decode(values[16])
        if values[17] is not None:
            dct['energy'] = values[17]
        if values[18] is not None:
            dct['free_energy'] = values[18]
        if values[19] is not None:
            dct['forces'] = deblob(values[19], shape=(-1, 3))
        if values[20] is not None:
            dct['stress'] = deblob(values[20])
        if values[21] is not None:
            dct['dipole'] = deblob(values[21])
        if values[22] is not None:
            dct['magmoms'] = deblob(values[22])
        if values[23] is not None:
            dct['magmom'] = values[23]
        if values[24] is not None:
            dct['charges'] = deblob(values[24])
        if values[25] != '{}':
            dct['key_value_pairs'] = decode(values[25])
        if len(values) >= 27 and values[26] != 'null':
            dct['data'] = decode(values[26], lazy=True)

        # Now we need to update with info from the external tables
        external_tab = self._get_external_table_names()
        tables = {}
        for tab in external_tab:
            row = self._read_external_table(tab, dct["id"])
            tables[tab] = row

        dct.update(tables)
        return AtomsRow(dct)

    def _old2new(self, values):
        if self.type == 'postgresql':
            assert self.version >= 8, 'Your db-version is too old!'
        assert self.version >= 4, 'Your db-file is too old!'
        if self.version < 5:
            pass  # should be ok for reading by convert.py script
        if self.version < 6:
            m = values[23]
            if m is not None and not isinstance(m, float):
                magmom = float(self.deblob(m, shape=()))
                values = values[:23] + (magmom,) + values[24:]
        return values

    def create_select_statement(self, keys, cmps,
                                sort=None, order=None, sort_table=None,
                                what='systems.*'):
        tables = ['systems']
        where = []
        args = []
        for key in keys:
            if key == 'forces':
                where.append('systems.fmax IS NOT NULL')
            elif key == 'strain':
                where.append('systems.smax IS NOT NULL')
            elif key in ['energy', 'fmax', 'smax',
                         'constraints', 'calculator']:
                where.append(f'systems.{key} IS NOT NULL')
            else:
                if '-' not in key:
                    q = 'systems.id in (select id from keys where key=?)'
                else:
                    key = key.replace('-', '')
                    q = 'systems.id not in (select id from keys where key=?)'
                where.append(q)
                args.append(key)

        # Special handling of "H=0" and "H<2" type of selections:
        bad = {}
        for key, op, value in cmps:
            if isinstance(key, int):
                bad[key] = bad.get(key, True) and ops[op](0, value)

        for key, op, value in cmps:
            if key in ['id', 'energy', 'magmom', 'ctime', 'user',
                       'calculator', 'natoms', 'pbc', 'unique_id',
                       'fmax', 'smax', 'volume', 'mass', 'charge']:
                if key == 'user':
                    key = 'username'
                elif key == 'pbc':
                    assert op in ['=', '!=']
                    value = int(np.dot([x == 'T' for x in value], [1, 2, 4]))
                elif key == 'magmom':
                    assert self.version >= 6, 'Update your db-file'
                where.append(f'systems.{key}{op}?')
                args.append(value)
            elif isinstance(key, int):
                if self.type == 'postgresql':
                    where.append(
                        'cardinality(array_positions(' +
                        f'numbers::int[], ?)){op}?')
                    args += [key, value]
                else:
                    if bad[key]:
                        where.append(
                            'systems.id not in (select id from species ' +
                            f'where Z=? and n{invop[op]}?)')
                        args += [key, value]
                    else:
                        where.append('systems.id in (select id from species ' +
                                     f'where Z=? and n{op}?)')
                        args += [key, value]

            elif self.type == 'postgresql':
                jsonop = '->'
                if isinstance(value, str):
                    jsonop = '->>'
                elif isinstance(value, bool):
                    jsonop = '->>'
                    value = str(value).lower()
                where.append("systems.key_value_pairs {} '{}'{}?"
                             .format(jsonop, key, op))
                args.append(str(value))

            elif isinstance(value, str):
                where.append('systems.id in (select id from text_key_values ' +
                             f'where key=? and value{op}?)')
                args += [key, value]
            else:
                where.append(
                    'systems.id in (select id from number_key_values ' +
                    f'where key=? and value{op}?)')
                args += [key, float(value)]

        if sort:
            if sort_table != 'systems':
                tables.append(f'{sort_table} AS sort_table')
                where.append('systems.id=sort_table.id AND '
                             'sort_table.key=?')
                args.append(sort)
                sort_table = 'sort_table'
                sort = 'value'

        sql = f'SELECT {what} FROM\n  ' + ', '.join(tables)
        if where:
            sql += '\n  WHERE\n  ' + ' AND\n  '.join(where)
        if sort:
            # XXX use "?" instead of "{}"
            sql += '\nORDER BY {0}.{1} IS NULL, {0}.{1} {2}'.format(
                sort_table, sort, order)

        return sql, args

    def _select(self, keys, cmps, explain=False, verbosity=0,
                limit=None, offset=0, sort=None, include_data=True,
                columns='all'):

        values = np.array([None for _ in range(27)])
        values[25] = '{}'
        values[26] = 'null'

        if columns == 'all':
            columnindex = list(range(26))
        else:
            columnindex = [c for c in range(26)
                           if self.columnnames[c] in columns]
        if include_data:
            columnindex.append(26)

        if sort:
            if sort[0] == '-':
                order = 'DESC'
                sort = sort[1:]
            else:
                order = 'ASC'
            if sort in ['id', 'energy', 'username', 'calculator',
                        'ctime', 'mtime', 'magmom', 'pbc',
                        'fmax', 'smax', 'volume', 'mass', 'charge', 'natoms']:
                sort_table = 'systems'
            else:
                for dct in self._select(keys + [sort], cmps=[], limit=1,
                                        include_data=False,
                                        columns=['key_value_pairs']):
                    if isinstance(dct['key_value_pairs'][sort], str):
                        sort_table = 'text_key_values'
                    else:
                        sort_table = 'number_key_values'
                    break
                else:
                    # No rows.  Just pick a table:
                    sort_table = 'number_key_values'

        else:
            order = None
            sort_table = None

        what = ', '.join('systems.' + name
                         for name in
                         np.array(self.columnnames)[np.array(columnindex)])

        sql, args = self.create_select_statement(keys, cmps, sort, order,
                                                 sort_table, what)

        if explain:
            sql = 'EXPLAIN QUERY PLAN ' + sql

        if limit:
            sql += f'\nLIMIT {limit}'

        if offset:
            sql += self.get_offset_string(offset, limit=limit)

        if verbosity == 2:
            print(sql, args)

        with self.managed_connection() as con:
            cur = con.cursor()
            cur.execute(sql, args)
            if explain:
                for row in cur.fetchall():
                    yield {'explain': row}
            else:
                n = 0
                for shortvalues in cur.fetchall():
                    values[columnindex] = shortvalues
                    yield self._convert_tuple_to_row(tuple(values))
                    n += 1

                if sort and sort_table != 'systems':
                    # Yield rows without sort key last:
                    if limit is not None:
                        if n == limit:
                            return
                        limit -= n
                    for row in self._select(keys + ['-' + sort], cmps,
                                            limit=limit, offset=offset,
                                            include_data=include_data,
                                            columns=columns):
                        yield row

    def get_offset_string(self, offset, limit=None):
        sql = ''
        if not limit:
            # In sqlite you cannot have offset without limit, so we
            # set it to -1 meaning no limit
            sql += '\nLIMIT -1'
        sql += f'\nOFFSET {offset}'
        return sql

    @parallel_function
    def count(self, selection=None, **kwargs):
        keys, cmps = parse_selection(selection, **kwargs)
        sql, args = self.create_select_statement(keys, cmps, what='COUNT(*)')

        with self.managed_connection() as con:
            cur = con.cursor()
            cur.execute(sql, args)
            return cur.fetchone()[0]

    def analyse(self):
        with self.managed_connection() as con:
            con.execute('ANALYZE')

    @parallel_function
    @lock
    def delete(self, ids):
        if len(ids) == 0:
            return
        table_names = self._get_external_table_names() + all_tables[::-1]
        with self.managed_connection() as con:
            self._delete(con.cursor(), ids,
                         tables=table_names)
        self.vacuum()

    def _delete(self, cur, ids, tables=None):
        tables = tables or all_tables[::-1]
        for table in tables:
            cur.execute('DELETE FROM {} WHERE id in ({});'.
                        format(table, ', '.join([str(id) for id in ids])))

    def vacuum(self):
        if self.type != 'db':
            return

        with self.managed_connection() as con:
            con.commit()
            con.cursor().execute("VACUUM")

    @property
    def metadata(self):
        if self._metadata is None:
            self._initialize(self._connect())
        return self._metadata.copy()

    @metadata.setter
    def metadata(self, dct):
        self._metadata = dct
        md = json.dumps(dct)
        with self.managed_connection() as con:
            cur = con.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM information WHERE name='metadata'")

            if cur.fetchone()[0]:
                cur.execute(
                    "UPDATE information SET value=? WHERE name='metadata'",
                    [md])
            else:
                cur.execute('INSERT INTO information VALUES (?, ?)',
                            ('metadata', md))

    def _get_external_table_names(self, db_con=None):
        """Return a list with the external table names."""
        sql = "SELECT value FROM information WHERE name='external_table_name'"
        with self.managed_connection() as con:
            cur = con.cursor()
            cur.execute(sql)
            ext_tab_names = [x[0] for x in cur.fetchall()]
        return ext_tab_names

    def _external_table_exists(self, name):
        """Return True if an external table name exists."""
        return name in self._get_external_table_names()

    def _create_table_if_not_exists(self, name, dtype):
        """Create a new table if it does not exits.

        Arguments
        ==========
        name: str
            Name of the new table
        dtype: str
            Datatype of the value field (typically REAL, INTEGER, TEXT etc.)
        """

        taken_names = set(all_tables + all_properties + self.columnnames)
        if name in taken_names:
            raise ValueError("External table can not be any of {}"
                             "".format(taken_names))

        if self._external_table_exists(name):
            return

        sql = f"CREATE TABLE IF NOT EXISTS {name} "
        sql += f"(key TEXT, value {dtype}, id INTEGER, "
        sql += "FOREIGN KEY (id) REFERENCES systems(id))"
        sql2 = "INSERT INTO information VALUES (?, ?)"
        with self.managed_connection() as con:
            cur = con.cursor()
            cur.execute(sql)
            # Insert an entry saying that there is a new external table
            # present and an entry with the datatype
            cur.execute(sql2, ("external_table_name", name))
            cur.execute(sql2, (name + "_dtype", dtype))

    def delete_external_table(self, name):
        """Delete an external table."""
        if not self._external_table_exists(name):
            return

        with self.managed_connection() as con:
            cur = con.cursor()

            sql = f"DROP TABLE {name}"
            cur.execute(sql)

            sql = "DELETE FROM information WHERE value=?"
            cur.execute(sql, (name,))
            sql = "DELETE FROM information WHERE name=?"
            cur.execute(sql, (name + "_dtype",))

    def _convert_to_recognized_types(self, value):
        """Convert Numpy types to python types."""
        if np.issubdtype(type(value), np.integer):
            return int(value)
        elif np.issubdtype(type(value), np.floating):
            return float(value)
        return value

    def _insert_in_external_table(self, cursor, name=None, entries=None):
        """Insert into external table"""
        if name is None or entries is None:
            # There is nothing to do
            return

        id = entries.pop("id")
        dtype = self._guess_type(entries)
        expected_dtype = self._get_value_type_of_table(cursor, name)
        if dtype != expected_dtype:
            raise ValueError("The provided data type for table {} "
                             "is {}, while it is initialized to "
                             "be of type {}"
                             "".format(name, dtype, expected_dtype))

        # First we check if entries already exists
        cursor.execute(f"SELECT key FROM {name} WHERE id=?", (id,))
        updates = []
        for item in cursor.fetchall():
            value = entries.pop(item[0], None)
            if value is not None:
                updates.append(
                    (value, id, self._convert_to_recognized_types(item[0])))

        # Update entry if key and ID already exists
        sql = f"UPDATE {name} SET value=? WHERE id=? AND key=?"
        cursor.executemany(sql, updates)

        # Insert the ones that does not already exist
        inserts = [(k, self._convert_to_recognized_types(v), id)
                   for k, v in entries.items()]
        sql = f"INSERT INTO {name} VALUES (?, ?, ?)"
        cursor.executemany(sql, inserts)

    def _guess_type(self, entries):
        """Guess the type based on the first entry."""
        values = [v for _, v in entries.items()]

        # Check if all datatypes are the same
        all_types = [type(v) for v in values]
        if any(t != all_types[0] for t in all_types):
            typenames = [t.__name__ for t in all_types]
            raise ValueError("Inconsistent datatypes in the table. "
                             "given types: {}".format(typenames))

        val = values[0]
        if isinstance(val, int) or np.issubdtype(type(val), np.integer):
            return "INTEGER"
        if isinstance(val, float) or np.issubdtype(type(val), np.floating):
            return "REAL"
        if isinstance(val, str):
            return "TEXT"
        raise ValueError("Unknown datatype!")

    def _get_value_type_of_table(self, cursor, tab_name):
        """Return the expected value name."""
        sql = "SELECT value FROM information WHERE name=?"
        cursor.execute(sql, (tab_name + "_dtype",))
        return cursor.fetchone()[0]

    def _read_external_table(self, name, id):
        """Read row from external table."""

        with self.managed_connection() as con:
            cur = con.cursor()
            cur.execute(f"SELECT * FROM {name} WHERE id=?", (id,))
            items = cur.fetchall()
            dictionary = {item[0]: item[1] for item in items}

        return dictionary

    def get_all_key_names(self):
        """Create set of all key names."""
        with self.managed_connection() as con:
            cur = con.cursor()
            cur.execute('SELECT DISTINCT key FROM keys;')
            all_keys = {row[0] for row in cur.fetchall()}
        return all_keys


if __name__ == '__main__':
    from ase.db import connect
    con = connect(sys.argv[1])
    con._initialize(con._connect())
    print('Version:', con.version)
