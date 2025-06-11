import ase.db
from ase.io.formats import string2index


def read_db(filename, index, **kwargs):
    with ase.db.connect(filename, serial=True, **kwargs) as db:
        if isinstance(index, str):
            try:
                index = string2index(index)
            except ValueError:
                pass

        if isinstance(index, int):
            index = slice(index, index + 1 or None)

        if isinstance(index, str):
            # index is a database query string:
            for row in db.select(index):
                yield row.toatoms()
        else:
            start, stop, step = index.indices(db.count())
            if start == stop:
                return
            assert step == 1
            for row in db.select(offset=start, limit=stop - start):
                yield row.toatoms()


def write_db(filename, images, append=False, **kwargs):
    with ase.db.connect(filename, serial=True, append=append, **kwargs) as con:
        for atoms in images:
            con.write(atoms)


read_json = read_db
write_json = write_db
read_postgresql = read_db
write_postgresql = write_db
read_mysql = read_db
write_mysql = write_db
