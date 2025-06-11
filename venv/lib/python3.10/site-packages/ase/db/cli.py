# fmt: off

import json
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

import ase.io
from ase.db import connect
from ase.db.core import convert_str_to_int_float_bool_or_str
from ase.db.row import row2dct
from ase.db.table import Table, all_columns
from ase.utils import plural


def count_keys(db, query):
    keys = defaultdict(int)
    for row in db.select(query):
        for key in row._keys:
            keys[key] += 1

    n = max(len(key) for key in keys) + 1
    for key, number in keys.items():
        print('{:{}} {}'.format(key + ':', n, number))
    return


def main(args):
    verbosity = 1 - args.quiet + args.verbose
    query = ','.join(args.query)

    if args.sort.endswith('-'):
        # Allow using "key-" instead of "-key" for reverse sorting
        args.sort = '-' + args.sort[:-1]

    if query.isdigit():
        query = int(query)

    add_key_value_pairs = {}
    if args.add_key_value_pairs:
        for pair in args.add_key_value_pairs.split(','):
            key, value = pair.split('=')
            add_key_value_pairs[key] = \
                convert_str_to_int_float_bool_or_str(value)

    if args.delete_keys:
        delete_keys = args.delete_keys.split(',')
    else:
        delete_keys = []

    db = connect(args.database, use_lock_file=not args.no_lock_file)

    def out(*args):
        if verbosity > 0:
            print(*args)

    if args.analyse:
        db.analyse()
        return

    if args.show_keys:
        count_keys(db, query)
        return

    if args.show_values:
        keys = args.show_values.split(',')
        values = {key: defaultdict(int) for key in keys}
        numbers = set()
        for row in db.select(query):
            kvp = row.key_value_pairs
            for key in keys:
                value = kvp.get(key)
                if value is not None:
                    values[key][value] += 1
                    if not isinstance(value, str):
                        numbers.add(key)

        n = max(len(key) for key in keys) + 1
        for key in keys:
            vals = values[key]
            if key in numbers:
                print('{:{}} [{}..{}]'
                      .format(key + ':', n, min(vals), max(vals)))
            else:
                print('{:{}} {}'
                      .format(key + ':', n,
                              ', '.join(f'{v}({n})'
                                        for v, n in vals.items())))
        return

    if args.add_from_file:
        filename = args.add_from_file
        configs = ase.io.read(filename)
        if not isinstance(configs, list):
            configs = [configs]
        for atoms in configs:
            db.write(atoms, key_value_pairs=add_key_value_pairs)
        out('Added ' + plural(len(configs), 'row'))
        return

    if args.count:
        n = db.count(query)
        print(f'{plural(n, "row")}')
        return

    if args.insert_into:
        if args.limit == -1:
            args.limit = 0

        progressbar = no_progressbar
        length = None

        if args.progress_bar:
            # Try to import the one from click.
            # People using ase.db will most likely have flask installed
            # and therfore also click.
            try:
                from click import progressbar
            except ImportError:
                pass
            else:
                length = db.count(query)

        nkvp = 0
        nrows = 0
        with connect(args.insert_into,
                     use_lock_file=not args.no_lock_file) as db2:
            with progressbar(db.select(query,
                                       sort=args.sort,
                                       limit=args.limit,
                                       offset=args.offset),
                             length=length) as rows:
                for row in rows:
                    kvp = row.get('key_value_pairs', {})
                    nkvp -= len(kvp)
                    kvp.update(add_key_value_pairs)
                    nkvp += len(kvp)
                    if args.strip_data:
                        db2.write(row.toatoms(), **kvp)
                    else:
                        db2.write(row, data=row.get('data'), **kvp)
                    nrows += 1

        out('Added %s (%s updated)' %
            (plural(nkvp, 'key-value pair'),
             plural(len(add_key_value_pairs) * nrows - nkvp, 'pair')))
        out(f'Inserted {plural(nrows, "row")}')
        return

    if args.limit == -1:
        args.limit = 20

    if args.explain:
        for row in db.select(query, explain=True,
                             verbosity=verbosity,
                             limit=args.limit, offset=args.offset):
            print(row['explain'])
        return

    if args.show_metadata:
        print(json.dumps(db.metadata, sort_keys=True, indent=4))
        return

    if args.set_metadata:
        with open(args.set_metadata) as fd:
            db.metadata = json.load(fd)
        return

    if add_key_value_pairs or delete_keys:
        ids = [row['id'] for row in db.select(query)]
        M = 0
        N = 0
        with db:
            for id in ids:
                m, n = db.update(id, delete_keys=delete_keys,
                                 **add_key_value_pairs)
                M += m
                N += n
        out('Added %s (%s updated)' %
            (plural(M, 'key-value pair'),
             plural(len(add_key_value_pairs) * len(ids) - M, 'pair')))
        out('Removed', plural(N, 'key-value pair'))

        return

    if args.delete:
        ids = [row['id'] for row in db.select(query, include_data=False)]
        if ids and not args.yes:
            msg = f'Delete {plural(len(ids), "row")}? (yes/No): '
            if input(msg).lower() != 'yes':
                return
        db.delete(ids)
        out(f'Deleted {plural(len(ids), "row")}')
        return

    if args.plot:
        if ':' in args.plot:
            tags, keys = args.plot.split(':')
            tags = tags.split(',')
        else:
            tags = []
            keys = args.plot
        keys = keys.split(',')
        plots = defaultdict(list)
        X = {}
        labels = []
        for row in db.select(query, sort=args.sort, include_data=False):
            name = ','.join(str(row[tag]) for tag in tags)
            x = row.get(keys[0])
            if x is not None:
                if isinstance(x, str):
                    if x not in X:
                        X[x] = len(X)
                        labels.append(x)
                    x = X[x]
                plots[name].append([x] + [row.get(key) for key in keys[1:]])
        import matplotlib.pyplot as plt
        for name, plot in plots.items():
            xyy = list(zip(*plot))
            x = xyy[0]
            for y, key in zip(xyy[1:], keys[1:]):
                plt.plot(x, y, label=name + ':' + key)
        if X:
            plt.xticks(range(len(labels)), labels, rotation=90)
        plt.legend()
        plt.show()
        return

    if args.json:
        row = db.get(query)
        db2 = connect(sys.stdout, 'json', use_lock_file=False)
        kvp = row.get('key_value_pairs', {})
        db2.write(row, data=row.get('data'), **kvp)
        return

    if args.long:
        row = db.get(query)
        print(row2str(row))
        return

    if args.open_web_browser:
        try:
            import flask  # noqa
        except ImportError:
            print('Please install Flask: python3 -m pip install flask')
            return
        check_jsmol()
        import ase.db.app as app
        app.DBApp().run_db(db)
        return

    columns = list(all_columns)
    c = args.columns
    if c and c.startswith('++'):
        keys = set()
        for row in db.select(query,
                             limit=args.limit, offset=args.offset,
                             include_data=False):
            keys.update(row._keys)
        columns.extend(keys)
        if c[2:3] == ',':
            c = c[3:]
        else:
            c = ''
    if c:
        if c[0] == '+':
            c = c[1:]
        elif c[0] != '-':
            columns = []
        for col in c.split(','):
            if col[0] == '-':
                columns.remove(col[1:])
            else:
                columns.append(col.lstrip('+'))

    table = Table(db, verbosity=verbosity, cut=args.cut)
    table.select(query, columns, args.sort, args.limit, args.offset)
    if args.csv:
        table.write_csv()
    else:
        table.write(query)


def row2str(row) -> str:
    t = row2dct(row, key_descriptions={})
    S = [t['formula'] + ':',
         'Unit cell in Ang:',
         'axis|periodic|          x|          y|          z|' +
         '    length|     angle']
    c = 1
    fmt = ('   {0}|     {1}|{2[0]:>11}|{2[1]:>11}|{2[2]:>11}|' +
           '{3:>10}|{4:>10}')
    for p, axis, L, A in zip(row.pbc, t['cell'], t['lengths'], t['angles']):
        S.append(fmt.format(c, [' no', 'yes'][p], axis, L, A))
        c += 1
    S.append('')

    if 'stress' in t:
        S += ['Stress tensor (xx, yy, zz, zy, zx, yx) in eV/Ang^3:',
              '   {}\n'.format(t['stress'])]

    if 'dipole' in t:
        S.append('Dipole moment in e*Ang: ({})\n'.format(t['dipole']))

    if 'constraints' in t:
        S.append('Constraints: {}\n'.format(t['constraints']))

    if 'data' in t:
        S.append('Data: {}\n'.format(t['data']))

    width0 = max(max(len(row[0]) for row in t['table']), 3)
    width1 = max(max(len(row[1]) for row in t['table']), 11)
    S.append('{:{}} | {:{}} | Value'
             .format('Key', width0, 'Description', width1))
    for key, desc, value in t['table']:
        S.append('{:{}} | {:{}} | {}'
                 .format(key, width0, desc, width1, value))
    return '\n'.join(S)


@contextmanager
def no_progressbar(iterable: Iterable,
                   length: int = None) -> Iterator[Iterable]:
    """A do-nothing implementation."""
    yield iterable


def check_jsmol():
    static = Path(__file__).parent / 'static'
    if not (static / 'jsmol/JSmol.min.js').is_file():
        print(f"""
    WARNING:
        You don't have jsmol on your system.

        Download Jmol-*-binary.tar.gz from
        https://sourceforge.net/projects/jmol/files/Jmol/,
        extract jsmol.zip, unzip it and create a soft-link:

            $ tar -xf Jmol-*-binary.tar.gz
            $ unzip jmol-*/jsmol.zip
            $ ln -s $PWD/jsmol {static}/jsmol
    """,
              file=sys.stderr)
