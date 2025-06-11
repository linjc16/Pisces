# fmt: off

# Note:
# Try to avoid module level import statements here to reduce
# import time during CLI execution


class CLICommand:
    """Manipulate and query ASE database.

    Query is a comma-separated list of
    selections where each selection is of the type "ID", "key" or
    "key=value".  Instead of "=", one can also use "<", "<=", ">=", ">"
    and  "!=" (these must be protected from the shell by using quotes).
    Special keys:

    * id
    * user
    * calculator
    * age
    * natoms
    * energy
    * magmom
    * charge

    Chemical symbols can also be used to select number of
    specific atomic species (H, He, Li, ...).  Selection examples:

        calculator=nwchem
        age<1d
        natoms=1
        user=alice
        2.2<bandgap<4.1
        Cu>=10

    See also: https://wiki.fysik.dtu.dk/ase/ase/db/db.html.
    """

    @staticmethod
    def add_arguments(parser):
        add = parser.add_argument
        add('database', help='SQLite3 file, JSON file or postgres URL.')
        add('query', nargs='*', help='Query string.')
        add('-v', '--verbose', action='store_true', help='More output.')
        add('-q', '--quiet', action='store_true', help='Less output.')
        add('-n', '--count', action='store_true',
            help='Count number of selected rows.')
        add('-l', '--long', action='store_true',
            help='Long description of selected row')
        add('-i', '--insert-into', metavar='db-name',
            help='Insert selected rows into another database.')
        add('-a', '--add-from-file', metavar='filename',
            help='Add configuration(s) from file.  '
            'If the file contains more than one configuration then you can '
            'use the syntax filename@: to add all of them.  Default is to '
            'only add the last.')
        add('-k', '--add-key-value-pairs', metavar='key1=val1,key2=val2,...',
            help='Add key-value pairs to selected rows.  Values must '
            'be numbers or strings and keys must follow the same rules as '
            'keywords.')
        add('-L', '--limit', type=int, default=-1, metavar='N',
            help='Show only first N rows.  Use --limit=0 '
            'to show all.  Default is 20 rows when listing rows and no '
            'limit when --insert-into is used.')
        add('--offset', type=int, default=0, metavar='N',
            help='Skip first N rows.  By default, no rows are skipped')
        add('--delete', action='store_true',
            help='Delete selected rows.')
        add('--delete-keys', metavar='key1,key2,...',
            help='Delete keys for selected rows.')
        add('-y', '--yes', action='store_true',
            help='Say yes.')
        add('--explain', action='store_true',
            help='Explain query plan.')
        add('-c', '--columns', metavar='col1,col2,...',
            help='Specify columns to show.  Precede the column specification '
            'with a "+" in order to add columns to the default set of '
            'columns.  Precede by a "-" to remove columns.  Use "++" for all.')
        add('-s', '--sort', metavar='column', default='id',
            help='Sort rows using "column".  Use "column-" for a descending '
            'sort.  Default is to sort after id.')
        add('--cut', type=int, default=35, help='Cut keywords and key-value '
            'columns after CUT characters.  Use --cut=0 to disable cutting. '
            'Default is 35 characters')
        add('-p', '--plot', metavar='x,y1,y2,...',
            help='Example: "-p x,y": plot y row against x row. Use '
            '"-p a:x,y" to make a plot for each value of a.')
        add('--csv', action='store_true',
            help='Write comma-separated-values file.')
        add('-w', '--open-web-browser', action='store_true',
            help='Open results in web-browser.')
        add('--no-lock-file', action='store_true', help="Don't use lock-files")
        add('--analyse', action='store_true',
            help='Gathers statistics about tables and indices to help make '
            'better query planning choices.')
        add('-j', '--json', action='store_true',
            help='Write json representation of selected row.')
        add('-m', '--show-metadata', action='store_true',
            help='Show metadata as json.')
        add('--set-metadata', metavar='something.json',
            help='Set metadata from a json file.')
        add('--strip-data', action='store_true',
            help='Strip data when using --insert-into.')
        add('--progress-bar', action='store_true',
            help='Show a progress bar when using --insert-into.')
        add('--show-keys', action='store_true',
            help='Show all keys.')
        add('--show-values', metavar='key1,key2,...',
            help='Show values for key(s).')

    @staticmethod
    def run(args):
        from ase.db.cli import main

        main(args)
