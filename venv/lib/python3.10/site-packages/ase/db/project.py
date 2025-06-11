# fmt: off

from pathlib import Path

from ase.db.core import KeyDescription
from ase.db.row import row2dct
from ase.formula import Formula


class DatabaseProject:
    """Settings for web view of a database.

    For historical reasons called a "Project".
    """
    _ase_templates = Path('ase/db/templates')

    def __init__(self, name, title, *,
                 key_descriptions,
                 database,
                 default_columns):
        self.name = name
        self.title = title
        self.uid_key = 'id'

        # The templates loop over "key descriptions" when they want to
        # loop over keys.
        #
        # Therefore, any key without description will not be rendered.
        # Therefore, we need to make dummy key descriptions of everything
        # in the database, ensuring that all keys are visible.

        all_keys = database.get_all_key_names()

        key_descriptions = {
            **{key: KeyDescription(key) for key in all_keys},
            **key_descriptions}

        for key, value in key_descriptions.items():
            assert isinstance(key, str), type(key)
            assert isinstance(value, KeyDescription), type(value)

        self.key_descriptions = key_descriptions
        self.database = database
        self.default_columns = default_columns

    def get_search_template(self):
        return self._ase_templates / 'search.html'

    def get_row_template(self):
        return self._ase_templates / 'row.html'

    def get_table_template(self):
        return self._ase_templates / 'table.html'

    def handle_query(self, args) -> str:
        """Convert request args to ase.db query string."""
        return args['query']

    def row_to_dict(self, row):
        """Convert row to dict for use in html template."""
        dct = row2dct(row, self.key_descriptions)
        dct['formula'] = Formula(row.formula).convert('abc').format('html')
        return dct

    def uid_to_row(self, uid):
        return self.database.get(f'{self.uid_key}={uid}')

    @classmethod
    def dummyproject(cls, **kwargs):
        class DummyDatabase:
            def select(self, *args, **kwargs):
                return iter([])

            def get_all_key_names(self):
                return set()

        _kwargs = dict(
            name='test',
            title='test',
            key_descriptions={},
            database=DummyDatabase(),  # XXX
            default_columns=[])
        _kwargs.update(kwargs)
        return cls(**_kwargs)

    # If we make this a classmethod, and try to instantiate the class,
    # it would fail on subclasses.  So we use staticmethod
    @staticmethod
    def load_db_as_ase_project(name, database):
        from ase.db.core import get_key_descriptions
        from ase.db.table import all_columns

        return DatabaseProject(
            name=name,
            title=database.metadata.get('title', ''),
            key_descriptions=get_key_descriptions(),
            database=database,
            default_columns=all_columns)
