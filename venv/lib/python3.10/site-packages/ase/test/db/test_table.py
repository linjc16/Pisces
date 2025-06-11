from types import SimpleNamespace

from ase.db.table import Table


class TestConnection:
    def select(self,
               query,
               verbosity,
               limit,
               offset,
               sort,
               include_data,
               columns):
        return [SimpleNamespace(id=1, a='hello'),
                SimpleNamespace(id=2, a='hi!!!', b=117)]


def test_hide_empty_columns():
    db = TestConnection()
    table = Table(db)
    for show in [True, False]:
        table.select('...', ['a', 'b', 'c'], '', 10, 0,
                     show_empty_columns=show)
        if show:
            assert table.columns == ['a', 'b', 'c']
        else:
            assert table.columns == ['a', 'b']
