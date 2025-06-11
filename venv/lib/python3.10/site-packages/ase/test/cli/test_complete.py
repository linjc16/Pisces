"""Check that our tab-completion script has been updated."""
from ase.cli.completion import path, update
from ase.cli.main import commands


def test_complete():
    update(path, commands, test=True)
