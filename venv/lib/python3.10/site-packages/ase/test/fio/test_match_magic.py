from ase.io.formats import ioformats


# flake8: noqa
def test_match_magic():

    text = b"""

      ___ ___ ___ _ _ _  
     |   |   |_  | | | | 
     | | | | | . | | | | 
     |__ |  _|___|_____|  19.8.2b1
     |___|_|             

    """

    gpaw = ioformats['gpaw-out']
    assert gpaw.match_magic(text)
