"""Tests for PubChem."""

import json
from io import BytesIO

import pytest

from ase.data import pubchem
from ase.data.pubchem import (
    analyze_input,
    pubchem_atoms_conformer_search,
    pubchem_atoms_search,
    pubchem_conformer_search,
    pubchem_search,
)


@pytest.fixture
def mock_pubchem(monkeypatch) -> None:
    """Mock `pubchem`."""

    def mock_search_pubchem_raw_222(*args, **kwargs):
        """Mock `search_pubchem_raw` for ammonia (CID=222)."""
        data222 = b'222\n  -OEChem-10071914343D\n\n  4  3  0     0  0  0  0  0  0999 V2000\n    0.0000    0.0000    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0\n   -0.4417    0.2906    0.8711 H   0  0  0  0  0  0  0  0  0  0  0  0\n    0.7256    0.6896   -0.1907 H   0  0  0  0  0  0  0  0  0  0  0  0\n    0.4875   -0.8701    0.2089 H   0  0  0  0  0  0  0  0  0  0  0  0\n  1  2  1  0  0  0  0\n  1  3  1  0  0  0  0\n  1  4  1  0  0  0  0\nM  END\n> <PUBCHEM_COMPOUND_CID>\n222\n\n> <PUBCHEM_CONFORMER_RMSD>\n0.4\n\n> <PUBCHEM_CONFORMER_DIVERSEORDER>\n1\n\n> <PUBCHEM_MMFF94_PARTIAL_CHARGES>\n4\n1 -1.08\n2 0.36\n3 0.36\n4 0.36\n\n> <PUBCHEM_EFFECTIVE_ROTOR_COUNT>\n0\n\n> <PUBCHEM_PHARMACOPHORE_FEATURES>\n1\n1 1 cation\n\n> <PUBCHEM_HEAVY_ATOM_COUNT>\n1\n\n> <PUBCHEM_ATOM_DEF_STEREO_COUNT>\n0\n\n> <PUBCHEM_ATOM_UDEF_STEREO_COUNT>\n0\n\n> <PUBCHEM_BOND_DEF_STEREO_COUNT>\n0\n\n> <PUBCHEM_BOND_UDEF_STEREO_COUNT>\n0\n\n> <PUBCHEM_ISOTOPIC_ATOM_COUNT>\n0\n\n> <PUBCHEM_COMPONENT_COUNT>\n1\n\n> <PUBCHEM_CACTVS_TAUTO_COUNT>\n1\n\n> <PUBCHEM_CONFORMER_ID>\n000000DE00000001\n\n> <PUBCHEM_MMFF94_ENERGY>\n0\n\n> <PUBCHEM_FEATURE_SELFOVERLAP>\n5.074\n\n> <PUBCHEM_SHAPE_FINGERPRINT>\n260 1 18410856563934756871\n\n> <PUBCHEM_SHAPE_MULTIPOLES>\n15.6\n0.51\n0.51\n0.51\n0\n0\n0\n0\n0\n0\n0\n0\n0\n0\n\n> <PUBCHEM_SHAPE_SELFOVERLAP>\n14.89\n\n> <PUBCHEM_SHAPE_VOLUME>\n15.6\n\n> <PUBCHEM_COORDINATE_TYPE>\n2\n5\n10\n\n$$$$\n'  # noqa
        r = BytesIO(data222)
        return r.read().decode('utf-8')

    def mock_available_conformer_search_222(*args, **kwargs):
        """Mock `available_conformer_search` for ammonia (CID=222)."""
        conformer222 = b'{\n  "InformationList": {\n    "Information": [\n      {\n        "CID": 222,\n        "ConformerID": [\n          "000000DE00000001"\n        ]\n      }\n    ]\n  }\n}\n'  # noqa
        r = BytesIO(conformer222)
        record = r.read().decode('utf-8')
        record = json.loads(record)
        return record['InformationList']['Information'][0]['ConformerID']

    f = mock_search_pubchem_raw_222
    monkeypatch.setattr(pubchem, 'search_pubchem_raw', f)

    f = mock_available_conformer_search_222
    monkeypatch.setattr(pubchem, 'available_conformer_search', f)


def test_pubchem_search(mock_pubchem) -> None:
    """Test if `pubchem_search` handles given arguments correctly."""
    data = pubchem_search('ammonia')

    atoms = data.get_atoms()
    assert atoms.get_chemical_symbols() == ['N', 'H', 'H', 'H']

    data.get_pubchem_data()


def test_pubchem(mock_pubchem) -> None:
    # check the various entry styles and the functions that return atoms
    pubchem_search(cid=241).get_atoms()
    pubchem_atoms_search(smiles='CCOH')
    pubchem_atoms_conformer_search('octane')


def test_pubchem_conformer_search(mock_pubchem) -> None:
    """Test if `pubchem_conformer_search` runs as expected."""
    confs = pubchem_conformer_search('octane')
    for _ in confs:
        pass


def test_multiple_search(mock_pubchem) -> None:
    """Check that you can't pass in two args."""
    with pytest.raises(ValueError):
        pubchem_search(name='octane', cid=222)


def test_empty_search(mock_pubchem) -> None:
    """Check that you must pass at least one arg."""
    with pytest.raises(ValueError):
        pubchem_search()


def test_triple_bond() -> None:
    """Check if hash (`#`) is converted to hex (`%23`)."""
    assert analyze_input(smiles='CC#N')[0] == 'CC%23N'
