from pathlib import Path

from ase.lattice import RHL


def test_ase_bandstructure(cli, plt, testdir):
    lat = RHL(3., 70.0)
    path = lat.bandpath()
    bs = path.free_electron_band_structure()

    bs_path = Path('bs.json')
    bs.write(bs_path)

    fig_path = Path('bs.png')

    cli.ase('band-structure', str(bs_path), '--output', str(fig_path))
    # If the CLI tool gave a text output, we could verify it.
    assert fig_path.is_file()

# Note: We don't have proper testing of --points, --range etc.  We
# test here on JSON input but the tool is in principle supposed to
# work on other formats, too (only gpw though as of now though).
