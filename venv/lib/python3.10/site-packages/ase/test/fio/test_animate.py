from ase.build import bulk, fcc111, molecule
from ase.io.animation import write_animation


def test_animate(plt):
    images = [molecule('H2O'), bulk('Cu'), fcc111('Au', size=(1, 1, 1))]

    # gif and mp4 writers may not be available.  Easiest solution is to only
    # test this using the html writer because it always exists whenever
    # matplotlib exists:
    write_animation('things.html', images, writer='html')
