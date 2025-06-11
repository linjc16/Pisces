import matplotlib.pyplot as plt
import numpy as np
from matplotlib.testing.compare import compare_images
from scipy.spatial.transform import Rotation

from ase.lattice import HEX2D


def test_repeat_transpose_bz(testdir, datadir) -> None:
    """Testing plot_bz."""

    hex2d = HEX2D(a=1.0)
    r = Rotation.from_rotvec([0, 0, np.deg2rad(10)])
    fig, ax = plt.subplots()
    hex2d.plot_bz(repeat=(2, 1), transforms=[r], ax=ax)
    test_image = 'test_bz.png'
    ref_image = str(datadir / 'rotated_bz.png')
    fig.savefig(test_image)
    assert compare_images(test_image, ref_image, 0.5) is None
