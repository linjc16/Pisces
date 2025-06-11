from ase.atoms import Atoms
from ase.md.analysis import DiffusionCoefficient
from ase.units import fs as fs_conversion

eps = 1e-10
timestep = 1 * fs_conversion  # fs
# Answer in \AA^2/<ASE time unit>
ans_orig = 5.0e-01 / fs_conversion


def test_atom():
    # Creating simple trajectories
    # Textbook case. The displacement coefficient should
    # be 0.5 A^2 / fs except for the final molecule

    # He atom

    he = Atoms('He', positions=[(0, 0, 0)])
    traj_he = [he.copy() for _ in range(2)]
    traj_he[1].set_positions([(1, 1, 1)])

    dc_he = DiffusionCoefficient(traj_he, timestep)
    dc_he.calculate(ignore_n_images=0, number_of_segments=1)
    ans = dc_he.get_diffusion_coefficients()[0][0]

    assert abs(ans - ans_orig) < eps


def test_co_molecule():
    co = Atoms('CO', positions=[(0, 0, 0), (0, 0, 1)])
    traj_co = [co.copy() for _ in range(2)]
    traj_co[1].set_positions([(-1, -1, -1), (-1, -1, 0)])

    dc_co = DiffusionCoefficient(traj_co, timestep, molecule=False)
    dc_co.calculate(ignore_n_images=0, number_of_segments=1)
    ans = dc_co.get_diffusion_coefficients()[0][0]

    assert abs(ans - ans_orig) < eps

    for index in range(2):
        dc_co = DiffusionCoefficient(traj_co, timestep, atom_indices=[index],
                                     molecule=False)
        dc_co.calculate()
        ans = dc_co.get_diffusion_coefficients()[0][0]
        assert abs(ans - ans_orig) < eps

    dc_co = DiffusionCoefficient(traj_co, timestep, molecule=True)
    dc_co.calculate(ignore_n_images=0, number_of_segments=1)
    ans = dc_co.get_diffusion_coefficients()[0][0]

    assert abs(ans - ans_orig) < eps
