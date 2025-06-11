import numpy as np
import pytest

from ase.build import fcc111
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.ga import set_raw_score
from ase.ga.cutandsplicepairing import CutAndSplicePairing
from ase.ga.data import DataConnection, PrepareDB
from ase.ga.offspring_creator import OperationSelector
from ase.ga.population import Population
from ase.ga.standard_comparators import InteratomicDistanceComparator
from ase.ga.standardmutations import (
    MirrorMutation,
    PermutationMutation,
    RattleMutation,
)
from ase.ga.startgenerator import StartGenerator
from ase.ga.utilities import closest_distances_generator, get_all_atom_types
from ase.io import write
from ase.optimize import BFGS

db_file = 'gadb.db'


@pytest.mark.slow()
def test_basic_example_main_run(seed, testdir):
    # set up the random number generator
    rng = np.random.RandomState(seed)

    # create the surface
    slab = fcc111('Au', size=(4, 4, 1), vacuum=10.0, orthogonal=True)
    slab.set_constraint(FixAtoms(mask=len(slab) * [True]))

    # define the volume in which the adsorbed cluster is optimized
    # the volume is defined by a corner position (p0)
    # and three spanning vectors (v1, v2, v3)
    pos = slab.get_positions()
    cell = slab.get_cell()
    p0 = np.array([0., 0., max(pos[:, 2]) + 2.])
    v1 = cell[0, :] * 0.8
    v2 = cell[1, :] * 0.8
    v3 = cell[2, :]
    v3[2] = 3.

    # Define the composition of the atoms to optimize
    atom_numbers = 2 * [47] + 2 * [79]

    # define the closest distance two atoms of a given species can be to each
    # other
    unique_atom_types = get_all_atom_types(slab, atom_numbers)
    blmin = closest_distances_generator(atom_numbers=unique_atom_types,
                                        ratio_of_covalent_radii=0.7)

    # create the starting population
    sg = StartGenerator(slab=slab,
                        blocks=atom_numbers,
                        blmin=blmin,
                        box_to_place_in=[p0, [v1, v2, v3]],
                        rng=rng)

    # generate the starting population
    population_size = 5
    starting_population = [sg.get_new_candidate()
                           for _ in range(population_size)]

    # from ase.visualize import view   # uncomment these lines
    # view(starting_population)        # to see the starting population

    # create the database to store information in
    d = PrepareDB(db_file_name=db_file,
                  simulation_cell=slab,
                  stoichiometry=atom_numbers)

    for a in starting_population:
        d.add_unrelaxed_candidate(a)

    # XXXXXXXXXX This should be the beginning of a new test,
    # but we are using some resources from the precious part.
    # Maybe refactor those things as (module-level?) fixtures.

    # Change the following three parameters to suit your needs
    population_size = 5
    mutation_probability = 0.3
    n_to_test = 5

    # Initialize the different components of the GA
    da = DataConnection('gadb.db')
    atom_numbers_to_optimize = da.get_atom_numbers_to_optimize()
    n_to_optimize = len(atom_numbers_to_optimize)
    slab = da.get_slab()
    all_atom_types = get_all_atom_types(slab, atom_numbers_to_optimize)
    blmin = closest_distances_generator(all_atom_types,
                                        ratio_of_covalent_radii=0.7)

    comp = InteratomicDistanceComparator(n_top=n_to_optimize,
                                         pair_cor_cum_diff=0.015,
                                         pair_cor_max=0.7,
                                         dE=0.02,
                                         mic=False)

    pairing = CutAndSplicePairing(slab, n_to_optimize, blmin, rng=rng)
    mutations = OperationSelector([1., 1., 1.],
                                  [MirrorMutation(blmin, n_to_optimize,
                                                  rng=rng),
                                   RattleMutation(
                                       blmin, n_to_optimize, rng=rng),
                                   PermutationMutation(n_to_optimize,
                                                       rng=rng)],
                                  rng=rng)

    # Relax all unrelaxed structures (e.g. the starting population)
    while da.get_number_of_unrelaxed_candidates() > 0:
        a = da.get_an_unrelaxed_candidate()
        a.calc = EMT()
        print('Relaxing starting candidate {}'.format(a.info['confid']))
        with BFGS(a, trajectory=None, logfile=None) as dyn:
            dyn.run(fmax=0.05, steps=100)
        set_raw_score(a, -a.get_potential_energy())
        da.add_relaxed_step(a)

    # create the population
    population = Population(data_connection=da,
                            population_size=population_size,
                            comparator=comp,
                            rng=rng)

    # test n_to_test new candidates
    for i in range(n_to_test):
        print(f'Now starting configuration number {i}')
        a1, a2 = population.get_two_candidates()
        a3, desc = pairing.get_new_individual([a1, a2])
        if a3 is None:
            continue
        da.add_unrelaxed_candidate(a3, description=desc)

        # Check if we want to do a mutation
        if rng.random() < mutation_probability:
            a3_mut, desc = mutations.get_new_individual([a3])
            if a3_mut is not None:
                da.add_unrelaxed_step(a3_mut, desc)
                a3 = a3_mut

        # Relax the new candidate
        a3.calc = EMT()
        with BFGS(a3, trajectory=None, logfile=None) as dyn:
            dyn.run(fmax=0.05, steps=100)
        set_raw_score(a3, -a3.get_potential_energy())
        da.add_relaxed_step(a3)
        population.update()

    write('all_candidates.traj', da.get_all_relaxed_candidates())
