import numpy as np
from pytest import mark

from ase import Atoms


def squeeze_dimer(atoms, d):
    """Squeeze the atoms together by the absolute distance ``d`` (Angstroms)
    """
    pos = atoms.get_positions()
    pos[0] += np.asarray([d, 0, 0])
    atoms.set_positions(pos)


def set_positions_to_orig(atoms, box_len, dimer_separation):
    pos0 = 0.5 * np.full(3, 0.5 * box_len)
    displacement = np.array([0.5 * dimer_separation, 0, 0])

    pos1 = pos0 - displacement
    pos2 = pos0 + displacement
    atoms.set_positions([pos1, pos2])


@mark.calculator_lite
def test_update_coords(KIM, testdir):
    """
    Check that the coordinates registered with the KIM API are updated
    appropriately when the atomic positions are updated.  This can go awry
    if the 'coords' attribute of the relevant NeighborList subclass is
    reassigned to a new memory location -- a problem which was indeed
    occurring at one point
    (see https://gitlab.com/ase/ase/merge_requests/1442)!
    """

    # We know that ex_model_Ar_P_Morse_07C has a cutoff of 8.15 Angstroms
    model = "ex_model_Ar_P_Morse_07C"
    model_cutoff = 8.15  # Angstroms

    # Create a dimer centered in a small box that's small enough so
    # that there will be padding atoms when we make it periodic
    box_len = 0.5 * model_cutoff
    dimer_separation = model_cutoff * 0.3

    atoms = Atoms("Ar" * 2, cell=[[box_len, 0, 0],
                  [0, box_len, 0], [0, 0, box_len]])

    # Create calculator.  Either the kimpy neighbor list library or
    # ASE's native neighbor lists should suffice to check this since
    # update_kim_coords() belongs to their parent class, NeighborList.
    # Here, we'll use the default mode (kimpy neighbor list).
    neigh_skin_ratio = 0.2
    skin = neigh_skin_ratio * model_cutoff
    calc = KIM(model, options={"neigh_skin_ratio": neigh_skin_ratio})
    atoms.calc = calc

    squeezed_energies_ref = {
        False: 5.784620078721877,  # finite
        True: 6.766293119162073,  # periodic
    }

    for pbc in [False, True]:

        # Reset dimer positions to original configuration
        set_positions_to_orig(atoms, box_len, dimer_separation)

        atoms.set_pbc(pbc)

        # Get potential energy so that it will get rid of "pbc" being
        # in the system_changes.  The only way that update_kim_coords
        # is called is when system_changes only contains "positions",
        # as otherwise a neighbor list rebuild is triggered.
        atoms.get_potential_energy()

        # First squeeze the dimer together by a distance less than the
        # skin and compute the energy.  This doesn't trigger a
        # neighbor list rebuild (which we avoid here because we're
        # only trying to test update_kim_coords, which is not called
        # in the event that a neighbor list rebuild is necessary).
        #
        # This will update the coordinate values in the ``coords``
        # attribute of the relevant NeighborList subclass instance
        # (see ase/calculators/kim/neighborlist.py) and, since
        # ``coords`` is pointing at the same memory the KIM API is
        # reading the coordinates from, the KIM Model will see the
        # updated coordinates with no problem.  *However*, it's
        # possible that after updating these values, the ``coords``
        # attribute is bound to a *new* object in memory if
        # update_kim_coords is broken somehow!
        squeeze_dimer(atoms, 0.2 * skin)
        atoms.get_potential_energy()

        # Now squeeze the dimer together by the same amount again,
        # where the sum of this squeeze and the last is still less
        # than the skin distance so that we still avoid a neighbor
        # list rebuild.  If all is well, the `coords` attribute of the
        # relevant NeighborList subclass still points at the same
        # location as before we did the previous squeeze.  Thus, when
        # we update the coordinates again, it should still be updating
        # the same coordinates being read by the KIM API, giving us
        # the expected value of the energy.  If update_kim_coords is
        # broken, ``coords`` will already be bound to a new object in
        # memory while the KIM API is still looking at its previous
        # address to read coordinates.  This means that when we update
        # the coordinates in the ``coords`` value, the KIM API never
        # sees them.  In fact, it's not even guaranteed that the KIM
        # API will read the same coordinates as before since the
        # memory where it's looking may have since been overwritten by
        # some other python objects.
        squeeze_dimer(atoms, 0.2 * skin)
        squeezed_energy = atoms.get_potential_energy()
        assert np.isclose(squeezed_energy, squeezed_energies_ref[pbc])
