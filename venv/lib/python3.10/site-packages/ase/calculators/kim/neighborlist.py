# fmt: off

from collections import defaultdict

import numpy as np

from ase import Atom
from ase.neighborlist import neighbor_list

from .kimpy_wrappers import c_double, c_int, check_call_wrapper, kimpy


class NeighborList:

    kimpy_arrays = {
        "num_particles": c_int,
        "coords": c_double,
        "particle_contributing": c_int,
        "species_code": c_int,
        "cutoffs": c_double,
        "padding_image_of": c_int,
        "need_neigh": c_int,
    }

    def __setattr__(self, name, value):
        """
        Override assignment to any of the attributes listed in
        kimpy_arrays to automatically cast the object to a numpy array.
        This is done to avoid a ton of explicit numpy.array() calls (and
        the possibility that we forget to do the cast).  It is important
        to use np.asarray() here instead of np.array() because using the
        latter will mean that incrementation (+=) will create a new
        object that the reference is bound to, which becomes a problem
        if update_compute_args isn't called to reregister the
        corresponding address with the KIM API.
        """
        if name in self.kimpy_arrays and value is not None:
            value = np.asarray(value, dtype=self.kimpy_arrays[name])
        self.__dict__[name] = value

    def __init__(
        self,
        neigh_skin_ratio,
        model_influence_dist,
        model_cutoffs,
        padding_not_require_neigh,
        debug,
    ):

        self.set_neigh_parameters(
            neigh_skin_ratio,
            model_influence_dist,
            model_cutoffs,
            padding_not_require_neigh,
        )

        self.debug = debug

        if self.debug:
            print()
            print(f"Calculator skin: {self.skin}")
            print(f"Model influence distance: {model_influence_dist}")
            print(
                "Calculator influence distance (including skin distance): {}"
                "".format(self.influence_dist)
            )
            print(f"Number of cutoffs: {model_cutoffs.size}")
            print(f"Model cutoffs: {model_cutoffs}")
            print(
                "Calculator cutoffs (including skin distance): {}"
                "".format(self.cutoffs)
            )
            print(
                "Model needs neighbors of padding atoms: {}"
                "".format(self.padding_need_neigh)
            )
            print()

        # Attributes to be set by subclasses
        self.neigh = None
        self.num_contributing_particles = None
        self.padding_image_of = None
        self.num_particles = None
        self.coords = None
        self.particle_contributing = None
        self.species_code = None
        self.need_neigh = None
        self.last_update_positions = None

    def set_neigh_parameters(
        self,
        neigh_skin_ratio,
        model_influence_dist,
        model_cutoffs,
        padding_not_require_neigh,
    ):
        self.skin = neigh_skin_ratio * model_influence_dist
        self.influence_dist = model_influence_dist + self.skin
        self.cutoffs = model_cutoffs + self.skin
        self.padding_need_neigh = not padding_not_require_neigh.all()

    def update_kim_coords(self, atoms):
        """Update atomic positions in self.coords, which is where the KIM
        API will look to find them in order to pass them to the model.
        """
        if self.padding_image_of.size != 0:
            disp_contrib = atoms.positions - self.coords[: len(atoms)]
            disp_pad = disp_contrib[self.padding_image_of]
            self.coords += np.concatenate((disp_contrib, disp_pad))
        else:
            np.copyto(self.coords, atoms.positions)

        if self.debug:
            print("Debug: called update_kim_coords")
            print()

    def need_neigh_update(self, atoms, system_changes):
        need_neigh_update = True
        if len(system_changes) == 1 and "positions" in system_changes:
            # Only position changes
            if self.last_update_positions is not None:
                a = self.last_update_positions
                b = atoms.positions
                if a.shape == b.shape:
                    delta = np.linalg.norm(a - b, axis=1)
                    # Indices of the two largest elements
                    try:
                        ind = np.argpartition(delta, -2)[-2:]

                        if sum(delta[ind]) <= self.skin:
                            need_neigh_update = False
                    except ValueError as error:
                        # if there is only a single atom that gets displaced
                        # np.argpartition(delta, -2) will fail with a
                        # ValueError, a single atom has no neighbors to update
                        if atoms.positions.shape[0] != 1:
                            raise error
                        need_neigh_update = False

        return need_neigh_update


class ASENeighborList(NeighborList):
    def __init__(
        self,
        compute_args,
        neigh_skin_ratio,
        model_influence_dist,
        model_cutoffs,
        padding_not_require_neigh,
        debug,
    ):
        super().__init__(
            neigh_skin_ratio,
            model_influence_dist,
            model_cutoffs,
            padding_not_require_neigh,
            debug,
        )

        self.neigh = {}
        compute_args.set_callback(
            kimpy.compute_callback_name.GetNeighborList, self.get_neigh,
            self.neigh
        )

    @staticmethod
    def get_neigh(data, cutoffs, neighbor_list_index, particle_number):
        """Retrieves the neighbors of each atom using ASE's native neighbor
        list library
        """
        # We can only return neighbors of particles that were stored
        number_of_particles = data["num_particles"]
        if particle_number >= number_of_particles or particle_number < 0:
            return (np.array([]), 1)

        neighbors = data["neighbors"][neighbor_list_index][particle_number]
        return (neighbors, 0)

    def build(self, orig_atoms):
        """Build the ASE neighbor list and return an Atoms object with
        all of the neighbors added.  First a neighbor list is created
        from ase.neighbor_list, having only information about the
        neighbors of the original atoms.  If neighbors of padding atoms
        are required, they are calculated using information from the
        first neighbor list.
        """
        syms = orig_atoms.get_chemical_symbols()
        orig_num_atoms = len(orig_atoms)
        orig_pos = orig_atoms.get_positions()

        # New atoms object that will contain the contributing atoms plus
        # the padding atoms
        new_atoms = orig_atoms.copy()

        neigh_list = defaultdict(list)
        neigh_dists = defaultdict(list)

        # Information for padding atoms
        padding_image_of = []
        padding_shifts = []

        # Ask ASE to build the neighbor list using the influence distance.
        # This is not a neighbor list for each atom, but rather a listing
        # of all neighbor pairs that exist.  Atoms with no neighbors will
        # not show up.
        (
            neigh_indices_i,
            neigh_indices_j,
            relative_pos,
            neigh_cell_offsets,
            dists,
        ) = neighbor_list("ijDSd", orig_atoms, self.influence_dist)

        # Loop over all neighbor pairs. Because this loop will generally
        # include image atoms (for periodic systems), we keep track of
        # which atoms/images we've accounted for in the `used` dictionary.
        used = {}
        for neigh_i, neigh_j, rel_pos, offset, dist in zip(
                neigh_indices_i, neigh_indices_j,
                relative_pos, neigh_cell_offsets, dists
        ):
            # Get neighbor position of neighbor
            # (mapped back into unit cell, so this may overlap with other atoms)
            wrapped_pos = orig_pos[neigh_i] + rel_pos

            shift = tuple(offset)
            uniq_index = (neigh_j,) + shift
            if shift == (0, 0, 0):
                # This atom is in the unit cell, i.e. it is contributing
                neigh_list[neigh_i].append(neigh_j)
                neigh_dists[neigh_i].append(dist)
                if uniq_index not in used:
                    used[uniq_index] = neigh_j
            else:
                # This atom is not in the unit cell, i.e. it is padding
                if uniq_index not in used:
                    # Add the neighbor as a padding atom
                    used[uniq_index] = len(new_atoms)
                    new_atoms.append(Atom(syms[neigh_j], position=wrapped_pos))
                    padding_image_of.append(neigh_j)
                    padding_shifts.append(offset)
                neigh_list[neigh_i].append(used[uniq_index])
                neigh_dists[neigh_i].append(dist)
        neighbor_list_size = orig_num_atoms

        # Add neighbors of padding atoms if the potential requires them
        if self.padding_need_neigh:
            neighbor_list_size = len(new_atoms)
            inv_used = {v: k for k, v in used.items()}
            # Loop over all the neighbors (k) and the image of that neighbor
            # in the cell (neigh)
            for k, neigh in enumerate(padding_image_of):
                # Shift from original atom in cell to neighbor
                shift = padding_shifts[k]
                for orig_neigh, orig_dist in zip(
                        neigh_list[neigh], neigh_dists[neigh]):
                    # Get the shift of the neighbor of the original atom
                    orig_shift = inv_used[orig_neigh][1:]

                    # Apply sum of original shift and current shift to
                    # neighbors of original atom
                    total_shift = orig_shift + shift

                    # Get the image in the cell of the original neighbor
                    if orig_neigh <= orig_num_atoms - 1:
                        orig_neigh_image = orig_neigh
                    else:
                        orig_neigh_image = padding_image_of[orig_neigh -
                                                            orig_num_atoms]

                    # If the original image with the total shift has been
                    # used before then it is also a neighbor of this atom
                    uniq_index = (orig_neigh_image,) + tuple(total_shift)
                    if uniq_index in used:
                        neigh_list[k + orig_num_atoms].append(used[uniq_index])
                        neigh_dists[k + orig_num_atoms].append(orig_dist)

        # If the model has multiple cutoffs, we need to return neighbor lists
        # corresponding to each of them
        neigh_lists = []
        for cut in self.cutoffs:
            neigh_list = [
                np.array(neigh_list[k], dtype=c_int)[neigh_dists[k] <= cut]
                for k in range(neighbor_list_size)
            ]
            neigh_lists.append(neigh_list)

        self.padding_image_of = padding_image_of

        self.neigh["neighbors"] = neigh_lists
        self.neigh["num_particles"] = neighbor_list_size

        return new_atoms

    def update(self, orig_atoms, species_map):
        """Create the neighbor list along with the other required
        parameters (which are stored as instance attributes). The
        required parameters are:

            - num_particles
            - coords
            - particle_contributing
            - species_code

        Note that the KIM API requires a neighbor list that has indices
        corresponding to each atom.
        """

        # Information of original atoms
        self.num_contributing_particles = len(orig_atoms)

        new_atoms = self.build(orig_atoms)

        # Save the number of atoms and all their neighbors and positions
        num_atoms = len(new_atoms)
        num_padding = num_atoms - self.num_contributing_particles
        self.num_particles = [num_atoms]
        self.coords = new_atoms.get_positions()

        # Save which coordinates are from original atoms and which are from
        # neighbors using a mask
        indices_mask = [1] * self.num_contributing_particles + [0] * num_padding
        self.particle_contributing = indices_mask

        # Species support and code
        try:
            self.species_code = [
                species_map[s] for s in new_atoms.get_chemical_symbols()
            ]
        except KeyError as e:
            raise RuntimeError(
                "Species not supported by KIM model; {}".format(
                    str(e)))

        self.last_update_positions = orig_atoms.get_positions()

        if self.debug:
            print("Debug: called update_ase_neigh")
            print()


class KimpyNeighborList(NeighborList):
    def __init__(
        self,
        compute_args,
        neigh_skin_ratio,
        model_influence_dist,
        model_cutoffs,
        padding_not_require_neigh,
        debug,
    ):
        super().__init__(
            neigh_skin_ratio,
            model_influence_dist,
            model_cutoffs,
            padding_not_require_neigh,
            debug,
        )

        self.neigh = kimpy.neighlist.NeighList()

        compute_args.set_callback_pointer(
            kimpy.compute_callback_name.GetNeighborList,
            kimpy.neighlist.get_neigh_kim(),
            self.neigh,
        )

    @check_call_wrapper
    def build(self):
        return self.neigh.build(
            self.coords, self.influence_dist, self.cutoffs, self.need_neigh
        )

    @check_call_wrapper
    def create_paddings(
        self, cell, pbc, contributing_coords, contributing_species_code
    ):
        # Cast things passed through kimpy to numpy arrays
        cell = np.asarray(cell, dtype=c_double)
        pbc = np.asarray(pbc, dtype=c_int)
        contributing_coords = np.asarray(contributing_coords, dtype=c_double)

        return kimpy.neighlist.create_paddings(
            self.influence_dist,
            cell,
            pbc,
            contributing_coords,
            contributing_species_code,
        )

    def update(self, atoms, species_map):
        """Create the neighbor list along with the other required
        parameters (which are stored as instance attributes). The
        required parameters are:

            - num_particles
            - coords
            - particle_contributing
            - species_code

        Note that the KIM API requires a neighbor list that has indices
        corresponding to each atom.
        """

        # Get info from Atoms object
        cell = np.asarray(atoms.get_cell(), dtype=c_double)
        pbc = np.asarray(atoms.get_pbc(), dtype=c_int)
        contributing_coords = np.asarray(atoms.get_positions(), dtype=c_double)
        self.num_contributing_particles = atoms.get_global_number_of_atoms()
        num_contributing = self.num_contributing_particles

        # Species support and code
        try:
            contributing_species_code = np.array(
                [species_map[s] for s in atoms.get_chemical_symbols()],
                dtype=c_int
            )
        except KeyError as e:
            raise RuntimeError(
                "Species not supported by KIM model; {}".format(
                    str(e)))

        if pbc.any():  # Need padding atoms
            # Create padding atoms

            (
                padding_coords,
                padding_species_code,
                self.padding_image_of,
            ) = self.create_paddings(
                cell, pbc, contributing_coords, contributing_species_code
            )
            num_padding = padding_species_code.size

            self.num_particles = [num_contributing + num_padding]
            self.coords = np.concatenate((contributing_coords, padding_coords))
            self.species_code = np.concatenate(
                (contributing_species_code, padding_species_code)
            )
            self.particle_contributing = [1] * \
                num_contributing + [0] * num_padding
            self.need_neigh = [1] * self.num_particles[0]
            if not self.padding_need_neigh:
                self.need_neigh[num_contributing:] = 0

        else:  # Do not need padding atoms
            self.padding_image_of = []
            self.num_particles = [num_contributing]
            self.coords = contributing_coords
            self.species_code = contributing_species_code
            self.particle_contributing = [1] * num_contributing
            self.need_neigh = self.particle_contributing

        # Create neighborlist
        self.build()

        self.last_update_positions = atoms.get_positions()

        if self.debug:
            print("Debug: called update_kimpy_neigh")
            print()
