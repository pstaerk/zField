"""Batching utilities for z_field package."""
import numpy as np


def unfolded_ghosts(atoms, cutoff,):
    """Builds all ghost atoms explicitly.

    :atoms: ase.Atoms object with periodic boundary conditions
    :cutoff: float, cutoff for neighbor list, needs to be effective cutoff

    returns:
    :all_nodes: (N, ) int array of atomic numbers including ghosts
    :all_positions: (N, 3) float array of positions including ghosts
    :sD: (E, 3) float array of all edge vectors including ghosts
    :si: (E, ) int array of source node indices for edges including ghosts
    :sj: (E, ) int array of target node indices for edges including ghosts
    :unit_cell_mask: (N, ) bool array, True for atoms in original unit cell
    :to_replicate: (N, ) int array, mapping from all_positions to original
                         atoms

    """
    import vesin
    nl_calc = vesin.NeighborList(cutoff=cutoff, full_list=True)

    # 1. Find all neighbors, including those in periodic images
    positions = atoms.positions
    i_p, j_p, S_p = nl_calc.compute(
        points=positions,
        box=atoms.cell,
        periodic=True,
        quantities="ijS"
    )
    # Find all unique (shift, atom_index) pairs
    replicas = np.unique(np.concatenate([S_p, j_p[:, None]], axis=-1), axis=0)
    is_original_cell = np.all(replicas[:, :3] == 0, axis=1)

    replicas = replicas[~is_original_cell]  # only replicas, not originals

    # Separate shifts from atom indices
    cell_shifts, to_replicate = np.split(replicas, [3], axis=-1)
    to_replicate = to_replicate.flatten().astype(int)

    # 3. Construct the supercell positions and node types
    cell_shifts = np.concatenate([np.zeros((len(positions), 3), dtype=int),
                                  cell_shifts], axis=0)
    to_replicate = np.concatenate([np.arange(len(positions), dtype=int),
                                   np.array(to_replicate, dtype=int)],
                                  dtype=int)

    unit_cell_mask = np.zeros(len(to_replicate), dtype=bool)
    unit_cell_mask[: len(positions)] = True

    offsets = np.einsum("pA,Aa->pa", cell_shifts, atoms.cell)
    all_positions = atoms.positions[to_replicate] + offsets
    all_nodes = atoms.get_atomic_numbers()[to_replicate]

    # 3. Construct the supercell positions and node types
    offsets = np.einsum("pA,Aa->pa", cell_shifts, atoms.cell)
    all_positions = atoms.positions[to_replicate] + offsets
    all_nodes = atoms.get_atomic_numbers()[to_replicate]

    # 5. Compute the new, non-periodic neighbor list for the supercell
    super_box = atoms.cell * np.max(np.abs(cell_shifts), axis=0) * 2
    si, sj, sD = nl_calc.compute(
        points=all_positions,
        periodic=False,
        quantities="ijD",
        box=super_box,
    )
    return all_nodes, all_positions, sD, si, sj, unit_cell_mask, to_replicate
