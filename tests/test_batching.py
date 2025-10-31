import numpy as np
from ase import Atoms
from ase.build import bulk

from z_field.batching import unfolded_ghosts


class TestUnfoldedGhosts:
    """Test the unfolded_ghosts function."""

    def test_simple_cubic_cell(self):
        """Test with a simple cubic cell."""
        atoms = Atoms("Si", positions=[(0, 0, 0)], cell=[5.0, 5.0, 5.0], pbc=True)
        cutoff = 3.0

        result = unfolded_ghosts(atoms, cutoff)
        all_nodes, all_positions, sD, si, sj, unit_cell_mask, to_replicate = result

        # Basic checks
        assert len(all_nodes) >= 1
        assert np.sum(unit_cell_mask) == 1
        assert all_nodes[0] == 14  # Silicon

    def test_fcc_copper(self):
        """Test with FCC copper structure."""
        atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
        cutoff = 3.0

        result = unfolded_ghosts(atoms, cutoff)
        all_nodes, all_positions, sD, si, sj, unit_cell_mask, to_replicate = result

        # Check edges exist
        assert len(si) > 0
        assert len(sj) > 0
        assert len(sD) > 0
        assert len(si) == len(sj) == len(sD)

        # Check edge indices are valid
        assert np.all(si >= 0)
        assert np.all(si < len(all_nodes))
        assert np.all(sj >= 0)
        assert np.all(sj < len(all_nodes))

    def test_output_shapes(self):
        """Test output array shapes."""
        atoms = bulk("Al", "fcc", a=2.05, cubic=True)
        atoms.cell *= 2  # Make cell larger such that there are no ghosts
        atoms.pbc = True
        cutoff = 4.0
        original_positions = atoms.positions.copy()

        result = unfolded_ghosts(atoms, cutoff)
        all_nodes, all_positions, sD, si, sj, unit_cell_mask, to_replicate = result

        n_nodes = len(all_nodes)
        n_edges = len(si)

        # Check shapes
        assert all_positions.shape == (n_nodes, 3)
        assert all_nodes.shape == (n_nodes,)
        assert unit_cell_mask.shape == (n_nodes,)
        assert to_replicate.shape == (n_nodes,)
        assert sD.shape == (n_edges, 3)
        assert si.shape == (n_edges,)
        assert sj.shape == (n_edges,)
        assert all_positions[unit_cell_mask].shape[0] == len(original_positions)
        assert np.all(all_positions[unit_cell_mask] == original_positions)
