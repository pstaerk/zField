import jax
import jax.numpy as jnp
import pytest
from flax import struct

from z_field.coupled import z_i_pbc


@struct.dataclass
class Batch:
    """JAX pytree-compatible batch structure."""

    unfolded_nodes: jnp.ndarray
    unfolded_positions: jnp.ndarray
    unit_cell_mask: jnp.ndarray
    to_replicate_idx: jnp.ndarray
    unfolded_centers: jnp.ndarray
    unfolded_others: jnp.ndarray


class TestCoupledModel:
    """Simple test suite for the coupled model z_i_pbc function."""

    @pytest.fixture
    def rng_key(self):
        """Provide a JAX random key."""
        return jax.random.PRNGKey(42)

    @pytest.fixture
    def simple_batch(self):
        """Create a simple batch structure for testing."""
        # Create a simple system with 2 atoms in unit cell, 4 total with ghosts
        return Batch(
            # 2 atoms in unit cell, 2 ghost atoms
            unfolded_nodes=jnp.array([1, 1, 1, 1]),  # Atomic numbers
            unfolded_positions=jnp.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],  # Ghost
                    [1.0, 1.0, 0.0],  # Ghost
                ]
            ),
            # Mask: True for unit cell atoms, False for ghosts
            unit_cell_mask=jnp.array([1.0, 1.0, 0.0, 0.0]),
            # Map ghost atoms back to original atoms
            to_replicate_idx=jnp.array([0, 1, 0, 1]),
            # Edge list (centers and neighbors)
            unfolded_centers=jnp.array([0, 0, 1, 1]),
            unfolded_others=jnp.array([1, 2, 0, 3]),
        )

    @pytest.fixture
    def simple_q_function(self):
        """Create a simple charge prediction function."""

        def q_func(batch, params, rijs):
            # Simple distance-based charge function
            distances = jnp.linalg.norm(rijs, axis=-1)
            return params["scale"] * jnp.exp(-distances / params["length_scale"])

        return q_func

    @pytest.fixture
    def simple_params(self):
        """Create simple parameters."""
        return {"scale": 1.0, "length_scale": 1.0}

    def test_basic_forward_pass(self, simple_batch, simple_params, simple_q_function):
        """Test that the function runs without errors."""
        Z = z_i_pbc(simple_batch, simple_params, simple_q_function)

        # Check output shape
        num_atoms = simple_batch.unfolded_nodes.shape[0]
        assert Z.shape == (num_atoms, 3, 3)

    def test_output_is_finite(self, simple_batch, simple_params, simple_q_function):
        """Test that output contains only finite values."""
        Z = z_i_pbc(simple_batch, simple_params, simple_q_function)
        assert jnp.all(jnp.isfinite(Z)), "Output contains NaN or Inf"

    def test_output_shape(self, simple_batch, simple_params, simple_q_function):
        """Test output has correct shape (N, 3, 3)."""
        Z = z_i_pbc(simple_batch, simple_params, simple_q_function)
        num_atoms = simple_batch.unfolded_nodes.shape[0]
        expected_shape = (num_atoms, 3, 3)
        assert Z.shape == expected_shape, f"Expected {expected_shape}, got {Z.shape}"

    def test_different_q_functions(self, simple_batch, simple_params):
        """Test with different charge functions."""

        # Constant charge function
        def const_q(batch, params, rijs):
            return jnp.ones(rijs.shape[0]) * params["scale"]

        Z1 = z_i_pbc(simple_batch, simple_params, const_q)
        assert jnp.all(jnp.isfinite(Z1))

        # Linear charge function
        def linear_q(batch, params, rijs):
            distances = jnp.linalg.norm(rijs, axis=-1)
            return params["scale"] * distances

        Z2 = z_i_pbc(simple_batch, simple_params, linear_q)
        assert jnp.all(jnp.isfinite(Z2))

    def test_gradient_computation(self, simple_batch, simple_params, simple_q_function):
        """Test that gradients can be computed with respect to parameters."""

        def loss_fn(params):
            Z = z_i_pbc(simple_batch, params, simple_q_function)
            return jnp.sum(Z**2)

        loss, grads = jax.value_and_grad(loss_fn)(simple_params)

        assert jnp.isfinite(loss), "Loss is not finite"
        assert all(
            jnp.all(jnp.isfinite(g)) for g in grads.values()
        ), "Gradients contain NaN or Inf"

    def test_deterministic_output(self, simple_batch, simple_params, simple_q_function):
        """Test that same input produces same output."""
        Z1 = z_i_pbc(simple_batch, simple_params, simple_q_function)
        Z2 = z_i_pbc(simple_batch, simple_params, simple_q_function)

        assert jnp.allclose(Z1, Z2, rtol=1e-6), "Outputs should be identical"
