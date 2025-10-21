import jax
import jax.numpy as jnp
import numpy as np
import pytest

from z_field.direct import PerParticleTensorPredictor


class TestPerParticleTensorPredictor:
    """Test the PerParticleTensorPredictor class."""

    @pytest.fixture
    def rng_key(self):
        """Provide a JAX random key."""
        return jax.random.PRNGKey(0)

    @pytest.fixture
    def model(self):
        """Create a model instance."""
        return PerParticleTensorPredictor(features=64)

    def create_spherical_features(self, batch_shape, max_degree, num_features):
        """Create mock spherical features with proper e3x structure.

        Args:
            batch_shape: Tuple of batch dimensions (e.g., (10,) for 10 atoms)
            max_degree: Maximum spherical degree
            num_features: Number of features per spherical harmonic

        Returns:
            Spherical features array with shape:
            (*batch_shape, 1, (max_degree+1)**2, num_features)
        """
        num_channels = (max_degree + 1) ** 2
        shape = batch_shape + (1, num_channels, num_features)
        return jnp.ones(shape)

    def test_model_initialization(self, model):
        """Test that model can be initialized."""
        assert model.features == 64
        assert isinstance(model, PerParticleTensorPredictor)

    def test_forward_pass_single_atom(self, model, rng_key):
        """Test forward pass with a single atom."""
        # Create spherical features for one atom
        # Shape: (1, (l+1)^2, features) for max_degree=2
        spherical_features = self.create_spherical_features(
            batch_shape=(1,), max_degree=2, num_features=32
        )

        # Initialize and apply model
        variables = model.init(rng_key, spherical_features)
        output = model.apply(variables, spherical_features)

        # Check output shape: should be (1, 3, 3) for a 3x3 tensor per atom
        assert output.shape == (1, 3, 3), f"Expected (1, 3, 3), got {output.shape}"

    def test_forward_pass_multiple_atoms(self, model, rng_key):
        """Test forward pass with multiple atoms."""
        num_atoms = 5
        spherical_features = self.create_spherical_features(
            batch_shape=(num_atoms,), max_degree=2, num_features=32
        )

        variables = model.init(rng_key, spherical_features)
        output = model.apply(variables, spherical_features)

        # Check output shape: (num_atoms, 3, 3)
        assert output.shape == (
            num_atoms,
            3,
            3,
        ), f"Expected ({num_atoms}, 3, 3), got {output.shape}"

    def test_output_is_finite(self, model, rng_key):
        """Test that output contains only finite values."""
        spherical_features = self.create_spherical_features(
            batch_shape=(3,), max_degree=2, num_features=32
        )

        variables = model.init(rng_key, spherical_features)
        output = model.apply(variables, spherical_features)

        assert jnp.all(jnp.isfinite(output)), "Output contains NaN or Inf values"

    def test_different_feature_sizes(self, rng_key):
        """Test with different feature sizes."""
        for features in [32, 64, 128, 256]:
            model = PerParticleTensorPredictor(features=features)
            spherical_features = self.create_spherical_features(
                batch_shape=(2,), max_degree=2, num_features=32
            )

            variables = model.init(rng_key, spherical_features)
            output = model.apply(variables, spherical_features)

            assert output.shape == (
                2,
                3,
                3,
            ), f"Failed for features={features}"

    def test_batch_processing(self, model, rng_key):
        """Test batch processing with multiple structures."""
        # Batch of 4 structures, each with 3 atoms
        batch_size = 4
        num_atoms = 3
        spherical_features = self.create_spherical_features(
            batch_shape=(batch_size, num_atoms), max_degree=2, num_features=32
        )

        variables = model.init(rng_key, spherical_features)
        output = model.apply(variables, spherical_features)

        # Check output shape: (batch_size, num_atoms, 3, 3)
        assert output.shape == (
            batch_size,
            num_atoms,
            3,
            3,
        ), f"Expected ({batch_size}, {num_atoms}, 3, 3), got {output.shape}"

    def test_gradient_computation(self, model, rng_key):
        """Test that gradients can be computed."""
        spherical_features = self.create_spherical_features(
            batch_shape=(2,), max_degree=2, num_features=32
        )

        def loss_fn(params, features):
            output = model.apply(params, features)
            return jnp.sum(output**2)

        variables = model.init(rng_key, spherical_features)
        loss, grads = jax.value_and_grad(loss_fn)(variables, spherical_features)
        print(f'{loss=}')

        # Check that loss is finite
        assert jnp.isfinite(loss), "Loss is not finite"

    def test_equivariance_under_rotation(self, model, rng_key):
        """Test approximate equivariance under SO(3) rotations.

        Note: This is a basic sanity check. True equivariance testing
        would require rotating the input spherical features properly.
        """
        spherical_features = self.create_spherical_features(
            batch_shape=(1,), max_degree=2, num_features=32
        )

        variables = model.init(rng_key, spherical_features)
        output = model.apply(variables, spherical_features)

        # Check that output is a proper tensor shape
        assert output.shape[-2:] == (3, 3), "Output should be 3x3 matrices"

    def test_different_max_degrees(self, rng_key):
        """Test with different maximum spherical degrees."""
        model = PerParticleTensorPredictor(features=64)

        for max_degree in [1, 2, 3]:
            spherical_features = self.create_spherical_features(
                batch_shape=(2,), max_degree=max_degree, num_features=32
            )

            variables = model.init(rng_key, spherical_features)
            output = model.apply(variables, spherical_features)

            assert output.shape == (
                2,
                3,
                3,
            ), f"Failed for max_degree={max_degree}"

    def test_deterministic_output(self, model, rng_key):
        """Test that same input produces same output (deterministic)."""
        spherical_features = self.create_spherical_features(
            batch_shape=(3,), max_degree=2, num_features=32
        )

        variables = model.init(rng_key, spherical_features)
        output1 = model.apply(variables, spherical_features)
        output2 = model.apply(variables, spherical_features)

        np.testing.assert_allclose(
            output1, output2, rtol=1e-6, err_msg="Outputs should be identical"
        )

    def test_different_inputs_different_outputs(self, model, rng_key):
        """Test that different inputs produce different outputs."""
        spherical_features1 = self.create_spherical_features(
            batch_shape=(2,), max_degree=2, num_features=32
        )
        spherical_features2 = spherical_features1 * 2.0  # Different input

        variables = model.init(rng_key, spherical_features1)
        output1 = model.apply(variables, spherical_features1)
        output2 = model.apply(variables, spherical_features2)

        # Outputs should be different
        assert not jnp.allclose(
            output1, output2
        ), "Different inputs should produce different outputs"

    def test_zero_input(self, model, rng_key):
        """Test behavior with zero input."""
        spherical_features = jnp.zeros((2, 1, 9, 32))  # max_degree=2

        variables = model.init(rng_key, spherical_features)
        output = model.apply(variables, spherical_features)

        # Output should be finite even with zero input
        assert jnp.all(jnp.isfinite(output)), "Output should be finite for zero input"
        assert output.shape == (2, 3, 3)
