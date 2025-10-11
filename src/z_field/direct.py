"""Direct prediction of polar tensors from local environments."""

import e3x
import jax.numpy as jnp
from flax import linen as nn


class PerParticleTensorPredictor(nn.Module):
    features: int = 128

    @nn.compact
    def __call__(self, spherical_features):
        # Weighting, linear combinations of spherical features.
        x = e3x.nn.Dense(features=self.features)(
            spherical_features
        )  # -> [...,1 or 2,(l+1)**2,sp_features]

        x = e3x.nn.activations.silu(x)

        x = e3x.nn.Dense(features=self.features)(
            spherical_features
        )  # -> [...,1 or 2,(l+1)**2,sp_features]

        x = e3x.nn.activations.silu(x)

        x = e3x.nn.Dense(features=self.features)(spherical_features)

        # coupling and weighting
        x = e3x.nn.TensorDense(
            features=1,
            max_degree=2,
        )(x)

        # Take only the non-pseudovector channels, feature away
        x = x[..., 0, :, 0]  # -> [...,N,9]

        cg = e3x.so3.clebsch_gordan(max_degree1=1, max_degree2=1,
                                    max_degree3=2)
        y = jnp.einsum("...l,nml->...nm", x, cg[1:, 1:, :])
        return y
