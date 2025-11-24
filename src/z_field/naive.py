"""Non-periodic boundary condition utilities for z_field package.

This module provides functions for calculating polarization and related tensors
without periodic boundary conditions, serving as a simpler alternative to the
phase approach for non-periodic systems.
"""

from typing import Any, Callable

import jax
import jax.numpy as jnp


def calc_rijs(batch, pbc=True):
    """Calculate relative position vectors between atoms.

    Args:
        batch: Batch object containing positions, centers, others,
               cell_shifts, and cell.
        pbc: Whether to apply periodic boundary conditions.

    Returns:
        jnp.ndarray: Relative position vectors.
    """
    rij = batch.positions[batch.others] - batch.positions[batch.centers]
    if pbc:
        rij += jnp.einsum("pA,Aa->pa", batch.cell_shifts, batch.cell)
    return rij


def pol_function_npbc(charge_fn: Callable, params: Any, batch: Any, alpha: int):
    """Calculate the polarization for non-periodic boundary conditions.

    This is a simpler alternative to the phase approach that directly computes
    the sum of position-weighted charges without considering periodicity.

    Args:
        charge_fn: Function that takes (params, batch) and returns charges.
                   Must be differentiable with respect to batch.positions.
                   The user is responsible for including screening or other
                   transformations in this function.
        params: Model parameters.
        batch: Batch object containing atomic structure information.
        alpha: Cartesian component index (0, 1, or 2 for x, y, z).

    Returns:
        float: Polarization component for the given direction.
    """
    charges = charge_fn(params, batch).flatten()
    return jnp.sum(batch.positions[:, alpha] * charges * batch.node_mask)


def z_i_alpha_beta_npbc(charge_fn: Callable, params: Any, batch: Any):
    """Calculate derivatives of polarization w.r.t. atomic positions (non-PBC).

    This computes z_i^{alpha,beta} = ∂P_alpha/∂r_i_beta for all atoms and
    all Cartesian components without periodic boundary conditions.

    Args:
        charge_fn: Function that takes (params, batch) and returns charges.
                   Must be differentiable with respect to batch.positions.
                   The user is responsible for including screening or other
                   transformations in this function.
        params: Model parameters.
        batch: Batch object containing atomic structure information.

    Returns:
        jnp.ndarray: Tensor of shape (n_atoms, 3, 3) where element [i, alpha, beta]
                     represents ∂P_alpha/∂r_i_beta.
    """

    def pol_wrapper(params, batch, alpha):
        return pol_function_npbc(charge_fn, params, batch, alpha)

    deriv_p_npbc = jax.value_and_grad(
        pol_wrapper, has_aux=False, argnums=1, allow_int=True
    )
    z_i_ab = jnp.zeros((batch.positions.shape[0], 3, 3), dtype=jnp.float32)

    for alpha in range(3):
        # we want the derivative of P_alpha with respect to r_beta
        _, deriv_alpha = deriv_p_npbc(params, batch, alpha)
        for beta in range(3):
            component = deriv_alpha.positions[:, beta]
            z_i_ab = z_i_ab.at[:, alpha, beta].set(component)
    return z_i_ab