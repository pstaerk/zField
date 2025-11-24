from typing import Any, Callable

import jax
import jax.numpy as jnp


def calc_rijs(batch):
    """Calculate differentiable relative position vectors between atom.

    Args:
        batch: Batch object containing positions, centers, others,
               cell_shifts, and cell.

    Returns:
        jnp.ndarray: Relative position vectors with periodic boundary
                     conditions applied.
    """
    rij = batch.positions[batch.others] - batch.positions[batch.centers]
    rij += jnp.einsum("pA,Aa->pa", batch.cell_shifts, batch.cell)
    return rij


def pol_function_diffable(charge_fn: Callable, params: Any, batch: Any, alpha: int):
    """Calculate the (alpha-component) of a sum of scalar properties using the phase
    approach.

    Args:
        charge_fn: Function that takes (params, batch) and returns charges.
                   Must be differentiable with respect to batch.positions.
                   The user is responsible for including screening or other
                   transformations in this function.
        params: Model parameters.
        batch: Batch object containing atomic structure information.
        alpha: Cartesian component index (0, 1, or 2 for x, y, z).

    Returns:
        complex: Complex sum of values for the given component.
    """
    charges = charge_fn(params, batch).flatten()
    box = batch.cell

    k = 2.0 * jnp.pi / box[alpha][alpha]
    phase = jnp.exp(1j * batch.positions[:, alpha] * k)
    S = jnp.sum(charges * phase, axis=0) / (1j * k)
    return S


def p_real(charge_fn: Callable, params: Any, batch: Any, alpha: int):
    """Calculate the real part of a complex phase of the sum of scalar properties.

    Args:
        charge_fn: Function that takes (params, batch) and returns charges.
        params: Model parameters.
        batch: Batch object.
        alpha: Cartesian component index.

    Returns:
        float: Real part of the sum.
    """
    return pol_function_diffable(charge_fn, params, batch, alpha).real


def p_imag(charge_fn: Callable, params: Any, batch: Any, alpha: int):
    """Calculate the imaginary part of a complex phase of the sum of scalar properties.

    Args:
        charge_fn: Function that takes (params, batch) and returns charges.
        params: Model parameters.
        batch: Batch object.
        alpha: Cartesian component index.

    Returns:
        float: Imaginary part of the sum.
    """
    return pol_function_diffable(charge_fn, params, batch, alpha).imag


def z_i_alpha_beta(charge_fn: Callable, params: Any, batch: Any):
    """Calculate a tensor that is related to derivatives of a global sum of
    properties w.r.t. atomic positions..

    This computes z_i^{alpha,beta} = Re(exp(-ik·r_alpha) * ∂P_alpha/∂r_beta)
    for all atoms and all Cartesian components.

    Args:
        charge_fn: Function that takes (params, batch) and returns charges.
                   Must be differentiable with respect to batch.positions.
                   The user is responsible for including screening or other
                   transformations in this function.
        params: Model parameters.
        batch: Batch object containing atomic structure information.

    Returns:
        jnp.ndarray: Effective (per atom) tensor of shape (n_atoms, 3, 3).
    """

    # Create gradient functions with charge_fn captured
    def p_real_wrapper(params, batch, alpha):
        return p_real(charge_fn, params, batch, alpha)

    def p_imag_wrapper(params, batch, alpha):
        return p_imag(charge_fn, params, batch, alpha)

    deriv_p_real = jax.value_and_grad(
        p_real_wrapper, has_aux=False, argnums=1, allow_int=True
    )
    deriv_p_imag = jax.value_and_grad(
        p_imag_wrapper, has_aux=False, argnums=1, allow_int=True
    )

    z_i_ab = jnp.zeros((batch.positions.shape[0], 3, 3), dtype=jnp.float32)

    box = batch.cell
    positions = batch.positions

    for alpha in range(3):
        pr, derivr = deriv_p_real(params, batch, alpha)
        pi, derivi = deriv_p_imag(params, batch, alpha)
        derivr = derivr.positions
        derivi = derivi.positions

        for beta in range(3):
            k_alpha = 2.0 * jnp.pi / box[alpha][alpha]
            de_phase = jnp.exp(-1j * k_alpha * positions[:, alpha])

            component = (de_phase * (derivr[:, beta] + 1j * derivi[:, beta])).real

            z_i_ab = z_i_ab.at[:, alpha, beta].set(component)

    return z_i_ab
