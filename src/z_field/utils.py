import jax.numpy as jnp


def distribute_excess_charge(apt, node_mask):
    """Distribute excess charge from acoustic sum rule equally to all atoms.

    Args:
        apt: jnp.ndarray of shape (..., N, 3, 3), predicted atomic polar
        tensors
        node_mask: jnp.ndarray of shape (..., N), mask for active atoms
    """
    excess_charge = acoustic_sum_rule(apt)
    num_active_atoms = jnp.sum(node_mask)
    charges_to_redistribute = jnp.where(
        num_active_atoms > 0,
        excess_charge / num_active_atoms,
        jnp.zeros_like(excess_charge),
    )  # Shape: (3, 3)
    apt -= charges_to_redistribute
    apt *= node_mask[..., None, None]  # apply mask
    return apt


def acoustic_sum_rule(apts):
    """Apply acoustic sum rule to predicted atomic polar tensors.

    Args:
        apts: jnp.ndarray of shape (..., N, 3, 3), predicted atomic polar
        tensors

    Returns:
        excess_charge: jnp.ndarray of shape (..., 3, 3), the excess over the
        system
    """
    excess_charge = jnp.einsum("...nij->...ij", apts)
    return excess_charge


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
