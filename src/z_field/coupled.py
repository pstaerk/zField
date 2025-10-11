import jax
import jax.numpy as jnp


def z_i_pbc(batch, params, q_function):
    mask = batch.unit_cell_mask
    to_replicate = batch.to_replicate_idx
    nr_nodes = batch.unfolded_nodes.shape[0]

    def calc_q_sc_sum(batch, params, mask):
        rijs = (batch.unfolded_positions[batch.unfolded_others] -
                batch.unfolded_positions[batch.unfolded_centers])
        qs = q_function(batch, params, rijs)
        qs *= mask
        return jnp.sum(qs), qs

    # gradient of sum_j q_j wrt. r_i,beta
    d_sumq_drib, qs = jax.grad(calc_q_sc_sum, allow_int=True,
                               has_aux=True, argnums=0)(batch, params, mask)

    # outer product of all positions with the corresponding gradient
    outer_product = (batch.unfolded_positions[:, :, None] *
                     d_sumq_drib.unfolded_positions[:, None, :])

    # sum of replica positions * gradient
    grad_q_sc_sum = jax.ops.segment_sum(
        outer_product,
        to_replicate,
        num_segments=nr_nodes,
    )

    # For the Barycenter calculation, we use positions which are not in the AD
    # graph in order to only be able to calc the derivative of the sum.
    stopgrad_positions = jax.lax.stop_gradient(batch.unfolded_positions)

    def z_alpha(alpha):
        def barycenter(batch, params, mask):
            rijs = (batch.unfolded_positions[batch.unfolded_others] -
                    batch.unfolded_positions[batch.unfolded_centers])
            qs = q_function(batch, params, rijs)
            qs *= mask  # Only restrict to simulation cell atoms, no ghosts

            return jnp.sum(qs[..., None] * stopgrad_positions, axis=0)[alpha]

        z_alpha = jax.grad(lambda b: barycenter(b, params, mask),
                           allow_int=True, argnums=0)(batch).unfolded_positions

        z_alpha = jax.ops.segment_sum(
            z_alpha,
            to_replicate,
            num_segments=nr_nodes,
            )
        return z_alpha

    s1 = jax.vmap(lambda a: z_alpha(a))(jnp.arange(3)).transpose((1, 0, 2))

    Z = s1 - grad_q_sc_sum

    # add q_i * delta_alpha_beta
    Z = Z + jnp.einsum('i,ab->iab', qs, jnp.eye(3))
    return Z
