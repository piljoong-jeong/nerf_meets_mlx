import numpy as onp
import mlx.core as mx
import mlx.nn as nn

__all__ = ["add_noise_z", "sample_from_inverse_cdf"]

def add_noise_z(
    z_vals, 
    strength=1.0 # TODO: refactor out
):
    if strength <= 0.0:
        return z_vals
    
    t_rand = mx.random.uniform(shape=z_vals.shape) * strength

    # NOTE: [mid(0, 1), mid(1, 2), ...]
    mids = 0.5 * (z_vals[..., :-1] + z_vals[..., 1:])
    upper = mx.concatenate(
        [mids, z_vals[..., -1]], axis=-1
    )
    lower = mx.concatenate(
        [z_vals[..., 0], mids], axis=-1
    )

    # NOTE: add randomness for each bin, where each strength is [0, size(bin)/2]
    z_vals = lower + (upper - lower) * t_rand

    return z_vals

# TODO: can this be stand as an independent sampler? 
def sample_from_inverse_cdf(
    z_vals, # [B, n]
    weights, # [B, n, 1]
    n_importance_samples, 
    eps=1e-5, 
    is_stratified_sampling=False, 
):
    
    weights = weights[..., 0] + (histogram_padding := 0.01) # [B, n]
    weights_sum = mx.sum(weights, axis=-1, keepdims=True)
    padding = nn.relu(eps - weights_sum)
    weights = weights + padding / weights.shape[-1]
    weights_sum += padding # [B, 1]

    # NOTE: PDF is proportional to `weights(=transmittance)`, from geometric probability's perspective
    pdf = weights / weights_sum
    cdf = mx.minimum(
        mx.ones_like(pdf), 
        mx.cumsum(pdf, axis=-1)
    ) # [B, n]
    cdf = mx.concatenate(
        [
            mx.zeros_like(cdf[..., :1]), 
            cdf
        ], axis=-1
    ) # [B, n+1?] TODO: figure out why - maybe for grid?

    # NOTE: similar with `t_vals` seen in samplers, but named as `u_vals` to indicate this is importance-sampled ones
    u_vals = None
    if is_stratified_sampling:
        u_vals = mx.linspace(0.0, 1.0, num=n_importance_samples)
        u_vals = mx.repeat(u_vals[None, ...], repeats=cdf.shape[0], axis=0) # TODO: double-check
    else: # NOTE: uniform sampling
        u_vals = mx.random.normal(
            tuple(list(cdf.shape[:-1]) + list(n_importance_samples))
        )

    # TODO: check if it's OK to use numpy's implementation; I guess so as no grad would be generated, but just to make sure...
    inds = onp.searchsorted(onp.array(cdf, copy=False), u_vals, side="right")
    # NOTE: clamp indices
    below = mx.clip(inds-1, 0, cdf.shape[-1]-1)
    above = mx.clip(inds-0, 0, cdf.shape[-1]-1)
    cdf_grid_from = mx.take(cdf, indices=below, axis=-1)
    cdf_grid_to = mx.take(cdf, indices=above, axis=-1)
    z_vals_mid = (z_vals[..., 1:] + z_vals[..., :-1]) / 2 # [B, n_samples-1]
    z_mid_from = mx.take(z_vals_mid, indices=below, axis=-1)
    z_mid_to = mx.take(z_vals_mid, indices=above, axis=-1)

    # NOTE: calculate importance
    t_numerator = u_vals - cdf_grid_from
    t_denominator = cdf_grid_to - cdf_grid_from
    t_denominator = mx.where(
        t_denominator < eps, 
        mx.ones_like(t_denominator), # NOTE: as this is denominator, set 1 to do nothing
        t_denominator
    )
    t_vals = mx.clip(
        (
            t_numerator / 
            t_denominator
        ), 
        a_min=0.0, a_max=1.0
    )
    z_vals = z_mid_from + t_vals * (z_mid_to - z_mid_from)

    return z_vals
