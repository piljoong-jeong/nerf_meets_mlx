import numpy as onp
import mlx.core as mx

__all__ = ["add_noise_z"]

def add_noise_z(
    z_vals, 
    perturb=0.0 # TODO: refactor out
):
    
    # TODO: refactor out
    if perturb <= 0.0:
        return z_vals
    
    t_rand = mx.random.uniform(shape=z_vals.shape)

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

def sample_from_inverse_cdf(
    z_vals, 
    weights, 
    n_importance_samples, 
    eps = 1e-5, 
    is_stratified_sampling=False, 
):
    z_vals_mid = (z_vals[..., 1:] + z_vals[..., :-1]) / 2
    weights = weights[..., 1:-1] + eps # [B, n_samples-1]
    weights_sum = mx.sum(weights, axis=-1, keepdims=True)

    # NOTE: PDF is proportional to `weights(=transmittance)`, from geometric probability's perspective
    pdf = weights / weights_sum
    cdf = mx.cumsum(pdf, axis=-1)
    cdf = mx.minimum(mx.ones_like(cdf), cdf) # NOTE: clip
    cdf = mx.concatenate(
        [
            mx.zeros_like(cdf[..., 0]), 
            cdf
        ], axis=-1
    ) # [B, n_samples]

    # NOTE: similar with `t_vals` seen in samplers, but named as `u_vals` to indicate they're importance-sampled ones
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

