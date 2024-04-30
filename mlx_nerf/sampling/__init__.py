import numpy as onp
import mlx.core as mx
import mlx.nn as nn
import torch


__all__ = ["add_noise_z", "sample_from_inverse_cdf", "sample_from_inverse_cdf_torch"]


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
    
    raise NotImplementedError(f"[ERROR] using `sample_from_inverse_cdf` introduces cast to `onp.array` back and forth to use `onp.searchsorted` method, which breaks computational graph construction in `mx.compile`! Hence we throw error when you attempted to use this method, but we encourage to use the alternative: `sample_from_inverse_cdf_torch`.")

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
            list(cdf.shape[:-1]) + [n_importance_samples]
        )

    # FIXME: https://github.com/ml-explore/mlx/issues/712#issuecomment-1954282098
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

@torch.no_grad()
def __sample_from_inverse_cdf_torch_impl(
    z_vals, # [B, n]
    weights, # [B, n, 1]
    n_importance_samples, 
    eps=1e-5, 
    is_stratified_sampling=False, 
) -> torch.Tensor:
    # NOTE: since `mlx` does not have `searchsorted` yet, 

    DEVICE = "cpu"

    weights = weights[..., 0] + (histogram_padding := 0.01) # [B, n]
    weights_sum = torch.sum(weights, dim=-1, keepdim=True)
    padding = torch.relu(eps - weights_sum)
    weights = weights + padding / weights.shape[-1]
    weights_sum += padding # [B, 1]

    # NOTE: PDF is proportional to `weights(=transmittance)`, from geometric probability's perspective
    pdf = weights / weights_sum
    cdf = torch.min(
        torch.ones_like(pdf), 
        torch.cumsum(pdf, axis=-1)
    ).to(DEVICE) # [B, n]
    cdf = torch.cat(
        [
            torch.zeros_like(cdf[..., :1]), 
            cdf
        ], dim=-1
    ).to(DEVICE) # [B, n+1?] TODO: figure out why - maybe for grid?

    # NOTE: similar with `t_vals` seen in samplers, but named as `u_vals` to indicate this is importance-sampled ones
    u_vals = None
    if is_stratified_sampling:
        u_vals = torch.linspace(0.0, 1.0, num=n_importance_samples)
        u_vals = u_vals.expand(list(cdf.shape[:-1], [n_importance_samples])) # TODO: double-check
    else: # NOTE: uniform sampling
        u_vals = torch.rand(
            list(cdf.shape[:-1]) + [n_importance_samples] # [B, n_importance_samples]
        )
    u_vals = u_vals.to(DEVICE)

    inds = torch.searchsorted(cdf, u_vals, side="right") 
    # NOTE: clamp indices
    below = torch.clip(inds-1, 0, cdf.shape[-1]-1) # [B, n_importance_samples]
    above = torch.clip(inds-0, 0, cdf.shape[-1]-1) # [B, n_importance_samples]
    cdf_grid_from = torch.gather(cdf, index=below, dim=-1) # [B, n_importance_samples]
    cdf_grid_to = torch.gather(cdf, index=above, dim=-1) # [B, n_importance_samples]
    z_vals_mid = (z_vals[..., 1:] + z_vals[..., :-1]) / 2 # FIXME: should be [B, n_samples], but [B, n_samples-1] for now
    
    # FIXME: `z_vals_mid` will have duplicated point samples in first and last element for each row
    # NOTE: as `below` and `above` can have values as indices in [0, n_samples]
    z_vals_mid = torch.cat(
        [
            z_vals_mid[..., 0, None], 
            z_vals_mid, 
            z_vals_mid[..., -1, None]
        ], dim=-1
    ) # TODO: check if this can lead NaN
    z_mid_from = torch.gather(z_vals_mid, index=below, dim=-1)
    z_mid_to = torch.gather(z_vals_mid, index=above, dim=-1)

    # NOTE: calculate importance
    t_numerator = u_vals - cdf_grid_from
    t_denominator = cdf_grid_to - cdf_grid_from
    t_denominator = torch.where(
        t_denominator < eps, 
        torch.ones_like(t_denominator), # NOTE: as this is denominator, set 1 to do nothing
        t_denominator
    )
    t_vals = torch.nan_to_num(t_numerator / t_denominator, 0)
    t_vals = torch.clip(
        t_vals, 
        min=0.0, max=1.0
    )
    z_vals = z_mid_from + t_vals * (z_mid_to - z_mid_from)

    return z_vals

def sample_from_inverse_cdf_torch(
    z_vals, # [B, n]
    weights, # [B, n, 1]
    n_importance_samples, 
    eps=1e-5, 
    is_stratified_sampling=False, 
) -> mx.array:
    # NOTE: cast to `torch.Tensor`
    # NOTE: `torch.searchsorted` not supports `mps` backend
    z_vals_torch = torch.from_numpy(onp.array(z_vals))# .to("mps")
    weights_torch = torch.from_numpy(onp.array(weights))# .to("mps")
    
    z_importance_samples = __sample_from_inverse_cdf_torch_impl(
        z_vals_torch, 
        weights_torch, 
        n_importance_samples, 
    )

    # NOTE: cast back to `mx.array`
    z_importance_samples = z_importance_samples.detach().cpu().numpy()
    z_importance_samples = mx.array(z_importance_samples)

    # NOTE: concat fine samples with original depth, and sort
    z_vals_fine = mx.sort(mx.concatenate([z_vals, z_importance_samples], axis=-1), axis=-1) # [B, n_samples + n_importance_samples]

    return z_vals_fine