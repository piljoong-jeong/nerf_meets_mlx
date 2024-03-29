"""### render_py
###### in `mlx_nerf/rendering`

Execution flow:
    1. render(...)
    2. batchify_rays(...)
    3. render_rays(...)
    4. raw2outputs(...)
"""

import numpy as onp
import mlx.core as mx
import mlx.nn as nn
import torch

from mlx_nerf.rendering import ray
from mlx_nerf import sampling
from mlx_nerf.sampling import uniform, linear_disparity

def raw2outputs(
    raw, 
    z_vals, # NOTE: [B, `n_depth_samples` from `render_rays(...)`]
    rays_d, 
    raw_noise_std=0, 
    white_bkgd=False, 
    pytest=False,
):
    """
    
    NOTE: here, 
        * alpha == density
        * weights == transmittance
    """

    # NOTE: decompose `raw`
    raw_rgb = raw[..., :3] # NOTE: dim = [B, n, 3]
    raw_density = raw[..., 3] # NOTE: dim = [B, n]
    # raw_density = mx.expand_dims(raw_density, axis=-1) # NOTE: dim = [B, n, 1]

    # NOTE: add noise if desired, to avoid overfitting
    if raw_noise_std > 0.0:
        noise = mx.random.normal(raw_density.shape) * raw_noise_std
        raw_density = raw_density + noise

    # NOTE: relative distance
    delta_dists = z_vals[..., 1:] - z_vals[..., :-1] # NOTE: [B, n_depth_samples-1]
    # NOTE: add infinite value at the end of `dists`
    DIST_LIMIT = mx.array(1e10)
    DIST_LIMIT = mx.repeat(DIST_LIMIT[None, ...], repeats=z_vals.shape[0], axis=0) # [B, 1]
    DIST_LIMIT = mx.expand_dims(DIST_LIMIT, axis=-1)
    delta_dists = mx.concatenate(
        [
            delta_dists, 
            DIST_LIMIT
        ], axis=-1
    ) # [B, n]
    # NOTE: rotate `dists` w.r.t. direction
    # delta_dists = mx.expand_dims(delta_dists, axis=-1)
    delta_dists = delta_dists * mx.linalg.norm(rays_d[..., None, :], axis=-1) # [B, n]
    

    # TODO: --------- double-check below ----------

    # NOTE: compute weight: composed alpha
    # NOTE: from last paragraph, below eq. (3)
    # alpha = 1.0 - mx.exp(-nn.relu(raw_alpha) * delta_dists)
    delta_densities = delta_dists * raw_density # [B, n]
    delta_densities = mx.expand_dims(delta_densities, axis=-1)
    alphas = 1.0 - mx.exp(-nn.relu(delta_densities)) # [B, n]
    
    transmittance = mx.cumsum(delta_densities[..., :-1, :], axis=-2) # FIXME: [B-1, n]???
    transmittance = mx.concatenate(
        [
            mx.zeros((*transmittance.shape[:1], 1, 1)), 
            transmittance
        ], 
        axis=-2
    )
    transmittance = mx.exp(-transmittance)
    weights = alphas * transmittance # [B, n, 1]

    # TODO: implement each as a renderer
    rgb_map = mx.sum(weights * raw_rgb, axis=-2) # [B, 3]
    depth_map = mx.sum(weights[..., 0] * z_vals, axis=-1)[..., None] # [B, 1]
    disp_map = 1.0 / mx.maximum(
        1e-10 * mx.ones_like(depth_map), 
        depth_map/mx.sum(weights, axis=-2)
    ) # [B, 1]
    acc_map = mx.sum(weights, axis=-2) # [B, 1]

    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map) # TODO: validate

    

    return rgb_map, disp_map, acc_map, weights, depth_map

def decompose_ray_batch(
    rays_batch_linear, # NOTE: [B, rays_o, rays_d, near, far, viewdirs (, time)]
    is_time_included: bool = False # TODO: make flexable, like `dict[str, int]`?
):
    
    rays_o, rays_d = rays_batch_linear[:, 0:3], rays_batch_linear[:, 3:6]
    
    bounds = mx.reshape(rays_batch_linear[..., 6:8+int(is_time_included)], [-1, 1, 2+int(is_time_included)])
    near, far = bounds[..., 0], bounds[..., 1]
    frame_time = bounds[..., 2] if is_time_included else None
    viewdirs = rays_batch_linear[:, -3:]
    
    return rays_o, rays_d, near, far, viewdirs, frame_time

def render_rays(
    rays_batch_linear, # NOTE: [B, rays_o, rays_d, near, far, viewdirs]
    network_coarse, 
    network_query_fn, 
    n_depth_samples, 
    retraw=False, 
    lindisp=False, 
    perturb=0.0, 
    N_importance=0, 
    network_fine=None, 
    white_bkgd=False, 
    raw_noise_std=0.0, 
    verbose=False, 
    pytest=False,
    **kwargs, 
):
    
    n_rays = rays_batch_linear.shape[0]
    rays_o, rays_d, near, far, viewdirs, _ = decompose_ray_batch(rays_batch_linear)

    # TODO: separate generation of `z_vals` - to allow users to choose which depth sampling they want to use (e.g., uniform, inverse_cdf, ...)
    # NOTE: sample z-values for coarse NeRF
    if not lindisp:
        z_vals = uniform.sample_z(near, far, n_depth_samples)
    else:
        z_vals = linear_disparity.sample_z(near, far, n_depth_samples)
    # TODO: use `expand` when `mlx` implements it - `repeat` copies data but `expand` provides view
    # z_vals = mx.repeat(z_vals[None, ...], repeats=n_rays, axis=0) # FIXME: validate: [n_rays, n_depth_samples] -> [n_rays, n_rays, n_depth_samples]
    z_vals = sampling.add_noise_z(z_vals, perturb)

    pos = rays_o[..., None, :] + (z_vals[..., :, None] * rays_d[..., None, :]) # TODO: validate

    raw = network_query_fn(pos, viewdirs, network_coarse) # returns [rgb, alpha]
    ret = {}
    if retraw: ret["raw"] = raw

    rgb_coarse, disp_coarse, acc_coarse, weights, depth_map = raw2outputs(
        raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest
    )
    ret["rgb_map"] = rgb_coarse
    ret["disp_map"] = disp_coarse
    ret["acc_map"] = acc_coarse
    ret["rgb_coarse"] = rgb_coarse
    ret["disp_coarse"] = disp_coarse
    ret["acc_coarse"] = acc_coarse

    # NOTE: for importance sampling
    ret["z_vals"] = z_vals
    ret["weights"] = weights
    
    return ret

def render_rays_eval(
    rays_batch_linear, # NOTE: [B, rays_o, rays_d, near, far, viewdirs]
    network_coarse, 
    network_query_fn, 
    n_depth_samples, 
    retraw=False, 
    lindisp=False, 
    perturb=0.0, 
    N_importance=0, 
    network_fine=None, 
    white_bkgd=False, 
    raw_noise_std=0.0, 
    verbose=False, 
    pytest=False,
    **kwargs, 
):
    
    n_rays = rays_batch_linear.shape[0]
    rays_o, rays_d, near, far, viewdirs, _ = decompose_ray_batch(rays_batch_linear)

    # TODO: separate generation of `z_vals` - to allow users to choose which depth sampling they want to use (e.g., uniform, inverse_cdf, ...)
    # NOTE: sample z-values for coarse NeRF
    if not lindisp:
        z_vals = uniform.sample_z(near, far, n_depth_samples)
    else:
        z_vals = linear_disparity.sample_z(near, far, n_depth_samples)
    # TODO: use `expand` when `mlx` implements it - `repeat` copies data but `expand` provides view
    # z_vals = mx.repeat(z_vals[None, ...], repeats=n_rays, axis=0) # FIXME: validate: [n_rays, n_depth_samples] -> [n_rays, n_rays, n_depth_samples]
    z_vals = sampling.add_noise_z(z_vals, perturb)

    pos = rays_o[..., None, :] + (z_vals[..., :, None] * rays_d[..., None, :]) # TODO: validate

    raw = network_query_fn(pos, viewdirs, network_coarse) # returns [rgb, alpha]
    ret = {}
    if retraw: ret["raw"] = raw

    rgb_coarse, disp_coarse, acc_coarse, weights, depth_map = raw2outputs(
        raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest
    )
    ret["rgb_map"] = rgb_coarse
    ret["disp_map"] = disp_coarse
    ret["acc_map"] = acc_coarse
    ret["rgb_coarse"] = rgb_coarse
    ret["disp_coarse"] = disp_coarse
    ret["acc_coarse"] = acc_coarse

    # NOTE: for importance sampling
    ret["z_vals"] = z_vals
    ret["weights"] = weights

    # NOTE: `torch.searchsorted` not supports `mps` backend
    z_vals_torch = torch.from_numpy(onp.array(z_vals))# .to("mps")
    weights_torch = torch.from_numpy(onp.array(weights))# .to("mps")
    z_importance_samples = sampling.sample_from_inverse_cdf_torch(
        z_vals_torch, 
        weights_torch, 
        N_importance, 
    )
    z_importance_samples = z_importance_samples.detach().cpu().numpy()
    z_importance_samples = mx.array(z_importance_samples)

    z_vals = mx.sort(mx.concatenate([z_vals, z_importance_samples], axis=-1), axis=-1) # TODO: double check
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    run_fn = network_fine if network_fine else network_coarse
    raw = network_query_fn(pts, viewdirs, run_fn)
    rgb, disp, acc, weight, depth = raw2outputs(
        raw, 
        z_vals, 
        rays_d, 
        raw_noise_std, 
        white_bkgd
    )
    ret["rgb_map"] = rgb
    ret["disp_map"] = disp
    ret["acc_map"] = acc
    
    return ret

def batchify_rays(
    rays_linear, 
    chunk=1024*32, 
    **kwargs
):
    
    render_rays_func = kwargs["render_rays_func"]
    
    results_batched = {}
    for i in range(0, rays_linear.shape[0], chunk):
        results = render_rays_func(rays_linear[i:i+chunk], **kwargs)

        # NOTE: accumulate per-batch results to `results_batched`
        for key, val in results.items():
            if key not in results_batched:
                results_batched[key] = []
            results_batched[key].append(val)

    results_batched = {
        key: mx.concatenate(val, axis=0)
        for key, val in results_batched.items()
    }

    return results_batched

def render(
    H, 
    W, 
    K, 
    chunk=1024*32, 
    rays=None,
    c2w=None, 
    ndc=True, 
    near=0.0, 
    far=1.0,
    use_viewdirs=False, 
    c2w_staticcam=None, 
    **kwargs
):

    if c2w is None and rays is not None:
        rays_o, rays_d = rays
    elif c2w is not None:
        rays_o, rays_d = ray.get_rays(H, W, K, c2w)
        rays_o = mx.array(rays_o)
        rays_d = mx.array(rays_d)

    rays_shape = rays_d.shape

    if c2w is not None:
        rays_o = mx.reshape(rays_o, [-1, 3]).astype(mx.float32)
        rays_d = mx.reshape(rays_d, [-1, 3]).astype(mx.float32)

    if use_viewdirs:
        viewdirs = rays_d

        # NOTE: if `c2w` is given, we generate rays from given view on demand
        if c2w_staticcam is not None:
            rays_o, rays_d = ray.get_rays(H, W, K, c2w_staticcam)
            rays_o = mx.array(rays_o)
            rays_d = mx.array(rays_d)
            # TODO: validate
            rays_o = mx.reshape(rays_o, [-1, 3]).astype(mx.float32)
            rays_d = mx.reshape(rays_d, [-1, 3]).astype(mx.float32)
        viewdirs = viewdirs / mx.linalg.norm(viewdirs, axis=-1, keepdims=True)
        viewdirs = mx.reshape(viewdirs, [-1, 3]).astype(mx.float32)

    

    if ndc:
        rays_o, rays_d = ray.ndc_rays(
            H, W, K[0][0], 
            1.0, 
            rays_o, rays_d
        )

    near = near * mx.ones_like(rays_d[..., :1])
    far = far * mx.ones_like(rays_d[..., :1])

    # NOTE: concat all ray-related features
    rays = mx.concatenate(
        [rays_o, rays_d, near, far], # FIXME: should `near` and `far` included for every ray tensor?
        axis=-1
    )
    if use_viewdirs:
        rays = mx.concatenate([rays, viewdirs], axis=-1)

    results_batched = batchify_rays(rays, chunk, **kwargs)
    # NOTE: shape back linearized rendered results to `rays.shape`
    for key, val in results_batched.items():
        results_batched[key] = mx.reshape(
            val, 
            tuple(list(rays_shape[:-1]) + list(val.shape[1:]))
        )

    k_extract = ["rgb_map", "disp_map", "acc_map"]
    ret_list = [results_batched[k] for k in k_extract]
    ret_dict = {
        k: v for k, v in results_batched.items()
        if k not in k_extract
    }

    return ret_list + [ret_dict]

