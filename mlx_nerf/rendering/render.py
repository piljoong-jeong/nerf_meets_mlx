"""### render_py
###### in `mlx_nerf/rendering`

Execution flow:
    1. render(...)
    2. batchify_rays(...)
    3. render_rays(...)
"""

import mlx.core as mx

from mlx_nerf.rendering import ray
from mlx_nerf.sampling import uniform

def decompose_ray_batch(
    rays_batch_linear, # NOTE: [B, rays_o, rays_d, near, far, viewdirs (, time)]
    is_time_included: bool = False # TODO: make flexable, like `dict[str, int]`?
):
    
    rays_o, rays_d = rays_batch_linear[:, 0:3], rays_batch_linear[:, 3:6]
    near = rays_batch_linear[:, 6]
    far = rays_batch_linear[:, 7]
    viewdirs = rays_batch_linear[:, 8:11]
    frame_time = rays_batch_linear[:, -1] if is_time_included else None
    
    return rays_o, rays_d, near, far, viewdirs, frame_time

def render_rays(
    rays_batch_linear, # NOTE: [B, rays_o, rays_d, near, far, viewdirs]
    network_fn, 
    network_query_fn, 
    n_samples, 
    retraw=False, 
    lindisp=False, 
    perturb=0.0, 
    N_imporatance=0, 
    network_fine=None, 
    white_bkgd=False, 
    raw_noise_std=0.0, 
    verbose=False, 
    pytest=False,
):
    
    n_rays = rays_batch_linear.shape[0]
    rays_o, rays_d, near, far, viewdirs, _ = decompose_ray_batch(rays_batch_linear)

    # NOTE: sample z-values for coarse NeRF
    z_vals = uniform.uniform_sample_z(near, far, n_samples)

    return

def batchify_rays(
    rays_linear, 
    chunk=1024*32, 
    **kwargs
):
    
    results_batched = {}
    for i in range(0, rays_linear.shape[0], chunk):
        results = render_rays(rays_linear[i:i+chunk], **kwargs)

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
    ndc=True, 
    near=0.0, 
    far=1.0,
    use_viewdirs=False, 
    c2w_staticcam=None, 
    **kwargs
):

    rays_o, rays_d = rays

    if use_viewdirs:
        viewdirs = rays_d
        if c2w_staticcam is not None:
            rays_o, rays_d = ray.get_rays(H, W, K, c2w_staticcam)
            rays_o = mx.array(rays_o)
            rays_d = mx.array(rays_d)
        viewdirs = viewdirs / mx.linalg.norm(viewdirs, axis=-1, keepdims=True)
        viewdirs = mx.reshape(viewdirs, [-1, 3]).astype(mx.float32)

    rays_shape = rays_d.shape

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
            list(rays_shape[:-1]) + list(val.shape[1:])
        )

    k_extract = ["rgb_map", "disp_map", "acc_map"]
    ret_list = [results_batched[k] for k in k_extract]
    ret_dict = {
        k: v for k, v in results_batched
        if k not in k_extract
    }

    return ret_list + [ret_dict]