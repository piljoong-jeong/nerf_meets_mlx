from typing import Union
import mlx.core as mx
import numpy as onp

def get_rays(H: int, W: int, K: Union[onp.ndarray, mx.array], c2w: Union[onp.ndarray, mx.array]):


    i, j = onp.meshgrid(
        onp.arange(W, dtype=onp.float32), 
        onp.arange(H, dtype=onp.float32), 
        indexing="xy"
    )
    
    fx = K[0][0]
    fy = K[1][1] # NOTE: in NeRF, fx == fy
    cx = K[0][2]
    cy = K[1][2]

    dirs = onp.stack(
        [
            (i-cx)/fx, 
            -(j-cy)/fy, 
            -onp.ones_like(i)
        ], axis=-1
    )

    rays_d = onp.sum(
        dirs[..., None, :] * c2w[:3, :3], axis=-1
    )
    rays_o = onp.broadcast_to(c2w[:3, -1], rays_d.shape) # c2w[:3, -1].expand(rays_d.shape)

    # TODO: cast? at least we don't have to store this as onp.ndarray
    return rays_o, rays_d
