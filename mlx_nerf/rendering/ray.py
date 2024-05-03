# TODO: move to `cameras`

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

# TODO: why `near` only?
# NOTE: `rays_o` to `near` plane, endpoint is `1` (in NDC)
def ndc_rays(H, W, focal, near, rays_o, rays_d):

    # NOTE: shift `rays_o` to near plane (z = -`near`)
    # NOTE: see last paragraph in Appendix. C
    o_z = rays_o[..., 2]
    d_z = rays_d[..., 2]
    # NOTE: o_n = o + t_n*d, for t_n = -(n + o_z) / d_z
    t_n = -(near + o_z) / d_z
    rays_o = rays_o + t_n[..., None] * rays_d

    # NOTE: map `rays_o` & `rays_d` to NDC space
    o_x = rays_o[..., 0]
    o_y = rays_o[..., 1]
    o_z = rays_o[..., 2]
    
    # NOTE: eq. (25)
    o0 = (-focal / (0.5*W)) * (o_x / o_z)
    o1 = (-focal / (0.5*H)) * (o_y / o_z)
    o2 = (1.0 + 2.0*near / o_z) 

    # NOTE: eq. (26)
    d_x = rays_d[..., 0]
    d_y = rays_d[..., 1]
    d_z = rays_d[..., 2]
    d0 = (-focal / (0.5*W)) * (d_x/d_z - o_x/o_z)
    d1 = (-focal / (0.5*H)) * (d_y/d_z - o_y/o_z)
    d2 = -2.0 * near * (1.0 / o_z)

    return (
        (rays_o := mx.stack([o0, o1, o2], axis=-1)), 
        (rays_d := mx.stack([d0, d1, d2], axis=-1))
    )

