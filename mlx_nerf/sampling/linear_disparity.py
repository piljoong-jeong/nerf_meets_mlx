"""### linear_disparity.py
###### in `mlx_nerf/sampling`

"""

import mlx.core as mx

def sample_z(
    near: float, 
    far: float, 
    n_samples: int
):
    
    t_vals = mx.linspace(0.0, 1.0, num=n_samples)
    z_from = 1.0 / (near * (1.0 - t_vals))
    z_to = 1.0 / (far * t_vals)
    z_vals = 1.0 / (z_from + z_to)

    return z_vals
    