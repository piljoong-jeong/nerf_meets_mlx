import mlx.core as mx

def uniform_sample_z(
    near: float, 
    far: float, 
    n_samples: int, 
):
    
    t_vals = mx.linspace(0.0, 1.0, num=n_samples)
    z_from = (near * (1.0 - t_vals))
    z_to = (far * t_vals)
    z_vals = z_from + z_to

    return z_vals
