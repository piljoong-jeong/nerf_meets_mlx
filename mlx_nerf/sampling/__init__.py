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