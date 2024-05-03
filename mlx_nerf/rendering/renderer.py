"""### renderer.py
###### in `mlx_nerf/rendering`

Collection of renderers of each feature
"""

import mlx.core as mx
import mlx.nn as nn

class RGBRenderer(nn.Module):
    def __init__(self):
        super().__init__()

    @classmethod
    def combine_rgb(
        cls, 
        rgb, 
        weights, 
        is_white_background = False,
    ):
        
        rgb_map = mx.sum(weights * rgb, axis=-2)
        acc_weight = mx.sum(weights, axis=-2)

        if is_white_background:
            rgb_map = rgb_map + (1.0 - acc_weight)

        return rgb_map
    
    def forward(
        self, 
        rgb, 
        weights, 
        is_white_background = False, 
    ):
        
        if not self.training:
            # rgb = mx.nan_to_num(rgb) # NOTE: `mlx` does not have `nan_to_num`
            pass
        rgb = self.combine_rgb(
            rgb, weights, is_white_background=is_white_background, 
        )
        if not self.training:
            rgb = mx.clip(rgb, min=0.0, max=1.0)

        return rgb