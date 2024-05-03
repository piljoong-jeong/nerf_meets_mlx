"""### renderer.py
###### in `mlx_nerf/rendering`

Collection of renderers of each feature
"""

import mlx.core as mx
import mlx.nn as nn

class RGBRenderer(nn.Module):
    def __init__(self):
        super().__init__()

    