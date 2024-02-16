"""


"""

import mlx.core as mx
import tyro


import this_project; this_project.import_project_root()
from mlx_nerf import entrypoints

if __name__ == "__main__":
    mx.set_default_device(mx.gpu)
    tyro.cli(
        # entrypoints.viser_record3d
        # entrypoints.viser_image_learning
        entrypoints.viser_gui
    )