import time
from pathlib import Path
from typing import List

import imageio.v3 as imageio
import numpy as onp
import mlx.core as mx
import mlx.nn as nn
import viser
import viser.extras
import viser.transforms as tf
from tqdm.auto import tqdm

from this_project import get_project_root, PJ_PINK
from mlx_nerf.models import embedding
from mlx_nerf.models.NeRF import NeRF
from mlx_nerf.ops.metric import MSE

def init_gui(server: viser.ViserServer, **config) -> None:

    num_frames = config.get("num_frames", 10000)

    with server.add_gui_folder("Playback"):
        gui_slider_iterations = server.add_gui_slider(
            "# Iterations",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=1000,
            disabled=False,
        )
        gui_btn_start = server.add_gui_button("Start Learning", disabled=True)
        
        

    server.configure_theme(
        # titlebar_content="NeRF using MLX", # FIXME: this results blank page
        control_layout="fixed",
        control_width="medium",
        dark_mode=True,
        show_logo=False,
        show_share_button=False,
        brand_color=PJ_PINK
    )

    return



def main(
    path_assets: Path = get_project_root() / "assets",
    downsample_factor: int = 4,
    max_frames: int = 100,
    share: bool = False,
):

    server = viser.ViserServer()

    init_gui(
        server, 
    )


    img_gt = mx.array(imageio.imread(str(path_img := path_assets / "images/albert.jpg")))
    server.add_image(
        "/gt",
        onp.array(img_gt, copy=False),
        4.0,
        4.0,
        format="png", # NOTE: `jpeg` gives strangely stretched image
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(4.0, 4.0, 0.0),
    )

    
    pred = mx.random.randint(0, 256, (400, 400, 3), dtype=mx.uint8)

    # NOTE: embedding func test
    N_INPUT_DIMS = 2
    embed, out_dim = embedding.get_embedder(10, n_input_dims=N_INPUT_DIMS)
    input = mx.zeros(N_INPUT_DIMS)
    output = embed(input)
    print(f"[DEBUG] {input=}")
    print(f"[DEBUG] {input.shape=}")
    print(f"[DEBUG] {output=}")
    print(f"[DEBUG] {output.shape=}")

    # NOTE: NeRF
    model = NeRF(
        channel_input=2, # NOTE: pixel position 
        channel_input_views=0, 
        channel_output=1, 
        is_use_view_directions=False, 
    )
    print(f"[DEBUG] evaluating {model=} ...")
    mx.eval(model.parameters())
    print(f"[DEBUG] evaluating {model=} ... done.")

    def mlx_mse(model, x, y):
        return mx.mean((model(x) - y) ** 2)
    loss_and_grad_fn = nn.value_and_grad(model, mlx_mse)




    while True:
        server.add_image(
            "/pred",
            onp.array(pred, copy=False), # NOTE: view
            4.0,
            4.0,
            format="png", # NOTE: `jpeg` gives strangely stretched image
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(4.0, 0.0, 0.0),
        )

        """
        TODO: learning

        - get pixel sample positions
        - pass into encoder => augment sample positions
        - (augmented i.e., encoded sample positions, pixel color) => MLP
        - `mlx`-dependent optimization implementations (say, `.eval()`?)
        """

        loss, grads = loss_and_grad_fn(model, ) # TODO

        time.sleep(0.1)
