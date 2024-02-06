import time
from pathlib import Path
from typing import List

import imageio.v3 as imageio
import numpy as onp
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
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

# FIXME
def batch_iterate(batch_size, X, y):
    perm = mx.array(onp.random.permutation(y.size))
    for s in range(0, y.size, batch_size):
        ids = perm[s : s + batch_size]
        yield X[ids], y[ids]

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
    img_gt = img_gt.astype(mx.float32) / 255.0
    # img_gt = mx.repeat(img_gt[..., None], repeats=3, axis=-1)
    server.add_image(
        "/gt",
        onp.array(mx.repeat(img_gt[..., None], repeats=3, axis=-1), copy=False),
        4.0,
        4.0,
        format="png", # NOTE: `jpeg` gives strangely stretched image
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(4.0, 4.0, 0.0),
    )

    
    img_pred = mx.random.randint(0, 256, img_gt.shape, dtype=mx.uint8)
    img_pred = img_pred.astype(mx.float32) / 255.0
    # pred = mx.repeat(pred, repeats=3, axis=-1)

    # NOTE: embedding func test
    N_INPUT_DIMS = 2
    embed, out_dim = embedding.get_embedder(10, n_input_dims=N_INPUT_DIMS)
    input = mx.zeros(N_INPUT_DIMS)
    output = embed(input)

    # NOTE: NeRF
    model = NeRF(
        channel_input=N_INPUT_DIMS, # NOTE: pixel position 
        channel_input_views=0, 
        channel_output=1, 
        is_use_view_directions=False, 
    )
    mx.eval(model.parameters())

    def mlx_mse(model, x, y):
        return mx.mean((model.forward(x) - y) ** 2)
    loss_and_grad_fn = nn.value_and_grad(model, mlx_mse)



    coords = onp.meshgrid(
        
        onp.arange(0, img_gt.shape[0]), 
        onp.arange(0, img_gt.shape[1]), 
        indexing="ij"
    ) # NOTE: `list`
    
    # TODO: convert all `np.ndarray`s into `mx.array`
    coords[0] = mx.array(coords[0])
    coords[1] = mx.array(coords[1])
    
    # TODO: stack meshgrids
    coords = mx.stack(coords, axis=-1)
    
    # TODO: reshape, now [H, W] has been flatten
    coords = mx.reshape(coords, [-1, 2])
    
    optimizer = optim.SGD(learning_rate=0.999)

    while True:
        server.add_image(
            "/pred",
            onp.array(mx.repeat(img_pred, repeats=3, axis=-1), copy=False), # NOTE: view
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

        # for coords, y in batch_iterate(batch_size:=1, pred, img_gt):
        #     print(f"[DEBUG] {coords.shape=}")
        #     print(f"[DEBUG] {y.shape=}")
        #     exit()
        #     loss, grads = loss_and_grad_fn(model, coords, y)
        #     optimizer.update(model, grads)
        #     mx.eval(model.parameters(), optimizer.state)


        # print(f"{img_pred.shape=}")
        # print(f"{img_gt.shape=}")
        # loss, grads = loss_and_grad_fn(model, img_pred, img_gt)
        # optimizer.update(model, grads)
        # mx.eval(model.parameters(), optimizer.state)
        # exit()

        time.sleep(0.1)
