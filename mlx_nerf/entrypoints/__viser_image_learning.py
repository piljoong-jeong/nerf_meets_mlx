import time
from pathlib import Path
from typing import List
from PIL import Image

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

    server.reset_scene()

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

def batch_iterate(batch_size: int, X: mx.array, y: mx.array):
    
    assert X.shape == y.shape
    
    coords = onp.meshgrid(
        
        onp.arange(0, y.shape[0]), 
        onp.arange(0, y.shape[1]), 
        indexing="ij"
    ) # NOTE: `list`
    
    # TODO: convert all `np.ndarray`s into `mx.array`
    coords[0] = mx.array(coords[0])
    coords[1] = mx.array(coords[1])
    
    # TODO: stack meshgrids
    coords = mx.stack(coords, axis=-1)
    
    # TODO: reshape, now [H, W] has been flatten
    coords = mx.reshape(coords, [-1, 2])
    
    choice = mx.array(onp.random.choice(coords.shape[0], size=[coords.shape[0]], replace=False)) # NOTE: [H*W]
    for s in range(0, y.size, batch_size):
        selected = choice[s : s + batch_size]
        
        X_batch = coords[selected] 
        y_batch = y[coords[selected][:, 0], coords[selected][:, 1]]

        yield (
            X_batch, 
            y_batch
        )

        

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


    img_gt = imageio.imread(str(path_img := path_assets / "images/albert.jpg"))
    img_gt = onp.asarray(Image.fromarray(img_gt).resize((100, 100)))
    img_gt = mx.array(img_gt)
    img_gt = img_gt.astype(mx.float32) / 255.0
    server.add_image(
        "/gt",
        onp.array(mx.repeat(img_gt[..., None], repeats=3, axis=-1), copy=False),
        4.0,
        4.0,
        format="png", # NOTE: `jpeg` gives strangely stretched image
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(4.0, 4.0, 0.0),
    )

    
    img_pred = mx.random.randint(255, 256, img_gt.shape, dtype=mx.uint8)
    img_pred = img_pred.astype(mx.float32) / 255.0
    # pred = mx.repeat(pred, repeats=3, axis=-1)

    # NOTE: embedding func test
    N_INPUT_DIMS = 2
    embed, out_dim = embedding.get_embedder(10, n_input_dims=N_INPUT_DIMS)
    input = mx.zeros(N_INPUT_DIMS)
    output = embed(input)
    print(f"[DEBUG] {out_dim=}")
    print(f"[DEBUG] {output.shape=}")

    # NOTE: NeRF
    model = NeRF(
        channel_input=out_dim, # NOTE: embedded pixel position 
        channel_input_views=0, 
        channel_output=1, 
        is_use_view_directions=False, 
    )
    mx.eval(model.parameters())

    def mlx_mse(model, x, y):
        
        x_flat = mx.reshape(x, [-1, x.shape[-1]])
        x_embedded = embed(x_flat)

        # NOTE: raw pixel position may cause optimization failure, which produces NaN
        # NOTE: hence we divide original pixel positions by its shape per axis
        if 2 == N_INPUT_DIMS:
            x_embedded[..., 0] /= img_gt.shape[0]
            x_embedded[..., 1] /= img_gt.shape[1]
        
        
        return mx.mean((model.forward(x_embedded) - y) ** 2)
    loss_and_grad_fn = nn.value_and_grad(model, mlx_mse)



    
    
    # optimizer = optim.SGD(learning_rate=0.001)
    learning_rate = 0.01# 5e-4
    optimizer = optim.Adam(learning_rate=learning_rate, betas=(0.9, 0.999))

    idx_iter = 0    

    decay_rate = 0.1
    decay_steps = (lrate_decay := 250) * 1000
    

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

        loss_mse = 0.0
        n_batch_iterate=0
        for X, y in batch_iterate(batch_size:=32*1024, img_pred, img_gt): # FIXME: problem in batching
            
            loss, grads = loss_and_grad_fn(model, X, y)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            loss_mse += loss
            n_batch_iterate += 1

            img_pred[X[..., 0], X[..., 1]] = model.forward(embed(mx.reshape(X, [-1, X.shape[-1]])))[0]


        print(f"[DEBUG] #iter={idx_iter} ... \t loss = {loss_mse / n_batch_iterate}")
        
        #new_lrate = learning_rate * (decay_rate ** (idx_iter+1 / decay_steps))
        #optimizer.learning_rate = new_lrate

        if idx_iter == 100:
            exit()


        idx_iter += 1
        time.sleep(0.1)
