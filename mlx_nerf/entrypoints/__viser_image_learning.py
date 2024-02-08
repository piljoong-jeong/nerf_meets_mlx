import os
import pprint
import time
from pathlib import Path
from typing import List, Union
from PIL import Image

import imageio.v2 as imageio
writer = imageio.get_writer(os.path.join("", f"learning.mp4"), fps=60)
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

def batch_iterate(batch_size: int, y: mx.array):
    
    
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
    
    onp.random.default_rng(seed=42) # NOTE: for debugging batch learning purpose
    choice = mx.array(onp.random.choice(coords.shape[0], size=[coords.shape[0]], replace=False)) # NOTE: [H*W]
    for s in range(0, y.size, batch_size):
        selected = choice[s : s + batch_size]
        
        X_batch = coords[selected] 
        y_batch = y[coords[selected][:, 0], coords[selected][:, 1]]

        yield (
            X_batch, 
            y_batch
        )

def load_mx_img_gt(path_img: Union[str, Path]) -> mx.array:
    """
    
    Loads an image.
    If there is only a single color channel, repeat to ensure 3 channels to render

    Returns:
        - An `mx.array` contains image pixel information in a dimension format [B=1, C=3, H, W]
    """

    if isinstance(path_img, Path):
        path_img = str(path_img)

    img_gt = imageio.imread(path_img)
    img_gt = Image.fromarray(img_gt).resize((2, 2)) # NOTE: debugging purpose
    img_gt = onp.asarray(img_gt)
    img_gt = mx.array(img_gt)

    if len(img_gt.shape) == 3:
        # TODO: check dimension with color channels, and ensure that is located at the first axis
        img_gt = mx.transpose(img_gt, -1, 0)
    # NOTE: if the image has only single channel, expand to 3 channels
    if len(img_gt.shape) == 2:
        img_gt = mx.repeat(img_gt[None, ...], repeats=3, axis=0)
    
    assert img_gt.shape[0] == 3
    
    # NOTE: add batch dimension
    img_gt = img_gt[None, ...]
    img_gt = img_gt.astype(mx.float32) / 255.0

    print(f"{img_gt.shape=}")

    return img_gt

def get_mx_img_pred(shape: tuple):

    img_pred = mx.random.randint(255, 256, shape, dtype=mx.uint8) # TODO: change to `ones`
    img_pred = img_pred.astype(mx.float32) / 255.0

    print(f"{img_pred.shape=}")

    return img_pred

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

    img_gt = load_mx_img_gt(path_assets / "images/albert.jpg")
    
    
    server.add_image(
        "/gt",
        onp.array(img_gt[0], copy=False),
        4.0,
        4.0,
        format="png", # NOTE: `jpeg` gives strangely stretched image
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(4.0, 4.0, 0.0),
    )

    img_pred = get_mx_img_pred(img_gt.shape)

    exit()
    
    
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
        # n_layers=5, # FIXME: dimension mismatching occurred
    )
    mx.eval(model.parameters())

    def mlx_mse(model, x, y):
        
        x_flat = mx.reshape(x, [-1, x.shape[-1]])
        x_embedded = embed(x_flat)
        
        mse = mx.mean((model.forward(x_embedded) - y) ** 2)

        return mse
        
    loss_and_grad_fn = nn.value_and_grad(model, mlx_mse)



    
    
    # NOTE: from https://github.com/NVlabs/tiny-cuda-nn/blob/master/data/config.json
    learning_rate = 1*1e-5
    optimizer = optim.Adam(learning_rate=learning_rate, betas=(0.9, 0.99))

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

        img_gt_visualized = onp.asarray(Image.fromarray(onp.array(img_gt * 255.0, copy=False).astype(onp.uint8)).resize((400, 400)))
        pixels_np = (onp.array(img_pred, dtype=onp.float32, copy=False) * 255.0).astype(onp.uint8) # [H, W]
        writer.append_data(
            # pixels_np.transpose()
            onp.hstack([
                img_gt_visualized, 
                onp.asarray(Image.fromarray(pixels_np).resize((400, 400))),
            ])
            # onp.asarray(Image.fromarray(pixels_np).resize((400, 400)))
        )

        # TODO: check `grads.shape` per `batch_size`, and come up with a way of aggregating them

        for X, y in batch_iterate(batch_size:=2, img_gt): # FIXME: learning fails when batch_size>1
            
            # FIXME: they should be evaluated once all pixels have been inferred   
            # TODO: maybe batch iterate inside of this function?
            loss, grads = loss_and_grad_fn(model, X, y)

            for k, v in grads.items():
                print(f"{k}")

                if isinstance(v, list):
                    for d in v:
                        for k2, v2 in d.items():
                            print(f"{k2} {v2.shape}")
            exit()

            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
    
            loss_mse += loss
            n_batch_iterate += 1

            # for k, v in grads.items():
            #     print(f"{k=} \t")
            # exit()

            #print(f"{batch_size=} {X.shape=}")

            X_flat = mx.reshape(X, [-1, X.shape[-1]])
            #print(f"{batch_size=} {X_flat.shape=}") # NOTE: OK
            
            X_embedded = embed(X_flat)
            #print(f"{batch_size=} {X_embedded.shape=}") # NOTE: OK
            # exit()
            values = model.forward(X_embedded)
            #print(f"{values.shape=}") # TODO: unflat `X_flat`
            #print(f"{img_pred[X[..., 0], X[..., 1]]=}")
            # exit()
            # print(f"{values=}")
            
            # print(f"{img_pred[X[..., 0], X[..., 1]]=}")

            # values = mx.reshape(values, )
            
            # FIXME: `squeeze` worked as there is only a single channel, adding color channel in images is encouraged
            img_pred[X[..., 0], X[..., 1]] = values.squeeze(axis=-1) # TODO: concatenate `values`, and assign at once?
            # img_pred[X[..., 0], X[..., 1], None] = values

            # break

        


        print(f"[DEBUG] #iter={idx_iter} ... \t loss = {loss_mse / n_batch_iterate}")
        
        # new_lrate = learning_rate * (decay_rate ** (idx_iter+1 / decay_steps))
        # optimizer.learning_rate = new_lrate

        

        if idx_iter == 60:
            writer.close()
            exit()


        idx_iter += 1
        #time.sleep(0.1)
