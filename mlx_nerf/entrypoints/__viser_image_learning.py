import os
from functools import partial, reduce
from pathlib import Path
from typing import List, Optional, Tuple, Union
from PIL import Image


import imageio.v2
import imageio.v3
import numpy as onp
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import viser
import viser.extras
import viser.transforms as tf
from imageio.core.format import Format
from tqdm.auto import tqdm

from this_project import get_project_root, PJ_PINK
from mlx_nerf.models import embedding
from mlx_nerf.models.NeRF import NeRF
from mlx_nerf.ops.metric import MSE

class GUIItem:
    cbox_learning: Optional[viser.GuiButtonHandle] = None
    is_learning = False
    slider_iter: Optional[viser.GuiInputHandle] = None
    idx_iter = 0

    writer: Optional[Format.Writer] = None

    img_gt = None
    img_pred = None

    pass

gui_items = GUIItem()

def toggle_learning() -> None:
    
    gui_items.is_learning = not gui_items.is_learning

    if gui_items.is_learning:
        # NOTE: toggled on
        gui_items.writer = imageio.v2.get_writer(os.path.join("results", f"learning.mp4"), fps=60)
        gui_items.img_pred = get_mx_img_pred(gui_items.img_gt.shape)
    else:
        # NOTE: toggled off - reset

        gui_items.slider_iter.value = 0
        if not None is gui_items.writer:
            gui_items.writer.close()
            gui_items.writer = None

    return
    

def init_gui_viser(**config) -> viser.ViserServer:

    server = viser.ViserServer()

    server.reset_scene()
    server.configure_theme(
        control_layout="fixed",
        control_width="medium",
        dark_mode=True,
        show_logo=False,
        show_share_button=False,
        brand_color=PJ_PINK
    )

    gui_items.cbox_learning = server.add_gui_checkbox("Learning?", False)
    gui_items.cbox_learning.on_update(lambda _: toggle_learning())
    gui_items.slider_iter = server.add_gui_slider(
        "# Iterations",
        min=0,
        max=(max_frames := config.get("max_frames", 10000)),
        step=1,
        initial_value=0,
        disabled=True,
    )

    return server

def batch_iterate(batch_size: int, y: mx.array):
    
    # NOTE: assuming [B, C, H, W]
    H = y.shape[-2]
    W = y.shape[-1]
    
    coords = onp.meshgrid(
        
        onp.arange(0, H), 
        onp.arange(0, W), 
        indexing="ij"
    ) # NOTE: `list`
    
    # TODO: convert all `np.ndarray`s into `mx.array`
    coords[0] = mx.array(coords[0])
    coords[1] = mx.array(coords[1])
    

    # TODO: stack meshgrids
    coords = mx.stack(coords, axis=-1)
    
    # TODO: reshape, now [H, W] has been flatten
    coords = mx.reshape(coords, [-1, 2])    
    
    # onp.random.default_rng(seed=42) # NOTE: for debugging batch learning purpose
    choice = mx.array(onp.random.choice(coords.shape[0], size=[coords.shape[0]], replace=False)) # NOTE: [H*W]
    for s in range(0, H*W, batch_size):
        
        selected = choice[s : s + batch_size]
        
        X_batch = coords[selected] # NOTE: [B, 2]
        
        y_ = y[0].moveaxis(0, -1)
        y_batch = y_[coords[selected][:, 0], coords[selected][:, 1]]

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

    img_gt = imageio.v3.imread(path_img)
    img_gt = Image.fromarray(img_gt).resize((400, 400)) # NOTE: debugging purpose
    img_gt = onp.asarray(img_gt)
    img_gt = mx.array(img_gt)

    if len(img_gt.shape) == 3:
        # TODO: check dimension with color channels, and ensure that is located at the first axis
        img_gt = mx.moveaxis(img_gt, -1, 0)
    # NOTE: if the image has only single channel, expand to 3 channels
    if len(img_gt.shape) == 2:
        img_gt = mx.repeat(img_gt[None, ...], repeats=3, axis=0)
    # NOTE: remove alpha channel
    if img_gt.shape[0] == 4:
        img_gt = img_gt[:3, ...]

    assert img_gt.shape[0] == 3
    
    # NOTE: add batch dimension
    img_gt = img_gt[None, ...]
    img_gt = img_gt.astype(mx.float32) / 255.0

    assert len(img_gt.shape) == 4

    return img_gt

def get_mx_img_pred(shape: tuple):

    img_pred = mx.random.randint(255, 256, shape, dtype=mx.uint8) # TODO: change to `ones`
    img_pred = img_pred.astype(mx.float32) / 255.0

    assert len(img_pred.shape) == 4

    return img_pred

def mx_to_img(
    a: mx.array, 
    size: Optional[Tuple], 
):
    
    result = onp.array(
        (mx.clip(a, 0.0, 1.0) * 255.0)[0].moveaxis(0, -1), copy=False).astype(onp.uint8)
    if not None is size:
        result = Image.fromarray(result).resize(size)
        result = onp.asarray(result)

    return result

def main(
    path_assets: Path = get_project_root() / "assets",
    batch_downsample_factor: int = 64,
    max_frames: int = 600,
):

    server = init_gui_viser(max_frames=max_frames)

    gui_items.img_gt = load_mx_img_gt(path_assets / "image_learning/albert.jpg")
    gui_items.img_pred = get_mx_img_pred(gui_items.img_gt.shape)
    
    N_INPUT_DIMS = 2
    embed, out_dim = embedding.get_embedder(10, n_input_dims=N_INPUT_DIMS)

    # NOTE: NeRF
    model = NeRF(
        channel_input=out_dim, # NOTE: embedded pixel position 
        channel_input_views=0, 
        channel_output=3, 
        is_use_view_directions=False, 
    )
    mx.eval(model.parameters())

    def mlx_mse(model, x, y):
        
        x_flat = mx.reshape(x, [-1, x.shape[-1]])
        x_embedded = embed(x_flat)
        
        mse = mx.mean((model.forward(x_embedded) - y) ** 2)

        result = mse
        return result
        
    loss_and_grad_fn = nn.value_and_grad(model, mlx_mse)
    
    # NOTE: from https://github.com/NVlabs/tiny-cuda-nn/blob/master/data/config.json
    optimizer = optim.Adam(
        learning_rate=(learning_rate := 1*1e-3), 
        betas=(0.9, 0.99)
    )
    
    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(X, y):
        loss_and_grad_fn = nn.value_and_grad(model, mlx_mse)
        loss, grads = loss_and_grad_fn(model, X, y)
        optimizer.update(model, grads)
        return loss
    
    while True:


        resize = (400, 400)
        img_gt_vis = mx_to_img(gui_items.img_gt, resize)
        img_pred_vis = mx_to_img(gui_items.img_pred, resize)

        server.add_image(
            "/gt",
            img_gt_vis, 
            4.0,
            4.0,
            format="png", # NOTE: `jpeg` gives strangely stretched image
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(4.0, 4.0, 0.0),
        )

        server.add_image(
            "/pred",
            img_pred_vis,
            4.0,
            4.0,
            format="png", # NOTE: `jpeg` gives strangely stretched image
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(4.0, 0.0, 0.0),
        )

        if not gui_items.is_learning:
            continue


        
        
        for X, y in batch_iterate(batch_size:=reduce(lambda a, b: a*b, resize)//batch_downsample_factor, gui_items.img_gt):
            
            # NOTE: without JIT
            #loss, grads = loss_and_grad_fn(model, X, y)
            #optimizer.update(model, grads)
            #mx.eval(model.parameters(), optimizer.state)
            
            # NOTE: with JIT
            loss = step(X, y)
            mx.eval(state)

            X_flat = mx.reshape(X, [-1, X.shape[-1]])
            X_embedded = embed(X_flat)
            values = model.forward(X_embedded)
            
            gui_items.img_pred = gui_items.img_pred[0].moveaxis(0, -1)
            gui_items.img_pred[X[..., 0], X[..., 1]] = values
            gui_items.img_pred = gui_items.img_pred.moveaxis(-1, 0)[None, ...]

    

        if gui_items.slider_iter.value == max_frames:

            # if not None is gui_items.writer:
            #     gui_items.writer.close()
            #     gui_items.writer = None

            print(f"[DEBUG] reached to maximum frame, learning stopped ...")
            toggle_learning()
            


        if gui_items.slider_iter.value >= max_frames:
            continue

        if gui_items.is_learning:
            gui_items.slider_iter.value += 1

            assert not None is gui_items.writer
            gui_items.writer.append_data(
                onp.hstack([
                    img_gt_vis, 
                    img_pred_vis,
                ])
            )
