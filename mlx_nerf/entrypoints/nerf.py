import json
import pprint
import os
import time
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple, Union
from PIL import Image

import imageio.v2
import numpy as onp
import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import viser
import viser.extras
from tqdm import trange

from this_project import get_project_root, PJ_PINK
from mlx_nerf import config_parser
from mlx_nerf.dataset.dataloader import load_blender_data, validate_dataset
from mlx_nerf.models.NeRF import create_NeRF
from mlx_nerf.rendering import ray, render


def main(
    path_dataset: Path = Path.home() / "Downloads" / "NeRF",
    max_iter: int = 10000,
    
):
    
    parser = config_parser.config_parser()
    args = parser.parse_args(args=[])
    args.config = "configs/ficus.txt"
    path_config = path_dataset / args.config
    
    configs = config_parser.load_config(None, path_config)
    # pprint.pprint(configs)
    args.expname = configs['expname']
    args.basedir = configs['basedir']
    args.datadir = configs['datadir']
    args.dataset_type = configs['dataset_type']
    args.no_batching = configs['no_batching']
    args.use_viewdirs = configs['use_viewdirs']
    args.white_bkgd = configs['white_bkgd']
    args.lrate_decay = int(configs['lrate_decay'])
    args.n_depth_samples = int(configs['N_samples'])
    args.N_importance = int(configs['N_importance'])
    args.N_rand = int(configs['N_rand'])
    args.precrop_iters = int(configs['precrop_iters'])
    args.precrop_frac = float(configs['precrop_frac'])
    args.half_res = configs['half_res']

    dir_dataset = configs["datadir"]
    images, poses, render_poses, hwf, i_split = load_blender_data(path_dataset / dir_dataset)
    # validate_dataset(path_dataset / dir_dataset)

    render_kwargs_train, render_kwargs_test, idx_iter, optimizer = create_NeRF(args)


    def mlx_mse(model, batch_rays, y_gt):
        """
        FIXME: 
            - ray generation
            - ray depth sampling
            - generate embedded inputs from sampled rays

        """

        rgb, disp, acc, extras = render.render(
            H, W, K, 
            args.chunk, 
            batch_rays, 
            # retraw=True, 
            **render_kwargs_train
        )

        # NOTE: fine first
        mse_fine = mx.mean((rgb - y_gt) ** 2)
        result = mse_fine
        if "rgb_coarse" in extras:
            rgb_coarse = extras["rgb_coarse"]
            mse_coarse = mx.mean((rgb_coarse - y_gt) ** 2)

            result = result + mse_coarse
        return result

    state_coarse = [render_kwargs_train["network_coarse"].state, optimizer.state]
    @partial(mx.compile, inputs=state_coarse, outputs=state_coarse)
    def step_coarse(X, y):
        model = render_kwargs_train["network_coarse"]
        loss_and_grad_fn = nn.value_and_grad(model, mlx_mse)
        loss, grads = loss_and_grad_fn(model, X, y)
        optimizer.update(model, grads)
        return loss
    

    state_fine = [render_kwargs_train["network_fine"].state, optimizer.state]
    @partial(mx.compile, inputs=state_fine, outputs=state_fine)
    def step_fine(X, y):
        model = render_kwargs_train["network_fine"]
        loss_and_grad_fn = nn.value_and_grad(model, mlx_mse)
        loss, grads = loss_and_grad_fn(model, X, y)
        optimizer.update(model, grads)
        return loss

    # NOTE: ---------------- from `train(args)` --------------------    
    i_train, i_val, i_test = i_split

    # NOTE: arbitrarily set bounds for synthetic data
    near= 2.0
    far = 6.0
    bds_dict = {
        "near": near,
        "far": far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # NOTE: blender image contains alpha, thus fill white
    if args.white_bkgd:
        images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
    else:
        images = images[..., :3]

    # NOTE: cast instrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    K = onp.array([
        [focal, 0, 0.5 * W], 
        [0, focal, 0.5 * H], 
        [0, 0, 1]
    ])

    
    if args.render_test:
        render_poses = onp.array(poses[i_test]) # NOTE: e.g., for PSNR etc
    render_poses = mx.array(render_poses)

    # NOTE: create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, "args.txt")
    with open(f, "w") as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write(f"{arg} = {attr}\n")
    if args.config is not None:
        f = os.path.join(basedir, expname, "config.txt")
        with open(f, "w") as file:
            file.write(open(path_config, "r").read())

    N_iters = 200000
    list_losses = []
    list_iters = []

    for i in trange(1, N_iters+1):
        # NOTE: randomize rays
        img_i = onp.random.choice(i_train)
        target = images[img_i]
        target = mx.array(target)
        pose = poses[img_i, :3, :4]
        N_rand = args.N_rand
        if not None is N_rand:
            rays_o, rays_d = ray.get_rays(H, W, K, mx.array(pose))

            rays_o = mx.array(rays_o) # [H, W, 3]
            rays_d = mx.array(rays_d) # [H, W, 3]

            
            """
            TODO: implement shuffling
            """
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

            choice = mx.array(onp.random.choice(coords.shape[0], size=[N_rand], replace=False)) # NOTE: [H*W]
            selected_coords = coords[choice]

            rays_o = rays_o[selected_coords[:, 0], selected_coords[:, 1]]
            rays_d = rays_d[selected_coords[:, 0], selected_coords[:, 1]]

            batch_rays = mx.stack([rays_o, rays_d], axis=0)
            target_selected = target[selected_coords[:, 0], selected_coords[:, 1]]
        else:
            raise NotImplementedError

        loss = step_coarse(batch_rays, target_selected)
        mx.eval(state_coarse)

        if False and render_kwargs_train["network_fine"]:
            loss_fine = step_fine(batch_rays, target_selected)
            mx.eval(state_fine)

        # print(f"[DEBUG] iter={i:06d} \t | loss={loss.item()=:0.6f}")

        list_iters.append(i)
        list_losses.append(loss.item())

    rgb, _, _, _ = render.render(
        H, W, K, 
        c2w=(testpose := mx.array(poses[len(poses)//2]))[:3, :4], 
        **render_kwargs_test
    )

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("Loss validation")
    ax1.set_ylim(0, 1.0)
    ax1.plot(list_iters, list_losses)
    ax2 = fig.add_subplot(1, 2, 2)
    to8b = lambda x: onp.array((mx.clip(x, 0.0, 1.0) * 255.0), copy=False).astype(onp.uint8)
    ax2.imshow(to8b(rgb))
    fig.savefig(f"results/iter={i}.png")

    print(f"[DEBUG] saving video...")
    writer = imageio.v2.get_writer(os.path.join("results", f"iter={i}.mp4"), fps=60)
    for i in trange(render_poses.shape[0]):
        render_pose = render_poses[i]
        rgb, _, _, _ = render.render(
            H, W, K, 
            c2w=render_pose[:3, :4], 
            **render_kwargs_test
        )
        writer.append_data(
            onp.hstack([
                to8b(rgb), 
            ])
        )
    writer.close()
    
    