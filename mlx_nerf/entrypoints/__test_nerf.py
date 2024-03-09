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
import torch
import viser
import viser.extras
from mlx.utils import tree_flatten
from tqdm import trange

from this_project import get_project_root, PJ_PINK
from mlx_nerf import config_parser
from mlx_nerf.dataset.dataloader import load_blender_data, validate_dataset
from mlx_nerf.models.NeRF import create_NeRF
from mlx_nerf.rendering import ray, render
from mlx_nerf.rendering.render import render_rays, raw2outputs
from mlx_nerf import sampling


def main(
    path_dataset: Path = Path.home() / "Downloads" / "NeRF",
    max_iter: int = 10000,
    
):
    
    parser = config_parser.config_parser()
    args = parser.parse_args(args=[])
    args.config = "configs/lego.txt"
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

    z_vals = None
    weights = None
    def mlx_mse_coarse(model, batch_rays, y_gt):
        """
        FIXME: 
            - ray generation
            - ray depth sampling
            - generate embedded inputs from sampled rays

        """

        # rgb, disp, acc, extras = render.render(
        #     H, W, K, 
        #     args.chunk, 
        #     batch_rays, 
        #     **render_kwargs_train
        # )


        near = render_kwargs_train["near"]
        far = render_kwargs_train["far"]

        rays_o, rays_d = batch_rays
        rays_shape = rays_d.shape
        viewdirs = rays_d
        viewdirs = viewdirs / mx.linalg.norm(viewdirs, axis=-1, keepdims=True)
        viewdirs = mx.reshape(viewdirs, [-1, 3]).astype(mx.float32)

        if False: # ndc:
            rays_o, rays_d = ray.ndc_rays(
                H, W, K[0][0], 
                1.0, 
                rays_o, rays_d
            )

        near = near * mx.ones_like(rays_d[..., :1])
        far = far * mx.ones_like(rays_d[..., :1])

        # NOTE: concat all ray-related features
        rays = mx.concatenate(
            [rays_o, rays_d, near, far], # FIXME: should `near` and `far` included for every ray tensor?
            axis=-1
        )
        
        rays_linear = mx.concatenate([rays, viewdirs], axis=-1)
        results = render_rays(rays_linear, **render_kwargs_train)

        rgb = results["rgb_coarse"]
        # nonlocal z_vals
        # nonlocal weights
        # z_vals = results["z_vals"]
        # weights = results["weights"]

        # NOTE: fine first
        mse_coarse = mx.mean((rgb - y_gt) ** 2)
        
        return mse_coarse
    
    optimizer_fine = optim.Adam(learning_rate=args.lrate, betas=(0.9, 0.999))
    def mlx_mse_fine(model, batch_rays, z_vals_fine, y_gt): # FIXME: in this way computational graph won't be established

        rays_o, rays_d = batch_rays
        rays_shape = rays_d.shape
        viewdirs = rays_d
        viewdirs = viewdirs / mx.linalg.norm(viewdirs, axis=-1, keepdims=True)
        viewdirs = mx.reshape(viewdirs, [-1, 3]).astype(mx.float32)


        
        
        run_fn = model
        network_query_fn = render_kwargs_train["network_query_fn"]
        white_bkgd=False 
        raw_noise_std=0.0 

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_fine[..., :, None]
        raw = network_query_fn(pts, viewdirs, run_fn)
        rgb, disp, acc, weight, depth = raw2outputs(
            raw, 
            z_vals_fine, 
            rays_d, 
            raw_noise_std, 
            white_bkgd
        )
        ret = {}
        ret["rgb_map"] = rgb
        ret["disp_map"] = disp
        ret["acc_map"] = acc

        # NOTE: fine first
        mse_fine = mx.mean((rgb - y_gt) ** 2)
        
        return mse_fine

    state_coarse = [render_kwargs_train["network_coarse"].state, optimizer.state]
    @partial(mx.compile, inputs=state_coarse, outputs=state_coarse)
    def step_coarse(X, y):
        model = render_kwargs_train["network_coarse"]
        loss_and_grad_fn = nn.value_and_grad(model, mlx_mse_coarse)
        loss, grads = loss_and_grad_fn(model, X, y)
        optimizer.update(model, grads)
        return loss
    

    state_fine = [render_kwargs_train["network_fine"].state, optimizer_fine.state]
    @partial(mx.compile, inputs=state_fine, outputs=state_fine)
    def step_fine(batch_rays, z_vals_fine, y):
        model = render_kwargs_train["network_fine"]
        loss_and_grad_fn = nn.value_and_grad(model, mlx_mse_fine)
        loss, grads = loss_and_grad_fn(model, batch_rays, z_vals_fine, y)
        optimizer_fine.update(model, grads)
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

    N_iters = 2000
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

        if render_kwargs_train["network_fine"]:
            
            mx.disable_compile()
            
            # TODO: `batch_rays` -> sampled rays
            
            N_importance = args.N_importance            
            near = render_kwargs_train["near"]
            far = render_kwargs_train["far"]

            rays_o, rays_d = batch_rays
            rays_shape = rays_d.shape
            viewdirs = rays_d
            viewdirs = viewdirs / mx.linalg.norm(viewdirs, axis=-1, keepdims=True)
            viewdirs = mx.reshape(viewdirs, [-1, 3]).astype(mx.float32)


            near = near * mx.ones_like(rays_d[..., :1])
            far = far * mx.ones_like(rays_d[..., :1])

            # NOTE: concat all ray-related features
            rays = mx.concatenate(
                [rays_o, rays_d, near, far], # FIXME: should `near` and `far` included for every ray tensor?
                axis=-1
            )
            
            rays_linear = mx.concatenate([rays, viewdirs], axis=-1)
            results = render_rays(rays_linear, **render_kwargs_train)
            z_vals = results["z_vals"]
            weights = results["weights"]

            z_vals_torch = torch.from_numpy(onp.array(z_vals)).to("mps")
            weights_torch = torch.from_numpy(onp.array(weights)).to("mps")
            
            z_importance_samples = sampling.sample_from_inverse_cdf_torch(
                z_vals_torch, 
                weights_torch, 
                N_importance, 
            )

            z_importance_samples = z_importance_samples.detach().cpu().numpy()
            z_importance_samples = mx.array(z_importance_samples)


            z_vals_fine = mx.sort(mx.concatenate([z_vals, z_importance_samples], axis=-1), axis=-1) # [B, n_samples + n_importance_samples]
            

            
            loss = step_fine(batch_rays, z_vals_fine, target_selected)
            mx.eval(state_fine)
            mx.enable_compile()

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
    
    