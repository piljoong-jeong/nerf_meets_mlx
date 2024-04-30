"""

"""

from argparse import ArgumentParser
from enum import Enum, auto

import numpy as onp
import mlx.core as mx
from tqdm import trange

from mlx_nerf.dataset.dataloader import DatasetType, load_blender_data
from mlx_nerf.integrator import Integrator
from mlx_nerf.rendering import ray

class Trainer:

    def __init__(
        self, 
        path_dataset: str, 
        args: ArgumentParser,
    ) -> None:
        
        self.path_dataset = path_dataset
        self.args = args
        
        self.dir_dataset = self.args.datadir

        self.max_iters = 2000

        return
    
    def load_dataset(
        self, 
        dataset_type: DatasetType
    ):
        
        func_dataset_loading = {
            DatasetType.BLENDER: load_blender_data, 
        }[dataset_type]

        self.images, self.poses, self.render_poses, self.hwf, self.i_split = func_dataset_loading(self.path_dataset / self.dir_dataset)

        self.i_train, self.i_val, self.i_test = self.i_split
        self.H, self.W, self.focal = self.hwf


    def select_pixels(
        self, 
        batch_size, 
        img_target, 
        H, W, focal, pose
    ):
        K = onp.array([
            [focal, 0, 0.5 * W], 
            [0, focal, 0.5 * H], 
            [0, 0, 1]
        ])
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

        choice = mx.array(onp.random.choice(coords.shape[0], size=[batch_size], replace=False)) # NOTE: [H*W]
        selected_coords = coords[choice]

        rays_o = rays_o[selected_coords[:, 0], selected_coords[:, 1]]
        rays_d = rays_d[selected_coords[:, 0], selected_coords[:, 1]]

        batch_rays = mx.stack([rays_o, rays_d], axis=0)
        target_selected = img_target[selected_coords[:, 0], selected_coords[:, 1]]

        return batch_rays, target_selected

    def train_using(
        self, 
        type_integrator: type,
    ):
        
        assert issubclass(type_integrator, Integrator), f"[ERROR] {type_integrator=} is not an {Integrator} type!"
        integrator = type_integrator((config := None))

        for i in trange(1, self.max_iters+1):
            idx_img = onp.random.choice(self.i_train)
            X, y = self.select_pixels(
                self.args.N_rand, 
                self.images[idx_img], 
                self.H, self.W, self.focal, 
                self.poses[idx_img, :3, :4])
            
            print(f"{X.shape=}")
            print(f"{y.shape=}")

            outputs = integrator.get_outputs(X, y)

            break
        
        return
