"""

"""

from argparse import ArgumentParser
from enum import Enum, auto

from mlx_nerf.dataset.dataloader import DatasetType, load_blender_data
from mlx_nerf.integrator import Integrator

class Trainer:

    def __init__(
        self, 
        path_dataset: str, 
        args: ArgumentParser,
    ) -> None:
        
        self.path_dataset = path_dataset
        self.args = args
        
        self.dir_dataset = self.args.datadir

        return
    
    def load_dataset(
        self, 
        dataset_type: DatasetType
    ):
        
        func_dataset_loading = {
            DatasetType.BLENDER: load_blender_data, 
        }[dataset_type]

        self.images, self.poses, self.render_poses, self.hwf, self.i_split = func_dataset_loading(self.path_dataset / self.dir_dataset)


    def train_using(
        self, 
        type_integrator: Integrator,
    ):
        
        assert isinstance(type_integrator, Integrator), f"[ERROR] {type(type_integrator)=} is not an {Integrator} type!"

        
        
        return
