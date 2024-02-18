import json
import os
import sys
from PIL import Image

import imageio.v2 as imageio
import numpy as np
import mlx.core as mx

if __name__ == "__main__":
    sys.path.append(DIR_PROJECT_ROOT := os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from mlx_nerf import config_parser
from mlx_nerf.ops import pose


# NOTE: implement Blender data loader
def load_blender_data(basedir, half_res: bool=False, testskip=1):

    splits = ["train", "val", "test"]
    metas = {}

    # NOTE: load poses
    for s in splits:
        with open(os.path.join(basedir, f"transforms_{s}.json"), "r") as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []

    counts = [0] # NOTE: start of `train` index
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []

        if s == "train" or testskip == 0: # NOTE: if `train`, use all frames
            skip = 1
        else:
            skip = testskip

        for frame in meta["frames"][::skip]:
            fname = os.path.join(basedir, frame["file_path"] + ".png")
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame["transform_matrix"]))
            

        imgs = (np.array(imgs) / 255.).astype(np.float32) # NOTE: keep all 4 channels
        poses = np.array(poses).astype(np.float32)
        all_imgs.append(imgs)
        all_poses.append(poses)

        counts.append(counts[-1] + imgs.shape[0]) # NOTE: end of each split index
 
    # NOTE: index for each split
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(len(splits))]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta["camera_angle_x"])
    focal_length = 0.5*W / np.tan(0.5*camera_angle_x) # NOTE: see handwritten note

    # NOTE: poses for inference/test
    render_poses = mx.stack(
        [
            pose.pose_spherical(theta=angle, phi=-30.0, radius=4.0) 
            for angle
            in np.linspace(-180, 180, 160+1)[:-1]
        ], axis=0
    )

    if True is half_res:
        H = H//2
        W = W//2

        # NOTE: IMPORTANT: change focal length!!!
        focal_length = focal_length/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for idx, image in enumerate(imgs):
            # imgs_half_res[idx] = cv2.resize(image, (H, W), interpolation=cv2.INTER_AREA)

            # NOTE: no perfect equivalence: see https://stackoverflow.com/questions/73836615/equivalent-to-cv2-inter-area-in-pil
            # TODO: validate
            imgs_half_res[idx] = np.asarray(Image.fromarray(image).resize((H, W), Image.Resampling.LANCZOS)) 
        imgs = imgs_half_res

    return imgs, poses, render_poses, [H, W, focal_length], i_split


def post_load_blender_data(i_split, images, is_white_bkgd):
    """
    Post blender data processing function to avoid redundancy
    """

    i_train, i_val, i_test = i_split

    # NOTE: arbitrarily set
    near = 2.0
    far = 6.0

    if is_white_bkgd:
        images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
    else:
        images = images[..., :3]

    return i_train, i_val, i_test, near, far, images



def main():
    from pathlib import Path
    dir_dataset = Path.home() / "Downloads" / "NeRF"
    print(f"{dir_dataset=}")

    path_config = dir_dataset / "configs/lego.txt"
    configs = config_parser.load_config(None, path_config)
    print(f"{configs=}")

    
    images, poses, render_poses, hwf, i_split = load_blender_data(dir_dataset / "data" / "nerf_synthetic" / "lego")
    print(f"{images.shape=}")
    print(f"{poses.shape=}")
    print(f"{poses[0]}")

    return

if __name__ == "__main__":
    main()