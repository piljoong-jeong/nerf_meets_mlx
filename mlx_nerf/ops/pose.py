import cv2
import numpy as np
import torch


# NOTE: synthetic pose generation from spherical coordinate
def pose_spherical(theta, phi, radius):
    """ 
    ### pose_spherical

    Returns camera-to-world matrix
    
    """
    
    # NOTE: WYSIWIG; row-major assignment
    def _trans_radius(r):
        return torch.Tensor([ 
            [1, 0, 0, 0], 
            [0, 1, 0, 0], 
            [0, 0, 1, r], # NOTE: lookat `-r` direction; from `r` to origin
            [0, 0, 0, 1]
        ]).float()

    def _rotate_phi(phi): # NOTE: pitch; right-thumb rule 👍
        return torch.Tensor([
            [1, 0, 0, 0], # NOTE: from X-axis, 
            [0, np.cos(phi), -np.sin(phi), 0], 
            [0, np.sin(phi), np.cos(phi), 0], 
            [0, 0, 0, 1]
        ]).float()

    def _rotate_theta(theta): # NOTE: Y-axis? yaw
        # NOTE: sign of `np.sin` inverted!; means inverted rotation direction (cw~ccw)
        # NOTE: it's up to developer's choice; no need to invert it, but it's convenient
        return torch.Tensor([
            [np.cos(theta), 0, -np.sin(theta), 0], 
            [0, 1, 0, 0], # NOTE: right thumb rule at Y-axis
            [np.sin(theta), 0, np.cos(theta), 0], 
            [0, 0, 0, 1]
        ]).float()


    # NOTE: generate pose
    render_pose = _trans_radius(radius)
    render_pose = _rotate_phi(phi / 180. * np.pi) @ render_pose
    render_pose = _rotate_theta(theta / 180. * np.pi) @ render_pose

    def _cam_to_world(pose):
        # NOTE: should we wrap this with `np.array`?
        return torch.Tensor(np.array([
            [-1, 0, 0, 0], # NOTE: invert X-axis
            [0, 0, 1, 0], # NOTE: switch Y<->Z
            [0, 1, 0, 0], # NOTE: switch Z<->Y
            [0, 0, 0, 1]
        ])) @ pose

    # NOTE: c2w: inverse of extrinsic matrix
    return _cam_to_world(render_pose)

# TODO: Rodrigues rotation? maybe not?
