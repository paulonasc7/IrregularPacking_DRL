import cv2
import numpy as np


class CrossEntropyLoss2d(object):
    def __init__(self, weight=None):
        import torch
        import torch.nn as nn
        self._torch = torch
        self.loss = nn.NLLLoss(weight)

    def cuda(self):
        self.loss = self.loss.cuda()
        return self

    def __call__(self, inputs, targets):
        return self.loss(self._torch.log_softmax(inputs, dim=1), targets)


def get_heightmap(color_img, depth_img, cam_intrinsics, cam_pose, workspace_limits, heightmap_resolution):
    """Project RGB-D into an orthographic top-down color/depth heightmap."""
    color_img = np.asarray(color_img)
    depth_img = np.asarray(depth_img)

    im_h, im_w = depth_img.shape
    fx, fy = cam_intrinsics[0, 0], cam_intrinsics[1, 1]
    cx, cy = cam_intrinsics[0, 2], cam_intrinsics[1, 2]

    pix_x, pix_y = np.meshgrid(np.arange(im_w), np.arange(im_h))
    cam_z = depth_img.reshape(-1)
    cam_x = (pix_x.reshape(-1) - cx) * cam_z / fx
    cam_y = (pix_y.reshape(-1) - cy) * cam_z / fy

    valid = np.isfinite(cam_z) & (cam_z > 0)
    cam_pts = np.stack((cam_x[valid], cam_y[valid], cam_z[valid]), axis=1)
    rgb = color_img.reshape(-1, 3)[valid]

    if cam_pts.size == 0:
        map_h = int(np.round((workspace_limits[1][1] - workspace_limits[1][0]) / heightmap_resolution))
        map_w = int(np.round((workspace_limits[0][1] - workspace_limits[0][0]) / heightmap_resolution))
        return np.zeros((map_h, map_w, 3), dtype=np.uint8), np.zeros((map_h, map_w), dtype=np.float32)

    rot = cam_pose[:3, :3]
    trans = cam_pose[:3, 3]
    world_pts = (rot @ cam_pts.T).T + trans

    xlim = workspace_limits[0]
    ylim = workspace_limits[1]
    zlim = workspace_limits[2]

    in_bounds = (
        (world_pts[:, 0] >= xlim[0]) & (world_pts[:, 0] < xlim[1]) &
        (world_pts[:, 1] >= ylim[0]) & (world_pts[:, 1] < ylim[1]) &
        (world_pts[:, 2] >= zlim[0]) & (world_pts[:, 2] < zlim[1])
    )

    world_pts = world_pts[in_bounds]
    rgb = rgb[in_bounds]

    map_h = int(np.round((ylim[1] - ylim[0]) / heightmap_resolution))
    map_w = int(np.round((xlim[1] - xlim[0]) / heightmap_resolution))
    color_heightmap = np.zeros((map_h, map_w, 3), dtype=np.uint8)
    depth_heightmap = np.zeros((map_h, map_w), dtype=np.float32)

    if world_pts.size == 0:
        return color_heightmap, depth_heightmap

    px = np.floor((world_pts[:, 0] - xlim[0]) / heightmap_resolution).astype(np.int32)
    py = np.floor((world_pts[:, 1] - ylim[0]) / heightmap_resolution).astype(np.int32)
    pz = (world_pts[:, 2] - zlim[0]).astype(np.float32)

    inside = (px >= 0) & (px < map_w) & (py >= 0) & (py < map_h)
    px, py, pz, rgb = px[inside], py[inside], pz[inside], rgb[inside]

    if px.size == 0:
        return color_heightmap, depth_heightmap

    # Keep top-most point per pixel: sort by height then overwrite.
    order = np.argsort(pz)
    px, py, pz, rgb = px[order], py[order], pz[order], rgb[order]

    depth_heightmap[py, px] = pz
    color_heightmap[py, px] = rgb

    return color_heightmap, depth_heightmap
