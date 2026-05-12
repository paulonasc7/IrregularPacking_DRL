import os
import cv2
import numpy as np
import torch


class Logger(object):
    def __init__(self, continue_logging, logging_directory):
        self.base_directory = logging_directory
        self.transitions_directory = os.path.join(self.base_directory, 'transitions')
        self.color_heightmaps_directory = os.path.join(self.base_directory, 'data', 'color-heightmaps')
        self.depth_heightmaps_directory = os.path.join(self.base_directory, 'data', 'depth-heightmaps')
        self.images_directory = os.path.join(self.base_directory, 'data', 'images')
        self.models_directory = os.path.join(self.base_directory, 'models')
        self.visualizations_directory = os.path.join(self.base_directory, 'visualizations')

        for path in [
            self.base_directory,
            self.transitions_directory,
            self.color_heightmaps_directory,
            self.depth_heightmaps_directory,
            self.images_directory,
            self.models_directory,
            self.visualizations_directory,
        ]:
            os.makedirs(path, exist_ok=True)

    def save_camera_info(self, cam_intrinsics, cam_pose, cam_depth_scale):
        np.save(os.path.join(self.base_directory, 'camera_intrinsics.npy'), cam_intrinsics)
        np.save(os.path.join(self.base_directory, 'camera_pose.npy'), cam_pose)
        np.save(os.path.join(self.base_directory, 'camera_depth_scale.npy'), np.array([cam_depth_scale]))

    def save_heightmap_info(self, workspace_limits, heightmap_resolution):
        np.save(os.path.join(self.base_directory, 'workspace_limits.npy'), workspace_limits)
        np.save(os.path.join(self.base_directory, 'heightmap_resolution.npy'), np.array([heightmap_resolution]))

    def write_to_log(self, log_name, log_data):
        log_path = os.path.join(self.transitions_directory, f'{log_name}.log.txt')
        if len(log_data) == 0:
            return
        arr = np.asarray(log_data)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        np.savetxt(log_path, arr, fmt='%.8f')

    def save_images(self, iteration, color_img, depth_img, suffix):
        color_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.images_directory, f'{iteration:06d}.{suffix}.color.png'), color_bgr)
        depth_uint16 = np.clip(depth_img * 100000, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        cv2.imwrite(os.path.join(self.images_directory, f'{iteration:06d}.{suffix}.depth.png'), depth_uint16)

    def save_heightmaps(self, iteration, color_heightmap, depth_heightmap, suffix):
        color_bgr = cv2.cvtColor(color_heightmap, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.color_heightmaps_directory, f'{iteration:06d}.{suffix}.color.png'), color_bgr)
        depth_uint16 = np.clip(depth_heightmap * 100000, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        cv2.imwrite(os.path.join(self.depth_heightmaps_directory, f'{iteration:06d}.{suffix}.depth.png'), depth_uint16)

    def save_visualizations(self, iteration, vis_img, vis_type):
        cv2.imwrite(os.path.join(self.visualizations_directory, f'{iteration:06d}.{vis_type}.png'), vis_img)

    def save_backup_model(self, model, method):
        torch.save(model.state_dict(), os.path.join(self.models_directory, f'backup.{method}.pth'))

    def save_model(self, iteration, model, method):
        torch.save(model.state_dict(), os.path.join(self.models_directory, f'{iteration:06d}.{method}.pth'))
