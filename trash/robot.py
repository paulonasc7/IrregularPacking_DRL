import glob
import math
import os
import time

import numpy as np
import pybullet as p
import pybullet_data


class Robot(object):
    def __init__(self, is_sim, obj_mesh_dir, num_obj, workspace_limits,
                 tcp_host_ip, tcp_port, rtc_host_ip, rtc_port,
                 is_testing, test_preset_cases, test_preset_file):
        self.is_sim = is_sim
        self.obj_mesh_dir = obj_mesh_dir
        self.num_obj = int(num_obj or 10)
        self.workspace_limits = np.asarray(workspace_limits, dtype=np.float32)
        self.is_testing = is_testing
        self._rng = np.random.default_rng(1234)
        self.object_ids = []

        if self.is_sim:
            if p.isConnected():
                p.disconnect()
            self.client_id = p.connect(p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.8)
            p.setTimeStep(1.0 / 240.0)

            self._setup_camera()
            self.restart_sim()
            self.add_objects()
        else:
            raise NotImplementedError('Real robot control is not available in this repository snapshot. Use --is_sim.')

    def _setup_camera(self):
        self.cam_width = 224
        self.cam_height = 224
        self.cam_depth_scale = 1.0

        xmid = float((self.workspace_limits[0, 0] + self.workspace_limits[0, 1]) / 2.0)
        ymid = float((self.workspace_limits[1, 0] + self.workspace_limits[1, 1]) / 2.0)
        zmax = float(self.workspace_limits[2, 1])

        self._cam_eye = [xmid, ymid, zmax + 0.55]
        self._cam_target = [xmid, ymid, zmax - 0.05]
        self._cam_up = [1.0, 0.0, 0.0]

        self._cam_fov = 60.0
        self._cam_aspect = float(self.cam_width) / float(self.cam_height)
        self._cam_near = 0.01
        self._cam_far = 2.0

        self._view_matrix = p.computeViewMatrix(self._cam_eye, self._cam_target, self._cam_up)
        self._proj_matrix = p.computeProjectionMatrixFOV(
            fov=self._cam_fov,
            aspect=self._cam_aspect,
            nearVal=self._cam_near,
            farVal=self._cam_far,
        )

        fy = (self.cam_height / 2.0) / math.tan(math.radians(self._cam_fov) / 2.0)
        fx = fy
        cx = (self.cam_width - 1) / 2.0
        cy = (self.cam_height - 1) / 2.0
        self.cam_intrinsics = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

        # Camera pose as world-from-camera transform for look-at view.
        eye = np.array(self._cam_eye, dtype=np.float32)
        target = np.array(self._cam_target, dtype=np.float32)
        up = np.array(self._cam_up, dtype=np.float32)
        forward = target - eye
        forward /= (np.linalg.norm(forward) + 1e-8)
        right = np.cross(forward, up)
        right /= (np.linalg.norm(right) + 1e-8)
        true_up = np.cross(right, forward)
        rot = np.stack([right, true_up, forward], axis=1)
        self.cam_pose = np.eye(4, dtype=np.float32)
        self.cam_pose[:3, :3] = rot
        self.cam_pose[:3, 3] = eye

    def _workspace_center(self):
        return np.array(
            [
                (self.workspace_limits[0, 0] + self.workspace_limits[0, 1]) / 2.0,
                (self.workspace_limits[1, 0] + self.workspace_limits[1, 1]) / 2.0,
                self.workspace_limits[2, 0],
            ],
            dtype=np.float32,
        )

    def _create_workspace(self):
        p.loadURDF('plane.urdf')

        x0, x1 = float(self.workspace_limits[0, 0]), float(self.workspace_limits[0, 1])
        y0, y1 = float(self.workspace_limits[1, 0]), float(self.workspace_limits[1, 1])
        z0 = float(self.workspace_limits[2, 0])

        wall_h = 0.08
        wall_t = 0.006

        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        lx = (x1 - x0)
        ly = (y1 - y0)

        base_half = [lx / 2.0, ly / 2.0, 0.005]
        base_pos = [cx, cy, z0 - 0.005]
        base_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=base_half)
        base_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=base_half, rgbaColor=[0.8, 0.8, 0.8, 1])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=base_col, baseVisualShapeIndex=base_vis, basePosition=base_pos)

        walls = [
            ([lx / 2.0, wall_t / 2.0, wall_h / 2.0], [cx, y0 - wall_t / 2.0, z0 + wall_h / 2.0]),
            ([lx / 2.0, wall_t / 2.0, wall_h / 2.0], [cx, y1 + wall_t / 2.0, z0 + wall_h / 2.0]),
            ([wall_t / 2.0, ly / 2.0, wall_h / 2.0], [x0 - wall_t / 2.0, cy, z0 + wall_h / 2.0]),
            ([wall_t / 2.0, ly / 2.0, wall_h / 2.0], [x1 + wall_t / 2.0, cy, z0 + wall_h / 2.0]),
        ]
        for half, pos in walls:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0.9, 0.9, 0.9, 1])
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=pos)

    def _candidate_urdfs(self):
        if not self.obj_mesh_dir or not os.path.isdir(self.obj_mesh_dir):
            return []
        return sorted(glob.glob(os.path.join(self.obj_mesh_dir, '**', '*.urdf'), recursive=True))

    def restart_sim(self):
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1.0 / 240.0)
        self.object_ids = []
        self._create_workspace()

    def add_objects(self):
        urdfs = self._candidate_urdfs()

        x0, x1 = float(self.workspace_limits[0, 0]), float(self.workspace_limits[0, 1])
        y0, y1 = float(self.workspace_limits[1, 0]), float(self.workspace_limits[1, 1])
        z0 = float(self.workspace_limits[2, 0])

        for i in range(self.num_obj):
            px = self._rng.uniform(x0 + 0.02, x1 - 0.02)
            py = self._rng.uniform(y0 + 0.02, y1 - 0.02)
            pz = z0 + 0.03 + 0.01 * (i % 4)
            yaw = float(self._rng.uniform(-math.pi, math.pi))
            quat = p.getQuaternionFromEuler([0, 0, yaw])

            body_id = None
            if urdfs:
                urdf_path = urdfs[i % len(urdfs)]
                try:
                    body_id = p.loadURDF(urdf_path, [px, py, pz], quat, useFixedBase=False)
                except Exception:
                    body_id = None

            if body_id is None:
                if i % 2 == 0:
                    hx = float(self._rng.uniform(0.01, 0.02))
                    hy = float(self._rng.uniform(0.01, 0.02))
                    hz = float(self._rng.uniform(0.01, 0.03))
                    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz])
                    vis = p.createVisualShape(
                        p.GEOM_BOX,
                        halfExtents=[hx, hy, hz],
                        rgbaColor=[float(self._rng.uniform(0.2, 1.0)), float(self._rng.uniform(0.2, 1.0)), float(self._rng.uniform(0.2, 1.0)), 1],
                    )
                else:
                    radius = float(self._rng.uniform(0.008, 0.018))
                    height = float(self._rng.uniform(0.02, 0.05))
                    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
                    vis = p.createVisualShape(
                        p.GEOM_CYLINDER,
                        radius=radius,
                        length=height,
                        rgbaColor=[float(self._rng.uniform(0.2, 1.0)), float(self._rng.uniform(0.2, 1.0)), float(self._rng.uniform(0.2, 1.0)), 1],
                    )
                body_id = p.createMultiBody(
                    baseMass=0.05,
                    baseCollisionShapeIndex=col,
                    baseVisualShapeIndex=vis,
                    basePosition=[px, py, pz],
                    baseOrientation=quat,
                )

            self.object_ids.append(body_id)

        for _ in range(360):
            p.stepSimulation()

    def check_sim(self):
        for _ in range(5):
            p.stepSimulation()
        self.object_ids = [oid for oid in self.object_ids if oid >= 0 and p.getBodyInfo(oid) is not None]

    def get_camera_data(self):
        img = p.getCameraImage(
            width=self.cam_width,
            height=self.cam_height,
            viewMatrix=self._view_matrix,
            projectionMatrix=self._proj_matrix,
            renderer=p.ER_TINY_RENDERER,
        )

        rgba = np.reshape(img[2], (self.cam_height, self.cam_width, 4))
        color = rgba[:, :, :3].astype(np.uint8)

        z_buffer = np.reshape(img[3], (self.cam_height, self.cam_width)).astype(np.float32)
        near, far = self._cam_near, self._cam_far
        depth = (2.0 * near * far) / (far + near - (2.0 * z_buffer - 1.0) * (far - near))
        return color, depth.astype(np.float32)

    def _nearest_object(self, x, y, max_dist=0.06):
        best_id = None
        best_d = float('inf')
        for oid in list(self.object_ids):
            try:
                pos, _ = p.getBasePositionAndOrientation(oid)
            except Exception:
                continue
            d = math.hypot(pos[0] - x, pos[1] - y)
            if d < best_d:
                best_d = d
                best_id = oid
        if best_id is None or best_d > max_dist:
            return None
        return best_id

    def push(self, primitive_position, best_rotation_angle, workspace_limits):
        x, y, _ = primitive_position
        oid = self._nearest_object(x, y, max_dist=0.07)
        if oid is None:
            return False

        pos, quat = p.getBasePositionAndOrientation(oid)
        dx = 0.04 * math.cos(best_rotation_angle)
        dy = 0.04 * math.sin(best_rotation_angle)

        nx = np.clip(pos[0] + dx, workspace_limits[0][0] + 0.01, workspace_limits[0][1] - 0.01)
        ny = np.clip(pos[1] + dy, workspace_limits[1][0] + 0.01, workspace_limits[1][1] - 0.01)
        nz = max(pos[2], workspace_limits[2][0] + 0.01)

        p.resetBasePositionAndOrientation(oid, [float(nx), float(ny), float(nz)], quat)
        for _ in range(120):
            p.stepSimulation()

        new_pos, _ = p.getBasePositionAndOrientation(oid)
        moved = math.hypot(new_pos[0] - pos[0], new_pos[1] - pos[1]) > 0.015
        return bool(moved)

    def grasp(self, primitive_position, best_rotation_angle, workspace_limits):
        x, y, _ = primitive_position
        oid = self._nearest_object(x, y, max_dist=0.05)
        if oid is None:
            return False

        try:
            pos, _ = p.getBasePositionAndOrientation(oid)
        except Exception:
            return False

        if pos[2] < workspace_limits[2][0] - 0.01:
            return False

        p.removeBody(oid)
        self.object_ids = [bid for bid in self.object_ids if bid != oid]
        for _ in range(30):
            p.stepSimulation()
        return True

    def restart_real(self):
        return False
