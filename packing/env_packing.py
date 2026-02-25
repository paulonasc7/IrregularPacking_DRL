import glob
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch


@dataclass
class PlacementResult:
    success: bool
    stable: bool
    done: bool
    info: dict[str, Any]


class PackingEnv:
    """Geometry-only packing environment for SLS-style planning."""

    def __init__(
        self,
        obj_dir: str,
        is_gui: bool = False,
        box_size: tuple[float, float, float] = (0.4, 0.4, 0.3),
        resolution: int = 200,
        seed: int = 1234,
        default_num_objects: int = 50,
        default_scale_range: tuple[float, float] = (0.8, 1.2),
        gravity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        use_cuda_state: bool = False,
    ) -> None:
        del is_gui  # Kept for API compatibility.
        self.obj_dir = obj_dir if obj_dir.endswith(os.sep) else obj_dir + os.sep
        self.box_size = box_size
        self.resolution = int(resolution)
        self.rng = np.random.default_rng(seed)
        self.default_num_objects = int(max(1, default_num_objects))
        self.default_scale_range = (float(default_scale_range[0]), float(default_scale_range[1]))
        self.gravity = (float(gravity[0]), float(gravity[1]), float(gravity[2]))
        self._use_cuda_state = bool(use_cuda_state) and torch.cuda.is_available()
        self._state_device = torch.device("cuda" if self._use_cuda_state else "cpu")

        self._catalog = self._load_object_catalog(self.obj_dir)
        if self._catalog.empty:
            raise ValueError(f"No objects found in '{self.obj_dir}'. Expected objects.csv or URDF files.")

        self._catalog_dims = [self._estimate_dims_from_urdf(path) for path in self._catalog["path"].tolist()]
        self._catalog_volumes = [self._estimate_volume_from_urdf(path) for path in self._catalog["path"].tolist()]

        self.loaded_ids: list[int] = []
        self.loaded_catalog_idx: list[int] = []
        self.loaded_scales: list[float] = []
        self._item_dims: dict[int, tuple[float, float, float]] = {}
        self._item_volumes: dict[int, float] = {}
        self._item_local_points: dict[int, np.ndarray] = {}
        self._item_local_points_torch: dict[tuple[int, str], torch.Tensor] = {}
        self.unpacked: list[int] = []
        self.packed: list[int] = []
        self.step_count: int = 0
        self.episode_id: int = 0
        self._box_hm: np.ndarray = np.zeros((self.resolution, self.resolution), dtype=np.float64)
        self._box_hm_t: torch.Tensor | None = None
        self._box_hm_dirty: bool = False
        self._item_hm_cache: dict[tuple[int, tuple[float, float, float]], tuple[np.ndarray, np.ndarray]] = {}
        self._item_hm_torch_cache: dict[
            tuple[int, tuple[float, float, float], str],
            tuple[torch.Tensor, torch.Tensor],
        ] = {}

    def _load_object_catalog(self, obj_dir: str) -> pd.DataFrame:
        csv_path = os.path.join(obj_dir, "objects.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if "dir" not in df.columns:
                raise ValueError(f"objects.csv at {csv_path} must contain a 'dir' column.")
            paths = [os.path.join(obj_dir, str(rel)) for rel in df["dir"].tolist()]
            df = pd.DataFrame({"path": paths})
        else:
            urdfs = sorted(glob.glob(os.path.join(obj_dir, "**", "*.urdf"), recursive=True))
            df = pd.DataFrame({"path": urdfs})

        df = df[df["path"].map(os.path.exists)].reset_index(drop=True)
        return df

    @staticmethod
    def _read_obj_bbox(mesh_path: str) -> tuple[np.ndarray, np.ndarray] | None:
        mins = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
        maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
        found = False
        try:
            with open(mesh_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if not line.startswith("v "):
                        continue
                    parts = line.strip().split()
                    if len(parts) < 4:
                        continue
                    v = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
                    mins = np.minimum(mins, v)
                    maxs = np.maximum(maxs, v)
                    found = True
        except Exception:
            return None
        if not found:
            return None
        return mins, maxs

    @staticmethod
    def _compute_mesh_volume_from_obj(mesh_path: str) -> float | None:
        """Compute actual mesh volume from OBJ file using signed tetrahedron method.
        
        For each triangle face, compute the signed volume of the tetrahedron formed
        by the triangle and the origin. The sum gives the total enclosed volume.
        This works for any closed mesh regardless of winding order (we take abs()).
        """
        vertices: list[np.ndarray] = []
        faces: list[tuple[int, int, int]] = []
        try:
            with open(mesh_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("v "):
                        parts = line.split()
                        if len(parts) >= 4:
                            vertices.append(np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64))
                    elif line.startswith("f "):
                        parts = line.split()[1:]
                        # Parse face indices (OBJ is 1-indexed, may have v/vt/vn format)
                        indices = []
                        for p in parts:
                            idx_str = p.split("/")[0]
                            try:
                                indices.append(int(idx_str) - 1)  # Convert to 0-indexed
                            except ValueError:
                                pass
                        # Triangulate polygon faces (fan triangulation)
                        if len(indices) >= 3:
                            for i in range(1, len(indices) - 1):
                                faces.append((indices[0], indices[i], indices[i + 1]))
        except Exception:
            return None
        
        if len(vertices) < 4 or len(faces) < 4:
            return None
        
        # Compute signed volume sum
        total_volume = 0.0
        for i0, i1, i2 in faces:
            if i0 >= len(vertices) or i1 >= len(vertices) or i2 >= len(vertices):
                continue
            v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]
            # Signed volume of tetrahedron with origin: (v0 · (v1 × v2)) / 6
            cross = np.cross(v1, v2)
            signed_vol = np.dot(v0, cross) / 6.0
            total_volume += signed_vol
        
        return abs(total_volume)

    def _estimate_volume_from_urdf(self, urdf_path: str) -> float:
        """Estimate actual mesh volume from URDF, falling back to AABB if mesh parsing fails."""
        try:
            root = ET.parse(urdf_path).getroot()
        except Exception:
            # Fallback to AABB volume
            dims = self._estimate_dims_from_urdf(urdf_path)
            return float(dims[0] * dims[1] * dims[2])
        
        mesh_node = root.find(".//collision/geometry/mesh")
        if mesh_node is None:
            mesh_node = root.find(".//visual/geometry/mesh")
        if mesh_node is None:
            dims = self._estimate_dims_from_urdf(urdf_path)
            return float(dims[0] * dims[1] * dims[2])
        
        filename = mesh_node.attrib.get("filename", "").strip()
        if not filename:
            dims = self._estimate_dims_from_urdf(urdf_path)
            return float(dims[0] * dims[1] * dims[2])
        
        mesh_path = filename if os.path.isabs(filename) else os.path.join(os.path.dirname(urdf_path), filename)
        
        # Get scale from URDF
        scale_text = mesh_node.attrib.get("scale", "1 1 1")
        try:
            scale = np.array([float(x) for x in scale_text.split()], dtype=np.float64)
            if scale.size != 3:
                scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        except Exception:
            scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        
        mesh_volume = self._compute_mesh_volume_from_obj(mesh_path)
        if mesh_volume is None:
            # Fallback to AABB
            dims = self._estimate_dims_from_urdf(urdf_path)
            return float(dims[0] * dims[1] * dims[2])
        
        # Scale volume: volume scales as product of scale factors (not cube of single scale)
        scale_factor = abs(scale[0] * scale[1] * scale[2])
        return float(mesh_volume * scale_factor)

    def _estimate_dims_from_urdf(self, urdf_path: str) -> tuple[float, float, float]:
        default_dims = (0.06, 0.06, 0.06)
        try:
            root = ET.parse(urdf_path).getroot()
        except Exception:
            return default_dims

        mesh_node = root.find(".//collision/geometry/mesh")
        if mesh_node is None:
            mesh_node = root.find(".//visual/geometry/mesh")
        if mesh_node is None:
            return default_dims

        filename = mesh_node.attrib.get("filename", "").strip()
        if not filename:
            return default_dims
        mesh_path = filename if os.path.isabs(filename) else os.path.join(os.path.dirname(urdf_path), filename)
        bbox = self._read_obj_bbox(mesh_path)
        if bbox is None:
            return default_dims

        mins, maxs = bbox
        dims = np.maximum(maxs - mins, 1e-4)
        scale_text = mesh_node.attrib.get("scale", "1 1 1")
        try:
            scale = np.array([float(x) for x in scale_text.split()], dtype=np.float64)
            if scale.size != 3:
                scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        except Exception:
            scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        dims = dims * np.abs(scale)
        return float(dims[0]), float(dims[1]), float(dims[2])

    def _load_items(self, catalog_indices: list[int], scales: list[float] | None = None) -> None:
        self.loaded_ids = []
        self.loaded_catalog_idx = list(catalog_indices)
        self.loaded_scales = []
        self._item_dims = {}
        self._item_volumes = {}
        self._item_local_points = {}
        self._item_local_points_torch = {}
        if scales is None:
            scales = [1.0] * len(catalog_indices)
        if len(scales) != len(catalog_indices):
            raise ValueError("scales length must match catalog_indices length.")

        for i, catalog_idx in enumerate(catalog_indices):
            scale = float(scales[i])
            base_dims = np.asarray(self._catalog_dims[catalog_idx], dtype=np.float64)
            base_volume = float(self._catalog_volumes[catalog_idx])
            dims = tuple((base_dims * scale).tolist())
            # Volume scales as scale^3 (uniform) or product of scale factors per axis
            volume = base_volume * (scale ** 3)
            item_id = int(i)
            self.loaded_ids.append(item_id)
            self.loaded_scales.append(scale)
            self._item_dims[item_id] = (float(dims[0]), float(dims[1]), float(dims[2]))
            self._item_volumes[item_id] = float(volume)
            self._item_local_points[item_id] = self._build_local_points(self._item_dims[item_id])

    def _remove_all_items(self) -> None:
        self.loaded_ids = []
        self.loaded_catalog_idx = []
        self.loaded_scales = []
        self._item_dims = {}
        self._item_volumes = {}
        self._item_local_points = {}
        self._item_local_points_torch = {}
        self._item_hm_cache = {}
        self._item_hm_torch_cache = {}
        self._box_hm_t = None
        self._box_hm_dirty = False

    def reset(self, object_ids: list[int] | None = None, object_scales: list[float] | None = None) -> dict[str, Any]:
        self.episode_id += 1
        self.step_count = 0

        self._remove_all_items()

        if object_ids is None:
            k = min(self.default_num_objects, len(self._catalog))
            sampled = self.rng.choice(len(self._catalog), size=k, replace=False)
            object_ids = [int(x) for x in sampled]
        if object_scales is None:
            s0, s1 = self.default_scale_range
            object_scales = self.rng.uniform(s0, s1, size=len(object_ids)).astype(np.float64).tolist()

        if len(object_ids) == 0:
            raise ValueError("reset requires at least one object.")
        bad = [obj for obj in object_ids if obj < 0 or obj >= len(self._catalog)]
        if bad:
            raise IndexError(f"Invalid object ids: {bad}. Catalog size={len(self._catalog)}")
        if len(object_scales) != len(object_ids):
            raise ValueError("object_scales length must match object_ids length.")

        self._load_items(object_ids, scales=object_scales)
        self.unpacked = list(self.loaded_ids)
        self.packed = []
        self._box_hm = np.zeros((self.resolution, self.resolution), dtype=np.float64)
        self._box_hm_t = (
            torch.zeros((self.resolution, self.resolution), dtype=torch.float32, device=self._state_device)
            if self._use_cuda_state
            else None
        )
        self._box_hm_dirty = False
        self._item_hm_cache = {}
        self._item_hm_torch_cache = {}
        return self.get_state()

    def get_state(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "step_count": self.step_count,
            "box_heightmap": self.box_heightmap(),
            "unpacked": list(self.unpacked),
            "packed": list(self.packed),
            "loaded_ids": list(self.loaded_ids),
            "catalog_indices": list(self.loaded_catalog_idx),
            "object_scales": list(self.loaded_scales),
            "box_size": self.box_size,
            "resolution": self.resolution,
        }

    @property
    def catalog_size(self) -> int:
        return int(len(self._catalog))

    def item_dims(self, item_id: int) -> tuple[float, float, float]:
        if item_id not in self._item_dims:
            raise ValueError(f"Unknown item id {item_id}.")
        return self._item_dims[item_id]

    def item_volume(self, item_id: int) -> float:
        """Return actual mesh volume of the item (not AABB volume)."""
        if item_id not in self._item_volumes:
            raise ValueError(f"Unknown item id {item_id}.")
        return self._item_volumes[item_id]

    def candidate_orientations(self, item_id: int) -> np.ndarray:
        if item_id not in self.loaded_ids:
            raise ValueError(f"Unknown item id {item_id}.")
        values = np.arange(0, 2 * np.pi, np.pi / 2, dtype=np.float64)
        yaw, pitch, roll = np.meshgrid(values, values, values)
        return np.stack([roll.reshape(-1), pitch.reshape(-1), yaw.reshape(-1)], axis=1)

    def box_heightmap(self, copy: bool = True) -> np.ndarray:
        if self._use_cuda_state and self._box_hm_t is not None and self._box_hm_dirty:
            self._box_hm = self._box_hm_t.detach().cpu().numpy().astype(np.float64)
            self._box_hm_dirty = False
        if copy:
            return self._box_hm.copy()
        return self._box_hm

    def box_heightmap_t(self, copy: bool = True) -> torch.Tensor:
        if self._box_hm_t is None:
            self._box_hm_t = torch.from_numpy(self._box_hm.astype(np.float32, copy=False)).to(self._state_device)
            self._box_hm_dirty = False
        if copy:
            return self._box_hm_t.clone()
        return self._box_hm_t

    @staticmethod
    def _euler_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
        ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
        rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
        return rz @ ry @ rx

    @staticmethod
    def _euler_to_matrix_torch(roll: float, pitch: float, yaw: float, device: torch.device) -> torch.Tensor:
        """Optimized Euler to rotation matrix conversion using batched tensor ops."""
        # Compute sin/cos values
        angles = torch.tensor([roll, pitch, yaw], dtype=torch.float32, device=device)
        cos_vals = torch.cos(angles)
        sin_vals = torch.sin(angles)
        cr, cp, cy = cos_vals[0], cos_vals[1], cos_vals[2]
        sr, sp, sy = sin_vals[0], sin_vals[1], sin_vals[2]
        
        # Build rotation matrix directly without multiple torch.tensor() calls
        zero = torch.tensor(0.0, dtype=torch.float32, device=device)
        one = torch.tensor(1.0, dtype=torch.float32, device=device)
        
        # Combined rotation matrix Rz @ Ry @ Rx computed directly
        # This is more efficient than 3 separate matrix multiplications
        r00 = cy * cp
        r01 = cy * sp * sr - sy * cr
        r02 = cy * sp * cr + sy * sr
        r10 = sy * cp
        r11 = sy * sp * sr + cy * cr
        r12 = sy * sp * cr - cy * sr
        r20 = -sp
        r21 = cp * sr
        r22 = cp * cr
        
        return torch.stack([
            torch.stack([r00, r01, r02]),
            torch.stack([r10, r11, r12]),
            torch.stack([r20, r21, r22]),
        ], dim=0)

    @staticmethod
    def _build_local_points(dims: tuple[float, float, float]) -> np.ndarray:
        dx, dy, dz = [float(v) for v in dims]
        # Cache dense volume samples once per item to avoid rebuilding per orientation.
        nx = max(6, int(np.ceil(dx / 0.01)))
        ny = max(6, int(np.ceil(dy / 0.01)))
        nz = max(6, int(np.ceil(dz / 0.01)))
        xs = np.linspace(-dx / 2.0, dx / 2.0, nx, dtype=np.float64)
        ys = np.linspace(-dy / 2.0, dy / 2.0, ny, dtype=np.float64)
        zs = np.linspace(-dz / 2.0, dz / 2.0, nz, dtype=np.float64)
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        return np.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], axis=1)

    def item_hm(self, item_id: int, orient: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(orient, torch.Tensor):
            orient_arr = orient.detach().cpu().numpy().astype(np.float64, copy=False).reshape(3)
        else:
            orient_arr = np.asarray(orient, dtype=np.float64).reshape(3)
        key = (int(item_id), tuple(np.round(orient_arr, 8).tolist()))
        cached = self._item_hm_cache.get(key)
        if cached is not None:
            return cached

        pts = self._item_local_points[item_id]

        R = self._euler_to_matrix(float(orient_arr[0]), float(orient_arr[1]), float(orient_arr[2]))
        rpts = pts @ R.T
        min_xyz = np.min(rpts, axis=0)
        max_xyz = np.max(rpts, axis=0)
        span = max_xyz - min_xyz
        sep = self.box_size[0] / self.resolution

        w = max(1, int(np.ceil(span[0] / max(1e-9, sep))))
        h = max(1, int(np.ceil(span[1] / max(1e-9, sep))))

        xi = np.floor((rpts[:, 0] - min_xyz[0]) / max(1e-9, sep)).astype(np.int32)
        yi = np.floor((rpts[:, 1] - min_xyz[1]) / max(1e-9, sep)).astype(np.int32)
        xi = np.clip(xi, 0, w - 1)
        yi = np.clip(yi, 0, h - 1)

        flat_idx = xi * h + yi
        zvals = rpts[:, 2]
        zmax = np.full(w * h, -np.inf, dtype=np.float64)
        zmin = np.full(w * h, np.inf, dtype=np.float64)
        np.maximum.at(zmax, flat_idx, zvals)
        np.minimum.at(zmin, flat_idx, zvals)
        zmax = zmax.reshape(w, h)
        zmin = zmin.reshape(w, h)

        global_min_z = float(np.min(rpts[:, 2]))
        ht = np.zeros((w, h), dtype=np.float64)
        hb = np.full((w, h), np.inf, dtype=np.float64)
        occ = np.isfinite(zmax) & np.isfinite(zmin)
        ht[occ] = zmax[occ] - global_min_z
        hb[occ] = zmin[occ] - global_min_z

        self._item_hm_cache[key] = (ht, hb)
        return self._item_hm_cache[key]

    def item_hm_torch(
        self,
        item_id: int,
        orient: np.ndarray,
        device: torch.device | str = "cuda",
        ht_np: np.ndarray | None = None,
        hb_np: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(orient, torch.Tensor):
            orient_arr = orient.detach().cpu().numpy().astype(np.float64, copy=False).reshape(3)
        else:
            orient_arr = np.asarray(orient, dtype=np.float64).reshape(3)
        ori_key = tuple(np.round(orient_arr, 8).tolist())
        dev = torch.device(device)
        cache_key = (int(item_id), ori_key, str(dev))
        cached = self._item_hm_torch_cache.get(cache_key)
        if cached is not None:
            return cached

        # If CPU maps are explicitly provided, convert directly.
        if ht_np is not None and hb_np is not None:
            ht_t = torch.from_numpy(np.asarray(ht_np, dtype=np.float32)).to(dev)
            hb_t = torch.from_numpy(np.asarray(hb_np, dtype=np.float32)).to(dev)
            self._item_hm_torch_cache[cache_key] = (ht_t, hb_t)
            return self._item_hm_torch_cache[cache_key]

        pts_key = (int(item_id), str(dev))
        pts_t = self._item_local_points_torch.get(pts_key)
        if pts_t is None:
            pts_t = torch.from_numpy(self._item_local_points[item_id].astype(np.float32, copy=False)).to(dev)
            self._item_local_points_torch[pts_key] = pts_t

        R_t = self._euler_to_matrix_torch(
            float(orient_arr[0]),
            float(orient_arr[1]),
            float(orient_arr[2]),
            dev,
        )
        rpts = pts_t @ R_t.transpose(0, 1)
        min_xyz = torch.amin(rpts, dim=0)
        max_xyz = torch.amax(rpts, dim=0)
        span = max_xyz - min_xyz
        sep = float(self.box_size[0] / self.resolution)

        w = max(1, int(torch.ceil(span[0] / max(1e-9, sep)).item()))
        h = max(1, int(torch.ceil(span[1] / max(1e-9, sep)).item()))

        xi = torch.floor((rpts[:, 0] - min_xyz[0]) / max(1e-9, sep)).to(torch.int64)
        yi = torch.floor((rpts[:, 1] - min_xyz[1]) / max(1e-9, sep)).to(torch.int64)
        xi = torch.clamp(xi, 0, w - 1)
        yi = torch.clamp(yi, 0, h - 1)

        flat_idx = xi * h + yi
        zvals = rpts[:, 2]
        size = w * h
        zmax = torch.full((size,), float("-inf"), dtype=torch.float32, device=dev)
        zmin = torch.full((size,), float("inf"), dtype=torch.float32, device=dev)
        zmax.scatter_reduce_(0, flat_idx, zvals, reduce="amax", include_self=True)
        zmin.scatter_reduce_(0, flat_idx, zvals, reduce="amin", include_self=True)
        zmax = zmax.view(w, h)
        zmin = zmin.view(w, h)

        global_min_z = torch.amin(zvals)
        ht_t = torch.zeros((w, h), dtype=torch.float32, device=dev)
        hb_t = torch.full((w, h), float("inf"), dtype=torch.float32, device=dev)
        occ = torch.isfinite(zmax) & torch.isfinite(zmin)
        ht_t = torch.where(occ, zmax - global_min_z, ht_t)
        hb_t = torch.where(occ, zmin - global_min_z, hb_t)

        self._item_hm_torch_cache[cache_key] = (ht_t, hb_t)
        return self._item_hm_torch_cache[cache_key]

    def pack_item(self, item_id: int, transform: np.ndarray | torch.Tensor) -> bool:
        if item_id not in self.loaded_ids:
            return False

        if isinstance(transform, torch.Tensor):
            t = transform.detach()
            target_euler_t = t[0:3].to(self._state_device, dtype=torch.float32)
            grid_x = int(torch.round(t[3]).item())
            grid_y = int(torch.round(t[4]).item())
            target_z = float(t[5].item())
            if not bool(torch.isfinite(target_euler_t).all().item()) or not np.isfinite(target_z):
                return False
        else:
            target_euler = np.asarray(transform[0:3], dtype=np.float64)
            grid_x = int(np.round(float(transform[3])))
            grid_y = int(np.round(float(transform[4])))
            target_z = float(transform[5])
            if not np.isfinite(target_euler).all() or not np.isfinite(target_z):
                return False

        if self._use_cuda_state and self._box_hm_t is not None:
            ht_t, hb_t = self.item_hm_torch(
                item_id,
                target_euler_t if isinstance(transform, torch.Tensor) else target_euler,
                device=self._state_device,
            )
            if int(ht_t.numel()) == 0 or int(hb_t.numel()) == 0:
                return False
            w, h = int(ht_t.shape[0]), int(ht_t.shape[1])
            if grid_x < 0 or grid_y < 0 or (grid_x + w) > self.resolution or (grid_y + h) > self.resolution:
                return False

            support_t = self._box_hm_t[grid_x : grid_x + w, grid_y : grid_y + h]
            required_z = float(torch.max(support_t - hb_t).item())
            if abs(target_z - required_z) > 1e-4:
                return False

            updated_t = torch.maximum((ht_t > 0).to(torch.float32) * (ht_t + float(target_z)), support_t)
            if float(torch.max(updated_t).item()) > float(self.box_size[2]) + 1e-6:
                return False

            self._box_hm_t[grid_x : grid_x + w, grid_y : grid_y + h] = updated_t
            self._box_hm_dirty = True
            return True

        ht, hb = self.item_hm(item_id, target_euler)
        if ht.size == 0 or hb.size == 0:
            return False
        w, h = ht.shape
        if grid_x < 0 or grid_y < 0 or (grid_x + w) > self.resolution or (grid_y + h) > self.resolution:
            return False

        support = self._box_hm[grid_x : grid_x + w, grid_y : grid_y + h]
        required_z = float(np.max(support - hb))
        if abs(target_z - required_z) > 1e-4:
            return False

        updated = np.maximum((ht > 0) * (ht + target_z), support)
        if float(np.max(updated)) > float(self.box_size[2]) + 1e-6:
            return False

        self._box_hm[grid_x : grid_x + w, grid_y : grid_y + h] = updated
        if self._box_hm_t is not None:
            self._box_hm_t = torch.from_numpy(self._box_hm.astype(np.float32, copy=False)).to(self._state_device)
            self._box_hm_dirty = False
        return True

    def step(self, item_id: int, placement: np.ndarray | torch.Tensor) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        self.step_count += 1
        info: dict[str, Any] = {"item_id": int(item_id), "valid_action": True}

        if item_id not in self.unpacked:
            info["valid_action"] = False
            info["reason"] = "item_not_unpacked"
            next_state = self.get_state()
            return next_state, 0.0, len(self.unpacked) == 0, info

        if isinstance(placement, torch.Tensor):
            if placement.numel() != 6:
                raise ValueError(f"placement must have shape (6,), got {tuple(placement.shape)}")
            if not bool(torch.isfinite(placement).all().item()):
                raise ValueError("placement contains non-finite values.")
            grid_x, grid_y = float(placement[3].item()), float(placement[4].item())
        else:
            placement = np.asarray(placement, dtype=np.float64)
            if placement.shape != (6,):
                raise ValueError(f"placement must have shape (6,), got {placement.shape}")
            if not np.isfinite(placement).all():
                raise ValueError("placement contains non-finite values.")
            grid_x, grid_y = float(placement[3]), float(placement[4])

        if grid_x < 0 or grid_x >= self.resolution or grid_y < 0 or grid_y >= self.resolution:
            info["valid_action"] = False
            info["reason"] = "grid_out_of_bounds"
            next_state = self.get_state()
            return next_state, 0.0, len(self.unpacked) == 0, info

        stable = self.pack_item(item_id, placement)
        if stable:
            self.unpacked.remove(item_id)
            self.packed.append(item_id)

        done = len(self.unpacked) == 0
        reward = 1.0 if stable else 0.0
        info["stable"] = bool(stable)
        info["packed_count"] = len(self.packed)
        info["remaining_count"] = len(self.unpacked)
        next_state = self.get_state()
        return next_state, reward, done, info

    def close(self) -> None:
        return None
