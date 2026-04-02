# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import os
import shutil
import time

import numpy as np
import torch
from loop_utils.sim3utils import merge_ply_files


def timing(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"[Timing] {func.__name__}: {time.time() - start:.3f}s")
        return result
    wrapper.__name__ = func.__name__
    return wrapper


def depth_to_point_cloud_vectorized(depth, intrinsics, extrinsics, device=None):
    """
    depth: [N, H, W] numpy array or torch tensor
    intrinsics: [N, 3, 3] numpy array or torch tensor
    extrinsics: [N, 3, 4] (w2c) numpy array or torch tensor
    Returns: point_cloud_world: [N, H, W, 3] same type as input
    """
    input_is_numpy = False
    if isinstance(depth, np.ndarray):
        input_is_numpy = True

        depth_tensor = torch.tensor(depth, dtype=torch.float32)
        intrinsics_tensor = torch.tensor(intrinsics, dtype=torch.float32)
        extrinsics_tensor = torch.tensor(extrinsics, dtype=torch.float32)

        if device is not None:
            depth_tensor = depth_tensor.to(device)
            intrinsics_tensor = intrinsics_tensor.to(device)
            extrinsics_tensor = extrinsics_tensor.to(device)
    else:
        depth_tensor = depth
        intrinsics_tensor = intrinsics
        extrinsics_tensor = extrinsics

    if device is not None:
        depth_tensor = depth_tensor.to(device)
        intrinsics_tensor = intrinsics_tensor.to(device)
        extrinsics_tensor = extrinsics_tensor.to(device)

    # main logic

    N, H, W = depth_tensor.shape

    device = depth_tensor.device

    u = torch.arange(W, device=device).float().view(1, 1, W, 1).expand(N, H, W, 1)
    v = torch.arange(H, device=device).float().view(1, H, 1, 1).expand(N, H, W, 1)
    ones = torch.ones((N, H, W, 1), device=device)
    pixel_coords = torch.cat([u, v, ones], dim=-1)

    intrinsics_inv = torch.inverse(intrinsics_tensor)  # [N, 3, 3]
    camera_coords = torch.einsum("nij,nhwj->nhwi", intrinsics_inv, pixel_coords)
    camera_coords = camera_coords * depth_tensor.unsqueeze(-1)
    camera_coords_homo = torch.cat([camera_coords, ones], dim=-1)

    extrinsics_4x4 = torch.zeros(N, 4, 4, device=device)
    extrinsics_4x4[:, :3, :4] = extrinsics_tensor
    extrinsics_4x4[:, 3, 3] = 1.0

    c2w = torch.inverse(extrinsics_4x4)
    world_coords_homo = torch.einsum("nij,nhwj->nhwi", c2w, camera_coords_homo)
    point_cloud_world = world_coords_homo[..., :3]

    if input_is_numpy:
        point_cloud_world = point_cloud_world.cpu().numpy()

    return point_cloud_world


def remove_duplicates(data_list):
    """
    data_list: [(67, (3386, 3406), 48, (2435, 2455)), ...]
    """
    seen = {}
    result = []

    for item in data_list:
        if item[0] == item[2]:
            continue

        key = (item[0], item[2])

        if key not in seen.keys():
            seen[key] = True
            result.append(item)

    return result


@timing
def merge_point_clouds(save_dir, delete_after_merge=False):
    """Merge point clouds for each group and handle segment distribution.

    For groups with segments (e.g., group_0 has group_0_0, group_0_1):
    1. Merge individual PLY files in parent group_X/pcd/ into combined_pcd.ply
    2. Copy combined_pcd.ply to each segment's pcd/ directory
    3. Delete parent group after distribution

    For groups without segments:
    - Just merge PLY files as before
    """
    # Find all parent group directories (group_0, group_1, not group_0_0)
    all_dirs = sorted(glob.glob(os.path.join(save_dir, "group_*")))
    parent_groups = [d for d in all_dirs if os.path.basename(d).count('_') == 1]

    for group_dir in parent_groups:
        pcd_dir = os.path.join(group_dir, "pcd")
        if not os.path.isdir(pcd_dir):
            continue

        # Merge individual PLY files into combined_pcd.ply
        combined_ply_path = os.path.join(pcd_dir, "combined_pcd.ply")
        print(f"Merging point clouds for {os.path.basename(group_dir)}")
        merge_ply_files(pcd_dir, combined_ply_path, delete_after_merge)

        # Check if this group has segments (e.g., group_0 has group_0_0, group_0_1)
        group_basename = os.path.basename(group_dir)
        segment_pattern = os.path.join(save_dir, f"{group_basename}_*")
        segment_dirs = sorted(glob.glob(segment_pattern))

        if segment_dirs:
            # Copy combined_pcd.ply to each segment's pcd directory
            print(f"  Copying combined PCD to {len(segment_dirs)} segments")
            for segment_dir in segment_dirs:
                segment_pcd_dir = os.path.join(segment_dir, "pcd")
                os.makedirs(segment_pcd_dir, exist_ok=True)
                dest_ply = os.path.join(segment_pcd_dir, "combined_pcd.ply")
                shutil.copy2(combined_ply_path, dest_ply)
                print(f"    Copied to {os.path.basename(segment_dir)}")

            # Delete parent group after distributing PCD
            print(f"  Deleting parent group: {group_dir}")
            shutil.rmtree(group_dir)


def copy_file(src_path, dst_dir):
    try:
        os.makedirs(dst_dir, exist_ok=True)

        dst_path = os.path.join(dst_dir, os.path.basename(src_path))

        shutil.copy2(src_path, dst_path)
        print(f"config yaml file has been copied to: {dst_path}")
        return dst_path

    except FileNotFoundError:
        print("File Not Found")
    except PermissionError:
        print("Permission Error")
    except Exception as e:
        print(f"Copy Error: {e}")
