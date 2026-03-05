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
#
# Adapted from [VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long)

"""
nohup python da3_streaming.py --image_dir /home/ubuntu/sky_workdir/nav_vid_data_pipeline/sample_dataset/z8At5dmBo6M/frames_2k --output_dir /home/ubuntu/sky_workdir/nav_vid_data_pipeline/sample_dataset/z8At5dmBo6M/da3_outputs/frames_2k > /home/ubuntu/sky_workdir/nav_vid_data_pipeline/logs/z8At5dmBo6M_2k_02222251.log 2>&1 &
"""

import argparse
import gc
import glob
import json
import os
import shutil
import sys
import time
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from loop_utils.alignment_torch import (
    apply_sim3_direct_torch,
    depth_to_point_cloud_optimized_torch,
)
from loop_utils.config_utils import load_config
from loop_utils.loop_detector import LoopDetector
from loop_utils.sim3loop import Sim3LoopOptimizer
from loop_utils.sim3utils import (
    accumulate_sim3_transforms,
    compute_sim3_ab,
    merge_ply_files,
    precompute_scale_chunks_with_depth,
    process_loop_list,
    save_confident_pointcloud_batch,
    warmup_numba,
    weighted_align_point_maps,
)
from safetensors.torch import load_file

from depth_anything_3.api import DepthAnything3

matplotlib.use("Agg")


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


class DA3_Streaming:
    @timing
    def __init__(self, image_dir, save_dir, config, first_frame=None, last_frame=None, config_path=None):
        self.config = config
        self.config_path = config_path
        self.first_frame = first_frame if first_frame is not None else 0
        self.last_frame = last_frame

        self.chunk_size = self.config["Model"]["chunk_size"]
        self.overlap = self.config["Model"]["overlap"]
        self.overlap_s = 0
        self.overlap_e = self.overlap - self.overlap_s
        self.conf_threshold = 1.5
        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = (
            torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        )

        self.img_dir = image_dir
        self.img_list = None
        self.output_dir = save_dir
        self.delete_temp_files = self.config["Model"]["delete_temp_files"]

        # Per-group state (initialized empty; populated in _prepare_for_group)
        self._group_chunk_id_map = None  # list mapping local -> global chunk idx
        self._temp_dirs_to_clean = []

        self._prepare_for_da3_inference(save_dir)
        print("init done.")

    def _prepare_for_alignment(self, save_dir):
        self.result_aligned_dir = os.path.join(save_dir, "_tmp_results_aligned")
        self.result_loop_dir = os.path.join(save_dir, "_tmp_results_loop")
        self.result_output_dir = os.path.join(save_dir, "results_output")
        self.pcd_dir = os.path.join(save_dir, "pcd")
        os.makedirs(self.result_aligned_dir, exist_ok=True)
        os.makedirs(self.result_loop_dir, exist_ok=True)
        os.makedirs(self.pcd_dir, exist_ok=True)

        # Track temp dirs for cleanup in close()
        self._temp_dirs_to_clean.extend([
            self.result_aligned_dir, self.result_loop_dir
        ])

        self.skyseg_session = None

        self.loop_list = []  # e.g. [(1584, 139), ...]

        self.loop_optimizer = Sim3LoopOptimizer(self.config)

        self.sim3_list = []  # [(s [1,], R [3,3], T [3,]), ...]

        self.loop_sim3_list = []  # [(chunk_idx_a, chunk_idx_b, s [1,], R [3,3], T [3,]), ...]

        self.loop_predict_list = []

        self.loop_enable = self.config["Model"]["loop_enable"]

        if self.loop_enable:
            self.loop_detector = LoopDetector(config=self.config)
            self.loop_detector.load_model()


    def _aggregate_pixel_to_frame_confidence(self, conf):
        """Aggregate pixel-wise confidence to per-frame boolean flags.

        For each frame, computes the ``pixel_to_frame_percentile``-th percentile
        of pixel-wise confidence values and compares it against
        ``frame_confidence_threshold``.

        Args:
            conf: np.ndarray of shape [N, H, W] with pixel-wise confidence.

        Returns:
            np.ndarray of shape [N] with dtype bool (True = selected).
        """
        conf_config = self.config.get("Confidence", {})
        percentile = conf_config.get("pixel_to_frame_percentile", 75)
        threshold = conf_config.get("frame_confidence_threshold", 0.2)

        frame_scores = np.percentile(conf, percentile, axis=(1, 2))
        return frame_scores >= threshold

    def _aggregate_frame_to_chunk_confidence(self, chunk_idx):
        """Decide whether a chunk should be kept based on frame-level flags.

        A chunk is dropped if it contains a continuous streak of False
        (unconfident) frames whose length is at least
        ``drop_continuous_ratio`` of the chunk length.

        Args:
            chunk_idx: Index into ``self.chunk_frame_confidence_flags``.

        Returns:
            bool: True if the chunk should be kept, False if it should be dropped.
        """
        conf_config = self.config.get("Confidence", {})
        drop_ratio = conf_config.get("drop_continuous_ratio", 0.2)

        frame_flags = self.chunk_frame_confidence_flags[chunk_idx]
        max_consecutive_false = int(drop_ratio * len(frame_flags))

        consecutive_false = 0
        for flag in frame_flags:
            if not flag:
                consecutive_false += 1
                if consecutive_false >= max_consecutive_false:
                    return False
            else:
                consecutive_false = 0
        return True

    def _compute_chunk_confidence_groups(self):
        """
        Phase 2: Compute chunk-level confidence and partition into groups
        of consecutive confident chunks.

        Returns:
            list of lists: Each inner list contains the original (global) chunk indices
                           that form one group. E.g., [[0,1,2,3,4], [8,9,10,11,12]]
        """
        conf_config = self.config.get("Confidence", {})
        if not conf_config.get("enable", True):
            print("[Confidence] Disabled — treating all chunks as one group.")
            return [list(range(len(self.chunk_indices)))]

        min_chunks = conf_config.get("min_chunks_per_group", 3)

        num_chunks = len(self.chunk_indices)
        chunk_keep = []

        for i in range(num_chunks):
            keep = self._aggregate_frame_to_chunk_confidence(i)
            chunk_keep.append(keep)
            flags = self.chunk_frame_confidence_flags[i]
            n_selected = int(np.sum(flags))
            local_start, local_end = self.chunk_indices[i]
            global_start = local_start + self.first_frame
            global_end = local_end + self.first_frame
            print(f"[Confidence] Chunk {i}, frames id [{global_start}, {global_end}): "
                  f"{n_selected}/{len(flags)} frames selected — {'OK' if keep else 'DROP'}")

        # Partition into groups of consecutive confident chunks
        groups = []
        current_group = []
        for i in range(num_chunks):
            if chunk_keep[i]:
                current_group.append(i)
            else:
                if len(current_group) >= min_chunks:
                    groups.append(current_group)
                elif current_group:
                    print(f"[Confidence] Dropping small group {current_group} "
                          f"({len(current_group)} < {min_chunks} chunks)")
                current_group = []
        # Finalize last group
        if len(current_group) >= min_chunks:
            groups.append(current_group)
        elif current_group:
            print(f"[Confidence] Dropping small group {current_group} "
                  f"({len(current_group)} < {min_chunks} chunks)")

        print(f"[Confidence] {num_chunks} total chunks -> {len(groups)} groups: "
              f"{[f'{g[0]}-{g[-1]}' for g in groups]}")

        return groups

    def _get_chunk_file_path(self, local_chunk_idx):
        """Map a group-local chunk index to the unaligned chunk file path on disk."""
        if self._group_chunk_id_map is not None:
            global_idx = self._group_chunk_id_map[local_chunk_idx]
        else:
            global_idx = local_chunk_idx
        return os.path.join(self.result_unaligned_dir, f"chunk_{global_idx}.npy")

    @staticmethod
    def _remap_chunk_metadata(all_data, group_chunk_ids, frame_offset):
        """Remap a list of ((start, end), payload) tuples to group-local frame indices."""
        result = []
        for gid in group_chunk_ids:
            (start, end), payload = all_data[gid]
            result.append(((start - frame_offset, end - frame_offset), payload))
        return result

    def _prepare_for_group(self, group_output_dir, group_chunk_ids,
                           all_chunk_indices, all_camera_poses,
                           all_camera_intrinsics, all_img_list):
        """
        Set up all per-group state before running post-processing.

        Remaps chunk_indices, img_list, camera poses/intrinsics to be
        group-local, then calls _prepare_for_alignment() for directory
        and optimizer setup.
        """
        self.output_dir = group_output_dir
        os.makedirs(group_output_dir, exist_ok=True)

        # Group-local -> global chunk index mapping (list; index = local, value = global)
        self._group_chunk_id_map = list(group_chunk_ids)

        # Frame range for this group (in the original full img_list)
        group_frame_start = all_chunk_indices[group_chunk_ids[0]][0]
        group_frame_end = all_chunk_indices[group_chunk_ids[-1]][1]

        # Slice img_list to group's frames
        self.img_list = all_img_list[group_frame_start:group_frame_end]

        # Remap chunk_indices to be relative to group_frame_start
        self.chunk_indices = [
            (start - group_frame_start, end - group_frame_start)
            for start, end in [all_chunk_indices[gid] for gid in group_chunk_ids]
        ]

        # Remap camera poses and intrinsics
        self.all_camera_poses = self._remap_chunk_metadata(
            all_camera_poses, group_chunk_ids, group_frame_start
        )
        self.all_camera_intrinsics = self._remap_chunk_metadata(
            all_camera_intrinsics, group_chunk_ids, group_frame_start
        )

        # Sanity checks
        assert self.chunk_indices[0][0] == 0, (
            f"First chunk in group should start at 0, got {self.chunk_indices[0][0]}"
        )
        assert len(self.img_list) == group_frame_end - group_frame_start, (
            f"img_list length {len(self.img_list)} != frame span "
            f"{group_frame_end - group_frame_start}"
        )

        print(f"[Group] output_dir={group_output_dir}")
        print(f"[Group] frames={group_frame_start}-{group_frame_end} "
              f"({len(self.img_list)} frames), "
              f"{len(self.chunk_indices)} chunks")

        # Initialize alignment state (dirs, sim3_list, loop detector, etc.)
        self._prepare_for_alignment(group_output_dir)

    def _prepare_for_da3_inference(self, save_dir):
        print("Loading model...")

        with open(self.config["Weights"]["DA3_CONFIG"]) as f:
            config = json.load(f)
        self.model = DepthAnything3(**config)
        weight = load_file(self.config["Weights"]["DA3"])
        self.model.load_state_dict(weight, strict=False)

        self.model.eval()
        self.model = self.model.to(self.device)

        # prepare temp dir to store unaligned prediction
        self.result_unaligned_dir = os.path.join(save_dir, "_tmp_results_unaligned")
        os.makedirs(self.result_unaligned_dir, exist_ok=True)

        # DA3 inference stores all poses and intrinsics with the following attr
        self.all_camera_poses = []
        self.all_camera_intrinsics = []
        self.chunk_frame_confidence_flags = []


    def get_loop_pairs(self):
        loop_info_save_path = os.path.join(self.output_dir, "loop_closures.txt")
        self.loop_detector.run(image_paths=self.img_list, output=loop_info_save_path)
        loop_list = self.loop_detector.get_loop_list()
        return loop_list

    def save_depth_conf_result(self, predictions, chunk_idx, s, R, T):
        if not self.config["Model"]["save_depth_conf_result"]:
            return
        os.makedirs(self.result_output_dir, exist_ok=True)

        chunk_start, chunk_end = self.chunk_indices[chunk_idx]

        if chunk_idx == 0:
            save_indices = list(range(0, chunk_end - chunk_start - self.overlap_e))
        elif chunk_idx == len(self.chunk_indices) - 1:
            save_indices = list(range(self.overlap_s, chunk_end - chunk_start))
        else:
            save_indices = list(range(self.overlap_s, chunk_end - chunk_start - self.overlap_e))

        print("[save_depth_conf_result] save_indices:")

        for local_idx in save_indices:
            global_idx = chunk_start + local_idx
            print(f"{global_idx}, ", end="")

            image = predictions.processed_images[local_idx]  # [H, W, 3] uint8
            depth = predictions.depth[local_idx]  # [H, W] float32
            conf = predictions.conf[local_idx]  # [H, W] float32
            intrinsics = predictions.intrinsics[local_idx]  # [3, 3] float32

            filename = f"frame_{global_idx}.npz"
            filepath = os.path.join(self.result_output_dir, filename)

            if self.config["Model"]["save_debug_info"]:
                np.savez_compressed(
                    filepath,
                    image=image,
                    depth=depth,
                    conf=conf,
                    intrinsics=intrinsics,
                    extrinsics=predictions.extrinsics[local_idx],
                    s=s,
                    R=R,
                    T=T,
                )
            else:
                np.savez_compressed(
                    filepath, image=image, depth=depth, conf=conf, intrinsics=intrinsics
                )
        print("")

    @timing
    def process_single_chunk(self, range_1, chunk_idx=None, range_2=None, is_loop=False):
        """
        self.img_list
        self.result_unaligned_dir

        self.all_camera_poses
        self.all_camera_intrinsics
        """
        start_idx, end_idx = range_1
        chunk_image_paths = self.img_list[start_idx:end_idx]
        if range_2 is not None:
            start_idx, end_idx = range_2
            chunk_image_paths += self.img_list[start_idx:end_idx]

        # images = load_and_preprocess_images(chunk_image_paths).to(self.device)
        # print(f"Loaded {len(chunk_image_paths)} images")

        ref_view_strategy = self.config["Model"][
            "ref_view_strategy" if not is_loop else "ref_view_strategy_loop"
        ]

        torch.cuda.empty_cache()
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                images = chunk_image_paths
                # images: ['xxx.png', 'xxx.png', ...]

                predictions = self.model.inference(images, ref_view_strategy=ref_view_strategy)

                predictions.depth = np.squeeze(predictions.depth)
                predictions.conf -= 1.0

                # print(predictions.processed_images.shape)  # [N, H, W, 3] uint8
                # print(predictions.depth.shape)  # [N, H, W] float32
                # print(predictions.conf.shape)  # [N, H, W] float32
                # print(predictions.extrinsics.shape)  # [N, 3, 4] float32 (w2c)
                # print(predictions.intrinsics.shape)  # [N, 3, 3] float32
        torch.cuda.empty_cache()

        # Save predictions to disk instead of keeping in memory
        if is_loop:
            save_dir = self.result_loop_dir
            filename = f"loop_{range_1[0]}_{range_1[1]}_{range_2[0]}_{range_2[1]}.npy"
        else:
            if chunk_idx is None:
                raise ValueError("chunk_idx must be provided when is_loop is False")
            save_dir = self.result_unaligned_dir
            filename = f"chunk_{chunk_idx}.npy"

        save_path = os.path.join(save_dir, filename)

        if not is_loop and range_2 is None:
            extrinsics = predictions.extrinsics
            intrinsics = predictions.intrinsics
            chunk_range = self.chunk_indices[chunk_idx]
            self.all_camera_poses.append((chunk_range, extrinsics))
            self.all_camera_intrinsics.append((chunk_range, intrinsics))

            frame_flags = self._aggregate_pixel_to_frame_confidence(predictions.conf)
            self.chunk_frame_confidence_flags.append(frame_flags)

        np.save(save_path, predictions)

        return predictions

    def get_chunk_indices(self):
        """Compute overlapping chunk start/end indices for the image list.

        Splits the image list into chunks of size ``self.chunk_size`` with
        ``self.overlap`` frames shared between consecutive chunks. When the
        total number of images fits within a single chunk, one chunk covering
        the entire list is returned.

        Returns:
            tuple[list[tuple[int, int]], int]: A list of [start, end) index
                pairs and the total number of chunks.
        """
        if len(self.img_list) <= self.chunk_size:
            num_chunks = 1
            chunk_indices = [(0, len(self.img_list))]
        else:
            step = self.chunk_size - self.overlap
            num_chunks = (len(self.img_list) - self.overlap + step - 1) // step
            chunk_indices = []
            for i in range(num_chunks):
                start_idx = i * step
                end_idx = min(start_idx + self.chunk_size, len(self.img_list))
                chunk_indices.append((start_idx, end_idx))
        return chunk_indices, num_chunks

    def align_2pcds(
        self,
        point_map1,
        conf1,
        point_map2,
        conf2,
        chunk1_depth,
        chunk2_depth,
        chunk1_depth_conf,
        chunk2_depth_conf,
    ):

        conf_threshold = min(np.median(conf1), np.median(conf2)) * 0.1

        scale_factor = None
        if self.config["Model"]["align_method"] == "scale+se3":
            scale_factor_return, quality_score, method_used = precompute_scale_chunks_with_depth(
                chunk1_depth,
                chunk1_depth_conf,
                chunk2_depth,
                chunk2_depth_conf,
                method=self.config["Model"]["scale_compute_method"],
            )
            print(
                f"[Depth Scale Precompute] scale: {scale_factor_return}, \
                    quality_score: {quality_score}, method_used: {method_used}"
            )
            scale_factor = scale_factor_return

        s, R, t = weighted_align_point_maps(
            point_map1,
            conf1,
            point_map2,
            conf2,
            conf_threshold=conf_threshold,
            config=self.config,
            precompute_scale=scale_factor,
        )
        print("Estimated Scale:", s)
        print("Estimated Rotation:\n", R)
        print("Estimated Translation:", t)

        return s, R, t

    def get_loop_sim3_from_loop_predict(self, loop_predict_list):
        loop_sim3_list = []
        for item in loop_predict_list:
            chunk_idx_a = item[0][0]
            chunk_idx_b = item[0][2]
            chunk_a_range = item[0][1]
            chunk_b_range = item[0][3]

            point_map_loop_org = depth_to_point_cloud_vectorized(
                item[1].depth, item[1].intrinsics, item[1].extrinsics
            )

            chunk_a_s = 0
            chunk_a_e = chunk_a_len = chunk_a_range[1] - chunk_a_range[0]
            chunk_b_s = -chunk_b_range[1] + chunk_b_range[0]
            chunk_b_e = point_map_loop_org.shape[0]
            chunk_b_len = chunk_b_range[1] - chunk_b_range[0]

            chunk_a_rela_begin = chunk_a_range[0] - self.chunk_indices[chunk_idx_a][0]
            chunk_a_rela_end = chunk_a_rela_begin + chunk_a_len
            chunk_b_rela_begin = chunk_b_range[0] - self.chunk_indices[chunk_idx_b][0]
            chunk_b_rela_end = chunk_b_rela_begin + chunk_b_len

            print("chunk_a align")

            point_map_loop_a = point_map_loop_org[chunk_a_s:chunk_a_e]
            conf_loop = item[1].conf[chunk_a_s:chunk_a_e]
            print(self.chunk_indices[chunk_idx_a])
            print(chunk_a_range)
            print(chunk_a_rela_begin, chunk_a_rela_end)
            chunk_data_a = np.load(
                self._get_chunk_file_path(chunk_idx_a),
                allow_pickle=True,
            ).item()

            point_map_a = depth_to_point_cloud_vectorized(
                chunk_data_a.depth, chunk_data_a.intrinsics, chunk_data_a.extrinsics
            )
            point_map_a = point_map_a[chunk_a_rela_begin:chunk_a_rela_end]
            conf_a = chunk_data_a.conf[chunk_a_rela_begin:chunk_a_rela_end]

            if self.config["Model"]["align_method"] == "scale+se3":
                chunk_a_depth = np.squeeze(chunk_data_a.depth[chunk_a_rela_begin:chunk_a_rela_end])
                chunk_a_depth_conf = np.squeeze(
                    chunk_data_a.conf[chunk_a_rela_begin:chunk_a_rela_end]
                )
                chunk_a_loop_depth = np.squeeze(item[1].depth[chunk_a_s:chunk_a_e])
                chunk_a_loop_depth_conf = np.squeeze(item[1].conf[chunk_a_s:chunk_a_e])
            else:
                chunk_a_depth = None
                chunk_a_loop_depth = None
                chunk_a_depth_conf = None
                chunk_a_loop_depth_conf = None

            s_a, R_a, t_a = self.align_2pcds(
                point_map_a,
                conf_a,
                point_map_loop_a,
                conf_loop,
                chunk_a_depth,
                chunk_a_loop_depth,
                chunk_a_depth_conf,
                chunk_a_loop_depth_conf,
            )

            print("chunk_b align")

            point_map_loop_b = point_map_loop_org[chunk_b_s:chunk_b_e]
            conf_loop = item[1].conf[chunk_b_s:chunk_b_e]
            print(self.chunk_indices[chunk_idx_b])
            print(chunk_b_range)
            print(chunk_b_rela_begin, chunk_b_rela_end)
            chunk_data_b = np.load(
                self._get_chunk_file_path(chunk_idx_b),
                allow_pickle=True,
            ).item()

            point_map_b = depth_to_point_cloud_vectorized(
                chunk_data_b.depth, chunk_data_b.intrinsics, chunk_data_b.extrinsics
            )
            point_map_b = point_map_b[chunk_b_rela_begin:chunk_b_rela_end]
            conf_b = chunk_data_b.conf[chunk_b_rela_begin:chunk_b_rela_end]

            if self.config["Model"]["align_method"] == "scale+se3":
                chunk_b_depth = np.squeeze(chunk_data_b.depth[chunk_b_rela_begin:chunk_b_rela_end])
                chunk_b_depth_conf = np.squeeze(
                    chunk_data_b.conf[chunk_b_rela_begin:chunk_b_rela_end]
                )
                chunk_b_loop_depth = np.squeeze(item[1].depth[chunk_b_s:chunk_b_e])
                chunk_b_loop_depth_conf = np.squeeze(item[1].conf[chunk_b_s:chunk_b_e])
            else:
                chunk_b_depth = None
                chunk_b_loop_depth = None
                chunk_b_depth_conf = None
                chunk_b_loop_depth_conf = None

            s_b, R_b, t_b = self.align_2pcds(
                point_map_b,
                conf_b,
                point_map_loop_b,
                conf_loop,
                chunk_b_depth,
                chunk_b_loop_depth,
                chunk_b_depth_conf,
                chunk_b_loop_depth_conf,
            )

            print("a -> b SIM 3")
            s_ab, R_ab, t_ab = compute_sim3_ab((s_a, R_a, t_a), (s_b, R_b, t_b))
            print("Estimated Scale:", s_ab)
            print("Estimated Rotation:\n", R_ab)
            print("Estimated Translation:", t_ab)

            loop_sim3_list.append((chunk_idx_a, chunk_idx_b, (s_ab, R_ab, t_ab)))

        return loop_sim3_list

    def plot_loop_closure(
        self, input_abs_poses, optimized_abs_poses, save_name="sim3_opt_result.png"
    ):
        def extract_xyz(pose_tensor):
            poses = pose_tensor.cpu().numpy()
            return poses[:, 0], poses[:, 1], poses[:, 2]

        x0, _, y0 = extract_xyz(input_abs_poses)
        x1, _, y1 = extract_xyz(optimized_abs_poses)

        # Visual in png format
        plt.figure(figsize=(8, 6))
        plt.plot(x0, y0, "o--", alpha=0.45, label="Before Optimization")
        plt.plot(x1, y1, "o-", label="After Optimization")
        for i, j, _ in self.loop_sim3_list:
            plt.plot(
                [x0[i], x0[j]],
                [y0[i], y0[j]],
                "r--",
                alpha=0.25,
                label="Loop (Before)" if i == 5 else "",
            )
            plt.plot(
                [x1[i], x1[j]],
                [y1[i], y1[j]],
                "g-",
                alpha=0.25,
                label="Loop (After)" if i == 5 else "",
            )
        plt.gca().set_aspect("equal")
        plt.title("Sim3 Loop Closure Optimization")
        plt.xlabel("x")
        plt.ylabel("z")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _run_cross_chunk_alignment(self):
        pre_predictions = None
        for chunk_idx in range(len(self.chunk_indices)):
            print(f"[Alignment Progress]: {chunk_idx}/{len(self.chunk_indices)}")

            cur_predictions = np.load(
                self._get_chunk_file_path(chunk_idx),
                allow_pickle=True,
            ).item()

            if chunk_idx > 0:
                print(
                    f"Aligning {chunk_idx-1} and {chunk_idx} (Total {len(self.chunk_indices)-1})"
                )
                chunk_data1 = pre_predictions
                chunk_data2 = cur_predictions

                point_map1 = depth_to_point_cloud_vectorized(
                    chunk_data1.depth, chunk_data1.intrinsics, chunk_data1.extrinsics
                )
                point_map2 = depth_to_point_cloud_vectorized(
                    chunk_data2.depth, chunk_data2.intrinsics, chunk_data2.extrinsics
                )

                point_map1 = point_map1[-self.overlap :]
                point_map2 = point_map2[: self.overlap]
                conf1 = chunk_data1.conf[-self.overlap :]
                conf2 = chunk_data2.conf[: self.overlap]

                if self.config["Model"]["align_method"] == "scale+se3":
                    chunk1_depth = np.squeeze(chunk_data1.depth[-self.overlap :])
                    chunk2_depth = np.squeeze(chunk_data2.depth[: self.overlap])
                    chunk1_depth_conf = np.squeeze(chunk_data1.conf[-self.overlap :])
                    chunk2_depth_conf = np.squeeze(chunk_data2.conf[: self.overlap])
                else:
                    chunk1_depth = None
                    chunk2_depth = None
                    chunk1_depth_conf = None
                    chunk2_depth_conf = None

                s, R, t = self.align_2pcds(
                    point_map1,
                    conf1,
                    point_map2,
                    conf2,
                    chunk1_depth,
                    chunk2_depth,
                    chunk1_depth_conf,
                    chunk2_depth_conf,
                )
                self.sim3_list.append((s, R, t))

            pre_predictions = cur_predictions

    @timing
    def _run_loop_closure_optimization(self):

        self.loop_list = self.get_loop_pairs()
        del self.loop_detector  # Save GPU Memory

        torch.cuda.empty_cache()

        print("Loop SIM(3) estimating...")
        loop_results = process_loop_list(
            self.chunk_indices,
            self.loop_list,
            half_window=int(self.config["Model"]["loop_chunk_size"] / 2),
        )
        loop_results = remove_duplicates(loop_results)
        print(loop_results)
        # return e.g. (31, (1574, 1594), 2, (129, 149))
        for item in loop_results:
            single_chunk_predictions = self.process_single_chunk(
                item[1], range_2=item[3], is_loop=True
            )

            self.loop_predict_list.append((item, single_chunk_predictions))
            print(item)

        self.loop_sim3_list = self.get_loop_sim3_from_loop_predict(self.loop_predict_list)

        input_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(
            self.sim3_list
        )  # just for plot
        self.sim3_list = self.loop_optimizer.optimize(self.sim3_list, self.loop_sim3_list)
        optimized_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(
            self.sim3_list
        )  # just for plot

        self.plot_loop_closure(
            input_abs_poses, optimized_abs_poses, save_name="sim3_opt_result.png"
        )

    @timing
    def _apply_alignment(self):
        print("Apply alignment")

        self.sim3_list = accumulate_sim3_transforms(self.sim3_list)
        for chunk_idx in range(len(self.chunk_indices) - 1):
            print(f"Applying {chunk_idx+1} -> {chunk_idx} (Total {len(self.chunk_indices)-1})")
            s, R, t = self.sim3_list[chunk_idx]

            chunk_data = np.load(
                self._get_chunk_file_path(chunk_idx + 1),
                allow_pickle=True,
            ).item()

            aligned_chunk_data = {}

            aligned_chunk_data["world_points"] = depth_to_point_cloud_optimized_torch(
                chunk_data.depth, chunk_data.intrinsics, chunk_data.extrinsics
            )
            aligned_chunk_data["world_points"] = apply_sim3_direct_torch(
                aligned_chunk_data["world_points"], s, R, t
            )

            aligned_chunk_data["conf"] = chunk_data.conf
            aligned_chunk_data["images"] = chunk_data.processed_images

            aligned_path = os.path.join(self.result_aligned_dir, f"chunk_{chunk_idx+1}.npy")
            np.save(aligned_path, aligned_chunk_data)

            if chunk_idx == 0:
                chunk_data_first = np.load(
                    self._get_chunk_file_path(0), allow_pickle=True
                ).item()
                np.save(os.path.join(self.result_aligned_dir, "chunk_0.npy"), chunk_data_first)
                points_first = depth_to_point_cloud_vectorized(
                    chunk_data_first.depth,
                    chunk_data_first.intrinsics,
                    chunk_data_first.extrinsics,
                )
                colors_first = chunk_data_first.processed_images
                confs_first = chunk_data_first.conf
                ply_path_first = os.path.join(self.pcd_dir, "0_pcd.ply")
                save_confident_pointcloud_batch(
                    points=points_first,  # shape: (H, W, 3)
                    colors=colors_first,  # shape: (H, W, 3)
                    confs=confs_first,  # shape: (H, W)
                    output_path=ply_path_first,
                    conf_threshold=np.mean(confs_first)
                    * self.config["Model"]["Pointcloud_Save"]["conf_threshold_coef"],
                    sample_ratio=self.config["Model"]["Pointcloud_Save"]["sample_ratio"],
                )
                if self.config["Model"]["save_depth_conf_result"]:
                    predictions = chunk_data_first
                    self.save_depth_conf_result(predictions, 0, 1, np.eye(3), np.array([0, 0, 0]))

            points = aligned_chunk_data["world_points"].reshape(-1, 3)
            colors = (aligned_chunk_data["images"].reshape(-1, 3)).astype(np.uint8)
            confs = aligned_chunk_data["conf"].reshape(-1)
            ply_path = os.path.join(self.pcd_dir, f"{chunk_idx+1}_pcd.ply")
            save_confident_pointcloud_batch(
                points=points,  # shape: (H, W, 3)
                colors=colors,  # shape: (H, W, 3)
                confs=confs,  # shape: (H, W)
                output_path=ply_path,
                conf_threshold=np.mean(confs)
                * self.config["Model"]["Pointcloud_Save"]["conf_threshold_coef"],
                sample_ratio=self.config["Model"]["Pointcloud_Save"]["sample_ratio"],
            )

            if self.config["Model"]["save_depth_conf_result"]:
                predictions = chunk_data
                predictions.depth *= s
                self.save_depth_conf_result(predictions, chunk_idx + 1, s, R, t)

    def process_long_sequence(self):
        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"[SETTING ERROR] Overlap ({self.overlap}) \
                    must be less than chunk size ({self.chunk_size})"
            )

        self.chunk_indices, num_chunks = self.get_chunk_indices()

        print(
            f"Processing {len(self.img_list)} images in {num_chunks} chunks of size {self.chunk_size} with {self.overlap} overlap"
        )

        # Phase 1: inference all chunks (independent)
        print(f"\n========== [Phase 1/3] START — DA3 inference ({num_chunks} chunks) ==========")
        self._run_da3_inference()
        print(f"========== [Phase 1/3] END ==========")

        # Phase 2: confidence-based grouping
        print(f"\n========== [Phase 2/3] START — confidence-based grouping ==========")
        chunk_groups = self._compute_chunk_confidence_groups()
        print(f"========== [Phase 2/3] END ==========")

        if not chunk_groups:
            print("[WARNING] No valid chunk groups found. Skipping post-processing.")
            return

        # Save original state before per-group remapping
        all_chunk_indices = self.chunk_indices
        all_camera_poses = self.all_camera_poses
        all_camera_intrinsics = self.all_camera_intrinsics
        all_img_list = self.img_list
        self._root_output_dir = self.output_dir
        self._temp_dirs_to_clean = []

        # Phase 3: per-group post-processing
        print(f"\n========== [Phase 3/3] START — per-group post-processing ({len(chunk_groups)} groups) ==========")
        for group_idx, group_chunk_ids in enumerate(chunk_groups):
            print(f"\n  ---------- [Group {group_idx + 1}/{len(chunk_groups)}] START — chunks {group_chunk_ids} ({len(group_chunk_ids)} chunks) ----------")

            group_output_dir = os.path.join(
                self._root_output_dir, f"group_{group_idx}"
            )

            if self.config_path:
                copy_file(self.config_path, group_output_dir)

            self._prepare_for_group(
                group_output_dir, group_chunk_ids,
                all_chunk_indices, all_camera_poses,
                all_camera_intrinsics, all_img_list,
            )
            self._run_post_processing()

            self._write_group_info(
                group_output_dir, group_idx, group_chunk_ids, all_chunk_indices,
            )

            # Free memory between groups
            gc.collect()
            torch.cuda.empty_cache()

            print(f"  ---------- [Group {group_idx + 1}/{len(chunk_groups)}] END ----------")

        print(f"========== [Phase 3/3] END ==========")

    def _write_group_info(self, group_output_dir, group_idx, group_chunk_ids, all_chunk_indices):
        """Write a JSON info file for a single group into its output dir."""
        frame_start = all_chunk_indices[group_chunk_ids[0]][0]
        frame_end = all_chunk_indices[group_chunk_ids[-1]][1]
        info = {
            "input_video_clip": {
                "video_frame_dir": self.img_dir,
                "first_frame_id": self.first_frame,
                "last_frame_id": self.last_frame + 1,  # plus one to be exclusive
            },
            "group_idx": group_idx,
            "chunk_ids": group_chunk_ids,
            "num_chunks": len(group_chunk_ids),
            "group_frame_id": [frame_start + self.first_frame, frame_end + self.first_frame],
        }
        info_path = os.path.join(group_output_dir, "group_info.json")
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
        print(f"Group info saved to {info_path}")

    def _run_da3_inference(self):
        for chunk_idx in range(len(self.chunk_indices)):
            print(f"[DA3 Inference Progress]: {chunk_idx}/{len(self.chunk_indices)}")

            self.process_single_chunk(
                self.chunk_indices[chunk_idx], chunk_idx=chunk_idx
            )

            torch.cuda.empty_cache()

    def _run_post_processing(self):

        # Phase 2: cross-chunk correction and alignment
        self._run_cross_chunk_alignment()

        if self.loop_enable:
            self._run_loop_closure_optimization()

        self._apply_alignment()

        self.save_camera_poses()

    def _load_image_list(self, first_frame=None, last_frame=None):
        """Load and optionally filter the sorted image list from self.img_dir.

        Args:
            first_frame: Optional integer frame ID of the first frame to include (inclusive).
                         Derives filename as "%06d.png", e.g. 100 -> "000100.png".
            last_frame: Optional integer frame ID of the last frame to include (inclusive).
                        Derives filename as "%06d.png", e.g. 200 -> "000200.png".
        """
        print(f"Loading images from {self.img_dir}...")
        img_list = sorted(
            glob.glob(os.path.join(self.img_dir, "*.jpg"))
            + glob.glob(os.path.join(self.img_dir, "*.png"))
        )
        if len(img_list) == 0:
            raise ValueError(f"[DIR EMPTY] No images found in {self.img_dir}!")

        if first_frame is not None or last_frame is not None:
            basenames = [os.path.basename(p) for p in img_list]

            if first_frame is not None:
                first_fname = f"{first_frame:06d}.png"
                if first_fname not in basenames:
                    raise ValueError(f"first_frame '{first_fname}' (id={first_frame}) not found in {self.img_dir}")
                start_idx = basenames.index(first_fname)
            else:
                start_idx = 0

            if last_frame is not None:
                last_fname = f"{last_frame:06d}.png"
                if last_fname not in basenames:
                    raise ValueError(f"last_frame '{last_fname}' (id={last_frame}) not found in {self.img_dir}")
                end_idx = basenames.index(last_fname) + 1
            else:
                end_idx = len(img_list)

            img_list = img_list[start_idx:end_idx]

            if len(img_list) == 0:
                raise ValueError(
                    f"No images remain after filtering with first_frame={first_frame}, "
                    f"last_frame={last_frame}. Check that first_frame < last_frame."
                )

        print(f"Found {len(img_list)} images")
        return img_list

    @timing
    def run(self):
        self.img_list = self._load_image_list(first_frame=self.first_frame, last_frame=self.last_frame)

        self.process_long_sequence()

    def save_camera_poses(self):
        """
        Save camera poses from all chunks to txt and ply files
        - txt file: Each line contains a 4x4 C2W matrix flattened into 16 numbers
        - ply file: Camera poses visualized as points with different colors for each chunk
        """
        chunk_colors = [
            [255, 0, 0],  # Red
            [0, 255, 0],  # Green
            [0, 0, 255],  # Blue
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cyan
            [128, 0, 0],  # Dark Red
            [0, 128, 0],  # Dark Green
            [0, 0, 128],  # Dark Blue
            [128, 128, 0],  # Olive
        ]
        print("Saving all camera poses to txt file...")

        all_poses = [None] * len(self.img_list)
        all_intrinsics = [None] * len(self.img_list)
        all_chunk_ids = [0] * len(self.img_list)

        first_chunk_range, first_chunk_extrinsics = self.all_camera_poses[0]
        _, first_chunk_intrinsics = self.all_camera_intrinsics[0]

        for i, idx in enumerate(
            range(first_chunk_range[0], first_chunk_range[1] - self.overlap_e)
        ):
            w2c = np.eye(4)
            w2c[:3, :] = first_chunk_extrinsics[i]
            c2w = np.linalg.inv(w2c)
            all_poses[idx] = c2w
            all_intrinsics[idx] = first_chunk_intrinsics[i]
            all_chunk_ids[idx] = 0

        for chunk_idx in range(1, len(self.all_camera_poses)):
            chunk_range, chunk_extrinsics = self.all_camera_poses[chunk_idx]
            _, chunk_intrinsics = self.all_camera_intrinsics[chunk_idx]
            s, R, t = self.sim3_list[
                chunk_idx - 1
            ]  # When call self.save_camera_poses(), all the sim3 are aligned to the first chunk.

            S = np.eye(4)
            S[:3, :3] = s * R
            S[:3, 3] = t

            chunk_range_end = (
                chunk_range[1] - self.overlap_e
                if chunk_idx < len(self.all_camera_poses) - 1
                else chunk_range[1]
            )

            for i, idx in enumerate(range(chunk_range[0] + self.overlap_s, chunk_range_end)):
                w2c = np.eye(4)
                w2c[:3, :] = chunk_extrinsics[i + self.overlap_s]
                c2w = np.linalg.inv(w2c)

                transformed_c2w = S @ c2w  # Be aware of the left multiplication!
                transformed_c2w[:3, :3] /= s  # Normalize rotation

                all_poses[idx] = transformed_c2w
                all_intrinsics[idx] = chunk_intrinsics[i + self.overlap_s]
                all_chunk_ids[idx] = chunk_idx

        poses_path = os.path.join(self.output_dir, "camera_poses.txt")
        with open(poses_path, "w") as f:
            for pose in all_poses:
                flat_pose = pose.flatten()
                f.write(" ".join([str(x) for x in flat_pose]) + "\n")

        print(f"Camera poses saved to {poses_path}")

        intrinsics_path = os.path.join(self.output_dir, "intrinsic.txt")
        with open(intrinsics_path, "w") as f:
            for intrinsic in all_intrinsics:
                fx = intrinsic[0, 0]
                fy = intrinsic[1, 1]
                cx = intrinsic[0, 2]
                cy = intrinsic[1, 2]
                f.write(f"{fx} {fy} {cx} {cy}\n")

        print(f"Camera intrinsics saved to {intrinsics_path}")

        # Build pyramid frustum geometry for each camera.
        # Each frustum has 5 vertices: apex (camera center) + 4 base corners.
        # The base corners are the 4 image corners un-projected to world space at
        # depth = frustum_scale, giving an oriented pyramid that encodes rotation.
        frustum_scale = 0.1  # depth of the frustum in scene units

        ply_vertices = []  # (x, y, z, r, g, b)
        ply_edges = []     # (v1_idx, v2_idx)
        vertex_offset = 0

        for pose, intrinsic, chunk_id in zip(all_poses, all_intrinsics, all_chunk_ids):
            color = chunk_colors[chunk_id % len(chunk_colors)]
            fx = intrinsic[0, 0]
            fy = intrinsic[1, 1]
            cx = intrinsic[0, 2]
            cy = intrinsic[1, 2]

            # Camera center (pyramid apex)
            center = pose[:3, 3]
            R = pose[:3, :3]

            # 4 image corner rays in camera space (principal-point-aware),
            # scaled to frustum_scale depth.  W ≈ 2*cx, H ≈ 2*cy.
            corners_cam = np.array([
                [-cx / fx, -cy / fy, 1.0],  # top-left  (1)
                [ cx / fx, -cy / fy, 1.0],  # top-right (2)
                [ cx / fx,  cy / fy, 1.0],  # bot-right (3)
                [-cx / fx,  cy / fy, 1.0],  # bot-left  (4)
            ]) * frustum_scale

            # Transform corners to world space
            corners_world = (R @ corners_cam.T).T + center

            # Apex vertex (index 0 within this frustum)
            ply_vertices.append((center[0], center[1], center[2],
                                  color[0], color[1], color[2]))
            # 4 base corner vertices (indices 1-4)
            for corner in corners_world:
                ply_vertices.append((corner[0], corner[1], corner[2],
                                     color[0], color[1], color[2]))

            # 4 edges: apex -> each corner
            for k in range(1, 5):
                ply_edges.append((vertex_offset, vertex_offset + k))
            # 4 edges: base rectangle
            ply_edges.append((vertex_offset + 1, vertex_offset + 2))
            ply_edges.append((vertex_offset + 2, vertex_offset + 3))
            ply_edges.append((vertex_offset + 3, vertex_offset + 4))
            ply_edges.append((vertex_offset + 4, vertex_offset + 1))

            vertex_offset += 5

        ply_path = os.path.join(self.output_dir, "camera_poses.ply")
        with open(ply_path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(ply_vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write(f"element edge {len(ply_edges)}\n")
            f.write("property int vertex1\n")
            f.write("property int vertex2\n")
            f.write("end_header\n")
            for v in ply_vertices:
                f.write(f"{v[0]} {v[1]} {v[2]} {v[3]} {v[4]} {v[5]}\n")
            for e in ply_edges:
                f.write(f"{e[0]} {e[1]}\n")

        print(f"Camera poses visualization saved to {ply_path}")

    def close(self):
        """
        Clean up temporary files and calculate reclaimed disk space.

        Deletes unaligned results dir and all per-group aligned/loop temp dirs.
        """
        if not self.delete_temp_files:
            return

        total_space = 0

        # Collect all temp dirs: the shared unaligned dir + per-group aligned/loop dirs
        dirs_to_clean = [self.result_unaligned_dir] + self._temp_dirs_to_clean

        for dir_path in dirs_to_clean:
            if not os.path.exists(dir_path):
                continue
            print(f"Deleting the temp dir {dir_path}")
            for filename in os.listdir(dir_path):
                file_path = os.path.join(dir_path, filename)
                if os.path.isfile(file_path):
                    total_space += os.path.getsize(file_path)
                    os.remove(file_path)
            os.rmdir(dir_path)
        print("Deleting temp dirs done.")

        print(f"Saved disk space: {total_space/1024/1024/1024:.4f} GiB")


@timing
def merge_point_clouds(save_dir):
    # Check if there are group subdirectories
    group_dirs = sorted(glob.glob(os.path.join(save_dir, "group_*")))

    for group_dir in group_dirs:
        pcd_dir = os.path.join(group_dir, "pcd")
        if os.path.isdir(pcd_dir):
            all_ply_path = os.path.join(pcd_dir, "combined_pcd.ply")
            print(f"Merging point clouds for {os.path.basename(group_dir)}")
            merge_ply_files(pcd_dir, all_ply_path)


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


if __name__ == "__main__":
    start_time_main = time.time()

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    parser = argparse.ArgumentParser(description="DA3-Streaming")
    parser.add_argument("--image_dir", type=str, required=True, help="Image path")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="/home/ubuntu/sky_workdir/Depth-Anything-3/da3_streaming/configs/base_config.yaml",
        help="Image path",
    )
    parser.add_argument("--output_dir", type=str, required=False, default=None, help="Output path")
    parser.add_argument("--first_frame", type=int, required=False, default=None,
                        help="Frame ID of the first frame to process (inclusive), e.g. 100")
    parser.add_argument("--last_frame", type=int, required=False, default=None,
                        help="Frame ID of the last frame to process (inclusive), e.g. 200")
    args = parser.parse_args()

    frame_range = f" frames [{args.first_frame}, {args.last_frame}]" if args.first_frame is not None or args.last_frame is not None else ""
    print(f"\n########## [Video] START — {args.image_dir}{frame_range} ##########")

    config = load_config(args.config)

    image_dir = args.image_dir

    if args.output_dir is not None:
        save_dir = args.output_dir
    else:
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        exp_dir = "./exps"
        save_dir = os.path.join(exp_dir, image_dir.replace("/", "_"), current_datetime)

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    print(f"The output will be saved under dir: {save_dir}")

    if config["Model"]["align_lib"] == "numba":
        warmup_numba()

    da3_streaming = DA3_Streaming(image_dir, save_dir, config, first_frame=args.first_frame, last_frame=args.last_frame, config_path=args.config)
    da3_streaming.run()
    da3_streaming.close()

    del da3_streaming
    torch.cuda.empty_cache()
    gc.collect()

    merge_point_clouds(save_dir)

    main_time = time.time() - start_time_main
    print(f"[Timing] total main execution: {main_time:.3f}s")

    print(f"\n########## [Video] END — {args.image_dir}{frame_range} ##########")

    print("DA3-Streaming done.")
    sys.exit()
