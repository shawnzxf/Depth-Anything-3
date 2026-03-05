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

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from geometry_utils import depth_to_point_cloud_vectorized, timing
from loop_utils.alignment_torch import (
    apply_sim3_direct_torch,
    depth_to_point_cloud_optimized_torch,
)
from loop_utils.sim3utils import (
    accumulate_sim3_transforms,
    compute_sim3_ab,
    precompute_scale_chunks_with_depth,
    save_confident_pointcloud_batch,
    weighted_align_point_maps,
)

matplotlib.use("Agg")


def align_2pcds(
    point_map1,
    conf1,
    point_map2,
    conf2,
    chunk1_depth,
    chunk2_depth,
    chunk1_depth_conf,
    chunk2_depth_conf,
    config,
):
    """Align two overlapping point clouds using weighted SIM(3) estimation.

    Args:
        point_map1, point_map2: np.ndarray [N, H, W, 3] point maps.
        conf1, conf2: np.ndarray [N, H, W] confidence maps.
        chunk1_depth, chunk2_depth: np.ndarray or None (for scale+se3).
        chunk1_depth_conf, chunk2_depth_conf: np.ndarray or None.
        config: dict -- the full config.

    Returns:
        tuple: (s, R, t) -- scale, rotation, translation.
    """
    conf_threshold = min(np.median(conf1), np.median(conf2)) * 0.1

    scale_factor = None
    if config["Model"]["align_method"] == "scale+se3":
        scale_factor_return, quality_score, method_used = precompute_scale_chunks_with_depth(
            chunk1_depth,
            chunk1_depth_conf,
            chunk2_depth,
            chunk2_depth_conf,
            method=config["Model"]["scale_compute_method"],
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
        config=config,
        precompute_scale=scale_factor,
    )
    print("Estimated Scale:", s)
    print("Estimated Rotation:\n", R)
    print("Estimated Translation:", t)

    return s, R, t


def run_cross_chunk_alignment(chunk_indices, overlap, get_chunk_file_path_fn, config):
    """Run pairwise alignment across all consecutive chunk pairs.

    Args:
        chunk_indices: list of (start, end) tuples.
        overlap: int -- number of overlap frames.
        get_chunk_file_path_fn: callable(local_chunk_idx) -> str path.
        config: dict.

    Returns:
        list: sim3_list of (s, R, t) tuples, length = len(chunk_indices) - 1.
    """
    sim3_list = []
    pre_predictions = None
    for chunk_idx in range(len(chunk_indices)):
        print(f"[Alignment Progress]: {chunk_idx}/{len(chunk_indices)}")

        cur_predictions = np.load(
            get_chunk_file_path_fn(chunk_idx),
            allow_pickle=True,
        ).item()

        if chunk_idx > 0:
            print(
                f"Aligning {chunk_idx-1} and {chunk_idx} (Total {len(chunk_indices)-1})"
            )
            chunk_data1 = pre_predictions
            chunk_data2 = cur_predictions

            point_map1 = depth_to_point_cloud_vectorized(
                chunk_data1.depth, chunk_data1.intrinsics, chunk_data1.extrinsics
            )
            point_map2 = depth_to_point_cloud_vectorized(
                chunk_data2.depth, chunk_data2.intrinsics, chunk_data2.extrinsics
            )

            point_map1 = point_map1[-overlap:]
            point_map2 = point_map2[:overlap]
            conf1 = chunk_data1.conf[-overlap:]
            conf2 = chunk_data2.conf[:overlap]

            if config["Model"]["align_method"] == "scale+se3":
                chunk1_depth = np.squeeze(chunk_data1.depth[-overlap:])
                chunk2_depth = np.squeeze(chunk_data2.depth[:overlap])
                chunk1_depth_conf = np.squeeze(chunk_data1.conf[-overlap:])
                chunk2_depth_conf = np.squeeze(chunk_data2.conf[:overlap])
            else:
                chunk1_depth = None
                chunk2_depth = None
                chunk1_depth_conf = None
                chunk2_depth_conf = None

            s, R, t = align_2pcds(
                point_map1,
                conf1,
                point_map2,
                conf2,
                chunk1_depth,
                chunk2_depth,
                chunk1_depth_conf,
                chunk2_depth_conf,
                config,
            )
            sim3_list.append((s, R, t))

        pre_predictions = cur_predictions

    return sim3_list


def get_loop_sim3_from_loop_predict(loop_predict_list, chunk_indices,
                                    get_chunk_file_path_fn, config):
    """Compute SIM(3) transforms for loop closure pairs.

    Args:
        loop_predict_list: list of ((chunk_a_idx, range_a, chunk_b_idx, range_b), predictions).
        chunk_indices: list of (start, end) tuples.
        get_chunk_file_path_fn: callable(local_chunk_idx) -> str path.
        config: dict.

    Returns:
        list: loop_sim3_list of (chunk_idx_a, chunk_idx_b, (s, R, t)).
    """
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

        chunk_a_rela_begin = chunk_a_range[0] - chunk_indices[chunk_idx_a][0]
        chunk_a_rela_end = chunk_a_rela_begin + chunk_a_len
        chunk_b_rela_begin = chunk_b_range[0] - chunk_indices[chunk_idx_b][0]
        chunk_b_rela_end = chunk_b_rela_begin + chunk_b_len

        print("chunk_a align")

        point_map_loop_a = point_map_loop_org[chunk_a_s:chunk_a_e]
        conf_loop = item[1].conf[chunk_a_s:chunk_a_e]
        print(chunk_indices[chunk_idx_a])
        print(chunk_a_range)
        print(chunk_a_rela_begin, chunk_a_rela_end)
        chunk_data_a = np.load(
            get_chunk_file_path_fn(chunk_idx_a),
            allow_pickle=True,
        ).item()

        point_map_a = depth_to_point_cloud_vectorized(
            chunk_data_a.depth, chunk_data_a.intrinsics, chunk_data_a.extrinsics
        )
        point_map_a = point_map_a[chunk_a_rela_begin:chunk_a_rela_end]
        conf_a = chunk_data_a.conf[chunk_a_rela_begin:chunk_a_rela_end]

        if config["Model"]["align_method"] == "scale+se3":
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

        s_a, R_a, t_a = align_2pcds(
            point_map_a,
            conf_a,
            point_map_loop_a,
            conf_loop,
            chunk_a_depth,
            chunk_a_loop_depth,
            chunk_a_depth_conf,
            chunk_a_loop_depth_conf,
            config,
        )

        print("chunk_b align")

        point_map_loop_b = point_map_loop_org[chunk_b_s:chunk_b_e]
        conf_loop = item[1].conf[chunk_b_s:chunk_b_e]
        print(chunk_indices[chunk_idx_b])
        print(chunk_b_range)
        print(chunk_b_rela_begin, chunk_b_rela_end)
        chunk_data_b = np.load(
            get_chunk_file_path_fn(chunk_idx_b),
            allow_pickle=True,
        ).item()

        point_map_b = depth_to_point_cloud_vectorized(
            chunk_data_b.depth, chunk_data_b.intrinsics, chunk_data_b.extrinsics
        )
        point_map_b = point_map_b[chunk_b_rela_begin:chunk_b_rela_end]
        conf_b = chunk_data_b.conf[chunk_b_rela_begin:chunk_b_rela_end]

        if config["Model"]["align_method"] == "scale+se3":
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

        s_b, R_b, t_b = align_2pcds(
            point_map_b,
            conf_b,
            point_map_loop_b,
            conf_loop,
            chunk_b_depth,
            chunk_b_loop_depth,
            chunk_b_depth_conf,
            chunk_b_loop_depth_conf,
            config,
        )

        print("a -> b SIM 3")
        s_ab, R_ab, t_ab = compute_sim3_ab((s_a, R_a, t_a), (s_b, R_b, t_b))
        print("Estimated Scale:", s_ab)
        print("Estimated Rotation:\n", R_ab)
        print("Estimated Translation:", t_ab)

        loop_sim3_list.append((chunk_idx_a, chunk_idx_b, (s_ab, R_ab, t_ab)))

    return loop_sim3_list


def plot_loop_closure(input_abs_poses, optimized_abs_poses, loop_sim3_list,
                      output_dir, save_name="sim3_opt_result.png"):
    """Plot before/after loop closure optimization results.

    Args:
        input_abs_poses: torch.Tensor of absolute poses before optimization.
        optimized_abs_poses: torch.Tensor of absolute poses after optimization.
        loop_sim3_list: list of (chunk_a, chunk_b, sim3) tuples.
        output_dir: str -- directory to save the plot.
        save_name: str -- filename for the plot.
    """
    def extract_xyz(pose_tensor):
        poses = pose_tensor.cpu().numpy()
        return poses[:, 0], poses[:, 1], poses[:, 2]

    x0, _, y0 = extract_xyz(input_abs_poses)
    x1, _, y1 = extract_xyz(optimized_abs_poses)

    # Visual in png format
    plt.figure(figsize=(8, 6))
    plt.plot(x0, y0, "o--", alpha=0.45, label="Before Optimization")
    plt.plot(x1, y1, "o-", label="After Optimization")
    for i, j, _ in loop_sim3_list:
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
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


@timing
def apply_alignment(chunk_indices, sim3_list, get_chunk_file_path_fn,
                    result_aligned_dir, pcd_dir, config,
                    save_depth_conf_fn=None):
    """Apply accumulated SIM(3) transforms to all chunks and save results.

    Args:
        chunk_indices: list of (start, end) tuples.
        sim3_list: list of (s, R, t) -- will be accumulated internally.
        get_chunk_file_path_fn: callable(local_chunk_idx) -> str path.
        result_aligned_dir: str -- directory for aligned chunk output.
        pcd_dir: str -- directory for point cloud PLY output.
        config: dict.
        save_depth_conf_fn: optional callable(predictions, chunk_idx, s, R, t)
                            for saving depth/conf results.

    Returns:
        list: The accumulated sim3_list.
    """
    print("Apply alignment")

    sim3_list = accumulate_sim3_transforms(sim3_list)
    for chunk_idx in range(len(chunk_indices) - 1):
        print(f"Applying {chunk_idx+1} -> {chunk_idx} (Total {len(chunk_indices)-1})")
        s, R, t = sim3_list[chunk_idx]

        chunk_data = np.load(
            get_chunk_file_path_fn(chunk_idx + 1),
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

        aligned_path = os.path.join(result_aligned_dir, f"chunk_{chunk_idx+1}.npy")
        np.save(aligned_path, aligned_chunk_data)

        if chunk_idx == 0:
            chunk_data_first = np.load(
                get_chunk_file_path_fn(0), allow_pickle=True
            ).item()
            np.save(os.path.join(result_aligned_dir, "chunk_0.npy"), chunk_data_first)
            points_first = depth_to_point_cloud_vectorized(
                chunk_data_first.depth,
                chunk_data_first.intrinsics,
                chunk_data_first.extrinsics,
            )
            colors_first = chunk_data_first.processed_images
            confs_first = chunk_data_first.conf
            ply_path_first = os.path.join(pcd_dir, "0_pcd.ply")
            save_confident_pointcloud_batch(
                points=points_first,  # shape: (H, W, 3)
                colors=colors_first,  # shape: (H, W, 3)
                confs=confs_first,  # shape: (H, W)
                output_path=ply_path_first,
                conf_threshold=np.mean(confs_first)
                * config["Model"]["Pointcloud_Save"]["conf_threshold_coef"],
                sample_ratio=config["Model"]["Pointcloud_Save"]["sample_ratio"],
            )
            if config["Model"]["save_depth_conf_result"] and save_depth_conf_fn is not None:
                predictions = chunk_data_first
                save_depth_conf_fn(predictions, 0, 1, np.eye(3), np.array([0, 0, 0]))

        points = aligned_chunk_data["world_points"].reshape(-1, 3)
        colors = (aligned_chunk_data["images"].reshape(-1, 3)).astype(np.uint8)
        confs = aligned_chunk_data["conf"].reshape(-1)
        ply_path = os.path.join(pcd_dir, f"{chunk_idx+1}_pcd.ply")
        save_confident_pointcloud_batch(
            points=points,  # shape: (H, W, 3)
            colors=colors,  # shape: (H, W, 3)
            confs=confs,  # shape: (H, W)
            output_path=ply_path,
            conf_threshold=np.mean(confs)
            * config["Model"]["Pointcloud_Save"]["conf_threshold_coef"],
            sample_ratio=config["Model"]["Pointcloud_Save"]["sample_ratio"],
        )

        if config["Model"]["save_depth_conf_result"] and save_depth_conf_fn is not None:
            predictions = chunk_data
            predictions.depth *= s
            save_depth_conf_fn(predictions, chunk_idx + 1, s, R, t)

    return sim3_list
