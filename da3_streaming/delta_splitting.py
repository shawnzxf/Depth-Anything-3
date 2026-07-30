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

"""Delta-based segment splitting for DA3 streaming.

This module provides functionality to split groups into segments based on
pose delta criteria (motion discontinuities) and quality filtering.
"""

import json
import os
import shutil
from dataclasses import dataclass
from typing import List, Dict

import numpy as np

from output import frame_filename

@dataclass
class DeltaSplitConfig:
    """Configuration for delta-based segment splitting."""
    enable: bool = True
    max_delta_threshold: float = 10.0
    rolling_avg_window: int = 50
    rolling_avg_multiplier: float = 10.0
    min_frames: int = 75
    min_p50_delta: float = 0.01
    max_p50_delta: float = 0.4


@dataclass
class SegmentInterval:
    """Represents a segment interval with quality metrics."""
    start_idx: int  # inclusive
    end_idx: int    # exclusive
    num_frames: int
    p50_delta: float


def load_absolute_poses(group_dir: str) -> List[Dict]:
    """Load absolute poses from camera_absolute_poses.json.

    Args:
        group_dir: Path to group directory

    Returns:
        List of pose dicts with keys: frame_id, rotation, translation

    Raises:
        FileNotFoundError: If camera_absolute_poses.json doesn't exist
        ValueError: If JSON is malformed
    """
    poses_path = os.path.join(group_dir, "camera_absolute_poses.json")

    if not os.path.exists(poses_path):
        raise FileNotFoundError(f"camera_absolute_poses.json not found in {group_dir}")

    with open(poses_path, 'r') as f:
        poses = json.load(f)

    if not isinstance(poses, list) or len(poses) == 0:
        raise ValueError(f"Invalid camera_absolute_poses.json format in {group_dir}")

    print(f"Loaded {len(poses)} poses from {poses_path}")
    return poses


def compute_deltas(poses: List[Dict]) -> np.ndarray:
    """Compute deltas between consecutive frames.

    Args:
        poses: List of pose dicts with translation keys (x, y, z)

    Returns:
        Array of shape (N-1, 3) containing [dx, dy, dz] for consecutive frames
    """
    if len(poses) < 2:
        return np.array([]).reshape(0, 3)

    deltas = []
    for i in range(len(poses) - 1):
        p0 = poses[i]["translation"]
        p1 = poses[i + 1]["translation"]
        dx = p1["x"] - p0["x"]
        dy = p1["y"] - p0["y"]
        dz = p1["z"] - p0["z"]
        deltas.append([dx, dy, dz])

    return np.array(deltas)


def compute_rolling_average(values: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling average with specified window size.

    Args:
        values: 1D array of values
        window: Window size for rolling average

    Returns:
        Array of same length as values with rolling averages
    """
    if len(values) < window:
        # Use smaller window if not enough values
        window = max(1, len(values))

    result = np.zeros_like(values)
    for i in range(len(values)):
        start_idx = max(0, i - window + 1)
        result[i] = np.mean(values[start_idx:i+1])

    return result


def split_by_delta_criteria(deltas: np.ndarray, config: DeltaSplitConfig, first_frame_id: int) -> List[int]:
    """Identify split points based on delta criteria.

    Args:
        deltas: (N-1, 3) array of [dx, dy, dz] for consecutive frames
        config: Configuration with thresholds
        first_frame_id: First frame ID for logging

    Returns:
        Sorted list of frame indices where splits occur
    """
    if len(deltas) == 0:
        return []

    split_points = set()

    # Criterion 1: Absolute delta threshold
    delta_magnitude = np.linalg.norm(deltas, axis=1)  # (N-1,)
    high_delta_mask = delta_magnitude > config.max_delta_threshold

    for i, is_high in enumerate(high_delta_mask):
        if is_high:
            split_points.add(i + 1)  # Split AFTER frame i
            print(f"Split point at frame_id {first_frame_id + i + 1} (idx {i+1}): delta magnitude {delta_magnitude[i]:.4f} > {config.max_delta_threshold}")

    # Criterion 2: Rolling average threshold
    if config.rolling_avg_window > 0 and len(deltas) >= config.rolling_avg_window:
        rolling_avg = compute_rolling_average(delta_magnitude, config.rolling_avg_window)
        threshold = config.rolling_avg_multiplier * rolling_avg

        for i in range(len(delta_magnitude)):
            if delta_magnitude[i] > threshold[i]:
                split_points.add(i + 1)
                print(f"Split point at frame_id {first_frame_id + i + 1} (idx {i+1}): delta {delta_magnitude[i]:.4f} > {config.rolling_avg_multiplier}x rolling avg {rolling_avg[i]:.4f}")

    split_list = sorted(list(split_points))
    print(f"Identified {len(split_list)} split points based on delta criteria")
    return split_list


def create_segments_from_splits(split_indices: List[int], num_frames: int,
                                deltas: np.ndarray, first_frame_id: int) -> List[SegmentInterval]:
    """Convert split points to segment intervals.

    Args:
        split_indices: Sorted list of frame indices where splits occur
        num_frames: Total number of frames
        deltas: (N-1, 3) array of deltas
        first_frame_id: First frame ID for logging

    Returns:
        List of SegmentInterval objects
    """
    if num_frames == 0:
        return []

    # Create boundaries: start with 0, add split points, end with num_frames
    boundaries = [0] + split_indices + [num_frames]

    segments = []
    delta_magnitude = np.linalg.norm(deltas, axis=1) if len(deltas) > 0 else np.array([])

    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]
        num_frames_in_segment = end_idx - start_idx

        # Compute average delta for this segment
        if start_idx < len(delta_magnitude) and end_idx <= len(delta_magnitude) + 1:
            segment_deltas = delta_magnitude[start_idx:min(end_idx-1, len(delta_magnitude))]
            p50_delta = np.percentile(segment_deltas, 50) if len(segment_deltas) > 0 else 0.0
        else:
            p50_delta = 0.0

        segments.append(SegmentInterval(
            start_idx=start_idx,
            end_idx=end_idx,
            num_frames=num_frames_in_segment,
            p50_delta=p50_delta
        ))

    print(f"Created {len(segments)} segments from split points")
    return segments


def filter_segments(segments: List[SegmentInterval], config: DeltaSplitConfig, first_frame_id: int) -> List[SegmentInterval]:
    """Filter segments based on quality criteria.

    Keeps segments that satisfy ALL criteria:
    - num_frames >= min_frames
    - p50_delta >= min_p50_delta
    - p50_delta <= max_p50_delta (if max_p50_delta > 0; filters fast-moving scenes like trains)

    Args:
        segments: List of SegmentInterval objects
        config: Configuration with filter thresholds
        first_frame_id: First frame ID for logging

    Returns:
        Filtered list of segments

    Raises:
        ValueError: If no valid segments remain after filtering
    """
    filtered = []

    for segment in segments:
        if segment.num_frames < config.min_frames:
            print(f"  Filtered out segment frame_ids [{first_frame_id + segment.start_idx}, {first_frame_id + segment.end_idx}): "
                       f"too few frames ({segment.num_frames} < {config.min_frames})")
            continue

        if segment.p50_delta < config.min_p50_delta:
            print(f"  Filtered out segment frame_ids [{first_frame_id + segment.start_idx}, {first_frame_id + segment.end_idx}): "
                       f"p50 delta too low ({segment.p50_delta:.6f} < {config.min_p50_delta})")
            continue

        if config.max_p50_delta > 0 and segment.p50_delta > config.max_p50_delta:
            print(f"  Filtered out segment frame_ids [{first_frame_id + segment.start_idx}, {first_frame_id + segment.end_idx}): "
                       f"p50 delta too high ({segment.p50_delta:.6f} > {config.max_p50_delta}), "
                       f"median speed ~{segment.p50_delta * 5:.2f} u/s assuming 5fps")
            continue

        filtered.append(segment)
        print(f"  Kept segment frame_ids [{first_frame_id + segment.start_idx}, {first_frame_id + segment.end_idx}): "
                   f"{segment.num_frames} frames, p50_delta={segment.p50_delta:.6f}")

    if len(filtered) == 0:
        raise ValueError("No valid segments after applying filter criteria")

    print(f"Kept {len(filtered)}/{len(segments)} segments after filtering")
    return filtered


def load_camera_poses_txt(parent_dir: str) -> List[str]:
    """Load camera poses from camera_poses.txt as raw lines.

    Args:
        parent_dir: Path to parent group directory

    Returns:
        List of pose lines (strings)
    """
    poses_path = os.path.join(parent_dir, "camera_poses.txt")

    with open(poses_path, 'r') as f:
        poses = [line for line in f if line.strip()]

    print(f"Loaded {len(poses)} pose lines from {poses_path}")
    return poses


def load_intrinsics_txt(parent_dir: str) -> List[str]:
    """Load camera intrinsics from intrinsics.txt as raw lines.

    Args:
        parent_dir: Path to parent group directory

    Returns:
        List of intrinsic lines (strings)
    """
    intrinsics_path = os.path.join(parent_dir, "intrinsic.txt")

    with open(intrinsics_path, 'r') as f:
        intrinsics = [line for line in f if line.strip()]

    print(f"Loaded {len(intrinsics)} intrinsic lines from {intrinsics_path}")
    return intrinsics


def write_camera_poses_txt(segment_dir: str, poses: List[str]):
    """Write camera poses to camera_poses.txt.

    Args:
        segment_dir: Path to segment directory
        poses: List of pose lines (strings)
    """
    poses_path = os.path.join(segment_dir, "camera_poses.txt")

    with open(poses_path, 'w') as f:
        f.writelines(poses)

    print(f"Wrote {len(poses)} poses to {poses_path}")


def write_absolute_poses_json(segment_dir: str, abs_poses: List[Dict]):
    """Write absolute poses to camera_absolute_poses.json.

    Args:
        segment_dir: Path to segment directory
        abs_poses: List of absolute pose dicts
    """
    poses_path = os.path.join(segment_dir, "camera_absolute_poses.json")

    with open(poses_path, 'w') as f:
        json.dump(abs_poses, f, indent=2)

    print(f"Wrote {len(abs_poses)} absolute poses to {poses_path}")


def write_intrinsics_txt(segment_dir: str, intrinsics: List[str]):
    """Write intrinsics to intrinsics.txt.

    Args:
        segment_dir: Path to segment directory
        intrinsics: List of intrinsic lines (strings)
    """
    intrinsics_path = os.path.join(segment_dir, "intrinsic.txt")

    with open(intrinsics_path, 'w') as f:
        f.writelines(intrinsics)

    print(f"Wrote {len(intrinsics)} intrinsics to {intrinsics_path}")


def parse_pose_lines_to_numpy(pose_lines: List[str]) -> List[np.ndarray]:
    """Parse pose lines (strings) into numpy arrays.

    Args:
        pose_lines: List of pose lines from camera_poses.txt

    Returns:
        List of 4x4 numpy arrays
    """
    poses = []
    for line in pose_lines:
        numbers = list(map(float, line.strip().split()))
        if len(numbers) == 16:
            poses.append(np.array(numbers).reshape(4, 4))
    return poses


def parse_intrinsic_lines_to_numpy(intrinsic_lines: List[str]) -> List[np.ndarray]:
    """Parse intrinsic lines (strings) into numpy arrays.

    Args:
        intrinsic_lines: List of intrinsic lines from intrinsics.txt

    Returns:
        List of 3x3 numpy arrays
    """
    intrinsics = []
    for line in intrinsic_lines:
        fx, fy, cx, cy = map(float, line.strip().split())
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        intrinsics.append(K)
    return intrinsics


def write_camera_poses_ply(segment_dir: str, poses: List[np.ndarray],
                           intrinsics: List[np.ndarray], frustum_scale: float = 0.1):
    """Write camera frustum visualization to camera_poses.ply.

    Args:
        segment_dir: Path to segment directory
        poses: List of 4x4 C2W matrices
        intrinsics: List of 3x3 intrinsic matrices
        frustum_scale: Depth of frustum base in scene units
    """
    # Reuse color scheme (simple single color for all cameras in segment)
    color = [255, 0, 0]  # Red for all cameras

    ply_vertices = []  # (x, y, z, r, g, b)
    ply_edges = []     # (v1_idx, v2_idx)
    vertex_offset = 0

    for pose, intrinsic in zip(poses, intrinsics):
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]

        # Camera center (apex of frustum)
        cam_center = pose[:3, 3]
        ply_vertices.append((*cam_center, *color))

        # Four corners of image plane at depth=frustum_scale
        corners_img = np.array([
            [0, 0],      # top-left
            [cx * 2, 0], # top-right (assuming image width = 2*cx)
            [cx * 2, cy * 2], # bottom-right
            [0, cy * 2]  # bottom-left
        ])

        for corner in corners_img:
            u, v = corner
            # Unproject to camera space
            x_cam = (u - cx) / fx * frustum_scale
            y_cam = (v - cy) / fy * frustum_scale
            z_cam = frustum_scale
            point_cam = np.array([x_cam, y_cam, z_cam, 1.0])

            # Transform to world space
            point_world = pose @ point_cam
            ply_vertices.append((*point_world[:3], *color))

        # Edges: apex to each corner, and rectangle around corners
        apex_idx = vertex_offset
        for i in range(1, 5):
            ply_edges.append((apex_idx, vertex_offset + i))
        for i in range(1, 5):
            ply_edges.append((vertex_offset + i, vertex_offset + (i % 4) + 1))

        vertex_offset += 5

    # Write PLY file
    ply_path = os.path.join(segment_dir, "camera_poses.ply")

    with open(ply_path, 'w') as f:
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
            f.write(f"{v[0]} {v[1]} {v[2]} {int(v[3])} {int(v[4])} {int(v[5])}\n")

        for e in ply_edges:
            f.write(f"{e[0]} {e[1]}\n")

    print(f"Wrote camera frustum PLY to {ply_path}")


def copy_segment_b2nz_files(parent_dir: str, segment_dir: str,
                           segment_abs_poses: List[Dict]):
    """Copy this segment's per-frame depth/conf result files from the parent.

    The parent group's per-frame results live in <parent_dir>/results_output/
    as frame_<frame_id>.b2nz, where <frame_id> matches the "frame_id" field of
    camera_absolute_poses.json (both use the same first_frame_offset). Only the
    frames belonging to this segment are copied, not the whole parent dir.

    Args:
        parent_dir: Path to parent group directory
        segment_dir: Path to segment directory to create
        segment_abs_poses: Absolute pose dicts for this segment (each has frame_id)
    """
    parent_results_dir = os.path.join(parent_dir, "results_output")
    if not os.path.isdir(parent_results_dir):
        print(f"  Skipped results_output copy (not found in {parent_dir})")
        return

    segment_results_dir = os.path.join(segment_dir, "results_output")
    os.makedirs(segment_results_dir, exist_ok=True)

    copied = 0
    missing = 0
    for pose in segment_abs_poses:
        frame_id = pose["frame_id"]
        filename = frame_filename(frame_id)
        src = os.path.join(parent_results_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(segment_results_dir, filename))
            copied += 1
        else:
            missing += 1

    print(f"  Copied {copied} frame files to {segment_results_dir}" +
          (f" ({missing} missing)" if missing else ""))


def write_segment_info(segment_dir: str, segment: SegmentInterval,
                       first_pose: Dict, parent_group_info: Dict):
    """Write segment metadata to group_info.json.

    Args:
        segment_dir: Path to segment directory
        segment: SegmentInterval with frame range
        first_pose: First pose dict for frame ID offset
        parent_group_info: Parent group info for reference
    """
    # Get absolute frame IDs
    first_frame_id = first_pose["frame_id"]

    info = {
        "parent_group": {
            "input_video_clip": parent_group_info["input_video_clip"],
            "group_idx": parent_group_info["group_idx"],
            "chunk_ids": parent_group_info["chunk_ids"],
            "num_chunks": parent_group_info["num_chunks"],
        },
        "segment_frame_range": {
            "start_idx": segment.start_idx,
            "end_idx": segment.end_idx,
        },
        "output_group": {
            "first_frame_id_inclusive": first_frame_id,
            "last_frame_id_exclusive": first_frame_id + segment.num_frames,
            "num_frames": segment.num_frames,
        },
        "quality_metrics": {
            "p50_delta": segment.p50_delta,
        }
    }

    info_path = os.path.join(segment_dir, "group_info.json")
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)

    print(f"Wrote segment info to {info_path}")


def create_segment_output(parent_dir: str, segment_dir: str,
                          segment: SegmentInterval,
                          parent_poses: List[str],
                          parent_abs_poses: List[Dict],
                          parent_intrinsics: List[str],
                          save_depth_conf_result: bool = False):
    """Create segment directory and write output files.

    Note: pcd/ directory is NOT copied here - it will be handled later in
    merge_point_clouds() after individual PLY files are merged.

    Args:
        parent_dir: Path to parent group directory
        segment_dir: Path to segment directory to create
        segment: SegmentInterval defining frame range
        parent_poses: Pre-loaded parent camera poses
        parent_abs_poses: Pre-loaded parent absolute poses
        parent_intrinsics: Pre-loaded parent intrinsics
        save_depth_conf_result: If True, copy this segment's per-frame depth/conf
            result files from the parent's results_output/ directory
    """
    os.makedirs(segment_dir, exist_ok=True)
    print(f"Creating segment output in {segment_dir}")

    # 1. Copy files from parent (if they exist), but NOT pcd/ directory
    files_to_copy = [
        'base_config.yaml',
        'loop_closures.txt',
        'sim3_opt_result.png'
    ]
    for filename in files_to_copy:
        src = os.path.join(parent_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, segment_dir)
            print(f"  Copied {filename}")
        else:
            print(f"  Skipped {filename} (not found)")

    # 2. Extract subset for this segment
    segment_poses = parent_poses[segment.start_idx:segment.end_idx]
    segment_abs_poses = parent_abs_poses[segment.start_idx:segment.end_idx]
    segment_intrinsics = parent_intrinsics[segment.start_idx:segment.end_idx]

    print(f"  Extracted {len(segment_poses)} frames for segment [{segment.start_idx}, {segment.end_idx})")

    # 3. Write output files for segment
    write_camera_poses_txt(segment_dir, segment_poses)
    write_absolute_poses_json(segment_dir, segment_abs_poses)
    write_intrinsics_txt(segment_dir, segment_intrinsics)

    # Parse string lines to numpy arrays for PLY generation
    segment_poses_np = parse_pose_lines_to_numpy(segment_poses)
    segment_intrinsics_np = parse_intrinsic_lines_to_numpy(segment_intrinsics)
    write_camera_poses_ply(segment_dir, segment_poses_np, segment_intrinsics_np)

    # 3b. Copy this segment's per-frame depth/conf result files from the parent
    if save_depth_conf_result:
        copy_segment_b2nz_files(parent_dir, segment_dir, segment_abs_poses)

    # 4. Load parent group_info.json for reference
    parent_info_path = os.path.join(parent_dir, "group_info.json")
    if os.path.exists(parent_info_path):
        with open(parent_info_path, 'r') as f:
            parent_group_info = json.load(f)
    else:
        parent_group_info = {}

    # 5. Write segment info
    write_segment_info(segment_dir, segment, segment_abs_poses[0], parent_group_info)


def split_group_into_segments(group_dir: str, config: DeltaSplitConfig,
                              save_depth_conf_result: bool = False) -> List[str]:
    """Split a group into segments based on pose delta criteria.

    This is the main entry point for delta-based segment splitting.

    Args:
        group_dir: Path to group directory
        config: DeltaSplitConfig with thresholds
        save_depth_conf_result: If True, copy each segment's per-frame depth/conf
            result files from the parent group's results_output/ directory

    Returns:
        List of segment directory paths created

    Raises:
        ValueError: If no valid segments remain after filtering
        FileNotFoundError: If required input files are missing
    """
    print(f"Starting delta-based segment splitting for {group_dir}")

    # 1. Load absolute poses from camera_absolute_poses.json
    abs_poses = load_absolute_poses(group_dir)
    first_frame_id = abs_poses[0]["frame_id"]
    # print(f"First frame_id: {first_frame_id}, total poses: {len(poses)}")

    # 2. Load parent output files once for all segments
    try:
        parent_c2w_poses = load_camera_poses_txt(group_dir)
        parent_intrinsics = load_intrinsics_txt(group_dir)
    except Exception as e:
        print(f"Failed to load parent output files: {e}")
        raise

    # 3. Compute deltas between consecutive frames
    deltas = compute_deltas(abs_poses)

    # 4. Find split points based on delta criteria
    split_indices = split_by_delta_criteria(deltas, config, first_frame_id)

    # 5. Convert split points to segments
    segments = create_segments_from_splits(split_indices, len(abs_poses), deltas, first_frame_id)

    # 6. Filter segments based on quality criteria (raises ValueError if none valid)
    filtered_segments = filter_segments(segments, config, first_frame_id)

    # 7. Create segment directories and output files
    parent_name = os.path.basename(group_dir)  # e.g., "group_0"
    parent_parent = os.path.dirname(group_dir)
    segment_dirs = []

    for segment_idx, segment in enumerate(filtered_segments):
        segment_name = f"{parent_name}_{segment_idx}"
        segment_dir = os.path.join(parent_parent, segment_name)

        print(f"Creating segment {segment_idx + 1}/{len(filtered_segments)}: {segment_name} "
              f"(frame_ids [{first_frame_id + segment.start_idx}, {first_frame_id + segment.end_idx}))")
        create_segment_output(group_dir, segment_dir, segment,
                              parent_c2w_poses, abs_poses, parent_intrinsics,
                              save_depth_conf_result=save_depth_conf_result)
        segment_dirs.append(segment_dir)

    # 8. Parent group is kept for now (will be deleted after PCD merge in geometry_utils.py)
    print(f"Successfully created {len(segment_dirs)} segments for {parent_name}")

    return segment_dirs
