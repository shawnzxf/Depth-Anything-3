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

import json
import os

import numpy as np


def save_depth_conf_result(predictions, chunk_idx, s, R, T,
                           chunk_indices, overlap_s, overlap_e,
                           result_output_dir, config, first_frame_offset=0):
    """Save per-frame depth, confidence, and intrinsics to compressed .npz files.

    Args:
        predictions: object with .processed_images, .depth, .conf, .intrinsics, .extrinsics.
        chunk_idx: int -- which chunk these predictions belong to.
        s, R, T: SIM(3) transform components (scale, rotation, translation).
        chunk_indices: list of (start, end) tuples.
        overlap_s: int -- overlap start offset.
        overlap_e: int -- overlap end offset.
        result_output_dir: str -- directory path for output .npz files.
        config: dict.
        first_frame_offset: int -- absolute video first frame id + relative group first frame id
    """
    if not config["Model"]["save_depth_conf_result"]:
        return
    os.makedirs(result_output_dir, exist_ok=True)

    chunk_start, chunk_end = chunk_indices[chunk_idx]

    if chunk_idx == 0:
        save_indices = list(range(0, chunk_end - chunk_start - overlap_e))
    elif chunk_idx == len(chunk_indices) - 1:
        save_indices = list(range(overlap_s, chunk_end - chunk_start))
    else:
        save_indices = list(range(overlap_s, chunk_end - chunk_start - overlap_e))

    print("[save_depth_conf_result] save_indices:")

    for local_idx in save_indices:
        global_idx = first_frame_offset + chunk_start + local_idx
        print(f"{global_idx}, ", end="")

        image = predictions.processed_images[local_idx]  # [H, W, 3] uint8
        depth = predictions.depth[local_idx]  # [H, W] float32
        conf = predictions.conf[local_idx]  # [H, W] float32
        intrinsics = predictions.intrinsics[local_idx]  # [3, 3] float32

        filename = f"frame_{global_idx}.npz"
        filepath = os.path.join(result_output_dir, filename)

        if config["Model"]["save_debug_info"]:
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


def _save_frustum_ply(all_poses, all_intrinsics, all_chunk_ids,
                      output_dir, frustum_scale=0.1):
    """Save camera frustum pyramids as an ASCII PLY file with per-chunk colors.

    Each camera is represented by a 5-vertex pyramid: the apex at the camera
    center and 4 base corners obtained by un-projecting the image corners to
    world space at the given *frustum_scale* depth.

    Args:
        all_poses: list of 4x4 C2W matrices.
        all_intrinsics: list of 3x3 intrinsic matrices.
        all_chunk_ids: list of int chunk IDs (used for coloring).
        output_dir: str -- directory for the output .ply file.
        frustum_scale: float -- depth of the frustum base in scene units.
    """
    chunk_colors = [
        [255, 0, 0],      # Red
        [0, 255, 0],      # Green
        [0, 0, 255],      # Blue
        [255, 255, 0],    # Yellow
        [255, 0, 255],    # Magenta
        [0, 255, 255],    # Cyan
        [128, 0, 0],      # Dark Red
        [0, 128, 0],      # Dark Green
        [0, 0, 128],      # Dark Blue
        [128, 128, 0],    # Olive
    ]
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
        R_mat = pose[:3, :3]

        # 4 image corner rays in camera space (principal-point-aware),
        # scaled to frustum_scale depth.  W ≈ 2*cx, H ≈ 2*cy.
        corners_cam = np.array([
            [-cx / fx, -cy / fy, 1.0],  # top-left  (1)
            [ cx / fx, -cy / fy, 1.0],  # top-right (2)
            [ cx / fx,  cy / fy, 1.0],  # bot-right (3)
            [-cx / fx,  cy / fy, 1.0],  # bot-left  (4)
        ]) * frustum_scale

        # Transform corners to world space
        corners_world = (R_mat @ corners_cam.T).T + center

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

    ply_path = os.path.join(output_dir, "camera_poses.ply")
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


def save_camera_poses(all_camera_poses, all_camera_intrinsics, sim3_list,
                      num_frames, overlap_s, overlap_e, output_dir,
                      first_frame_offset=0):
    """Save camera poses from all chunks to txt and ply files.

    - txt file: Each line contains a 4x4 C2W matrix flattened into 16 numbers
    - ply file: Camera poses visualized as frustum pyramids with per-chunk colors

    Args:
        all_camera_poses: list of ((start, end), extrinsics) tuples.
        all_camera_intrinsics: list of ((start, end), intrinsics) tuples.
        sim3_list: list of (s, R, t) tuples (already accumulated).
        num_frames: int -- total number of frames (len(img_list)).
        overlap_s: int -- overlap start offset.
        overlap_e: int -- overlap end offset.
        output_dir: str -- directory for output files.
    """
    print("Saving all camera poses to txt file...")

    all_poses = [None] * num_frames
    all_intrinsics_out = [None] * num_frames
    all_chunk_ids = [0] * num_frames

    first_chunk_range, first_chunk_extrinsics = all_camera_poses[0]
    _, first_chunk_intrinsics = all_camera_intrinsics[0]

    for i, idx in enumerate(
        range(first_chunk_range[0], first_chunk_range[1] - overlap_e)
    ):
        w2c = np.eye(4)
        w2c[:3, :] = first_chunk_extrinsics[i]
        c2w = np.linalg.inv(w2c)
        all_poses[idx] = c2w
        all_intrinsics_out[idx] = first_chunk_intrinsics[i]
        all_chunk_ids[idx] = 0

    for chunk_idx in range(1, len(all_camera_poses)):
        chunk_range, chunk_extrinsics = all_camera_poses[chunk_idx]
        _, chunk_intrinsics = all_camera_intrinsics[chunk_idx]
        s, R, t = sim3_list[
            chunk_idx - 1
        ]  # When call save_camera_poses(), all the sim3 are aligned to the first chunk.

        S = np.eye(4)
        S[:3, :3] = s * R
        S[:3, 3] = t

        chunk_range_end = (
            chunk_range[1] - overlap_e
            if chunk_idx < len(all_camera_poses) - 1
            else chunk_range[1]
        )

        for i, idx in enumerate(range(chunk_range[0] + overlap_s, chunk_range_end)):
            w2c = np.eye(4)
            w2c[:3, :] = chunk_extrinsics[i + overlap_s]
            c2w = np.linalg.inv(w2c)

            transformed_c2w = S @ c2w  # Be aware of the left multiplication!
            transformed_c2w[:3, :3] /= s  # Normalize rotation

            all_poses[idx] = transformed_c2w
            all_intrinsics_out[idx] = chunk_intrinsics[i + overlap_s]
            all_chunk_ids[idx] = chunk_idx

    poses_path = os.path.join(output_dir, "camera_poses.txt")
    with open(poses_path, "w") as f:
        for pose in all_poses:
            flat_pose = pose.flatten()
            f.write(" ".join([str(x) for x in flat_pose]) + "\n")
    print(f"Camera poses saved to {poses_path}")

    save_c2w_as_absolute_poses(all_poses, output_dir, first_frame_offset=first_frame_offset)

    intrinsics_path = os.path.join(output_dir, "intrinsic.txt")
    with open(intrinsics_path, "w") as f:
        for intrinsic in all_intrinsics_out:
            fx = intrinsic[0, 0]
            fy = intrinsic[1, 1]
            cx = intrinsic[0, 2]
            cy = intrinsic[1, 2]
            f.write(f"{fx} {fy} {cx} {cy}\n")
    print(f"Camera intrinsics saved to {intrinsics_path}")

    _save_frustum_ply(all_poses, all_intrinsics_out, all_chunk_ids, output_dir)


def save_c2w_as_absolute_poses(all_poses_in_c2w, output_dir, need_sanity_check=False, first_frame_offset=0):
    """Convert all_poses_in_c2w (a list of 4x4 cam-to-world matrices) to absolute poses in a
    ground-plane coordinate system compatible with public_compute_deltas_from_two_frames.

    DA3/OpenCV camera convention: +X=right, +Y=down, +Z=forward.
    The DA3 world ground plane is XZ with Y roughly vertical (down).

    Target convention (matching public_compute_deltas_from_two_frames):
    - x, y = ground plane coordinates
    - z    = vertical (up)
    - yaw  = heading on ground plane (radians), such that body +x=forward, +y=right

    Coordinate mapping:
    target_x =  da3_x
    target_y = -da3_z
    target_z = -da3_y  (up)
    yaw      = atan2(-R[2,2], R[0,2])  (from camera forward projected onto ground plane)
    """

    # Read camera poses (each line = 16 floats of a 4x4 camera-to-world matrix)
    poses = all_poses_in_c2w
    print(f"Read {len(poses)} poses")

    # Convert each pose to target coordinate system
    result = []
    for idx, pose in enumerate(poses):
        frame_id = first_frame_offset + idx
        R = pose[:3, :3]  # 3x3 rotation (camera-to-world)
        t = pose[:3, 3]   # translation (camera position in world)

        # Remap position: target_x = da3_x, target_y = -da3_z, target_z = -da3_y
        target_x = t[0]
        target_y = -t[2]
        target_z = -t[1]  # height (up)

        # Camera forward in DA3 world = R[:, 2] (third column)
        # Projected onto target ground plane: (R[0,2], -R[2,2])
        yaw = np.arctan2(-R[2, 2], R[0, 2])

        result.append(
            {
                "frame_id": frame_id,
                "rotation": {"yaw": float(yaw)},
                "translation": {
                    "x": float(target_x),
                    "y": float(target_y),
                    "z": float(target_z),
                },
            }
        )

    poses_path = os.path.join(output_dir, "camera_absolute_poses.json")
    with open(poses_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Camera absolute poses saved to {poses_path}")

    if need_sanity_check:
        print("\nFirst 10 poses:")
        for i, entry in enumerate(result[:10]):
            t = entry["translation"]
            r = entry["rotation"]
            print(
                f"  Frame {i}: x={t['x']:.4f}, y={t['y']:.4f}, z={t['z']:.4f}, yaw={r['yaw']:.4f} rad"
            )

        # Verify: compute body-frame deltas for consecutive pose pairs
        from scipy.spatial.transform import Rotation as Rot

        n_deltas = min(10, len(result) - 1)
        print(f"\nBody-frame deltas for first {n_deltas} consecutive pairs:")
        for i in range(n_deltas):
            p0, p1 = result[i], result[i + 1]
            dx_w = p1["translation"]["x"] - p0["translation"]["x"]
            dy_w = p1["translation"]["y"] - p0["translation"]["y"]
            dz_w = p1["translation"]["z"] - p0["translation"]["z"]
            rot0 = Rot.from_euler("z", p0["rotation"]["yaw"], degrees=False)
            rot1 = Rot.from_euler("z", p1["rotation"]["yaw"], degrees=False)
            body = rot0.inv().apply([dx_w, dy_w, dz_w])
            drot = (rot1 * rot0.inv()).as_matrix()
            dyaw = np.arctan2(drot[1, 0], drot[0, 0])
            label = f"{i}->{i+1}:"
            print(
                f"  {label:<7} "
                f"dx(forward)={body[0]:>10.6f},  "
                f"dy(right)={body[1]:>10.6f},  "
                f"dz(upward)={body[2]:>10.6f},  "
                f"dyaw(clockwise rotate)={dyaw:>10.6f} rad"
            )


def write_group_info(group_output_dir, group_idx, group_chunk_ids,
                     all_chunk_indices, img_dir, first_frame, last_frame):
    """Write a JSON info file for a single group into its output dir.

    Args:
        group_output_dir: str -- output directory for this group.
        group_idx: int -- group index.
        group_chunk_ids: list of int -- chunk indices in this group.
        all_chunk_indices: list of (start, end) tuples.
        img_dir: str -- path to image directory.
        first_frame: int -- global first frame ID.
        last_frame: int -- global last frame ID (exclusive).
    """
    frame_start = all_chunk_indices[group_chunk_ids[0]][0]
    frame_end = all_chunk_indices[group_chunk_ids[-1]][1]
    info = {
        "input_video_clip": {
            "video_frame_dir": img_dir,
            "first_frame_id_inclusive": first_frame,
            "last_frame_id_exclusive": last_frame,
        },
        "group_idx": group_idx,
        "chunk_ids": group_chunk_ids,
        "num_chunks": len(group_chunk_ids),
        "output_group": {
            "first_frame_id_inclusive": frame_start + first_frame,
            "last_frame_id_exclusive": frame_end + first_frame,
            "num_frames": frame_end - frame_start,
        },
    }
    info_path = os.path.join(group_output_dir, "group_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"Group info saved to {info_path}")
