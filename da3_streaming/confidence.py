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

import cv2
import numpy as np


def apply_mask_to_confidence(conf, image_paths, mask_dir):
    """Zero out confidence where the external binary mask is 255.

    For each frame, loads the corresponding mask (uint8, values 0 or 255)
    from *mask_dir* (same basename as the image file).  Pixels where the
    mask is 255 get their confidence set to 0.0; pixels where the mask is 0
    keep their original confidence.

    Args:
        conf: np.ndarray of shape [N, H, W] -- pixel-wise confidence scores.
        image_paths: list of str -- full paths of the chunk's input images
            (used to derive mask filenames).
        mask_dir: str -- directory that contains binary mask images (uint8,
            0 or 255), one per frame, with the same basename as the
            corresponding input image.

    Returns:
        np.ndarray of shape [N, H, W] -- masked confidence scores.
    """
    conf = conf.copy()
    h, w = conf.shape[1], conf.shape[2]
    for i, img_path in enumerate(image_paths):
        mask_path = os.path.join(mask_dir, os.path.basename(img_path))
        if not os.path.exists(mask_path):
            print(f"[Warning] Mask not found for frame: {mask_path}")
            continue
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[Warning] Failed to read mask: {mask_path}")
            continue
        if mask.shape[0] != h or mask.shape[1] != w:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        conf[i][mask == 255] = 0.0
    return conf


def aggregate_pixel_to_frame_confidence(conf, config):
    """Aggregate pixel-wise confidence to per-frame boolean flags.

    For each frame, computes the ``pixel_to_frame_percentile``-th percentile
    of pixel-wise confidence values and compares it against
    ``frame_confidence_threshold``.

    Args:
        conf: np.ndarray of shape [N, H, W] with pixel-wise confidence.
        config: dict -- the full config; reads config["Confidence"].

    Returns:
        np.ndarray of shape [N] with dtype bool (True = selected).
    """
    conf_config = config.get("Confidence", {})
    percentile = conf_config.get("pixel_to_frame_percentile", 75)
    threshold = conf_config.get("frame_confidence_threshold", 0.2)

    frame_scores = np.percentile(conf, percentile, axis=(1, 2))
    return frame_scores >= threshold


def aggregate_frame_to_chunk_confidence(frame_flags, config):
    """Decide whether a chunk should be kept based on frame-level flags.

    A chunk is dropped if it contains a continuous streak of False
    (unconfident) frames whose length is at least
    ``drop_continuous_ratio`` of the chunk length.

    Args:
        frame_flags: np.ndarray of bool -- per-frame confidence flags for one chunk.
        config: dict -- the full config; reads config["Confidence"].

    Returns:
        bool: True if the chunk should be kept, False if it should be dropped.
    """
    conf_config = config.get("Confidence", {})
    drop_ratio = conf_config.get("drop_continuous_ratio", 0.2)

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


def compute_chunk_confidence_groups(chunk_indices, chunk_frame_confidence_flags,
                                    first_frame, config):
    """Compute chunk-level confidence and partition into groups
    of consecutive confident chunks.

    Args:
        chunk_indices: list of (start, end) tuples -- chunk frame ranges.
        chunk_frame_confidence_flags: list of np.ndarray -- per-chunk frame flags.
        first_frame: int -- global frame offset for logging.
        config: dict -- the full config; reads config["Confidence"].

    Returns:
        list of lists: Each inner list contains the original (global) chunk indices
                       that form one group. E.g., [[0,1,2,3,4], [8,9,10,11,12]]
    """
    conf_config = config.get("Confidence", {})
    if not conf_config.get("enable", True):
        print("[Confidence] Disabled — treating all chunks as one group.")
        return [list(range(len(chunk_indices)))]

    min_chunks = conf_config.get("min_chunks_per_group", 3)

    num_chunks = len(chunk_indices)
    chunk_keep = []

    for i in range(num_chunks):
        keep = aggregate_frame_to_chunk_confidence(
            chunk_frame_confidence_flags[i], config
        )
        chunk_keep.append(keep)
        flags = chunk_frame_confidence_flags[i]
        n_selected = int(np.sum(flags))
        local_start, local_end = chunk_indices[i]
        global_start = local_start + first_frame
        global_end = local_end + first_frame
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
