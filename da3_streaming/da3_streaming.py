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

import numpy as np
import torch
from loop_utils.config_utils import load_config
from loop_utils.loop_detector import LoopDetector
from loop_utils.sim3loop import Sim3LoopOptimizer
from loop_utils.sim3utils import process_loop_list, warmup_numba
from safetensors.torch import load_file

from depth_anything_3.api import DepthAnything3

# Local modules
from geometry_utils import (
    copy_file,
    depth_to_point_cloud_vectorized,  # re-exported for npz_output_process.py
    merge_point_clouds,
    remove_duplicates,
    timing,
)
from confidence import (
    aggregate_pixel_to_frame_confidence,
    compute_chunk_confidence_groups,
)
from alignment import (
    align_2pcds as _align_2pcds,
    apply_alignment,
    get_loop_sim3_from_loop_predict,
    plot_loop_closure,
    run_cross_chunk_alignment,
)
from output import (
    save_camera_poses,
    save_depth_conf_result,
    write_group_info,
)


class DA3_Streaming:
    @timing
    def __init__(self, image_dir, save_dir, config, first_frame=None, last_frame=None, config_path=None):
        self.config = config
        self.config_path = config_path
        self.first_frame = first_frame if first_frame is not None else 0
        self.last_frame = last_frame
        self._group_frame_start = 0

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

    # -- Confidence filtering (delegates to confidence.py) --

    def _aggregate_pixel_to_frame_confidence(self, conf):
        return aggregate_pixel_to_frame_confidence(conf, self.config)

    def _compute_chunk_confidence_groups(self):
        return compute_chunk_confidence_groups(
            self.chunk_indices, self.chunk_frame_confidence_flags,
            self.first_frame, self.config,
        )

    # -- Chunk file path resolution --

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

        # Store group frame offset so save_depth_conf_result can compute
        # correct global frame indices (chunk_indices will be remapped to 0-based below).
        self._group_frame_start = group_frame_start

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

    # -- Loop closure --

    def get_loop_pairs(self):
        loop_info_save_path = os.path.join(self.output_dir, "loop_closures.txt")
        self.loop_detector.run(image_paths=self.img_list, output=loop_info_save_path)
        loop_list = self.loop_detector.get_loop_list()
        return loop_list

    # -- Output (delegates to output.py) --

    def _save_depth_conf_result(self, predictions, chunk_idx, s, R, T):
        save_depth_conf_result(
            predictions, chunk_idx, s, R, T,
            self.chunk_indices, self.overlap_s, self.overlap_e,
            self.result_output_dir, self.config,
            # abs video clip first frame + relative group first frame
            first_frame_offset=self.first_frame + self._group_frame_start,
        )

    def _save_camera_poses(self):
        save_camera_poses(
            self.all_camera_poses, self.all_camera_intrinsics,
            self.sim3_list, len(self.img_list),
            self.overlap_s, self.overlap_e, self.output_dir,
            first_frame_offset=self.first_frame + self._group_frame_start,
        )

    def _write_group_info(self, group_output_dir, group_idx, group_chunk_ids, all_chunk_indices):
        write_group_info(
            group_output_dir, group_idx, group_chunk_ids,
            all_chunk_indices, self.img_dir,
            self.first_frame, self.last_frame,
        )

    # -- Inference --

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

    # -- Alignment (delegates to alignment.py) --

    def align_2pcds(self, point_map1, conf1, point_map2, conf2,
                    chunk1_depth, chunk2_depth,
                    chunk1_depth_conf, chunk2_depth_conf):
        return _align_2pcds(
            point_map1, conf1, point_map2, conf2,
            chunk1_depth, chunk2_depth,
            chunk1_depth_conf, chunk2_depth_conf,
            self.config,
        )

    def _run_cross_chunk_alignment(self):
        self.sim3_list = run_cross_chunk_alignment(
            self.chunk_indices, self.overlap,
            self._get_chunk_file_path, self.config,
        )

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

        self.loop_sim3_list = get_loop_sim3_from_loop_predict(
            self.loop_predict_list, self.chunk_indices,
            self._get_chunk_file_path, self.config,
        )

        input_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(
            self.sim3_list
        )  # just for plot
        self.sim3_list = self.loop_optimizer.optimize(self.sim3_list, self.loop_sim3_list)
        optimized_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(
            self.sim3_list
        )  # just for plot

        plot_loop_closure(
            input_abs_poses, optimized_abs_poses,
            self.loop_sim3_list, self.output_dir,
            save_name="sim3_opt_result.png",
        )

    @timing
    def _apply_alignment(self):
        self.sim3_list = apply_alignment(
            self.chunk_indices, self.sim3_list,
            self._get_chunk_file_path, self.result_aligned_dir,
            self.pcd_dir, self.config,
            save_depth_conf_fn=self._save_depth_conf_result,
        )

    # -- Pipeline orchestration --

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

        self._save_camera_poses()

    def _load_image_list(self, first_frame=None, last_frame=None):
        """Load and optionally filter the sorted image list from self.img_dir.

        Args:
            first_frame: Optional integer frame ID of the first frame to include (inclusive).
                         Derives filename as "%06d.png", e.g. 100 -> "000100.png".
            last_frame: Optional integer frame ID of the last frame to include (exclusive).
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
                end_idx = basenames.index(last_fname)
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
                        help="Frame ID of the last frame to process (exclusive), e.g. 200")
    args = parser.parse_args()

    frame_range = f" frames [{args.first_frame}, {args.last_frame})" if args.first_frame is not None or args.last_frame is not None else ""
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
    print(f"The exp will be saved under dir: {save_dir}")

    if config["Model"]["align_lib"] == "numba":
        warmup_numba()

    da3_streaming = DA3_Streaming(image_dir, save_dir, config, first_frame=args.first_frame, last_frame=args.last_frame, config_path=args.config)
    da3_streaming.run()
    da3_streaming.close()

    del da3_streaming
    torch.cuda.empty_cache()
    gc.collect()

    merge_point_clouds(save_dir, delete_after_merge=True)

    main_time = time.time() - start_time_main
    print(f"[Timing] total main execution: {main_time:.3f}s")

    print(f"\n########## [Video] END — {args.image_dir}{frame_range} ##########")

    print("DA3-Streaming done.")
    sys.exit()
