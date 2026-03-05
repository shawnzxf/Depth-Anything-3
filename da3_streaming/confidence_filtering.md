## General Goal
We expect to use DA3 streaming to get camera intrinsics and extrinsics from a list of video frames. So the input is a list of video frames, and the outputs are camera intrinsics, extrinsics, and optional point clouds for visual validation.

## Issue
DA3 streaming relies on processing chunk by chunk. Each chunk's depth/confidence prediction is independent (via `process_single_chunk()`), but the alignment step connects chunks sequentially — so the accuracy of the global scene depends on every chunk-to-chunk alignment being correct.

However, DA3 can be unconfident about its depth prediction when video frames aim at objects far-away or textureless. When this happens, the alignment between a low-confidence chunk and its neighbors will be inaccurate, which in turn ruins all downstream alignments that build on top of it.

Specifically, the output within future chunks is locally accurate (camera intrinsics and extrinsics relative to the local scene are correct), but they are inaccurate in the global scene because the bad alignment at the low-confidence chunk propagates through the sequential chain. A typical case: the first half of the scene is accurate and so is the second half, but they are connected incorrectly at the low-confidence chunk, ruining the whole scene. There can be more than one low-confidence region, so this issue may compound.

## Task
We need to solve this issue where predictions for local scenes are good, but the combined prediction for the whole scene is not.

## Priority
- Accurate extrinsics and intrinsics relative to the env is a HARD requirement
- One global scene is not a hard requirement, so no need to put them together if not confident

## Notes and Observations
- The frame-level average confidence score tends to be low (< 0.01) across multiple consecutive frames, not just one isolated frame. This is why the 20th percentile aggregation works well for chunk-level scoring.
- Threshold selection (static vs. adaptive) is an open item. Initial approach: use a fixed threshold and tune empirically on representative sequences.

## Proposal

### Core Idea
Split the chunk sequence at low-confidence boundaries. Process each resulting group independently — each group gets its own coordinate frame, its own alignment, and its own loop correction. Groups are never stitched together (per the priority that one global scene is not required).

### Phase 1 — Inference (all chunks, independently)
Run `process_single_chunk()` for ALL chunks upfront, outside the alignment loop. Each chunk's inference (depth, confidence, intrinsics/extrinsics prediction) is independent of other chunks — confirmed by inspecting the code: the model call within `process_single_chunk()` takes only the current chunk's images and produces predictions without referencing any other chunk's state.

This gives us predictions and per-pixel confidence scores for every chunk before any alignment happens.

### Phase 2 — Confidence-based grouping
Compute a chunk-level confidence score:
1. For each frame in the chunk, compute the average of its per-pixel confidence values (spatial mean)
2. Take the 20th percentile of these frame-level averages as the chunk's confidence score

This accounts for the observation that low-confidence regions tend to span multiple consecutive frames — the 20th percentile captures this without being thrown off by a single bad frame.

If a chunk's confidence score falls below a threshold, mark it as low-confidence. Partition the chunks into groups of consecutive confident chunks, dropping all low-confidence chunks. For example, if chunks 0-4 are confident, chunks 5-7 are low-confidence, and chunks 8-12 are confident, we get two groups: [0, 4] and [8, 12].

### Phase 3 — Per-group alignment, loop detection, and correction
For each group independently:
1. Run sequential alignment between consecutive chunks (using overlap regions)
2. Run loop detection within the group
3. Run loop correction/optimization within the group

Each group produces extrinsics/intrinsics in its own coordinate frame. Loop detection operates only within each group — cross-group loops are not attempted since the groups do not share a coordinate frame.

### Output
The final output is a list of chunk groups, each with its own set of aligned extrinsics and intrinsics. Downstream consumers receive per-group results rather than a single global scene.

### Implementation Notes
- The current da3_streaming.py has a clear separation between phase 1 vs phase 3. But there are still some attributes shared across phases. So we need to insert a phase 2 in between, reset the attributes before phase 3 (potentially inside the self._prepare_for_alignment()) for each of the group and then run phase 3 for each group.
  - a key config to update is the self.output_dir because we need to output to a different sub-dir for each group
  - we also need to update the self.img_list for each group because it is used by the loop detector and we don't want cross-group loop detection
  - we might also need to update the self.chunk_indices for each group, please double check 
- For phase 2, please start a new sub-section for DA3 confidence threshold and percentile config, please also set a config for min num of chunks in a group so we can filter out group that is too small
- Please prioritize clear accurate code structure over performance for now, like we are fine to load the loop detector for each group
