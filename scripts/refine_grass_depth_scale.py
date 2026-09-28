#!/usr/bin/env python3
"""Correct monocular depth with a robust spatial log-scale from sparse 3D tracks.

This uses the sparse-seed idea from MTD, not MTD's unreleased fitting code.
Tracks with IDs divisible by five are never used for fitting. Frames 0-20 are
the method-development pilot; later frames provide a stricter evaluation set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from grass_track_depth import image_and_anchors, read_colmap, undistort, write_ply


def sample_depth(
    depth: np.ndarray, xy: np.ndarray, width: int, height: int, preprocess: str
) -> tuple[np.ndarray, np.ndarray]:
    if preprocess == "pad_portrait_14":
        if height < width or depth.shape[0] != depth.shape[1]:
            raise ValueError("padded depth expects a square output from portrait input")
        resized_width = round(width / height * depth.shape[0] / 14) * 14
        pad_left = (depth.shape[1] - resized_width) // 2
        x = (xy[:, 0] + 0.5) * resized_width / width - 0.5 + pad_left
        y = (xy[:, 1] + 0.5) * depth.shape[0] / height - 0.5
    else:
        x = (xy[:, 0] + 0.5) * depth.shape[1] / width - 0.5
        y = (xy[:, 1] + 0.5) * depth.shape[0] / height - 0.5
    sampled = cv2.remap(
        depth,
        x.astype(np.float32).reshape(1, -1),
        y.astype(np.float32).reshape(1, -1),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    ).ravel()
    return np.stack([x, y], axis=1), sampled


def correction_features(xy: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.column_stack(
        [
            np.ones(len(xy)),
            (xy[:, 0] - width / 2) / width,
            (xy[:, 1] - height / 2) / height,
        ]
    )


def fuse_frame(
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    tracked_xy: np.ndarray,
    displacement: np.ndarray,
    pose: tuple[np.ndarray, np.ndarray, tuple[float, ...]],
    pixel_stride: int,
    preprocess: str,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape
    yy, xx = np.mgrid[0:h:pixel_stride, 0:w:pixel_stride]
    yy, xx = yy.ravel(), xx.ravel()
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    green = (hsv[yy, xx, 0] >= 18) & (hsv[yy, xx, 0] <= 95) & (hsv[yy, xx, 1] >= 28)
    good = green & mask[yy, xx] & np.isfinite(depth[yy, xx]) & (depth[yy, xx] > 0)
    xx, yy = xx[good], yy[good]
    if not len(xx) or not len(tracked_xy):
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)
    tree = cKDTree(tracked_xy)
    distance, neighbor = tree.query(
        np.stack([xx, yy], axis=1), k=min(4, len(tracked_xy))
    )
    if distance.ndim == 1:
        distance, neighbor = distance[:, None], neighbor[:, None]
    near = distance[:, 0] < 12
    xx, yy, distance, neighbor = xx[near], yy[near], distance[near], neighbor[near]
    if not len(xx):
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)
    weights = 1 / np.maximum(distance, 2) ** 2
    weights /= weights.sum(axis=1, keepdims=True)
    local_motion = np.sum(displacement[neighbor] * weights[..., None], axis=1)
    orig_w, orig_h = int(pose[2][0]), int(pose[2][1])
    if preprocess == "pad_portrait_14":
        resized_width = round(orig_w / orig_h * h / 14) * 14
        pad_left = (w - resized_width) // 2
        x_orig = (xx - pad_left + 0.5) * orig_w / resized_width - 0.5
        y_orig = (yy + 0.5) * orig_h / h - 0.5
    else:
        x_orig = (xx + 0.5) * orig_w / w - 0.5
        y_orig = (yy + 0.5) * orig_h / h - 0.5
    xu, yu = undistort(x_orig, y_orig, pose[2])
    z = depth[yy, xx]
    camera_xyz = np.stack([xu * z, yu * z, z], axis=1)
    world = (camera_xyz - pose[1]) @ pose[0] - local_motion
    return world.astype(np.float32), rgb[yy, xx]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--colmap-model", type=Path, required=True)
    parser.add_argument("--fit-state", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument(
        "--depth-input", "--moge-output", dest="depth_input", type=Path, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-name", choices=["moge3", "omega", "zipmap"], default="moge3"
    )
    parser.add_argument(
        "--preprocess", choices=["direct", "pad_portrait_14"], default="direct"
    )
    parser.add_argument("--fusion-every", type=int, default=4)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    if args.fusion_every < 1 or args.pixel_stride < 1:
        parser.error("fusion and pixel strides must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    poses = read_colmap(args.colmap_model)
    with np.load(args.fit_state, allow_pickle=False) as archive:
        state = {key: archive[key] for key in archive.files}
    with np.load(args.tracks, allow_pickle=False) as archive:
        tracks = {key: archive[key] for key in archive.files}
    rows, fused_xyz, fused_rgb = [], [], []
    depth_paths = sorted(args.depth_input.glob("[0-9][0-9][0-9][0-9][0-9].npz"))
    if args.max_frames is not None:
        depth_paths = depth_paths[: args.max_frames]
    if not depth_paths:
        raise FileNotFoundError("No numbered depth NPZ in input directory")
    for ordinal, path in enumerate(depth_paths):
        frame = int(path.stem) - 1
        if frame not in poses:
            raise ValueError(f"COLMAP pose missing for frame {frame}")
        bgr = cv2.imread(str(args.images / f"{frame + 1:05d}.jpg"))
        if bgr is None:
            raise FileNotFoundError(f"image missing for frame {frame}")
        height, width = bgr.shape[:2]
        pose = poses[frame]
        with np.load(path, allow_pickle=False) as data:
            global_depth = data["depth_sfm"]
            mask = data["mask"].astype(bool)
        xy_orig, z_sfm, displacement, anchor_ids = image_and_anchors(
            frame, bgr, state, tracks, pose
        )
        xy_small, sampled = sample_depth(
            global_depth, xy_orig, width, height, args.preprocess
        )
        sampled_mask = (
            cv2.remap(
                mask.astype(np.uint8),
                xy_small[:, 0].astype(np.float32).reshape(1, -1),
                xy_small[:, 1].astype(np.float32).reshape(1, -1),
                cv2.INTER_NEAREST,
            )
            .ravel()
            .astype(bool)
        )
        valid = sampled_mask & np.isfinite(sampled) & (sampled > 0) & (z_sfm > 0)
        train = valid & (anchor_ids % 5 != 4)
        hold = valid & (anchor_ids % 5 == 4)
        if train.sum() < 20 or hold.sum() < 5:
            raise ValueError(f"frame {frame}: too few calibration/holdout anchors")
        features = correction_features(
            xy_small, global_depth.shape[1], global_depth.shape[0]
        )
        log_residual = np.log(z_sfm[train] / sampled[train])
        beta = least_squares(
            lambda value: log_residual - features[train] @ value,
            [float(np.median(log_residual)), 0.0, 0.0],
            loss="soft_l1",
            f_scale=0.1,
        ).x
        yy, xx = np.mgrid[: global_depth.shape[0], : global_depth.shape[1]]
        factors = np.exp(
            beta[0]
            + beta[1] * (xx - global_depth.shape[1] / 2) / global_depth.shape[1]
            + beta[2] * (yy - global_depth.shape[0] / 2) / global_depth.shape[0]
        )
        factors = np.clip(factors, 0.5, 2.0)
        affine_depth = (global_depth * factors).astype(np.float32)
        _, sampled_affine = sample_depth(
            affine_depth, xy_orig, width, height, args.preprocess
        )
        baseline_err = np.abs(sampled[hold] - z_sfm[hold]) / z_sfm[hold]
        affine_err = np.abs(sampled_affine[hold] - z_sfm[hold]) / z_sfm[hold]
        row = {
            "frame": frame,
            "evaluation_split": "pilot" if frame < 21 else "later_frames",
            "train_anchors": int(train.sum()),
            "holdout_anchors": int(hold.sum()),
            "log_scale_coefficients": beta.tolist(),
            "global_median_rel": float(np.median(baseline_err)),
            "affine_median_rel": float(np.median(affine_err)),
            "global_p90_rel": float(np.quantile(baseline_err, 0.9)),
            "affine_p90_rel": float(np.quantile(affine_err, 0.9)),
        }
        rows.append(row)
        np.savez_compressed(
            args.output / path.name,
            depth_sfm_affine=affine_depth,
            mask=mask,
            coefficients=beta,
        )
        if ordinal % args.fusion_every == 0:
            out_h, out_w = global_depth.shape
            if args.preprocess == "pad_portrait_14":
                resized_width = round(width / height * out_h / 14) * 14
                pad_left = (out_w - resized_width) // 2
                resized = cv2.cvtColor(
                    cv2.resize(
                        bgr, (resized_width, out_h), interpolation=cv2.INTER_AREA
                    ),
                    cv2.COLOR_BGR2RGB,
                )
                rgb = np.full((out_h, out_w, 3), 255, dtype=np.uint8)
                rgb[:, pad_left : pad_left + resized_width] = resized
            else:
                rgb = cv2.cvtColor(
                    cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_AREA),
                    cv2.COLOR_BGR2RGB,
                )
            xyz, colors = fuse_frame(
                rgb,
                affine_depth,
                mask,
                xy_small,
                displacement,
                pose,
                args.pixel_stride,
                args.preprocess,
            )
            fused_xyz.append(xyz)
            fused_rgb.append(colors)
        print(json.dumps(row), flush=True)
    xyz = np.concatenate(fused_xyz)
    colors = np.concatenate(fused_rgb)
    anchors = state["canonical"][state["accepted"]]
    anchor_tree = cKDTree(anchors)
    local_spacing = anchor_tree.query(anchors, k=min(5, len(anchors)))[0][:, -1]
    radius = float(max(0.75, 2 * np.quantile(local_spacing, 0.9)))
    near_anchor = anchor_tree.query(xyz, k=1)[0] < radius
    raw_points = len(xyz)
    xyz, colors = xyz[near_anchor], colors[near_anchor]
    write_ply(
        args.output / f"grass_{args.model_name}_spatial_track_aligned_mean.ply",
        xyz,
        colors,
    )
    report = {
        "method": "robust affine log-scale from sparse CoW 3D tracks (MTD-inspired; not MTD code)",
        "model_name": args.model_name,
        "preprocess": args.preprocess,
        "coordinates": "arbitrary COLMAP SfM units, approximate CoW time-mean shape",
        "fused_points": len(xyz),
        "pre_filter_points": raw_points,
        "anchor_proximity_radius_sfm_units": radius,
        "frames": rows,
    }
    for split in ("pilot", "later_frames"):
        subset = [row for row in rows if row["evaluation_split"] == split]
        if subset:
            report[split] = {
                "frames": len(subset),
                "global_median_of_frame_medians": float(
                    np.median([r["global_median_rel"] for r in subset])
                ),
                "affine_median_of_frame_medians": float(
                    np.median([r["affine_median_rel"] for r in subset])
                ),
                "global_median_of_frame_p90": float(
                    np.median([r["global_p90_rel"] for r in subset])
                ),
                "affine_median_of_frame_p90": float(
                    np.median([r["affine_p90_rel"] for r in subset])
                ),
                "improved_frame_count": sum(
                    r["affine_median_rel"] < r["global_median_rel"] for r in subset
                ),
            }
    strict = [r for r in rows if r["frame"] >= 21 and r["frame"] % 5 == 4]
    if strict:
        report["later_unfitted_frames"] = {
            "frames": len(strict),
            "global_median_of_frame_medians": float(
                np.median([r["global_median_rel"] for r in strict])
            ),
            "affine_median_of_frame_medians": float(
                np.median([r["affine_median_rel"] for r in strict])
            ),
            "improved_frame_count": sum(
                r["affine_median_rel"] < r["global_median_rel"] for r in strict
            ),
        }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
