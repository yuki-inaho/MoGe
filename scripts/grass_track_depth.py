#!/usr/bin/env python3
"""Align MoGe-3 depth to COLMAP units using held-out CoWTracker 3D tracks.

The input fit is a time-mean deforming grass model. Output is in arbitrary SfM
units; neither COLMAP nor the fitted tracks establish a metric scale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from moge.model.v3 import MoGeModel


def read_colmap(
    model_dir: Path,
) -> dict[int, tuple[np.ndarray, np.ndarray, tuple[float, ...]]]:
    cameras = {}
    for line in (model_dir / "cameras.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if fields[1] != "SIMPLE_RADIAL":
            raise ValueError(f"unsupported COLMAP camera: {fields[1]}")
        cameras[int(fields[0])] = tuple(map(float, fields[2:8]))
    poses = {}
    for line in (model_dir / "images.txt").read_text().splitlines():
        fields = line.split()
        if len(fields) < 10 or not fields[0].isdigit():
            continue
        stem = Path(fields[9]).stem
        if not stem.isdigit():
            continue
        quat = np.array([*map(float, fields[2:5]), float(fields[1])])
        poses[int(stem) - 1] = (
            Rotation.from_quat(quat).as_matrix(),
            np.array([float(v) for v in fields[5:8]]),
            cameras[int(fields[8])],
        )
    if not poses:
        raise ValueError("no numbered registered images in COLMAP model")
    return poses


def motion_at_frame(state: np.lib.npyio.NpzFile, frame: int) -> np.ndarray:
    coefficients = np.einsum("nbd,b->nd", state["motion"], state["phi"][frame])
    return np.sum(
        coefficients[state["neighbor_ids"]] * state["neighbor_weights"][..., None],
        axis=1,
    )


def undistort(
    x: np.ndarray, y: np.ndarray, camera: tuple[float, ...]
) -> tuple[np.ndarray, np.ndarray]:
    _, _, focal, cx, cy, k = camera
    xd, yd = (x - cx) / focal, (y - cy) / focal
    xu, yu = xd.copy(), yd.copy()
    for _ in range(8):
        radial = 1 + k * (xu * xu + yu * yu)
        xu, yu = xd / radial, yd / radial
    return xu, yu


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    with path.open("w") as file:
        file.write("ply\nformat ascii 1.0\n")
        file.write(
            "comment COLMAP SfM units, approximate CoW time-mean warp; not metric or no-wind\n"
        )
        file.write(f"element vertex {len(xyz)}\n")
        file.write("property float x\nproperty float y\nproperty float z\n")
        file.write(
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        )
        for point, color in zip(xyz, rgb):
            file.write("{:.7g} {:.7g} {:.7g} {} {} {}\n".format(*point, *color))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_and_anchors(
    frame: int,
    image: np.ndarray,
    state: np.lib.npyio.NpzFile,
    tracks: np.lib.npyio.NpzFile,
    pose: tuple[np.ndarray, np.ndarray, tuple[float, ...]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rotation, translation, camera = pose
    height, width = image.shape[:2]
    if (width, height) != tuple(map(int, camera[:2])):
        raise ValueError("image dimensions do not match COLMAP camera")
    yx = state["selected_yx"]
    accepted = state["accepted"]
    track_height, track_width = tracks["tracks"].shape[1:3]
    tracked = tracks["tracks"][frame, yx[:, 0], yx[:, 1]]
    observed = np.stack(
        [
            (tracked[:, 0] + 0.5) * width / track_width - 0.5,
            (tracked[:, 1] + 0.5) * height / track_height - 0.5,
        ],
        axis=1,
    )
    displacement = motion_at_frame(state, frame)
    camera_xyz = (state["canonical"] + displacement) @ rotation.T + translation
    z = camera_xyz[:, 2]
    _, _, focal, cx, cy, k = camera
    xy = camera_xyz[:, :2] / np.maximum(z[:, None], 1e-9)
    radial = 1 + k * np.sum(xy * xy, axis=1)
    projected = np.stack(
        [focal * xy[:, 0] * radial + cx, focal * xy[:, 1] * radial + cy], axis=1
    )
    reprojection = np.linalg.norm(projected - observed, axis=1)
    visibility = tracks["vis"][frame, yx[:, 0], yx[:, 1]]
    confidence = tracks["conf"][frame, yx[:, 0], yx[:, 1]]
    good = (
        accepted
        & (visibility > 0.6)
        & (confidence > 0.15)
        & (z > 0)
        & (reprojection < 3.0)
        & (observed[:, 0] >= 0)
        & (observed[:, 0] < width)
        & (observed[:, 1] >= 0)
        & (observed[:, 1] < height)
    )
    return observed[good], z[good], displacement[good], np.flatnonzero(good)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--colmap-model", type=Path, required=True)
    parser.add_argument("--fit-state", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", default="Ruicheng/moge-3-vitl")
    parser.add_argument("--max-size", type=int, default=512)
    parser.add_argument("--num-tokens", type=int, default=1200)
    parser.add_argument("--refine-steps", type=int, default=1)
    parser.add_argument("--fusion-every", type=int, default=4)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("MoGe-3 ViT-L inference requires CUDA for this workflow")
    if (
        args.max_size < 128
        or args.num_tokens < 100
        or args.fusion_every < 1
        or args.pixel_stride < 1
    ):
        parser.error("invalid size, token count, or sampling stride")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    poses = read_colmap(args.colmap_model)
    with (
        np.load(args.fit_state, allow_pickle=False) as state,
        np.load(args.tracks, allow_pickle=False) as tracks,
    ):
        if len(state["phi"]) != len(tracks["tracks"]):
            raise ValueError("fit and CoW track frame counts differ")
        frames = sorted(poses)
        if args.max_frames is not None:
            frames = frames[: args.max_frames]
        model = MoGeModel.from_pretrained(args.checkpoint).to("cuda").eval()
        fused_points, fused_colors = [], []
        rows = []
        for ordinal, frame in enumerate(frames):
            image_path = args.images / f"{frame + 1:05d}.jpg"
            bgr = cv2.imread(str(image_path))
            if bgr is None:
                raise FileNotFoundError(image_path)
            orig_h, orig_w = bgr.shape[:2]
            scale_image = args.max_size / max(orig_h, orig_w)
            small_w = max(1, round(orig_w * scale_image))
            small_h = max(1, round(orig_h * scale_image))
            rgb = cv2.cvtColor(
                cv2.resize(bgr, (small_w, small_h), interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2RGB,
            )
            tensor = (
                torch.from_numpy(rgb.copy())
                .permute(2, 0, 1)
                .float()
                .div_(255)
                .to("cuda")
            )
            pose = poses[frame]
            fov_x = float(np.degrees(2 * np.arctan(orig_w / (2 * pose[2][2]))))
            with torch.inference_mode():
                prediction = model.infer(
                    tensor,
                    num_tokens=args.num_tokens,
                    refine_steps=args.refine_steps,
                    fov_x=fov_x,
                    use_fp16=True,
                )
            depth = prediction["depth"].float().cpu().numpy()
            mask = prediction["mask"].cpu().numpy().astype(bool)
            del prediction, tensor
            xy_orig, z_sfm, displacement, indices = image_and_anchors(
                frame, bgr, state, tracks, pose
            )
            xy_small = np.stack(
                [
                    (xy_orig[:, 0] + 0.5) * small_w / orig_w - 0.5,
                    (xy_orig[:, 1] + 0.5) * small_h / orig_h - 0.5,
                ],
                axis=1,
            )
            sampled_z = cv2.remap(
                depth,
                xy_small[:, 0].astype(np.float32).reshape(1, -1),
                xy_small[:, 1].astype(np.float32).reshape(1, -1),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=np.nan,
            ).ravel()
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
            valid = (
                sampled_mask
                & np.isfinite(sampled_z)
                & (sampled_z > 0)
                & np.isfinite(z_sfm)
            )
            calib = valid & (indices % 5 != 4)
            hold = valid & (indices % 5 == 4)
            if calib.sum() < 20 or hold.sum() < 5:
                raise ValueError(
                    f"frame {frame}: only {calib.sum()} calibration and {hold.sum()} held-out anchors"
                )
            log_ratio = np.log(z_sfm[calib] / sampled_z[calib])
            center = np.median(log_ratio)
            mad = np.median(np.abs(log_ratio - center))
            inlier = np.abs(log_ratio - center) <= max(3 * 1.4826 * mad, 0.15)
            if inlier.sum() < 20:
                raise ValueError(f"frame {frame}: too few robust scale anchors")
            scale = float(np.exp(np.median(log_ratio[inlier])))
            hold_rel = np.abs(scale * sampled_z[hold] - z_sfm[hold]) / z_sfm[hold]
            out_depth = (depth * scale).astype(np.float32)
            np.savez_compressed(
                output / f"{frame + 1:05d}.npz",
                depth_sfm=out_depth,
                mask=mask,
                scale_to_sfm=scale,
                anchor_ids=indices[valid],
            )
            row = {
                "frame": frame,
                "image": image_path.name,
                "size": [small_w, small_h],
                "calibration_anchors": int(calib.sum()),
                "calibration_inliers": int(inlier.sum()),
                "holdout_anchors": int(hold.sum()),
                "scale_to_sfm": scale,
                "holdout_median_relative_error": float(np.median(hold_rel)),
                "holdout_p90_relative_error": float(np.quantile(hold_rel, 0.9)),
            }
            rows.append(row)
            if ordinal % args.fusion_every == 0:
                yy, xx = np.mgrid[
                    0 : small_h : args.pixel_stride, 0 : small_w : args.pixel_stride
                ]
                yy, xx = yy.ravel(), xx.ravel()
                hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
                green = (
                    (hsv[yy, xx, 0] >= 18)
                    & (hsv[yy, xx, 0] <= 95)
                    & (hsv[yy, xx, 1] >= 28)
                )
                good_pixel = (
                    green
                    & mask[yy, xx]
                    & np.isfinite(out_depth[yy, xx])
                    & (out_depth[yy, xx] > 0)
                )
                yy, xx = yy[good_pixel], xx[good_pixel]
                if len(xx):
                    tree = cKDTree(xy_small)
                    distances, neighbors = tree.query(
                        np.stack([xx, yy], axis=1), k=min(4, len(xy_small))
                    )
                    if distances.ndim == 1:
                        distances, neighbors = distances[:, None], neighbors[:, None]
                    near = distances[:, 0] < 12
                    yy, xx, distances, neighbors = (
                        yy[near],
                        xx[near],
                        distances[near],
                        neighbors[near],
                    )
                    weights = 1 / np.maximum(distances, 2) ** 2
                    weights /= weights.sum(axis=1, keepdims=True)
                    local_motion = np.sum(
                        displacement[neighbors] * weights[..., None], axis=1
                    )
                    x_orig = (xx + 0.5) * orig_w / small_w - 0.5
                    y_orig = (yy + 0.5) * orig_h / small_h - 0.5
                    xu, yu = undistort(x_orig, y_orig, pose[2])
                    z = out_depth[yy, xx]
                    cam_xyz = np.stack([xu * z, yu * z, z], axis=1)
                    world = (cam_xyz - pose[1]) @ pose[0] - local_motion
                    fused_points.append(world.astype(np.float32))
                    fused_colors.append(rgb[yy, xx])
            print(json.dumps(row), flush=True)
        if fused_points:
            xyz = np.concatenate(fused_points)
            colors = np.concatenate(fused_colors)
            write_ply(output / "grass_moge3_track_aligned_mean.ply", xyz, colors)
        report = {
            "model": "MoGe-3 ViT-L",
            "checkpoint": args.checkpoint,
            "checkpoint_sha256": file_sha256(Path(args.checkpoint))
            if Path(args.checkpoint).is_file()
            else None,
            "gpu": torch.cuda.get_device_name(0),
            "frames": rows,
            "fused_points": int(sum(map(len, fused_points))),
            "coordinates": "COLMAP arbitrary SfM units; approximate CoW time-mean deformation correction",
            "calibration": "per-frame robust log-depth scale from good CoW 3D tracks; spatial point holdout by id mod 5",
        }
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
