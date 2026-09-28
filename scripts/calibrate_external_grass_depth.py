#!/usr/bin/env python3
"""Scale external monocular depth maps to COLMAP using good 3D track anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth-input", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="external")
    parser.add_argument(
        "--preprocess", choices=["direct", "pad_portrait_14"], default="direct"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.depth_input.glob("[0-9][0-9][0-9][0-9][0-9].npz"))
    if not paths:
        raise FileNotFoundError("no numbered depth NPZ files")
    rows = []
    for path in paths:
        frame = int(path.stem) - 1
        with np.load(path, allow_pickle=False) as source:
            raw = source["depth"].astype(np.float32)
        with np.load(args.anchors / path.name, allow_pickle=False) as anchors:
            xy = anchors["xy_original"]
            z_sfm = anchors["z_sfm"]
            ids = anchors["track_id"]
            width, height = map(int, anchors["original_size"])
        if raw.ndim != 2 or raw.shape[0] <= 0 or raw.shape[1] <= 0:
            raise ValueError(f"invalid depth shape in {path}")
        if args.preprocess == "pad_portrait_14":
            if height < width or raw.shape[0] != raw.shape[1]:
                raise ValueError(
                    "padded ZipMap depth expects a square output from portrait input"
                )
            resized_width = round(width / height * raw.shape[0] / 14) * 14
            pad_left = (raw.shape[1] - resized_width) // 2
            x = (xy[:, 0] + 0.5) * resized_width / width - 0.5 + pad_left
            y = (xy[:, 1] + 0.5) * raw.shape[0] / height - 0.5
        else:
            x = (xy[:, 0] + 0.5) * raw.shape[1] / width - 0.5
            y = (xy[:, 1] + 0.5) * raw.shape[0] / height - 0.5
        sampled = cv2.remap(
            raw,
            x.astype(np.float32).reshape(1, -1),
            y.astype(np.float32).reshape(1, -1),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=np.nan,
        ).ravel()
        valid = np.isfinite(sampled) & (sampled > 0) & (z_sfm > 0)
        train = valid & (ids % 5 != 4)
        holdout = valid & (ids % 5 == 4)
        if train.sum() < 20 or holdout.sum() < 5:
            raise ValueError(
                f"frame {frame}: insufficient calibration or holdout anchors"
            )
        log_ratio = np.log(z_sfm[train] / sampled[train])
        center = np.median(log_ratio)
        mad = np.median(np.abs(log_ratio - center))
        inlier = np.abs(log_ratio - center) <= max(3 * 1.4826 * mad, 0.15)
        scale = float(np.exp(np.median(log_ratio[inlier])))
        relative_error = (
            np.abs(scale * sampled[holdout] - z_sfm[holdout]) / z_sfm[holdout]
        )
        mask = np.isfinite(raw) & (raw > 0)
        np.savez_compressed(
            args.output / path.name,
            depth_sfm=(raw * scale).astype(np.float32),
            mask=mask,
            scale_to_sfm=scale,
        )
        rows.append(
            {
                "frame": frame,
                "train_anchors": int(train.sum()),
                "holdout_anchors": int(holdout.sum()),
                "scale_to_sfm": scale,
                "holdout_median_relative_error": float(np.median(relative_error)),
                "holdout_p90_relative_error": float(np.quantile(relative_error, 0.9)),
            }
        )
    report = {
        "model": args.model_name,
        "preprocess": args.preprocess,
        "coordinates": "arbitrary COLMAP SfM units",
        "calibration": "per-frame robust log-depth scale from 3D track IDs mod 5 != 4",
        "frames": rows,
    }
    strict = [row for row in rows if row["frame"] % 5 == 4]
    report["heldout_from_3d_fit"] = {
        "frames": len(strict),
        "median_of_frame_medians": float(
            np.median([row["holdout_median_relative_error"] for row in strict])
        ),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "frames"}))


if __name__ == "__main__":
    main()
