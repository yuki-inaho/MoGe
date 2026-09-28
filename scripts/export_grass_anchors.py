#!/usr/bin/env python3
"""Export high-quality CoWTracker 3D depth anchors for cross-model comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from grass_track_depth import image_and_anchors, read_colmap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--colmap-model", type=Path, required=True)
    parser.add_argument("--fit-state", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    poses = read_colmap(args.colmap_model)
    with np.load(args.fit_state, allow_pickle=False) as archive:
        state = {key: archive[key] for key in archive.files}
    with np.load(args.tracks, allow_pickle=False) as archive:
        tracks = {key: archive[key] for key in archive.files}
    for frame, pose in sorted(poses.items()):
        image_path = args.images / f"{frame + 1:05d}.jpg"
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        xy, z_sfm, _, ids = image_and_anchors(frame, image, state, tracks, pose)
        np.savez_compressed(
            args.output / f"{frame + 1:05d}.npz",
            xy_original=xy.astype(np.float32),
            z_sfm=z_sfm.astype(np.float32),
            track_id=ids.astype(np.int32),
            original_size=np.array([image.shape[1], image.shape[0]], dtype=np.int32),
            heldout_from_3d_fit=bool(frame % 5 == 4),
        )
    print(f"exported {len(poses)} frame anchor files to {args.output}")


if __name__ == "__main__":
    main()
