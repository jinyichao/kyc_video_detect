"""Uniform time-based frame sampling from a video file using OpenCV."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SampledFrame:
    index: int
    timestamp_sec: float
    image_rgb: np.ndarray


def sample_frames(video_path: str, max_frames: int = 16) -> list[SampledFrame]:
    """Uniformly sample up to `max_frames` frames across the full duration.

    Uniform sampling (rather than the first N frames) matters here because
    eKYC clips often start with instructions or a static ID card before the
    subject appears, and a vehicle interior may only be identifiable once
    the camera framing settles.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

        if total_frames <= 0:
            return _sample_by_sequential_read(cap, max_frames, fps)

        n = min(max_frames, total_frames)
        frame_indices = np.linspace(0, total_frames - 1, n, dtype=int)

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, bgr = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            timestamp = (idx / fps) if fps > 0 else 0.0
            frames.append(SampledFrame(index=int(idx), timestamp_sec=float(timestamp), image_rgb=rgb))

        if not frames:
            raise ValueError(f"No frames could be decoded from: {video_path}")
        return frames
    finally:
        cap.release()


def _sample_by_sequential_read(cap: cv2.VideoCapture, max_frames: int, fps: float) -> list[SampledFrame]:
    """Fallback for containers that don't report a reliable frame count.

    Reads sequentially and keeps an evenly spaced subset, which costs more
    decode time but works for streams where CAP_PROP_FRAME_COUNT is 0/-1
    (common with some mp4 variants and piped inputs).
    """
    all_frames = []
    idx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        all_frames.append((idx, bgr))
        idx += 1

    if not all_frames:
        raise ValueError("No frames could be decoded from video")

    n = min(max_frames, len(all_frames))
    pick_indices = np.linspace(0, len(all_frames) - 1, n, dtype=int)

    frames = []
    for i in pick_indices:
        frame_idx, bgr = all_frames[i]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        timestamp = (frame_idx / fps) if fps > 0 else 0.0
        frames.append(SampledFrame(index=frame_idx, timestamp_sec=float(timestamp), image_rgb=rgb))
    return frames
