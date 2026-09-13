"""Smoke test for the full pipeline: synthetic video -> detection result.

Downloads the CLIP checkpoint (~600MB) on first run and requires network
access, so this is a slow integration test rather than a unit test.
"""

import subprocess
from pathlib import Path

import pytest

from ekyc_vehicle_detect.detector import VehicleContextDetector


def _make_synthetic_video(tmp_path: Path) -> Path:
    """Build a short clip from solid-color frames as a structural smoke test.

    This does not exercise real-world visual accuracy (there's no genuine
    car interior in a solid color frame) — it only proves that frame
    sampling, batching, and aggregation run end-to-end without error.
    """
    video_path = tmp_path / "synthetic.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=3:size=320x240:rate=10",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
            "-loglevel",
            "error",
        ],
        check=True,
    )
    return video_path


@pytest.mark.slow
def test_detect_runs_end_to_end(tmp_path):
    video_path = _make_synthetic_video(tmp_path)

    detector = VehicleContextDetector(max_frames=6)
    result = detector.detect(str(video_path))

    assert result.num_frames_sampled == 6
    assert 0.0 <= result.mean_vehicle_score <= 1.0
    assert 0.0 <= result.vehicle_frame_ratio <= 1.0
    assert isinstance(result.is_in_vehicle, bool)
    assert len(result.frame_results) == 6
    timestamps = [f.timestamp_sec for f in result.frame_results]
    assert timestamps == sorted(timestamps)
