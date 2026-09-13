"""High-level API: video path in, vehicle-context decision out."""

from dataclasses import dataclass, field

from .classifier import DEFAULT_MODEL_NAME, ClipZeroShotClassifier
from .frame_sampler import sample_frames


@dataclass
class FrameResult:
    frame_index: int
    timestamp_sec: float
    vehicle_score: float
    label: str


@dataclass
class DetectionResult:
    is_in_vehicle: bool
    mean_vehicle_score: float
    vehicle_frame_ratio: float
    num_frames_sampled: int
    model_name: str
    frame_results: list[FrameResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "is_in_vehicle": self.is_in_vehicle,
            "mean_vehicle_score": self.mean_vehicle_score,
            "vehicle_frame_ratio": self.vehicle_frame_ratio,
            "num_frames_sampled": self.num_frames_sampled,
            "model_name": self.model_name,
            "frame_results": [
                {
                    "frame_index": f.frame_index,
                    "timestamp_sec": f.timestamp_sec,
                    "vehicle_score": f.vehicle_score,
                    "label": f.label,
                }
                for f in self.frame_results
            ],
        }


class VehicleContextDetector:
    """Zero-shot detector for whether an eKYC video was recorded inside a vehicle.

    Two thresholds gate the final decision instead of one, to avoid a single
    frame's false positive (e.g. a car visible through a window behind the
    subject) tipping the whole video: `score_threshold` requires the average
    per-frame confidence to clear a bar, and `ratio_threshold` additionally
    requires a minimum fraction of sampled frames to individually agree.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
        max_frames: int = 16,
        score_threshold: float = 0.55,
        ratio_threshold: float = 0.4,
    ):
        self.max_frames = max_frames
        self.score_threshold = score_threshold
        self.ratio_threshold = ratio_threshold
        self.model_name = model_name
        self._classifier = ClipZeroShotClassifier(model_name=model_name, device=device)

    def detect(self, video_path: str) -> DetectionResult:
        sampled = sample_frames(video_path, max_frames=self.max_frames)
        images = [f.image_rgb for f in sampled]
        scores = self._classifier.score_frames(images)

        frame_results = [
            FrameResult(
                frame_index=sf.index,
                timestamp_sec=sf.timestamp_sec,
                vehicle_score=sc.vehicle_score,
                label=sc.label,
            )
            for sf, sc in zip(sampled, scores)
        ]

        mean_score = sum(f.vehicle_score for f in frame_results) / len(frame_results)
        vehicle_ratio = sum(1 for f in frame_results if f.label == "in_vehicle") / len(frame_results)

        is_in_vehicle = mean_score >= self.score_threshold and vehicle_ratio >= self.ratio_threshold

        return DetectionResult(
            is_in_vehicle=is_in_vehicle,
            mean_vehicle_score=mean_score,
            vehicle_frame_ratio=vehicle_ratio,
            num_frames_sampled=len(frame_results),
            model_name=self.model_name,
            frame_results=frame_results,
        )
