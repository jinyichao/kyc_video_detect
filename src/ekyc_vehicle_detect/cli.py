import argparse
import json
import sys

from .classifier import DEFAULT_MODEL_NAME
from .detector import VehicleContextDetector


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-shot detection of whether an eKYC video was recorded inside a vehicle."
    )
    parser.add_argument("video", help="Path to the video file")
    parser.add_argument("--frames", type=int, default=16, help="Number of frames to sample (default: 16)")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.55,
        help="Minimum mean vehicle-class probability to flag the video (default: 0.55)",
    )
    parser.add_argument(
        "--ratio-threshold",
        type=float,
        default=0.4,
        help="Minimum fraction of sampled frames individually classified as vehicle (default: 0.4)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help=f"HF CLIP model id (default: {DEFAULT_MODEL_NAME})")
    parser.add_argument("--device", default=None, help="torch device, e.g. cpu / cuda (default: auto)")
    parser.add_argument("--json", action="store_true", help="Print full result as JSON, including per-frame scores")
    args = parser.parse_args()

    detector = VehicleContextDetector(
        model_name=args.model,
        device=args.device,
        max_frames=args.frames,
        score_threshold=args.score_threshold,
        ratio_threshold=args.ratio_threshold,
    )
    result = detector.detect(args.video)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        verdict = "INSIDE A VEHICLE" if result.is_in_vehicle else "not inside a vehicle"
        print(f"Verdict: {verdict}")
        print(f"  mean_vehicle_score:   {result.mean_vehicle_score:.3f}")
        print(f"  vehicle_frame_ratio:  {result.vehicle_frame_ratio:.3f}")
        print(f"  frames_sampled:       {result.num_frames_sampled}")
        print(f"  model:                {result.model_name}")

    sys.exit(0)


if __name__ == "__main__":
    main()
