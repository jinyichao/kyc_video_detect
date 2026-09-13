# eKYC Vehicle-Context Detector

Zero-shot detection of whether an eKYC verification video was recorded
**inside a car or other vehicle** — a common fraud/compliance signal (e.g.
a subject completing liveness checks while driving, or a video staged in a
car to avoid a controlled environment).

No training data or fine-tuning is required. The detector uses a pretrained
CLIP model (`openai/clip-vit-base-patch32` by default) to compare sampled
video frames against natural-language descriptions of "inside a vehicle" vs.
"not inside a vehicle" (see `src/ekyc_vehicle_detect/prompts.py`), so it works
out of the box on frame content alone.

## How it works

1. **Sample frames** uniformly across the full video duration (`frame_sampler.py`).
   Uniform sampling matters because eKYC clips often start with instructions
   or a static ID card before the subject appears.
2. **Score each frame** with CLIP zero-shot classification against two
   prompt-ensembled classes, `in_vehicle` / `not_in_vehicle` (`classifier.py`).
3. **Aggregate** across frames (`detector.py`): the video is flagged only if
   *both* the mean per-frame vehicle probability and the fraction of
   individually-agreeing frames clear their thresholds. Two thresholds
   (rather than one) avoid a single frame's false positive — e.g. a car
   visible through a window behind the subject — tipping the whole video.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

First run downloads the CLIP checkpoint (~600MB) from Hugging Face.

## Usage

```bash
# CLI
python -m ekyc_vehicle_detect.cli path/to/video.mp4
python -m ekyc_vehicle_detect.cli path/to/video.mp4 --json   # full per-frame breakdown

# Library
from ekyc_vehicle_detect import VehicleContextDetector

detector = VehicleContextDetector()
result = detector.detect("path/to/video.mp4")
print(result.is_in_vehicle, result.mean_vehicle_score)
```

CLI options: `--frames` (frames to sample, default 16), `--score-threshold`
(default 0.55), `--ratio-threshold` (default 0.4), `--model` (any HF CLIP
checkpoint id), `--device` (`cpu`/`cuda`, default auto).

## Validated behavior

Sanity-checked against real photos (Wikimedia Commons): two genuine car
interior photos scored ~0.998–0.999 vehicle probability; a bedroom and a
home office both scored ~0.005–0.008. A synthetic clip that starts in a
non-vehicle scene and cuts to a car interior was correctly flagged
(`is_in_vehicle: true`, `vehicle_frame_ratio: 0.67`) with per-frame scores
showing the exact transition point — see `tests/test_detector.py` for the
runnable pipeline smoke test (`pytest -m slow`, requires network for the
model download).

## Limitations & recommended next steps

- **Zero-shot ≠ calibrated.** The default thresholds (0.55 / 0.4) are
  reasonable starting points, not validated against a labeled dataset of
  real eKYC submissions. Before production use, tune them against your own
  labeled sample (particularly to check false-positive rate on cluttered
  indoor backgrounds, and false-negative rate on rear-seat/night/parked-car
  footage where vehicle cues are subtler).
- **CLIP sees content, not physics.** It cannot distinguish a stationary
  parked car from a moving one, or detect a car interior shown on a screen
  behind the subject (deepfake/replay attacks) — pair this with existing
  liveness/anti-spoofing checks, not as a standalone fraud signal.
- **Frame sampling is time-uniform, not scene-aware.** A very short glimpse
  of the vehicle context (e.g. only visible for 1 of 30 seconds) may be
  missed if it falls between sampled frames; increase `--frames` for longer
  videos if this matters for your workflow.
- **Swap the model** via `--model`/`model_name=` if you need higher accuracy
  (e.g. `openai/clip-vit-large-patch14`) at the cost of more compute per frame.
