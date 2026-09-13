# eKYC Vehicle-Context Detector

Two independent eKYC fraud/compliance checks live in this package:

1. Zero-shot detection of whether a verification **video** was recorded
   **inside a car or other vehicle**.
2. Heuristic detection of an overly simple **signature image** (a single
   letter, a basic geometric shape, or disconnected dots).

## Vehicle-context video detection

Detects whether an eKYC verification video was recorded **inside a car or
other vehicle** — a common fraud/compliance signal (e.g. a subject
completing liveness checks while driving, or a video staged in a car to
avoid a controlled environment).

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

## Signature complexity check

Flags a signature image as **too simple** to plausibly be a genuine
signature: a single letter, a basic geometric shape (square, circle, star,
triangle, ...), or scattered disconnected dots. Unlike the video detector
above, this is classical contour/shape analysis
(`ekyc_signature_check/signature_complexity.py`) rather than a learned
model — no image encoder or GPU/CPU inference needed, and every threshold
is a plain geometric quantity you can reason about. It's a separate,
independent package (`src/ekyc_signature_check/`) from the video detector
(`src/ekyc_vehicle_detect/`), since the two checks share no code.

```bash
# CLI
python -m ekyc_signature_check.cli path/to/signature.png
python -m ekyc_signature_check.cli path/to/signature.png --json

# Library
from ekyc_signature_check import analyze_signature_complexity

result = analyze_signature_complexity("path/to/signature.png")
print(result.is_too_simple, result.reasons)  # e.g. ['single_letter_or_compact_mark']
```

It checks four things independently, each contributing its own reason code:

- **`disconnected_dots`** — several separate ink components that are each
  individually round/blob-shaped. (A printed, non-cursive signature also
  splits into several disconnected components — one per letter — but letter
  shapes are far less circular than dots, so this doesn't misfire on those.)
- **`circular_shape`** / **`simple_polygon_shape`** / **`symmetric_shape`**
  — the main ink shape is a circle/oval, a low-vertex convex polygon
  (square, triangle, ...), or has rotational symmetry (catches a 5-point
  star, which is concave and so isn't caught by the polygon check alone).
- **`single_letter_or_compact_mark`** — few ink components with a
  near-square bounding box. A genuine signature is written left-to-right
  across several letters and is reliably much wider than tall; a single
  glyph is roughly as tall as it is wide *regardless of how jagged its
  outline is* (a single bold serif letter's outline can be just as complex
  as real cursive writing, which is why this needs its own check rather
  than folding into a general "stroke complexity" score).
- **`insufficient_stroke_complexity`** — low ink perimeter relative to the
  size of the mark, for shapes not caught by the above.

### Validated behavior

Calibrated against 8 synthetic test cases (`tests/test_signature_complexity.py`):
correctly flags a square, circle, 5-point star, single letter, and
scattered dots as too simple, and correctly leaves alone a winding
scribble, a printed multi-letter name ("Sam Lee" — initially a false
positive under `disconnected_dots` before the per-component circularity
check was added, since printed letters are also disconnected from each
other), and a two-letter monogram ("JD").

### Limitations

- **Tuned on synthetic shapes, not real signatures.** All default
  thresholds (`dot_count_threshold`, `circularity_threshold`,
  `min_signature_aspect_ratio`, etc. — all exposed as keyword arguments)
  need calibration against your own accepted/rejected signature samples
  before production use.
- **Assumes ink is the minority color** in the image (true for a scanned
  signature box or a signature-pad capture); a very heavily inked or noisy
  scan can confuse the Otsu thresholding step.
- **A short but genuine cursive signature** (e.g. a fast scribbled initial)
  could still trip `single_letter_or_compact_mark` if its bounding box
  happens to be near-square — this is a precision/recall tradeoff to tune
  via `min_signature_aspect_ratio`, not a bug.
