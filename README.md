# eKYC Vehicle-Context Detector

Two independent, zero-training eKYC fraud/compliance checks:

| Package | Detects | Method |
|---|---|---|
| `ekyc_vehicle_detect` | Video recorded **inside a car** | CLIP zero-shot frame classification |
| `ekyc_signature_check` | Signature image **too simple** (letter/shape/dots) | Classical contour & shape analysis |

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Vehicle-context video detection

Flags an eKYC video recorded inside a car — e.g. a subject completing
liveness checks while driving, or staging a video in a car to dodge a
controlled environment. Uses a pretrained CLIP model
(`openai/clip-vit-base-patch32`) to score sampled frames against
`in_vehicle`/`not_in_vehicle` text prompts (`prompts.py`) — no labeled
video data needed. First run downloads the checkpoint (~600MB).

```bash
python -m ekyc_vehicle_detect.cli video.mp4 [--json]
```
```python
from ekyc_vehicle_detect import VehicleContextDetector
result = VehicleContextDetector().detect("video.mp4")
```

**How it decides:** frames are sampled uniformly across the video
(`frame_sampler.py`, since eKYC clips often open with instructions or a
static ID card), each is scored by CLIP (`classifier.py`), and the video is
flagged only if *both* the mean score and the fraction of individually
agreeing frames clear their thresholds (`detector.py`) — two thresholds
instead of one so a single false-positive frame (car glimpsed through a
window) can't tip the verdict alone.

Options: `--frames` (16), `--score-threshold` (0.55), `--ratio-threshold`
(0.4), `--model` (any HF CLIP id), `--device` (auto).

**Validated** against real Wikimedia Commons footage: car-interior photos
scored 0.998–0.999, indoor scenes scored 0.005–0.008, and two real dashcam
clips (minimal visible cabin, windshield-POV) were both correctly flagged
despite being a much harder case than a clean interior photo. See
`tests/test_detector.py` (`pytest -m slow`, needs network for the model).

**Limitations:** thresholds are untuned defaults, not validated against real
eKYC submissions — calibrate before production. CLIP judges content, not
physics (can't tell parked vs. moving, or a car shown on a screen behind
the subject) — pair with existing liveness checks, don't rely on it alone.
Frame sampling is time-uniform, not scene-aware, so a very brief glimpse of
the vehicle context can be missed between samples — raise `--frames` if
that matters for you.

## Signature complexity check

Flags a signature image as too simple to plausibly be genuine: a single
letter, a basic shape (square/circle/star/triangle), or scattered dots.
Classical contour/shape analysis (`signature_complexity.py`) — no model,
fully deterministic.

```bash
python -m ekyc_signature_check.cli signature.png [--json]
```
```python
from ekyc_signature_check import analyze_signature_complexity
result = analyze_signature_complexity("signature.png")
print(result.is_too_simple, result.reasons)
```

**Reason codes**, each independent:

- `disconnected_dots` — several components that are each individually
  round/blob-shaped. (A *printed* multi-letter signature also splits into
  several components, one per letter, but letters aren't circular — that
  was an actual false positive caught during calibration and fixed by
  checking per-component circularity instead of "no dominant component.")
- `circular_shape` / `simple_polygon_shape` / `symmetric_shape` — the main
  shape is a circle, a low-vertex convex polygon, or rotationally symmetric
  (catches a 5-point star, which is concave and slips past the polygon check).
- `single_letter_or_compact_mark` — few components with a near-square
  minimum-area bounding rectangle (rotation-invariant, so a signature
  photographed or written sideways isn't penalized). A real signature spans
  several letters and is reliably wider than tall; a single glyph is
  square-ish *regardless of how jagged its outline is* (a bold serif
  letter's silhouette can be as complex as real cursive), which is why this
  is a separate check rather than folded into stroke complexity.
- `trivial_mark` — very few vertices in the polygon approximation of the
  main shape, regardless of solidity or aspect ratio. Catches a checkmark
  or an "X": genuine signatures, even short ones, wind through several
  letters and need more vertices to approximate (observed gap: trivial
  marks score 6-8, every genuine case tested scores 9+).
- `insufficient_stroke_complexity` — low ink perimeter relative to the
  mark's size, for anything the above miss.

**Validated** against 16 synthetic cases (`tests/test_signature_complexity.py`),
including adversarial ones added after the fact: a heart (mirror- but not
rotationally-symmetric), a spiral, an infinity symbol, a checkmark and an
"X", a genuine signature rotated 90° (this was a real false positive until
the bounding box was made rotation-invariant), a short signature in an
actual script font, a printed name, and a monogram — all correctly
classified.

**Limitations:** thresholds (all keyword args, e.g. `dot_count_threshold`,
`circularity_threshold`, `min_signature_aspect_ratio`) are tuned on
synthetic shapes, not real signatures — calibrate before production.
Assumes ink is the minority color (true for a scanned box or pad capture).
A fast, genuinely scribbled short signature with a near-square bounding box
could still trip `single_letter_or_compact_mark` — a precision/recall
tradeoff to tune via `min_signature_aspect_ratio`, not a bug. The heart
shape is caught via bounding-box aspect ratio, not reflection symmetry —
the symmetry check is rotation-only, so a mirror-symmetric-but-elongated
shape could in principle slip through undetected.
