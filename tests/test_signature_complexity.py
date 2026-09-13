"""Calibration test: synthetic shapes/marks vs. a genuine-looking scribble.

Thresholds in signature_complexity.py were tuned against exactly these
synthetic cases (plus a printed multi-letter name and a two-letter monogram,
to check for false positives) — see the module docstring for the caveat
that real signatures should still be used to calibrate for production.
"""

import random
import sys
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ekyc_signature_check.signature_complexity import analyze_signature_complexity  # noqa: E402

W, H = 400, 200


def _blank() -> Image.Image:
    return Image.new("L", (W, H), 255)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Georgia.ttf", size)
    except OSError:
        return ImageFont.load_default()


def make_square() -> np.ndarray:
    img = _blank()
    ImageDraw.Draw(img).rectangle([130, 50, 270, 150], outline=0, width=6)
    return np.array(img)


def make_circle() -> np.ndarray:
    img = _blank()
    ImageDraw.Draw(img).ellipse([130, 40, 270, 160], outline=0, width=6)
    return np.array(img)


def make_star() -> np.ndarray:
    img = _blank()
    d = ImageDraw.Draw(img)
    cx, cy, r_out, r_in = 200, 100, 70, 28
    pts = []
    for i in range(10):
        ang = -np.pi / 2 + i * np.pi / 5
        r = r_out if i % 2 == 0 else r_in
        pts.append((cx + r * np.cos(ang), cy + r * np.sin(ang)))
    d.polygon(pts, outline=0, width=6)
    return np.array(img)


def make_single_letter() -> np.ndarray:
    img = _blank()
    ImageDraw.Draw(img).text((140, 20), "A", fill=0, font=_font(140))
    return np.array(img)


def make_dots() -> np.ndarray:
    img = _blank()
    d = ImageDraw.Draw(img)
    rng = random.Random(0)
    for _ in range(6):
        x, y = rng.randint(80, 320), rng.randint(60, 140)
        d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=0)
    return np.array(img)


def make_cursive_scribble() -> np.ndarray:
    arr = np.array(_blank())
    t = np.linspace(0, 4 * np.pi, 2000)
    x = 200 + 130 * np.cos(t * 0.9) * np.exp(-0.02 * t) + 20 * np.sin(t * 3.3)
    y = 100 + 40 * np.sin(t * 1.7) + 15 * np.cos(t * 5.1)
    x = np.clip(x, 5, W - 5).astype(int)
    y = np.clip(y, 5, H - 5).astype(int)
    pts = np.stack([x, y], axis=1).reshape(-1, 1, 2)
    cv2.polylines(arr, [pts], isClosed=False, color=0, thickness=5)
    return arr


def make_printed_name() -> np.ndarray:
    img = _blank()
    ImageDraw.Draw(img).text((40, 70), "Sam Lee", fill=0, font=_font(60))
    return np.array(img)


def make_monogram() -> np.ndarray:
    img = _blank()
    ImageDraw.Draw(img).text((120, 60), "JD", fill=0, font=_font(60))
    return np.array(img)


CASES = {
    "square": (make_square, True),
    "circle": (make_circle, True),
    "star": (make_star, True),
    "single_letter": (make_single_letter, True),
    "dots": (make_dots, True),
    "cursive_scribble": (make_cursive_scribble, False),
    "printed_name": (make_printed_name, False),
    "monogram": (make_monogram, False),
}


def test_calibration_cases():
    failures = []
    for name, (make_image, expected_too_simple) in CASES.items():
        result = analyze_signature_complexity(make_image())
        if result.is_too_simple != expected_too_simple:
            failures.append(f"{name}: expected too_simple={expected_too_simple}, got {result.is_too_simple} (reasons={result.reasons})")
    assert not failures, "\n".join(failures)
