"""Calibration test: synthetic shapes/marks vs. a genuine-looking scribble.

Thresholds in signature_complexity.py were tuned against exactly these
synthetic cases (plus a printed multi-letter name and a two-letter monogram,
to check for false positives) — see the module docstring for the caveat
that real signatures should still be used to calibrate for production.
"""

import random

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from ekyc_signature_check.signature_complexity import analyze_signature_complexity

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


def make_heart() -> np.ndarray:
    """Mirror-symmetric but NOT rotationally symmetric — stresses the
    rotational-only symmetry check (it's caught via bounding-box aspect
    ratio instead, so this passing doesn't mean reflection symmetry is
    actually detected)."""
    img = _blank()
    t = np.linspace(0, 2 * np.pi, 300)
    x = 16 * np.sin(t) ** 3
    y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
    pts = list(zip((200 + x * 7).astype(int), (100 + y * 7).astype(int)))
    ImageDraw.Draw(img).polygon(pts, outline=0, width=5)
    return np.array(img)


def make_checkmark() -> np.ndarray:
    img = _blank()
    ImageDraw.Draw(img).line([(150, 100), (185, 135), (250, 65)], fill=0, width=8, joint="curve")
    return np.array(img)


def make_x_mark() -> np.ndarray:
    img = _blank()
    d = ImageDraw.Draw(img)
    d.line([(150, 60), (250, 140)], fill=0, width=8)
    d.line([(150, 140), (250, 60)], fill=0, width=8)
    return np.array(img)


def make_scribble_rotated_90() -> np.ndarray:
    """The genuine cursive_scribble rotated 90 degrees — a real signature
    photographed or written sideways must not become a false positive."""
    arr = make_cursive_scribble()
    return cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE)


def make_cursive_monogram_script() -> np.ndarray:
    """A short (2-letter) but genuine signature in an actual script font,
    as opposed to the block-letter make_monogram case above."""
    img = _blank()
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/SnellRoundhand.ttc", 90)
    except OSError:
        font = _font(90)
    ImageDraw.Draw(img).text((110, 40), "AB", fill=0, font=font)
    return np.array(img)


def make_spiral() -> np.ndarray:
    img = _blank()
    t = np.linspace(0, 6 * np.pi, 500)
    r = t * 6
    pts = list(zip((200 + r * np.cos(t)).astype(int), (100 + r * np.sin(t)).astype(int)))
    ImageDraw.Draw(img).line(pts, fill=0, width=4, joint="curve")
    return np.array(img)


def make_infinity() -> np.ndarray:
    """A lemniscate: 2-fold rotationally symmetric, unlike the heart above."""
    img = _blank()
    t = np.linspace(0, 2 * np.pi, 300)
    a = 80
    x = a * np.cos(t) / (1 + np.sin(t) ** 2)
    y = a * np.sin(t) * np.cos(t) / (1 + np.sin(t) ** 2)
    pts = list(zip((200 + x).astype(int), (100 + y).astype(int)))
    ImageDraw.Draw(img).line(pts, fill=0, width=6, joint="curve")
    return np.array(img)


def make_zigzag() -> np.ndarray:
    """Geometrically elaborate-looking but semantically trivial (repetitive,
    no letterforms) — happens to have near-180-degree rotational symmetry."""
    img = _blank()
    pts = [(60, 100), (100, 60), (140, 100), (180, 60), (220, 100), (260, 60), (300, 100), (340, 60)]
    ImageDraw.Draw(img).line(pts, fill=0, width=5, joint="curve")
    return np.array(img)


CASES = {
    "square": (make_square, True),
    "circle": (make_circle, True),
    "star": (make_star, True),
    "single_letter": (make_single_letter, True),
    "dots": (make_dots, True),
    "heart": (make_heart, True),
    "checkmark": (make_checkmark, True),
    "x_mark": (make_x_mark, True),
    "spiral": (make_spiral, True),
    "infinity": (make_infinity, True),
    "zigzag": (make_zigzag, True),
    "cursive_scribble": (make_cursive_scribble, False),
    "scribble_rotated_90": (make_scribble_rotated_90, False),
    "printed_name": (make_printed_name, False),
    "monogram": (make_monogram, False),
    "cursive_monogram_script": (make_cursive_monogram_script, False),
}


def test_calibration_cases():
    failures = []
    for name, (make_image, expected_too_simple) in CASES.items():
        result = analyze_signature_complexity(make_image())
        if result.is_too_simple != expected_too_simple:
            failures.append(f"{name}: expected too_simple={expected_too_simple}, got {result.is_too_simple} (reasons={result.reasons})")
    assert not failures, "\n".join(failures)
