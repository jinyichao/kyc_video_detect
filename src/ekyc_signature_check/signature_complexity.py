"""Heuristic detection of overly simple "signatures".

Flags a single letter, a basic geometric shape (square/circle/star/
triangle), or scattered disconnected dots — marks that are easy to produce
without actually signing, and weak evidence of identity compared to a
genuine cursive signature.

Unlike the CLIP-based vehicle-context detector, this uses classical contour
and shape analysis rather than a learned model: "too simple" is a structural
property of the ink strokes (how much they wind, whether they're
symmetric, how many separate pieces there are), not scene content, so it
doesn't benefit from a vision-language model and classical CV is cheaper
and fully deterministic.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class SignatureComplexityResult:
    is_too_simple: bool
    reasons: list[str]
    metrics: dict = field(default_factory=dict)


def _binarize_ink(gray: np.ndarray) -> np.ndarray:
    """Otsu-threshold to a binary ink mask (255 = ink), auto-fixing polarity.

    Assumes ink covers a minority of the image, true for essentially any
    signature capture (a signature pad, a scanned signature box, a cropped
    photo of a signature line): whichever class is smaller after
    thresholding is treated as ink.
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.count_nonzero(binary) > binary.size // 2:
        binary = cv2.bitwise_not(binary)
    return binary


def _rotational_symmetry_score(mask: np.ndarray, n_fold: int) -> float:
    """IoU between `mask` and itself rotated by 360/n_fold degrees about its centroid.

    A value near 1 means the ink is unchanged by that rotation, i.e. it has
    n-fold rotational symmetry — true of a square (n=4) or a 5-point star
    (n=5) but essentially never true of genuine cursive handwriting.
    """
    ys, xs = np.nonzero(mask)
    cx, cy = float(xs.mean()), float(ys.mean())
    h, w = mask.shape
    rot = cv2.getRotationMatrix2D((cx, cy), 360.0 / n_fold, 1.0)
    rotated = cv2.warpAffine(mask, rot, (w, h), flags=cv2.INTER_NEAREST)
    intersection = np.count_nonzero((mask > 0) & (rotated > 0))
    union = np.count_nonzero((mask > 0) | (rotated > 0))
    return intersection / union if union else 0.0


def analyze_signature_complexity(
    image,
    min_component_area_frac: float = 0.0005,
    dot_count_threshold: int = 4,
    dot_circularity_threshold: float = 0.5,
    dot_like_fraction_threshold: float = 0.7,
    circularity_threshold: float = 0.75,
    max_shape_vertices: int = 8,
    shape_solidity_threshold: float = 0.85,
    symmetry_threshold: float = 0.55,
    min_stroke_complexity: float = 2.2,
    min_signature_aspect_ratio: float = 1.6,
    compact_mark_max_components: int = 2,
) -> SignatureComplexityResult:
    """Flag a signature image as too simple to plausibly be a genuine signature.

    `image` is a file path or an already-loaded grayscale/BGR numpy array.
    All thresholds are exposed because "too simple" is a policy decision,
    not a fact about the image — calibrate them against your own accepted
    and rejected signature samples rather than trusting the defaults (which
    were only checked against synthetic shapes; see tests/test_signature_complexity.py).

    Detects, independently: disconnected dots (several separate components
    that are each individually round/blob-shaped, as opposed to several
    disconnected *letters* in a printed — non-cursive — signature, which are
    not round), circular/simple-polygon/rotationally-symmetric shapes
    (square, circle, star, triangle, ...), and a single letter or other
    compact mark (few components with a near-square bounding box, where a
    genuine signature's is much wider than tall).
    """
    if isinstance(image, str):
        gray = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"Could not read image: {image}")
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    binary = _binarize_ink(gray)
    h, w = binary.shape

    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]  # skip label 0 (background)
    min_area = max(4, int(min_component_area_frac * h * w))
    keep_mask = areas >= min_area
    component_labels = np.nonzero(keep_mask)[0] + 1
    areas = areas[keep_mask]

    if len(component_labels) == 0:
        return SignatureComplexityResult(
            is_too_simple=True,
            reasons=["no_ink_detected"],
            metrics={"num_components": 0},
        )

    total_ink_area = int(areas.sum())
    num_components = int(len(component_labels))
    largest_idx = int(np.argmax(areas))
    largest_label = int(component_labels[largest_idx])
    largest_area_ratio = float(areas[largest_idx] / total_ink_area)

    kept_mask = np.isin(labels, component_labels)
    ys, xs = np.nonzero(kept_mask)
    bbox_w = int(xs.max() - xs.min()) or 1
    bbox_h = int(ys.max() - ys.min()) or 1
    bbox_aspect_ratio = bbox_w / bbox_h

    # Per-component circularity distinguishes round dots from disconnected
    # *letters* (a printed, non-cursive signature also splits into several
    # components, but letter shapes are far less circular than dots).
    per_component_circularity = []
    for lbl in component_labels:
        comp_mask = np.where(labels == lbl, np.uint8(255), np.uint8(0))
        comp_contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        comp_contour = max(comp_contours, key=cv2.contourArea)
        comp_area = cv2.contourArea(comp_contour)
        comp_perimeter = cv2.arcLength(comp_contour, True)
        circ = (4 * np.pi * comp_area / (comp_perimeter**2)) if comp_perimeter > 0 else 0.0
        per_component_circularity.append(circ)
    dot_like_count = int(sum(c >= dot_circularity_threshold for c in per_component_circularity))
    dot_like_fraction = dot_like_count / num_components

    reasons: list[str] = []
    metrics: dict = {
        "num_components": num_components,
        "total_ink_area": total_ink_area,
        "largest_component_area_ratio": largest_area_ratio,
        "bbox_aspect_ratio": bbox_aspect_ratio,
        "dot_like_fraction": dot_like_fraction,
    }

    if num_components >= dot_count_threshold and dot_like_fraction >= dot_like_fraction_threshold:
        reasons.append("disconnected_dots")

    # A genuine signature is written left-to-right across several letters and
    # is almost always noticeably wider than tall; a single glyph or a
    # compact shape has a roughly square bounding box regardless of its
    # outline complexity (e.g. a single serif letter can have as jagged a
    # silhouette as real cursive writing, so stroke complexity alone misses it).
    if num_components <= compact_mark_max_components and bbox_aspect_ratio < min_signature_aspect_ratio:
        reasons.append("single_letter_or_compact_mark")

    largest_mask = np.where(labels == largest_label, np.uint8(255), np.uint8(0))
    contours, _ = cv2.findContours(largest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    _, _, cw, ch = cv2.boundingRect(contour)
    comp_diag = float(np.hypot(cw, ch)) or 1.0

    circularity = (4 * np.pi * area / (perimeter**2)) if perimeter > 0 else 0.0
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull) or 1.0
    solidity = area / hull_area
    approx = cv2.approxPolyDP(contour, 0.015 * perimeter, True)
    vertex_count = len(approx)
    stroke_complexity = perimeter / comp_diag
    best_symmetry = max(_rotational_symmetry_score(largest_mask, n) for n in (2, 3, 4, 5, 6, 8))

    metrics.update(
        circularity=circularity,
        solidity=solidity,
        approx_vertex_count=vertex_count,
        stroke_complexity=stroke_complexity,
        rotational_symmetry=best_symmetry,
    )

    if circularity >= circularity_threshold:
        reasons.append("circular_shape")
    elif vertex_count <= max_shape_vertices and solidity >= shape_solidity_threshold:
        reasons.append("simple_polygon_shape")
    elif best_symmetry >= symmetry_threshold:
        reasons.append("symmetric_shape")

    if "disconnected_dots" not in reasons and num_components <= 2 and stroke_complexity < min_stroke_complexity:
        reasons.append("insufficient_stroke_complexity")

    return SignatureComplexityResult(
        is_too_simple=len(reasons) > 0,
        reasons=reasons,
        metrics=metrics,
    )
