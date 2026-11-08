"""Compact row-major RLE masks and O(runs) interval algebra."""

from __future__ import annotations

from typing import Any

import numpy as np


def _validated_runs(payload: Any) -> tuple[int, int, list[tuple[int, int]]]:
    """Validate and extract canonical sorted disjoint (start, end) run intervals."""
    if not isinstance(payload, dict):
        raise ValueError("RLE payload must be a dict")
    size = payload.get("size")
    counts = payload.get("counts")
    if (
        not isinstance(size, list)
        or len(size) != 2
        or type(size[0]) is not int
        or type(size[1]) is not int
        or size[0] < 0
        or size[1] < 0
        or not isinstance(counts, list)
        or (len(counts) & 1) != 0
    ):
        raise ValueError("RLE payload must contain valid 'size' [H, W] and even 'counts'")
    height, width = size[0], size[1]
    limit = height * width
    intervals: list[tuple[int, int]] = []
    prev_end = -1
    it = iter(counts)
    for start in it:
        length = next(it)
        if type(start) is not int or type(length) is not int or length <= 0:
            raise ValueError("RLE run start and length must be positive integers")
        end = start + length
        if start < 0 or end > limit:
            raise ValueError("RLE run exceeds image bounds")
        if start < prev_end:
            raise ValueError("RLE runs must be sorted and non-overlapping")
        if start == prev_end:
            intervals[-1] = (intervals[-1][0], end)
        else:
            intervals.append((start, end))
        prev_end = end
    return height, width, intervals


def _intervals_to_rle(
    height: int, width: int, intervals: list[tuple[int, int]]
) -> dict[str, list[int]]:
    counts: list[int] = []
    for start, end in intervals:
        if end > start:
            counts.append(int(start))
            counts.append(int(end - start))
    return {"size": [int(height), int(width)], "counts": counts}


def encode_mask_rle(mask: np.ndarray) -> dict[str, list[int]]:
    """Encode a 2D boolean mask into row-major (start, length) run intervals."""
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError("mask must be a two-dimensional array")
    flat = values.reshape(-1)
    if not flat.size:
        counts: list[int] = []
    else:
        # Find both run boundaries in one scan; preserve row-major wire bytes.
        edges = np.flatnonzero(flat[1:] != flat[:-1]) + 1
        if flat[0]:
            edges = np.concatenate(([0], edges))
        if flat[-1]:
            edges = np.concatenate((edges, [flat.size]))
        runs = edges.reshape(-1, 2)
        counts = np.column_stack((runs[:, 0], runs[:, 1] - runs[:, 0])).reshape(-1).tolist()
    return {"size": [int(values.shape[0]), int(values.shape[1])], "counts": counts}


def decode_mask_rle(payload: Any, *, width: int, height: int) -> np.ndarray | None:
    """Decode a row-major RLE dictionary into a (height, width) boolean ndarray."""
    if not isinstance(payload, dict):
        return None
    size = payload.get("size")
    counts = payload.get("counts")
    if not (
        isinstance(size, list)
        and len(size) == 2
        and int(size[0]) == int(height)
        and int(size[1]) == int(width)
        and isinstance(counts, list)
        and len(counts) % 2 == 0
    ):
        return None
    flat = np.zeros(int(width) * int(height), dtype=bool)
    for raw_start, raw_length in zip(counts[0::2], counts[1::2], strict=False):
        if not isinstance(raw_start, int) or not isinstance(raw_length, int) or raw_length <= 0:
            return None
        end = raw_start + raw_length
        if raw_start < 0 or end > len(flat):
            return None
        flat[raw_start:end] = True
    return flat.reshape((int(height), int(width)))


def rle_area(payload: Any) -> int:
    """Compute exact foreground pixel count in O(R) time without decoding."""
    _h, _w, intervals = _validated_runs(payload)
    return sum(end - start for start, end in intervals)


def rle_intersection_runs(rle_a: Any, rle_b: Any) -> dict[str, list[int]]:
    """Compute exact intersection R_A ∩ R_B in O(R_A + R_B) time."""
    h_a, w_a, runs_a = _validated_runs(rle_a)
    h_b, w_b, runs_b = _validated_runs(rle_b)
    if (h_a, w_a) != (h_b, w_b):
        raise ValueError("RLE masks must share the same [height, width] dimensions")
    out: list[tuple[int, int]] = []
    i, j = 0, 0
    n_a, n_b = len(runs_a), len(runs_b)
    while i < n_a and j < n_b:
        s_a, e_a = runs_a[i]
        s_b, e_b = runs_b[j]
        start = s_a if s_a > s_b else s_b
        end = e_a if e_a < e_b else e_b
        if start < end:
            out.append((start, end))
        if e_a <= e_b:
            i += 1
        else:
            j += 1
    return _intervals_to_rle(h_a, w_a, out)


def rle_union_runs(rle_a: Any, rle_b: Any) -> dict[str, list[int]]:
    """Compute exact union R_A ∪ R_B in O(R_A + R_B) time."""
    h_a, w_a, runs_a = _validated_runs(rle_a)
    h_b, w_b, runs_b = _validated_runs(rle_b)
    if (h_a, w_a) != (h_b, w_b):
        raise ValueError("RLE masks must share the same [height, width] dimensions")
    out: list[tuple[int, int]] = []
    i, j = 0, 0
    n_a, n_b = len(runs_a), len(runs_b)
    while i < n_a or j < n_b:
        if j >= n_b or (i < n_a and runs_a[i][0] <= runs_b[j][0]):
            cur_s, cur_e = runs_a[i]
            i += 1
        else:
            cur_s, cur_e = runs_b[j]
            j += 1
        if out and cur_s <= out[-1][1]:
            if cur_e > out[-1][1]:
                out[-1] = (out[-1][0], cur_e)
        else:
            out.append((cur_s, cur_e))
    return _intervals_to_rle(h_a, w_a, out)


def rle_difference_runs(rle_a: Any, rle_b: Any) -> dict[str, list[int]]:
    """Compute exact set difference R_A \\ R_B in O(R_A + R_B) time."""
    h_a, w_a, runs_a = _validated_runs(rle_a)
    h_b, w_b, runs_b = _validated_runs(rle_b)
    if (h_a, w_a) != (h_b, w_b):
        raise ValueError("RLE masks must share the same [height, width] dimensions")
    out: list[tuple[int, int]] = []
    j = 0
    n_b = len(runs_b)
    for s_a, e_a in runs_a:
        cur = s_a
        while j < n_b and runs_b[j][1] <= cur:
            j += 1
        k = j
        while k < n_b and runs_b[k][0] < e_a:
            s_b, e_b = runs_b[k]
            if s_b > cur:
                out.append((cur, min(s_b, e_a)))
            cur = max(cur, e_b)
            if cur >= e_a:
                break
            k += 1
        if cur < e_a:
            out.append((cur, e_a))
    return _intervals_to_rle(h_a, w_a, out)


def rle_intersection_area(rle_a: Any, rle_b: Any) -> int:
    """Compute exact intersection pixel count |R_A ∩ R_B| in O(R_A + R_B) time."""
    h_a, w_a, runs_a = _validated_runs(rle_a)
    h_b, w_b, runs_b = _validated_runs(rle_b)
    if (h_a, w_a) != (h_b, w_b):
        raise ValueError("RLE masks must share the same [height, width] dimensions")
    inter = 0
    i, j = 0, 0
    n_a, n_b = len(runs_a), len(runs_b)
    while i < n_a and j < n_b:
        s_a, e_a = runs_a[i]
        s_b, e_b = runs_b[j]
        start = s_a if s_a > s_b else s_b
        end = e_a if e_a < e_b else e_b
        if start < end:
            inter += end - start
        if e_a <= e_b:
            i += 1
        else:
            j += 1
    return inter


def rle_iou(rle_a: Any, rle_b: Any) -> float:
    """Compute exact pairwise IoU(R_A, R_B) in O(R_A + R_B) time without decoding."""
    h_a, w_a, runs_a = _validated_runs(rle_a)
    h_b, w_b, runs_b = _validated_runs(rle_b)
    if (h_a, w_a) != (h_b, w_b):
        raise ValueError("RLE masks must share the same [height, width] dimensions")
    area_a = sum(e - s for s, e in runs_a)
    area_b = sum(e - s for s, e in runs_b)
    if area_a == 0 and area_b == 0:
        return 1.0
    if area_a == 0 or area_b == 0:
        return 0.0
    inter = 0
    i, j = 0, 0
    n_a, n_b = len(runs_a), len(runs_b)
    while i < n_a and j < n_b:
        s_a, e_a = runs_a[i]
        s_b, e_b = runs_b[j]
        start = s_a if s_a > s_b else s_b
        end = e_a if e_a < e_b else e_b
        if start < end:
            inter += end - start
        if e_a <= e_b:
            i += 1
        else:
            j += 1
    union = area_a + area_b - inter
    return float(inter) / float(union) if union > 0 else 0.0
