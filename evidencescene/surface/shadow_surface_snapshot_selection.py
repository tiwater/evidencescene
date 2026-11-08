"""Stateless existing-point snapshot selection for offline shadow evaluation.

The live surface is owned by CausalObjectSurface/RevisableSurfelMap. This
module does not mutate that map, run an estimator, or authorize publication.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from evidencescene.surface.object_surface_fusion import _pack_3d_keys


def select_existing_points_shadow(
    points: Any, support: Any, cameras: Any, voxel_keys: Any, *, budget: int = 1500,
    minimum_coarse_support: int = 10, voxel_size_m: float = 0.004,
    coarse_cell_m: float = 0.012,
) -> dict[str, Any]:
    """Return indices into one complete, identity-checked owner snapshot.

    When the coarse candidate cannot supply the budget, return the exact
    existing owner priority instead. Keys must be actual object-gauge voxel
    keys for the supplied points; the adapter does not infer object identity.
    """
    if type(budget) is not int or budget <= 0 or type(minimum_coarse_support) is not int or minimum_coarse_support <= 0:
        raise ValueError('budget and support floor must be positive integers')
    for value in (voxel_size_m, coarse_cell_m):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('cell sizes must be finite positive numbers')
    try:
        xyz = np.asarray(points, dtype=np.float64)
        counts = np.asarray(support)
        views = np.asarray(cameras)
        keys = np.asarray(voxel_keys)
    except (TypeError, ValueError) as exc:
        raise ValueError('invalid snapshot arrays') from exc
    n = len(xyz) if xyz.ndim == 2 else -1
    if (xyz.shape != (n, 3) or not np.isfinite(xyz).all()
            or keys.shape != (n, 3) or not np.issubdtype(keys.dtype, np.signedinteger)
            or counts.shape != (n,) or views.shape != (n,)
            or not np.issubdtype(counts.dtype, np.signedinteger)
            or not np.issubdtype(views.dtype, np.signedinteger)
            or np.any(counts < 2) or np.any(views < 1)):
        raise ValueError('invalid supported snapshot arrays')
    scaled = np.floor(xyz / voxel_size_m)
    # Keep enough headroom to avoid undefined integer casts at the endpoints.
    if (not np.isfinite(scaled).all() or np.any(np.abs(scaled) >= 2**62)
            or not np.array_equal(scaled.astype(np.int64), keys)):
        raise ValueError('ambiguous voxel keys')
    packed_keys, key_span = _pack_3d_keys(keys.astype(np.int64, copy=False))
    if packed_keys is not None:
        if n <= 1 or bool(np.all(packed_keys[1:] > packed_keys[:-1])):
            lex = np.arange(n, dtype=np.intp)
        else:
            lex = np.argsort(packed_keys, kind="stable")
            sorted_pk = packed_keys[lex]
            if bool(np.any(sorted_pk[1:] == sorted_pk[:-1])):
                raise ValueError('ambiguous voxel keys')
        if n == 0:
            priority = np.empty(0, dtype=np.intp)
        else:
            c_max, c_min = int(counts.max()), int(counts.min())
            v_max, v_min = int(views.max()), int(views.min())
            span_c = c_max - c_min + 1
            span_v = v_max - v_min + 1
            if span_c == 1 and span_v == 1:
                priority = lex
            elif span_v * span_c * key_span <= np.iinfo(np.int64).max:
                packed_prio = (
                    ((v_max - views) * span_c + (c_max - counts)) * key_span
                    + packed_keys
                )
                if bool(np.all(packed_prio[1:] > packed_prio[:-1])):
                    priority = np.arange(n, dtype=np.intp)
                else:
                    priority = np.argsort(packed_prio, kind="stable")
            else:
                priority = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0], -counts, -views))
    else:
        if len(np.unique(keys, axis=0)) != n:
            raise ValueError('ambiguous voxel keys')
        lex = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
        priority = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0], -counts, -views))
    owner = priority[:budget] if n > budget else lex
    eligible = priority[counts[priority] >= minimum_coarse_support]
    base = {'wire_authorized': False, 'eligible_count': len(eligible), 'input_count': n}
    if n <= budget or len(eligible) < budget:
        return {**base, 'mode': 'owner_priority', 'indices': owner,
                'reason': 'within_budget' if n <= budget else 'insufficient_coarse_support'}
    coarse = np.floor(xyz[eligible] / coarse_cell_m)
    if not np.isfinite(coarse).all() or np.any(np.abs(coarse) >= 2**62):
        raise ValueError('invalid coarse keys')
    coarse_i64 = coarse.astype(np.int64)
    packed_coarse, _ = _pack_3d_keys(coarse_i64)
    if packed_coarse is not None:
        _, first = np.unique(packed_coarse, return_index=True)
    else:
        _, first = np.unique(coarse_i64, axis=0, return_index=True)
    representatives = eligible[np.sort(first)]
    if len(representatives) >= budget:
        selected = representatives[:budget]
    else:
        used = np.zeros(n, dtype=bool)
        used[representatives] = True
        selected = np.concatenate((representatives, eligible[~used[eligible]][:budget-len(representatives)]))
    return {**base, 'mode': 'coarse_shadow', 'indices': selected, 'reason': None}
