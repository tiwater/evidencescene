"""Pure conditional diameter diagnostics; no association, filtering or pose gate."""
import numpy as np


def bound_witness(points, maximum_diameter):
    if isinstance(maximum_diameter, bool) or not np.isfinite(maximum_diameter) or maximum_diameter <= 0:
        raise ValueError("maximum diameter must be finite and positive")
    xyz = np.asarray(points, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("points must be finite Nx3")
    result = {"maximum_diameter_m": float(maximum_diameter), "point_count": len(xyz),
              "points_filtered": 0, "publication_authorized": False}
    if not len(xyz):
        return {**result, "status": "unobservable", "diameter_lower_bound_m": None,
                "diameter_upper_bound_m": None, "witness_camera_points_m": None}
    indices = np.unique(np.concatenate((xyz.argmin(axis=0), xyz.argmax(axis=0))))
    extremes = xyz[indices]
    differences = extremes[:, None, :] - extremes[None, :, :]
    distances = np.linalg.norm(differences, axis=2)
    first, second = np.unravel_index(distances.argmax(), distances.shape)
    lower = float(distances[first, second])
    upper = float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0)))
    status = ("conflict_with_declared_prior" if lower > maximum_diameter else
              "within_bound_not_association_proof" if upper <= maximum_diameter else "inconclusive")
    return {**result, "status": status, "diameter_lower_bound_m": lower,
            "diameter_upper_bound_m": upper,
            "witness_camera_points_m": [extremes[first].tolist(), extremes[second].tolist()],
            "witness_source_point_indices": [int(indices[first]), int(indices[second])],
            "camera_depth_min_m": float(xyz[:, 2].min()),
            "camera_depth_median_m": float(np.median(xyz[:, 2])),
            "camera_depth_max_m": float(xyz[:, 2].max())}

