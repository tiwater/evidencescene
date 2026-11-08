"""Smoke and contract verification tests for standalone evidencescene package."""

from __future__ import annotations

import numpy as np

import evidencescene
from evidencescene.observability.shadow_pose_adapter import (
    adapt_shadow_pose,
    adapt_shadow_poses,
)
from evidencescene.runtime.request_evidence_deduplication import (
    RequestDeduplicationStats,
    rle_mask_box_and_centroid,
)
from evidencescene.surface.causal_object_surface import CausalObjectSurface
from evidencescene.surface.object_surface_fusion import RevisableSurfelMap
from evidencescene.tracking.object_centric_observation_session import (
    ObjectCentricObservationSession,
)
from evidencescene.tracking.pinned_bundletrack_runner import PinnedBundleTrackSession


def test_package_exports_and_version() -> None:
    assert evidencescene.__version__ == "2.0.0"
    assert CausalObjectSurface is not None
    assert RevisableSurfelMap is not None
    assert ObjectCentricObservationSession is not None
    assert PinnedBundleTrackSession is not None
    assert RequestDeduplicationStats is not None


def test_shadow_pose_adapter_fail_closed_and_valid_composition() -> None:
    eye = np.eye(4, dtype=np.float64)
    blocked = adapt_shadow_pose(
        camera_current_from_camera_initial=eye,
        camera_initial_from_object=eye,
        lost=True,
    )
    assert blocked["status"] == "not_observable"
    assert blocked["camera_from_object"] is None
    assert blocked["wire_authorized"] is False

    candidate = adapt_shadow_pose(
        camera_current_from_camera_initial=eye,
        camera_initial_from_object=eye,
        lost=False,
    )
    assert candidate["status"] == "candidate"
    assert candidate["wire_authorized"] is False
    assert np.allclose(np.asarray(candidate["camera_from_object"]), eye)

    batch = adapt_shadow_poses(
        object_keys=["carton"],
        camera_current_from_camera_initial=[eye],
        camera_initial_from_object={"carton": eye},
        lost=[False],
    )
    assert len(batch) == 1
    assert batch[0]["status"] == "candidate"


def test_rle_run_interval_geometry() -> None:
    # 10x10 image with two runs on rows 2..3, cols 4..6 (start=24, len=3; start=34, len=3)
    width, height = 10, 10
    rle = {"size": [height, width], "counts": [24, 3, 34, 3]}

    geom = rle_mask_box_and_centroid(rle, width=width, height=height)
    assert geom is not None
    bbox, centroid = geom
    assert bbox == (4, 2, 6, 3)
    assert np.isclose(centroid[0], 5.0)
    assert np.isclose(centroid[1], 2.5)


def test_surface_snapshot_matches_reference_priority() -> None:
    import numpy as np

    from evidencescene.surface.object_surface_fusion import RevisableSurfelMap

    base = np.array(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]],
        dtype=np.float64,
    )
    surfel_map = RevisableSurfelMap(
        voxel_size_m=0.005,
        maximum_voxels=128,
        maximum_live_bytes=1 << 20,
    )
    surfel_map.update({"chest": ("chest", base), "head": ("head", base)})
    # Add a support-1 and a three-camera voxel to exercise both sort keys.
    surfel_map.update({"wrist": ("wrist", np.array([[0.02, 0.0, 0.0]]))})

    for budget in (1, 3, 8):
        points, support, cameras, keys = surfel_map.snapshot(
            minimum_observation_support=1,
            maximum_point_count=budget,
            include_voxel_keys=True,
        )
        rows = []
        for key, value in surfel_map._totals.items():
            if value[1] >= 1:
                rows.append((key, value[1], len(value[2]), value[0] / value[1]))
        rows.sort(key=lambda row: (-row[1], -row[2], row[0][0], row[0][1], row[0][2]))
        rows = rows[:budget]
        assert len(points) == len(rows)
        np.testing.assert_array_equal(keys, np.asarray([row[0] for row in rows]))
        np.testing.assert_array_equal(support, np.asarray([row[1] for row in rows]))
        np.testing.assert_array_equal(cameras, np.asarray([row[2] for row in rows]))
        np.testing.assert_allclose(points, np.asarray([row[3] for row in rows]), rtol=0, atol=0)


def test_voxel_centroid_accumulation_is_exact() -> None:
    import numpy as np

    from evidencescene.surface.object_surface_fusion import observation_voxel_centroids

    rng = np.random.default_rng(11)
    keys = rng.integers(0, 8, size=(400, 3), dtype=np.int64)
    points = keys.astype(np.float64) * 0.004 + rng.normal(0.0, 1e-7, (400, 3))
    unique, centroids = observation_voxel_centroids(points, 0.004)
    cell_keys = np.floor(points / 0.004).astype(np.int64)
    groups: dict[tuple[int, int, int], list[np.ndarray]] = {}
    for key, point in zip(map(tuple, cell_keys.tolist()), points, strict=True):
        groups.setdefault(key, []).append(point)
    ordered = sorted(groups)
    np.testing.assert_array_equal(unique, np.asarray(ordered, dtype=np.int64))
    expected = np.asarray([np.mean(groups[key], axis=0) for key in ordered])
    np.testing.assert_array_equal(centroids, expected)
