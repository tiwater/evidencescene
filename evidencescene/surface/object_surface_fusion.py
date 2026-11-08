"""Revisable form of the original per-observation multicamera surfel fusion.

Each observation contributes one centroid per voxel, regardless of its pixel
count. Retiring a native frame commits its last contribution; it does not erase
the observed surface. Object identity, coordinate gauge and chronology belong
to the caller. This module estimates neither pose nor unobserved geometry.
"""
import itertools
import math

import numpy as np


def measured_free_space_margin(
    object_points, depth_m, intrinsics, camera_from_object, *,
    minimum_depth_m, maximum_depth_m, depth_scale_m=1.0,
    return_projection_details=False,
):
    """Return (known, margin_m) using four surrounding registered depth pixels.

    Positive margin means the point is in front of every sampled depth endpoint.
    The caller supplies the uncertainty threshold; unknown margins are NaN.
    Missing depth, image edges and points behind the camera provide no evidence.
    depth_scale_m converts input depth units to meters after sampling, so an
    integer depth image need not be copied and converted for every point set.
    This is an offline evidence primitive, not an object-membership decision.

    Existing producer: CausalObjectSurface.update; state owner: RevisableSurfelMap.
    Neither is modified here. Callers own registered input provenance, timestamps,
    calibration and the BundleTrack object gauge, including pose revisions.
    No pose, mask, surface state or publication is produced by this function.
    """
    if type(return_projection_details)is not bool:
        raise ValueError('Explicit projection-details boolean required')
    points = np.asarray(object_points, dtype=np.float64)
    depth = np.asarray(depth_m)
    K = np.asarray(intrinsics, dtype=np.float64)
    T = np.asarray(camera_from_object, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("object points must be finite Nx3")
    if depth.ndim != 2 or min(depth.shape) < 2:
        raise ValueError("registered metric depth must be at least 2x2")
    if (not np.issubdtype(depth.dtype, np.number)
            or np.issubdtype(depth.dtype, np.complexfloating)):
        raise ValueError("registered depth must contain real numeric samples")
    if not math.isfinite(depth_scale_m) or depth_scale_m <= 0:
        raise ValueError("depth scale must be finite and positive")
    if (K.shape != (3, 3) or not np.isfinite(K).all()
            or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.array_equal(K[2], [0., 0., 1.])):
        raise ValueError("finite pinhole intrinsics required")
    if (T.shape != (4, 4) or not np.isfinite(T).all()
            or not np.allclose(T[3], [0., 0., 0., 1.], atol=1e-8, rtol=0)
            or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5, rtol=0)
            or not np.isclose(np.linalg.det(T[:3, :3]), 1., atol=1e-5, rtol=0)):
        raise ValueError("rigid camera_from_object transform required")
    if not (math.isfinite(minimum_depth_m) and math.isfinite(maximum_depth_m)
            and 0 < minimum_depth_m < maximum_depth_m):
        raise ValueError("explicit positive finite depth interval required")
    known = np.zeros(len(points), dtype=bool)
    margin = np.full(len(points), np.nan)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        camera_points = points @ T[:3, :3].T + T[:3, 3]
        indices = np.flatnonzero(
            np.isfinite(camera_points).all(axis=1) & (camera_points[:, 2] > 0))
        projected = camera_points[indices] @ K.T
        uv = projected[:, :2] / projected[:, 2:]
    inside = (np.isfinite(uv).all(axis=1)
              & (uv[:, 0] >= 0) & (uv[:, 0] < depth.shape[1] - 1)
              & (uv[:, 1] >= 0) & (uv[:, 1] < depth.shape[0] - 1))
    indices = indices[inside]
    x, y = np.floor(uv[inside]).astype(np.int64).T
    samples = np.column_stack((depth[y, x], depth[y, x + 1],
                               depth[y + 1, x], depth[y + 1, x + 1])).astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        samples *= depth_scale_m
    valid = (np.isfinite(samples) & (samples >= minimum_depth_m)
             & (samples <= maximum_depth_m)).all(axis=1)
    selected = indices[valid]
    known[selected] = True
    margin[selected] = (samples[valid] - camera_points[selected, 2, None]).min(axis=1)
    if return_projection_details:
        return known, margin, dict(point_count=len(points),source_point_indices=selected,
            pixels_floor_uv=np.column_stack((x[valid],y[valid])),depth_samples_m=samples[valid],
            projected_depth_m=camera_points[selected,2])
    return known, margin


def conditional_mask_depth_conflict(details, mask, *, maximum_depth_residual_m=.03, maximum_span_m=.02):
    """Shadow ambiguity only, using original projection's actual four samples.

    No visibility/material truth is inferred. Unknown/occluded/mixed samples
    cannot supply a conditional current-mask conflict.
    """
    mask=np.asarray(mask)
    if mask.ndim!=2 or mask.dtype!=np.bool_ or any(not math.isfinite(v) or v<=0 for v in (maximum_depth_residual_m,maximum_span_m)):
        raise ValueError('Boolean current mask and positive finite bounds required')
    count=details['point_count'];ids=np.asarray(details['source_point_indices']);xy=np.asarray(details['pixels_floor_uv'])
    samples=np.asarray(details['depth_samples_m']);z=np.asarray(details['projected_depth_m'])
    if (type(count)is not int or count<0 or ids.ndim!=1 or xy.shape!=(len(ids),2) or samples.shape!=(len(ids),4) or z.shape!=(len(ids),)
            or not np.issubdtype(ids.dtype,np.integer) or not np.issubdtype(xy.dtype,np.integer)
            or ((ids<0)|(ids>=count)).any() or len(np.unique(ids))!=len(ids)
            or not np.isfinite(samples).all() or not np.isfinite(z).all() or (samples<=0).any() or (z<=0).any()):
        raise ValueError('Original finite valid projection samples required')
    x,y=xy.T
    if ((x<0)|(x>=mask.shape[1]-1)|(y<0)|(y>=mask.shape[0]-1)).any():raise ValueError('Projection/mask grid mismatch')
    outside=~(mask[y,x]|mask[y,x+1]|mask[y+1,x]|mask[y+1,x+1])
    compatible=(np.abs(samples-z[:,None])<=maximum_depth_residual_m).all(axis=1)
    coherent=np.ptp(samples,axis=1)<=maximum_span_m
    result=np.zeros(count,dtype=bool);result[ids]=outside&compatible&coherent
    return result


def _pack_3d_keys(keys: np.ndarray) -> tuple[np.ndarray | None, int]:
    """Pack Nx3 int64 voxel keys into 1D int64 values when bounding box span fits."""
    if len(keys) == 0:
        return np.empty(0, dtype=np.int64), 1
    lower = keys.min(axis=0)
    upper = keys.max(axis=0)
    s0 = int(upper[0]) - int(lower[0]) + 1
    s1 = int(upper[1]) - int(lower[1]) + 1
    s2 = int(upper[2]) - int(lower[2]) + 1
    total_span = s0 * s1 * s2
    if total_span <= 0 or total_span > np.iinfo(np.int64).max:
        return None, 0
    packed = (
        (keys[:, 0] - int(lower[0])) * (s1 * s2)
        + (keys[:, 1] - int(lower[1])) * s2
        + (keys[:, 2] - int(lower[2]))
    )
    return packed, total_span


def observation_voxel_centroids(raw_points, voxel_size_m):
    points = np.asarray(raw_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("each surface observation must be finite Nx3")
    if not math.isfinite(voxel_size_m) or voxel_size_m <= 0:
        raise ValueError("surfel voxel size must be finite and positive")
    scaled = np.floor(points / voxel_size_m)
    if np.any(np.abs(scaled) >= float(np.iinfo(np.int64).max)):
        raise ValueError("surface voxel index overflow")
    keys = scaled.astype(np.int64)
    if not len(keys):
        return np.empty((0, 3), dtype=np.int64), np.empty((0, 3), dtype=np.float64)
    # Preserve lexicographic voxel order and original point-order summation.
    packed, _ = _pack_3d_keys(keys)
    if packed is not None:
        if len(packed) <= 1 or bool(np.all(packed[1:] > packed[:-1])):
            return keys, points.copy()
        order = np.argsort(packed, kind="stable")
        sorted_packed = packed[order]
        starts = np.r_[0, np.flatnonzero(sorted_packed[1:] != sorted_packed[:-1]) + 1]
        unique = keys[order[starts]]
    else:
        order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
        sorted_keys = keys[order]
        starts = np.r_[0, np.flatnonzero(np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)) + 1]
        unique = sorted_keys[starts]
    counts = np.diff(np.r_[starts, len(keys)])
    inverse = np.empty(len(keys), dtype=np.intp)
    inverse[order] = np.repeat(np.arange(len(starts)), counts)
    # np.add.at is a notoriously slow unbuffered ufunc; bincount performs the
    # same sequential accumulation in C and is bit-identical for these inputs.
    sums = np.empty((len(unique), 3), dtype=np.float64)
    for axis in range(3):
        sums[:, axis] = np.bincount(
            inverse,
            weights=points[:, axis],
            minlength=len(unique),
        )
    return unique, sums / counts[:, None]


def _index_surface_events(live, prepared):
    staged = []
    for identity, contribution in prepared.items():
        old = [(live[identity], -1)] if identity in live else []
        for (camera, keys, centroids), sign in [*old, (contribution, 1)]:
            staged.append((camera, keys, centroids, sign))
    if len(staged) == 1:
        camera, keys, centroids, sign = staged[0]
        n = len(keys)
        indices = {key: i for i, key in enumerate(zip(*keys.T.tolist()))}
        return [(camera, np.arange(n, dtype=np.intp), centroids, sign)], indices
    count = sum(len(keys) for _, keys, _, _ in staged)
    if count >= 4096:
        keys = np.concatenate([k for _, k, _, _ in staged])
        packed, _ = _pack_3d_keys(keys)
        if packed is not None:
            _, first, inverse = np.unique(packed, return_index=True, return_inverse=True)
            order = np.argsort(first)
            remap = np.empty(len(order), dtype=np.intp)
            remap[order] = np.arange(len(order))
            ids = remap[inverse]
            ordered_keys = keys[first[order]]
            indices = {key: i for i, key in enumerate(zip(*ordered_keys.T.tolist()))}
            events, start = [], 0
            for camera, keys, centroids, sign in staged:
                end = start + len(keys)
                events.append((camera, ids[start:end], centroids, sign))
                start = end
            return events, indices
    events, indices = [], {}
    for camera, keys, centroids, sign in staged:
        ids = [indices.setdefault(key, len(indices))
               for key in zip(*keys.T.tolist())]
        events.append((camera, np.asarray(ids, dtype=np.intp), centroids, sign))
    return events, indices


class RevisableSurfelMap:
    def __init__(self, *, voxel_size_m, maximum_voxels, maximum_live_bytes):
        if not math.isfinite(voxel_size_m) or voxel_size_m <= 0:
            raise ValueError("positive finite voxel size required")
        if any(type(v) is not int or v <= 0 for v in (maximum_voxels, maximum_live_bytes)):
            raise ValueError("positive explicit surface resource budgets required")
        self.voxel_size_m = voxel_size_m
        self.maximum_voxels = maximum_voxels
        self.maximum_live_bytes = maximum_live_bytes
        self._totals_dict = None
        self._col_keys = np.empty((0, 3), dtype=np.int64)
        self._col_sums = np.empty((0, 3), dtype=np.float64)
        self._col_support = np.empty(0, dtype=np.int64)
        self._col_cam_counts = {}
        self._live = {}

    @property
    def _totals(self):
        if self._totals_dict is None:
            totals = {}
            keys_list = list(zip(*self._col_keys.T.tolist())) if len(self._col_keys) else []
            sup_list = self._col_support.tolist()
            cam_items = [(cam, arr.tolist()) for cam, arr in self._col_cam_counts.items()]
            for i, key in enumerate(keys_list):
                cams = {}
                for cam, c_list in cam_items:
                    cnt = c_list[i]
                    if cnt:
                        cams[cam] = cnt
                totals[key] = [self._col_sums[i].copy(), sup_list[i], cams]
            self._totals_dict = totals
        return self._totals_dict

    @_totals.setter
    def _totals(self, value):
        self._totals_dict = value

    def _sync_columnar_from_totals_dict(self):
        if self._totals_dict is None:
            return
        totals = self._totals_dict
        if not totals:
            self._col_keys = np.empty((0, 3), dtype=np.int64)
            self._col_sums = np.empty((0, 3), dtype=np.float64)
            self._col_support = np.empty(0, dtype=np.int64)
            self._col_cam_counts = {}
            self._totals_dict = None
            return
        n = len(totals)
        keys = np.fromiter(
            itertools.chain.from_iterable(totals.keys()),
            dtype=np.int64,
            count=n * 3,
        ).reshape(n, 3)
        order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
        vals = list(totals.values())
        self._col_keys = keys[order]
        self._col_sums = np.array([vals[i][0] for i in order], dtype=np.float64)
        self._col_support = np.fromiter((vals[i][1] for i in order), dtype=np.int64, count=n)
        all_cams = set()
        for v in vals:
            all_cams.update(v[2].keys())
        cam_counts = {}
        for cam in all_cams:
            arr = np.fromiter((vals[i][2].get(cam, 0) for i in order), dtype=np.int64, count=n)
            if np.any(arr):
                cam_counts[cam] = arr
        self._col_cam_counts = cam_counts
        self._totals_dict = None

    def update(self, observations, *, retire=()):
        """Atomically replace live contributions, then retire native ownership.

        Voxel keys are unique within each contribution. Batched additions keep
        observation order and reset sums exactly when support becomes zero.
        Camera staging is dense only when its arrays are no larger than the
        coordinate-sum/support staging; broader camera sets stay sparse.
        Resource exhaustion rejects the update before changing owner state.
        """
        prepared = {}
        for identity, (camera, points) in observations.items():
            if not isinstance(camera, str) or not camera:
                raise ValueError('surface camera identity required')
            keys, centroids = observation_voxel_centroids(points, self.voxel_size_m)
            prepared[identity] = (camera, keys, centroids)
        live = {**self._live, **prepared}
        for identity in retire:
            if identity not in live:
                raise ValueError('cannot retire an unknown surface observation')
            del live[identity]
        if sum(k.nbytes + p.nbytes for _, k, p in live.values()) > self.maximum_live_bytes:
            raise ValueError('surface live contribution memory budget exceeded')

        if self._totals_dict is not None:
            self._sync_columnar_from_totals_dict()

        staged = []
        for identity, contribution in prepared.items():
            old = [(self._live[identity], -1)] if identity in self._live else []
            for (camera, keys, centroids), sign in [*old, (contribution, 1)]:
                if len(keys):
                    staged.append((camera, keys, centroids, sign))

        if not staged:
            self._live = live
            return

        v_old = len(self._col_keys)
        # Fast Path 1: Inserting a single observation into an empty map
        if v_old == 0 and len(staged) == 1 and staged[0][3] == 1:
            camera, keys, centroids, _ = staged[0]
            n = len(keys)
            if n > self.maximum_voxels:
                raise ValueError('surface voxel budget exceeded')
            self._col_keys = keys
            self._col_sums = centroids.copy()
            self._col_support = np.ones(n, dtype=np.int64)
            self._col_cam_counts = {camera: np.ones(n, dtype=np.int64)}
            self._live = live
            return

        # Fast Path 2: Single addition with identical voxel keys as current map
        if v_old > 0 and len(staged) == 1 and staged[0][3] == 1:
            camera, keys, centroids, _ = staged[0]
            if len(keys) == v_old and np.array_equal(self._col_keys, keys):
                self._col_sums = self._col_sums + centroids
                self._col_support = self._col_support + 1
                if camera in self._col_cam_counts:
                    self._col_cam_counts[camera] = self._col_cam_counts[camera] + 1
                else:
                    self._col_cam_counts[camera] = np.ones(v_old, dtype=np.int64)
                self._live = live
                return

        # General Columnar Update Path
        key_blocks = [self._col_keys] if v_old > 0 else []
        key_blocks.extend(k for _, k, _, _ in staged)
        all_keys = np.concatenate(key_blocks, axis=0) if len(key_blocks) > 1 else key_blocks[0]
        packed, _ = _pack_3d_keys(all_keys)
        if packed is not None:
            _, first_idx, inverse = np.unique(packed, return_index=True, return_inverse=True)
            u_count = len(first_idx)
            unique_keys = all_keys[first_idx]
        else:
            order = np.lexsort((all_keys[:, 2], all_keys[:, 1], all_keys[:, 0]))
            s_keys = all_keys[order]
            starts = np.r_[0, np.flatnonzero(np.any(s_keys[1:] != s_keys[:-1], axis=1)) + 1]
            unique_keys = s_keys[starts]
            u_count = len(starts)
            counts = np.diff(np.r_[starts, len(all_keys)])
            inverse = np.empty(len(all_keys), dtype=np.intp)
            inverse[order] = np.repeat(np.arange(u_count), counts)

        sums = np.zeros((u_count, 3), dtype=np.float64)
        support = np.zeros(u_count, dtype=np.int64)
        cam_names = set(self._col_cam_counts.keys())
        for camera, _, _, _ in staged:
            cam_names.add(camera)
        cam_counts = {cam: np.zeros(u_count, dtype=np.int64) for cam in cam_names}

        offset = 0
        if v_old > 0:
            old_ids = inverse[:v_old]
            sums[old_ids] = self._col_sums
            support[old_ids] = self._col_support
            for cam, arr in self._col_cam_counts.items():
                cam_counts[cam][old_ids] = arr
            offset = v_old

        for camera, keys, centroids, sign in staged:
            n_k = len(keys)
            ids = inverse[offset : offset + n_k]
            offset += n_k
            sums[ids] += sign * centroids
            support[ids] += sign
            cam_counts[camera][ids] += sign
            zero_mask = support[ids] == 0
            if np.any(zero_mask):
                sums[ids[zero_mask]] = 0.0

        keep = support > 0
        resulting_count = int(np.count_nonzero(keep))
        if resulting_count > self.maximum_voxels:
            raise ValueError('surface voxel budget exceeded')

        if resulting_count == u_count:
            self._col_keys = unique_keys
            self._col_sums = sums
            self._col_support = support
            self._col_cam_counts = {
                cam: arr for cam, arr in cam_counts.items() if np.any(arr)
            }
        else:
            self._col_keys = unique_keys[keep]
            self._col_sums = sums[keep]
            self._col_support = support[keep]
            self._col_cam_counts = {
                cam: arr[keep] for cam, arr in cam_counts.items() if np.any(arr[keep])
            }
        self._live = live

    def shadow_discard_retired_contributions(self, voxel_keys):
        """Offline experiment primitive; caller must supply scored evidence.

        Not called by the production owner. Preserve every live contribution
        so later native pose revisions can still subtract/replace it exactly.
        The original replay is the recovery source for discarded retired data.
        This method makes no visibility or validity decision of its own.
        """
        keys = set()
        for raw in voxel_keys:
            if len(raw) != 3 or any(not isinstance(v, (int, np.integer)) or isinstance(v, bool) for v in raw):
                raise ValueError("integer voxel keys required")
            keys.add(tuple(int(v) for v in raw))
        keys.intersection_update(self._totals)
        if not keys:
            return dict(mode="offline_shadow", discarded_retired_observation_voxel_contributions=0,
                        publication_authorized=False)
        # Exact integer-key membership in NumPy; only matching contributions
        # enter Python. Keep their original order for identical accumulation.
        key_dtype = np.dtype((np.void, 3 * np.dtype(np.int64).itemsize))
        target_keys = np.ascontiguousarray(list(keys), dtype=np.int64).view(key_dtype).ravel()
        retained = {key: [np.zeros(3), 0, {}] for key in keys}
        for camera, live_keys, centroids in self._live.values():
            matches = np.isin(np.ascontiguousarray(live_keys, dtype=np.int64).view(key_dtype).ravel(), target_keys)
            for raw_key, centroid in zip(live_keys[matches], centroids[matches], strict=True):
                key = tuple(raw_key)
                if key not in retained:
                    continue
                value = retained[key]
                value[0] += centroid
                value[1] += 1
                value[2][camera] = value[2].get(camera, 0) + 1
        removed = 0
        for key, value in retained.items():
            removed += self._totals[key][1] - value[1]
            if value[1]:
                self._totals[key] = value
            else:
                self._totals.pop(key)
        return dict(mode="offline_shadow", discarded_retired_observation_voxel_contributions=removed,
                    publication_authorized=False)

    def snapshot(self, *, minimum_observation_support, maximum_point_count,
                 include_voxel_keys=False):
        """Export existing totals; optional keys follow the same point order."""
        if any(type(v) is not int or v <= 0 for v in (minimum_observation_support, maximum_point_count)):
            raise ValueError("positive surface support and output count required")
        if type(include_voxel_keys) is not bool:
            raise ValueError("explicit voxel-key export boolean required")
        if self._totals_dict is not None:
            self._sync_columnar_from_totals_dict()
        if len(self._col_support) == 0:
            empty = (np.empty((0, 3)), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
            return (*empty, np.empty((0, 3), dtype=np.int64)) if include_voxel_keys else empty

        if int(self._col_support.min()) >= minimum_observation_support:
            grid = self._col_keys
            sums = self._col_sums
            support = self._col_support
            cam_arrays = list(self._col_cam_counts.values())
        else:
            mask = self._col_support >= minimum_observation_support
            if not np.any(mask):
                empty = (np.empty((0, 3)), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
                return (*empty, np.empty((0, 3), dtype=np.int64)) if include_voxel_keys else empty
            grid = self._col_keys[mask]
            sums = self._col_sums[mask]
            support = self._col_support[mask]
            cam_arrays = [arr[mask] for arr in self._col_cam_counts.values()]

        count = len(support)
        if len(cam_arrays) == 1:
            cameras = (cam_arrays[0] > 0).astype(np.int64)
        else:
            cameras = np.zeros(count, dtype=np.int64)
            for arr in cam_arrays:
                cameras += arr > 0

        if count > maximum_point_count:
            k = maximum_point_count
            c_max, c_min = int(cameras.max()), int(cameras.min())
            s_max, s_min = int(support.max()), int(support.min())
            span_c = c_max - c_min + 1
            span_s = s_max - s_min + 1
            if span_c == 1 and span_s == 1:
                order = slice(0, k)
            elif span_c * span_s * count <= np.iinfo(np.int64).max:
                packed_prio = (
                    ((c_max - cameras) * span_s + (s_max - support)) * count
                    + np.arange(count, dtype=np.int64)
                )
                part = np.argpartition(packed_prio, k - 1)[:k]
                order = part[np.argsort(packed_prio[part])]
            else:
                order = np.lexsort((grid[:, 2], grid[:, 1], grid[:, 0], -support, -cameras))[:k]
            support = support[order].copy()
            cameras = cameras[order].copy()
            points = sums[order] / support[:, None]
            if include_voxel_keys:
                return points, support, cameras, grid[order].copy()
            return points, support, cameras

        support = support.copy()
        points = sums / support[:, None]
        if include_voxel_keys:
            return points, support, cameras, grid.copy()
        return points, support, cameras

    def supported_point_count(self, *, minimum_observation_support):
        if type(minimum_observation_support) is not int or minimum_observation_support <= 0:
            raise ValueError("positive surface support required")
        if self._totals_dict is not None:
            self._sync_columnar_from_totals_dict()
        return int(np.count_nonzero(self._col_support >= minimum_observation_support))

    def close(self):
        self._totals_dict = None
        self._col_keys = np.empty((0, 3), dtype=np.int64)
        self._col_sums = np.empty((0, 3), dtype=np.float64)
        self._col_support = np.empty(0, dtype=np.int64)
        self._col_cam_counts.clear()
        self._live.clear()
