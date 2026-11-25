"""Compact 4-byte-aligned binary wire codec for EvidenceScene surface meshes.

Serializes :class:`MeasuredSurfaceMesh`, :class:`PersistentSurfaceSnapshot`,
:class:`MultiPlaneSurfaceSnapshot`, or compatible mapping payloads into the
``ESTM`` binary wire format designed for zero-copy WebGL / robotics transport:

Layout (little-endian):
  - ``[0:4]``   ASCII magic ``b"ESTM"``
  - ``[4:8]``   ``uint32`` JSON metadata header byte length ``H``
  - ``[8:8+H]`` UTF-8 compact JSON scalar metadata header
  - ``0..3``    zero padding bytes to align next section to 4-byte boundary
  - ``uint16[V, 3]`` quantized vertex coordinates ``round((p - bbox_min) / scale)``
  - ``uint8[V, 3]``  per-vertex RGB colors + ``0..3`` zero pad bytes to 4-byte boundary
  - ``uint16[F, 3]`` (when ``V < 65536``) or ``uint32[F, 3]`` triangle indices
  - Optional trailing multi-plane section (when ``header["has_plane_indices"]`` is True):
    ``0..3`` zero pad bytes to 4-byte boundary + ``uint8[V]`` ``vertex_plane_indices``
    + ``uint8[F]`` ``triangle_plane_indices`` (when ``len(plane_ids) <= 255``;
    ``int32`` otherwise).
"""

from __future__ import annotations

import json
import struct
from typing import Any, Mapping

import numpy as np

from evidencescene.surface.measured_mesh import MeasuredSurfaceMesh
from evidencescene.surface.persistent_multi_plane import MultiPlaneSurfaceSnapshot
from evidencescene.surface.persistent_static_surface import PersistentSurfaceSnapshot

BINARY_MESH_MAGIC = b"ESTM"

_OMITTED_ARRAY_KEYS = frozenset(
    {
        "vertices_m",
        "colors_rgb_u8",
        "triangles",
        "face_normals",
        "source_point_indices",
        "vertex_plane_indices",
        "triangle_plane_indices",
        "vertex_observation_counts",
        "vertex_last_observed_ns",
    }
)


def _extract_payload(
    mesh_or_payload: (
        MeasuredSurfaceMesh
        | PersistentSurfaceSnapshot
        | MultiPlaneSurfaceSnapshot
        | Mapping[str, Any]
    ),
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    if isinstance(mesh_or_payload, MultiPlaneSurfaceSnapshot):
        header: dict[str, Any] = {
            "ready": bool(mesh_or_payload.ready),
            "reason": str(mesh_or_payload.reason),
            "version": int(mesh_or_payload.version),
            "epoch": int(mesh_or_payload.epoch),
            "coordinate_frame": str(mesh_or_payload.coordinate_frame),
            "plane_ids": list(mesh_or_payload.plane_ids),
            "support_count": int(mesh_or_payload.support_count),
            "surface_area_m2": round(float(mesh_or_payload.surface_area_m2), 6),
            "maximum_edge_m": round(float(mesh_or_payload.maximum_edge_m), 6),
            "estimated_memory_bytes": int(mesh_or_payload.estimated_memory_bytes),
        }
        if mesh_or_payload.latest_observation_timestamp_ns is not None:
            header["latest_observation_timestamp_ns"] = str(
                int(mesh_or_payload.latest_observation_timestamp_ns)
            )
        return (
            header,
            np.asarray(mesh_or_payload.vertices_m, dtype=np.float64),
            np.asarray(mesh_or_payload.colors_rgb_u8, dtype=np.uint8),
            np.asarray(mesh_or_payload.triangles, dtype=np.uint32),
            np.asarray(mesh_or_payload.vertex_plane_indices, dtype=np.int32),
            np.asarray(mesh_or_payload.triangle_plane_indices, dtype=np.int32),
        )

    if isinstance(mesh_or_payload, PersistentSurfaceSnapshot):
        header = {
            "ready": bool(mesh_or_payload.ready),
            "reason": str(mesh_or_payload.reason),
            "version": int(mesh_or_payload.version),
            "epoch": int(mesh_or_payload.epoch),
            "coordinate_frame": str(mesh_or_payload.coordinate_frame),
            "support_count": int(mesh_or_payload.support_count),
            "surface_area_m2": round(float(mesh_or_payload.surface_area_m2), 6),
            "maximum_edge_m": round(float(mesh_or_payload.maximum_edge_m), 6),
            "estimated_memory_bytes": int(mesh_or_payload.estimated_memory_bytes),
        }
        if mesh_or_payload.latest_observation_timestamp_ns is not None:
            header["latest_observation_timestamp_ns"] = str(
                int(mesh_or_payload.latest_observation_timestamp_ns)
            )
        return (
            header,
            np.asarray(mesh_or_payload.vertices_m, dtype=np.float64),
            np.asarray(mesh_or_payload.colors_rgb_u8, dtype=np.uint8),
            np.asarray(mesh_or_payload.triangles, dtype=np.uint32),
            None,
            None,
        )

    if isinstance(mesh_or_payload, MeasuredSurfaceMesh):
        header = {
            "ready": bool(mesh_or_payload.ready),
            "reason": str(mesh_or_payload.reason),
            "coordinate_frame": str(mesh_or_payload.coordinate_frame),
            "timestamp_ns": str(int(mesh_or_payload.timestamp_ns)),
            "source_id": str(mesh_or_payload.source_id),
            "support_count": int(mesh_or_payload.support_count),
            "surface_area_m2": round(float(mesh_or_payload.surface_area_m2), 6),
            "maximum_edge_m": round(float(mesh_or_payload.maximum_edge_m), 6),
        }
        return (
            header,
            np.asarray(mesh_or_payload.vertices_m, dtype=np.float64),
            np.asarray(mesh_or_payload.colors_rgb_u8, dtype=np.uint8),
            np.asarray(mesh_or_payload.triangles, dtype=np.uint32),
            None,
            None,
        )

    if isinstance(mesh_or_payload, Mapping):
        header = {
            k: v for k, v in mesh_or_payload.items() if k not in _OMITTED_ARRAY_KEYS
        }
        ready = bool(mesh_or_payload.get("ready"))
        if not ready:
            return (
                header,
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3), dtype=np.uint8),
                np.empty((0, 3), dtype=np.uint32),
                None,
                None,
            )
        verts = np.asarray(mesh_or_payload.get("vertices_m", ()), dtype=np.float64)
        cols = np.asarray(mesh_or_payload.get("colors_rgb_u8", ()), dtype=np.uint8)
        tris = np.asarray(mesh_or_payload.get("triangles", ()), dtype=np.uint32)
        vp = mesh_or_payload.get("vertex_plane_indices")
        tp = mesh_or_payload.get("triangle_plane_indices")
        vp_arr = None if vp is None else np.asarray(vp, dtype=np.int32)
        tp_arr = None if tp is None else np.asarray(tp, dtype=np.int32)
        return header, verts, cols, tris, vp_arr, tp_arr

    raise TypeError(
        "mesh_or_payload must be a MeasuredSurfaceMesh, PersistentSurfaceSnapshot, "
        "MultiPlaneSurfaceSnapshot, or mapping"
    )


def encode_surface_mesh_binary(
    mesh_or_payload: (
        MeasuredSurfaceMesh
        | PersistentSurfaceSnapshot
        | MultiPlaneSurfaceSnapshot
        | Mapping[str, Any]
    ),
    *,
    min_quantization_m: float = 0.0001,
) -> bytes:
    """Encode an EvidenceScene surface mesh or dict payload into compact ``ESTM`` binary."""
    if not np.isfinite(min_quantization_m) or min_quantization_m <= 0.0:
        raise ValueError("min_quantization_m must be finite and positive")

    header, verts, cols, tris, vp_arr, tp_arr = _extract_payload(mesh_or_payload)
    ready = bool(header.get("ready"))
    if not ready or len(verts) == 0:
        header["vertex_count"] = 0
        header["triangle_count"] = 0
        header["index_dtype"] = "uint16"
        header["bbox_min_m"] = [0.0, 0.0, 0.0]
        header["quantization_scale_m"] = float(min_quantization_m)
        hdr_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        pad = (-(8 + len(hdr_bytes))) & 3
        return BINARY_MESH_MAGIC + struct.pack("<I", len(hdr_bytes)) + hdr_bytes + (b"\x00" * pad)

    if verts.ndim != 2 or verts.shape[1] != 3 or cols.shape != verts.shape:
        raise ValueError("vertices_m and colors_rgb_u8 must be matching Vx3 arrays")
    if tris.ndim != 2 or tris.shape[1] != 3:
        raise ValueError("triangles must be an Fx3 array")

    n_v = int(len(verts))
    n_f = int(len(tris))
    v_min = np.min(verts, axis=0)
    v_max = np.max(verts, axis=0)
    span = float(np.max(v_max - v_min))
    scale = max(float(min_quantization_m), span / 65535.0)
    q_verts = np.clip(np.rint((verts - v_min[None, :]) / scale), 0.0, 65535.0).astype("<u2")
    cols_u8 = np.ascontiguousarray(cols, dtype=np.uint8)
    idx_dtype = "uint16" if n_v < 65536 else "uint32"
    tris_bin = np.ascontiguousarray(tris, dtype=("<u2" if idx_dtype == "uint16" else "<u4"))

    has_planes = vp_arr is not None and tp_arr is not None and len(vp_arr) == n_v and len(tp_arr) == n_f
    plane_dtype = "uint8"
    if has_planes:
        max_p = int(max(int(np.max(vp_arr)) if n_v else 0, int(np.max(tp_arr)) if n_f else 0))
        plane_dtype = "uint8" if max_p <= 255 else "int32"
        header["has_plane_indices"] = True
        header["plane_index_dtype"] = plane_dtype

    header["vertex_count"] = n_v
    header["triangle_count"] = n_f
    header["index_dtype"] = idx_dtype
    header["bbox_min_m"] = [float(v_min[0]), float(v_min[1]), float(v_min[2])]
    header["quantization_scale_m"] = float(scale)
    header["coordinate_quantization_m"] = float(scale)

    hdr_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad_hdr = (-(8 + len(hdr_bytes))) & 3
    v_bytes = q_verts.tobytes()
    c_bytes = cols_u8.tobytes()
    pad_c = (-(len(v_bytes) + len(c_bytes))) & 3
    t_bytes = tris_bin.tobytes()

    parts = [
        BINARY_MESH_MAGIC,
        struct.pack("<I", len(hdr_bytes)),
        hdr_bytes,
        b"\x00" * pad_hdr,
        v_bytes,
        c_bytes,
        b"\x00" * pad_c,
        t_bytes,
    ]
    if has_planes:
        assert vp_arr is not None and tp_arr is not None
        pad_t = (-len(t_bytes)) & 3
        p_dt = np.uint8 if plane_dtype == "uint8" else "<i4"
        parts.append(b"\x00" * pad_t)
        parts.append(np.ascontiguousarray(vp_arr, dtype=p_dt).tobytes())
        parts.append(np.ascontiguousarray(tp_arr, dtype=p_dt).tobytes())
    return b"".join(parts)


def decode_surface_mesh_binary(blob: bytes) -> dict[str, Any]:
    """Decode an ``ESTM`` binary buffer back into a dictionary with NumPy mesh arrays."""
    if len(blob) < 8 or blob[:4] != BINARY_MESH_MAGIC:
        raise ValueError("Invalid ESTM binary surface mesh magic")
    (hdr_len,) = struct.unpack_from("<I", blob, 4)
    if 8 + hdr_len > len(blob):
        raise ValueError("Truncated ESTM header")
    header = json.loads(blob[8 : 8 + hdr_len].decode("utf-8"))
    n_v = int(header.get("vertex_count", 0))
    n_f = int(header.get("triangle_count", 0))
    if not header.get("ready") or n_v == 0:
        header["vertices_m"] = np.empty((0, 3), dtype=np.float64)
        header["colors_rgb_u8"] = np.empty((0, 3), dtype=np.uint8)
        header["triangles"] = np.empty((0, 3), dtype=np.int32)
        return header

    offset = 8 + hdr_len + ((-(8 + hdr_len)) & 3)
    v_nbytes = n_v * 3 * 2
    c_nbytes = n_v * 3
    if offset + v_nbytes + c_nbytes > len(blob):
        raise ValueError("Truncated ESTM vertex/color buffer")
    q_verts = np.frombuffer(blob, dtype="<u2", count=n_v * 3, offset=offset).reshape(n_v, 3)
    offset += v_nbytes
    cols = np.frombuffer(blob, dtype=np.uint8, count=n_v * 3, offset=offset).reshape(n_v, 3).copy()
    offset += c_nbytes
    offset += (-(v_nbytes + c_nbytes)) & 3

    idx_dtype = header.get("index_dtype", "uint16")
    t_dt = np.dtype("<u2") if idx_dtype == "uint16" else np.dtype("<u4")
    t_nbytes = n_f * 3 * t_dt.itemsize
    if offset + t_nbytes > len(blob):
        raise ValueError("Truncated ESTM triangle buffer")
    tris = np.frombuffer(blob, dtype=t_dt, count=n_f * 3, offset=offset).reshape(n_f, 3).astype(np.int32)
    offset += t_nbytes

    v_min = np.asarray(header["bbox_min_m"], dtype=np.float64)
    scale = float(header["quantization_scale_m"])
    header["vertices_m"] = v_min[None, :] + q_verts.astype(np.float64) * scale
    header["colors_rgb_u8"] = cols
    header["triangles"] = tris

    if header.get("has_plane_indices"):
        offset += (-t_nbytes) & 3
        p_dt = np.dtype(np.uint8) if header.get("plane_index_dtype", "uint8") == "uint8" else np.dtype("<i4")
        p_nbytes = (n_v + n_f) * p_dt.itemsize
        if offset + p_nbytes > len(blob):
            raise ValueError("Truncated ESTM plane index buffer")
        header["vertex_plane_indices"] = np.frombuffer(
            blob, dtype=p_dt, count=n_v, offset=offset
        ).astype(np.int32)
        offset += n_v * p_dt.itemsize
        header["triangle_plane_indices"] = np.frombuffer(
            blob, dtype=p_dt, count=n_f, offset=offset
        ).astype(np.int32)

    return header


__all__ = [
    "BINARY_MESH_MAGIC",
    "decode_surface_mesh_binary",
    "encode_surface_mesh_binary",
]
