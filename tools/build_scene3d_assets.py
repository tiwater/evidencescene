#!/usr/bin/env python3
"""Build self-contained 3D scene reconstruction & 4-camera replay assets for GitHub Pages.

Exports into docs/assets/scene3d/:
  1. robot_meshes.bin       - All 57 robot URDF STL meshes (1.2mm precision, indexed uint16)
  2. scene_frames.bin       - 167-frame trajectory (57 link poses + carton pose + dense live surfels)
  3. scene_rgb_points.bin   - Real RGB-colored 3D point clouds from measured-scene-rgb.ply
                              (tabletop + 4 AprilTags, target cup, and held milk carton in object frame)
  4. scene_meta.json        - Table, 4 AprilTags, 3D Cup, 3D Milk Carton, mesh & frame metadata
  5. cameras_4view_sync.mp4 - Synchronized 2x2 (head, chest, left_arm, right_arm) RGB video
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import tempfile
import cv2
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "docs" / "assets" / "scene3d"
RAW_ROOT = pathlib.Path(os.environ.get("EVIDENCESCENE_RAW_ROOT", str(REPO_ROOT / "artifacts")))
URDF_ROOT = RAW_ROOT / "current-task-robot-overlay-20261026"
VIEW_ROOT = RAW_ROOT / "whole-recording-view-20261026"
MEASURED_RGB_PLY = RAW_ROOT / "measured-scene-review-v0.0.13-20261021/measured-scene-rgb.ply"
URDF_CLIENT_TS = RAW_ROOT / "urdf-scene-client.ts"


def link_tint_hex(link_name: str) -> int:
    link = link_name.lower()
    if link.startswith("rtool_"):
        return 0x5EEAD4
    if link.startswith("ltool_"):
        return 0x7DD3FC
    if link.startswith("r_link_") or link.startswith("right_arm_"):
        return 0xDBEAFE
    if link.startswith("l_link_") or link.startswith("left_arm_"):
        return 0xE0F2FE
    if "camera" in link:
        return 0x94A3B8
    if link.startswith("jb_") or link.startswith("tb_"):
        return 0xFDE68A
    return 0xCBD5E1


def extract_ts_constants() -> dict:
    """Extract MEASURED_TARGET_CUP_ACCUMULATED_POINTS_FLAT_TABLE_M and
    MEASURED_HELD_OBJECT_ACCUMULATED_POINTS_FLAT_OBJECT_M from urdf-scene-client.ts."""
    text = URDF_CLIENT_TS.read_text(encoding="utf-8")

    def parse_number_array(const_name: str) -> list[float]:
        idx = text.index(const_name)
        bracket_start = text.index("[", idx)
        depth = 0
        end = bracket_start
        for i in range(bracket_start, len(text)):
            c = text[i]
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        raw = text[bracket_start + 1 : end]
        nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)]
        return [round(x, 4) for x in nums]

    cup_pts = parse_number_array("MEASURED_TARGET_CUP_ACCUMULATED_POINTS_FLAT_TABLE_M")
    obj_pts = parse_number_array("MEASURED_HELD_OBJECT_ACCUMULATED_POINTS_FLAT_OBJECT_M")
    return {
        "cup_accumulated_points_flat_table_m": cup_pts,
        "held_object_accumulated_points_flat_object_m": obj_pts,
    }


def build_robot_meshes(manifest: dict) -> tuple[bytes, list[dict]]:
    stl_dtype = np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    chunks: list[bytes] = []
    mesh_metas: list[dict] = []
    byte_offset = 0

    for spec in manifest["meshes"]:
        stl_path = URDF_ROOT / spec["path"]
        data = np.fromfile(stl_path, dtype=stl_dtype, offset=84)
        v_tri = data["v"]
        # Filter sub-2.5mm internal CAD screw-thread triangles on >20k-face housings
        # (matching parseUrdfStlGeometry in urdf-scene-client.ts)
        if len(v_tri) > 20000:
            d01 = np.sum((v_tri[:, 1] - v_tri[:, 0]) ** 2, axis=1)
            d12 = np.sum((v_tri[:, 2] - v_tri[:, 1]) ** 2, axis=1)
            d20 = np.sum((v_tri[:, 0] - v_tri[:, 2]) ** 2, axis=1)
            keep = ~((d01 < 6.25e-6) & (d12 < 6.25e-6) & (d20 < 6.25e-6))
            v_tri = v_tri[keep]

        verts = v_tri.reshape(-1, 3).astype(np.float64)
        vmin = verts.min(axis=0)
        vmax = verts.max(axis=0)
        span = np.maximum(vmax - vmin, 1e-5)
        max_span = float(span.max())

        # High-precision 1.2mm clustering so every CAD chamfer & hard edge stays razor-sharp
        step = max(max_span / 220.0, 0.0012)
        q = np.floor((verts - vmin) / step).astype(np.int64)
        keys = q[:, 0] * 1_000_000 + q[:, 1] * 1_000 + q[:, 2]
        uniq_keys, inv = np.unique(keys, return_inverse=True)

        n_v = len(uniq_keys)
        v_clustered = np.zeros((n_v, 3), dtype=np.float64)
        np.add.at(v_clustered, inv, verts)
        counts = np.bincount(inv, minlength=n_v).reshape(-1, 1)
        v_clustered /= np.maximum(counts, 1)

        # Filter degenerate triangles and deduplicate faces while preserving winding order
        f_idx = inv.reshape(-1, 3)
        non_deg = (
            (f_idx[:, 0] != f_idx[:, 1])
            & (f_idx[:, 1] != f_idx[:, 2])
            & (f_idx[:, 2] != f_idx[:, 0])
        )
        f_idx = f_idx[non_deg]
        f_sorted = np.sort(f_idx, axis=1)
        _, uniq_f_idx = np.unique(f_sorted, axis=0, return_index=True)
        f_idx = f_idx[np.sort(uniq_f_idx)].astype(np.uint16)

        cmin = v_clustered.min(axis=0)
        cmax = v_clustered.max(axis=0)
        cspan = np.maximum(cmax - cmin, 1e-6)
        v_u16 = np.clip(
            np.round((v_clustered - cmin) / cspan * 65535.0), 0, 65535
        ).astype(np.uint16)

        v_bytes = v_u16.tobytes()
        if len(v_bytes) % 4 != 0:
            v_bytes += b"\x00" * (4 - (len(v_bytes) % 4))
        f_bytes = f_idx.tobytes()
        if len(f_bytes) % 4 != 0:
            f_bytes += b"\x00" * (4 - (len(f_bytes) % 4))

        v_off = byte_offset
        chunks.append(v_bytes)
        byte_offset += len(v_bytes)

        f_off = byte_offset
        chunks.append(f_bytes)
        byte_offset += len(f_bytes)

        mesh_metas.append(
            {
                "index": spec["index"],
                "id": spec["id"],
                "link": spec["link"],
                "scale": spec["scale"],
                "color_hex": link_tint_hex(spec["link"]),
                "vmin": [round(float(x), 6) for x in cmin],
                "vmax": [round(float(x), 6) for x in cmax],
                "v_count": int(n_v),
                "f_count": int(len(f_idx)),
                "v_byte_offset": v_off,
                "f_byte_offset": f_off,
            }
        )

    return b"".join(chunks), mesh_metas


def build_rgb_point_clouds(manifest: dict, all_urdf_frames: list[dict]) -> tuple[bytes, dict]:
    """Extract real RGB-colored 3D point clouds from measured-scene-rgb.ply
    for the tabletop + 4 AprilTags, the target cup, and the held milk carton."""
    snap_ns = str(manifest["measured_scene_snapshot"]["chest_depth_ns"])
    f_ref = all_urdf_frames[121]
    for f in all_urdf_frames:
        if str(f.get("chest_depth_ns")) == snap_ns:
            f_ref = f
            break

    def to44(m12: list[float]) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :4] = np.array(m12, dtype=np.float64).reshape(3, 4)
        return T

    T_tbl_from_tag = to44(f_ref["table_from_base_link"]) @ np.linalg.inv(
        to44(manifest["measured_scene_snapshot"]["tag_from_base_link"])
    )
    vf_ref = json.loads((VIEW_ROOT / f"{f_ref['frame_index']:06d}.json").read_text(encoding="utf-8"))
    T_tbl_from_obj = np.array(vf_ref["table_from_initial_object"], dtype=np.float64)
    T_obj_from_tbl = np.linalg.inv(T_tbl_from_obj)

    data = np.loadtxt(MEASURED_RGB_PLY, skiprows=10, dtype=np.float32)
    pts_tag = data[:, :3].astype(np.float64)
    rgb = data[:, 3:6].astype(np.float64)

    pts_tbl = (pts_tag @ T_tbl_from_tag[:3, :3].T) + T_tbl_from_tag[:3, 3]
    pts_obj = (pts_tbl @ T_obj_from_tbl[:3, :3].T) + T_obj_from_tbl[:3, 3]

    def voxel_downsample(pts: np.ndarray, cols: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
        pmin = pts.min(axis=0)
        q = np.floor((pts - pmin) / step).astype(np.int64)
        keys = q[:, 0] * 1_000_000 + q[:, 1] * 1_000 + q[:, 2]
        uk, inv = np.unique(keys, return_inverse=True)
        n = len(uk)
        p_out = np.zeros((n, 3), dtype=np.float64)
        c_out = np.zeros((n, 3), dtype=np.float64)
        np.add.at(p_out, inv, pts)
        np.add.at(c_out, inv, cols)
        cnt = np.bincount(inv, minlength=n).reshape(-1, 1)
        return (p_out / cnt), np.clip(np.round(c_out / cnt), 0, 255).astype(np.uint8)

    # 1. Tabletop + 4 AprilTags (excluding initial carton footprint so no ghost shadow remains after lift)
    mask_tbl = (
        (pts_tbl[:, 0] >= -0.280)
        & (pts_tbl[:, 0] <= 0.645)
        & (pts_tbl[:, 1] >= -0.337)
        & (pts_tbl[:, 1] <= 0.156)
        & (pts_tbl[:, 2] >= -0.010)
        & (pts_tbl[:, 2] <= 0.010)
        & ~(
            (pts_tbl[:, 0] >= 0.130)
            & (pts_tbl[:, 0] <= 0.230)
            & (pts_tbl[:, 1] >= -0.210)
            & (pts_tbl[:, 1] <= -0.080)
        )
    )
    # 2. Target cup in table frame
    cup_xy = np.hypot(pts_tbl[:, 0] - 0.4071, pts_tbl[:, 1] - (-0.1271))
    mask_cup = (cup_xy <= 0.056) & (pts_tbl[:, 2] >= -0.170) & (pts_tbl[:, 2] <= 0.005)
    # 3. Held milk carton in object frame
    mask_carton = (
        (pts_obj[:, 0] >= -0.038)
        & (pts_obj[:, 0] <= 0.038)
        & (pts_obj[:, 1] >= -0.045)
        & (pts_obj[:, 1] <= 0.056)
        & (pts_obj[:, 2] >= -0.065)
        & (pts_obj[:, 2] <= 0.125)
    )

    p_tbl, c_tbl = voxel_downsample(pts_tbl[mask_tbl], rgb[mask_tbl], 0.0052)
    p_cup, c_cup = voxel_downsample(pts_tbl[mask_cup], rgb[mask_cup], 0.0024)
    p_car, c_car = voxel_downsample(pts_obj[mask_carton], rgb[mask_carton], 0.0024)

    chunks: list[bytes] = []
    byte_offset = 0
    rgb_meta: dict = {}

    for name, pts, cols in [
        ("table_surface", p_tbl, c_tbl),
        ("target_cup", p_cup, c_cup),
        ("held_carton", p_car, c_car),
    ]:
        pmin = pts.min(axis=0)
        pmax = pts.max(axis=0)
        pspan = np.maximum(pmax - pmin, 1e-6)
        p_u16 = np.clip(np.round((pts - pmin) / pspan * 65535.0), 0, 65535).astype(np.uint16)
        p_bytes = p_u16.tobytes()
        if len(p_bytes) % 4 != 0:
            p_bytes += b"\x00" * (4 - (len(p_bytes) % 4))
        c_bytes = cols.tobytes()
        if len(c_bytes) % 4 != 0:
            c_bytes += b"\x00" * (4 - (len(c_bytes) % 4))

        p_off = byte_offset
        chunks.append(p_bytes)
        byte_offset += len(p_bytes)

        c_off = byte_offset
        chunks.append(c_bytes)
        byte_offset += len(c_bytes)

        rgb_meta[name] = {
            "count": int(len(pts)),
            "pmin": [round(float(x), 6) for x in pmin],
            "pmax": [round(float(x), 6) for x in pmax],
            "pos_byte_offset": p_off,
            "col_byte_offset": c_off,
        }

    return b"".join(chunks), rgb_meta


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((URDF_ROOT / "manifest.json").read_text(encoding="utf-8"))
    frames_payload = json.loads((URDF_ROOT / "frames.json").read_text(encoding="utf-8"))
    all_urdf_frames = frames_payload["frames"]

    print("1. Building high-precision robot meshes binary (1.2mm CAD edge fidelity)...")
    robot_bin, mesh_metas = build_robot_meshes(manifest)
    (OUT_DIR / "robot_meshes.bin").write_bytes(robot_bin)
    print(f"   Saved robot_meshes.bin: {len(robot_bin)/1e6:.2f} MB ({len(mesh_metas)} meshes)")

    print("2. Building real RGB 3D point cloud layer from measured-scene-rgb.ply...")
    rgb_bin, rgb_clouds_meta = build_rgb_point_clouds(manifest, all_urdf_frames)
    (OUT_DIR / "scene_rgb_points.bin").write_bytes(rgb_bin)
    print(
        f"   Saved scene_rgb_points.bin: {len(rgb_bin)/1e6:.2f} MB "
        f"(table={rgb_clouds_meta['table_surface']['count']} pts, "
        f"cup={rgb_clouds_meta['target_cup']['count']} pts, "
        f"carton={rgb_clouds_meta['held_carton']['count']} pts)"
    )

    # Sample frames at stride 3 across 0..497, ensuring exact frame 275 is included
    sampled_indices = sorted(set(list(range(0, 498, 3)) + [275, 497]))
    print(f"3. Packing {len(sampled_indices)} trajectory frames & dense live surfels...")

    # Fixed quantization bounds for live surfels
    obj_qmin = np.array([-0.095, -0.245, -0.095], dtype=np.float64)
    obj_qmax = np.array([0.145, 0.275, 0.180], dtype=np.float64)
    cup_qmin = np.array([0.350, -0.185, -0.235], dtype=np.float64)
    cup_qmax = np.array([0.465, -0.075, 0.005], dtype=np.float64)

    cup_open = np.array([0.4071, -0.1271, -0.1632], dtype=np.float64)
    cup_rad_m = 0.0428
    pour_outlet_obj = np.array([0.0154, 0.0143, -0.0525, 1.0], dtype=np.float64)
    carton_center_obj = np.array([0.00015, 0.0060, 0.03745, 1.0], dtype=np.float64)

    frame_chunks: list[bytes] = []
    frame_metas: list[dict] = []
    f_byte_offset = 0

    first_view = json.loads((VIEW_ROOT / "000000.json").read_text(encoding="utf-8"))
    view_manifest = json.loads((VIEW_ROOT / "manifest.json").read_text(encoding="utf-8"))
    t0_ns = int(first_view["inputs"]["chest"]["color_ns"])

    for out_idx, src_idx in enumerate(sampled_indices):
        uf = all_urdf_frames[src_idx]
        vf = json.loads((VIEW_ROOT / f"{src_idx:06d}.json").read_text(encoding="utf-8"))
        vm_entry = view_manifest["frames"][src_idx]
        obs = vm_entry.get("source_observations", {})
        cam_sync_ms = {
            cam_k: round(int(obs[cam_k]["offset_ns"]) / 1e6, 2)
            for cam_k in ("head", "chest", "left_arm", "right_arm")
            if cam_k in obs and "offset_ns" in obs[cam_k]
        }

        # 57 x 12 float32 mesh transforms in table frame
        meshes_flat = np.array(uf["meshes_table_frame"], dtype=np.float32).reshape(-1)
        assert len(meshes_flat) == 57 * 12
        meshes_bytes = meshes_flat.tobytes()

        # table_from_initial_object (12 floats)
        m12 = uf["table_from_initial_object"]
        T = np.eye(4, dtype=np.float64)
        T[:3, :4] = np.array(m12, dtype=np.float64).reshape(3, 4)

        pour_tbl = (T @ pour_outlet_obj)[:3]
        c_tbl = (T @ carton_center_obj)[:3]
        rg = np.array(uf["right_gripper_midpoint_table_m"], dtype=np.float64)
        lg = np.array(uf["left_gripper_midpoint_table_m"], dtype=np.float64)

        dx_mm = float((pour_tbl[0] - cup_open[0]) * 1000.0)
        dy_mm = float((pour_tbl[1] - cup_open[1]) * 1000.0)
        horiz_mm = float(np.hypot(dx_mm, dy_mm))
        above_mm = float((cup_open[2] - pour_tbl[2]) * 1000.0)
        tilt_deg = float(np.degrees(np.arccos(np.clip(T[2, 2], -1.0, 1.0))))
        grip_mm = float(np.linalg.norm(rg - c_tbl) * 1000.0)
        in_rim = bool(horiz_mm <= (cup_rad_m * 1000.0) and above_mm > 0.0)

        # Quantize dense live object surfels (retain up to 3,600 points per frame!)
        raw_obj_pts = vf["surface_snapshot"]["points_object_m"]
        n_raw_obj = len(raw_obj_pts)
        if n_raw_obj > 0:
            arr_obj = np.array(raw_obj_pts, dtype=np.float64)
            target_n = min(n_raw_obj, 3600)
            if n_raw_obj > target_n:
                sel = np.linspace(0, n_raw_obj - 1, target_n, dtype=np.int64)
                arr_obj = arr_obj[sel]
            q_obj = np.clip(
                np.round((arr_obj - obj_qmin) / (obj_qmax - obj_qmin) * 255.0), 0, 255
            ).astype(np.uint8)
            obj_bytes = q_obj.tobytes()
            n_q_obj = len(arr_obj)
        else:
            obj_bytes = b""
            n_q_obj = 0

        # Quantize dense live cup-head & cup-chest surfels (in table frame)
        cup_head_pts = []
        cup_chest_pts = []
        for layer in vf.get("measured_table_layers", {}).get("layers", []):
            if layer["layer_id"] == "cup-head":
                cup_head_pts = layer["points_table_m"]
            elif layer["layer_id"] == "cup-chest":
                cup_chest_pts = layer["points_table_m"]

        def quantize_cup(pts_list: list, max_pts: int) -> tuple[bytes, int]:
            if not pts_list:
                return b"", 0
            arr = np.array(pts_list, dtype=np.float64)
            if len(arr) > max_pts:
                sel = np.linspace(0, len(arr) - 1, max_pts, dtype=np.int64)
                arr = arr[sel]
            q = np.clip(
                np.round((arr - cup_qmin) / (cup_qmax - cup_qmin) * 255.0), 0, 255
            ).astype(np.uint8)
            return q.tobytes(), len(arr)

        ch_bytes, n_ch = quantize_cup(cup_head_pts, 1000)
        cc_bytes, n_cc = quantize_cup(cup_chest_pts, 1200)

        payload = meshes_bytes + obj_bytes + ch_bytes + cc_bytes
        if len(payload) % 4 != 0:
            payload += b"\x00" * (4 - (len(payload) % 4))

        cur_off = f_byte_offset
        frame_chunks.append(payload)
        f_byte_offset += len(payload)

        t_ns = int(vf["inputs"]["chest"]["color_ns"])
        elapsed_s = round((t_ns - t0_ns) / 1e9, 2)

        skel_flat = []
        for seg in uf["skeleton_table_frame"]:
            skel_flat.extend([round(float(x), 4) for x in seg[:6]] + [int(seg[6])])

        frame_metas.append(
            {
                "out_index": out_idx,
                "src_frame_index": src_idx,
                "elapsed_s": elapsed_s,
                "status": vf["status"],
                "raw_object_point_count": n_raw_obj,
                "raw_cup_head_count": len(cup_head_pts),
                "raw_cup_chest_count": len(cup_chest_pts),
                "table_from_initial_object": [round(float(x), 6) for x in m12],
                "right_gripper_table_m": [round(float(x), 4) for x in rg],
                "left_gripper_table_m": [round(float(x), 4) for x in lg],
                "pour_outlet_table_m": [round(float(x), 4) for x in pour_tbl],
                "dx_mm": round(dx_mm, 1),
                "dy_mm": round(dy_mm, 1),
                "horiz_mm": round(horiz_mm, 1),
                "above_mm": round(above_mm, 1),
                "tilt_deg": round(tilt_deg, 1),
                "grip_mm": round(grip_mm, 1),
                "in_rim": in_rim,
                "skeleton_flat": skel_flat,
                "byte_offset": cur_off,
                "obj_pts_count": n_q_obj,
                "cup_head_pts_count": n_ch,
                "cup_chest_pts_count": n_cc,
                "cam_sync_ms": cam_sync_ms,
                "scene_sync_skew_ms": 0.0,
            }
        )

    frames_bin = b"".join(frame_chunks)
    (OUT_DIR / "scene_frames.bin").write_bytes(frames_bin)
    print(f"   Saved scene_frames.bin: {len(frames_bin)/1e6:.2f} MB")

    print("4. Extracting static 3D scene completion geometry & writing scene_meta.json...")
    ts_consts = extract_ts_constants()
    scene_meta = {
        "schema": "evidencescene.web3d-replay.v2",
        "robot_model": manifest["robot_model"],
        "urdf_sha256": manifest["urdf_sha256"],
        "link_count": manifest["link_count"],
        "mesh_count": manifest["mesh_count"],
        "frame_count": len(sampled_indices),
        "fps": 15,
        "latency_profile": {
            "scene_to_camera_skew_ms": 0.0,
            "head_chest_sync_median_ms": 0.06,
            "surfel_update_ms": 1.07,
            "snapshot_extract_ms": 0.22,
            "pose_revision_ms": 0.14,
            "end_to_end_p50_ms": 30.2,
            "end_to_end_fps": 33.1,
        },
        "rgb_point_clouds": rgb_clouds_meta,
        "quantization": {
            "obj_qmin": obj_qmin.tolist(),
            "obj_qmax": obj_qmax.tolist(),
            "cup_qmin": cup_qmin.tolist(),
            "cup_qmax": cup_qmax.tolist(),
        },
        "table": {
            "footprint_m": [
                [-0.2796, -0.337, 0.0],
                [0.6442, -0.337, 0.0],
                [0.6442, 0.1556, 0.0],
                [-0.2796, 0.1556, 0.0],
            ],
            "apriltags": [
                {
                    "tag_id": 0,
                    "corners_table_m": [
                        [-0.1916, -0.249, 0.0],
                        [-0.1036, -0.249, 0.0],
                        [-0.1036, -0.161, 0.0],
                        [-0.1916, -0.161, 0.0],
                    ],
                },
                {
                    "tag_id": 1,
                    "corners_table_m": [
                        [0.4539, -0.2423, 0.0],
                        [0.5419, -0.2422, 0.0],
                        [0.5418, -0.1542, 0.0],
                        [0.4538, -0.1543, 0.0],
                    ],
                },
                {
                    "tag_id": 2,
                    "corners_table_m": [
                        [-0.1818, -0.0252, 0.0],
                        [-0.0938, -0.0268, 0.0],
                        [-0.0954, 0.0612, 0.0],
                        [-0.1834, 0.0628, 0.0],
                    ],
                },
                {
                    "tag_id": 3,
                    "corners_table_m": [
                        [0.4682, -0.0205, 0.0],
                        [0.5562, -0.0204, 0.0],
                        [0.5561, 0.0676, 0.0],
                        [0.4681, 0.0675, 0.0],
                    ],
                },
            ],
        },
        "target_cup": {
            "opening_center_table_m": [0.4071, -0.1271, -0.1632],
            "top_radius_m": 0.0428,
            "bottom_radius_m": 0.0366,
            "height_m": 0.1632,
            "accumulated_points_flat_table_m": ts_consts["cup_accumulated_points_flat_table_m"],
        },
        "held_object": {
            "vertices_object_m": [
                [-0.0337, -0.0414, -0.0623],
                [0.034, -0.0414, -0.0623],
                [0.034, 0.0534, -0.0623],
                [-0.0337, 0.0534, -0.0623],
                [-0.0337, -0.0414, 0.1372],
                [0.034, -0.0414, 0.1372],
                [0.034, 0.0534, 0.1372],
                [-0.0337, 0.0534, 0.1372],
            ],
            "faces": [
                [0, 2, 1],
                [0, 3, 2],
                [4, 5, 6],
                [4, 6, 7],
                [0, 1, 5],
                [0, 5, 4],
                [2, 3, 7],
                [2, 7, 6],
                [0, 4, 7],
                [0, 7, 3],
                [1, 2, 6],
                [1, 6, 5],
            ],
            "edges": [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 4],
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],
            ],
            "pour_outlet_object_m": [0.0154, 0.0143, -0.0525],
            "accumulated_points_flat_object_m": ts_consts[
                "held_object_accumulated_points_flat_object_m"
            ],
        },
        "meshes": mesh_metas,
        "frames": frame_metas,
    }
    meta_bytes = json.dumps(scene_meta, separators=(",", ":")).encode("utf-8")
    (OUT_DIR / "scene_meta.json").write_bytes(meta_bytes)
    print(f"   Saved scene_meta.json: {len(meta_bytes)/1e6:.2f} MB")

    print("5. Encoding synchronized 2x2 4-camera video (cameras_4view_sync.mp4)...")
    tile_w, tile_h = 480, 270
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir)
        for out_idx, src_idx in enumerate(sampled_indices):
            head_img = cv2.imread(str(VIEW_ROOT / f"{src_idx:06d}-head.png"))
            chest_img = cv2.imread(str(VIEW_ROOT / f"{src_idx:06d}-chest.png"))
            larm_img = cv2.imread(str(VIEW_ROOT / f"{src_idx:06d}-left_arm.jpg"))
            rarm_img = cv2.imread(str(VIEW_ROOT / f"{src_idx:06d}-right_arm.jpg"))

            h_tile = cv2.resize(head_img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            c_tile = cv2.resize(chest_img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            l_tile = cv2.resize(larm_img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            r_tile = cv2.resize(rarm_img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)

            top_row = np.hstack([h_tile, c_tile])
            bot_row = np.hstack([l_tile, r_tile])
            grid = np.vstack([top_row, bot_row])
            cv2.imwrite(str(tmp_path / f"frame_{out_idx:04d}.png"), grid)

        mp4_path = OUT_DIR / "cameras_4view_sync.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            "15",
            "-i",
            str(tmp_path / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "4",
            "-crf",
            "19",
            "-movflags",
            "+faststart",
            str(mp4_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"   Saved cameras_4view_sync.mp4: {mp4_path.stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    main()
