"""Generate MP4 demonstration videos for the EvidenceScene project page."""

from __future__ import annotations

import glob
import json
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "assets" / "videos"


def _encode_mp4(frames_bgr: list[np.ndarray], out_path: Path, fps: int = 24) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, frame in enumerate(frames_bgr):
            cv2.imwrite(os.path.join(tmpdir, f"{idx:05d}.png"), frame)
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            os.path.join(tmpdir, "%05d.png"),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Wrote {out_path} ({len(frames_bgr)} frames, {out_path.stat().st_size / 1024:.1f} KB)")


def _load_ascii_ply(ply_path: str) -> tuple[np.ndarray, np.ndarray]:
    pts = []
    colors = []
    with open(ply_path, "r", encoding="latin1") as f:
        in_Import = False
        n_verts = 0
        count = 0
        for line in f:
            s = line.strip()
            if s.startswith("element vertex"):
                n_verts = int(s.split()[-1])
            elif s == "end_header":
                in_Import = True
                continue
            elif in_Import:
                parts = s.split()
                if len(parts) >= 6:
                    pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    # BGR for OpenCV
                    colors.append([int(parts[5]), int(parts[4]), int(parts[3])])
                count += 1
                if count >= n_verts:
                    break
    return np.asarray(pts, dtype=np.float64), np.asarray(colors, dtype=np.uint8)


def _draw_axes(img: np.ndarray, K: np.ndarray, T: np.ndarray, length: float = 0.08) -> None:
    origin = np.array([[0.0, 0.0, 0.0], [length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]])
    pts_c = (T[:3, :3] @ origin.T).T + T[:3, 3]
    if np.any(pts_c[:, 2] <= 0.05):
        return
    uv = (K @ pts_c.T).T
    uv = (uv[:, :2] / uv[:, 2:3]).astype(np.int32)
    p0 = tuple(uv[0])
    cv2.line(img, p0, tuple(uv[1]), (40, 50, 235), 2, cv2.LINE_AA)   # X: Red
    cv2.line(img, p0, tuple(uv[2]), (50, 205, 50), 2, cv2.LINE_AA)   # Y: Green
    cv2.line(img, p0, tuple(uv[3]), (235, 130, 30), 2, cv2.LINE_AA)  # Z: Blue


def _draw_3d_box(img: np.ndarray, K: np.ndarray, T: np.ndarray, pmin: np.ndarray, pmax: np.ndarray, color=(0, 215, 255)) -> None:
    corners = np.array([
        [pmin[0], pmin[1], pmin[2]],
        [pmax[0], pmin[1], pmin[2]],
        [pmax[0], pmax[1], pmin[2]],
        [pmin[0], pmax[1], pmin[2]],
        [pmin[0], pmin[1], pmax[2]],
        [pmax[0], pmin[1], pmax[2]],
        [pmax[0], pmax[1], pmax[2]],
        [pmin[0], pmax[1], pmax[2]],
    ])
    pts_c = (T[:3, :3] @ corners.T).T + T[:3, 3]
    if np.any(pts_c[:, 2] <= 0.05):
        return
    uv = (K @ pts_c.T).T
    uv = (uv[:, :2] / uv[:, 2:3]).astype(np.int32)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i, j in edges:
        cv2.line(img, tuple(uv[i]), tuple(uv[j]), color, 2, cv2.LINE_AA)


def generate_bimanual_pouring_video() -> None:
    src_dir = Path("/home/hugh/tc/ticos-edge/artifacts/whole-recording-view-20261026")
    if not src_dir.exists():
        print("Skipping bimanual pouring video (source directory not found)")
        return

    W, H = 1280, 540
    frames_out: list[np.ndarray] = []
    traj_xy: list[tuple[float, float]] = []
    traj_xz: list[tuple[float, float]] = []

    indices = list(range(10, 465, 3))
    for fidx in indices:
        json_path = src_dir / f"{fidx:06d}.json"
        chest_path = src_dir / f"{fidx:06d}-chest.png"
        head_path = src_dir / f"{fidx:06d}-head.png"
        left_path = src_dir / f"{fidx:06d}-left_arm.jpg"
        right_path = src_dir / f"{fidx:06d}-right_arm.jpg"
        if not (json_path.exists() and chest_path.exists() and head_path.exists()):
            continue

        canvas = np.full((H, W, 3), 248, dtype=np.uint8)

        # Left column (480x540): 2x2 synchronized camera views
        cam_w, cam_h = 228, 172
        cams = [
            (head_path, "Head RGB-D", 12, 42),
            (chest_path, "Chest RGB-D", 246, 42),
            (left_path, "Left Wrist RGB-D", 12, 232),
            (right_path, "Right Wrist RGB-D", 246, 232),
        ]
        for cpath, label, x0, y0 in cams:
            im = cv2.imread(str(cpath))
            if im is not None:
                im_r = cv2.resize(im, (cam_w, cam_h), interpolation=cv2.INTER_AREA)
                canvas[y0 : y0 + cam_h, x0 : x0 + cam_w] = im_r
                cv2.rectangle(canvas, (x0, y0), (x0 + cam_w, y0 + cam_h), (200, 205, 212), 1)
                cv2.rectangle(canvas, (x0, y0), (x0 + 132, y0 + 22), (25, 32, 44), -1)
                cv2.putText(canvas, label, (x0 + 6, y0 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        data = json.loads(json_path.read_text())
        ss = data.get("surface_snapshot") or {}
        support = np.asarray(ss.get("observation_support") or [], dtype=np.float32)
        n_surfels = int(ss.get("point_count") or len(support))
        revised_cnt = int(ss.get("revised_observation_count") or 0)

        layers = (data.get("measured_table_layers") or {}).get("layers") or []
        carton_pts = np.zeros((0, 3))
        cup_pts = []
        for lyr in layers:
            lid = lyr.get("layer_id", "")
            pts = np.asarray(lyr.get("points_table_m") or [], dtype=np.float64)
            if len(pts) == 0:
                continue
            if lid == "primary-object":
                carton_pts = pts
            elif "cup" in lid:
                cup_pts.append(pts)
        cup_pts_all = np.vstack(cup_pts) if cup_pts else np.zeros((0, 3))

        # Status bar on bottom-left
        cv2.rectangle(canvas, (12, 418), (474, 524), (255, 255, 255), -1)
        cv2.rectangle(canvas, (12, 418), (474, 524), (210, 215, 222), 1)
        cv2.putText(canvas, f"Frame {fidx:03d} / 465  |  4-Camera Synchronized Pouring Replay", (24, 442), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (20, 28, 40), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"RevisableSurfelMap: {n_surfels:,} corroborated surfels (revised: {revised_cnt})", (24, 468), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (45, 95, 195), 1, cv2.LINE_AA)
        c10_ratio = float(np.mean(support >= 10) * 100.0) if len(support) else 0.0
        cv2.putText(canvas, f"Multi-View Support (c >= 10): {c10_ratio:.1f}%  |  Voxel: 4.0 mm", (24, 494), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (16, 125, 80), 1, cv2.LINE_AA)

        # Center Panel (486..896, 42..524): 3D Isometric Tabletop + Reconstructed Surfaces
        px0, py0, pw, ph = 488, 42, 400, 482
        cv2.rectangle(canvas, (px0, py0), (px0 + pw, py0 + ph), (20, 26, 36), -1)
        cv2.putText(canvas, "Online 3D Reconstruction (RevisableSurfelMap)", (px0 + 14, py0 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 240, 248), 1, cv2.LINE_AA)

        def proj_iso(pts3d: np.ndarray) -> np.ndarray:
            # Table frame: X forward [0.05..0.50], Y lateral [-0.35..0.05], E = -Z elevation [0..0.38]
            x = pts3d[:, 0] - 0.28
            y = pts3d[:, 1] + 0.15
            e = -pts3d[:, 2]
            yaw = np.deg2rad(-38.0)
            pitch = np.deg2rad(26.0)
            xr = x * np.cos(yaw) - y * np.sin(yaw)
            yr = x * np.sin(yaw) + y * np.cos(yaw)
            u = px0 + pw * 0.50 + yr * 920.0
            v = py0 + ph * 0.78 - (e * np.cos(pitch) - xr * np.sin(pitch)) * 920.0
            return np.stack([u, v], axis=-1).astype(np.int32)

        # Draw tabletop grid
        for gx in np.linspace(0.08, 0.48, 9):
            seg = np.array([[gx, -0.34, 0.0], [gx, 0.04, 0.0]])
            uv = proj_iso(seg)
            cv2.line(canvas, tuple(uv[0]), tuple(uv[1]), (48, 58, 74), 1, cv2.LINE_AA)
        for gy in np.linspace(-0.34, 0.04, 9):
            seg = np.array([[0.08, gy, 0.0], [0.48, gy, 0.0]])
            uv = proj_iso(seg)
            cv2.line(canvas, tuple(uv[0]), tuple(uv[1]), (48, 58, 74), 1, cv2.LINE_AA)

        # Draw cup surfels (cyan/emerald)
        if len(cup_pts_all):
            uv_cup = proj_iso(cup_pts_all[::2])
            for u, v in uv_cup:
                if px0 + 4 <= u < px0 + pw - 4 and py0 + 36 <= v < py0 + ph - 4:
                    canvas[v - 1 : v + 1, u - 1 : u + 1] = (210, 220, 45)

        # Draw carton surfels colored by observation support
        if len(carton_pts):
            c_cent = carton_pts.mean(axis=0)
            traj_xy.append((float(c_cent[0]), float(c_cent[1])))
            traj_xz.append((float(c_cent[0]), float(-c_cent[2])))
            step = max(1, len(carton_pts) // 2200)
            pts_sub = carton_pts[::step]
            sup_sub = support[::step] if len(support) == len(carton_pts) else np.full(len(pts_sub), 12.0)
            uv_car = proj_iso(pts_sub)
            for (u, v), s_val in zip(uv_car, sup_sub):
                if px0 + 4 <= u < px0 + pw - 4 and py0 + 36 <= v < py0 + ph - 4:
                    color = (95, 210, 65) if s_val >= 10 else (55, 165, 245)
                    canvas[v - 1 : v + 2, u - 1 : u + 2] = color

        # Draw 3D trajectory trail in isometric view
        if len(traj_xy) >= 2:
            trail3d = np.array([[xy[0], xy[1], -xz[1]] for xy, xz in zip(traj_xy, traj_xz)])
            uv_t = proj_iso(trail3d)
            for i in range(1, len(uv_t)):
                cv2.line(canvas, tuple(uv_t[i - 1]), tuple(uv_t[i]), (80, 180, 255), 2, cv2.LINE_AA)

        cv2.putText(canvas, "Green: Corroborated (c>=10)  Cyan: Target Cup", (px0 + 14, py0 + ph - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 190, 205), 1, cv2.LINE_AA)

        # Right Column: Top-XY Bullseye (top) and Side-XZ Clearance (bottom)
        rx0, rw = 900, 368
        # 1) Top-XY Bullseye (py0..py0+234)
        by0, bh = 42, 234
        cv2.rectangle(canvas, (rx0, by0), (rx0 + rw, by0 + bh), (20, 26, 36), -1)
        cv2.putText(canvas, "Orthographic Top-XY Bullseye", (rx0 + 12, by0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 240, 248), 1, cv2.LINE_AA)

        def to_xy(pts3d: np.ndarray) -> np.ndarray:
            u = rx0 + 30 + (pts3d[:, 0] - 0.10) / 0.40 * (rw - 60)
            v = by0 + bh - 24 - (pts3d[:, 1] + 0.34) / 0.36 * (bh - 56)
            return np.stack([u, v], axis=-1).astype(np.int32)

        if len(cup_pts_all):
            cup_c = cup_pts_all.mean(axis=0)
            cup_uv = to_xy(cup_c[None, :])[0]
            cv2.circle(canvas, tuple(cup_uv), 26, (210, 220, 45), 1, cv2.LINE_AA)
            cv2.circle(canvas, tuple(cup_uv), 12, (210, 220, 45), 1, cv2.LINE_AA)
            for u, v in to_xy(cup_pts_all[::4]):
                if rx0 + 4 <= u < rx0 + rw - 4 and by0 + 30 <= v < by0 + bh - 4:
                    canvas[v, u] = (210, 220, 45)
        if len(carton_pts):
            for u, v in to_xy(carton_pts[::6]):
                if rx0 + 4 <= u < rx0 + rw - 4 and by0 + 30 <= v < by0 + bh - 4:
                    canvas[v, u] = (95, 210, 65)
        if len(traj_xy) >= 2:
            t_pts = np.array([[x, y, 0.0] for x, y in traj_xy])
            uv_xy = to_xy(t_pts)
            for i in range(1, len(uv_xy)):
                cv2.line(canvas, tuple(uv_xy[i - 1]), tuple(uv_xy[i]), (80, 180, 255), 2, cv2.LINE_AA)

        # 2) Side-XZ Clearance Profile (290..524)
        sy0, sh = 290, 234
        cv2.rectangle(canvas, (rx0, sy0), (rx0 + rw, sy0 + sh), (20, 26, 36), -1)
        cv2.putText(canvas, "Orthographic Side-XZ Rim Clearance", (rx0 + 12, sy0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 240, 248), 1, cv2.LINE_AA)
        table_y = sy0 + sh - 28
        cv2.line(canvas, (rx0 + 20, table_y), (rx0 + rw - 20, table_y), (110, 120, 135), 2)

        def to_xz(pts3d: np.ndarray) -> np.ndarray:
            u = rx0 + 30 + (pts3d[:, 0] - 0.10) / 0.40 * (rw - 60)
            v = table_y - (-pts3d[:, 2]) / 0.36 * (sh - 62)
            return np.stack([u, v], axis=-1).astype(np.int32)

        if len(cup_pts_all):
            for u, v in to_xz(cup_pts_all[::4]):
                if rx0 + 4 <= u < rx0 + rw - 4 and sy0 + 30 <= v < sy0 + sh - 4:
                    canvas[v, u] = (210, 220, 45)
        if len(carton_pts):
            for u, v in to_xz(carton_pts[::6]):
                if rx0 + 4 <= u < rx0 + rw - 4 and sy0 + 30 <= v < sy0 + sh - 4:
                    canvas[v, u] = (95, 210, 65)
            if len(cup_pts_all):
                carton_bottom_z = float((-carton_pts[:, 2]).min())
                cup_rim_z = float((-cup_pts_all[:, 2]).max())
                clearance_mm = (carton_bottom_z - cup_rim_z) * 1000.0
                cv2.putText(canvas, f"Rim Z: {cup_rim_z*100:.1f} cm | Clearance: {clearance_mm:+.1f} mm", (rx0 + 12, sy0 + sh - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (180, 220, 250), 1, cv2.LINE_AA)

        # Header title bar
        cv2.putText(canvas, "EvidenceScene: Multi-Camera RGB-D 6-DoF Tracking & RevisableSurfelMap Teleop Assistance", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (15, 23, 42), 2, cv2.LINE_AA)
        frames_out.append(canvas)

    if frames_out:
        _encode_mp4(frames_out, OUT_DIR / "bimanual_pouring_teleop_hud.mp4", fps=24)


def generate_sequence_tracking_video(
    run_dir: Path,
    title: str,
    subtitle: str,
    out_name: str,
    start_frame: int = 0,
    max_frames: int = 420,
    step: int = 3,
) -> None:
    if not run_dir.exists():
        print(f"Skipping {out_name} ({run_dir} not found)")
        return

    K = np.loadtxt(run_dir / "cam_K.txt").reshape(3, 3)
    ply_pts0, ply_clr = _load_ascii_ply(str(run_dir / "cloud_init.ply"))
    T0 = np.loadtxt(run_dir / "ob_in_cam" / "000000.txt").reshape(4, 4)
    T0_inv = np.linalg.inv(T0)
    ply_pts = (T0_inv[:3, :3] @ ply_pts0.T).T + T0_inv[:3, 3]
    pmin, pmax = np.percentile(ply_pts, 2, axis=0), np.percentile(ply_pts, 98, axis=0)

    color_files = sorted(glob.glob(str(run_dir / "color" / "*.png")))[start_frame:max_frames:step]
    frames_out: list[np.ndarray] = []
    traj_centers: list[np.ndarray] = []

    for cfile in color_files:
        stem = Path(cfile).stem
        pose_file = run_dir / "ob_in_cam" / f"{stem}.txt"
        norm_file = run_dir / "normal" / f"{stem}.png"
        seg_file = run_dir / "color_segmented" / f"{stem}.png"
        if not pose_file.exists():
            continue

        rgb = cv2.imread(cfile)
        if rgb is None:
            continue
        T = np.loadtxt(pose_file).reshape(4, 4)
        traj_centers.append(T[:3, 3].copy())

        # Panel 1: RGB + Projected 3D Surface Cloud + 3D Bounding Box + 6-DoF Axes
        p1 = rgb.copy()
        pts_c = (T[:3, :3] @ ply_pts[::3].T).T + T[:3, 3]
        valid = pts_c[:, 2] > 0.05
        if np.any(valid):
            uv = (K @ pts_c[valid].T).T
            uv = (uv[:, :2] / uv[:, 2:3]).astype(np.int32)
            overlay = p1.copy()
            h_im, w_im = p1.shape[:2]
            for (u, v), clr in zip(uv, ply_clr[::3][valid]):
                if 1 <= u < w_im - 1 and 1 <= v < h_im - 1:
                    overlay[v - 1 : v + 1, u - 1 : u + 1] = (50, 220, 100)
            p1 = cv2.addWeighted(p1, 0.68, overlay, 0.32, 0)

        _draw_3d_box(p1, K, T, pmin, pmax, color=(0, 215, 255))
        _draw_axes(p1, K, T, length=0.075)

        # Project 3D trajectory trail onto image
        if len(traj_centers) >= 2:
            tc = np.asarray(traj_centers)
            uv_t = (K @ tc.T).T
            uv_t = (uv_t[:, :2] / uv_t[:, 2:3]).astype(np.int32)
            for i in range(1, len(uv_t)):
                cv2.line(p1, tuple(uv_t[i - 1]), tuple(uv_t[i]), (255, 190, 40), 2, cv2.LINE_AA)

        # Panel 2: Segmented Object & Surface Normals
        norm_im = cv2.imread(str(norm_file)) if norm_file.exists() else np.zeros_like(rgb)
        seg_im = cv2.imread(str(seg_file)) if seg_file.exists() else np.zeros_like(rgb)
        p1_r = cv2.resize(p1, (440, 330), interpolation=cv2.INTER_AREA)
        seg_r = cv2.resize(seg_im, (220, 165), interpolation=cv2.INTER_AREA)
        norm_r = cv2.resize(norm_im, (220, 165), interpolation=cv2.INTER_AREA)
        p2_r = np.vstack([seg_r, norm_r])

        canvas = np.full((386, 684, 3), 248, dtype=np.uint8)
        canvas[46:376, 10:450] = p1_r
        canvas[46:376, 456:676] = p2_r

        cv2.rectangle(canvas, (10, 46), (225, 68), (20, 28, 40), -1)
        cv2.putText(canvas, f"6-DoF Pose & Surface (Frame {int(stem)})", (16, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (456, 46), (610, 66), (20, 28, 40), -1)
        cv2.putText(canvas, "Segmented Object RGB", (462, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (456, 211), (610, 231), (20, 28, 40), -1)
        cv2.putText(canvas, "Measured Surface Normals", (462, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(canvas, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (15, 23, 42), 1, cv2.LINE_AA)
        cv2.putText(canvas, subtitle, (12, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (71, 85, 105), 1, cv2.LINE_AA)
        frames_out.append(canvas)

    if frames_out:
        _encode_mp4(frames_out, OUT_DIR / out_name, fps=24)


def main() -> None:
    generate_bimanual_pouring_video()
    generate_sequence_tracking_video(
        Path("/home/hugh/tc/ticos-edge/artifacts/public-benchmark-ycbineoat-20261028/bleach0-bundletrack-reference-mask"),
        "YCBInEOAT — bleach0 Dexterous Manipulation",
        "EvidenceScene (Mean ADD: 26.8 mm | Duty: 96.3% | Repeat Spread: 0.00 mm / 0.00 deg)",
        "ycb_bleach0_tracking.mp4",
        start_frame=110,
        max_frames=530,
        step=3,
    )
    generate_sequence_tracking_video(
        Path("/home/hugh/tc/ticos-edge/artifacts/public-benchmark-ycbineoat-20261028/mustard0-bundletrack-reference-mask"),
        "YCBInEOAT — mustard0 Bimanual Handover & Rotation",
        "EvidenceScene Deterministic Native Backend + Bounded Recovery (0.00 mm Spread)",
        "ycb_mustard0_tracking.mp4",
        start_frame=40,
        max_frames=430,
        step=3,
    )
    generate_sequence_tracking_video(
        Path("/home/hugh/tc/ticos-edge/artifacts/laval-mocap-development-20261029/bundletrack-zfar2-v2"),
        "Laval Vicon Motion Capture — Clock 6-DoF Tracking",
        "Full-Precision CUDACache::storeFrame SE(3) Fix + Stable-Winner RANSAC (AUC: 0.917)",
        "laval_clock_tracking.mp4",
        max_frames=203,
        step=2,
    )
    generate_sequence_tracking_video(
        Path("/home/hugh/tc/ticos-edge/artifacts/laval-cookiejar-development-20261029/bundletrack-zfar2-v2"),
        "Laval Vicon Motion Capture — Cookiejar Occlusion & Rotation",
        "Contiguous-NO_BA Frame Retirement (-1.99 GiB RSS) + Observability Gate",
        "laval_cookiejar_tracking.mp4",
        max_frames=240,
        step=2,
    )


if __name__ == "__main__":
    main()
