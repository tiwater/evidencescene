#!/usr/bin/env python3
"""Render the flagship 1920x1080 split-screen MP4 video (bimanual_pouring_teleop_hud.mp4).

Left side : 4 synchronized on-site RGB-D cameras (head, chest, left_arm, right_arm) + live object inventory
Right side: Real Three.js 3D WebGL reconstructed scene (57-STL Robot + Table + 4 AprilTags +
            Target Cup + Milk Carton + Surfels + Spout-to-Cup Alignment HUD) smoothly orbiting in 3D.
"""

from __future__ import annotations

import base64
import http.server
import json
import math
import pathlib
import socketserver
import subprocess
import tempfile
import threading
import cv2
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
VIEW_ROOT = pathlib.Path("/home/hugh/tc/ticos-edge/artifacts/whole-recording-view-20261026")
OUT_MP4 = DOCS_DIR / "assets" / "videos" / "bimanual_pouring_teleop_hud.mp4"


def camera_trajectory(idx: int, total: int) -> tuple[float, float, float, list[float], str]:
    """Smooth 3D orbital camera trajectory (yaw, pitch, distance, target, view_label)."""
    u = idx / max(1, total - 1)
    # Smoothly orbit in 3D around the robot, table, cup, and milk carton
    if idx < 42:
        # Phase 1: Full workspace 3D orbit (Reach & Setup)
        t = idx / 42.0
        s = 0.5 - 0.5 * math.cos(math.pi * t)
        yaw = 0.82 - 0.52 * s
        pitch = 0.36 + 0.08 * s
        dist = 1.34 - 0.14 * s
        target = [0.22 + 0.04 * s, -0.30 + 0.08 * s, -0.20 + 0.04 * s]
        label = "3D Orbit View | Full Workspace (Robot + Table + Cup + Carton)"
    elif idx < 82:
        # Phase 2: Orbit & zoom toward grasp and transport to cup rim
        t = (idx - 42) / 40.0
        s = 0.5 - 0.5 * math.cos(math.pi * t)
        yaw = 0.30 - 0.72 * s
        pitch = 0.44 + 0.12 * math.sin(math.pi * t)
        dist = 1.20 - 0.30 * s
        target = [0.26 + 0.05 * s, -0.22 + 0.06 * s, -0.16]
        label = "3D Orbit View | Grasp & Transport (Incremental Carton Surfels)"
    elif idx < 122:
        # Phase 3: Active pouring over cup rim - continuous 3D rotation around cup & carton
        t = (idx - 82) / 40.0
        s = 0.5 - 0.5 * math.cos(math.pi * t)
        yaw = -0.42 + 1.10 * s
        # Sweep pitch up to show top-oblique XY alignment then down for side XZ clearance
        pitch = 0.44 + 0.32 * math.sin(2 * math.pi * t)
        dist = 0.90 + 0.10 * math.sin(math.pi * t)
        target = [0.31 - 0.03 * s, -0.16 - 0.04 * s, -0.16]
        if t < 0.45:
            label = "3D Orbit View | Top-Oblique Pour Alignment (Spout In Cup Rim)"
        else:
            label = "3D Orbit View | Side-Oblique Rim Clearance & Pour Tilt"
    else:
        # Phase 4: Orbit back to full workspace overview during placement & return
        t = (idx - 122) / max(1, (total - 1 - 122))
        s = 0.5 - 0.5 * math.cos(math.pi * t)
        yaw = 0.68 - 0.14 * s
        pitch = 0.44 - 0.06 * s
        dist = 0.96 + 0.36 * s
        target = [0.28 - 0.06 * s, -0.20 - 0.10 * s, -0.16 - 0.04 * s]
        label = "3D Orbit View | Carton Placement & Bimanual Return"
    return yaw, pitch, dist, target, label


def draw_rounded_box(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    bg_bgr: tuple[int, int, int],
    border_bgr: tuple[int, int, int] | None = None,
    alpha: float = 0.88,
) -> None:
    x1, y1 = max(0, pt1[0]), max(0, pt1[1])
    x2, y2 = min(img.shape[1], pt2[0]), min(img.shape[0], pt2[1])
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    fill = np.full_like(roi, bg_bgr)
    cv2.addWeighted(fill, alpha, roi, 1.0 - alpha, 0, roi)
    if border_bgr is not None:
        cv2.rectangle(img, (x1, y1), (x2 - 1, y2 - 1), border_bgr, 1, cv2.LINE_AA)


def main() -> None:
    meta = json.loads((DOCS_DIR / "assets" / "scene3d" / "scene_meta.json").read_text(encoding="utf-8"))
    frames_meta = meta["frames"]
    n_frames = len(frames_meta)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir)
        webgl_dir = tmp_path / "webgl"
        comp_dir = tmp_path / "comp"
        webgl_dir.mkdir()
        comp_dir.mkdir()

        # Precompute trajectory array for browser
        traj_list = []
        for i in range(n_frames):
            yaw, pitch, dist, target, label = camera_trajectory(i, n_frames)
            traj_list.append(
                {
                    "yaw": yaw,
                    "pitch": pitch,
                    "distance": dist,
                    "target": target,
                    "label": label,
                }
            )

        # Write temporary driver HTML inside docs/
        driver_html_path = DOCS_DIR / "_render_flagship_driver.html"
        driver_html_path.write_text(
            f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ width: 1120px; height: 964px; background: #0b1017; overflow: hidden; }}
  #vwrap {{ width: 1120px; height: 964px; position: relative; }}
  canvas {{ width: 1120px; height: 964px; display: block; }}
</style>
</head>
<body>
<div id="vwrap">
  <canvas data-scene3d-canvas width="1120" height="964"></canvas>
</div>
<script type="module">
  import * as THREE from "./assets/js/three.module.min.js";
  import {{ initScene3DViewer }} from "./assets/js/scene3d-viewer.js";

  const traj = {json.dumps(traj_list)};
  async function run() {{
    const viewer = await initScene3DViewer(document.getElementById("vwrap"), {{
      autoOrbit: false,
      autoPlay: false,
      initialFrame: 0
    }});
    const canvas = document.querySelector("[data-scene3d-canvas]");
    for (let i = 0; i < traj.length; i++) {{
      const t = traj[i];
      viewer.viewState.yaw = t.yaw;
      viewer.viewState.pitch = t.pitch;
      viewer.viewState.distance = t.distance;
      viewer.viewState.target.set(t.target[0], t.target[1], t.target[2]);
      viewer.applyFrame(i, false);
      viewer.updateCameraAndRender();
      const dataUrl = canvas.toDataURL("image/jpeg", 0.96);
      await fetch(`/save_webgl_frame?idx=${{i}}`, {{
        method: "POST",
        body: dataUrl
      }});
    }}
    await fetch("/done_rendering", {{ method: "POST" }});
  }}
  run();
</script>
</body>
</html>""",
            encoding="utf-8",
        )

        done_event = threading.Event()

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(DOCS_DIR), **kwargs)

            def log_message(self, format, *args):
                pass

            def do_POST(self):
                if self.path.startswith("/save_webgl_frame"):
                    idx = int(self.path.split("idx=")[1])
                    length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(length).decode("utf-8")
                    b64 = body.split(",", 1)[1]
                    (webgl_dir / f"webgl_{idx:04d}.jpg").write_bytes(base64.b64decode(b64))
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"ok")
                elif self.path.startswith("/done_rendering"):
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"done")
                    done_event.set()

        class ThreadedServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
            allow_reuse_address = True

        server = ThreadedServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
        srv_thread.start()

        print(f"1. Rendering {n_frames} Three.js 3D WebGL frames via headless Chrome on port {port}...")
        chrome_cmd = [
            "/usr/bin/google-chrome",
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu-sandbox",
            "--use-gl=angle",
            "--use-angle=swiftshader",
            "--window-size=1120,964",
            f"http://127.0.0.1:{port}/_render_flagship_driver.html",
        ]
        proc = subprocess.Popen(chrome_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        done_event.wait(timeout=90)
        proc.terminate()
        server.shutdown()
        driver_html_path.unlink(missing_ok=True)

        rendered_count = len(list(webgl_dir.glob("webgl_*.jpg")))
        print(f"   Rendered {rendered_count}/{n_frames} WebGL 3D frames.")
        assert rendered_count == n_frames, f"Expected {n_frames} WebGL frames, got {rendered_count}"

        print("2. Compositing 1920x1080 split-screen frames (Left: 4 RGB-D cameras; Right: 3D scene + HUD)...")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cup_rad_mm = 42.8

        for i, fmeta in enumerate(frames_meta):
            src_idx = fmeta["src_frame_index"]
            _, _, _, _, view_label = camera_trajectory(i, n_frames)

            canvas = np.full((1080, 1920, 3), (18, 13, 9), dtype=np.uint8)

            # Top header bar (1920 x 54)
            draw_rounded_box(canvas, (0, 0), (1920, 54), (28, 20, 14), (65, 50, 38), alpha=1.0)
            cv2.putText(
                canvas,
                "ON-SITE MULTI-CAMERA RGB-D STREAMS (4 SYNCHRONIZED CAMERAS)",
                (20, 34),
                font,
                0.62,
                (248, 250, 252),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                "3D RECONSTRUCTED WORKSPACE (ROBOT + TABLE + CUP + MILK CARTON)",
                (818, 34),
                font,
                0.62,
                (252, 211, 77),
                2,
                cv2.LINE_AA,
            )

            # Right panel: 3D WebGL scene (1120 x 964) placed at x=800, y=54
            webgl_img = cv2.imread(str(webgl_dir / f"webgl_{i:04d}.jpg"))
            if webgl_img.shape[:2] != (964, 1120):
                webgl_img = cv2.resize(webgl_img, (1120, 964), interpolation=cv2.INTER_AREA)
            canvas[54 : 54 + 964, 800:1920] = webgl_img

            # Divider line between Left and Right panels
            cv2.line(canvas, (800, 0), (800, 1080), (75, 55, 40), 2)

            # Left panel: 2x2 grid of the 4 on-site cameras (x: 16..784, y: 68..624)
            cam_specs = [
                ("head", "png", "HEAD RGB-D", "Surface Input", (248, 189, 56), 16, 68),
                ("chest", "png", "CHEST RGB-D", "Surface Input", (248, 189, 56), 404, 68),
                ("left_arm", "jpg", "LEFT ARM WRIST", "Context View", (180, 165, 148), 16, 348),
                ("right_arm", "jpg", "RIGHT ARM WRIST", "Context View", (180, 165, 148), 404, 348),
            ]
            tw, th = 380, 214
            for cam_id, ext, title, role, badge_col, cx, cy in cam_specs:
                draw_rounded_box(
                    canvas, (cx, cy), (cx + tw, cy + th + 52), (30, 22, 15), (68, 52, 38), alpha=1.0
                )
                cv2.putText(canvas, title, (cx + 10, cy + 20), font, 0.46, (240, 244, 248), 1, cv2.LINE_AA)
                cv2.putText(canvas, role, (cx + tw - 110, cy + 20), font, 0.42, badge_col, 1, cv2.LINE_AA)

                im_path = VIEW_ROOT / f"{src_idx:06d}-{cam_id}.{ext}"
                cam_im = cv2.imread(str(im_path))
                cam_im = cv2.resize(cam_im, (tw - 4, th), interpolation=cv2.INTER_AREA)
                canvas[cy + 28 : cy + 28 + th, cx + 2 : cx + 2 + tw - 4] = cam_im

                cap_txt = f"Source frame #{src_idx + 1:03d} | t = {fmeta['elapsed_s']:.2f}s | Synced"
                cv2.putText(
                    canvas, cap_txt, (cx + 10, cy + th + 45), font, 0.38, (148, 163, 184), 1, cv2.LINE_AA
                )

            # Left panel bottom card: Reconstructed 3D Scene Inventory & Evidence Telemetry (y: 636..1006)
            draw_rounded_box(canvas, (16, 636), (784, 1006), (26, 19, 13), (72, 56, 42), alpha=0.96)
            cv2.putText(
                canvas,
                "RECONSTRUCTED 3D SCENE ENTITIES (SYNCHRONIZED IN TABLE FRAME Z=0)",
                (32, 666),
                font,
                0.50,
                (248, 250, 252),
                1,
                cv2.LINE_AA,
            )

            n_obj = fmeta["raw_object_point_count"]
            n_cup = fmeta["raw_cup_head_count"] + fmeta["raw_cup_chest_count"]
            items = [
                (
                    (212, 234, 94),
                    "1. Bimanual Robot (URDF Overlay)",
                    f"LSRX_V0.0.13 | 61 links & 57 calibrated STL meshes | Grip->Carton: {fmeta['grip_mm']:.1f} mm",
                ),
                (
                    (248, 189, 56),
                    "2. Held Milk Carton (Object Gauge -> Table)",
                    f"3D Box (68x95x200 mm) + Pour Spout + {n_obj:,} live incremental surfels | Tilt: {fmeta['tilt_deg']:.1f} deg",
                ),
                (
                    (16, 185, 129),
                    "3. Target Cup on Table",
                    f"3D Tapered Cylinder (Dia 85.6x163.2 mm) + Rim Bullseye + {n_cup:,} live & 360 deg surfels",
                ),
                (
                    (21, 204, 250),
                    "4. Calibrated Table & 4 AprilTags",
                    "Table plane (z = 0) + 5cm/10cm metric grid + 4 yellow AprilTag landmarks (Tag 0..3)",
                ),
            ]
            iy = 702
            for dot_col, it_title, it_desc in items:
                cv2.circle(canvas, (40, iy - 5), 6, dot_col, -1, cv2.LINE_AA)
                cv2.putText(canvas, it_title, (56, iy), font, 0.48, (241, 245, 249), 1, cv2.LINE_AA)
                cv2.putText(canvas, it_desc, (56, iy + 22), font, 0.41, (168, 182, 198), 1, cv2.LINE_AA)
                iy += 56

            # Bottom note inside left inventory card
            draw_rounded_box(canvas, (32, 932), (768, 992), (38, 28, 18), (94, 72, 52), alpha=0.9)
            cv2.putText(
                canvas,
                "Interactive WebGL Viewer on Page: Drag right 3D scene with mouse to rotate yaw/pitch & scroll to zoom.",
                (46, 956),
                font,
                0.41,
                (252, 211, 77),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"Active Camera Mode: {view_label}",
                (46, 980),
                font,
                0.41,
                (125, 211, 252),
                1,
                cv2.LINE_AA,
            )

            # Right panel overlays: Top-Left Spout-to-Cup Pour Alignment HUD (x: 820..1460, y: 72..182)
            in_rim = fmeta["in_rim"]
            horiz_mm = fmeta["horiz_mm"]
            above_mm = fmeta["above_mm"]
            tilt_deg = fmeta["tilt_deg"]
            grip_mm = fmeta["grip_mm"]

            hud_border = (129, 185, 16) if in_rim else (56, 189, 248)
            draw_rounded_box(canvas, (820, 72), (1515, 184), (22, 16, 11), hud_border, alpha=0.88)

            # Bullseye reticle at (876, 128)
            rcx, rcy = 876, 128
            cv2.circle(canvas, (rcx, rcy), 40, (64, 50, 38), 1, cv2.LINE_AA)
            cv2.line(canvas, (rcx - 42, rcy), (rcx + 42, rcy), (55, 42, 32), 1, cv2.LINE_AA)
            cv2.line(canvas, (rcx, rcy - 42), (rcx, rcy + 42), (55, 42, 32), 1, cv2.LINE_AA)
            rim_col = (129, 185, 16) if in_rim else (11, 158, 245)
            cv2.circle(canvas, (rcx, rcy), 20, rim_col, 2, cv2.LINE_AA)

            scale = 20.0 / cup_rad_mm
            raw_x = fmeta["dy_mm"] * scale
            raw_y = -fMeta_dx if False else -fmeta["dx_mm"] * scale
            raw_r = math.hypot(raw_x, raw_y)
            if raw_r > 38.0:
                raw_x = raw_x / raw_r * 38.0
                raw_y = raw_y / raw_r * 38.0
            dx_px, dy_px = int(round(rcx + raw_x)), int(round(rcy + raw_y))
            cv2.line(canvas, (rcx, rcy), (dx_px, dy_px), rim_col, 2, cv2.LINE_AA)
            cv2.circle(canvas, (dx_px, dy_px), 5, rim_col, -1, cv2.LINE_AA)

            if in_rim:
                st_txt = f"POUR ALIGNED - IN CUP RIM (+{cup_rad_mm - horiz_mm:.1f} mm margin)"
                st_col = (153, 211, 52)
            elif horiz_mm <= cup_rad_mm * 2.0:
                st_txt = f"APPROACHING CUP RIM ({horiz_mm - cup_rad_mm:.1f} mm outside)"
                st_col = (77, 211, 252)
            elif grip_mm <= 38.0:
                st_txt = "GRASP ENGAGED - TRANSPORTING MILK CARTON"
                st_col = (248, 189, 56)
            else:
                st_txt = "BIMANUAL REACH / WORKSPACE STANDBY"
                st_col = (203, 213, 225)

            cv2.putText(canvas, st_txt, (934, 102), font, 0.52, st_col, 2, cv2.LINE_AA)
            r1 = f"Spout dXY: {horiz_mm:.1f} mm / R {cup_rad_mm:.1f} mm | dX {fmeta['dx_mm']:+.0f} mm | dY {fmeta['dy_mm']:+.0f} mm"
            r2 = f"Rim Clearance dZ: {above_mm:+.1f} mm | Pour Tilt: {tilt_deg:.1f} deg | Grip->Carton: {grip_mm:.1f} mm"
            cv2.putText(canvas, r1, (934, 134), font, 0.44, (226, 232, 240), 1, cv2.LINE_AA)
            cv2.putText(canvas, r2, (934, 164), font, 0.44, (226, 232, 240), 1, cv2.LINE_AA)

            # Right panel Top-Right badge: 3D View Orbit indicator
            draw_rounded_box(canvas, (1528, 72), (1900, 132), (22, 16, 11), (82, 64, 48), alpha=0.88)
            cv2.putText(
                canvas,
                "3D VIEWPORT (ROTATABLE)",
                (1542, 98),
                font,
                0.45,
                (252, 211, 77),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                view_label.replace("3D Orbit View | ", ""),
                (1542, 120),
                font,
                0.39,
                (186, 230, 253),
                1,
                cv2.LINE_AA,
            )

            # Right panel Bottom legend bar (x: 820..1900, y: 966..1006)
            draw_rounded_box(canvas, (820, 964), (1900, 1006), (22, 16, 11), (68, 52, 38), alpha=0.90)
            leg_items = [
                ((212, 234, 94), "Robot (57 STL)"),
                ((248, 189, 56), "Milk Carton + Surfels"),
                ((11, 158, 245), "Target Cup + Surfels"),
                ((21, 204, 250), "Table (z=0) + 4 AprilTags"),
                ((94, 63, 244), "Spout->Cup Vector"),
            ]
            lx = 838
            for lcol, ltxt in leg_items:
                cv2.circle(canvas, (lx, 985), 5, lcol, -1, cv2.LINE_AA)
                cv2.putText(canvas, ltxt, (lx + 12, 990), font, 0.42, (226, 232, 240), 1, cv2.LINE_AA)
                lx += 210

            # Bottom transport & timeline bar (1920 x 62 at y: 1018..1080)
            draw_rounded_box(canvas, (0, 1018), (1920, 1080), (24, 17, 12), (65, 50, 38), alpha=1.0)
            cv2.putText(
                canvas,
                f"FRAME {src_idx + 1:03d} / 498  ({fmeta['elapsed_s']:04.1f}s / 33.1s)",
                (24, 1055),
                font,
                0.50,
                (241, 245, 249),
                1,
                cv2.LINE_AA,
            )
            # Progress track (x: 380..1540, y: 1044..1054)
            bar_x1, bar_x2 = 380, 1540
            cv2.rectangle(canvas, (bar_x1, 1044), (bar_x2, 1054), (55, 42, 32), -1)
            # Highlight Pour Aligned window on timeline (frames 255..345 out of 498)
            pa_x1 = bar_x1 + int((255 / 497.0) * (bar_x2 - bar_x1))
            pa_x2 = bar_x1 + int((345 / 497.0) * (bar_x2 - bar_x1))
            cv2.rectangle(canvas, (pa_x1, 1044), (pa_x2, 1054), (65, 110, 25), -1)

            prog_x = bar_x1 + int((src_idx / 497.0) * (bar_x2 - bar_x1))
            cv2.rectangle(canvas, (bar_x1, 1044), (prog_x, 1054), (248, 189, 56), -1)
            cv2.circle(canvas, (prog_x, 1049), 8, (252, 211, 77), -1, cv2.LINE_AA)

            tot_pts = n_obj + n_cup
            cv2.putText(
                canvas,
                f"Live Surfels: {tot_pts:,} pts",
                (1570, 1055),
                font,
                0.50,
                (153, 211, 52),
                1,
                cv2.LINE_AA,
            )

            cv2.imwrite(str(comp_dir / f"frame_{i:04d}.png"), canvas)

        print("3. Encoding flagship MP4 (docs/assets/videos/bimanual_pouring_teleop_hud.mp4)...")
        OUT_MP4.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            "15",
            "-i",
            str(comp_dir / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "19",
            "-movflags",
            "+faststart",
            str(OUT_MP4),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"   Saved {OUT_MP4}: {OUT_MP4.stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    main()
