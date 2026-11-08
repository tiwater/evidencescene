#!/usr/bin/env python3
"""Render the flagship split-screen 4-camera + 3D reconstructed workspace video
(docs/assets/videos/bimanual_pouring_teleop_hud.mp4) directly from the real
Three.js WebGL scene (#interactive-scene3d) in headless Chrome via Playwright.

Supports HTTP 206 Partial Content (Byte-Range requests) so HTML5 <video>
seeking in Chromium locks the 4 RGB-D camera canvases to the exact millisecond
of every 3D trajectory frame.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import tempfile

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_MP4 = REPO_ROOT / "docs" / "assets" / "videos" / "bimanual_pouring_teleop_hud.mp4"
NODE_BIN = "/home/hugh/.local/node-v22.22.3/bin/node"
NODE_PATH = "/home/hugh/.supen/daemon/node_modules"
CHROME_BIN = "/usr/bin/google-chrome"

PLAYWRIGHT_SCRIPT = r"""
const http = require("http");
const fs = require("fs");
const path = require("path");
const { chromium } = require("playwright");

const root = process.env.DOCS_ROOT;
const outDir = process.env.FRAMES_DIR;

const mime = {
  ".html": "text/html",
  ".js": "application/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".bin": "application/octet-stream",
  ".mp4": "video/mp4",
  ".png": "image/png",
  ".svg": "image/svg+xml"
};

function smoothstep(a, b, x) {
  const t = Math.max(0, Math.min(1, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

const server = http.createServer((req, res) => {
  const cleanUrl = req.url.split("?")[0];
  let fp = path.join(root, cleanUrl === "/" ? "index.html" : cleanUrl);
  if (fs.existsSync(fp) && fs.statSync(fp).isDirectory()) fp = path.join(fp, "index.html");
  if (!fs.existsSync(fp)) {
    res.writeHead(404);
    res.end("Not found");
    return;
  }
  const stat = fs.statSync(fp);
  const total = stat.size;
  const ext = path.extname(fp);
  const contentType = mime[ext] || "application/octet-stream";
  const range = req.headers.range;

  if (range) {
    const parts = range.replace(/bytes=/, "").split("-");
    const start = parseInt(parts[0], 10);
    const end = parts[1] ? parseInt(parts[1], 10) : total - 1;
    const chunksize = (end - start) + 1;
    res.writeHead(206, {
      "Content-Range": `bytes ${start}-${end}/${total}`,
      "Accept-Ranges": "bytes",
      "Content-Length": chunksize,
      "Content-Type": contentType
    });
    fs.createReadStream(fp, { start, end }).pipe(res);
  } else {
    res.writeHead(200, {
      "Content-Length": total,
      "Accept-Ranges": "bytes",
      "Content-Type": contentType
    });
    fs.createReadStream(fp).pipe(res);
  }
});

server.listen(0, async () => {
  const port = server.address().port;
  const browser = await chromium.launch({
    executablePath: process.env.CHROME_BIN,
    args: ["--no-sandbox", "--use-gl=angle", "--use-angle=swiftshader", "--enable-webgl"]
  });
  const page = await browser.newPage({
    viewport: { width: 1520, height: 1080 },
    deviceScaleFactor: 1.25
  });
  await page.goto(`http://127.0.0.1:${port}/index.html`, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => {
    const v = window.__scene3dViewer;
    return v && v.meta && v.videoEl && v.videoEl.readyState >= 2;
  });

  const nFrames = await page.evaluate(() => {
    const v = window.__scene3dViewer;
    v.viewState.playing = false;
    v.viewState.autoOrbit = false;
    if (v.videoEl) v.videoEl.pause();
    return v.meta.frames.length;
  });

  const el = await page.$("#interactive-scene3d");
  console.log(`Capturing ${nFrames} synchronized 4-camera + WebGL 3D frames...`);

  for (let i = 0; i < nFrames; i++) {
    const u = i / Math.max(1, nFrames - 1);
    const wClose = smoothstep(0.26, 0.45, u) * (1.0 - smoothstep(0.72, 0.90, u));
    const wTopTilt = smoothstep(0.54, 0.64, u) * (1.0 - smoothstep(0.68, 0.78, u));

    const tx = lerp(0.24, 0.34, wClose);
    const ty = lerp(-0.25, -0.14, wClose);
    const tz = lerp(-0.25, -0.15, wClose);
    const dist = lerp(1.46, 0.68, wClose);
    const pitch = lerp(0.35, 0.34, wClose) + 0.24 * wTopTilt;
    const yaw = 0.48 + 0.42 * Math.sin(u * Math.PI * 2.0);

    await page.evaluate(async ({ idx, tx, ty, tz, dist, pitch, yaw }) => {
      const v = window.__scene3dViewer;
      v.viewState.target.set(tx, ty, tz);
      v.viewState.distance = dist;
      v.viewState.pitch = pitch;
      v.viewState.yaw = yaw;
      await v.seekToFrame(idx);
    }, { idx: i, tx, ty, tz, dist, pitch, yaw });

    const framePath = path.join(outDir, `frame_${String(i).padStart(4, "0")}.png`);
    await el.screenshot({ path: framePath });
    if ((i + 1) % 25 === 0 || i === nFrames - 1) {
      console.log(`  Rendered frame ${i + 1} / ${nFrames}`);
    }
  }

  await browser.close();
  server.close();
});
"""


def main() -> None:
    OUT_MP4.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        env = os.environ.copy()
        env["NODE_PATH"] = NODE_PATH
        env["DOCS_ROOT"] = str(REPO_ROOT / "docs")
        env["FRAMES_DIR"] = tmpdir
        env["CHROME_BIN"] = CHROME_BIN

        subprocess.run([NODE_BIN, "-e", PLAYWRIGHT_SCRIPT], env=env, check=True)

        print("Encoding H.264 MP4 via ffmpeg...")
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            "15",
            "-i",
            str(pathlib.Path(tmpdir) / "frame_%04d.png"),
            "-vf",
            "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=#070b11",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
            str(OUT_MP4),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Saved {OUT_MP4} ({OUT_MP4.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
