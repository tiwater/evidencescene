"""Shared dataset loaders, provenance helpers, and SO(3)/gauge utilities for the public suite."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class YcbSequenceSpec:
    seq_id: str
    short_label: str
    ycb_object: str
    mesh_relpath: str
    prior_p2p_ref: str | None
    prior_p2p_gen: str | None
    prior_fp_cad: str | None


YCBINEOAT_SPECS: tuple[YcbSequenceSpec, ...] = (
    YcbSequenceSpec(
        seq_id="cracker_box_reorient",
        short_label="Cracker",
        ycb_object="003_cracker_box",
        mesh_relpath="cracker_ycbv_full_material/textured_simple.obj",
        prior_p2p_ref="point2pose-reference-mask/results.json",
        prior_p2p_gen="point2pose-single-seed",
        prior_fp_cad="cracker-foundationpose-cad-full/results.json",
    ),
    YcbSequenceSpec(
        seq_id="cracker_box_yalehand0",
        short_label="Cracker-hand",
        ycb_object="003_cracker_box",
        mesh_relpath="cracker_ycbv_full_material/textured_simple.obj",
        prior_p2p_ref=None,
        prior_p2p_gen=None,
        prior_fp_cad=None,
    ),
    YcbSequenceSpec(
        seq_id="mustard0",
        short_label="Mustard",
        ycb_object="006_mustard_bottle",
        mesh_relpath="mustard_ycbv_full_material/textured_simple.obj",
        prior_p2p_ref="mustard0-point2pose-reference-mask/results.json",
        prior_p2p_gen="mustard0-point2pose-single-seed",
        prior_fp_cad="mustard-foundationpose-cad-full/results.json",
    ),
    YcbSequenceSpec(
        seq_id="mustard_easy_00_02",
        short_label="Mustard-easy",
        ycb_object="006_mustard_bottle",
        mesh_relpath="mustard_ycbv_full_material/textured_simple.obj",
        prior_p2p_ref=None,
        prior_p2p_gen=None,
        prior_fp_cad=None,
    ),
    YcbSequenceSpec(
        seq_id="bleach0",
        short_label="Bleach",
        ycb_object="021_bleach_cleanser",
        mesh_relpath="bleach_ycbv_full_material/textured_simple.obj",
        prior_p2p_ref="bleach0-point2pose-reference-mask/results.json",
        prior_p2p_gen="bleach0-point2pose-single-seed",
        prior_fp_cad="bleach0-foundationpose-cad-full/results.json",
    ),
    YcbSequenceSpec(
        seq_id="bleach_hard_00_03_chaitanya",
        short_label="Bleach-hard",
        ycb_object="021_bleach_cleanser",
        mesh_relpath="bleach_ycbv_full_material/textured_simple.obj",
        prior_p2p_ref="bleach-hard-point2pose-reference-mask/results.json",
        prior_p2p_gen=None,
        prior_fp_cad=None,
    ),
    YcbSequenceSpec(
        seq_id="sugar_box1",
        short_label="Sugar-1",
        ycb_object="004_sugar_box",
        mesh_relpath="sugar_ycbv_full_material/textured_simple.obj",
        prior_p2p_ref=None,
        prior_p2p_gen=None,
        prior_fp_cad=None,
    ),
    YcbSequenceSpec(
        seq_id="sugar_box_yalehand0",
        short_label="Sugar-hand",
        ycb_object="004_sugar_box",
        mesh_relpath="sugar_ycbv_full_material/textured_simple.obj",
        prior_p2p_ref="sugar-box-point2pose-reference-mask/results.json",
        prior_p2p_gen="sugar-point2pose-generated-mask",
        prior_fp_cad="sugar-foundationpose-cad-full/results.json",
    ),
    YcbSequenceSpec(
        seq_id="tomato_soup_can_yalehand0",
        short_label="Tomato",
        ycb_object="005_tomato_soup_can",
        mesh_relpath="tomato_ycbv_full_material/textured_simple.obj",
        prior_p2p_ref="tomato-can-point2pose-reference-mask/results.json",
        prior_p2p_gen="tomato-point2pose-generated-mask",
        prior_fp_cad="tomato-foundationpose-cad-full/results.json",
    ),
)

YCB_SPEC_BY_ID: dict[str, YcbSequenceSpec] = {s.seq_id: s for s in YCBINEOAT_SPECS}
LAVAL_ACTION_IDS: tuple[str, ...] = ("clock-hard", "clock-full", "cookiejar")


def default_paths() -> dict[str, Path]:
    run_cache = Path(
        os.environ.get(
            "EVIDENCESCENE_RUN_CACHE_ROOT",
            str(Path.home() / ".cache" / "evidencescene-public-runs-20261121"),
        )
    )
    links = run_cache / "links"
    return {
        "run_cache": run_cache,
        "links": links,
        "ycb_root": Path(
            os.environ.get(
                "EVIDENCESCENE_YCB_ROOT",
                str(Path.home() / ".cache" / "evidencescene-public-ycbineoat"),
            )
        ),
        "laval_root": Path(
            os.environ.get(
                "EVIDENCESCENE_LAVAL_CACHE_ROOT",
                str(links / "laval"),
            )
        ),
        "prior_ycb_root": Path(
            os.environ.get(
                "EVIDENCESCENE_PRIOR_YCB_ROOT",
                str(links / "prior-ycb"),
            )
        ),
        "foundationpose_root": Path(
            os.environ.get(
                "EVIDENCESCENE_FOUNDATIONPOSE_ROOT",
                str(Path.home() / ".cache" / "nvlabs-foundationpose-a1b694"),
            )
        ),
        "point2pose_root": Path(
            os.environ.get(
                "EVIDENCESCENE_POINT2POSE_ROOT",
                str(
                    links / "point2pose-root"
                    if (links / "point2pose-root").exists()
                    else Path.home() / "src" / "point-to-pose"
                ),
            )
        ),
        "bundlesdf_source": Path(
            os.environ.get(
                "EVIDENCESCENE_BUNDLESDF_SOURCE",
                str(links / "composed-native" / "source"),
            )
        ),
        "bundlesdf_build": Path(
            os.environ.get(
                "EVIDENCESCENE_BUNDLESDF_BUILD",
                str(links / "composed-native" / "build"),
            )
        ),
        "bundlesdf_prefix": Path(
            os.environ.get(
                "EVIDENCESCENE_BUNDLESDF_PREFIX",
                str(links / "bundlesdf-prefix"),
            )
        ),
        "loftr_checkpoint": Path(
            os.environ.get(
                "EVIDENCESCENE_LOFTR_CHECKPOINT",
                str(
                    links / "bundlesdf-weights" / "BundleTrack" / "LoFTR" / "weights" / "outdoor_ds.ckpt"
                    if (links / "bundlesdf-weights" / "BundleTrack" / "LoFTR" / "weights" / "outdoor_ds.ckpt").is_file()
                    else links / "composed-native" / "source" / "BundleTrack/LoFTR/weights/outdoor_ds.ckpt"
                ),
            )
        ),
        "sam2_checkpoint": Path(
            os.environ.get(
                "EVIDENCESCENE_SAM2_CHECKPOINT",
                str(
                    Path.home()
                    / ".cache/huggingface/hub/models--facebook--sam2.1-hiera-large"
                    / "snapshots/665f8e2ad61cf5f53d65644ff27c8ee525124610"
                    / "sam2.1_hiera_large.pt"
                ),
            )
        ),
        "bootstapir_checkpoint": Path(
            os.environ.get(
                "EVIDENCESCENE_BOOTSTAPIR_CHECKPOINT",
                str(
                    (links / "point2pose-root" if (links / "point2pose-root").exists() else Path.home() / "src" / "point-to-pose")
                    / "checkpoints/tapir/causal_bootstapir_checkpoint.pt"
                ),
            )
        ),
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while buf := f.read(1024 * 1024):
            h.update(buf)
    return h.hexdigest()


def polar_so3(pose: Any) -> np.ndarray | None:
    if pose is None:
        return None
    try:
        arr = np.asarray(pose, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.shape != (4, 4) or not np.isfinite(arr).all():
        return None
    u, _, vt = np.linalg.svd(arr[:3, :3])
    if np.linalg.det(u @ vt) < 0.0:
        u[:, -1] *= -1.0
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = u @ vt
    out[:3, 3] = arr[:3, 3]
    return out


def laval_vicon_to_opencv(raw_pose: np.ndarray) -> np.ndarray:
    """Convert Laval OpenGL (-Z forward) Vicon pose to OpenCV (+Z forward) gauge."""
    mat = polar_so3(raw_pose)
    if mat is None:
        raise ValueError("Invalid Laval Vicon pose matrix")
    gl_to_cv = np.diag([1.0, -1.0, -1.0, 1.0])
    return gl_to_cv @ mat


def erode_mask_3x3(mask_bool: np.ndarray) -> np.ndarray:
    mask_u8 = (np.asarray(mask_bool) > 0).astype(np.uint8)
    eroded = cv2.erode(mask_u8, np.ones((3, 3), dtype=np.uint8))
    return eroded > 0


def load_ycb_sequence_stems(seq_dir: Path) -> list[str]:
    stems = sorted(p.stem for p in (seq_dir / "rgb").glob("*.png"))
    if not stems:
        raise FileNotFoundError(f"No RGB PNGs found under {seq_dir / 'rgb'}")
    for stem in stems:
        if not (seq_dir / "depth" / f"{stem}.png").is_file():
            raise FileNotFoundError(f"Missing depth frame {stem}.png under {seq_dir / 'depth'}")
        if not (seq_dir / "gt_mask" / f"{stem}.png").is_file():
            raise FileNotFoundError(f"Missing gt_mask frame {stem}.png under {seq_dir / 'gt_mask'}")
    return stems


def load_ycb_gt_poses(seq_dir: Path, stems: list[str]) -> list[np.ndarray]:
    pose_dir = seq_dir / "annotated_poses"
    if (pose_dir / f"{stems[0]}.txt").is_file():
        paths = [pose_dir / f"{s}.txt" for s in stems]
    else:
        paths = sorted(pose_dir.glob("*.txt"))
    if len(paths) != len(stems):
        raise ValueError(f"Pose file count {len(paths)} != frame count {len(stems)} in {seq_dir}")
    poses: list[np.ndarray] = []
    for p in paths:
        mat = polar_so3(np.loadtxt(p).reshape(4, 4))
        if mat is None:
            raise ValueError(f"Invalid GT pose in {p}")
        poses.append(mat)
    return poses


def load_obj_vertices(obj_path: Path) -> np.ndarray:
    verts = np.asarray(
        [[float(x) for x in line.split()[1:4]] for line in obj_path.open("r", encoding="utf-8") if line.startswith("v ")],
        dtype=np.float64,
    )
    if verts.ndim != 2 or verts.shape[1] != 3 or len(verts) == 0:
        raise ValueError(f"Invalid OBJ vertices in {obj_path}")
    return verts


def sample_mesh_surface_points(mesh_path: Path, *, num_samples: int = 100_000, seed: int = 17) -> np.ndarray:
    import trimesh

    mesh = trimesh.load(mesh_path.resolve(), force="mesh", process=False)
    np.random.seed(seed)
    pts, _ = trimesh.sample.sample_surface(mesh, num_samples, seed=seed)
    return np.asarray(pts, dtype=np.float64)


def load_laval_frames_and_gt(
    laval_action_dir: Path,
) -> tuple[list[dict[str, Any]], np.ndarray, list[np.ndarray], Path]:
    """Load registered 1080p Laval RGB-D frames, SAM 2 owner masks, intrinsics, Vicon GT, and model."""
    sample_dir = laval_action_dir / "sample"
    sam_dir = laval_action_dir / "sam-owner"
    model_path = laval_action_dir / "geometry.ply"

    sam = json.loads((sam_dir / "results.json").read_text(encoding="utf-8"))
    cam = json.loads((sample_dir / "camera.json").read_text(encoding="utf-8"))
    intrinsics = np.array(
        [[cam["focalX"], 0.0, cam["centerX"]], [0.0, cam["focalY"], cam["centerY"]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    raw_poses = np.load(sample_dir / "poses.npy").reshape(-1, 4, 4)
    records = sam["records"]
    if len(records) != len(raw_poses):
        raise ValueError(f"SAM record count {len(records)} != pose count {len(raw_poses)} in {laval_action_dir}")

    gt_poses = [laval_vicon_to_opencv(p) for p in raw_poses]
    frames: list[dict[str, Any]] = []
    for idx, rec in enumerate(records):
        rgb_path = sample_dir / f"{idx}.png"
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(rgb_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rec["status"] == "available" and rec.get("mask_artifact"):
            npz_path = sam_dir / rec["mask_artifact"]
            with np.load(npz_path, allow_pickle=False) as z:
                depth_mm = np.asarray(z["depth_mm"], dtype=np.uint16)
                mask_bool = np.asarray(z["mask"], dtype=bool)
        else:
            depth_path = sample_dir / f"{idx}d.png"
            depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.uint16)
            mask_bool = np.zeros(bgr.shape[:2], dtype=bool)
        frames.append(
            {
                "index": idx,
                "bgr": bgr,
                "rgb": rgb,
                "depth_mm": depth_mm,
                "depth_m": depth_mm.astype(np.float32) / 1000.0,
                "mask": mask_bool,
                "status": rec["status"],
            }
        )
    return frames, intrinsics, gt_poses, model_path


def p50_p95(samples_ms: list[float], *, warmup: int = 5) -> dict[str, float]:
    warm = samples_ms[warmup:] if len(samples_ms) > warmup else samples_ms
    if not warm:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0, "samples": 0}
    arr = np.sort(np.asarray(warm, dtype=np.float64))
    return {
        "p50_ms": round(float(arr[len(arr) // 2]), 3),
        "p95_ms": round(float(arr[min(len(arr) - 1, int(0.95 * len(arr)))]), 3),
        "mean_ms": round(float(np.mean(arr)), 3),
        "samples": int(len(warm)),
        "warmup": int(warmup),
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
