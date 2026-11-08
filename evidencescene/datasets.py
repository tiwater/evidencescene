"""Standard RGB-D dataset readers and deterministic synthetic manipulation sequences.

Allows external research teams to feed standard RGB-D + segmentation directories
(YCBInEOAT, BOP, Open3D/TUM layouts) directly into EvidenceScene without custom
internal manifest files, and provides a zero-dependency synthetic manipulation
sequence generator for reproducible end-to-end benchmarking and tutorials.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from evidencescene.primitives.bundletrack_live_input import RegisteredBundleTrackInput
from evidencescene.primitives.image_mask import encode_mask_rle


@dataclass(frozen=True)
class SyntheticFrameBundle:
    """One frame of a synthetic RGB-D manipulation sequence with ground truth."""

    frame_index: int
    timestamp_ns: int
    registered_input: RegisteredBundleTrackInput
    mask_rle: dict[str, list[int]]
    true_camera_from_object: np.ndarray
    true_world_from_object: np.ndarray
    noisy_estimated_pose: np.ndarray | None
    is_occluded: bool
    is_rank_deficient_frame: bool
    observed_landmark_ids: tuple[int, ...]
    camera_id: str


@dataclass(frozen=True)
class SyntheticManipulationScenario:
    """Self-contained multi-frame manipulation scenario for end-to-end verification."""

    object_id: str
    intrinsics: np.ndarray
    world_from_camera: np.ndarray
    model_points_object_m: np.ndarray
    landmarks_object_m: dict[int, np.ndarray]
    target_cup_rim_world_m: np.ndarray
    spout_point_object_m: np.ndarray
    spout_direction_object: np.ndarray
    frames: tuple[SyntheticFrameBundle, ...]


class FolderRgbdSequenceReader:
    """Read standard RGB-D + mask sequences (YCBInEOAT / BOP / custom folders).

    Supported directory layouts under ``sequence_root``:
    - Color images: ``rgb/*.png`` or ``color/*.png`` or ``rgb/*.jpg``
    - Depth maps:   ``depth/*.png`` (uint16 millimeters or scaled units)
    - Masks:        ``gt_mask/*.png``, ``masks/*.png``, or ``mask/*.png``
    - Intrinsics:   ``cam_K.txt`` (3x3 whitespace-separated) or ``camera.json``
    - GT Poses:     optional ``annotated_poses/*.txt`` or ``gt_poses.npy``
    """

    def __init__(
        self,
        sequence_root: str | Path,
        *,
        object_id: str = "object0",
        camera_id: str = "cam0",
        session_id: str | None = None,
        max_frames: int | None = None,
    ) -> None:
        self.root = Path(sequence_root).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Sequence directory does not exist: {self.root}")
        self.object_id = str(object_id)
        self.camera_id = str(camera_id)
        self.session_id = str(session_id or self.root.name)

        self.rgb_dir = self._find_subdir(("rgb", "color"))
        self.depth_dir = self._find_subdir(("depth",))
        self.mask_dir = self._find_subdir(("gt_mask", "masks", "mask"))
        self.intrinsics = self._load_intrinsics()

        rgb_files = sorted(
            [*self.rgb_dir.glob("*.png"), *self.rgb_dir.glob("*.jpg")]
        )
        if not rgb_files:
            raise ValueError(f"No RGB images found in {self.rgb_dir}")
        if max_frames is not None:
            if max_frames <= 0:
                raise ValueError("max_frames must be positive")
            rgb_files = rgb_files[:max_frames]
        self.stems = tuple(p.stem for p in rgb_files)
        self._rgb_paths = tuple(rgb_files)

    def _find_subdir(self, candidates: Sequence[str]) -> Path:
        for name in candidates:
            candidate = self.root / name
            if candidate.is_dir():
                return candidate
        raise FileNotFoundError(
            f"None of the expected subdirectories {candidates} found under {self.root}"
        )

    def _load_intrinsics(self) -> np.ndarray:
        cam_k = self.root / "cam_K.txt"
        if cam_k.is_file():
            mat = np.asarray(np.loadtxt(cam_k), dtype=np.float64).reshape(3, 3)
            return mat
        cam_json = self.root / "camera.json"
        if cam_json.is_file():
            data = json.loads(cam_json.read_text(encoding="utf-8"))
            if "K" in data:
                return np.asarray(data["K"], dtype=np.float64).reshape(3, 3)
            fx, fy = float(data["fx"]), float(data["fy"])
            cx, cy = float(data["cx"]), float(data["cy"])
            return np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
            )
        raise FileNotFoundError(
            f"Neither cam_K.txt nor camera.json found under {self.root}"
        )

    def __len__(self) -> int:
        return len(self.stems)

    def __iter__(self) -> Iterator[RegisteredBundleTrackInput]:
        import cv2

        for idx, (stem, rgb_path) in enumerate(
            zip(self.stems, self._rgb_paths, strict=True)
        ):
            depth_path = self.depth_dir / f"{stem}.png"
            mask_path = self.mask_dir / f"{stem}.png"
            if not depth_path.is_file():
                raise FileNotFoundError(f"Missing depth file: {depth_path}")
            if not mask_path.is_file():
                raise FileNotFoundError(f"Missing mask file: {mask_path}")

            bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Failed to decode RGB image: {rgb_path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise ValueError(f"Failed to decode depth image: {depth_path}")
            depth_mm = np.asarray(depth, dtype=np.uint16)
            raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if raw_mask is None:
                raise ValueError(f"Failed to decode mask image: {mask_path}")
            if raw_mask.ndim == 3:
                raw_mask = raw_mask[:, :, 0]
            mask = np.asarray(raw_mask > 0, dtype=bool)

            if stem.isdecimal() and int(stem) > 0:
                ts_ns = int(stem)
                if ts_ns < 1_000_000_000:
                    ts_ns = 1_000_000_000 + idx * 33_333_333
            else:
                ts_ns = 1_000_000_000 + idx * 33_333_333

            src_hash = hashlib.sha256(
                f"{self.session_id}:{stem}:{idx}".encode("ascii")
            ).hexdigest()

            yield RegisteredBundleTrackInput(
                stream_key=(
                    "dataset",
                    self.session_id,
                    self.camera_id,
                    self.object_id,
                    "v1",
                    "rgbd",
                    self.object_id,
                ),
                color_timestamp_ns=ts_ns,
                depth_timestamp_ns=ts_ns,
                color_rgb=rgb,
                depth_mm=depth_mm,
                mask=mask,
                intrinsics=self.intrinsics,
                source_evidence_hash=src_hash,
            )


def generate_synthetic_manipulation_sequence(
    *,
    num_frames: int = 24,
    width: int = 160,
    height: int = 120,
    seed: int = 42,
) -> SyntheticManipulationScenario:
    """Generate a deterministic RGB-D manipulation sequence with ground-truth geometry.

    Includes:
    - Dense 3D object surface points (a rectangular beverage carton)
    - 8 tracked 3D landmarks on the carton
    - Smooth SE(3) pouring trajectory approaching a circular target cup aperture
    - A simulated rank-deficient collinear observation frame (frame 6)
    - A simulated full occlusion gap (frames 10-11) followed by re-acquisition (frame 12)
    - Simulated early pose drift on frames 4-5 that is corrected by a pose-graph revision
    """
    if num_frames < 15:
        raise ValueError("num_frames must be at least 15 to exercise all scenario phases")
    rng = np.random.default_rng(seed)

    intrinsics = np.array(
        [
            [200.0, 0.0, width / 2.0],
            [0.0, 200.0, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # Camera looks down at the table from z = 0.65m
    world_from_camera = np.eye(4, dtype=np.float64)
    world_from_camera[:3, :3] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    world_from_camera[:3, 3] = [0.25, 0.0, 0.65]
    camera_from_world = np.linalg.inv(world_from_camera)

    # Dense carton model points in object frame (box of 0.06 x 0.06 x 0.12 m)
    gx, gy, gz = np.meshgrid(
        np.linspace(-0.03, 0.03, 9),
        np.linspace(-0.03, 0.03, 9),
        np.linspace(-0.06, 0.06, 13),
    )
    all_box = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    on_surface = (
        np.isclose(np.abs(all_box[:, 0]), 0.03)
        | np.isclose(np.abs(all_box[:, 1]), 0.03)
        | np.isclose(np.abs(all_box[:, 2]), 0.06)
    )
    model_points_object_m = all_box[on_surface].astype(np.float64)

    # 8 3D landmarks on the object; landmarks 0, 1, 2 are intentionally collinear
    # along the top edge x = 0.03, z = 0.06 so observing only {0, 1, 2} yields rank r_i = 5 < 6!
    landmarks_object_m = {
        0: np.array([0.03, -0.025, 0.06], dtype=np.float64),
        1: np.array([0.03, 0.000, 0.06], dtype=np.float64),
        2: np.array([0.03, 0.025, 0.06], dtype=np.float64),
        3: np.array([-0.03, -0.025, 0.06], dtype=np.float64),
        4: np.array([-0.03, 0.025, 0.06], dtype=np.float64),
        5: np.array([0.03, -0.025, -0.05], dtype=np.float64),
        6: np.array([0.03, 0.025, -0.05], dtype=np.float64),
        7: np.array([-0.03, 0.000, -0.05], dtype=np.float64),
    }

    # Target cup circular aperture on the table at (0.30, 0.00, 0.05), radius 0.045 m
    angles = np.linspace(-math.pi, math.pi, 120, endpoint=False)
    target_cup_rim_world_m = np.column_stack(
        [
            0.30 + 0.045 * np.cos(angles) + rng.normal(0.0, 0.0003, len(angles)),
            0.00 + 0.045 * np.sin(angles) + rng.normal(0.0, 0.0003, len(angles)),
            np.full(len(angles), 0.05) + rng.normal(0.0, 0.0002, len(angles)),
        ]
    )

    spout_point_object_m = np.array([0.03, 0.0, 0.06], dtype=np.float64)
    spout_direction_object = np.array([0.25, 0.0, -0.9682458365518543], dtype=np.float64)

    frames: list[SyntheticFrameBundle] = []
    cameras = ("chest", "head", "wrist")

    for idx in range(num_frames):
        alpha = idx / float(max(1, num_frames - 1))
        timestamp_ns = 1_000_000_000 + idx * 33_333_333

        # Smooth trajectory moving the carton spout to pouring position above the cup
        T_wo = np.eye(4, dtype=np.float64)
        yaw_deg = 8.0 * math.sin(alpha * math.pi)
        pitch_deg = 12.0 * alpha
        T_wo[:3, :3] = Rotation.from_euler(
            "zyx", [yaw_deg, pitch_deg, 0.0], degrees=True
        ).as_matrix()
        T_wo[:3, 3] = [
            0.22 + 0.035 * alpha,
            -0.02 * (1.0 - alpha),
            0.11 - 0.02 * alpha,
        ]
        T_co = camera_from_world @ T_wo

        is_occluded = idx in (10, 11)
        is_rank_deficient = idx == 6
        observed_lms = (0, 1, 2) if is_rank_deficient else (0, 1, 2, 3, 4, 5, 6, 7)

        # Render synthetic depth and segmentation mask in camera frame
        depth_mm = np.zeros((height, width), dtype=np.uint16)
        mask = np.zeros((height, width), dtype=bool)
        rgb = np.full((height, width, 3), 30, dtype=np.uint8)

        if not is_occluded:
            pts_c = model_points_object_m @ T_co[:3, :3].T + T_co[:3, 3]
            valid_z = pts_c[:, 2] > 0.1
            pts_c = pts_c[valid_z]
            uv = (pts_c[:, :2] / pts_c[:, 2:3]) @ intrinsics[:2, :2].T + intrinsics[:2, 2]
            cols = np.round(uv[:, 0]).astype(np.int64)
            rows = np.round(uv[:, 1]).astype(np.int64)
            in_img = (cols >= 1) & (cols < width - 1) & (rows >= 1) & (rows < height - 1)
            for r, c, z_m in zip(rows[in_img], cols[in_img], pts_c[in_img, 2], strict=False):
                z_mm = int(round(float(z_m) * 1000.0))
                # Splat 2x2 neighborhood for dense depth coverage
                for dr in (0, 1):
                    for dc in (0, 1):
                        rr, cc = r + dr, c + dc
                        if depth_mm[rr, cc] == 0 or z_mm < depth_mm[rr, cc]:
                            depth_mm[rr, cc] = z_mm
                        mask[rr, cc] = True
                        rgb[rr, cc] = (60, 140, 220)

        if is_occluded:
            noisy_pose = None
        else:
            noisy_pose = T_co.copy()
            # Frame 4 and 5 have transient initial drift (+18 mm) before pose-graph revision
            if idx in (4, 5):
                noisy_pose[:3, 3] += np.array([0.018, -0.012, 0.008], dtype=np.float64)
            elif is_rank_deficient:
                # Collinear degeneracy induces large rotational/translational error if un-gated
                noisy_pose[:3, :3] = (
                    noisy_pose[:3, :3]
                    @ Rotation.from_euler("x", 18.0, degrees=True).as_matrix()
                )
                noisy_pose[:3, 3] += np.array([0.035, 0.025, -0.020], dtype=np.float64)
            else:
                noisy_pose[:3, 3] += rng.normal(0.0, 0.0008, size=3)

        cam_id = cameras[idx % len(cameras)]
        reg_input = RegisteredBundleTrackInput(
            stream_key=(
                "synthetic",
                "pouring_demo",
                cam_id,
                "carton",
                "v1",
                "rgbd",
                "carton",
            ),
            color_timestamp_ns=timestamp_ns,
            depth_timestamp_ns=timestamp_ns,
            color_rgb=rgb,
            depth_mm=depth_mm,
            mask=mask,
            intrinsics=intrinsics,
            source_evidence_hash=hashlib.sha256(
                f"synthetic:{idx}:{timestamp_ns}".encode("ascii")
            ).hexdigest(),
        )

        frames.append(
            SyntheticFrameBundle(
                frame_index=idx,
                timestamp_ns=timestamp_ns,
                registered_input=reg_input,
                mask_rle=encode_mask_rle(mask),
                true_camera_from_object=T_co,
                true_world_from_object=T_wo,
                noisy_estimated_pose=noisy_pose,
                is_occluded=is_occluded,
                is_rank_deficient_frame=is_rank_deficient,
                observed_landmark_ids=observed_lms,
                camera_id=cam_id,
            )
        )

    return SyntheticManipulationScenario(
        object_id="carton",
        intrinsics=intrinsics,
        world_from_camera=world_from_camera,
        model_points_object_m=model_points_object_m,
        landmarks_object_m=landmarks_object_m,
        target_cup_rim_world_m=target_cup_rim_world_m,
        spout_point_object_m=spout_point_object_m,
        spout_direction_object=spout_direction_object,
        frames=tuple(frames),
    )
