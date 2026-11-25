"""Causal SE(3) Lie-algebra trajectory filtering, C1 join blending, and multi-horizon prediction.

Designed for L2 teleoperation assistance:
- Strictly causal updates (no future frames or acausal smoothing).
- C1-continuous cubic Hermite join blending across re-acquisition gaps
  (zero position jump at join, bounded velocity jump).
- Strictly monotonic 95% uncertainty intervals across prediction horizons
  (100 ms, 250 ms, 500 ms).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from evidencescene._accel import NATIVE

CHI2_3DOF_95_SCALE = math.sqrt(7.814727903251179)  # sqrt(chi2.ppf(0.95, df=3))


def _so3_exp(rotvec: np.ndarray) -> np.ndarray:
    """Analytical Rodrigues exponential map so(3) -> SO(3)."""
    vx = float(rotvec[0])
    vy = float(rotvec[1])
    vz = float(rotvec[2])
    theta2 = vx * vx + vy * vy + vz * vz
    if theta2 < 1e-8:
        a = 1.0 - theta2 * (1.0 / 6.0) + theta2 * theta2 * (1.0 / 120.0)
        b = 0.5 - theta2 * (1.0 / 24.0) + theta2 * theta2 * (1.0 / 720.0)
    else:
        theta = math.sqrt(theta2)
        a = math.sin(theta) / theta
        b = (1.0 - math.cos(theta)) / theta2
    wx2 = vx * vx
    wy2 = vy * vy
    wz2 = vz * vz
    wxy = vx * vy
    wxz = vx * vz
    wyz = vy * vz
    return np.array(
        [
            [1.0 - b * (wy2 + wz2), b * wxy - a * vz, b * wxz + a * vy],
            [b * wxy + a * vz, 1.0 - b * (wx2 + wz2), b * wyz - a * vx],
            [b * wxz - a * vy, b * wyz + a * vx, 1.0 - b * (wx2 + wy2)],
        ],
        dtype=np.float64,
    )


def _so3_log(r: np.ndarray) -> np.ndarray:
    """Analytical Rodrigues logarithm map SO(3) -> so(3)."""
    cos_theta = 0.5 * (float(r[0, 0]) + float(r[1, 1]) + float(r[2, 2]) - 1.0)
    if cos_theta < -0.9999:
        return Rotation.from_matrix(r).as_rotvec()
    rx = float(r[2, 1]) - float(r[1, 2])
    ry = float(r[0, 2]) - float(r[2, 0])
    rz = float(r[1, 0]) - float(r[0, 1])
    if cos_theta >= 1.0 - 1e-8:
        sin2 = 0.25 * (rx * rx + ry * ry + rz * rz)
        factor = 0.5 + sin2 * (1.0 / 12.0) + sin2 * sin2 * (3.0 / 80.0)
    else:
        cos_c = -1.0 if cos_theta < -1.0 else (1.0 if cos_theta > 1.0 else cos_theta)
        theta = math.acos(cos_c)
        sin_theta = math.sqrt(max(0.0, 1.0 - cos_c * cos_c))
        factor = theta / (2.0 * sin_theta)
    return np.array([factor * rx, factor * ry, factor * rz], dtype=np.float64)


def _validate_se3(pose: np.ndarray, name: str = "pose") -> np.ndarray:
    matrix = np.asarray(pose, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must be a finite right-handed 4x4 SE(3) matrix")
    if NATIVE is not None:
        mat_c = np.ascontiguousarray(matrix, dtype=np.float64)
        if not NATIVE.is_valid_se3(mat_c, 1e-8, 1e-5, 1e-5):
            raise ValueError(f"{name} must be a finite right-handed 4x4 SE(3) matrix")
        out = mat_c.copy()
        out.flags.writeable = False
        return out
    if (
        not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8)
        or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
        or not math.isclose(float(np.linalg.det(matrix[:3, :3])), 1.0, abs_tol=1e-5)
    ):
        raise ValueError(f"{name} must be a finite right-handed 4x4 SE(3) matrix")
    out = matrix.copy()
    out.flags.writeable = False
    return out


@dataclass(frozen=True)
class C1JoinState:
    """Active C1-continuous re-acquisition transition state."""

    join_start_timestamp_ns: int
    join_duration_s: float
    position_offset_m: np.ndarray
    velocity_offset_mps: np.ndarray
    rotation_offset_rotvec_rad: np.ndarray
    angular_velocity_offset_rad_s: np.ndarray
    position_join_jump_m: float
    velocity_join_jump_mps: float


@dataclass(frozen=True)
class TrajectoryHorizonPrediction:
    """Predicted SE(3) pose and 95% confidence envelope at one future horizon."""

    horizon_ms: float
    target_timestamp_ns: int
    predicted_pose: np.ndarray
    linear_velocity_mps: np.ndarray
    angular_velocity_rad_s: np.ndarray
    position_std_m: np.ndarray
    rotation_std_deg: np.ndarray
    interval_half_width_95_m: float
    rotation_half_width_95_deg: float
    covariance_6x6: np.ndarray
    c1_join_blended: bool


@dataclass(frozen=True)
class MultiHorizonTrajectoryReport:
    """Multi-horizon causal trajectory bundle with C1 join and monotonicity diagnostics."""

    source_timestamp_ns: int
    horizons: tuple[TrajectoryHorizonPrediction, ...]
    interval_width_monotonic_violation_count: int
    position_join_jump_m: float
    velocity_join_jump_mps: float
    translation_speed_mps: float
    angular_speed_deg_s: float
    c1_join_active: bool
    publication_authorized: bool = False


def _hermite_decay_factors(elapsed_s: float, duration_s: float) -> tuple[float, float, float]:
    """Return (h00, h10, dh00_dt) for decaying an initial offset to zero over [0, duration_s].

    At elapsed_s == 0:
      offset(0) = 1 * delta_p + 0 * delta_v = delta_p
      d/dt offset(0) = 0 * delta_p + 1 * delta_v = delta_v
    At elapsed_s >= duration_s:
      offset(T) = 0, d/dt offset(T) = 0
    """
    if duration_s <= 1e-9 or elapsed_s >= duration_s:
        return 0.0, 0.0, 0.0
    if elapsed_s <= 0.0:
        return 1.0, 0.0, 0.0
    u = elapsed_s / duration_s
    u2 = u * u
    u3 = u2 * u
    h00 = 1.0 - 3.0 * u2 + 2.0 * u3
    h10 = duration_s * (u - 2.0 * u2 + u3)
    dh00_dt = (-6.0 * u + 6.0 * u2) / duration_s
    return h00, h10, dh00_dt


def predict_multi_horizon_se3_trajectory(
    *,
    current_pose: np.ndarray,
    current_timestamp_ns: int,
    linear_velocity_mps: np.ndarray,
    angular_velocity_rad_s: np.ndarray,
    base_position_std_m: np.ndarray | Sequence[float] = (0.002, 0.002, 0.002),
    base_velocity_std_mps: np.ndarray | Sequence[float] = (0.010, 0.010, 0.010),
    base_rotation_std_deg: np.ndarray | Sequence[float] = (0.5, 0.5, 0.5),
    base_angular_velocity_std_rad_s: np.ndarray | Sequence[float] = (0.03, 0.03, 0.03),
    linear_acceleration_noise_mps2: float = 0.15,
    angular_acceleration_noise_rad_s2: float = 0.25,
    horizons_ms: Sequence[float] = (100.0, 250.0, 500.0),
    maximum_translation_speed_mps: float = 1.2,
    maximum_angular_speed_deg_s: float = 180.0,
    c1_join_state: C1JoinState | None = None,
) -> MultiHorizonTrajectoryReport:
    """Extrapolate a causal SE(3) state across multiple horizons with monotonic 95% bounds."""
    pose = _validate_se3(current_pose, "current_pose")
    if type(current_timestamp_ns) is not int or current_timestamp_ns <= 0:
        raise ValueError("current_timestamp_ns must be a positive integer")
    vel = np.asarray(linear_velocity_mps, dtype=np.float64)
    omega = np.asarray(angular_velocity_rad_s, dtype=np.float64)
    if vel.shape != (3,) or not np.isfinite(vel).all():
        raise ValueError("linear_velocity_mps must be a finite 3-vector")
    if omega.shape != (3,) or not np.isfinite(omega).all():
        raise ValueError("angular_velocity_rad_s must be a finite 3-vector")
    pos_std0 = np.asarray(base_position_std_m, dtype=np.float64)
    vel_std0 = np.asarray(base_velocity_std_mps, dtype=np.float64)
    rot_std0_deg = np.asarray(base_rotation_std_deg, dtype=np.float64)
    if (
        pos_std0.shape != (3,)
        or vel_std0.shape != (3,)
        or rot_std0_deg.shape != (3,)
        or not np.isfinite(pos_std0).all()
        or not np.isfinite(vel_std0).all()
        or not np.isfinite(rot_std0_deg).all()
        or np.any(pos_std0 <= 0.0)
        or np.any(vel_std0 <= 0.0)
        or np.any(rot_std0_deg <= 0.0)
    ):
        raise ValueError("base uncertainties must be positive finite 3-vectors")
    if (
        not math.isfinite(linear_acceleration_noise_mps2)
        or linear_acceleration_noise_mps2 <= 0.0
        or not math.isfinite(angular_acceleration_noise_rad_s2)
        or angular_acceleration_noise_rad_s2 <= 0.0
    ):
        raise ValueError("acceleration noise parameters must be finite and positive")

    horizon_list = tuple(float(h) for h in horizons_ms)
    if (
        not horizon_list
        or any(not math.isfinite(h) or h < 0.0 for h in horizon_list)
        or any(h1 >= h2 for h1, h2 in zip(horizon_list, horizon_list[1:]))
    ):
        raise ValueError("horizons_ms must be a nonempty strictly increasing nonnegative sequence")

    speed = float(np.linalg.norm(vel))
    if speed > maximum_translation_speed_mps:
        vel = vel * (maximum_translation_speed_mps / speed)
        speed = float(maximum_translation_speed_mps)

    ang_speed_rad = float(np.linalg.norm(omega))
    max_ang_rad = math.radians(maximum_angular_speed_deg_s)
    if ang_speed_rad > max_ang_rad:
        omega = omega * (max_ang_rad / ang_speed_rad)
        ang_speed_rad = max_ang_rad

    rot_std0_rad = np.radians(rot_std0_deg)
    predictions: list[TrajectoryHorizonPrediction] = []
    previous_width: float | None = None
    violations = 0

    for h_ms in horizon_list:
        dt_s = h_ms / 1000.0
        target_ns = current_timestamp_ns + int(round(h_ms * 1_000_000.0))

        pred_pos = pose[:3, 3] + vel * dt_s
        pred_vel = vel.copy()
        pred_rot = pose[:3, :3] @ _so3_exp(omega * dt_s)
        pred_omega = omega.copy()
        join_blended = False

        if c1_join_state is not None:
            elapsed_s = (target_ns - c1_join_state.join_start_timestamp_ns) / 1e9
            if 0.0 <= elapsed_s < c1_join_state.join_duration_s:
                h00, h10, dh00_dt = _hermite_decay_factors(
                    elapsed_s, c1_join_state.join_duration_s
                )
                u = elapsed_s / c1_join_state.join_duration_s
                dh10_dt = 1.0 - 4.0 * u + 3.0 * u * u
                pred_pos = (
                    pred_pos
                    + h00 * c1_join_state.position_offset_m
                    + h10 * c1_join_state.velocity_offset_mps
                )
                pred_vel = (
                    pred_vel
                    + dh00_dt * c1_join_state.position_offset_m
                    + dh10_dt * c1_join_state.velocity_offset_mps
                )
                rot_decay = h00 * c1_join_state.rotation_offset_rotvec_rad
                pred_rot = pred_rot @ _so3_exp(rot_decay)
                join_blended = True

        pred_T = np.eye(4, dtype=np.float64)
        pred_T[:3, :3] = pred_rot
        pred_T[:3, 3] = pred_pos
        pred_T.flags.writeable = False

        # Strictly increasing continuous-time variance growth in translation and rotation
        pos_var = (
            np.square(pos_std0)
            + np.square(vel_std0) * (dt_s * dt_s)
            + (1.0 / 3.0) * (linear_acceleration_noise_mps2 ** 2) * (dt_s ** 3)
            + 1e-8 * dt_s
        )
        rot_vel_std0 = np.asarray(base_angular_velocity_std_rad_s, dtype=np.float64)
        rot_var_rad = (
            np.square(rot_std0_rad)
            + np.square(rot_vel_std0) * (dt_s * dt_s)
            + (1.0 / 3.0) * (angular_acceleration_noise_rad_s2 ** 2) * (dt_s ** 3)
            + 1e-8 * dt_s
        )
        pos_std = np.sqrt(pos_var)
        rot_std_deg = np.degrees(np.sqrt(rot_var_rad))

        # 95% confidence radial half-width in 3D Euclidean space
        width_95_m = float(CHI2_3DOF_95_SCALE * math.sqrt(float(np.mean(pos_var))))
        rot_width_95_deg = float(
            CHI2_3DOF_95_SCALE * math.degrees(math.sqrt(float(np.mean(rot_var_rad))))
        )
        if previous_width is not None and width_95_m <= previous_width:
            violations += 1
        previous_width = width_95_m

        cov6 = np.zeros((6, 6), dtype=np.float64)
        cov6[0, 0] = rot_var_rad[0]
        cov6[1, 1] = rot_var_rad[1]
        cov6[2, 2] = rot_var_rad[2]
        cov6[3, 3] = pos_var[0]
        cov6[4, 4] = pos_var[1]
        cov6[5, 5] = pos_var[2]
        cov6.flags.writeable = False
        pos_std.flags.writeable = False
        rot_std_deg.flags.writeable = False
        pred_vel.flags.writeable = False
        pred_omega.flags.writeable = False

        predictions.append(
            TrajectoryHorizonPrediction(
                horizon_ms=h_ms,
                target_timestamp_ns=target_ns,
                predicted_pose=pred_T,
                linear_velocity_mps=pred_vel,
                angular_velocity_rad_s=pred_omega,
                position_std_m=pos_std,
                rotation_std_deg=rot_std_deg,
                interval_half_width_95_m=width_95_m,
                rotation_half_width_95_deg=rot_width_95_deg,
                covariance_6x6=cov6,
                c1_join_blended=join_blended,
            )
        )

    pos_jump = c1_join_state.position_join_jump_m if c1_join_state is not None else 0.0
    vel_jump = c1_join_state.velocity_join_jump_mps if c1_join_state is not None else 0.0
    return MultiHorizonTrajectoryReport(
        source_timestamp_ns=current_timestamp_ns,
        horizons=tuple(predictions),
        interval_width_monotonic_violation_count=violations,
        position_join_jump_m=pos_jump,
        velocity_join_jump_mps=vel_jump,
        translation_speed_mps=speed,
        angular_speed_deg_s=math.degrees(ang_speed_rad),
        c1_join_active=any(p.c1_join_blended for p in predictions),
        publication_authorized=False,
    )


class CausalSe3TrajectoryPredictor:
    """Stateful causal SE(3) trajectory filter with C1 re-acquisition join blending."""

    def __init__(
        self,
        *,
        velocity_ema_alpha: float = 0.45,
        reacquisition_gap_threshold_s: float = 0.12,
        c1_join_duration_s: float = 0.20,
        maximum_translation_speed_mps: float = 1.2,
        maximum_angular_speed_deg_s: float = 180.0,
        maximum_join_velocity_jump_mps: float = 0.25,
        horizons_ms: Sequence[float] = (100.0, 250.0, 500.0),
    ) -> None:
        if not 0.0 < velocity_ema_alpha <= 1.0:
            raise ValueError("velocity_ema_alpha must be in (0, 1]")
        if reacquisition_gap_threshold_s <= 0.0 or c1_join_duration_s <= 0.0:
            raise ValueError("gap threshold and C1 join duration must be positive")
        self.velocity_ema_alpha = float(velocity_ema_alpha)
        self.reacquisition_gap_threshold_s = float(reacquisition_gap_threshold_s)
        self.c1_join_duration_s = float(c1_join_duration_s)
        self.maximum_translation_speed_mps = float(maximum_translation_speed_mps)
        self.maximum_angular_speed_deg_s = float(maximum_angular_speed_deg_s)
        self.maximum_join_velocity_jump_mps = float(maximum_join_velocity_jump_mps)
        self.horizons_ms = tuple(float(h) for h in horizons_ms)

        self._last_pose: np.ndarray | None = None
        self._last_timestamp_ns: int | None = None
        self._linear_velocity_mps = np.zeros(3, dtype=np.float64)
        self._angular_velocity_rad_s = np.zeros(3, dtype=np.float64)
        self._c1_join_state: C1JoinState | None = None

    def observe(
        self,
        *,
        pose: np.ndarray,
        timestamp_ns: int,
        base_position_std_m: Sequence[float] = (0.003, 0.003, 0.003),
        base_rotation_std_deg: Sequence[float] = (0.8, 0.8, 0.8),
        is_reacquisition: bool = False,
    ) -> MultiHorizonTrajectoryReport:
        """Advance the causal trajectory filter with one current SE(3) observation."""
        T_obs = _validate_se3(pose, "pose")
        if type(timestamp_ns) is not int or timestamp_ns <= 0:
            raise ValueError("timestamp_ns must be a positive integer")
        if self._last_timestamp_ns is not None and timestamp_ns <= self._last_timestamp_ns:
            raise ValueError("timestamp_ns must increase strictly")

        if self._last_pose is None or self._last_timestamp_ns is None:
            self._last_pose = T_obs
            self._last_timestamp_ns = timestamp_ns
            return predict_multi_horizon_se3_trajectory(
                current_pose=T_obs,
                current_timestamp_ns=timestamp_ns,
                linear_velocity_mps=self._linear_velocity_mps,
                angular_velocity_rad_s=self._angular_velocity_rad_s,
                base_position_std_m=base_position_std_m,
                base_rotation_std_deg=base_rotation_std_deg,
                horizons_ms=self.horizons_ms,
                maximum_translation_speed_mps=self.maximum_translation_speed_mps,
                maximum_angular_speed_deg_s=self.maximum_angular_speed_deg_s,
                c1_join_state=None,
            )

        dt_s = (timestamp_ns - self._last_timestamp_ns) / 1e9
        raw_vel = (T_obs[:3, 3] - self._last_pose[:3, 3]) / dt_s
        raw_speed = float(np.linalg.norm(raw_vel))
        if raw_speed > self.maximum_translation_speed_mps:
            raw_vel = raw_vel * (self.maximum_translation_speed_mps / raw_speed)

        raw_omega = _so3_log(self._last_pose[:3, :3].T @ T_obs[:3, :3]) / dt_s
        raw_ang_speed = float(np.linalg.norm(raw_omega))
        max_ang_rad = math.radians(self.maximum_angular_speed_deg_s)
        if raw_ang_speed > max_ang_rad:
            raw_omega = raw_omega * (max_ang_rad / raw_ang_speed)

        gap_triggered = is_reacquisition or (dt_s > self.reacquisition_gap_threshold_s)
        if gap_triggered:
            # Extrapolate where the pre-join trajectory was at timestamp_ns
            prior_extrap_pos = self._last_pose[:3, 3] + self._linear_velocity_mps * dt_s
            prior_extrap_rot = (
                self._last_pose[:3, :3]
                @ _so3_exp(self._angular_velocity_rad_s * dt_s)
            )
            # Dampen velocity update across a gap and bound the velocity jump
            new_vel = (1.0 - 0.5 * self.velocity_ema_alpha) * self._linear_velocity_mps + (
                0.5 * self.velocity_ema_alpha
            ) * raw_vel
            vel_diff = new_vel - self._linear_velocity_mps
            vel_jump = float(np.linalg.norm(vel_diff))
            if vel_jump > self.maximum_join_velocity_jump_mps:
                new_vel = self._linear_velocity_mps + vel_diff * (
                    self.maximum_join_velocity_jump_mps / vel_jump
                )
                vel_jump = self.maximum_join_velocity_jump_mps

            new_omega = (1.0 - 0.5 * self.velocity_ema_alpha) * self._angular_velocity_rad_s + (
                0.5 * self.velocity_ema_alpha
            ) * raw_omega

            # C1 Hermite offset so blended(t_join) == prior_extrap_pos (zero position jump)
            pos_offset = prior_extrap_pos - T_obs[:3, 3]
            vel_offset = self._linear_velocity_mps - new_vel
            rot_offset = _so3_log(T_obs[:3, :3].T @ prior_extrap_rot)
            omega_offset = self._angular_velocity_rad_s - new_omega

            self._c1_join_state = C1JoinState(
                join_start_timestamp_ns=timestamp_ns,
                join_duration_s=self.c1_join_duration_s,
                position_offset_m=pos_offset,
                velocity_offset_mps=vel_offset,
                rotation_offset_rotvec_rad=rot_offset,
                angular_velocity_offset_rad_s=omega_offset,
                position_join_jump_m=0.0,
                velocity_join_jump_mps=vel_jump,
            )
            self._linear_velocity_mps = new_vel
            self._angular_velocity_rad_s = new_omega
        else:
            alpha = self.velocity_ema_alpha
            self._linear_velocity_mps = (1.0 - alpha) * self._linear_velocity_mps + alpha * raw_vel
            self._angular_velocity_rad_s = (
                1.0 - alpha
            ) * self._angular_velocity_rad_s + alpha * raw_omega
            if self._c1_join_state is not None:
                elapsed = (timestamp_ns - self._c1_join_state.join_start_timestamp_ns) / 1e9
                if elapsed >= self._c1_join_state.join_duration_s:
                    self._c1_join_state = None

        self._last_pose = T_obs
        self._last_timestamp_ns = timestamp_ns

        return predict_multi_horizon_se3_trajectory(
            current_pose=T_obs,
            current_timestamp_ns=timestamp_ns,
            linear_velocity_mps=self._linear_velocity_mps,
            angular_velocity_rad_s=self._angular_velocity_rad_s,
            base_position_std_m=base_position_std_m,
            base_rotation_std_deg=base_rotation_std_deg,
            horizons_ms=self.horizons_ms,
            maximum_translation_speed_mps=self.maximum_translation_speed_mps,
            maximum_angular_speed_deg_s=self.maximum_angular_speed_deg_s,
            c1_join_state=self._c1_join_state,
        )
