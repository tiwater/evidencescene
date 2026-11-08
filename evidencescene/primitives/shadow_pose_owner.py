"""Shadow pose-owner contracts and controlled re-anchor policy.

Shadow owners may score camera-local evidence. They must not authorize wire
publication. Production BundleTrack remains authoritative until cutover.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol


SHADOW_POSE_OWNER_SCHEMA = "ticos.shadow-pose-owner-tick.v1"

# Reasons that may request a surface-preserving controlled re-anchor in shadow.
CONTROLLED_REANCHOR_REASONS = frozenset(
    {
        "original_world_reanchor_required",
        "sam_memory_exhausted",
        "sam2_memory_exhausted",
        # SAM lost the current mask: allow foundation re-seed of the same owner
        # after the ordinary failure streak (or first-miss for unavailable).
        # Official NO_OBJ fill is intentionally NOT an immediate close trigger:
        # same-tick reseed in OriginalPersistentWorld handles gate-passing
        # exact foundation masks, and a single pour-window NO_OBJ must keep
        # the temporal SAM owner open instead of closing into nearest-mask limbo.
        "sam_mask_unavailable",
        # sam_object_score_not_appearing intentionally omitted: closing on NO_OBJ
        # streak forced far-template limbo; recovery is same-tick exact reseed.
        "original_world_foundation_mask_corroboration_required",
    }
)

# Explicitly rejected as re-anchor triggers (see tools/patches/README.md).
REJECTED_REANCHOR_REASONS = frozenset(
    {
        "sam_depth_support_insufficient",
    }
)


@dataclass(frozen=True)
class ShadowPoseTick:
    """One scored observation for a shadow or baseline arm."""

    frame_index: int
    status: str
    reason: str | None
    mask_available: bool
    camera_from_object: list[list[float]] | None
    total_ms: float | None
    pose_ms: float | None
    wire_authorized: bool = False
    arm: str = "unknown"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SHADOW_POSE_OWNER_SCHEMA,
            "arm": self.arm,
            "frame_index": self.frame_index,
            "status": self.status,
            "reason": self.reason,
            "mask_available": self.mask_available,
            "camera_from_object": self.camera_from_object,
            "runtime_timing_ms": {
                "total_ms": self.total_ms,
                "pose_ms": self.pose_ms,
            },
            "wire_authorized": self.wire_authorized,
            "world_pose_authorized": False,
        }


class ShadowPoseOwner(Protocol):
    """Advance one shadow tick without mutating the live world wire."""

    @property
    def arm_name(self) -> str: ...

    def advance(self, observation: Mapping[str, Any]) -> ShadowPoseTick: ...


def should_request_controlled_reanchor(
    *,
    not_observable_streak: int,
    has_retained_surface: bool,
    last_reason: str | None,
    failure_streak_threshold: int = 3,
) -> bool:
    """Return whether a surface-preserving controlled re-anchor is allowed.

    Production currently reanchors empty sessions after a failure streak. This
    policy additionally allows a surface-preserving re-anchor when a retained
    surface exists for SAM-memory exhaustion, explicit reanchor-required, or
    ``sam_mask_unavailable`` (so foundation masks can re-seed the same owner).
    Weak depth support remains a rejected trigger.

    ``sam_mask_unavailable`` re-anchors on the first miss so foundation_mask_cache
    can re-seed when the temporal path had bypassed the detector. Official
    ``sam_object_score_not_appearing`` does *not*: closing on the first NO_OBJ
    forced a nearest-neighbor / template open that failed far/extent gates
    (308–313 limbo) while a healthy 297-seeded SAM memory could have continued.
    NO_OBJ still qualifies after the ordinary failure streak, and same-tick
    reseed remains the path when an exact gate-passing foundation mask is
    already in hand this tick.
    """
    if type(not_observable_streak) is not int or not_observable_streak < 0:
        raise ValueError("not_observable_streak must be a non-negative int")
    if type(failure_streak_threshold) is not int or failure_streak_threshold <= 0:
        raise ValueError("failure_streak_threshold must be a positive int")
    reason = str(last_reason or "")
    if reason in REJECTED_REANCHOR_REASONS:
        return False
    # Pour-window official NO_OBJ must keep the temporal owner open.
    # Same-tick exact foundation reseed recovers when a gate-passing
    # mask is already in hand; empty-session fallback must not close.
    # A far-surface mask is discarded from SAM memory inside the session.
    # Closing the owner here drops the carton track, and this recording has
    # no later foundation mask that can reopen it.
    if reason in {
        "sam_object_score_not_appearing",
        "sam_depth_far_contaminated",
        # Unmatched frame kept in the BundleTrack bank. Closing the session
        # would drop that reference and the carton track with it.
        "tracker_reference_retained",
    }:
        return False
    if reason == "sam_mask_unavailable" and not_observable_streak >= 1:
        return True
    if not_observable_streak < failure_streak_threshold:
        return False
    if reason in CONTROLLED_REANCHOR_REASONS:
        return True
    # Empty-session recovery: no surface yet and streak exceeded. A missing
    # reason still qualifies — the live path historically re-anchored empty
    # sessions from streak alone so foundation masks are not bypassed forever.
    return not has_retained_surface


def score_controlled_reanchor_policy(
    ticks: list[Mapping[str, Any]],
    *,
    failure_streak_threshold: int = 3,
) -> list[dict[str, Any]]:
    """Annotate a baseline tick stream with controlled-reanchor decisions."""
    streak = 0
    events: list[dict[str, Any]] = []
    for index, tick in enumerate(ticks):
        status = str(tick.get("status") or "")
        reason = tick.get("reason")
        reason_text = str(reason) if isinstance(reason, str) else None
        if status == "candidate":
            streak = 0
            would = False
        else:
            streak += 1
            surface = tick.get("has_retained_surface")
            if surface is None:
                surface = tick.get("object_gauge_surface") is not None
            would = should_request_controlled_reanchor(
                not_observable_streak=streak,
                has_retained_surface=bool(surface),
                last_reason=reason_text,
                failure_streak_threshold=failure_streak_threshold,
            )
        events.append(
            {
                "frame_index": int(tick.get("frame_index", index)),
                "status": status,
                "reason": reason_text,
                "not_observable_streak": streak,
                "would_request_controlled_reanchor": would,
                "wire_authorized": False,
            }
        )
    return events

