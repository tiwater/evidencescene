"""Versioned task semantics and physical priors for scene reconstruction.

The scene runtime must not infer product semantics from class-name literals.
Task profiles are immutable release artifacts that map detector labels to task
roles, prompts, physical models, and acceptance gates.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Literal

try:
    from pydantic import (
        BaseModel,
        ConfigDict,
        Field,
        ValidationError,
        field_validator,
        model_validator,
    )
except ImportError:  # pragma: no cover - standalone fallback when pydantic is absent
    class ValidationError(ValueError):
        pass

    def ConfigDict(**kwargs: Any) -> dict[str, Any]:
        return dict(kwargs)

    def Field(default: Any = None, **kwargs: Any) -> Any:
        if "default_factory" in kwargs and callable(kwargs["default_factory"]):
            return kwargs["default_factory"]()
        return default

    def field_validator(*_args: Any, **_kwargs: Any) -> Any:
        def _decorator(fn: Any) -> Any:
            return fn
        return _decorator

    def model_validator(*_args: Any, **_kwargs: Any) -> Any:
        def _decorator(fn: Any) -> Any:
            return fn
        return _decorator

    class BaseModel:
        def __init__(self, **kwargs: Any) -> None:
            for cls in reversed(type(self).__mro__):
                for key, val in cls.__dict__.items():
                    if not key.startswith("_") and not callable(val) and not isinstance(val, (classmethod, staticmethod, property)):
                        object.__setattr__(self, key, val)
            for key, val in kwargs.items():
                object.__setattr__(self, key, val)
            if hasattr(self, "maximum_diameter_m"):
                d = getattr(self, "maximum_diameter_m", None)
                if isinstance(d, bool) or not isinstance(d, (int, float)) or not math.isfinite(d) or d <= 0:
                    raise ValidationError("maximum_diameter_m must be finite and positive")
            if hasattr(self, "source"):
                s = getattr(self, "source", "")
                if not isinstance(s, str) or not s.strip():
                    raise ValidationError("source must be nonempty")

        def model_dump(self, **_kwargs: Any) -> dict[str, Any]:
            return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

        @classmethod
        def model_validate(cls, payload: dict[str, Any]) -> Any:
            return cls(**payload)


TASK_PROFILE_SCHEMA = "ticos.scene-task-profile.v1"
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class TaskProfileError(ValueError):
    """Raised when a task profile cannot safely configure the runtime."""


def normalize_task_label(value: str) -> str:
    """Return the canonical comparison form for detector and profile labels."""

    return " ".join(value.strip().lower().replace("_", " ").replace("-", " ").split())


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class SupportedSquarePrismGeometry(_FrozenModel):
    kind: Literal["supported_square_prism"]
    side_m: float = Field(gt=0.0, le=10.0)

    @field_validator("side_m")
    @classmethod
    def finite_side(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("side_m must be finite")
        return value


class ConfiguredDimensionPrior(_FrozenModel):
    """A sourced scene constraint, never a reconstructed object dimension."""

    quantity: Literal[
        "horizontal_side_short",
        "horizontal_side_long",
        "height",
        "top_diameter",
        "bottom_diameter",
    ]
    minimum_m: float = Field(strict=True, gt=0.0, allow_inf_nan=False)
    maximum_m: float = Field(strict=True, gt=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def ordered_range(self) -> "ConfiguredDimensionPrior":
        if self.minimum_m > self.maximum_m:
            raise ValueError("configured dimension minimum cannot exceed maximum")
        return self


class PhysicalBounds(_FrozenModel):
    """Explicit configuration prior; never measured dimensions or geometry."""

    maximum_diameter_m: float = Field(strict=True, gt=0.0, allow_inf_nan=False)
    source: str = Field(strict=True, min_length=1)
    semantics: Literal["configured_prior_not_measured_geometry"] = "configured_prior_not_measured_geometry"
    dimension_priors: tuple[ConfiguredDimensionPrior, ...] = ()

    @field_validator("source")
    @classmethod
    def nonempty_source(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("physical bound source must be nonempty")
        return value.strip()

    @field_validator("dimension_priors")
    @classmethod
    def unique_dimension_quantities(
        cls, values: tuple[ConfiguredDimensionPrior, ...]
    ) -> tuple[ConfiguredDimensionPrior, ...]:
        quantities = [value.quantity for value in values]
        if len(quantities) != len(set(quantities)):
            raise ValueError("configured dimension quantities must be unique")
        return values


class TaskFeatureSpec(_FrozenModel):
    """A current-frame feature-support request, not geometric configuration."""

    feature_id: str
    geometry_kind: Literal["point", "circular_aperture", "circular_aperture_plane"]
    grounding_scope: Literal["full_image", "object_crop"] = "full_image"
    foundation_prompts: tuple[str, ...] = ()
    direction_foundation_prompts: tuple[str, ...] = ()

    @field_validator("feature_id")
    @classmethod
    def safe_feature_id(cls, value: str) -> str:
        if not _IDENTIFIER_RE.fullmatch(value):
            raise ValueError("feature_id must be a safe 1-128 character identifier")
        return value

    @field_validator("foundation_prompts")
    @classmethod
    def unique_feature_prompts(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(value.strip() for value in values)
        if any(not value for value in cleaned):
            raise ValueError("feature prompts must be nonempty strings")
        normalized = tuple(normalize_task_label(value) for value in cleaned)
        if len(normalized) != len(set(normalized)):
            raise ValueError("feature prompts must be unique after normalization")
        return cleaned

    @field_validator("direction_foundation_prompts")
    @classmethod
    def unique_direction_prompts(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(value.strip() for value in values)
        if any(not value for value in cleaned):
            raise ValueError("direction feature prompts must be nonempty strings")
        normalized = tuple(normalize_task_label(value) for value in cleaned)
        if len(normalized) != len(set(normalized)):
            raise ValueError(
                "direction feature prompts must be unique after normalization"
            )
        return cleaned

    @model_validator(mode="after")
    def direction_prompts_require_point_geometry(self) -> "TaskFeatureSpec":
        if self.direction_foundation_prompts and self.geometry_kind != "point":
            raise ValueError(
                "direction feature prompts require point geometry"
            )
        return self


class TaskObject(_FrozenModel):
    object_id: str
    labels: tuple[str, ...] = Field(min_length=1)
    roles: tuple[str, ...] = Field(min_length=1)
    foundation_prompts: tuple[str, ...] = ()
    task_features: tuple[TaskFeatureSpec, ...] = ()
    geometry: SupportedSquarePrismGeometry | None = None
    physical_bounds: PhysicalBounds | None = None

    @field_validator("object_id")
    @classmethod
    def safe_object_id(cls, value: str) -> str:
        if not _IDENTIFIER_RE.fullmatch(value):
            raise ValueError("object_id must be a safe 1-128 character identifier")
        return value

    @field_validator("labels", "roles", "foundation_prompts")
    @classmethod
    def nonempty_unique_strings(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(value.strip() for value in values)
        if any(not value for value in cleaned):
            raise ValueError("values must be nonempty strings")
        normalized = tuple(normalize_task_label(value) for value in cleaned)
        if len(normalized) != len(set(normalized)):
            raise ValueError("values must be unique after normalization")
        return cleaned

    @model_validator(mode="after")
    def unique_task_features(self) -> "TaskObject":
        feature_ids = [feature.feature_id for feature in self.task_features]
        if len(feature_ids) != len(set(feature_ids)):
            raise ValueError("task feature IDs must be unique per object")
        return self

    @property
    def canonical_label(self) -> str:
        return self.labels[0]

    def feature_for_id(self, feature_id: str) -> TaskFeatureSpec | None:
        return next(
            (feature for feature in self.task_features if feature.feature_id == feature_id),
            None,
        )


class ObjectSurfaceQualityGate(_FrozenModel):
    contract_version: str = "g4-current-object-surface.v1"
    valid_depth_ratio_min: float = Field(ge=0.0, le=1.0)
    heldout_mask_iou_min: float = Field(ge=0.0, le=1.0)
    required_semantic_classes: tuple[str, ...] = Field(min_length=1)
    foreground_background_depth_separation_error_p95_m_max: float = Field(ge=0.0)

    @field_validator("contract_version")
    @classmethod
    def nonempty_contract(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("contract_version must be nonempty")
        return value.strip()

    @field_validator("required_semantic_classes")
    @classmethod
    def unique_semantic_classes(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(value.strip() for value in values)
        if any(not value for value in cleaned):
            raise ValueError("required semantic classes must be nonempty")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("required semantic classes must be unique")
        return cleaned

    @field_validator("foreground_background_depth_separation_error_p95_m_max")
    @classmethod
    def finite_depth_error(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("depth separation threshold must be finite")
        return value


class TaskQualityGates(_FrozenModel):
    object_surface: ObjectSurfaceQualityGate


class TaskEvidenceRequirement(_FrozenModel):
    """A fact a task needs from an existing perception/state owner."""

    requirement_id: str
    subject_type: Literal["task_object", "scene"] = "task_object"
    object_id: str | None = None
    evidence_kind: str
    feature_id: str | None = None
    frame_scope: Literal[
        "any",
        "image",
        "camera_local",
        "object_gauge",
        "measured_table_local",
        "measured_shared",
        "robot_base",
    ] = "any"
    current_update_required: bool = Field(default=True, strict=True)
    required: bool = Field(default=True, strict=True)
    maximum_age_ms: int | None = Field(default=None, strict=True, ge=0)
    minimum_confidence: float | None = Field(
        default=None, strict=True, ge=0.0, le=1.0, allow_inf_nan=False
    )
    maximum_uncertainty_m: float | None = Field(
        default=None, strict=True, ge=0.0, allow_inf_nan=False
    )
    minimum_authority: Literal["owner_output", "diagnostic"] = "owner_output"
    source_owner: str | None = None
    camera_id: str | None = None
    camera_topic: str | None = None

    @field_validator("requirement_id", "evidence_kind")
    @classmethod
    def safe_evidence_identifier(cls, value: str) -> str:
        if not _IDENTIFIER_RE.fullmatch(value):
            raise ValueError("evidence identifiers must be safe 1-128 character identifiers")
        return value

    @field_validator("object_id")
    @classmethod
    def safe_optional_object_id(cls, value: str | None) -> str | None:
        if value is not None and not _IDENTIFIER_RE.fullmatch(value):
            raise ValueError("object_id must be a safe 1-128 character identifier")
        return value

    @model_validator(mode="after")
    def validate_subject(self) -> "TaskEvidenceRequirement":
        if self.subject_type == "task_object" and self.object_id is None:
            raise ValueError("task-object evidence requirements need object_id")
        if self.subject_type == "scene" and self.object_id is not None:
            raise ValueError("scene evidence requirements cannot set object_id")
        return self

    @field_validator("feature_id")
    @classmethod
    def safe_optional_feature_id(cls, value: str | None) -> str | None:
        if value is not None and not _IDENTIFIER_RE.fullmatch(value):
            raise ValueError("feature_id must be a safe 1-128 character identifier")
        return value

    @field_validator("source_owner")
    @classmethod
    def safe_optional_source_owner(cls, value: str | None) -> str | None:
        if value is not None and not _IDENTIFIER_RE.fullmatch(value):
            raise ValueError("source_owner must be a safe 1-128 character identifier")
        return value

    @field_validator("camera_id", "camera_topic")
    @classmethod
    def nonempty_camera_selector(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("camera selectors must be nonempty when configured")
        return normalized


class TaskRelationRequirement(_FrozenModel):
    """A relation evaluated only from facts selected by evidence requirements."""

    relation_id: str
    source_requirement_id: str
    target_requirement_id: str
    operator: Literal[
        "center_translation",
        "visible_surface_centroid_translation",
        "relative_transform",
        "relative_aperture_geometry",
        "point_to_aperture",
        "point_to_aperture_plane",
    ]
    require_same_frame: bool = Field(default=True, strict=True)
    maximum_timestamp_skew_ms: int = Field(default=50, strict=True, ge=0)
    required: bool = Field(default=True, strict=True)

    @field_validator(
        "relation_id", "source_requirement_id", "target_requirement_id"
    )
    @classmethod
    def safe_relation_identifier(cls, value: str) -> str:
        if not _IDENTIFIER_RE.fullmatch(value):
            raise ValueError("relation identifiers must be safe 1-128 character identifiers")
        return value

    @model_validator(mode="after")
    def supported_operators_require_a_shared_frame(self) -> "TaskRelationRequirement":
        # These operators perform vector, transform or feature-geometry algebra. There
        # is no meaningful result if their inputs use different coordinate frames.
        if not self.require_same_frame:
            raise ValueError(f"{self.operator} requires the same coordinate frame")
        return self


class TaskProfile(_FrozenModel):
    schema_version: Literal["ticos.scene-task-profile.v1"] = Field(
        default=TASK_PROFILE_SCHEMA,
        alias="schema",
    )
    profile_id: str
    revision: str
    primary_object_id: str
    objects: tuple[TaskObject, ...] = Field(min_length=1)
    quality_gates: TaskQualityGates
    evidence_requirements: tuple[TaskEvidenceRequirement, ...] = ()
    relation_requirements: tuple[TaskRelationRequirement, ...] = ()

    @field_validator("profile_id", "revision", "primary_object_id")
    @classmethod
    def safe_identifier(cls, value: str) -> str:
        if not _IDENTIFIER_RE.fullmatch(value):
            raise ValueError("value must be a safe 1-128 character identifier")
        return value

    @model_validator(mode="after")
    def validate_identity_and_aliases(self) -> "TaskProfile":
        object_ids = [item.object_id for item in self.objects]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("object IDs must be unique")
        if self.primary_object_id not in set(object_ids):
            raise ValueError("primary_object_id must reference an object")
        requirement_ids = [item.requirement_id for item in self.evidence_requirements]
        if len(requirement_ids) != len(set(requirement_ids)):
            raise ValueError("evidence requirement IDs must be unique")
        unknown_requirement_objects = sorted(
            {
                item.object_id
                for item in self.evidence_requirements
                if item.subject_type == "task_object" and item.object_id is not None
            }
            - set(object_ids)
        )
        if unknown_requirement_objects:
            raise ValueError(
                "evidence requirements reference unknown objects: "
                + ", ".join(unknown_requirement_objects)
            )
        relation_ids = [item.relation_id for item in self.relation_requirements]
        if len(relation_ids) != len(set(relation_ids)):
            raise ValueError("relation requirement IDs must be unique")
        unknown_relation_evidence = sorted(
            {
                requirement_id
                for relation in self.relation_requirements
                for requirement_id in (
                    relation.source_requirement_id,
                    relation.target_requirement_id,
                )
                if requirement_id not in set(requirement_ids)
            }
        )
        if unknown_relation_evidence:
            raise ValueError(
                "relation requirements reference unknown evidence requirements: "
                + ", ".join(unknown_relation_evidence)
            )
        owners: dict[str, str] = {}
        for item in self.objects:
            for label in item.labels:
                normalized = normalize_task_label(label)
                previous = owners.setdefault(normalized, item.object_id)
                if previous != item.object_id:
                    raise ValueError(
                        f"normalized label {normalized!r} belongs to multiple objects"
                    )
        return self

    @property
    def primary_object(self) -> TaskObject:
        return next(item for item in self.objects if item.object_id == self.primary_object_id)

    def object_for_label(self, label: str) -> TaskObject | None:
        normalized = normalize_task_label(label)
        for item in self.objects:
            if normalized in {normalize_task_label(alias) for alias in item.labels}:
                return item
        return None

    def supported_square_prism_sides_by_label(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for item in self.objects:
            if item.geometry is None or item.geometry.kind != "supported_square_prism":
                continue
            for label in item.labels:
                result[normalize_task_label(label)] = item.geometry.side_m
        return result

    def canonical_json_bytes(self) -> bytes:
        payload = self.model_dump(mode="json", by_alias=True)
        # Preserve v1 content identity for profiles without this optional prior.
        # Do not globally exclude None: legacy geometry:null is part of the hash.
        for item in payload["objects"]:
            if item.get("physical_bounds") is None:
                item.pop("physical_bounds", None)
            if item.get("task_features"):
                for feature in item["task_features"]:
                    if not feature.get("direction_foundation_prompts"):
                        feature.pop("direction_foundation_prompts", None)
            else:
                item.pop("task_features", None)
        # Keep the v1 content identity of profiles written before task
        # requirements became part of the optional scene contract.
        if not payload.get("evidence_requirements"):
            payload.pop("evidence_requirements", None)
        else:
            for requirement in payload["evidence_requirements"]:
                if requirement.get("subject_type") == "task_object":
                    requirement.pop("subject_type", None)
                for optional_threshold in (
                    "minimum_confidence",
                    "maximum_uncertainty_m",
                    "camera_id",
                    "camera_topic",
                ):
                    if requirement.get(optional_threshold) is None:
                        requirement.pop(optional_threshold, None)
        if not payload.get("relation_requirements"):
            payload.pop("relation_requirements", None)
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json_bytes()).hexdigest()


def load_task_profile(path: Path | str) -> TaskProfile:
    source = Path(path)
    if not source.is_absolute() and not source.exists():
        repo_candidate = Path(__file__).resolve().parents[4] / source
        if repo_candidate.exists():
            source = repo_candidate
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        return TaskProfile.model_validate(payload)
    except OSError as error:
        raise TaskProfileError(f"task profile is not readable: {source}") from error
    except (json.JSONDecodeError, ValidationError) as error:
        raise TaskProfileError(f"invalid task profile {source}: {error}") from error

