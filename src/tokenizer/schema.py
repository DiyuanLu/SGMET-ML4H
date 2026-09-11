from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class FeatureType(IntEnum):
    """Feature type ids consumed by the name-value transformer."""

    NUMERICAL = 0
    CATEGORICAL = 1


@dataclass(frozen=True)
class FeatureSpec:
    """Metadata for one tabular feature token."""

    name: str
    feature_type: FeatureType
    unit: str | None = None
    description: str | None = None
    categories: tuple[str, ...] = ()
    apply_log: bool = False

    def to_feature_text(self) -> str:
        """Build the semantic text prompt encoded by a frozen text embedding model."""
        parts = [
            f"Type: {self.feature_type.name.lower()}.",
        ]
        if self.unit and (not self.description or "Unit:" not in self.description):
            parts.append(f"Unit: {self.unit}.")
        if self.description:
            parts.append(self.description)
        return "\n".join(parts)
