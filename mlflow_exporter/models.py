"""Domain value objects shared across exporter components."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class MlflowSnapshot:
    """Aggregated MLflow state exported to Prometheus."""

    experiments_total: int
    experiments_active_total: int
    experiments_deleted_total: int
    runs_total: int
    runs_by_status: Mapping[str, int]
    registered_models_total: int
    model_versions_total: int
    model_versions_by_stage: Mapping[str, int]

    def __post_init__(self) -> None:
        """Freeze nested mappings so published snapshots stay immutable."""
        object.__setattr__(
            self,
            "runs_by_status",
            MappingProxyType(dict(self.runs_by_status)),
        )
        object.__setattr__(
            self,
            "model_versions_by_stage",
            MappingProxyType(dict(self.model_versions_by_stage)),
        )
