"""Behavioral detectors for identifying concerning agent patterns."""

from agentsmith.detectors.survival import SurvivalDetector
from agentsmith.detectors.replication import ReplicationDetector
from agentsmith.detectors.deception import DeceptionDetector

ALL_DETECTORS: dict[str, type] = {
    "survival": SurvivalDetector,
    "replication": ReplicationDetector,
    "deception": DeceptionDetector,
}

__all__ = [
    "SurvivalDetector",
    "ReplicationDetector",
    "DeceptionDetector",
    "ALL_DETECTORS",
]
