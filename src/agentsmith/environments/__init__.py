"""Sandboxed test environments for evaluating agent behaviors."""

from agentsmith.environments.base import BaseEnvironment, EnvironmentResult
from agentsmith.environments.shutdown import ShutdownEnvironment
from agentsmith.environments.replication import ReplicationEnvironment
from agentsmith.environments.resource import ResourceEnvironment
from agentsmith.environments.goal_mod import GoalModificationEnvironment

ALL_ENVIRONMENTS: dict[str, type[BaseEnvironment]] = {
    "shutdown": ShutdownEnvironment,
    "replication": ReplicationEnvironment,
    "resource": ResourceEnvironment,
    "goal_modification": GoalModificationEnvironment,
}

__all__ = [
    "BaseEnvironment",
    "EnvironmentResult",
    "ShutdownEnvironment",
    "ReplicationEnvironment",
    "ResourceEnvironment",
    "GoalModificationEnvironment",
    "ALL_ENVIRONMENTS",
]
