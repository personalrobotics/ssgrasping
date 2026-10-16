"""Structured outcomes of staged skills.

A skill runs its stages in order and records each one: its name, whether it succeeded,
why, the runtime time and scene revision when it ended, and what the skill then
believed about the object (held, attached). The result names the stage that decided it.
Stages are skill-specific; only the terminal status is shared.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class SkillStatus(enum.StrEnum):
    """How a skill ended."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Evidence(enum.StrEnum):
    """Where a stage's evidence came from."""

    PORTABLE = "portable"
    """An observation a physical robot also reports, such as a gripper opening."""
    PRIVILEGED = "privileged"
    """Simulator state a physical robot does not report, such as an object's pose or the
    attachment monitor."""
    PLANNED = "planned"
    """The planner and validator, on the snapshot the stage planned in."""
    COMMANDED = "commanded"
    """A command's terminal status, from the runtime."""


@dataclass(frozen=True, slots=True)
class Stage:
    """One stage of a skill, as it ended."""

    name: str
    succeeded: bool
    detail: str
    time_ns: int
    revision: int
    held: bool | None
    """Whether the skill then believed the object held: None when it cannot tell."""
    attached: bool
    """Whether the scene then declared the object attached."""
    evidence: Evidence


@dataclass(frozen=True, slots=True)
class SkillResult:
    """A skill's stages, in order, and how it ended."""

    skill: str
    status: SkillStatus
    stages: tuple[Stage, ...]
    values: dict[str, object] = field(default_factory=dict)
    """Measured values worth keeping, by name: an opening, a reached distance."""

    @property
    def stage(self) -> Stage:
        """The stage that decided the result: the failed one, or the last."""
        return self.stages[-1]

    @property
    def succeeded(self) -> bool:
        return self.status is SkillStatus.SUCCEEDED
