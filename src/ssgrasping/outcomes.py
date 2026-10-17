"""Structured outcomes of staged skills.

A skill runs its stages in order and records each one: its name, whether it succeeded,
why, the runtime time and scene revision when it ended, and what the skill then
believed about the object (held, attached). The result names the stage that decided it
and the attachment the scene declared at the end. Stages are skill-specific; only the
state is shared.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from ssrobot import Attachment


class SkillState(enum.StrEnum):
    """Where a skill execution is: not started, running, or how it ended."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Evidence(enum.StrEnum):
    """Where a stage's evidence came from."""

    PORTABLE = "portable"
    """An observation a physical robot also reports, such as a gripper opening or its
    joint positions."""
    PRIVILEGED = "privileged"
    """Simulator state a physical robot does not report, such as an object's pose or the
    attachment monitor."""
    PLANNED = "planned"
    """The description, the planner, and the validator, on the snapshot the stage
    planned in."""
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
    """A skill's stages, in order, how it ended, and the attachment it left declared."""

    skill: str
    state: SkillState
    stages: tuple[Stage, ...]
    attachment: Attachment | None
    """The object's attachment as the scene declared it at the end, if any."""
    values: dict[str, object] = field(default_factory=dict)
    """Measured values worth keeping, by name: an opening, a reached distance."""

    @property
    def stage(self) -> Stage:
        """The stage that decided the result: the failed one, or the last."""
        return self.stages[-1]

    @property
    def succeeded(self) -> bool:
        return self.state is SkillState.SUCCEEDED
