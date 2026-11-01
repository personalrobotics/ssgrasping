"""Structured outcomes of staged skills.

A skill runs its stages in order and records each one: its name, whether it succeeded,
why, the runtime time and scene revision when it ended, and what the skill then
believed about the object (held, attached). The result names the stage that decided it
and the attachment the scene declared at the end. Stages are skill-specific; only the
state is shared.

A failed stage says why twice: as a ``Reason``, with the ssrobot ``cause`` behind it
when there is one, for an application to act on; and as ``detail``, prose for a person.
The reason and cause are the protocol; the detail may be reworded at any time.

A skill also estimates the manipulation transitions it passes through (#28) from the
signals a physical robot reports, as ``TransitionEstimate``s. Privileged state never
enters an estimate: where the simulator has a privileged reference, it is kept beside
the estimate in a ``TransitionEvaluation``, for scoring, and never decides a stage. A
reference is not ground truth: the attachment monitor, for one, can report an object
lost that is still in the hand.
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


class Reason(enum.StrEnum):
    """Why a stage failed, stable across releases; the README tabulates each reason's
    meaning, the stages that report it, and its cause's vocabulary."""

    UNAVAILABLE = "unavailable"
    NOT_PLANNED = "not_planned"
    REFUSED = "refused"
    COMMAND_FAILED = "command_failed"
    COMMAND_CANCELED = "command_canceled"
    CANCELLED = "cancelled"
    NOT_ATTACHED = "not_attached"
    STILL_ATTACHED = "still_attached"
    ATTACHMENT_LOST = "attachment_lost"
    MISSED = "missed"
    LOST = "lost"
    IN_CONTACT = "in_contact"
    RELEASE_UNCONFIRMED = "release_unconfirmed"
    NOT_AT_REST = "not_at_rest"
    NO_CONTACT = "no_contact"
    OUTSIDE_REGIONS = "outside_regions"
    TRANSITION_REFUTED = "transition_refuted"


class Transition(enum.StrEnum):
    """A manipulation transition: of the hand and the object, or of the task's outcome."""

    ACQUIRED = "acquired"
    """The fingers closed on the object."""
    RETAINED = "retained"
    """The object is still in hand, throughout a carry."""
    RELEASED = "released"
    """The object has left the hand."""
    PLACED = "placed"
    """The object rests as the placement intended."""
    LANDED = "landed"
    """The object came to rest in the intended container region."""


class TransitionOutcome(enum.StrEnum):
    """What the evidence says of a transition. ``unobserved`` is not success: nothing
    available confirmed or refuted it, and what to do then is the application's."""

    CONFIRMED = "confirmed"
    REFUTED = "refuted"
    UNOBSERVED = "unobserved"


class EstimateReason(enum.StrEnum):
    """Why an estimate came out as it did, stable across releases."""

    OPENING_HELD = "opening_held"
    """The gripper opening stayed at or above the guard's bound through the motion."""
    OPENING_FELL = "opening_fell"
    """The gripper opening fell below the guard's bound: the fingers closed past the
    object's width."""
    NO_OPENING_MARGIN = "no_opening_margin"
    """The opening the object was held at is not above the closing target, so the
    fingers have nowhere to close to if it leaves."""


@dataclass(frozen=True, slots=True)
class Observed:
    """A reading an estimate or an evaluation used."""

    channel: str
    time_ns: int
    value: tuple[float, ...]
    evidence: Evidence
    """``portable`` or ``privileged``."""


@dataclass(frozen=True, slots=True)
class Criterion:
    """A value an estimate was decided by, and where it came from."""

    name: str
    value: float
    source: str


@dataclass(frozen=True, slots=True)
class TransitionEstimate:
    """A transition, estimated in ``stage`` from portable evidence only."""

    transition: Transition
    stage: str
    outcome: TransitionOutcome
    reason: EstimateReason
    evidence: tuple[Observed, ...]
    criteria: tuple[Criterion, ...]


@dataclass(frozen=True, slots=True)
class TransitionEvaluation:
    """An estimate and the privileged reference it is scored against, for evaluation in
    simulation only: never read by a stage."""

    estimate: TransitionEstimate
    reference: TransitionOutcome
    """``confirmed`` or ``refuted``, as the privileged reference judged it."""
    source: str
    """What judged ``reference``, such as ``attachment_monitor``."""
    evidence: tuple[Observed, ...]
    """The privileged readings behind ``reference``."""

    @property
    def agrees(self) -> bool:
        return self.estimate.outcome is self.reference


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
    reason: Reason | None = None
    """Why it failed: None when it succeeded."""
    cause: str | None = None
    """The ssrobot code behind ``reason``, in the vocabulary the reason names (a plan
    status, an error code, an execution diagnostic): None when there is none."""


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
    transitions: tuple[TransitionEstimate, ...] = ()
    """The transitions estimated, in order."""
    evaluations: tuple[TransitionEvaluation, ...] = ()
    """Each estimate the runtime had a privileged reference for, scored against it."""

    @property
    def stage(self) -> Stage:
        """The stage that decided the result: the failed one, or the last."""
        return self.stages[-1]

    @property
    def succeeded(self) -> bool:
        return self.state is SkillState.SUCCEEDED
