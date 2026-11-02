"""Robot-neutral grasping and manipulation skills over ssrobot.

ssgrasping orchestrates ssrobot's public planning, execution, observation, and
attachment APIs into staged skills: grasp, verify, retract, place, drop, and release,
and observes where a dropped object lands. It owns no simulator, planner, kinematics,
retiming, or runtime control, and imports none.
"""

from ssgrasping._skill import Lookahead
from ssgrasping.candidates import Candidate, Candidates, Selection
from ssgrasping.drop import DropSpec, drop
from ssgrasping.execution import SkillExecution, run_skill, run_together
from ssgrasping.grasp import GraspSpec, grasp
from ssgrasping.landing import LandingSpec, landing
from ssgrasping.outcomes import (
    Criterion,
    EstimateReason,
    Evidence,
    Observed,
    Reason,
    SkillResult,
    SkillState,
    Stage,
    Transition,
    TransitionEstimate,
    TransitionEvaluation,
    TransitionOutcome,
)
from ssgrasping.place import PlaceSpec, place

__all__ = [
    "Candidate",
    "Candidates",
    "Criterion",
    "DropSpec",
    "EstimateReason",
    "Evidence",
    "GraspSpec",
    "LandingSpec",
    "Lookahead",
    "Observed",
    "PlaceSpec",
    "Reason",
    "Selection",
    "SkillExecution",
    "SkillResult",
    "SkillState",
    "Stage",
    "Transition",
    "TransitionEstimate",
    "TransitionEvaluation",
    "TransitionOutcome",
    "drop",
    "grasp",
    "landing",
    "place",
    "run_skill",
    "run_together",
]
