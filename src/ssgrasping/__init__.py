"""Robot-neutral grasping and manipulation skills over ssrobot.

ssgrasping orchestrates ssrobot's public planning, execution, observation, and
attachment APIs into staged skills: grasp, verify, retract, place, drop, and release. It owns
no simulator, planner, kinematics, retiming, or runtime control, and imports none.
"""

from ssgrasping._skill import Ahead, Kept, Lookahead
from ssgrasping.candidates import Candidate, Candidates, Selection
from ssgrasping.drop import DropSpec, drop
from ssgrasping.execution import SkillExecution, run_skill, run_together
from ssgrasping.grasp import GraspSpec, grasp
from ssgrasping.outcomes import Evidence, SkillResult, SkillState, Stage
from ssgrasping.place import PlaceSpec, place

__all__ = [
    "Ahead",
    "Candidate",
    "Candidates",
    "DropSpec",
    "Evidence",
    "GraspSpec",
    "Kept",
    "Lookahead",
    "PlaceSpec",
    "Selection",
    "SkillExecution",
    "SkillResult",
    "SkillState",
    "Stage",
    "drop",
    "grasp",
    "place",
    "run_skill",
    "run_together",
]
