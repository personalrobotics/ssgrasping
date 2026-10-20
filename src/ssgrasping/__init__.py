"""Robot-neutral grasping and manipulation skills over ssrobot.

ssgrasping orchestrates ssrobot's public planning, execution, observation, and
attachment APIs into staged skills: grasp, verify, retract, place, drop, and release. It owns
no simulator, planner, kinematics, retiming, or runtime control, and imports none.
"""

from ssgrasping.drop import DropSpec, drop
from ssgrasping.execution import SkillExecution, run_skill
from ssgrasping.grasp import GraspSpec, grasp
from ssgrasping.outcomes import Evidence, SkillResult, SkillState, Stage
from ssgrasping.place import PlaceSpec, place

__all__ = [
    "DropSpec",
    "Evidence",
    "GraspSpec",
    "PlaceSpec",
    "SkillExecution",
    "SkillResult",
    "SkillState",
    "Stage",
    "drop",
    "grasp",
    "place",
    "run_skill",
]
