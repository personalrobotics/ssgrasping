"""Robot-neutral grasping and manipulation skills over ssrobot.

ssgrasping orchestrates ssrobot's public planning, execution, observation, and
attachment APIs into staged skills: grasp, verify, retract, place, and release. It owns
no simulator, planner, kinematics, retiming, or runtime control, and imports none.
"""

from ssgrasping.grasp import GraspSpec, grasp
from ssgrasping.outcomes import Evidence, SkillResult, SkillStatus, Stage
from ssgrasping.place import PlaceSpec, place

__all__ = [
    "Evidence",
    "GraspSpec",
    "PlaceSpec",
    "SkillResult",
    "SkillStatus",
    "Stage",
    "grasp",
    "place",
]
