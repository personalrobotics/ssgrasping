"""Robot-neutral grasping and manipulation skills over ssrobot.

ssgrasping orchestrates ssrobot's public planning, execution, observation, and
attachment APIs into staged skills: grasp, verify, retract, place, and release. It owns
no simulator, planner, kinematics, retiming, or runtime control, and imports none.
"""
