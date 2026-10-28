# ssgrasping

Picking up an object and putting it somewhere else is not a single motion. The robot
moves to a pose from which it can approach the object, approaches it, and closes its
fingers. Only then does it learn whether it is holding anything. The motions that
follow, and what it is safe to believe about the object, depend on that answer. A
skill that hides this structure either retries blindly or reports a success it cannot
support.

ssgrasping writes grasping, placing, and dropping as staged skills. Each stage either
moves the robot through [ssrobot](https://github.com/personalrobotics/ssrobot)'s
checked execution boundary or establishes a fact, and the skill records where each fact
came from. A skill stops at the first stage that fails and reports the object's state
conservatively, so an application can decide how to recover.

The skills are written against robot-neutral contracts. Everything that differs between
robots, such as the arm, the hand frame, the gripper, the regions in which a grasp or
placement is acceptable, and the planners, is supplied as explicit data. ssgrasping owns
no simulator, planner, kinematics, retiming, or runtime control. Its wheel depends only
on ssrobot, and importing it loads none of those systems.

## Skills and stages

A skill is a fixed sequence of named stages. `grasp` opens the gripper, moves to a
pregrasp pose, approaches along the hand's axis, and closes the fingers. It then
checks that the close stopped short of its target, as it does on something between the
fingers, before declaring an attachment. It lifts along the support's normal, then
closes again: only if the fingers still stop short is the object acquired. It finally
confirms that the object is clear of everything. `place` carries a held object to a target region, descends
under a force guard until contact, releases it, and checks where it rests. `drop`
carries the object over a container and lets it fall. It ends at the confirmed
release, so the robot can move on at once. `landing` then observes where the object
comes to rest, alongside whatever the robot does next; it commands nothing. Each module
documents its stages and their exact conditions.

Failure is reported in terms of the physical situation rather than as a bare error.
Before anything is commanded, a `preflight` stage confirms that every frame, command,
and channel the skill will use is available. A missed grasp reopens the gripper,
because that cleanup has an unambiguous meaning. Once an object has been declared
attached, a later failure leaves the attachment in place and reports the object as
held, because that is the physically conservative assumption. Retrying and recovery
belong to the application.

## Robot-specific choices are data

`GraspSpec`, `PlaceSpec`, `DropSpec`, and `LandingSpec` name everything a skill needs: the manipulator,
end effector, gripper, and gripper-opening channel; the object and the support it rests
on, with the support's outward normal; the regions in which the hand or the object
must end, as task space regions; and the distances, tolerances, and attempt limits for
each stage. Nothing is inferred from names.

The regions come from the libraries that own them. In the tests and in Geodude's demo,
grasp regions come from [sstsr](https://github.com/personalrobotics/sstsr)'s gripper
templates and placement regions from its stable placer. The application also chooses the
planners. A skill takes a main planner and, optionally, a second planner with a small
iteration budget for feasibility checks: a short straight move that is feasible plans in
a few iterations, so a bounded planner rejects an infeasible one quickly and
deterministically.

## Evidence

A stage's result is only as trustworthy as the information it used, and that differs
between simulation and a physical robot. Every stage therefore records the kind of
evidence it relied on:

| Evidence | Source |
| --- | --- |
| `portable` | An observation a physical robot also reports, such as the gripper opening or joint positions |
| `commanded` | The terminal status of a command, reported by the runtime |
| `planned` | The robot description, the planner, and its validator, on the snapshot the stage planned in |
| `privileged` | Simulator state that a physical robot does not report, such as an object's pose or the attachment monitor |

The distinction determines what a skill may rely on. Whether a grasp succeeded is
decided by `acquired`, from how the gripper's close ends after the lift, which a physical
gripper reports. The close before the lift only shows the fingers are `obstructed`: the
support could stop them too. The later `retained` stage reads the simulator's
attachment monitor and is recorded as privileged; it adds information in simulation but
never substitutes for `acquired`.

## Passive execution

A skill advances only when its caller advances it. `grasp`, `place`, `drop`, and
`landing` return a `SkillExecution` that reads the runtime clock, execution states, and
finished plans, but never steps or polls the runtime itself. `run_skill` drives one
skill through the context's own calls, stepping a manually clocked runtime or updating
an externally clocked one. `run_together` advances several items at once, such as one
object's drop and the race for the next object, or one object's landing and the next
object's grasp. ssgrasping creates no threads, clocks, or event loops
of its own.

When every unfinished item is waiting only for plans being computed on worker
threads, `run_together` waits for those plans instead of stepping the clock. No runtime
time passes while the robot waits for a plan, exactly as when a stage plans on the
calling thread. A run's timeline is therefore independent of how long planning takes,
and repeated runs are deterministic for deterministic planners.

## Planning ahead

Some motions are fully determined before the motion that precedes them has finished.
The approach to an object depends only on where the pregrasp ends, so it can be
planned while the pregrasp executes, or even before it starts. Other motions depend on
an outcome that has not yet been observed. Whether the fingers hold the object decides
whether a lift is needed at all, and declaring the attachment changes the scene that
the lift must be planned in.

ssgrasping uses ssrobot's projections, predictions, and continuations to plan ahead
exactly where the request is determined. A pregrasp is kept only if the straight
approach also plans from where it ends. With a `Lookahead`, that approach plan is kept
and continued when the pregrasp arrives, after ssrobot confirms that the arm is at rest,
that it arrived where the plan expected, and that the scene has not changed. An arrival
outside the plan's tolerance is planned again from the actual state. An unpredicted
change to the scene, or an arm that does not come to rest, fails the stage without
submitting anything, so that the application can recover. Lookahead stops at stages
whose outcome is not yet known, and the skill records each place where it stopped.

A grasp can also plan the next skill's first motion while its lift runs, such as a
drop's transfer to the container: give it a `Follow`, for example with
`ssgrasping.drop.transfer_request`. The lift's prediction expects the grasp's
`restrict` to narrow what the held object may touch to the hand, the one scene change
between the lift and the transfer, and ssrobot checks on arrival that it happened
exactly so. If the grasp succeeds, it hands the plan on as `SkillExecution.upcoming`,
and `drop(planned=...)` continues it from where the lift ended, so the robot does not
wait for the transfer to be planned. Any other scene change fails the transfer for
recovery.

`Candidates` extends the same idea to a choice between grasps, for example one per arm.
Each candidate's whole feasibility prefix is planned as an ssrobot chain before anything
moves: the pregrasp, the approach, the lift with the object held, and, when a drop
follows, the transfer to the container and the retreat. ssrobot validates each stage
on a fresh scene, prepares each candidate for the runtime before it can be selected, and
selects deterministically by priority, then by the planned duration of the prefix, then
by declaration order. The race can also be planned on the prediction of a motion that
is still executing, so the next object's grasp is chosen while the current object is
carried.

![A pick and drop runs pregrasp, approach, close, lift, transfer, and release. Before anything moves, a race checks each candidate's whole prefix on projected states. The approach planned with the pregrasp is kept and continued on arrival. Closing the fingers and releasing the object are observed outcomes, after which stages are planned again from the actual state. While the transfer executes, the next object's race is planned on its prediction.](docs/figures/planning-ahead.svg)

*The race checks a whole prefix before the robot moves, but only the pregrasp and its
approach are reused. After each observed outcome, the next stages are planned from the
state the robot actually reached.*

The prefix is a search policy rather than a proof. When every transfer seed fails for a
candidate, no other pregrasp is tried, which bounds the work but means that a different
pregrasp might in principle have succeeded.

## Grasping a can with Geodude

From a clone, [uv](https://docs.astral.sh/uv/) installs ssrobot's MuJoCo and planning
integrations and the Geodude and ADA robot packages used by the tests:

```sh
git clone https://github.com/personalrobotics/ssgrasping
cd ssgrasping
uv sync
```

The following example places a soda can on Geodude's work surface, builds grasp regions
from sstsr's side grasps for its Robotiq 2F-140 gripper, and grasps and lifts the can
with the right arm:

```python
import dataclasses
import math
from pathlib import Path

import numpy as np
from sscbirrt import CBiRRTConfig
from ssrobot import Pose, RobotContext, Tsr, load_installed_package
from ssrobot.mujoco import MujocoRuntime
from ssrobot.planning import CBiRRT
from tsr.hands.parallel_jaw import Robotiq2F140

from ssgrasping import GraspSpec, grasp, run_skill

RADIUS, HEIGHT = 0.033, 0.123  # a soda can
SURFACE_Z = 0.753  # the top of Geodude's work surface
Path("can.xml").write_text(f"""<mujoco><worldbody>
  <body name="can" pos="0.35 -0.40 {SURFACE_Z + HEIGHT / 2}">
    <freejoint/>
    <geom type="cylinder" size="{RADIUS} {HEIGHT / 2}" mass="0.05" friction="0.6 0.005 0.0001"/>
  </body>
</worldbody></mujoco>""")


def pose(T):
    """A 4x4 homogeneous transform as an ssrobot Pose."""
    R = T[:3, :3]
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return Pose(position=tuple(map(float, T[:3, 3])), quat_wxyz=(w / n, x / n, y / n, z / n))


def region(t):
    """An instantiated sstsr region as an ssrobot Tsr."""
    b = [tuple(map(float, row)) for row in t.Bw]
    return Tsr(
        origin=pose(t.T0_w),
        offset=pose(t.Tw_e),
        position_bounds=tuple(b[:3]),
        rotation_bounds=tuple(b[3:]),
    )


package = load_installed_package("geodude_assets")
runtime = MujocoRuntime(package, scene="can.xml", attach_tolerance_rad=0.2)
planner = CBiRRT.from_package(package, "right_arm", config=CBiRRTConfig(edge_resolution=0.01))
checker = CBiRRT.from_package(
    package, "right_arm", config=CBiRRTConfig(edge_resolution=0.01, max_iterations=300)
)

with RobotContext(package.description, runtime) as ctx:
    # sstsr frames a grasped cylinder at its base; each side grasp is backed off 8 cm
    # along the hand's approach axis, so the skill can approach in a straight line.
    base = np.eye(4)
    base[:3, 3] = (0.35, -0.40, SURFACE_Z)
    back = np.eye(4)
    back[2, 3] = -0.08
    pregrasps = [
        region(dataclasses.replace(t, Tw_e=t.Tw_e @ back).instantiate(base))
        for t in Robotiq2F140().grasp_cylinder_side(RADIUS, HEIGHT)
    ]
    spec = GraspSpec(
        manipulator="right_arm",
        end_effector="right_hand",
        gripper="right_gripper",
        opening_channel="right_gripper_opening",
        object="can",
        support="vention_base",
        normal=(0.0, 0.0, 1.0),
        pregrasps=pregrasps,
        approach_m=0.08,
        lift_m=0.06,
    )
    result = run_skill(ctx, grasp(ctx, planner, spec, check_planner=checker), max_ticks=200_000)

print(result.state.value)
for stage in result.stages:
    print(f"{stage.name:10} {'ok' if stage.succeeded else 'failed':7} {stage.evidence.value}")
```

It prints `succeeded`, then each stage with its outcome and the kind of evidence it
used:

```text
succeeded
preflight  ok      planned
open       ok      portable
pregrasp   ok      commanded
approach   ok      commanded
close      ok      portable
obstructed ok      portable
attach     ok      planned
lift       ok      commanded
acquired   ok      portable
restrict   ok      planned
clear      ok      planned
retained   ok      privileged
```

Run it with `uv run python` from the repository root, where it writes `can.xml`. CI runs
this same example. The two helpers convert sstsr's matrices into ssrobot's `Pose` and `Tsr`
values. The planners check validity every 0.01 rad in joint space (`edge_resolution`),
the spacing at which this example's straight approach and lift pass end to end. The
attachment monitor tolerates the can swinging between the pads as the arm accelerates.

## Current capabilities

| Capability | Status |
| --- | --- |
| Grasp, place under a force guard, and drop into a container | Implemented; demonstrated on Geodude in MuJoCo |
| Portable grasp verification and conservative failure reporting | Implemented; missed grasps, slips, and unconfirmed releases are tested on Geodude |
| Planning ahead within a skill, and continuing the kept approach | Implemented; demonstrated on Geodude |
| Racing grasp candidates across arms, including during an earlier motion | Implemented; used by Geodude's four-can recycling demo |
| Skills on ADA | Planned; the ADA package opens in the test environment, but no skill has been demonstrated on it |
| Physical robots and learned policies | Not yet supported; they depend on ssrobot's hardware runtimes and policy execution |

Each test writes a deterministic evidence artifact under `artifacts/` and documents how
to reproduce it. A skill counts as robot-neutral once it has been demonstrated on both
Geodude and ADA.

## Related packages

| Package | Owns |
| --- | --- |
| [ssrobot](https://github.com/personalrobotics/ssrobot) | Robot semantics, runtimes, planning contracts, the execution boundary, and evidence |
| [sstsr](https://github.com/personalrobotics/sstsr), [sscbirrt](https://github.com/personalrobotics/sscbirrt), [ssik](https://github.com/personalrobotics/ssik) | Task space regions and gripper templates, constrained planning, and inverse kinematics |
| [Geodude](https://github.com/personalrobotics/geodude) | The bimanual robot application, including the recycling demo built on these skills |

A skill that works across robots belongs here. A choice that is specific to one robot or
task, such as which regions to offer or how to recover from a failure, belongs in that
robot's application.

## Development

`uv sync` installs the development environment, and `uv run pytest` runs the tests.
[CONTRIBUTING.md](CONTRIBUTING.md) lists every check CI runs.
