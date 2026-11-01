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
confirms that the object is clear of everything. `place` carries a held object to a target region, waits until the arm
is observed at rest, so the force guard tares a settled wrist, descends under that guard
until contact, releases the object, and retreats. `drop`
carries the object over a container and lets it fall. It ends at the confirmed
release, so the robot can move on at once. Where a placed or dropped object came to
rest is not something a physical robot observes yet, so it is never a stage (see
*Manipulation transitions*). `landing` scores a drop's `landed` estimate against the
object's privileged pose, for evaluation only, alongside whatever the robot does next;
it commands nothing. Each module
documents its stages and their exact conditions.

Failure is reported in terms of the physical situation rather than as a bare error.
Before anything is commanded, a `preflight` stage confirms that every frame, command,
and channel the skill will use is available. A missed grasp reopens the gripper,
because that cleanup has an unambiguous meaning. Once an object has been declared
attached, a later failure leaves the attachment in place and reports the object as
held, because that is the physically conservative assumption, unless the gripper opening
showed the object leave the hand during a carry. Retrying and recovery
belong to the application.

### Why a stage failed

A failed stage says why twice. Its `reason`, a `Reason`, and its `cause` are for
software: an application chooses its recovery from the stage, the reason, and the
stage's `held` and `attached` state. Its `detail` is prose for a person, and may be
reworded in any release. A successful stage has no reason and no cause.

The reason names the vocabulary of its cause, and a cause is a code, never prose:

| Reason | Meaning | Reported by | Cause |
| --- | --- | --- | --- |
| `unavailable` | Something the skill needs is not declared: a frame, fixture, command, channel, guard, or the object | `preflight`, an allowance change, `observed` | None |
| `not_planned` | No plan for the stage's motion | any planned motion | The `PlanStatus`; None when every seed of a choice failed (`values[<stage>_attempts]`) |
| `refused` | ssrobot refused a plan, command, continuation, or watch | any stage | The ssrobot error code, or the continuation's refusal |
| `command_failed` | A command ended without success: failed, timed out, or rejected | any commanded stage | The execution's diagnostic code, otherwise its state |
| `command_canceled` | A command was canceled | any commanded stage | The execution's diagnostic code, otherwise its state |
| `cancelled` | The skill was cancelled (`SkillExecution.cancel`) | the stage in progress | None |
| `not_attached` | The object must be attached and is not | `transfer`, an allowance change | None |
| `still_attached` | The object must be released and is still attached | `observed` | None |
| `attachment_lost` | The attachment monitor reports the object left its declared transform | an allowance change | None |
| `missed` | The fingers closed to their target: nothing is between them | `obstructed` | None |
| `lost` | After the lift, the fingers closed to their target | `acquired` | None |
| `in_contact` | After the lift, the arm or the object is in contact with the scene | `clear` | None |
| `release_unconfirmed` | The fingers did not reach the released opening | `release` | None |
| `not_at_rest` | The arm was not observed at rest before the tare | `descend` | The `RestOutcome` |
| `no_contact` | The guarded descent ended without its guard triggering | `descend` | The `GuardCause` |
| `outside_regions` | The held object is outside every target region | `contact` | None |
| `transition_refuted` | Portable evidence refuted a manipulation transition the stage needs: for `retained`, the opening guard stopped a carry when the fingers closed past the object's width | `lift`, `transfer` | The `Transition` |

A stage that replans after a refusal and then succeeds has no reason; the refusal is in
its lookahead log. A race's candidates record each plan's `plan_status`, the planner's
own status, so a planning timeout is told from an infeasible goal without its detail.

## Robot-specific choices are data

`GraspSpec`, `PlaceSpec`, `DropSpec`, and `LandingSpec` name everything a skill needs: the manipulator,
end effector, gripper, and gripper-opening channel; the object and the support it rests
on, with the support's outward normal; the regions in which the hand or the object
must end, as task space regions; and the distances, tolerances, and attempt limits for
each stage. Nothing is inferred from names. Joint rest uses the criteria the runtime
declares for the arm's channel.

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
support could stop them too.

### Manipulation transitions

A skill also estimates the transitions it passes through (#28) as
`TransitionEstimate`s in `SkillResult.transitions`: the transition, the stage, an
outcome (`confirmed`, `refuted` or `unobserved`), a stable `EstimateReason`, the
portable readings it used, and the criteria that decided it, each with its source.
`unobserved` is not success: nothing available decided it, and what to do then is the
application's.

Today one transition is estimated: `retained`, over every carry (`grasp`'s `lift`, and
the `transfer` of `drop` and `place`). The carry runs under an ssrobot `OpeningGuard`
midway between the opening the object is held at and the closing target. An object that
leaves the hand lets the fingers run on toward the target, so the guard stops the arm
where the loss was seen, `retained` is refuted, and the stage fails at once
(`transition_refuted`), with the object reported not held and still declared attached:
detaching it is the application's choice. `place`'s descent runs under its wrench
guard alone, so `retained` is not monitored through it.

The evidence for this is
[`evidence/transitions/`](evidence/transitions/README.md). In Geodude's MuJoCo model the
opening falls by at least 0.448 after every slip and at most 0.007 while held. That
depends on the simulated 2F-140 closing on after the object leaves, which is not yet
confirmed on the physical gripper.

`placed` and `landed` are estimated at a confirmed release, and are always `unobserved`
(`no_portable_signal`): no signal a physical robot reports yet tells where the object
came to rest. Assuming them, or not, is the application's policy. `drop` and `place`
hand the object to the caller's `released` policy once the release is confirmed and
the object detached, before anything else is planned.

The skills run unchanged in a context with registered object poses (ssrobot's
`ObjectPoses.REGISTERED`), where the scene holds what the application registered, not
what the simulator measures. There a released object's pose is unknown until the
`released` policy registers or removes it.

Privileged state never decides a stage. Where the runtime has the object's privileged
pose, each estimate is scored in a `TransitionEvaluation` (`SkillResult.evaluations`),
for evaluation only:
- `retained`: against the runtime's attachment monitor (`ctx.violation`), with
  `source` `attachment_monitor`, and the object's pose at the carry's start and end;
- `placed`, after `place`'s retreat, and `landed`, by `landing`: against the object's
  pose in the target regions, with `source` the pose channel.

For `retained`, the reference is privileged, not ground truth: in Geodude's
development corpus the monitor reported a box lost that had only swung in the hand,
while the opening correctly showed it held. Ground truth is reserved for physical
state, as the study above judges it.

## Passive execution

A skill advances only when its caller advances it. `grasp`, `place`, `drop`, and
`landing` return a `SkillExecution` that reads the runtime clock, execution states,
ssrobot's passive rest watches, and finished plans, but never steps or polls the
runtime itself. A wait on the robot is on an observed event; a timeout only bounds how
long it may take before the stage fails. `landing`'s fixed `land_ns` is the one
exception, until a dropped object can be observed at rest (see CHANGELOG.md). `run_skill` drives one
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
and continued when the pregrasp arrives, after ssrobot observes the arm at rest
(within `Lookahead.rest_timeout_ns`),
that it arrived where the plan expected, and that the scene has not changed. An arrival
outside the plan's tolerance is planned again from the actual state. An unpredicted
change to the scene, or an arm that does not come to rest, fails the stage without
submitting anything, so that the application can recover. Lookahead stops at stages
whose outcome is not yet known, and the skill records each place where it stopped.

A grasp can also plan the next skill's first motion while its lift runs, such as a
drop's transfer to the container: give it a `Follow`, for example with
`ssgrasping.drop.transfer_request`, for a drop without a retreat. The lift's prediction expects the grasp's
`restrict` to narrow what the held object may touch to the hand, the one scene change
between the lift and the transfer, and ssrobot checks on arrival that it happened
exactly so. If the grasp succeeds, it hands the plan on as `SkillExecution.upcoming`,
and `drop(planned=...)` continues it from where the lift ended, so the robot does not
wait for the transfer to be planned. Any other scene change fails the transfer for
recovery. A drop with a retreat refuses a planned transfer: its transfer is kept only if
the retreat plans from where it ends, which would have to be planned with it.

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
retained   ok      portable
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
| Estimating `retained` from the gripper opening, stopping a carry that loses its object | Implemented; demonstrated on Geodude in MuJoCo. `placed` and `landed` are recorded `unobserved`, scored against privileged poses; `acquired` and `released` estimates are planned (#28) |
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
